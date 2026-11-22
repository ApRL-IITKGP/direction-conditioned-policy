import time

import jax
import numpy as np
from brax import envs
from brax.training import acting
from brax.training.types import Metrics, PolicyParams


# TODO: make only single type of Evaluator
class Evaluator(acting.Evaluator):
    """
    Class for running evaluation epochs in a training process.

    This class extends the base acting.Evaluator and is designed to evaluate the
    performance of policies based on given parameters and training metrics.
    It provides detailed evaluations by aggregating various metrics over episodes
    and calculating statistical summaries such as mean, standard deviation,
    maximum, and minimum values.

    This is an evaluator that behaves in the exact same way as brax Evaluator,
    but additionally it aggregates metrics with max, min.
    It also logs in how many episodes there was any success.
    """

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        # Brax's generate_eval_unroll signature is (donated_env_state, policy_params, key);
        # update self._eval_state_to_donate with the returned state for next call.
        eval_state = self._generate_eval_unroll(
            self._eval_state_to_donate, policy_params, unroll_key
        )
        self._eval_state_to_donate = eval_state
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            (np.std, "_std"),
            (np.max, "_max"),
            (np.min, "_min"),
        ]

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (fn(value) if aggregate_episodes else value)
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(eval_metrics.episode_metrics["success"] > 0.0)

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray


def generate_unroll(actor_step, training_state, env, env_state, unroll_length, extra_fields=()):
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state = carry
        nstate, transition = actor_step(training_state, env, state, extra_fields=extra_fields)
        return nstate, transition

    final_state, data = jax.lax.scan(f, env_state, (), length=unroll_length)
    return final_state, data


def _reaching_sustaining_metrics(reward_traj, episode_length):
    """Compute time-to-first-success and success-given-reached from reward trajectory.

    Assumes sparse reward (reward == success indicator, values in {0, 1}).
    Both metrics are only logged when at least one episode reached the goal.

    Args:
        reward_traj: per-step rewards, shape (T, N). Only this is transferred from GPU,
                     so obs/action/extras are never materialised (XLA DCE).
        episode_length: total steps per episode T.

    Returns:
        dict with keys:
          eval/time_to_first_success   -- mean step of first success (raw step count
                                          in [0, T]) among episodes that reached. Lower
                                          = faster reaching.
          eval/success_given_reached   -- mean fraction (in [0, 1]) of post-reach
                                          steps spent at goal, among episodes that
                                          reached. Higher = better sustaining.
    """
    metrics = {}
    success_traj = np.asarray(reward_traj)               # (T, N)
    T = success_traj.shape[0]
    ever_succeeded = success_traj.max(axis=0) > 0        # (N,) bool
    # When no episode reached, time-to-first should encode "very poor" (lower=better),
    # so we report a large sentinel rather than 0 (which would falsely look optimal).
    NEVER_REACHED = 1e6
    if not ever_succeeded.any():
        metrics["eval/time_to_first_success"] = float(NEVER_REACHED)
        metrics["eval/success_given_reached"] = 0.0
        return metrics
    first_step = np.argmax(success_traj > 0, axis=0)     # (N,) — valid only where ever_succeeded
    metrics["eval/time_to_first_success"] = float(
        np.mean(first_step[ever_succeeded])
    )
    step_idx = np.arange(T)[:, None]                     # (T, 1)
    after_first = step_idx >= first_step[None, :]        # (T, N)
    steps_remaining = np.maximum(T - first_step, 1)      # (N,)
    success_after = (success_traj * after_first).sum(axis=0) / steps_remaining  # (N,)
    metrics["eval/success_given_reached"] = float(
        np.mean(success_after[ever_succeeded])
    )
    return metrics


class ActorEvaluator:
    """Single GPU evaluator that evaluates an arbitrary actor function. Used by the CRL agent."""

    def __init__(self, actor_step, eval_env, num_eval_envs, episode_length, key):
        self._key = key
        self._eval_walltime = 0.0
        self._episode_length = episode_length

        eval_env = envs.training.EvalWrapper(eval_env)

        def generate_eval_unroll(training_state, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            final_state, traj_data = generate_unroll(
                actor_step,
                training_state,
                eval_env,
                eval_first_state,
                unroll_length=episode_length,
            )
            # Only return reward from trajectory so XLA can DCE obs/action/extras.
            # reward shape: (T, N) — 2MB for T=1000, N=512. Everything else is dropped.
            return final_state, traj_data.reward

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self, training_state, training_metrics, aggregate_episodes=True):
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state, reward_traj = self._generate_eval_unroll(training_state, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            # (np.std, "_std"),
            # (np.max, "_max"),
            # (np.min, "_min"),
        ]

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(eval_metrics.episode_metrics[name])
                        if aggregate_episodes
                        else eval_metrics.episode_metrics[name]
                    )
                    for name in [
                        "reward",
                        "success",
                        "success_easy",
                        "dist",
                        "distance_from_origin",
                        "contact",
                        "puck_displacement",
                        "goal_actuation_ratio",
                        "behind_alignment",
                    ]
                    if name in eval_metrics.episode_metrics
                }
            )

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(eval_metrics.episode_metrics["success"] > 0.0)

        # Reaching vs. sustaining diagnostics (sparse-reward envs only)
        metrics.update(_reaching_sustaining_metrics(reward_traj, self._episode_length))

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics
