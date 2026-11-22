"""Direction interventions on trained DCP puck-pushing policies.

Every task configuration of the environment (robot, puck, and target
positions) is evaluated `--repeats` times from rest. The trained actor is
rolled out with the learned direction input and with edited inputs (reversed,
wrong-goal, orthogonal, or zero direction; zero magnitude), holding the
physical state, target, actor, encoder, and reset keys fixed. Outcomes
separate contact, puck transport, and goal reaching.

Example:
    python -m scripts.controlled.push_bot_direction_interventions --runs-dir runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from jaxgcrl.utils.env import create_env
from scripts.controlled.common import (
    dcp_actor_input,
    initial_states,
    load_params,
    load_run,
    make_actor,
    make_encoder,
    require,
    run_path,
    seed_summary,
    write_json,
)

INTERVENTIONS = (
    "correct",
    "flip",
    "wrong_goal",
    "orthogonal",
    "zero_direction",
    "zero_magnitude",
)
METRICS = ("success_rate", "contact_rate", "mean_puck_path_length")
RESET_SEED = 20260726


def evaluate(run_dir: Path, horizon: int, repeats: int) -> dict:
    agent, run = load_run(run_dir)
    require(run, agent, "DCP", "push_bot_")
    env = create_env(run.env, backend=run.backend)
    actor = make_actor(agent, env.action_size)
    encoder = make_encoder(agent)
    params = load_params(run_dir)
    actor_params, encoder_params = params[1], params[2]["g_encoder"]
    state_dim = env.state_dim
    goal_indices = jnp.asarray(env.goal_indices, dtype=jnp.int32)

    configurations = np.repeat(np.asarray(env.configs), repeats, axis=0)
    count = len(configurations)
    initial_state = initial_states(
        env, RESET_SEED, count, lambda q: q.at[:, :6].set(jnp.asarray(configurations))
    )

    def actions(state, intervention):
        # wrong_goal uses the target of the next task configuration.
        inputs = dcp_actor_input(
            encoder, encoder_params, state.obs, state_dim, goal_indices,
            agent.use_magnitude, intervention, shift=repeats,
        )
        mean, _ = actor.apply(actor_params, inputs)
        return nn.tanh(mean)

    batched_step = jax.vmap(env.step)

    def rollout(intervention):
        def step(carry, _):
            state, reached, contacted, first_contact, puck_path, previous_puck, t = carry
            next_state = batched_step(state, actions(state, intervention))
            contact = next_state.metrics["contact"] > 0
            first_contact = jnp.where(contact & ~contacted, t, first_contact)
            contacted |= contact
            reached |= next_state.metrics["success"] > 0
            puck = next_state.obs[:, 2:4]
            puck_path += jnp.linalg.norm(puck - previous_puck, axis=1)
            return (next_state, reached, contacted, first_contact, puck_path, puck, t + 1), None

        carry = (
            initial_state,
            jnp.zeros((count,), dtype=bool),
            jnp.zeros((count,), dtype=bool),
            jnp.full((count,), horizon, dtype=jnp.int32),
            jnp.zeros((count,)),
            initial_state.obs[:, 2:4],
            jnp.asarray(0, dtype=jnp.int32),
        )
        (state, reached, contacted, first_contact, puck_path, _, _), _ = jax.lax.scan(
            step, carry, None, length=horizon
        )
        goal_distance = jnp.linalg.norm(state.obs[:, 2:4] - state.obs[:, state_dim:], axis=1)
        agent_puck_distance = jnp.linalg.norm(state.obs[:, :2] - state.obs[:, 2:4], axis=1)
        return reached, contacted, first_contact, puck_path, goal_distance, agent_puck_distance

    correct_action = np.asarray(actions(initial_state, "correct"))
    result = {
        "run_dir": str(run_dir),
        "env": run.env,
        "seed": run.seed,
        "configurations": len(env.configs),
        "repeats": repeats,
        "cases": count,
        "interventions": {},
    }
    for intervention in INTERVENTIONS:
        reached, contacted, first_contact, puck_path, goal_distance, agent_puck_distance = (
            np.asarray(v) for v in jax.jit(rollout, static_argnums=(0,))(intervention)
        )
        initial_action = np.asarray(actions(initial_state, intervention))
        row = {
            "success_rate": float(np.mean(reached)),
            "contact_rate": float(np.mean(contacted)),
            "mean_first_contact_if_contacted": (
                float(np.mean(first_contact[contacted])) if np.any(contacted) else None
            ),
            "mean_puck_path_length": float(np.mean(puck_path)),
            "median_final_goal_distance": float(np.median(goal_distance)),
            "median_final_agent_puck_distance": float(np.median(agent_puck_distance)),
            "initial_action_l2_from_correct_mean": float(
                np.mean(np.linalg.norm(initial_action - correct_action, axis=1))
            ),
        }
        result["interventions"][intervention] = row
        print(
            f"seed={run.seed} {intervention:<15} success={row['success_rate']:.3f} "
            f"contact={row['contact_rate']:.3f} puck_path={row['mean_puck_path_length']:.3f}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--env", default="push_bot_open")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123])
    parser.add_argument("--run-dirs", type=Path, nargs="+", help="overrides --runs-dir/--seeds")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    run_dirs = args.run_dirs or [run_path(args.runs_dir, f"dcp_{args.env}", s) for s in args.seeds]
    results = [evaluate(run_dir, args.horizon, args.repeats) for run_dir in run_dirs]

    summary = {"interventions": {}, "correct_minus_intervention": {}}
    print("\nmean over seeds")
    for intervention in INTERVENTIONS:
        summary["interventions"][intervention] = {
            metric: seed_summary({r["seed"]: r["interventions"][intervention][metric] for r in results})
            for metric in METRICS
        }
        if intervention != "correct":
            summary["correct_minus_intervention"][intervention] = {
                metric: seed_summary(
                    {
                        r["seed"]: r["interventions"]["correct"][metric]
                        - r["interventions"][intervention][metric]
                        for r in results
                    }
                )
                for metric in METRICS
            }
        s = summary["interventions"][intervention]
        print(
            f"{intervention:<15} success={100 * s['success_rate']['mean']:.1f}% "
            f"contact={100 * s['contact_rate']['mean']:.1f}% "
            f"puck_path={s['mean_puck_path_length']['mean']:.3f}"
        )
    write_json(args.json_out, {"runs": results, "summary": summary})


if __name__ == "__main__":
    main()
