"""Direction interventions on trained DCP policies for the differential-drive U-maze.

The robot (state x, y, heading, and their velocities) is actuated only by
forward thrust and yaw torque. Every directed pair of free maze-cell centers
is evaluated at four initial headings. For each case the trained actor is
rolled out with the learned direction input and with edited inputs (reversed,
wrong-goal, orthogonal, or zero direction; zero or shuffled magnitude), holding
the physical state, goal, heading, actor, encoder, and reset keys fixed.

Example:
    python -m scripts.controlled.bot_direction_interventions --runs-dir runs
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
    SCALING,
    build_fine_graph,
    dcp_actor_input,
    free_macro_centers,
    initial_states,
    layout_from_env,
    load_params,
    load_run,
    make_actor,
    make_encoder,
    nearest_nodes,
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
    "shuffled_magnitude",
)
HEADINGS = np.asarray([-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0], dtype=np.float32)
PAIR_SEED = 20260726
RESET_SEED = 20260726
SUBSETS = ("all", "detour", "long_horizon")


def sample_pairs(points: np.ndarray, maximum: int, seed: int):
    source, goal = np.where(~np.eye(len(points), dtype=bool))
    if len(source) > maximum:
        keep = np.random.default_rng(seed).choice(len(source), maximum, replace=False)
        source, goal = source[keep], goal[keep]
    return points[source], points[goal]


def summarize(reached, first_success, final_distance, shortest, path_length, mask) -> dict:
    successful = reached[mask] > 0
    return {
        "cases": int(np.count_nonzero(mask)),
        "success_rate": float(np.mean(reached[mask])),
        "median_final_distance": float(np.median(final_distance[mask])),
        "mean_first_success_if_reached": (
            float(np.mean(first_success[mask][successful])) if np.any(successful) else None
        ),
        "mean_path_efficiency_if_reached": (
            float(np.mean(shortest[mask][successful] / (path_length[mask][successful] + 1e-8)))
            if np.any(successful)
            else None
        ),
    }


def evaluate(run_dir: Path, horizon: int, max_pairs: int, subdivisions: int) -> dict:
    agent, run = load_run(run_dir)
    require(run, agent, "DCP", "bot_")
    _, layout = layout_from_env(run.env)
    env = create_env(run.env, backend=run.backend)
    actor = make_actor(agent, env.action_size)
    encoder = make_encoder(agent)
    params = load_params(run_dir)
    actor_params, encoder_params = params[1], params[2]["g_encoder"]
    state_dim = env.state_dim
    goal_indices = jnp.asarray(env.goal_indices, dtype=jnp.int32)

    source_pair, goal_pair = sample_pairs(free_macro_centers(layout), max_pairs, PAIR_SEED)
    # Each physical pair is evaluated at every initial heading.
    source_xy = np.repeat(source_pair, len(HEADINGS), axis=0)
    goal_xy = np.repeat(goal_pair, len(HEADINGS), axis=0)
    heading = np.tile(HEADINGS, len(source_pair))
    count = len(source_xy)

    fine_xy, _, fine_distance = build_fine_graph(layout, subdivisions)
    source_node, _ = nearest_nodes(source_xy, fine_xy)
    goal_node, _ = nearest_nodes(goal_xy, fine_xy)
    shortest = fine_distance[source_node, goal_node] * SCALING / subdivisions
    euclidean = np.linalg.norm(goal_xy - source_xy, axis=1)
    masks = {
        "all": np.ones(count, dtype=bool),
        "detour": shortest > 1.25 * euclidean,
        "long_horizon": shortest >= np.quantile(shortest, 2.0 / 3.0),
    }

    initial_state = initial_states(
        env,
        RESET_SEED,
        count,
        lambda q: q.at[:, :2]
        .set(jnp.asarray(source_xy))
        .at[:, 2]
        .set(jnp.asarray(heading))
        .at[:, -2:]
        .set(jnp.asarray(goal_xy)),
    )

    def actions(state, intervention):
        inputs = dcp_actor_input(
            encoder, encoder_params, state.obs, state_dim, goal_indices,
            agent.use_magnitude, intervention, shift=4,
        )
        mean, _ = actor.apply(actor_params, inputs)
        return nn.tanh(mean)

    batched_step = jax.vmap(env.step)

    def rollout(intervention):
        def step(carry, _):
            state, reached, first_success, path_length, previous_xy, t = carry
            next_state = batched_step(state, actions(state, intervention))
            xy = next_state.obs[:, :2]
            path_length += jnp.linalg.norm(xy - previous_xy, axis=1)
            success = next_state.metrics["success"] > 0
            first_success = jnp.where(success & ~reached, t, first_success)
            reached |= success
            return (next_state, reached, first_success, path_length, xy, t + 1), None

        carry = (
            initial_state,
            jnp.zeros((count,), dtype=bool),
            jnp.full((count,), horizon, dtype=jnp.int32),
            jnp.zeros((count,)),
            jnp.asarray(source_xy),
            jnp.asarray(0, dtype=jnp.int32),
        )
        (state, reached, first_success, path_length, _, _), _ = jax.lax.scan(
            step, carry, None, length=horizon
        )
        final_distance = jnp.linalg.norm(state.obs[:, :2] - state.obs[:, state_dim:], axis=1)
        return reached, first_success, path_length, final_distance

    correct_action = np.asarray(actions(initial_state, "correct"))
    result = {
        "run_dir": str(run_dir),
        "env": run.env,
        "seed": run.seed,
        "physical_pairs": len(source_pair),
        "headings_per_pair": len(HEADINGS),
        "cases": count,
        "interventions": {},
    }
    for intervention in INTERVENTIONS:
        reached, first_success, path_length, final_distance = (
            np.asarray(v) for v in jax.jit(rollout, static_argnums=(0,))(intervention)
        )
        initial_action = np.asarray(actions(initial_state, intervention))
        row = {
            "initial_action_l2_from_correct_mean": float(
                np.mean(np.linalg.norm(initial_action - correct_action, axis=1))
            ),
        }
        for subset, mask in masks.items():
            row[subset] = summarize(
                reached, first_success, final_distance, shortest, path_length, mask
            )
        result["interventions"][intervention] = row
        print(
            f"seed={run.seed} {intervention:<18} success={row['all']['success_rate']:.3f} "
            f"detour={row['detour']['success_rate']:.3f} "
            f"long={row['long_horizon']['success_rate']:.3f}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123])
    parser.add_argument("--run-dirs", type=Path, nargs="+", help="overrides --runs-dir/--seeds")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--max-pairs", type=int, default=256)
    parser.add_argument("--subdivisions", type=int, default=8)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    run_dirs = args.run_dirs or [run_path(args.runs_dir, "dcp_bot_u_maze", s) for s in args.seeds]
    results = [
        evaluate(run_dir, args.horizon, args.max_pairs, args.subdivisions) for run_dir in run_dirs
    ]

    def success(result, intervention, subset):
        return result["interventions"][intervention][subset]["success_rate"]

    summary = {"interventions": {}, "correct_minus_intervention": {}}
    print("\nsuccess rate across seeds (mean +/- sample sd)")
    for intervention in INTERVENTIONS:
        summary["interventions"][intervention] = {
            subset: seed_summary({r["seed"]: success(r, intervention, subset) for r in results})
            for subset in SUBSETS
        }
        if intervention != "correct":
            summary["correct_minus_intervention"][intervention] = {
                subset: seed_summary(
                    {
                        r["seed"]: success(r, "correct", subset) - success(r, intervention, subset)
                        for r in results
                    }
                )
                for subset in SUBSETS
            }
        s = summary["interventions"][intervention]["all"]
        print(f"{intervention:<18} {100 * s['mean']:.1f}% +/- {100 * (s['sample_sd'] or 0):.1f}")
    write_json(args.json_out, {"runs": results, "summary": summary})


if __name__ == "__main__":
    main()
