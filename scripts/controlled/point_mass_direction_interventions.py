"""Direction interventions on trained DCP point-mass policies in the U-maze.

For every directed pair of free maze-cell centers, the trained actor is rolled
out from rest with the learned direction input and with edited inputs
(reversed, wrong-goal, orthogonal, or zero direction; zero or shuffled
magnitude). The physical state, goal, actor, encoder, and reset keys are held
fixed; only the direction or magnitude given to the actor changes. Success is
reported overall, on detour pairs (graph distance > 1.25 x Euclidean), and by
graph-distance horizon.

Example:
    python -m scripts.controlled.point_mass_direction_interventions --runs-dir runs
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
    LAYOUTS,
    SCALING,
    build_fine_graph,
    dcp_actor_input,
    free_macro_centers,
    initial_states,
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
RESET_SEED = 20260725


def summarize(reached, first_success, path_length, shortest, final_distance, mask) -> dict:
    successful = reached[mask] > 0
    return {
        "pairs": int(np.count_nonzero(mask)),
        "success_rate": float(np.mean(reached[mask])),
        "mean_first_success_if_reached": (
            float(np.mean(first_success[mask][successful])) if np.any(successful) else None
        ),
        "mean_path_efficiency_if_reached": (
            float(np.mean(shortest[mask][successful] / (path_length[mask][successful] + 1e-8)))
            if np.any(successful)
            else None
        ),
        "median_final_distance": float(np.median(final_distance[mask])),
    }


def evaluate(run_dir: Path, horizon: int, subdivisions: int) -> dict:
    agent, run = load_run(run_dir)
    require(run, agent, "DCP", "simple_u_maze")
    env = create_env(run.env, backend=run.backend)
    actor = make_actor(agent, env.action_size)
    encoder = make_encoder(agent)
    params = load_params(run_dir)
    actor_params, encoder_params = params[1], params[2]["g_encoder"]
    state_dim = env.state_dim
    goal_indices = jnp.asarray(env.goal_indices, dtype=jnp.int32)

    layout = LAYOUTS["u_maze"]
    points = free_macro_centers(layout)
    source_index, goal_index = np.where(~np.eye(len(points), dtype=bool))
    source_xy, goal_xy = points[source_index], points[goal_index]
    count = len(source_xy)

    fine_xy, _, fine_distances = build_fine_graph(layout, subdivisions)
    source_node, _ = nearest_nodes(source_xy, fine_xy)
    goal_node, _ = nearest_nodes(goal_xy, fine_xy)
    shortest = fine_distances[source_node, goal_node] * SCALING / subdivisions
    euclidean = np.linalg.norm(goal_xy - source_xy, axis=1)
    detour = shortest / (euclidean + 1e-12) > 1.25
    # U-maze cell-center routes are one to six 4-unit cell steps long.
    horizon_label = np.where(
        shortest <= 8.0 + 1e-6, "short", np.where(shortest <= 16.0 + 1e-6, "medium", "long")
    )

    initial_state = initial_states(
        env,
        RESET_SEED,
        count,
        lambda q: q.at[:, :2].set(jnp.asarray(source_xy)).at[:, -2:].set(jnp.asarray(goal_xy)),
    )

    def actions(state, intervention):
        inputs = dcp_actor_input(
            encoder, encoder_params, state.obs, state_dim, goal_indices,
            agent.use_magnitude, intervention, shift=1,
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
    result = {"run_dir": str(run_dir), "seed": run.seed, "pairs": count, "interventions": {}}
    for intervention in INTERVENTIONS:
        reached, first_success, path_length, final_distance = (
            np.asarray(v) for v in jax.jit(rollout, static_argnums=(0,))(intervention)
        )

        def stats(mask):
            return summarize(reached, first_success, path_length, shortest, final_distance, mask)

        initial_action = np.asarray(actions(initial_state, intervention))
        row = {
            "initial_action_l2_from_correct_mean": float(
                np.mean(np.linalg.norm(initial_action - correct_action, axis=1))
            ),
            "all": stats(np.ones(count, dtype=bool)),
            "detour": stats(detour),
        }
        for label in ("short", "medium", "long"):
            row[label] = stats(horizon_label == label)
        result["interventions"][intervention] = row
        print(
            f"seed={run.seed} {intervention:<18} success={row['all']['success_rate']:.3f} "
            f"detour={row['detour']['success_rate']:.3f} long={row['long']['success_rate']:.3f}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123, 256, 1234])
    parser.add_argument("--run-dirs", type=Path, nargs="+", help="overrides --runs-dir/--seeds")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--subdivisions", type=int, default=8)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    run_dirs = args.run_dirs or [
        run_path(args.runs_dir, "dcp_simple_u_maze", seed) for seed in args.seeds
    ]
    results = [evaluate(run_dir, args.horizon, args.subdivisions) for run_dir in run_dirs]

    summary = {}
    print("\nsuccess rate across seeds (mean +/- sample sd)")
    for intervention in INTERVENTIONS:
        summary[intervention] = {
            subset: seed_summary(
                {r["seed"]: r["interventions"][intervention][subset]["success_rate"] for r in results}
            )
            for subset in ("all", "detour", "long")
        }
        s = summary[intervention]["all"]
        print(f"{intervention:<18} {100 * s['mean']:.1f}% +/- {100 * (s['sample_sd'] or 0):.1f}")
    write_json(args.json_out, {"runs": results, "summary": summary})


if __name__ == "__main__":
    main()
