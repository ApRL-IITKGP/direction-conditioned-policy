"""Waypoint substitution for trained DCP point-mass policies in the U-maze.

For start-goal pairs on the occupancy graph (at least two edges apart), the
actor's action at the start (at rest) is computed with three direction and
magnitude targets: the final goal, an exact graph-shortest-path midpoint, and
an off-path control whose graph distance from the start is matched to the
midpoint. Reports the cosine between each substituted action and the
final-goal action, and how often the action has positive cosine with a
graph-optimal first move. Actor, encoder, physical states, and pairs are
fixed across targets.

Example:
    python -m scripts.controlled.waypoint_substitution --runs-dir runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from jaxgcrl.utils.env import create_env
from scripts.controlled.common import (
    LAYOUTS,
    SCALING,
    build_fine_graph,
    goal_embeddings,
    load_params,
    load_run,
    make_actor,
    make_encoder,
    optimal_first_steps,
    require,
    run_path,
    seed_summary,
    write_json,
)

PAIR_SEED = 20260727
TARGETS = ("onpath_midpoint", "offpath_control")


def choose_waypoints(distances, source, goal):
    """On-path midpoint and distance-matched off-path node for each pair."""
    nodes = np.arange(distances.shape[0])
    onpath = np.empty(len(source), dtype=np.int32)
    offpath = np.empty(len(source), dtype=np.int32)
    for index, (src, dst) in enumerate(zip(source, goal, strict=True)):
        d_sz = distances[src]
        slack = d_sz + distances[:, dst] - distances[src, dst]
        interior = (nodes != src) & (nodes != dst)
        exact = nodes[interior & np.isclose(slack, 0.0, atol=1e-7)]
        onpath[index] = exact[np.argmin(np.abs(d_sz[exact] - 0.5 * distances[src, dst]))]
        # Closest start distance to the midpoint among nodes with positive
        # slack; ties broken by the largest slack.
        off = nodes[interior & (slack > 1e-7)]
        order = np.lexsort((-slack[off], np.abs(d_sz[off] - d_sz[onpath[index]])))
        offpath[index] = off[order[0]]
    return onpath, offpath


def cosine(first, second):
    return np.sum(first * second, axis=1) / (
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1) + 1e-12
    )


def summarize(direction, magnitude, action, final, optimal_step, mask) -> dict:
    final_direction, final_magnitude, final_action = final
    optimal_alignment = cosine(action, optimal_step)
    direction_cosine = cosine(direction[mask], final_direction[mask])
    return {
        "pairs": int(np.count_nonzero(mask)),
        "direction_cosine_to_final_mean": float(np.mean(direction_cosine)),
        "direction_cosine_to_final_median": float(np.median(direction_cosine)),
        "relative_magnitude_error_to_final_median": float(
            np.median(
                np.abs(magnitude[mask] - final_magnitude[mask]) / (final_magnitude[mask] + 1e-12)
            )
        ),
        "action_cosine_to_final_mean": float(np.mean(cosine(action[mask], final_action[mask]))),
        "action_l2_to_final_mean": float(
            np.mean(np.linalg.norm(action[mask] - final_action[mask], axis=1))
        ),
        "graph_optimal_action_positive_rate": float(np.mean(optimal_alignment[mask] > 0.0)),
        "graph_optimal_action_cosine_mean": float(np.mean(optimal_alignment[mask])),
    }


def evaluate(run_dir: Path, pairs: int, subdivisions: int) -> dict:
    agent, run = load_run(run_dir)
    require(run, agent, "DCP", "simple_u_maze")
    env = create_env(run.env, backend=run.backend)
    encoder = make_encoder(agent)
    actor = make_actor(agent, env.action_size)
    params = load_params(run_dir)
    actor_params, encoder_params = params[1], params[2]["g_encoder"]

    coordinates, adjacency, distances = build_fine_graph(LAYOUTS["u_maze"], subdivisions)
    rng = np.random.default_rng(PAIR_SEED)
    source, goal = [], []
    # At least two graph edges apart, so an interior waypoint exists.
    while len(source) < pairs:
        candidate_source = rng.integers(0, len(coordinates), size=pairs)
        candidate_goal = rng.integers(0, len(coordinates), size=pairs)
        keep = distances[candidate_source, candidate_goal] >= 2.0 - 1e-7
        source.extend(candidate_source[keep].tolist())
        goal.extend(candidate_goal[keep].tolist())
    source = np.asarray(source[:pairs], dtype=np.int32)
    goal = np.asarray(goal[:pairs], dtype=np.int32)
    onpath, offpath = choose_waypoints(distances, source, goal)

    embeddings = goal_embeddings(encoder, encoder_params, coordinates)
    physical = np.zeros((pairs, env.state_dim), dtype=np.float32)
    physical[:, :2] = coordinates[source]

    def endpoint_values(endpoint):
        gap = embeddings[endpoint] - embeddings[source]
        magnitude = np.linalg.norm(gap, axis=1)
        direction = gap / (magnitude[:, None] + 1e-12)
        pieces = [physical, direction]
        if agent.use_magnitude:
            pieces.append(magnitude[:, None])
        mean, _ = actor.apply(actor_params, jnp.asarray(np.concatenate(pieces, axis=1)))
        return direction, magnitude, np.asarray(nn.tanh(mean))

    final = endpoint_values(goal)
    final_action = final[2]

    # One graph-optimal first move per pair; among tied shortest paths, the
    # one best aligned with the final-goal action.
    graph_step = SCALING / subdivisions
    optimal_step = np.zeros_like(final_action)
    for index, (src, dst) in enumerate(zip(source, goal, strict=True)):
        directions = optimal_first_steps(coordinates, adjacency, distances, src, dst, graph_step)
        optimal_step[index] = directions[np.argmax(directions @ final_action[index])]

    graph_world = distances[source, goal] * graph_step
    cuts = np.quantile(graph_world, (1.0 / 3.0, 2.0 / 3.0))
    masks = {
        "all": np.ones(pairs, dtype=bool),
        "short": graph_world <= cuts[0],
        "medium": (graph_world > cuts[0]) & (graph_world <= cuts[1]),
        "long": graph_world > cuts[1],
    }
    result = {"run_dir": str(run_dir), "seed": int(run.seed), "pairs": pairs, "targets": {}}
    for label, endpoint in zip(TARGETS, (onpath, offpath)):
        direction, magnitude, action = endpoint_values(endpoint)
        rows = {
            horizon: summarize(direction, magnitude, action, final, optimal_step, mask)
            for horizon, mask in masks.items()
        }
        rows["graph_slack_mean"] = float(
            np.mean(distances[source, endpoint] + distances[endpoint, goal] - distances[source, goal])
        )
        result["targets"][label] = rows
        print(
            f"seed={run.seed} {label:<16} "
            f"action_cosine={rows['all']['action_cosine_to_final_mean']:.3f} "
            f"optimal_positive={100 * rows['all']['graph_optimal_action_positive_rate']:.1f}%"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123])
    parser.add_argument("--run-dirs", type=Path, nargs="+", help="overrides --runs-dir/--seeds")
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument("--subdivisions", type=int, default=8)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    run_dirs = args.run_dirs or [
        run_path(args.runs_dir, "dcp_simple_u_maze", seed) for seed in args.seeds
    ]
    results = [evaluate(run_dir, args.pairs, args.subdivisions) for run_dir in run_dirs]

    summary = {}
    print("\nmean over seeds")
    for label in TARGETS:
        summary[label] = {
            metric: seed_summary({r["seed"]: r["targets"][label]["all"][metric] for r in results})
            for metric in ("action_cosine_to_final_mean", "graph_optimal_action_positive_rate")
        }
        print(
            f"{label:<16} action_cosine={summary[label]['action_cosine_to_final_mean']['mean']:.3f} "
            f"optimal_positive="
            f"{100 * summary[label]['graph_optimal_action_positive_rate']['mean']:.1f}%"
        )
    write_json(args.json_out, {"runs": results, "summary": summary})


if __name__ == "__main__":
    main()
