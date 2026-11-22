"""Representation geometry of trained goal encoders in the point-mass mazes.

For each run, compares the goal-encoder distance ||psi(x) - psi(y)|| with
occupancy-graph shortest-path distance and with Euclidean distance (Spearman
correlation), overall and within graph-distance terciles. The U-maze uses all
unordered node pairs; the open room uses a fixed sample of 100,000 pairs.

In the U-maze it also evaluates graph-correct local descent: on every directed
pair for which no graph-optimal first move has positive alignment with the
straight-to-goal vector, the fraction of pairs where the negative spatial
gradient of the learned distance has positive cosine with some graph-optimal
first move.

Example:
    python -m scripts.controlled.geometry --runs-dir runs --json-out results/geometry.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from scripts.controlled.common import (
    LAYOUTS,
    build_fine_graph,
    goal_embeddings,
    graph_step_size,
    load_goal_encoder,
    optimal_first_steps,
    run_path,
    sample_distinct_pairs,
    seed_summary,
    write_json,
)

ENV_LAYOUTS = {"simple_u_maze": "u_maze", "simple_open": "open"}
OPEN_PAIR_SEED = 20260725


def distance_metrics(embeddings, coordinates, distances, pairs) -> dict:
    source, goal = pairs
    latent = np.linalg.norm(embeddings[source] - embeddings[goal], axis=1)
    euclidean = np.linalg.norm(coordinates[source] - coordinates[goal], axis=1)
    graph = distances[source, goal]
    graph_rho = float(spearmanr(latent, graph).statistic)
    euclidean_rho = float(spearmanr(latent, euclidean).statistic)

    # Terciles of graph distance over the same pair set.
    cut_one, cut_two = np.quantile(graph, [1.0 / 3.0, 2.0 / 3.0])
    horizons = {}
    for label, mask in (
        ("short", graph <= cut_one),
        ("medium", (graph > cut_one) & (graph <= cut_two)),
        ("long", graph > cut_two),
    ):
        horizons[label] = {
            "pairs": int(np.count_nonzero(mask)),
            "graph_spearman": float(spearmanr(latent[mask], graph[mask]).statistic),
            "euclidean_spearman": float(
                spearmanr(latent[mask], euclidean[mask]).statistic
            ),
        }
    return {
        "pairs": int(len(source)),
        "graph_spearman": graph_rho,
        "euclidean_spearman": euclidean_rho,
        "topology_gain": graph_rho - euclidean_rho,
        "distance_horizons": horizons,
    }


def topology_critical_pairs(coordinates, adjacency, distances):
    """Directed pairs where no graph-optimal first move points toward the goal."""
    graph_step = graph_step_size(coordinates, adjacency)
    critical_source, critical_goal = [], []
    nodes = np.arange(len(coordinates))
    for source in nodes:
        goals = nodes[nodes != source]
        direct = coordinates[goals] - coordinates[source]
        direct /= np.linalg.norm(direct, axis=1, keepdims=True)
        neighbors = adjacency[source]
        delta = coordinates[neighbors] - coordinates[source]
        edge = np.linalg.norm(delta, axis=1) / graph_step
        neighbor_direction = delta / np.linalg.norm(delta, axis=1, keepdims=True)
        optimal = np.isclose(
            edge[:, None] + distances[neighbors][:, goals],
            distances[source, goals][None, :],
            atol=1e-7,
        )
        alignment = neighbor_direction @ direct.T
        best = np.max(np.where(optimal, alignment, -np.inf), axis=0)
        keep = best <= 1e-9
        critical_source.append(np.full(np.count_nonzero(keep), source, dtype=np.int32))
        critical_goal.append(goals[keep].astype(np.int32))
    return np.concatenate(critical_source), np.concatenate(critical_goal)


def local_descent_metrics(encoder, variables, coordinates, adjacency, distances, source, goal):
    """Alignment of -grad_x ||psi(x) - psi(g)|| with graph-optimal first moves."""

    def learned_distance(xy, goal_xy):
        psi_x = encoder.apply(variables, xy[None])[0]
        psi_goal = encoder.apply(variables, goal_xy[None])[0]
        return jnp.linalg.norm(psi_x - psi_goal)

    gradient = np.asarray(
        jax.vmap(jax.grad(learned_distance))(
            jnp.asarray(coordinates[source]), jnp.asarray(coordinates[goal])
        )
    )
    descent = -gradient
    descent /= np.linalg.norm(descent, axis=1, keepdims=True) + 1e-12
    graph_step = graph_step_size(coordinates, adjacency)
    best = np.array(
        [
            np.max(
                optimal_first_steps(coordinates, adjacency, distances, src, dst, graph_step)
                @ descent[index]
            )
            for index, (src, dst) in enumerate(zip(source, goal, strict=True))
        ]
    )
    return {
        "directed_pairs": int(len(source)),
        "mean_best_cosine": float(np.mean(best)),
        "positive_rate": float(np.mean(best > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--agents", nargs="+", default=["crl", "dcp"])
    parser.add_argument("--envs", nargs="+", default=list(ENV_LAYOUTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123, 256, 1234])
    parser.add_argument("--subdivisions", type=int, default=8)
    parser.add_argument("--open-pairs", type=int, default=100_000)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    graphs, pairs, critical = {}, {}, {}
    for env in args.envs:
        coordinates, adjacency, distances = build_fine_graph(
            LAYOUTS[ENV_LAYOUTS[env]], args.subdivisions
        )
        graphs[env] = (coordinates, adjacency, distances)
        if env == "simple_u_maze":
            pairs[env] = np.triu_indices(len(coordinates), k=1)
            critical[env] = topology_critical_pairs(coordinates, adjacency, distances)
            print(
                f"{env}: nodes={len(coordinates)} pairs={len(pairs[env][0])} "
                f"critical_pairs={len(critical[env][0])}"
            )
        else:
            pairs[env] = sample_distinct_pairs(
                np.random.default_rng(OPEN_PAIR_SEED), len(coordinates), args.open_pairs
            )
            print(f"{env}: nodes={len(coordinates)} pairs={args.open_pairs}")

    rows = []
    for env in args.envs:
        coordinates, adjacency, distances = graphs[env]
        for agent_name in args.agents:
            for seed in args.seeds:
                run_dir = run_path(args.runs_dir, f"{agent_name}_{env}", seed)
                agent, run, encoder, variables = load_goal_encoder(run_dir)
                if run.env != env:
                    raise ValueError(f"{run_dir}: trained on {run.env}, expected {env}")
                embeddings = goal_embeddings(encoder, variables, coordinates)
                row = {
                    "env": env,
                    "agent": agent_name,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "distance": distance_metrics(embeddings, coordinates, distances, pairs[env]),
                }
                if env in critical:
                    row["local_descent"] = local_descent_metrics(
                        encoder, variables, coordinates, adjacency, distances, *critical[env]
                    )
                rows.append(row)
                line = (
                    f"{env} {agent_name} seed={seed} "
                    f"graph_rho={row['distance']['graph_spearman']:.3f} "
                    f"euclid_rho={row['distance']['euclidean_spearman']:.3f}"
                )
                if "local_descent" in row:
                    line += f" descent={100 * row['local_descent']['positive_rate']:.1f}%"
                print(line)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["env"], row["agent"])].append(row)
    metrics = {
        "graph_spearman": lambda r: r["distance"]["graph_spearman"],
        "euclidean_spearman": lambda r: r["distance"]["euclidean_spearman"],
        "long_horizon_graph_spearman": lambda r: r["distance"]["distance_horizons"]["long"][
            "graph_spearman"
        ],
        "local_descent_positive_rate": lambda r: r["local_descent"]["positive_rate"],
    }
    summary = {}
    print("\nmean +/- sample sd over seeds")
    for (env, agent_name), items in grouped.items():
        summary[f"{env}/{agent_name}"] = {
            name: seed_summary({r["seed"]: get(r) for r in items})
            for name, get in metrics.items()
            if name != "local_descent_positive_rate" or "local_descent" in items[0]
        }
        print(
            f"{env} {agent_name}: "
            + " ".join(
                f"{name}={s['mean']:.3f}+/-{s['sample_sd'] or 0:.3f}"
                for name, s in summary[f"{env}/{agent_name}"].items()
            )
        )
    write_json(args.json_out, {"subdivisions": args.subdivisions, "rows": rows, "summary": summary})


if __name__ == "__main__":
    main()
