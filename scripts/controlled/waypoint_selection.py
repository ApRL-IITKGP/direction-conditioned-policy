"""Waypoint selection by fixed trained DCP goal encoders in the U-maze.

Start, goal, and candidate positions are drawn uniformly from the occupancy
graph, and every ranking rule receives the identical draws. A candidate z is
forward on-path when D(s,z) + D(z,g) - D(s,g) = 0 and D(z,g) < D(s,g), with D
the graph distance. For each rule this reports how often a forward on-path
candidate is available and selected, the selected candidate's relative slack
[D(s,z) + D(z,g) - D(s,g)] / D(s,g), and its relative progress
[D(s,g) - D(z,g)] / D(s,g), overall and by graph-distance tercile.

Ranking rules:
  uniform        first candidate of the uniform draw
  inner_product  argmax <psi(z), psi(g)>, the DCP scorer
  cosine         argmax cos(psi(z) - psi(s), psi(g) - psi(s))
  latent_slack   argmin ||psi(z) - psi(s)|| + ||psi(g) - psi(z)||

A second test uses nested candidate sets (the first N of one draw of 128 per
trial) to measure how availability and inner-product selection scale with N.

Example:
    python -m scripts.controlled.waypoint_selection --runs-dir runs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.controlled.common import (
    LAYOUTS,
    build_fine_graph,
    goal_embeddings,
    load_goal_encoder,
    require,
    run_path,
    sample_distinct_pairs,
    write_json,
)

RANDOM_SEED = 20260725


def selector_choices(psi_source, psi_goal, psi_candidate) -> dict[str, np.ndarray]:
    inner_product = np.einsum("tcd,td->tc", psi_candidate, psi_goal)
    gap_goal = psi_goal - psi_source
    gap_candidate = psi_candidate - psi_source[:, None, :]
    candidate_norm = np.linalg.norm(gap_candidate, axis=2)
    cosine = np.einsum("tcd,td->tc", gap_candidate, gap_goal) / (
        candidate_norm * np.linalg.norm(gap_goal, axis=1)[:, None] + 1e-12
    )
    cosine[candidate_norm < 1e-8] = -np.inf
    latent_slack = candidate_norm + np.linalg.norm(psi_candidate - psi_goal[:, None, :], axis=2)
    return {
        "uniform": np.zeros(len(psi_source), dtype=np.int64),
        "inner_product": np.argmax(inner_product, axis=1),
        "cosine": np.argmax(cosine, axis=1),
        "latent_slack": np.argmin(latent_slack, axis=1),
    }


def summarize_choices(source, goal, candidates, choices, distances) -> dict:
    rows = np.arange(len(source))
    d_sg = distances[source, goal].astype(np.float64)
    slack = (
        distances[source[:, None], candidates]
        + distances[candidates, goal[:, None]]
        - d_sg[:, None]
    )
    progress = d_sg[:, None] - distances[candidates, goal[:, None]]
    useful = np.isclose(slack, 0.0, atol=1e-7) & (progress > 1e-7)
    available = np.any(useful, axis=1)
    best_slack = np.min(slack, axis=1)

    cut_one, cut_two = np.quantile(d_sg, [1.0 / 3.0, 2.0 / 3.0])
    masks = {
        "all": np.ones(len(source), dtype=bool),
        "short": d_sg <= cut_one,
        "medium": (d_sg > cut_one) & (d_sg <= cut_two),
        "long": d_sg > cut_two,
    }
    payload = {
        "trials": len(source),
        "candidate_count": candidates.shape[1],
        "horizon_cutoffs": [float(cut_one), float(cut_two)],
        "useful_candidate_availability": float(np.mean(available)),
        "selectors": {},
    }
    for name, choice in choices.items():
        chosen_slack = slack[rows, choice]
        chosen_progress = progress[rows, choice]
        chosen_useful = useful[rows, choice]
        by_horizon = {}
        for horizon, mask in masks.items():
            if not np.any(mask):
                by_horizon[horizon] = {"trials": 0}
                continue
            given = mask & available
            by_horizon[horizon] = {
                "trials": int(np.count_nonzero(mask)),
                "useful_candidate_availability": float(np.mean(available[mask])),
                "exact_on_path_rate": float(np.mean(np.isclose(chosen_slack[mask], 0.0, atol=1e-7))),
                "useful_on_path_rate": float(np.mean(chosen_useful[mask])),
                "capture_given_available": (
                    float(np.mean(chosen_useful[given])) if np.any(given) else None
                ),
                "median_absolute_slack": float(np.median(chosen_slack[mask])),
                "p90_absolute_slack": float(np.quantile(chosen_slack[mask], 0.90)),
                "median_relative_slack": float(np.median(chosen_slack[mask] / d_sg[mask])),
                "p90_relative_slack": float(np.quantile(chosen_slack[mask] / d_sg[mask], 0.90)),
                "mean_relative_progress": float(np.mean(chosen_progress[mask] / d_sg[mask])),
                "positive_progress_rate": float(np.mean(chosen_progress[mask] > 1e-7)),
                "median_slack_regret_to_best_candidate": float(
                    np.median(chosen_slack[mask] - best_slack[mask])
                ),
            }
        payload["selectors"][name] = by_horizon
    return payload


def fixed_candidate_test(embeddings, distances, trials, candidate_count) -> dict:
    rng = np.random.default_rng(RANDOM_SEED)
    source, goal = sample_distinct_pairs(rng, len(embeddings), trials, dtype=np.int32)
    candidates = rng.integers(0, len(embeddings), size=(trials, candidate_count), dtype=np.int32)
    choices = selector_choices(embeddings[source], embeddings[goal], embeddings[candidates])
    return summarize_choices(source, goal, candidates, choices, distances)


def nested_candidate_test(embeddings, distances, trials, counts) -> list[dict]:
    rng = np.random.default_rng(RANDOM_SEED)
    source, goal = sample_distinct_pairs(rng, len(embeddings), trials, dtype=np.int32)
    candidates = rng.integers(0, len(embeddings), size=(trials, max(counts)), dtype=np.int32)

    # Fraction of all nodes that are forward on-path; with uniform draws, N
    # candidates contain one with probability 1 - (1 - fraction)^N.
    d_sg = distances[source, goal][:, None]
    nodes = np.arange(len(embeddings))
    to_goal = distances[nodes, goal[:, None]]
    slack = distances[source[:, None], nodes] + to_goal - d_sg
    useful_fraction = np.mean(
        np.isclose(slack, 0.0, atol=1e-7) & (d_sg - to_goal > 1e-7), axis=1
    )
    rows = []
    for count in counts:
        candidate = candidates[:, :count]
        choices = selector_choices(embeddings[source], embeddings[goal], embeddings[candidate])
        quality = summarize_choices(source, goal, candidate, choices, distances)
        rows.append(
            {
                "candidate_count": count,
                "predicted_useful_availability": float(
                    np.mean(1.0 - (1.0 - useful_fraction) ** count)
                ),
                "useful_candidate_availability": quality["useful_candidate_availability"],
                "inner_product": quality["selectors"]["inner_product"]["all"],
                "uniform": quality["selectors"]["uniform"]["all"],
            }
        )
    return rows


def load_embeddings(run_dir: Path, coordinates: np.ndarray):
    agent, run, encoder, variables = load_goal_encoder(run_dir)
    require(run, agent, "DCP", "simple_u_maze")
    return goal_embeddings(encoder, variables, coordinates)


def value_range(values) -> str:
    return f"{100 * min(values):.1f}-{100 * max(values):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123, 256, 1234])
    parser.add_argument("--nested-seeds", type=int, nargs="+", default=[1, 42, 123])
    parser.add_argument("--subdivisions", type=int, default=8)
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--candidate-count", type=int, default=32)
    parser.add_argument(
        "--candidate-counts", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    coordinates, _, distances = build_fine_graph(LAYOUTS["u_maze"], args.subdivisions)
    print(f"u_maze graph: subdivisions={args.subdivisions} nodes={len(coordinates)}")
    embeddings = {
        seed: load_embeddings(run_path(args.runs_dir, "dcp_simple_u_maze", seed), coordinates)
        for seed in dict.fromkeys(args.seeds + args.nested_seeds)
    }

    fixed = {
        seed: fixed_candidate_test(embeddings[seed], distances, args.trials, args.candidate_count)
        for seed in args.seeds
    }
    print(f"\nFixed candidate sets: {args.trials} trials x {args.candidate_count} candidates")
    for seed, result in fixed.items():
        line = f"seed={seed} available={100 * result['useful_candidate_availability']:.1f}%"
        for name, rows in result["selectors"].items():
            a = rows["all"]
            line += (
                f" | {name}: on_path={100 * a['useful_on_path_rate']:.1f}% "
                f"slack={a['median_relative_slack']:.3f} progress={a['mean_relative_progress']:+.3f}"
            )
        print(line)
    print("range across seeds:")
    for name in ("uniform", "inner_product", "cosine", "latent_slack"):
        rows = [r["selectors"][name]["all"] for r in fixed.values()]
        print(
            f"  {name:<14} on_path={value_range([r['useful_on_path_rate'] for r in rows])} "
            f"median_relative_slack={min(r['median_relative_slack'] for r in rows):.3f}-"
            f"{max(r['median_relative_slack'] for r in rows):.3f} "
            f"mean_relative_progress={min(r['mean_relative_progress'] for r in rows):+.3f}-"
            f"{max(r['mean_relative_progress'] for r in rows):+.3f}"
        )

    counts = sorted(set(args.candidate_counts))
    nested = {
        seed: nested_candidate_test(embeddings[seed], distances, args.trials, counts)
        for seed in args.nested_seeds
    }
    print("\nNested candidate sets: availability and inner-product on-path selection by N")
    for index, count in enumerate(counts):
        rows = [nested[seed][index] for seed in args.nested_seeds]
        print(
            f"N={count:<4} available={100 * np.mean([r['useful_candidate_availability'] for r in rows]):.1f}% "
            f"(predicted {100 * np.mean([r['predicted_useful_availability'] for r in rows]):.1f}%) "
            "inner_product="
            + " ".join(f"{100 * r['inner_product']['useful_on_path_rate']:.1f}%" for r in rows)
        )

    write_json(
        args.json_out,
        {
            "subdivisions": args.subdivisions,
            "nodes": len(coordinates),
            "fixed_candidates": {str(seed): result for seed, result in fixed.items()},
            "nested_candidates": {str(seed): rows for seed, rows in nested.items()},
        },
    )


if __name__ == "__main__":
    main()
