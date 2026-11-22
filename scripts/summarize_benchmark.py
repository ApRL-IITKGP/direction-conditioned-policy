"""Summarize benchmark runs: final success and time near goal, mean over seeds.

Reads runs/run_<method>_<env>_s<seed>_s_<seed>/metrics.jsonl written during training and
reports, at the final evaluation, success (% of episodes reaching the goal at least once)
and time near goal (steps in the goal region per 1,000-step episode), with 95% Student-t
confidence-interval half-widths across seeds.

    uv run python scripts/summarize_benchmark.py --methods crl ssgc dcp direct_dcp
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

TASKS = (
    ("ant_big_maze", "AntMaze Big"),
    ("ant_hardest_maze", "AntMaze Hardest"),
    ("ant_u_maze", "Ant U-Maze"),
    ("humanoid_u_maze", "Humanoid U-Maze"),
    ("ant_push", "AntPush"),
    ("ant_ball", "AntSoccer"),
    ("pusher_easy", "PusherEasy"),
    ("pusher_hard", "PusherHard"),
    ("pusher_hard_far", "PusherHard Far"),
)
SEEDS = (1, 42, 123, 256, 1234)


def final_metrics(run_dir: Path):
    path = run_dir / "metrics.jsonl"
    if not (run_dir / "ckpt" / "final").exists() or not path.exists():
        return None
    last = json.loads(path.read_text().strip().splitlines()[-1])
    return 100.0 * last["eval/episode_success_any"], last["eval/episode_success"]


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values.mean(), float("nan")
    return values.mean(), t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", default=["crl", "ssgc", "dcp", "direct_dcp"])
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    summary = {}
    header = f"{'Task':<17}" + "".join(f"{m:>28}" for m in args.methods)
    print("Success (%) / time near goal (steps), mean +- 95% CI half-width")
    print(header)
    for env, name in TASKS:
        row = f"{name:<17}"
        for method in args.methods:
            results = [
                final_metrics(args.runs / f"run_{method}_{env}_s{seed}_s_{seed}") for seed in SEEDS
            ]
            results = [r for r in results if r is not None]
            if not results:
                row += f"{'-':>28}"
                continue
            success, near = zip(*results)
            (s_mean, s_ci), (n_mean, n_ci) = mean_ci(success), mean_ci(near)
            summary.setdefault(env, {})[method] = {
                "seeds": len(results),
                "success": s_mean,
                "success_ci": s_ci,
                "time_near_goal": n_mean,
                "time_near_goal_ci": n_ci,
            }
            cell = f"{s_mean:.1f}+-{s_ci:.1f} / {n_mean:.0f}+-{n_ci:.0f}"
            if len(results) < len(SEEDS):
                cell += f" (n={len(results)})"
            row += f"{cell:>28}"
        print(row)

    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
