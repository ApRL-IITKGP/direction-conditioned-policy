"""Success summary for the four-way factorial (CRL, Direct-DCP, SSGC, DCP).

Reads the evaluation history that `run.py` logs during training (one line per
evaluation, `step: N, eval/episode_success_any: v +/- s`) from
`<run_dir>/train.log`, as written by `train_point_mass.sh`. For each run it
reports the mean of the evaluation checkpoints ("mean success over training")
and related summaries, then aggregates across seeds with a 95% Student-t
confidence interval. Paired per-seed contrasts isolate the collection target
(scoring) and the actor input (direction).

Example:
    python -m scripts.controlled.factorial_table --runs-dir runs
    python -m scripts.controlled.factorial_table --envs push_bot_open \
        --variants crl dcp --seeds 1 42 123
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from scripts.controlled.common import run_path, write_json

VARIANTS = {"crl": "CRL", "direct_dcp": "Direct-DCP", "ssgc": "SSGC", "dcp": "DCP"}
METRICS = ("eval/episode_success_any", "eval/episode_success")
METRIC_PATTERN = re.compile(
    r"step: ([0-9]+), ([^:]+): ([-+]?(?:[0-9]*[.])?[0-9]+(?:[eE][-+]?[0-9]+)?)"
)
CONTRASTS = {
    "direction_without_scoring": {"direct_dcp": 1, "crl": -1},
    "direction_with_scoring": {"dcp": 1, "ssgc": -1},
    "scoring_with_raw_actor": {"ssgc": 1, "crl": -1},
    "scoring_with_direction_actor": {"dcp": 1, "direct_dcp": -1},
    "interaction": {"dcp": 1, "direct_dcp": -1, "ssgc": -1, "crl": 1},
    "dcp_minus_crl": {"dcp": 1, "crl": -1},
}


def history(log: Path, key: str) -> list[tuple[int, float]]:
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        match = METRIC_PATTERN.search(line)
        if match and match.group(2) == key:
            rows.append((int(match.group(1)), float(match.group(3))))
    return rows


def curve_summary(rows: list[tuple[int, float]]) -> dict:
    values = [value for _, value in rows]
    return {
        "auc_mean": float(np.mean(values)),
        "first_half_mean": float(np.mean(values[: len(values) // 2])),
        "last5_mean": float(np.mean(values[-5:])),
        "last5_minus_previous5": float(np.mean(values[-5:]) - np.mean(values[-10:-5])),
        "final": values[-1],
        "best": max(values),
        "history": [{"step": step, "value": value} for step, value in rows],
    }


def seed_interval(values: list[float]) -> dict:
    """Mean with a 95% Student-t interval across seeds."""
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    half_width = float(student_t.ppf(0.975, len(array) - 1) * sd / math.sqrt(len(array)))
    return {
        "mean": mean,
        "sample_sd": sd,
        "ci95_half_width": half_width,
        "ci95": [mean - half_width, mean + half_width],
        "positive": int(np.sum(array > 0)),
        "negative": int(np.sum(array < 0)),
        "values": array.tolist(),
    }


def actor_parameter_count(checkpoint: Path) -> int:
    with checkpoint.open("rb") as handle:
        params = pickle.load(handle)
    # CRL/DCP save (alpha, actor, critic); SSGC saves the actor parameters.
    actor = params[1] if isinstance(params, tuple) else params
    return sum(int(np.asarray(leaf).size) for leaf in _leaves(actor))


def _leaves(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _leaves(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from _leaves(value)
    else:
        yield tree


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--envs", nargs="+", default=["simple_open", "simple_u_maze"])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123, 256, 1234])
    parser.add_argument("--log-name", default="train.log")
    parser.add_argument("--num-evals", type=int, default=20)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    runs = {}
    for env in args.envs:
        for variant in args.variants:
            for seed in args.seeds:
                run_dir = run_path(args.runs_dir, f"{variant}_{env}", seed)
                metrics = {}
                for key in METRICS:
                    rows = history(run_dir / args.log_name, key)
                    if len(rows) != args.num_evals:
                        raise ValueError(
                            f"{run_dir / args.log_name}: expected {args.num_evals} "
                            f"{key} evaluations, found {len(rows)}"
                        )
                    metrics[key] = curve_summary(rows)
                runs[(env, variant, seed)] = {
                    "env": env,
                    "variant": variant,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "actor_parameters": actor_parameter_count(run_dir / "ckpt" / "final"),
                    "metrics": metrics,
                }

    def values(env, variant, field="auc_mean", metric="eval/episode_success_any"):
        return [runs[(env, variant, s)]["metrics"][metric][field] for s in args.seeds]

    aggregate = {
        env: {variant: seed_interval(values(env, variant)) for variant in args.variants}
        for env in args.envs
    }
    contrasts = {}
    for env in args.envs:
        contrasts[env] = {}
        for name, weights in CONTRASTS.items():
            if not set(weights) <= set(args.variants):
                continue
            per_seed = np.sum(
                [w * np.asarray(values(env, v)) for v, w in weights.items()], axis=0
            )
            contrasts[env][name] = seed_interval(per_seed.tolist())

    print("Mean success over training (eval/episode_success_any), mean +/- 95% t half-width")
    print(f"{'Method':<12}" + "".join(f"{env:>22}" for env in args.envs))
    for variant in args.variants:
        cells = [aggregate[env][variant] for env in args.envs]
        print(
            f"{VARIANTS[variant]:<12}"
            + "".join(f"{c['mean']:>14.3f} +/- {c['ci95_half_width']:.3f}" for c in cells)
        )
    print("\nPaired contrasts across seeds: mean [95% CI] (positive/negative seeds)")
    for env in args.envs:
        for name, c in contrasts[env].items():
            print(
                f"{env:<16} {name:<30} {c['mean']:+.3f} "
                f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] ({c['positive']}/{c['negative']})"
            )
    counts = {
        variant: sorted({runs[(e, variant, s)]["actor_parameters"] for e in args.envs for s in args.seeds})
        for variant in args.variants
    }
    print("\nActor parameters: " + ", ".join(f"{VARIANTS[v]}={c}" for v, c in counts.items()))

    write_json(
        args.json_out,
        {
            "seeds": args.seeds,
            "runs": list(runs.values()),
            "aggregate": aggregate,
            "contrasts": contrasts,
            "actor_parameters": counts,
        },
    )


if __name__ == "__main__":
    main()
