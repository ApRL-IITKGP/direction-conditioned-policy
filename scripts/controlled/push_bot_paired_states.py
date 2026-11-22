"""Paired-state puck-pushing evaluation of trained DCP and CRL policies.

Cases come in quartets: within a quartet the puck and target are identical and
the robot starts behind, left of, right of, or in front of the puck, so the
goal (and DCP's goal-space direction) is the same while the physical state
differs. Quartets vary the target direction, target distance, absolute
position, and robot-puck separation. Each case is rolled out for one episode
and the contact-to-success chain is recorded: any contact, contact from the
correct side, puck movement, goal progress, and success. DCP is also rolled
out with reversed, orthogonal, and zero direction inputs.

Example:
    python -m scripts.controlled.push_bot_paired_states --runs-dir runs
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
    run_path,
    seed_summary,
    write_json,
)

DCP_MODES = ("correct", "flip", "orthogonal", "zero_direction")
POSES = ("behind", "side_left", "side_right", "front")
CHAIN = (
    "contact_rate",
    "correct_side_contact_rate",
    "puck_movement_rate",
    "goal_progress_rate",
    "success_rate",
)
RESET_SEED = 20260727


def quartet_configurations(directions: int, repeats: int):
    """Rows of (robot xy, puck xy, target xy) and the robot pose label of each."""
    configurations, poses = [], []
    for angle in np.linspace(0.0, 2.0 * np.pi, directions, endpoint=False):
        goal_unit = np.asarray([np.cos(angle), np.sin(angle)])
        perpendicular = np.asarray([-goal_unit[1], goal_unit[0]])
        for repeat in range(repeats):
            repeat_angle = 2.0 * np.pi * repeat / repeats
            puck = 0.20 * np.asarray([np.cos(repeat_angle), np.sin(repeat_angle)])
            target_distance = 2.6 + 0.4 * (repeat % 4) / 3.0
            contact_radius = 0.90 + 0.30 * ((repeat // 4) % 4) / 3.0
            target = puck + target_distance * goal_unit
            robots = {
                "behind": puck - contact_radius * goal_unit,
                "side_left": puck + contact_radius * perpendicular,
                "side_right": puck - contact_radius * perpendicular,
                "front": puck + contact_radius * goal_unit,
            }
            for pose in POSES:
                configurations.append(np.concatenate((robots[pose], puck, target)))
                poses.append(pose)
    return np.asarray(configurations, dtype=np.float32), np.asarray(poses)


def summarize(values: dict, mask: np.ndarray) -> dict:
    contacted, correct_side = values["contacted"], values["correct_side"]
    moved = values["puck_path"] > 0.25
    progressed = values["best_progress"] > 0.25

    def rate(x, given=None):
        selected = mask if given is None else mask & given
        return float(np.mean(x[selected])) if np.any(selected) else None

    return {
        "cases": int(np.count_nonzero(mask)),
        "contact_rate": rate(contacted),
        "correct_side_contact_rate": rate(correct_side),
        "correct_side_given_contact": rate(correct_side, contacted),
        "puck_movement_rate": rate(moved),
        "movement_given_correct_side": rate(moved, correct_side),
        "goal_progress_rate": rate(progressed),
        "progress_given_movement": rate(progressed, moved),
        "success_rate": rate(values["reached"]),
        "success_given_progress": rate(values["reached"], progressed),
        "mean_puck_path": rate(values["puck_path"]),
        "mean_positive_goal_progress": rate(values["positive_progress"]),
        "mean_best_goal_progress": rate(values["best_progress"]),
        "mean_net_goal_progress": rate(values["net_progress"]),
        "median_final_goal_distance": float(np.median(values["final_goal_distance"][mask])),
        "mean_first_contact_if_contacted": rate(values["first_contact"], contacted),
    }


def evaluate(run_dir: Path, horizon: int, directions: int, repeats: int) -> dict:
    agent, run = load_run(run_dir)
    agent_name = type(agent).__name__
    if agent_name not in ("DCP", "CRL") or run.env != "push_bot_open":
        raise ValueError(f"expected DCP or CRL on push_bot_open, got {agent_name} on {run.env}")
    env = create_env(run.env, backend=run.backend)
    actor = make_actor(agent, env.action_size)
    params = load_params(run_dir)
    actor_params = params[1]
    state_dim = env.state_dim
    goal_indices = jnp.asarray(env.goal_indices, dtype=jnp.int32)
    if agent_name == "DCP":
        encoder = make_encoder(agent)
        encoder_params = params[2]["g_encoder"]
        modes = DCP_MODES
    else:
        modes = ("native",)

    configurations, poses = quartet_configurations(directions, repeats)
    count = len(configurations)
    initial_state = initial_states(
        env, RESET_SEED, count, lambda q: q.at[:, :6].set(jnp.asarray(configurations))
    )

    def actions(state, mode):
        if mode == "native":
            inputs = state.obs  # CRL: raw state and goal
        else:
            inputs = dcp_actor_input(
                encoder, encoder_params, state.obs, state_dim, goal_indices,
                agent.use_magnitude, mode,
            )
        mean, _ = actor.apply(actor_params, inputs)
        return nn.tanh(mean)

    batched_step = jax.vmap(env.step)
    initial_puck = initial_state.obs[:, 2:4]
    initial_goal_distance = jnp.linalg.norm(initial_puck - initial_state.obs[:, state_dim:], axis=1)

    def rollout(mode):
        def step(carry, _):
            (state, reached, contacted, correct_side, first_contact, puck_path,
             positive_progress, best_progress, previous_puck, previous_distance, t) = carry
            next_state = batched_step(state, actions(state, mode))
            contact = next_state.metrics["contact"] > 0
            first_contact = jnp.where(contact & ~contacted, t, first_contact)
            contacted |= contact
            correct_side |= contact & (next_state.metrics["behind_alignment"] > 0.5)
            reached |= next_state.metrics["success"] > 0
            puck = next_state.obs[:, 2:4]
            distance = jnp.linalg.norm(puck - next_state.obs[:, state_dim:], axis=1)
            puck_path += jnp.linalg.norm(puck - previous_puck, axis=1)
            positive_progress += jnp.maximum(previous_distance - distance, 0.0)
            best_progress = jnp.maximum(best_progress, initial_goal_distance - distance)
            return (next_state, reached, contacted, correct_side, first_contact, puck_path,
                    positive_progress, best_progress, puck, distance, t + 1), None

        false = jnp.zeros((count,), dtype=bool)
        zero = jnp.zeros((count,))
        carry = (
            initial_state, false, false, false,
            jnp.full((count,), horizon, dtype=jnp.int32),
            zero, zero, zero, initial_puck, initial_goal_distance,
            jnp.asarray(0, dtype=jnp.int32),
        )
        final, _ = jax.lax.scan(step, carry, None, length=horizon)
        (_, reached, contacted, correct_side, first_contact, puck_path,
         positive_progress, best_progress, _, final_distance, _) = final
        return {
            "reached": reached,
            "contacted": contacted,
            "correct_side": correct_side,
            "first_contact": first_contact,
            "puck_path": puck_path,
            "positive_progress": positive_progress,
            "best_progress": best_progress,
            "net_progress": initial_goal_distance - final_distance,
            "final_goal_distance": final_distance,
        }

    result = {
        "run_dir": str(run_dir),
        "agent": agent_name,
        "seed": run.seed,
        "cases": count,
        "quartets": count // len(POSES),
        "modes": {},
    }
    for mode in modes:
        values = {k: np.asarray(v) for k, v in jax.jit(rollout, static_argnums=(0,))(mode).items()}
        rows = {"all": summarize(values, np.ones(count, dtype=bool))}
        for pose in POSES:
            rows[pose] = summarize(values, poses == pose)
        result["modes"][mode] = rows
        a = rows["all"]
        print(
            f"{agent_name} seed={run.seed} {mode:<15} contact={a['contact_rate']:.3f} "
            f"correct_side={a['correct_side_contact_rate']:.3f} "
            f"progress={a['goal_progress_rate']:.3f} success={a['success_rate']:.3f}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 42, 123])
    parser.add_argument("--dcp-run-dirs", type=Path, nargs="+")
    parser.add_argument("--crl-run-dirs", type=Path, nargs="+")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--directions", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    def dirs(agent, explicit):
        return explicit or [run_path(args.runs_dir, f"{agent}_push_bot_open", s) for s in args.seeds]

    dcp, crl = (
        [evaluate(d, args.horizon, args.directions, args.repeats) for d in dirs(agent, explicit)]
        for agent, explicit in (("dcp", args.dcp_run_dirs), ("crl", args.crl_run_dirs))
    )

    def chain(results, mode):
        return {
            metric: seed_summary({r["seed"]: r["modes"][mode]["all"][metric] for r in results})
            for metric in CHAIN
        }

    def paired(first, first_mode, second, second_mode):
        return {
            metric: seed_summary(
                {
                    a["seed"]: a["modes"][first_mode]["all"][metric]
                    - b["modes"][second_mode]["all"][metric]
                    for a, b in zip(first, second, strict=True)
                }
            )
            for metric in CHAIN
        }

    summary = {f"dcp_{mode}": chain(dcp, mode) for mode in DCP_MODES}
    summary["crl"] = chain(crl, "native")
    differences = {"dcp_minus_crl": paired(dcp, "correct", crl, "native")}
    for mode in DCP_MODES[1:]:
        differences[f"dcp_correct_minus_{mode}"] = paired(dcp, "correct", dcp, mode)

    print("\nmean over seeds (%)")
    labels = ("contact", "correct side", "puck moved", "goal progress", "success")
    print(f"{'':<20}" + "".join(f"{label:>15}" for label in labels))
    for name, rows in summary.items():
        print(f"{name:<20}" + "".join(f"{100 * rows[m]['mean']:>15.1f}" for m in CHAIN))
    write_json(
        args.json_out,
        {"dcp": dcp, "crl": crl, "summary": summary, "paired_differences": differences},
    )


if __name__ == "__main__":
    main()
