"""Planar puck pushing with a force-controlled point agent.

The agent must push an unactuated puck to a target. The goal encoder receives puck
coordinates (``goal_indices = [2, 3]``); the puck only moves after agent--puck contact.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp


_BOUNDARY_WALLS = (
    # (x, y, half_x, half_y)
    (-4.75, 0.0, 0.25, 5.0),
    (4.75, 0.0, 0.25, 5.0),
    (0.0, -4.75, 5.0, 0.25),
    (0.0, 4.75, 5.0, 0.25),
)

_WALL_GAP = (
    (0.0, -3.0, 0.22, 1.75),
    (0.0, 3.0, 0.22, 1.75),
)


def _configs(scenario: str) -> jax.Array:
    """Returns rows ``[agent_x,y, puck_x,y, target_x,y]``."""
    if scenario == "open":
        return jnp.array(
            [
                [-2.0, 0.0, 0.0, 0.0, 2.5, 0.0],
                [2.0, 0.0, 0.0, 0.0, -2.5, 0.0],
                [0.0, -2.0, 0.0, 0.0, 0.0, 2.5],
                [0.0, 2.0, 0.0, 0.0, 0.0, -2.5],
                [-2.0, -2.0, 0.0, 0.0, 2.5, 2.5],
                [2.0, 2.0, 0.0, 0.0, -2.5, -2.5],
                [-2.0, 2.0, 0.0, 0.0, 2.5, -2.5],
                [2.0, -2.0, 0.0, 0.0, -2.5, 2.5],
            ],
            dtype=jnp.float32,
        )
    if scenario == "reposition":
        # The agent begins between puck and target. Successful transport
        # requires first moving around to the opposite side of the puck.
        return jnp.array(
            [
                [-1.0, 0.0, 0.0, 0.0, -3.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, -3.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 3.0],
                [-0.8, -0.8, 0.0, 0.0, -2.6, -2.6],
                [0.8, 0.8, 0.0, 0.0, 2.6, 2.6],
                [-0.8, 0.8, 0.0, 0.0, -2.6, 2.6],
                [0.8, -0.8, 0.0, 0.0, 2.6, -2.6],
            ],
            dtype=jnp.float32,
        )
    if scenario == "wall_gap":
        # Agent and puck start in the left chamber. The target is in the
        # right chamber; successful puck paths must pass through the gap.
        return jnp.array(
            [
                [-3.7, -2.5, -2.5, -2.5, 2.8, 2.5],
                [-3.7, 2.5, -2.5, 2.5, 2.8, -2.5],
                [-3.7, -1.8, -2.5, -1.8, 3.0, 1.8],
                [-3.7, 1.8, -2.5, 1.8, 3.0, -1.8],
            ],
            dtype=jnp.float32,
        )
    raise ValueError(f"Unknown push-bot scenario: {scenario}")


def _make_world(scenario: str) -> bytes:
    xml_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "assets",
        "push_bot.xml",
    )
    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")
    walls = _BOUNDARY_WALLS + (_WALL_GAP if scenario == "wall_gap" else ())
    for index, (x, y, half_x, half_y) in enumerate(walls):
        ET.SubElement(
            worldbody,
            "geom",
            name=f"wall_{index}",
            pos=f"{x} {y} 0.5",
            size=f"{half_x} {half_y} 0.5",
            type="box",
            contype="1",
            conaffinity="1",
            friction="1.2 0.4 0.4",
            rgba="0.35 0.28 0.22 1",
        )
    return ET.tostring(tree.getroot())


class PushBot(PipelineEnv):
    """Holonomic locomotor pushing an unactuated puck in a planar workspace."""

    def __init__(
        self,
        scenario: str = "open",
        backend: str = "spring",
        reset_noise_scale: float = 0.05,
        ctrl_cost_weight: float = 0.02,
        **kwargs,
    ):
        self.scenario = scenario
        self.configs = _configs(scenario)
        sys = mjcf.loads(_make_world(scenario))
        n_frames = 10
        if backend in ("spring", "positional"):
            sys = sys.tree_replace({"opt.timestep": 0.005})
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self._reset_noise_scale = reset_noise_scale
        self._ctrl_cost_weight = ctrl_cost_weight
        # state = [agent_xy, puck_xy, agent_vxy, puck_vxy], goal = target_xy
        self.state_dim = 8
        self.goal_indices = jnp.array([2, 3])
        self.goal_reach_thresh = 0.55

    def reset(self, rng: jax.Array) -> State:
        rng, config_key, position_key, velocity_key = jax.random.split(rng, 4)
        index = jax.random.randint(config_key, (), 0, len(self.configs))
        configuration = self.configs[index]
        position_noise = jax.random.uniform(
            position_key,
            (4,),
            minval=-self._reset_noise_scale,
            maxval=self._reset_noise_scale,
        )

        q = self.sys.init_q
        q = q.at[:4].set(configuration[:4] + position_noise)
        q = q.at[4:6].set(configuration[4:6])
        qd = jnp.zeros((self.sys.qd_size(),))
        qd = qd.at[:4].set(
            0.01 * jax.random.normal(velocity_key, (4,))
        )
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        zero = jnp.zeros(())
        metrics = {
            "success": zero,
            "success_easy": zero,
            "puck_goal_dist": zero,
            "agent_puck_dist": zero,
            "contact": zero,
            "puck_displacement": zero,
            "agent_displacement": zero,
            "goal_actuation_ratio": zero,
            "behind_alignment": zero,
            "reward_ctrl": zero,
        }
        return State(pipeline_state, obs, zero, zero, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        before = state.pipeline_state
        pipeline_state = self.pipeline_step(before, action)
        obs = self._get_obs(pipeline_state)

        agent_before, puck_before = before.q[:2], before.q[2:4]
        target_before = before.q[4:6]
        agent_xy, puck_xy, target_xy = (
            pipeline_state.q[:2],
            pipeline_state.q[2:4],
            pipeline_state.q[4:6],
        )
        agent_displacement = math.safe_norm(agent_xy - agent_before)
        puck_displacement = math.safe_norm(puck_xy - puck_before)
        previous_puck_goal_dist = math.safe_norm(
            target_before - puck_before
        )
        agent_puck = puck_xy - agent_xy
        puck_goal = target_xy - puck_xy
        agent_puck_dist = math.safe_norm(agent_puck)
        puck_goal_dist = math.safe_norm(puck_goal)
        behind_alignment = jnp.dot(
            agent_puck / (agent_puck_dist + 1e-8),
            puck_goal / (puck_goal_dist + 1e-8),
        )
        contact = jnp.asarray(agent_puck_dist < 0.85, dtype=jnp.float32)
        success = jnp.asarray(
            puck_goal_dist < self.goal_reach_thresh, dtype=jnp.float32
        )
        success_easy = jnp.asarray(puck_goal_dist < 1.25, dtype=jnp.float32)
        ctrl_cost = self._ctrl_cost_weight * jnp.square(action).sum()
        reward = -puck_goal_dist - ctrl_cost

        state.metrics.update(
            success=success,
            success_easy=success_easy,
            puck_goal_dist=puck_goal_dist,
            agent_puck_dist=agent_puck_dist,
            contact=contact,
            puck_displacement=puck_displacement,
            agent_displacement=agent_displacement,
            # Bounded fraction of puck motion directed toward the target.
            # The previous displacement ratio was undefined when the agent
            # was stationary and could explode to O(1e8), making it unusable
            # as a diagnostic.  This metric is in [-1, 1]: +1 means all puck
            # motion reduced target distance, -1 means it moved directly
            # away, and 0 means no goal-directed puck motion.
            goal_actuation_ratio=jnp.clip(
                (previous_puck_goal_dist - puck_goal_dist)
                / (puck_displacement + 1e-8),
                -1.0,
                1.0,
            ),
            behind_alignment=behind_alignment,
            reward_ctrl=-ctrl_cost,
        )
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=jnp.zeros(()),
        )

    def _get_obs(self, pipeline_state) -> jax.Array:
        return jnp.concatenate(
            [
                pipeline_state.q[:4],
                pipeline_state.qd[:4],
                pipeline_state.q[4:6],
            ]
        )
