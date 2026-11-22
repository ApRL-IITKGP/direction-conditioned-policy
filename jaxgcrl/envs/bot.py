import os
import xml.etree.ElementTree as ET

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

from jaxgcrl.envs.simple_maze import (
    BOTTLENECK_MAZE,
    MAZE_HEIGHT,
    BIG_MAZE,
    DECEPTIVE_MAZE,
    HARDEST_MAZE,
    OPEN,
    S_CORRIDOR_MAZE,
    TWO_ROUTE_MAZE,
    U_MAZE,
    find_goals,
    find_starts,
)

# Differential-drive (unicycle) mobile robot: a nonholonomic toy environment.
#
# Action space: a = (v, w) in [-1,1]^2 — forward thrust along the current heading
# and yaw torque. The bot cannot produce lateral force: in world frame the
# controllable subspace at state s is C(s) = span{(cos th, sin th), rotation},
# which instantiates the controllable-subspace failure characterisation
# (Theorem 3) with an *analytically known* C(s). Residual lateral velocity is
# braked in the control mapping to emulate wheel lateral friction.
#
# Open + maze layouts reuse the simple_maze machinery, so ψ-landscape analysis
# is directly comparable between the holonomic point bot (SimpleMaze) and this
# nonholonomic bot on identical geometry.


def make_bot_world(maze_layout_name, maze_size_scaling):
    if maze_layout_name == "open":
        maze_layout = OPEN
    elif maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
    elif maze_layout_name == "deceptive_maze":
        maze_layout = DECEPTIVE_MAZE
    elif maze_layout_name == "two_route_maze":
        maze_layout = TWO_ROUTE_MAZE
    elif maze_layout_name == "s_corridor_maze":
        maze_layout = S_CORRIDOR_MAZE
    elif maze_layout_name == "bottleneck_maze":
        maze_layout = BOTTLENECK_MAZE
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")

    xml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "bot.xml")

    possible_starts = find_starts(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                ET.SubElement(
                    worldbody,
                    "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f"
                    % (
                        i * maze_size_scaling,
                        j * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    size="%f %f %f"
                    % (
                        0.5 * maze_size_scaling,
                        0.5 * maze_size_scaling,
                        MAZE_HEIGHT / 2 * maze_size_scaling,
                    ),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)

    return xml_string, possible_starts, possible_goals


class Bot(PipelineEnv):
    """Nonholonomic differential-drive bot in an open room or maze."""

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        lateral_friction=2.0,
        reset_noise_scale=0.1,
        backend="spring",
        maze_layout_name="open",
        maze_size_scaling=4.0,
        **kwargs,
    ):
        xml_string, possible_starts, possible_goals = make_bot_world(maze_layout_name, maze_size_scaling)

        sys = mjcf.loads(xml_string)
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._lateral_friction = lateral_friction
        self._reset_noise_scale = reset_noise_scale

        # obs = [x, y, th, vx, vy, w | target_xy]
        self.state_dim = 6
        self.goal_indices = jnp.array([0, 1])
        self.goal_reach_thresh = 0.5

    @property
    def action_size(self) -> int:
        return 2  # (v, w) — underlying sys has 3 actuators, mapped in step()

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng1, (self.sys.qd_size(),))

        start = self._random_start(rng2)
        q = q.at[:2].set(start)
        theta0 = jax.random.uniform(rng4, (), minval=-jnp.pi, maxval=jnp.pi)
        q = q.at[2].set(theta0)

        target = self._random_target(rng3)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_ctrl": zero,
            "x_position": zero,
            "y_position": zero,
            "heading": zero,
            "distance_from_origin": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        """Map (v, w) into world-frame joint controls, then step the pipeline.

        Forward thrust acts only along the heading; lateral velocity is braked
        (wheels don't slide sideways), so translation is confined to the
        heading direction — the nonholonomic constraint of a wheeled robot.
        """
        pipeline_state0 = state.pipeline_state
        theta = pipeline_state0.q[2]
        heading = jnp.array([jnp.cos(theta), jnp.sin(theta)])
        lateral = jnp.array([-jnp.sin(theta), jnp.cos(theta)])

        v, w = action[0], action[1]
        v_lat = jnp.dot(pipeline_state0.qd[:2], lateral)
        force_xy = v * heading - self._lateral_friction * v_lat * lateral
        ctrl = jnp.array([force_xy[0], force_xy[1], w])

        pipeline_state = self.pipeline_step(pipeline_state0, ctrl)

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self._get_obs(pipeline_state)
        dist = jnp.linalg.norm(obs[:2] - obs[-2:])
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(dist < 2.0, dtype=float)
        reward = -dist - ctrl_cost

        state.metrics.update(
            reward_ctrl=-ctrl_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            heading=theta,
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=jnp.zeros(()))

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]
        target_pos = pipeline_state.x.pos[-1][:2]
        return jnp.concatenate([qpos] + [qvel] + [target_pos])

    def _random_target(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]

    def _random_start(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]
