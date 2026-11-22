"""Shared helpers for the controlled studies.

Run loading, network construction, the weighted occupancy graph of the
point-mass layouts, and the direction input of a trained DCP actor.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from brax.io import model
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

from jaxgcrl.agents.crl.networks import Actor, Encoder
from jaxgcrl.envs.simple_maze import OPEN, U_MAZE

# Width of one maze cell in world units.
SCALING = 4.0
LAYOUTS = {"open": OPEN, "u_maze": U_MAZE}


# Runs and networks


def run_path(runs_dir: Path, name: str, seed: int) -> Path:
    """Directory written by `run.py --exp-name <name> --seed <seed>`."""
    return Path(runs_dir) / f"run_{name}_s_{seed}"


def load_run(run_dir: Path):
    """Returns the saved (agent config, run config) of a training run."""
    with (Path(run_dir) / "args.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    return saved["agent"], saved["run"]


def load_params(run_dir: Path):
    """Final checkpoint: (alpha params, actor params, critic params)."""
    return model.load_params(Path(run_dir) / "ckpt" / "final")


def make_encoder(agent) -> Encoder:
    return Encoder(
        repr_dim=agent.repr_dim,
        network_width=agent.h_dim,
        network_depth=agent.n_hidden,
        skip_connections=agent.skip_connections,
        use_relu=agent.use_relu,
        use_ln=agent.use_ln,
    )


def make_actor(agent, action_size: int) -> Actor:
    return Actor(
        action_size=action_size,
        network_width=agent.h_dim,
        network_depth=agent.n_hidden,
        skip_connections=agent.skip_connections,
        use_relu=agent.use_relu,
    )


def load_goal_encoder(run_dir: Path):
    """Returns (agent, run, encoder, goal-encoder parameters)."""
    agent, run = load_run(run_dir)
    encoder = make_encoder(agent)
    variables = load_params(run_dir)[2]["g_encoder"]
    return agent, run, encoder, variables


def require(run, agent, agent_name: str, env_prefix: str) -> None:
    if type(agent).__name__ != agent_name or not run.env.startswith(env_prefix):
        raise ValueError(
            f"expected {agent_name} on {env_prefix}*, got "
            f"{type(agent).__name__} on {run.env}"
        )


def goal_embeddings(encoder: Encoder, variables, coordinates: np.ndarray) -> np.ndarray:
    return np.asarray(encoder.apply(variables, jnp.asarray(coordinates)))


# Occupancy graph


def layout_from_env(env_name: str) -> tuple[str, list[list[object]]]:
    for prefix in ("simple_", "bot_"):
        if env_name.startswith(prefix):
            name = env_name[len(prefix):]
            if name in LAYOUTS:
                return name, LAYOUTS[name]
    raise ValueError(f"unsupported environment: {env_name!r}")


def build_fine_graph(
    layout: list[list[object]], subdivisions: int
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Returns node coordinates, 8-neighbor adjacency, and graph distances.

    Every free maze cell is split into `subdivisions**2` nodes. Cardinal edges
    have weight 1 and diagonal edges weight sqrt(2); a diagonal is allowed only
    when both adjacent cardinal nodes are free, so paths cannot cut corners.
    Distances are in units of the node spacing `SCALING / subdivisions`.
    """
    if subdivisions < 2:
        raise ValueError("subdivisions must be at least 2")
    rows, cols = len(layout) * subdivisions, len(layout[0]) * subdivisions
    free = np.zeros((rows, cols), dtype=bool)
    for row in range(len(layout)):
        for col in range(len(layout[0])):
            if layout[row][col] != 1:
                free[
                    row * subdivisions : (row + 1) * subdivisions,
                    col * subdivisions : (col + 1) * subdivisions,
                ] = True

    cells = np.argwhere(free)
    index = {tuple(cell): idx for idx, cell in enumerate(cells)}
    adjacency: list[np.ndarray] = []
    edge_rows, edge_cols, edge_weights = [], [], []
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for row, col in cells:
        source = index[(row, col)]
        neighbors = []
        for d_row, d_col in offsets:
            neighbor = (row + d_row, col + d_col)
            if neighbor not in index:
                continue
            if d_row and d_col:
                if not (free[row + d_row, col] and free[row, col + d_col]):
                    continue
                weight = np.sqrt(2.0)
            else:
                weight = 1.0
            target = index[neighbor]
            neighbors.append(target)
            edge_rows.append(source)
            edge_cols.append(target)
            edge_weights.append(weight)
        adjacency.append(np.asarray(neighbors, dtype=np.int32))

    graph = sparse.csr_matrix(
        (edge_weights, (edge_rows, edge_cols)), shape=(len(cells), len(cells))
    )
    distances = shortest_path(graph, directed=False, unweighted=False)
    if not np.isfinite(distances).all():
        raise AssertionError("occupancy graph is disconnected")
    # Maze cell (row, col) is centered at (row, col) * SCALING.
    coordinates = ((cells.astype(np.float64) + 0.5) / subdivisions - 0.5) * SCALING
    return coordinates, adjacency, distances


def graph_step_size(coordinates: np.ndarray, adjacency: list[np.ndarray]) -> float:
    """World length of one cardinal graph edge."""
    return float(
        min(
            np.linalg.norm(coordinates[neighbors] - coordinates[source], axis=1).min()
            for source, neighbors in enumerate(adjacency)
            if len(neighbors)
        )
    )


def optimal_first_steps(
    coordinates: np.ndarray,
    adjacency: list[np.ndarray],
    distances: np.ndarray,
    source: int,
    goal: int,
    graph_step: float,
) -> np.ndarray:
    """Unit world directions of every graph-optimal first move from source."""
    neighbors = adjacency[int(source)]
    delta = coordinates[neighbors] - coordinates[int(source)]
    edge = np.linalg.norm(delta, axis=1) / graph_step
    optimal = np.isclose(
        edge + distances[neighbors, int(goal)],
        distances[int(source), int(goal)],
        atol=1e-7,
    )
    directions = delta[optimal]
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)


def nearest_nodes(points: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    squared = np.sum((np.asarray(points)[:, None, :] - coordinates[None]) ** 2, axis=2)
    nodes = np.argmin(squared, axis=1)
    return nodes, np.sqrt(squared[np.arange(len(points)), nodes])


def sample_distinct_pairs(
    rng: np.random.Generator, node_count: int, count: int, dtype=np.int64
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform ordered pairs of distinct nodes."""
    source = rng.integers(0, node_count, size=count, dtype=dtype)
    goal = rng.integers(0, node_count - 1, size=count, dtype=dtype)
    goal += goal >= source
    return source, goal


def free_macro_centers(layout) -> np.ndarray:
    """World centers of all free maze cells."""
    return np.asarray(
        [
            (row * SCALING, col * SCALING)
            for row in range(len(layout))
            for col in range(len(layout[0]))
            if layout[row][col] != 1
        ],
        dtype=np.float32,
    )


# Fixed-policy rollouts


def initial_states(env, seed: int, count: int, set_q):
    """Batched reset whose positions are overwritten by `set_q(q)`, at rest."""
    keys = jax.random.split(jax.random.PRNGKey(seed), count)
    state = jax.jit(jax.vmap(env.reset))(keys)
    q = set_q(state.pipeline_state.q)
    qd = jnp.zeros_like(state.pipeline_state.qd)
    pipeline_state = jax.vmap(env.pipeline_init)(q, qd)
    obs = jax.vmap(env._get_obs)(pipeline_state)
    return state.replace(pipeline_state=pipeline_state, obs=obs)


def _unit(x):
    return x / (jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def dcp_actor_input(
    encoder,
    encoder_params,
    obs,
    state_dim: int,
    goal_indices,
    use_magnitude: bool,
    intervention: str = "correct",
    shift: int = 1,
):
    """DCP actor input [state, direction, magnitude] with an optional edit.

    The direction is the unit vector from psi(q(s)) to psi(g). Interventions
    change only the direction or magnitude:
      flip                 negated direction
      wrong_goal           direction toward the goal of the case `shift` rows away
      orthogonal           unit vector orthogonal to the learned direction
      zero_direction       zero vector
      zero_magnitude       zero magnitude
      shuffled_magnitude   magnitude of the case `shift` rows away
    """
    physical = obs[:, :state_dim]
    goal = obs[:, state_dim:]
    psi_state = encoder.apply(encoder_params, physical[:, goal_indices])
    gap = encoder.apply(encoder_params, goal) - psi_state
    magnitude = jnp.linalg.norm(gap, axis=1, keepdims=True)
    direction = gap / (magnitude + 1e-8)

    if intervention == "flip":
        direction = -direction
    elif intervention == "wrong_goal":
        wrong_goal = jnp.roll(goal, shift, axis=0)
        direction = _unit(encoder.apply(encoder_params, wrong_goal) - psi_state)
    elif intervention == "orthogonal":
        candidate = jnp.roll(direction, 1, axis=1)
        candidate -= jnp.sum(candidate * direction, axis=1, keepdims=True) * direction
        direction = _unit(candidate)
    elif intervention == "zero_direction":
        direction = jnp.zeros_like(direction)
    elif intervention == "zero_magnitude":
        magnitude = jnp.zeros_like(magnitude)
    elif intervention == "shuffled_magnitude":
        magnitude = jnp.roll(magnitude, shift, axis=0)
    elif intervention != "correct":
        raise ValueError(intervention)

    pieces = [physical, direction]
    if use_magnitude:
        pieces.append(magnitude)
    return jnp.concatenate(pieces, axis=1)


# Aggregation and output


def seed_summary(values_by_seed: dict) -> dict:
    """Mean and sample standard deviation across seeds."""
    values = np.asarray(list(values_by_seed.values()), dtype=float)
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "values_by_seed": {str(k): float(v) for k, v in values_by_seed.items()},
    }


def write_json(path: Path | None, payload) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
