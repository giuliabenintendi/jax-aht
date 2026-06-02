"""Multi-Destination Spread environment (image observations).

A four-agent fully cooperative gridworld on a 5x5 grid (Multi-Destination Spread).
All agents spawn on the central cell and must spread out to cover four distinct
goals placed symmetrically around the centre. The reward is shared and shaped by
the number of *distinct* goals covered on the current step: 4 goals -> 10, 3 -> 5,
2 -> 2, 1 -> 1, 0 -> 0. The reward is dense (paid every step) and there is no early
termination — each episode runs the full `max_steps` (100 in the paper). Because
the layout is symmetric the agent->goal assignment is ambiguous, which is the
source of the large self-play vs cross-play gap this task is designed to expose.

Each agent has 9 actions: the eight grid directions plus stay. A move off-grid is
clipped to the boundary; when several agents would land on the same cell, all of
them are reverted to their previous positions (a single, non-iterated revert, so
agents may still overlap on the cells they came from — including the shared centre
at spawn).

Observations are ego-centric RGB renders, flattened to a vector in `[0, 1]`: the
ego agent red, the partners blue, goals green, walls grey, on a white background.
The partners share one colour (the image, unlike the paper's identity-indexed
vector obs, does not distinguish individual partners). The grid/tile conventions
mirror the Dual Destination and card-game envs so the same image (JA-)IPPO
pipeline applies (`grid_height`, `grid_width`, `tile_size`, `_img_h`, `_img_w`,
`num_scalar_obs` are exposed). Joint-attention wiring is intentionally out of scope.
"""
from __future__ import annotations

from functools import partial
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv, WrappedEnvState
from envs.multi_destination_spread.layouts import LAYOUTS, Layout, layout_to_arrays

TILE_SIZE = 7
AGENT_RADIUS = 2
NUM_ACTIONS = 9  # N, NE, E, SE, S, SW, W, NW, stay

# Movement deltas in [x, y], y increasing downward (image coords), indexed by
# action: eight compass directions clockwise from north, then stay.
ACTION_TO_DIR = jnp.array(
    [
        [0, -1],   # N
        [1, -1],   # NE
        [1, 0],    # E
        [1, 1],    # SE
        [0, 1],    # S
        [-1, 1],   # SW
        [-1, 0],   # W
        [-1, -1],  # NW
        [0, 0],    # stay
    ],
    dtype=jnp.int32,
)

# Shared reward by number of distinct goals covered (index 0..4): {0:0,1:1,2:2,3:5,4:10}.
REWARD_BY_COVERAGE = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=jnp.float32)

_RED = jnp.array([255, 0, 0], dtype=jnp.uint8)
_BLUE = jnp.array([0, 0, 255], dtype=jnp.uint8)
_GREEN = jnp.array([0, 255, 0], dtype=jnp.uint8)
_WHITE = jnp.array([255, 255, 255], dtype=jnp.uint8)
_GREY = jnp.array([100, 100, 100], dtype=jnp.uint8)


@dataclass
class MultiDestinationSpreadState:
    agent_pos: chex.Array  # (num_agents, 2) int32, [x, y] per agent
    time: chex.Array       # scalar int32


class MultiDestinationSpreadEnv(BaseEnv):
    """Four-agent cooperative spread gridworld with flattened image observations."""

    def __init__(
        self,
        layout: str | Layout = "cardinal_5x5",
        max_steps: int = 100,
        random_reset: bool = False,
        **kwargs,  # noqa: ARG002 - absorbs obs_type and other passthrough kwargs
    ):
        if isinstance(layout, str):
            layout = LAYOUTS[layout]
        h, w, n, wall_map, goal_pos, agent_start = layout_to_arrays(layout)

        self.height = int(h)
        self.width = int(w)
        self.wall_map = wall_map          # (h, w) bool
        self.goal_pos = goal_pos          # (num_goals, 2) int32 [x, y]
        self.agent_start = agent_start    # (num_agents, 2) int32 [x, y]
        self.max_steps = max_steps
        self.random_reset = random_reset

        self.num_agents = int(n)
        self.num_goals = int(goal_pos.shape[0])
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.name = "MultiDestinationSpread"

        # Image dims, exposed for the (JA-)IPPO image pipeline.
        self.tile_size = TILE_SIZE
        self.grid_height = self.height
        self.grid_width = self.width
        self._img_h = self.height * TILE_SIZE
        self._img_w = self.width * TILE_SIZE
        self.num_scalar_obs = 0
        self._obs_dim = self._img_h * self._img_w * 3

        # Free cells (not wall, not goal) for optional random agent placement.
        occupied = self.wall_map.at[goal_pos[:, 1], goal_pos[:, 0]].set(True)
        self._free_prob = (~occupied).reshape(-1).astype(jnp.float32)
        self._free_prob = self._free_prob / self._free_prob.sum()

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

    def observation_space(self, agent: str) -> jaxmarl_spaces.Box:
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str) -> jaxmarl_spaces.Discrete:
        return jaxmarl_spaces.Discrete(num_categories=NUM_ACTIONS)

    @property
    def action_dim(self) -> int:
        return NUM_ACTIONS

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> tuple[dict[str, chex.Array], WrappedEnvState]:
        agent_pos = self._sample_agent_pos(key) if self.random_reset else self.agent_start
        env_state = MultiDestinationSpreadState(agent_pos=agent_pos, time=jnp.int32(0))
        obs = self._make_obs(env_state)
        return obs, WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=jnp.zeros(self.num_agents),
            step=jnp.int32(0),
        )

    def _sample_agent_pos(self, key: chex.PRNGKey) -> jnp.ndarray:
        """Place agents on distinct free cells, weighting by `_free_prob`."""
        flats = jax.random.choice(
            key, self.height * self.width, shape=(self.num_agents,),
            replace=False, p=self._free_prob,
        )
        xs = flats % self.width
        ys = flats // self.width
        return jnp.stack([xs, ys], axis=-1).astype(jnp.int32)

    def _move_and_reward(
        self, agent_pos: jnp.ndarray, actions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply moves with wall/collision handling and the coverage reward.

        Returns `(next_pos, reward, num_covered)`. A move into a wall or off-grid is
        reverted; then any agents whose targets coincide are reverted to their
        previous cells (single pass — agents may still overlap on those cells).
        """
        next_pos = agent_pos + ACTION_TO_DIR[actions]
        next_pos = next_pos.at[:, 0].set(jnp.clip(next_pos[:, 0], 0, self.width - 1))
        next_pos = next_pos.at[:, 1].set(jnp.clip(next_pos[:, 1], 0, self.height - 1))

        hits_wall = self.wall_map[next_pos[:, 1], next_pos[:, 0]]
        next_pos = jnp.where(hits_wall[:, None], agent_pos, next_pos)

        # Revert every agent that shares its target cell with another agent.
        same_target = jnp.all(
            next_pos[:, None, :] == next_pos[None, :, :], axis=-1
        )  # (n, n)
        collides = same_target.sum(axis=1) > 1
        next_pos = jnp.where(collides[:, None], agent_pos, next_pos).astype(jnp.int32)

        # Distinct goals covered: a goal counts if at least one agent stands on it.
        on = jnp.all(
            next_pos[:, None, :] == self.goal_pos[None, :, :], axis=-1
        )  # (n_agents, n_goals)
        goal_covered = jnp.any(on, axis=0)  # (n_goals,)
        num_covered = goal_covered.sum().astype(jnp.int32)
        reward = REWARD_BY_COVERAGE[num_covered]
        return next_pos, reward, num_covered

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> tuple[dict[str, chex.Array], WrappedEnvState, dict[str, float], dict[str, bool], dict]:
        key, key_reset = jax.random.split(key)
        env_state = state.env_state
        acts = jnp.array([actions[a] for a in self.agents]).astype(jnp.int32)

        next_pos, reward_val, num_covered = self._move_and_reward(env_state.agent_pos, acts)
        new_time = env_state.time + 1
        # Dense reward, no early termination — episodes only end on timeout.
        done = new_time >= self.max_steps
        all_covered = num_covered >= self.num_goals

        new_env_state = MultiDestinationSpreadState(agent_pos=next_pos, time=new_time)
        obs_st = self._make_obs(new_env_state)

        base_reward_arr = jnp.full((self.num_agents,), reward_val)
        base_return = state.base_return_so_far + base_reward_arr
        state_st = WrappedEnvState(
            env_state=new_env_state,
            base_return_so_far=base_return,
            avail_actions=jnp.zeros(self.num_agents),
            step=new_time,
        )

        reward = {agent: reward_val for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        info = {
            "base_reward": base_reward_arr,
            "base_return": base_return,
            "num_goals_covered": jnp.broadcast_to(num_covered, (self.num_agents,)),
            "task_success": jnp.broadcast_to(all_covered, (self.num_agents,)),
            "step_count": jnp.broadcast_to(new_time, (self.num_agents,)),
        }

        obs_reset, state_reset = self.reset(key_reset)
        obs, new_state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y),
            (obs_reset, state_reset),
            (obs_st, state_st),
        )
        new_state = new_state.replace(
            base_return_so_far=jax.lax.select(
                done, jnp.zeros(self.num_agents), new_state.base_return_so_far,
            )
        )
        return obs, new_state, reward, dones, info

    def _make_obs(self, state: MultiDestinationSpreadState) -> dict[str, jnp.ndarray]:
        obs = {}
        for i in range(self.num_agents):
            img = self._render_agent_view(state, i)
            obs[self.agents[i]] = img.flatten().astype(jnp.float32) / 255.0
        return obs

    def _render_agent_view(self, state: MultiDestinationSpreadState, agent_idx: int) -> jnp.ndarray:
        """Ego-centric uint8 RGB render: ego red, partners blue, goals green, walls grey.

        Partners are drawn first and the ego last, so the ego is always visible even
        when agents overlap (e.g. all on the centre cell at spawn).
        """
        tile_img = jnp.tile(_WHITE[None, None, :], (self.height, self.width, 1))
        tile_img = jnp.where(self.wall_map[..., None], _GREY[None, None, :], tile_img)
        tile_img = tile_img.at[self.goal_pos[:, 1], self.goal_pos[:, 0], :].set(_GREEN)
        img = jnp.repeat(jnp.repeat(tile_img, TILE_SIZE, axis=0), TILE_SIZE, axis=1)
        for j in range(self.num_agents):
            if j != agent_idx:
                img = self._draw_circle(img, state.agent_pos[j], _BLUE)
        img = self._draw_circle(img, state.agent_pos[agent_idx], _RED)
        return img

    def _draw_circle(self, img: jnp.ndarray, pos_xy: jnp.ndarray, color: jnp.ndarray) -> jnp.ndarray:
        cx = pos_xy[0] * TILE_SIZE + TILE_SIZE // 2
        cy = pos_xy[1] * TILE_SIZE + TILE_SIZE // 2
        yy, xx = jnp.meshgrid(
            jnp.arange(self._img_h), jnp.arange(self._img_w), indexing="ij",
        )
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= AGENT_RADIUS ** 2
        return jnp.where(mask[..., None], color[None, None, :], img)

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> dict[str, jnp.ndarray]:
        mask = jnp.ones(NUM_ACTIONS, dtype=jnp.float32)
        return {agent: mask for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.ndarray:
        return state.step


## Tests

# Action indices used by the tests (see ACTION_TO_DIR above).
_N, _NE, _E, _SE, _S, _SW, _W, _NW, _STAY = range(9)


def test_obs_shapes_and_spaces():
    env = MultiDestinationSpreadEnv(layout="cardinal_5x5", max_steps=20)
    obs, state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert env.num_agents == 4
    for a in env.agents:
        assert obs[a].shape == (env._obs_dim,)
    assert env._obs_dim == 5 * TILE_SIZE * 5 * TILE_SIZE * 3
    assert float(obs["agent_0"].min()) >= 0.0 and float(obs["agent_0"].max()) <= 1.0
    assert env.action_space("agent_0").n == NUM_ACTIONS
    # All agents spawn on the same centre cell, so the ego views coincide there;
    # once they spread out (N/W/E/S) each sees a distinct frame (own cell red).
    obs2, _, _, _, _ = jax.jit(env.step)(
        jax.random.PRNGKey(1), state,
        {"agent_0": _N, "agent_1": _W, "agent_2": _E, "agent_3": _S},
    )
    assert not bool(jnp.allclose(obs2["agent_0"], obs2["agent_1"]))


def test_observation_is_rendered_view():
    """The obs is the flattened, [0,1]-normalised ego render; save a montage to inspect.

    Run `uv run pytest -s -k observation_is_rendered_view
    envs/multi_destination_spread/multi_destination_spread.py` to write
    artifacts/multi_destination_spread_obs.png and eyeball what the agents see.
    """
    env = MultiDestinationSpreadEnv(layout="cardinal_5x5", max_steps=100)
    obs, state = env.reset(jax.random.PRNGKey(0))
    recon = (obs["agent_0"].reshape(env._img_h, env._img_w, 3) * 255).astype(jnp.uint8)
    assert jnp.array_equal(recon, env._render_agent_view(state.env_state, 0))

    from envs.multi_destination_spread.rendering import save_observation_montage
    out = save_observation_montage(env, "artifacts/multi_destination_spread_obs.png")
    print(f"[multi_destination_spread] saved observation montage -> {out}")


def test_all_goals_covered_gives_max_reward():
    """From the centre, one step out to all four cardinal goals -> reward 10, no done."""
    env = MultiDestinationSpreadEnv(layout="cardinal_5x5", max_steps=100)
    _, state = env.reset(jax.random.PRNGKey(0))
    # Goals are 2 steps away; place agents adjacent first, then step onto them.
    # Centre (2,2): move agent_0 N twice -> (2,0); agent_1 W twice -> (0,2);
    # agent_2 E twice -> (4,2); agent_3 S twice -> (2,4).
    acts = {"agent_0": _N, "agent_1": _W, "agent_2": _E, "agent_3": _S}
    _, state, reward, dones, info = jax.jit(env.step)(jax.random.PRNGKey(1), state, acts)
    # After one step nobody is on a goal yet (intermediate cells).
    assert float(reward["agent_0"]) == 0.0
    _, state, reward, dones, info = jax.jit(env.step)(jax.random.PRNGKey(2), state, acts)
    assert int(info["num_goals_covered"][0]) == 4
    assert float(reward["agent_0"]) == 10.0
    assert bool(info["task_success"][0])
    assert not bool(dones["__all__"])  # dense reward, no early termination


def test_partial_coverage_reward_table():
    """Two distinct goals covered -> reward 2."""
    env = MultiDestinationSpreadEnv(layout="cardinal_5x5", max_steps=100)
    _, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    # agent_0 -> N goal, agent_1 -> W goal; agent_2/agent_3 stay near centre.
    acts = {"agent_0": _N, "agent_1": _W, "agent_2": _STAY, "agent_3": _STAY}
    _, state, _, _, _ = step(jax.random.PRNGKey(1), state, acts)
    _, state, reward, _, info = step(jax.random.PRNGKey(2), state, acts)
    assert int(info["num_goals_covered"][0]) == 2
    assert float(reward["agent_0"]) == 2.0


def test_collision_reverts_to_previous_cell():
    """Two agents targeting the same cell are both reverted to where they were."""
    env = MultiDestinationSpreadEnv(layout="cardinal_5x5", max_steps=100)
    _, state = env.reset(jax.random.PRNGKey(0))
    # All start at centre (2,2). agent_0 stays, agent_1 moves W then back E onto
    # agent_0's cell. Simpler: agent_0 N, agent_1 N -> both target (2,1) -> revert.
    acts = {"agent_0": _N, "agent_1": _N, "agent_2": _STAY, "agent_3": _STAY}
    _, state, _, _, _ = jax.jit(env.step)(jax.random.PRNGKey(1), state, acts)
    pos = state.env_state.agent_pos
    # agent_0 and agent_1 both wanted (2,1); both reverted to the centre (2,2).
    assert tuple(int(v) for v in pos[0]) == (2, 2)
    assert tuple(int(v) for v in pos[1]) == (2, 2)


def test_timeout_triggers_auto_reset():
    env = MultiDestinationSpreadEnv(layout="cardinal_5x5", max_steps=3)
    _, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    stay = {a: _STAY for a in env.agents}
    for _ in range(2):
        _, state, _, dones, _ = step(jax.random.PRNGKey(2), state, stay)
        assert not bool(dones["__all__"])
    _, state, _, dones, _ = step(jax.random.PRNGKey(2), state, stay)
    assert bool(dones["__all__"])
    assert int(state.env_state.time) == 0  # auto-reset


def test_vmap_reset_step_and_random_reset():
    env = MultiDestinationSpreadEnv(layout="cardinal_5x5", max_steps=20, random_reset=True)
    n = 8
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    obs, state = jax.vmap(env.reset)(keys)
    assert obs["agent_0"].shape == (n, env._obs_dim)
    acts = {a: jnp.zeros(n, dtype=jnp.int32) for a in env.agents}
    obs2, _, reward, _, _ = jax.vmap(env.step)(keys, state, acts)
    assert obs2["agent_0"].shape == (n, env._obs_dim)
    assert reward["agent_0"].shape == (n,)
