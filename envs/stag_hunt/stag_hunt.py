"""Stag Hunt environment (image observations).

A two-player temporally/spatially-extended Stag Hunt, ported from the Google
Research `social_rl` multigrid StagHunt (Lee et al. 2021, "Joint Attention for
Multi-Agent Coordination and Social Learning"). Two agents move on an open grid
containing *plants* and *stags*:

  - Stepping onto a **plant** gives the stepping agent +`plant_reward` (default 1).
    Safe, individual.
  - Stepping onto a **stag** is the risky cooperative act: if the *other* agent is
    Manhattan-distance 1 from the stag at that moment, *both* agents receive
    +`stag_reward` (default 5); otherwise the lone agent is penalised
    -`penalty` (default 1). Either way the stag is consumed.

Consumed plants/stags respawn on a random free cell. Episodes run a fixed
`max_steps` (no terminal goal). This is the canonical risk-vs-payoff coordination
dilemma: the safe-solo berry equilibrium vs the payoff-dominant joint-stag one.

Observations are ego-centric RGB renders flattened to `[0, 1]` (ego red, partner
blue, stags green, plants yellow, walls grey, white bg). The grid/tile conventions
mirror Dual Destination so the same image / joint-attention pipeline applies
(`grid_height`, `grid_width`, `tile_size`, `_img_h`, `_img_w` are exposed).
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
from envs.stag_hunt.layouts import LAYOUTS, Layout, layout_to_arrays

TILE_SIZE = 7
AGENT_RADIUS = 2
NUM_ACTIONS = 5  # right, down, left, up, stay

# Movement deltas in [x, y], indexed by action.
ACTION_TO_DIR = jnp.array(
    [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]], dtype=jnp.int32
)

_RED = jnp.array([255, 0, 0], dtype=jnp.uint8)
_BLUE = jnp.array([0, 0, 255], dtype=jnp.uint8)
_GREEN = jnp.array([0, 200, 0], dtype=jnp.uint8)
_YELLOW = jnp.array([230, 210, 0], dtype=jnp.uint8)
_WHITE = jnp.array([255, 255, 255], dtype=jnp.uint8)
_GREY = jnp.array([100, 100, 100], dtype=jnp.uint8)


@dataclass
class StagHuntState:
    agent_pos: chex.Array   # (2, 2) int32, [x, y] per agent
    stag_pos: chex.Array    # (num_stags, 2) int32
    plant_pos: chex.Array   # (num_plants, 2) int32
    time: chex.Array        # scalar int32


class StagHuntEnv(BaseEnv):
    """Two-agent Stag Hunt gridworld with flattened image observations."""

    def __init__(
        self,
        layout: str | Layout = "empty_7x7",
        num_stags: int = 1,
        num_plants: int = 2,
        penalty: float = 1.0,
        stag_reward: float = 5.0,
        plant_reward: float = 1.0,
        max_steps: int = 100,
        random_reset: bool = True,  # noqa: ARG002 - StagHunt always regenerates each episode
        **kwargs,  # noqa: ARG002 - absorbs obs_type and other passthrough kwargs
    ):
        if isinstance(layout, str):
            layout = LAYOUTS[layout]
        h, w, wall_map = layout_to_arrays(layout)

        self.height = int(h)
        self.width = int(w)
        self.wall_map = wall_map          # (h, w) bool
        self.num_stags = int(num_stags)
        self.num_plants = int(num_plants)
        self.penalty = float(penalty)
        self.stag_reward = float(stag_reward)
        self.plant_reward = float(plant_reward)
        self.max_steps = max_steps

        self.num_agents = 2
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.name = "StagHunt"

        # Image dims, exposed for the (JA-)IPPO image pipeline.
        self.tile_size = TILE_SIZE
        self.grid_height = self.height
        self.grid_width = self.width
        self._img_h = self.height * TILE_SIZE
        self._img_w = self.width * TILE_SIZE
        self.num_scalar_obs = 0
        self._obs_dim = self._img_h * self._img_w * 3

        # Uniform sampling weight over non-wall cells.
        free = (~self.wall_map).reshape(-1).astype(jnp.float32)
        self._base_prob = free / free.sum()

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

    def observation_space(self, agent: str) -> jaxmarl_spaces.Box:
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str) -> jaxmarl_spaces.Discrete:
        return jaxmarl_spaces.Discrete(num_categories=NUM_ACTIONS)

    @property
    def action_dim(self) -> int:
        return NUM_ACTIONS

    def _flat_to_xy(self, flats: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([flats % self.width, flats // self.width], axis=-1).astype(jnp.int32)

    def _sample_cells(self, key: chex.PRNGKey, n: int, prob: jnp.ndarray) -> jnp.ndarray:
        """Sample `n` distinct cells (as [x, y]) weighted by `prob` over flat indices."""
        flats = jax.random.choice(
            key, self.height * self.width, shape=(n,), replace=False, p=prob
        )
        return self._flat_to_xy(flats)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> tuple[dict[str, chex.Array], WrappedEnvState]:
        n = self.num_agents + self.num_stags + self.num_plants
        cells = self._sample_cells(key, n, self._base_prob)  # (n, 2), distinct
        agent_pos = cells[: self.num_agents]
        stag_pos = cells[self.num_agents : self.num_agents + self.num_stags]
        plant_pos = cells[self.num_agents + self.num_stags :]

        env_state = StagHuntState(
            agent_pos=agent_pos, stag_pos=stag_pos, plant_pos=plant_pos, time=jnp.int32(0)
        )
        obs = self._make_obs(env_state)
        return obs, WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=jnp.zeros(self.num_agents),
            step=jnp.int32(0),
        )

    def _move(self, agent_pos: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """Cardinal move with wall / off-grid / agent-collision reverts."""
        next_pos = agent_pos + ACTION_TO_DIR[actions]
        next_pos = next_pos.at[:, 0].set(jnp.clip(next_pos[:, 0], 0, self.width - 1))
        next_pos = next_pos.at[:, 1].set(jnp.clip(next_pos[:, 1], 0, self.height - 1))
        hits_wall = self.wall_map[next_pos[:, 1], next_pos[:, 0]]
        next_pos = jnp.where(hits_wall[:, None], agent_pos, next_pos)
        would_collide = jnp.all(next_pos[0] == next_pos[1])
        next_pos = jnp.where(would_collide, agent_pos, next_pos)
        return next_pos.astype(jnp.int32)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,  # noqa: ARG002
    ) -> tuple[dict[str, chex.Array], WrappedEnvState, dict[str, float], dict[str, bool], dict]:
        key, key_respawn, key_reset = jax.random.split(key, 3)
        env_state = state.env_state
        acts = jnp.array([actions["agent_0"], actions["agent_1"]]).astype(jnp.int32)

        next_pos = self._move(env_state.agent_pos, acts)  # (2, 2)
        p0, p1 = next_pos[0], next_pos[1]

        # Plants: stepping onto one gives +plant_reward to that agent; respawn it.
        on_plant_0 = jnp.all(p0 == env_state.plant_pos, axis=-1)  # (P,)
        on_plant_1 = jnp.all(p1 == env_state.plant_pos, axis=-1)
        plant_eaten = on_plant_0 | on_plant_1
        r0_plant = self.plant_reward * jnp.sum(on_plant_0)
        r1_plant = self.plant_reward * jnp.sum(on_plant_1)

        # Stags: stepping onto one is cooperative iff the OTHER agent is Manhattan
        # distance 1 from the stag; then +stag_reward to BOTH, else -penalty to the
        # lone stepper. Consumed either way.
        on_stag_0 = jnp.all(p0 == env_state.stag_pos, axis=-1)  # (S,)
        on_stag_1 = jnp.all(p1 == env_state.stag_pos, axis=-1)
        dist0 = jnp.sum(jnp.abs(p0 - env_state.stag_pos), axis=-1)  # (S,) agent0 -> each stag
        dist1 = jnp.sum(jnp.abs(p1 - env_state.stag_pos), axis=-1)
        coop = (on_stag_0 & (dist1 == 1)) | (on_stag_1 & (dist0 == 1))  # (S,)
        stepped = on_stag_0 | on_stag_1
        solo = stepped & ~coop
        shared_stag = self.stag_reward * jnp.sum(coop)  # to both agents
        r0_stag = shared_stag - self.penalty * jnp.sum(on_stag_0 & solo)
        r1_stag = shared_stag - self.penalty * jnp.sum(on_stag_1 & solo)

        reward_arr = jnp.array([r0_plant + r0_stag, r1_plant + r1_stag])

        # Respawn consumed objects on free cells, avoiding the agents' new cells.
        agent_flats = next_pos[:, 1] * self.width + next_pos[:, 0]
        respawn_free = (~self.wall_map).reshape(-1).astype(jnp.float32).at[agent_flats].set(0.0)
        respawn_prob = respawn_free / respawn_free.sum()
        new_cells = self._sample_cells(
            key_respawn, self.num_plants + self.num_stags, respawn_prob
        )
        new_plant_pos = jnp.where(plant_eaten[:, None], new_cells[: self.num_plants], env_state.plant_pos)
        new_stag_pos = jnp.where(stepped[:, None], new_cells[self.num_plants :], env_state.stag_pos)

        new_time = env_state.time + 1
        done = new_time >= self.max_steps  # no terminal goal; fixed-length episodes

        new_env_state = StagHuntState(
            agent_pos=next_pos, stag_pos=new_stag_pos, plant_pos=new_plant_pos, time=new_time
        )
        obs_st = self._make_obs(new_env_state)

        base_return = state.base_return_so_far + reward_arr
        state_st = WrappedEnvState(
            env_state=new_env_state,
            base_return_so_far=base_return,
            avail_actions=jnp.zeros(self.num_agents),
            step=new_time,
        )

        reward = {self.agents[i]: reward_arr[i] for i in range(self.num_agents)}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done
        info = {
            "base_reward": reward_arr,
            "base_return": base_return,
            "stags_caught": jnp.broadcast_to(jnp.sum(coop), (self.num_agents,)),
            "step_count": jnp.broadcast_to(new_time, (self.num_agents,)),
        }

        # Auto-reset on episode end.
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

    def _make_obs(self, state: StagHuntState) -> dict[str, jnp.ndarray]:
        obs = {}
        for i in range(self.num_agents):
            img = self._render_agent_view(state, i)
            obs[self.agents[i]] = img.flatten().astype(jnp.float32) / 255.0
        return obs

    def _render_agent_view(self, state: StagHuntState, agent_idx: int) -> jnp.ndarray:
        """Ego-centric uint8 RGB: ego red, partner blue, stags green, plants yellow, walls grey."""
        other_idx = 1 - agent_idx
        tile_img = jnp.tile(_WHITE[None, None, :], (self.height, self.width, 1))
        tile_img = jnp.where(self.wall_map[..., None], _GREY[None, None, :], tile_img)
        tile_img = tile_img.at[state.plant_pos[:, 1], state.plant_pos[:, 0], :].set(_YELLOW)
        tile_img = tile_img.at[state.stag_pos[:, 1], state.stag_pos[:, 0], :].set(_GREEN)
        img = jnp.repeat(jnp.repeat(tile_img, TILE_SIZE, axis=0), TILE_SIZE, axis=1)
        img = self._draw_circle(img, state.agent_pos[other_idx], _BLUE)
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

def _fixed_state(env, agents, stags, plants):
    """Build a StagHuntState with explicit [x, y] positions for deterministic tests."""
    return StagHuntState(
        agent_pos=jnp.asarray(agents, dtype=jnp.int32),
        stag_pos=jnp.asarray(stags, dtype=jnp.int32),
        plant_pos=jnp.asarray(plants, dtype=jnp.int32),
        time=jnp.int32(0),
    )


def test_obs_shapes_and_spaces():
    env = StagHuntEnv(layout="empty_7x7", num_stags=1, num_plants=2, max_steps=50)
    obs, state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert obs["agent_0"].shape == (env._obs_dim,)
    assert env._obs_dim == 7 * TILE_SIZE * 7 * TILE_SIZE * 3
    assert float(obs["agent_0"].min()) >= 0.0 and float(obs["agent_0"].max()) <= 1.0
    # Ego-centric: agents see different frames.
    assert not bool(jnp.allclose(obs["agent_0"], obs["agent_1"]))
    assert env.action_space("agent_0").n == NUM_ACTIONS


def test_reset_places_distinct_cells():
    env = StagHuntEnv(layout="empty_7x7", num_stags=2, num_plants=3)
    _, state = env.reset(jax.random.PRNGKey(3))
    s = state.env_state
    allpos = jnp.concatenate([s.agent_pos, s.stag_pos, s.plant_pos], axis=0)
    flats = allpos[:, 1] * env.width + allpos[:, 0]
    assert int(jnp.unique(flats).shape[0]) == allpos.shape[0]  # all distinct


def test_plant_gives_individual_reward():
    env = StagHuntEnv(layout="empty_7x7", num_stags=1, num_plants=1, max_steps=50)
    # agent0 at (1,1), plant at (2,1) -> agent0 moves right onto it. Stag far away.
    st = _fixed_state(env, [[1, 1], [5, 5]], [[6, 6]], [[2, 1]])
    state = WrappedEnvState(st, jnp.zeros(2), jnp.zeros(2), jnp.int32(0))
    right, stay = 0, 4
    _, _, reward, _, _ = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, {"agent_0": right, "agent_1": stay}
    )
    assert float(reward["agent_0"]) == 1.0
    assert float(reward["agent_1"]) == 0.0


def test_cooperative_stag_rewards_both():
    env = StagHuntEnv(layout="empty_7x7", num_stags=1, num_plants=1, penalty=1.0, max_steps=50)
    # stag at (3,3); agent0 at (2,3) steps right onto stag; agent1 at (3,4) is dist 1.
    st = _fixed_state(env, [[2, 3], [3, 4]], [[3, 3]], [[0, 0]])
    state = WrappedEnvState(st, jnp.zeros(2), jnp.zeros(2), jnp.int32(0))
    right, stay = 0, 4
    _, _, reward, _, _ = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, {"agent_0": right, "agent_1": stay}
    )
    assert float(reward["agent_0"]) == 5.0
    assert float(reward["agent_1"]) == 5.0


def test_solo_stag_penalises_stepper():
    env = StagHuntEnv(layout="empty_7x7", num_stags=1, num_plants=1, penalty=1.0, max_steps=50)
    # stag at (3,3); agent0 steps onto it; agent1 far (dist > 1).
    st = _fixed_state(env, [[2, 3], [6, 6]], [[3, 3]], [[0, 0]])
    state = WrappedEnvState(st, jnp.zeros(2), jnp.zeros(2), jnp.int32(0))
    right, stay = 0, 4
    _, _, reward, _, _ = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, {"agent_0": right, "agent_1": stay}
    )
    assert float(reward["agent_0"]) == -1.0
    assert float(reward["agent_1"]) == 0.0


def test_timeout_triggers_auto_reset():
    env = StagHuntEnv(layout="empty_7x7", num_stags=1, num_plants=2, max_steps=3)
    _, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    stay = {"agent_0": 4, "agent_1": 4}
    for _ in range(2):
        _, state, _, dones, _ = step(jax.random.PRNGKey(2), state, stay)
        assert not bool(dones["__all__"])
    _, state, _, dones, _ = step(jax.random.PRNGKey(2), state, stay)
    assert bool(dones["__all__"])
    assert int(state.env_state.time) == 0  # auto-reset


def test_vmap_reset_step():
    env = StagHuntEnv(layout="empty_7x7", num_stags=2, num_plants=3, max_steps=20)
    n = 8
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    obs, state = jax.vmap(env.reset)(keys)
    assert obs["agent_0"].shape == (n, env._obs_dim)
    acts = {a: jnp.zeros(n, dtype=jnp.int32) for a in env.agents}
    obs2, _, reward, _, _ = jax.vmap(env.step)(keys, state, acts)
    assert obs2["agent_0"].shape == (n, env._obs_dim)
    assert reward["agent_0"].shape == (n,)
