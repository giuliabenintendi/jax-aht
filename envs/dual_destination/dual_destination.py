"""Dual Destination environment (image observations).

A two-player cooperative gridworld, adapted from the DreamTeam `toy_coop`
environment (itself from KJha02/crossEnvCooperation). Two agents move on a grid
and must simultaneously occupy two *distinct* goal cells; the pair receives
reward 1 on the step both agents sit on different goals, 0 otherwise. Because the
goals are identical and the layout symmetric, the task is a pure symmetry-breaking
(anti-)coordination problem: the agents must agree on who takes which goal.

Observations are ego-centric RGB renders, flattened to a vector in `[0, 1]`:
ego agent red, partner blue, goals green, walls grey, on a white background. The
grid/tile conventions mirror the card game env so the same image pipeline applies
(`grid_height`, `grid_width`, `tile_size`, `_img_h`, `_img_w` are exposed).

This adapts DreamTeam's jaxued/UED env (`reset_to_level(level)` + `Level`) to the
jax-aht `BaseEnv` interface: the layout is fixed per env instance and `reset(rng)`
builds the state directly. Joint-attention wiring is intentionally out of scope here.
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
from envs.dual_destination.layouts import LAYOUTS, Layout, layout_to_arrays

TILE_SIZE = 7
AGENT_RADIUS = 2
NUM_ACTIONS = 5  # right, down, left, up, stay

# Movement deltas in [x, y], indexed by action.
ACTION_TO_DIR = jnp.array(
    [[1, 0], [0, 1], [-1, 0], [0, -1], [0, 0]], dtype=jnp.int32
)

_RED = jnp.array([255, 0, 0], dtype=jnp.uint8)
_BLUE = jnp.array([0, 0, 255], dtype=jnp.uint8)
_GREEN = jnp.array([0, 255, 0], dtype=jnp.uint8)
_WHITE = jnp.array([255, 255, 255], dtype=jnp.uint8)
_GREY = jnp.array([100, 100, 100], dtype=jnp.uint8)


@dataclass
class DualDestinationState:
    agent_pos: chex.Array  # (2, 2) int32, [x, y] per agent
    time: chex.Array       # scalar int32


class DualDestinationEnv(BaseEnv):
    """Two-agent anti-coordination gridworld with flattened image observations."""

    def __init__(
        self,
        layout: str | Layout = "symmetric_5x5",
        max_steps: int = 50,
        random_reset: bool = False,
        one_step_delay: bool = False,
        **kwargs,  # noqa: ARG002 - absorbs obs_type and other passthrough kwargs
    ):
        if isinstance(layout, str):
            layout = LAYOUTS[layout]
        h, w, wall_map, goal_pos, agent_start = layout_to_arrays(layout)

        self.height = int(h)
        self.width = int(w)
        self.wall_map = wall_map          # (h, w) bool
        self.goal_pos = goal_pos          # (2, 2) int32 [x, y]
        self.agent_start = agent_start    # (2, 2) int32 [x, y]
        self.max_steps = max_steps
        self.random_reset = random_reset
        self.one_step_delay = one_step_delay

        self.num_agents = 2
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.name = "DualDestination"

        # Image dims, exposed for the (JA-)IPPO image pipeline.
        self.tile_size = TILE_SIZE
        self.grid_height = self.height
        self.grid_width = self.width
        self._img_h = self.height * TILE_SIZE
        self._img_w = self.width * TILE_SIZE
        self.num_scalar_obs = 0
        self._obs_dim = self._img_h * self._img_w * 3

        # Free cells (not wall, not goal) for random agent placement.
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
        env_state = DualDestinationState(agent_pos=agent_pos, time=jnp.int32(0))
        obs = self._make_obs(env_state)
        return obs, WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=jnp.zeros(self.num_agents),
            step=jnp.int32(0),
        )

    def _sample_agent_pos(self, key: chex.PRNGKey) -> jnp.ndarray:
        """Place the two agents on distinct free cells, weighting by `_free_prob`.

        Uses `choice` over all cells with zero probability on occupied cells, which
        keeps the sample jit-safe (static population size) without a fixed-size
        `flatnonzero` dance.
        """
        flats = jax.random.choice(
            key, self.height * self.width, shape=(2,), replace=False, p=self._free_prob
        )
        xs = flats % self.width
        ys = flats // self.width
        return jnp.stack([xs, ys], axis=-1).astype(jnp.int32)

    def _move_and_reward(
        self, agent_pos: jnp.ndarray, actions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply moves with wall/collision handling and compute the sparse reward.

        Returns `(next_pos, reward, goal_reached)`. A move into a wall, off-grid,
        or into the partner's target cell is reverted for that agent.
        """
        next_pos = agent_pos + ACTION_TO_DIR[actions]
        next_pos = next_pos.at[:, 0].set(jnp.clip(next_pos[:, 0], 0, self.width - 1))
        next_pos = next_pos.at[:, 1].set(jnp.clip(next_pos[:, 1], 0, self.height - 1))

        hits_wall = self.wall_map[next_pos[:, 1], next_pos[:, 0]]
        next_pos = jnp.where(hits_wall[:, None], agent_pos, next_pos)

        would_collide = jnp.all(next_pos[0] == next_pos[1])
        next_pos = jnp.where(would_collide, agent_pos, next_pos)

        a0_on = jnp.all(next_pos[0] == self.goal_pos, axis=-1)  # (num_goals,)
        a1_on = jnp.all(next_pos[1] == self.goal_pos, axis=-1)
        both_on = jnp.logical_and(jnp.any(a0_on), jnp.any(a1_on))
        on_same_goal = jnp.any(jnp.logical_and(a0_on, a1_on))
        goal_reached = jnp.logical_and(both_on, ~on_same_goal)
        if self.one_step_delay:
            # Require both agents to hold position for one step on the goals.
            goal_reached = jnp.logical_and(goal_reached, jnp.all(agent_pos == next_pos))

        return next_pos.astype(jnp.int32), goal_reached.astype(jnp.float32), goal_reached

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
        acts = jnp.array([actions["agent_0"], actions["agent_1"]]).astype(jnp.int32)

        next_pos, reward_val, task_success = self._move_and_reward(
            env_state.agent_pos, acts
        )
        new_time = env_state.time + 1
        timeout = new_time >= self.max_steps
        done = jnp.logical_or(task_success, timeout)

        new_env_state = DualDestinationState(agent_pos=next_pos, time=new_time)
        obs_st = self._make_obs(new_env_state)

        base_reward_arr = jnp.array([reward_val, reward_val])
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
            "task_success": jnp.broadcast_to(task_success, (self.num_agents,)),
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

    def _make_obs(self, state: DualDestinationState) -> dict[str, jnp.ndarray]:
        obs = {}
        for i in range(self.num_agents):
            img = self._render_agent_view(state, i)
            obs[self.agents[i]] = img.flatten().astype(jnp.float32) / 255.0
        return obs

    def _render_agent_view(self, state: DualDestinationState, agent_idx: int) -> jnp.ndarray:
        """Ego-centric uint8 RGB render: ego red, partner blue, goals green, walls grey."""
        other_idx = 1 - agent_idx
        tile_img = jnp.tile(_WHITE[None, None, :], (self.height, self.width, 1))
        tile_img = jnp.where(self.wall_map[..., None], _GREY[None, None, :], tile_img)
        tile_img = tile_img.at[self.goal_pos[:, 1], self.goal_pos[:, 0], :].set(_GREEN)
        img = jnp.repeat(jnp.repeat(tile_img, TILE_SIZE, axis=0), TILE_SIZE, axis=1)
        img = self._draw_circle(img, state.agent_pos[agent_idx], _RED)
        img = self._draw_circle(img, state.agent_pos[other_idx], _BLUE)
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

def _coordination_layout() -> Layout:
    """3x3 empty room with each agent one step from a distinct goal."""
    # goals at (0,0)->0 and (2,2)->8; agents at (0,1)->3 and (2,1)->5.
    return {"height": 3, "width": 3, "wall_idx": [], "agent_idx": [3, 5], "goal_idx": [0, 8]}


def test_obs_shapes_and_spaces():
    env = DualDestinationEnv(layout="symmetric_5x5", max_steps=20)
    obs, state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert obs["agent_0"].shape == (env._obs_dim,)
    assert obs["agent_1"].shape == (env._obs_dim,)
    assert env._obs_dim == 5 * TILE_SIZE * 5 * TILE_SIZE * 3
    assert float(obs["agent_0"].min()) >= 0.0 and float(obs["agent_0"].max()) <= 1.0
    # Ego-centric: the two agents see different frames (red/blue swapped).
    assert not bool(jnp.allclose(obs["agent_0"], obs["agent_1"]))
    assert env.action_space("agent_0").n == NUM_ACTIONS


def test_observation_is_rendered_view():
    """The obs is the flattened, [0,1]-normalised ego render; save a montage to inspect.

    Run `uv run pytest -s -k observation_is_rendered_view envs/dual_destination/dual_destination.py`
    to write artifacts/dual_destination_obs.png and eyeball what the agents see.
    """
    env = DualDestinationEnv(layout="symmetric_5x5", max_steps=100)
    obs, state = env.reset(jax.random.PRNGKey(0))
    recon = (obs["agent_0"].reshape(env._img_h, env._img_w, 3) * 255).astype(jnp.uint8)
    assert jnp.array_equal(recon, env._render_agent_view(state.env_state, 0))

    from envs.dual_destination.rendering import save_observation_montage
    out = save_observation_montage(env, "artifacts/dual_destination_obs.png")
    print(f"[dual_destination] saved observation montage -> {out}")


def test_distinct_goals_give_reward_and_done():
    env = DualDestinationEnv(layout=_coordination_layout(), max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    # agent_0 up -> (0,0); agent_1 down -> (2,2): distinct goals.
    up, down = 3, 1
    _, new_state, reward, dones, info = jax.jit(env.step)(
        jax.random.PRNGKey(1), state, {"agent_0": up, "agent_1": down},
    )
    assert float(reward["agent_0"]) == 1.0
    assert bool(dones["__all__"])
    assert bool(info["task_success"][0])
    # Auto-reset wipes the clock back to 0.
    assert int(new_state.env_state.time) == 0


def test_non_goal_step_no_reward():
    env = DualDestinationEnv(layout=_coordination_layout(), max_steps=20)
    _, state = env.reset(jax.random.PRNGKey(0))
    stay = 4
    _, _, reward, dones, _ = env.step(
        jax.random.PRNGKey(1), state, {"agent_0": stay, "agent_1": stay},
    )
    assert float(reward["agent_0"]) == 0.0
    assert not bool(dones["__all__"])


def test_timeout_triggers_auto_reset():
    env = DualDestinationEnv(layout="symmetric_5x5", max_steps=3)
    _, state = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    stay = {"agent_0": 4, "agent_1": 4}
    for _ in range(2):
        _, state, _, dones, _ = step(jax.random.PRNGKey(2), state, stay)
        assert not bool(dones["__all__"])
    _, state, _, dones, _ = step(jax.random.PRNGKey(2), state, stay)
    assert bool(dones["__all__"])
    assert int(state.env_state.time) == 0


def test_vmap_reset_step_and_random_reset():
    env = DualDestinationEnv(layout="symmetric_5x5", max_steps=20, random_reset=True)
    n = 8
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    obs, state = jax.vmap(env.reset)(keys)
    assert obs["agent_0"].shape == (n, env._obs_dim)
    acts = {a: jnp.zeros(n, dtype=jnp.int32) for a in env.agents}
    obs2, _, reward, _, _ = jax.vmap(env.step)(keys, state, acts)
    assert obs2["agent_0"].shape == (n, env._obs_dim)
    assert reward["agent_0"].shape == (n,)
