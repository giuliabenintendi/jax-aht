"""Card Flip environment for testing joint attention.

Two agents see 5 face-down (gray) cards. During 3 deliberation steps, each
agent flips one card per step to reveal its color. Each agent only sees
their own flips. On the decision step, both pick a COLOR. Reward +1 if
both pick the same color.

JA helps: the partner's attention map reveals which positions they flipped,
enabling coordination on a shared revealed card.
"""
from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax import struct

from jaxmarl import spaces as jaxmarl_spaces
from envs.base_env import BaseEnv, WrappedEnvState
from envs.card_game.rendering import (
    render_card_game,
    TILE_PIXELS,
    GRID_ROWS,
    GRID_COLS,
    NUM_CARDS,
)

_EGO_HIGHLIGHT_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)

_AGENT_POSITIONS = jnp.array([
    [0, 2],  # agent 0: top center
    [2, 2],  # agent 1: bottom center
], dtype=jnp.int32)


def _draw_border(img, row, col, tile_size, color):
    """Draw a 1-pixel border around the tile at grid position (row, col)."""
    y = jnp.int32(row) * tile_size
    x = jnp.int32(col) * tile_size

    top_row = jnp.broadcast_to(color, (1, tile_size, 3))
    img = jax.lax.dynamic_update_slice(img, top_row, (y, x, 0))
    img = jax.lax.dynamic_update_slice(img, top_row, (y + tile_size - 1, x, 0))

    left_col = jnp.broadcast_to(color, (tile_size, 1, 3))
    img = jax.lax.dynamic_update_slice(img, left_col, (y, x, 0))
    img = jax.lax.dynamic_update_slice(img, left_col, (y, x + tile_size - 1, 0))

    return img


@struct.dataclass
class CardFlipState:
    card_permutation: chex.Array  # (5,) color at each position
    step_count: chex.Array        # scalar int32
    agent_choices: chex.Array     # (2,) decision color picks, -1 until decision
    revealed_0: chex.Array        # (5,) bool — agent 0's flipped cards
    revealed_1: chex.Array        # (5,) bool — agent 1's flipped cards


class CardFlipEnv(BaseEnv):
    def __init__(
        self,
        max_steps: int = 4,
        obs_type: str = "image",
        **kwargs,
    ):
        super().__init__(num_agents=2)
        self.max_steps = max_steps
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.name = "CardFlip"

        self.grid_height = GRID_ROWS
        self.grid_width = GRID_COLS
        self.tile_size = TILE_PIXELS

        self._img_h = self.grid_height * self.tile_size
        self._img_w = self.grid_width * self.tile_size
        self.num_scalar_obs = 0
        self._obs_dim = self._img_h * self._img_w * 3

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

    @property
    def num_cards(self):
        return NUM_CARDS

    def observation_space(self, agent: str):
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str):
        return jaxmarl_spaces.Discrete(num_categories=NUM_CARDS)

    def _make_obs(self, env_state: CardFlipState) -> Dict[str, jnp.ndarray]:
        """Render per-agent observation with only their revealed cards visible."""
        obs = {}
        is_decision = (env_state.step_count + 1) >= self.max_steps
        white = jnp.array([255, 255, 255], dtype=jnp.uint8)
        revealed_masks = [env_state.revealed_0, env_state.revealed_1]

        for i in range(self.num_agents):
            img = render_card_game(env_state.card_permutation, revealed=revealed_masks[i])
            row, col = _AGENT_POSITIONS[i]
            agent_img = _draw_border(img, row, col, self.tile_size, _EGO_HIGHLIGHT_COLOR)
            # Decision square at top-left
            decision_img = agent_img.at[0:4, 0:4, :].set(white[None, None, :])
            agent_img = jnp.where(is_decision, decision_img, agent_img)
            flat = agent_img.flatten().astype(jnp.float32) / 255.0
            obs[self.agents[i]] = flat
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        perm = jax.random.permutation(key, self.num_cards)
        env_state = CardFlipState(
            card_permutation=perm,
            step_count=jnp.int32(0),
            agent_choices=jnp.full(2, -1, dtype=jnp.int32),
            revealed_0=jnp.zeros(self.num_cards, dtype=jnp.bool_),
            revealed_1=jnp.zeros(self.num_cards, dtype=jnp.bool_),
        )
        obs = self._make_obs(env_state)
        return obs, WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=jnp.zeros(self.num_agents),
            step=jnp.int32(0),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: WrappedEnvState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[WrappedEnvState] = None,
    ) -> Tuple[Dict[str, chex.Array], WrappedEnvState, Dict[str, float], Dict[str, bool], Dict]:
        key, key_reset = jax.random.split(key)
        env_state = state.env_state
        new_step = env_state.step_count + 1

        is_decision = new_step >= self.max_steps
        a0 = actions["agent_0"]
        a1 = actions["agent_1"]

        # Deliberation: flip card at chosen position
        new_revealed_0 = env_state.revealed_0.at[a0].set(True)
        new_revealed_1 = env_state.revealed_1.at[a1].set(True)
        # Only update revealed during deliberation, freeze during decision
        new_revealed_0 = jnp.where(is_decision, env_state.revealed_0, new_revealed_0)
        new_revealed_1 = jnp.where(is_decision, env_state.revealed_1, new_revealed_1)

        # Decision: reward if both pick same color
        match = jnp.equal(a0, a1)
        reward_val = jnp.where(is_decision & match, 1.0, 0.0)

        reward = {agent: reward_val for agent in self.agents}
        done = is_decision
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        choices = jnp.where(
            is_decision,
            jnp.array([a0, a1], dtype=jnp.int32),
            jnp.full(2, -1, dtype=jnp.int32),
        )
        new_env_state = CardFlipState(
            card_permutation=env_state.card_permutation,
            step_count=new_step,
            agent_choices=choices,
            revealed_0=new_revealed_0,
            revealed_1=new_revealed_1,
        )
        obs_st = self._make_obs(new_env_state)

        base_reward_arr = jnp.array([reward_val, reward_val])
        base_return = state.base_return_so_far + base_reward_arr

        state_st = WrappedEnvState(
            env_state=new_env_state,
            base_return_so_far=base_return,
            avail_actions=jnp.zeros(self.num_agents),
            step=new_step,
        )

        info = {
            "base_reward": base_reward_arr,
            "base_return": base_return,
        }

        # Auto-reset on episode end
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

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: WrappedEnvState) -> Dict[str, jnp.ndarray]:
        # All 5 actions always available
        mask = jnp.ones(self.num_cards, dtype=jnp.float32)
        return {agent: mask for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step
