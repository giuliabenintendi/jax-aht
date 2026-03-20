"""Card Game environment for testing joint attention.

Two agents observe 5 shuffled colored cards for several deliberation steps,
then simultaneously pick a card position. Reward +1 if both pick the same
position, 0 otherwise. The only coordination channel is the partner's
spatial attention map (fed as a 4th observation channel by JA-IPPO).

Layout (3×5 grid, TILE_PIXELS=7 → 21×35 px):
  Row 0: [ ] [ ] [agent_0 ▽] [ ] [ ]
  Row 1: [card] [card] [card] [card] [card]
  Row 2: [ ] [ ] [agent_1 △] [ ] [ ]

Actions: 0-4 (pick card at that position). Ignored during deliberation steps.
Episode: max_steps total (default 10). Steps 1..max_steps-1 are deliberation,
step max_steps is the decision step.
"""
from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv, WrappedEnvState
from envs.card_game.rendering import (
    render_card_game,
    TILE_PIXELS,
    GRID_ROWS,
    GRID_COLS,
    NUM_CARDS,
)

_EGO_HIGHLIGHT_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)

# Fixed agent grid positions
_AGENT_POSITIONS = jnp.array([
    [0, 2],  # agent 0: top center
    [2, 2],  # agent 1: bottom center
], dtype=jnp.int32)


@dataclass
class CardGameState:
    card_permutation: chex.Array  # (NUM_CARDS,) card identity at each position
    step_count: chex.Array        # scalar int32


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


class CardGameEnv(BaseEnv):
    """Card coordination game with image observations.

    Exposes grid_height, grid_width, tile_size for compatibility with
    the JA-IPPO agent initialization pipeline.
    """

    def __init__(self, max_steps: int = 10, **kwargs):
        self.max_steps = max_steps
        self.num_cards = NUM_CARDS
        self.num_agents = 2
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.name = "CardGame"

        # Image dimensions (exposed for JA-IPPO pipeline)
        self.grid_height = GRID_ROWS
        self.grid_width = GRID_COLS
        self.tile_size = TILE_PIXELS

        self._img_h = self.grid_height * self.tile_size
        self._img_w = self.grid_width * self.tile_size
        self._obs_dim = self._img_h * self._img_w * 3

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

    def observation_space(self, agent: str):
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str):
        return jaxmarl_spaces.Discrete(num_categories=self.num_cards)

    def _make_obs(self, env_state: CardGameState) -> Dict[str, jnp.ndarray]:
        """Render image with per-agent ego highlight (magenta border)."""
        img = render_card_game(env_state.card_permutation)

        obs = {}
        for i in range(self.num_agents):
            row, col = _AGENT_POSITIONS[i]
            agent_img = _draw_border(
                img, row, col, self.tile_size, _EGO_HIGHLIGHT_COLOR
            )
            obs[self.agents[i]] = agent_img.flatten().astype(jnp.float32) / 255.0
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        perm = jax.random.permutation(key, self.num_cards)
        env_state = CardGameState(
            card_permutation=perm,
            step_count=jnp.int32(0),
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

        # Reward only on the final (decision) step
        is_decision = new_step >= self.max_steps
        a0 = actions["agent_0"]
        a1 = actions["agent_1"]
        match = jnp.equal(a0, a1)
        reward_val = jnp.where(is_decision & match, 1.0, 0.0)

        reward = {agent: reward_val for agent in self.agents}
        done = is_decision
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        new_env_state = env_state.replace(step_count=new_step)
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
        return {agent: jnp.ones(self.num_cards) for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step
