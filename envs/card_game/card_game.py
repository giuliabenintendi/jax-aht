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
    agent_choices: chex.Array     # (2,) chosen positions, -1 until decision step
    messages: chex.Array          # (2,) last message per agent, int32


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

    def __init__(self, max_steps: int = 10, shuffle: bool = True, fixed_partner_pos: int = -1,
                 communication: bool = False, **kwargs):
        self.max_steps = max_steps
        self.shuffle = shuffle
        self.fixed_partner_pos = fixed_partner_pos
        self.communication = communication
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
        if self.communication:
            self._obs_dim += self.num_cards  # partner's message as one-hot

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

    def observation_space(self, agent: str):
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str):
        if self.communication:
            return jaxmarl_spaces.Discrete(num_categories=self.num_cards * self.num_cards)
        return jaxmarl_spaces.Discrete(num_categories=self.num_cards)

    def _make_obs(self, env_state: CardGameState) -> Dict[str, jnp.ndarray]:
        """Render image with per-agent ego highlight (magenta border).

        When communication is enabled, appends the partner's last message
        as a one-hot vector (NUM_CARDS floats) to the flat observation.
        """
        img = render_card_game(env_state.card_permutation)

        obs = {}
        for i in range(self.num_agents):
            row, col = _AGENT_POSITIONS[i]
            agent_img = _draw_border(
                img, row, col, self.tile_size, _EGO_HIGHLIGHT_COLOR
            )
            flat = agent_img.flatten().astype(jnp.float32) / 255.0
            if self.communication:
                partner_msg = env_state.messages[1 - i]
                msg_onehot = jax.nn.one_hot(partner_msg, self.num_cards)
                flat = jnp.concatenate([flat, msg_onehot])
            obs[self.agents[i]] = flat
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        perm = jax.random.permutation(key, self.num_cards) if self.shuffle else jnp.arange(self.num_cards)
        env_state = CardGameState(
            card_permutation=perm,
            step_count=jnp.int32(0),
            agent_choices=jnp.full(2, -1, dtype=jnp.int32),
            messages=jnp.zeros(2, dtype=jnp.int32),
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
        raw_a0 = actions["agent_0"]
        raw_a1 = actions["agent_1"]

        # Decode joint action into card choice and message
        if self.communication:
            a0 = raw_a0 // self.num_cards
            a1 = raw_a1 // self.num_cards
            new_messages = jnp.array(
                [raw_a0 % self.num_cards, raw_a1 % self.num_cards], dtype=jnp.int32)
        else:
            a0 = raw_a0
            a1 = raw_a1
            new_messages = env_state.messages

        # Override agent 1's action if fixed partner is set
        if self.fixed_partner_pos >= 0:
            a1 = jnp.int32(self.fixed_partner_pos)
        match = jnp.equal(a0, a1)
        reward_val = jnp.where(is_decision & match, 1.0, 0.0)

        reward = {agent: reward_val for agent in self.agents}
        done = is_decision
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        # Store choices on decision step, keep -1 otherwise
        choices = jnp.where(
            is_decision,
            jnp.array([a0, a1], dtype=jnp.int32),
            jnp.full(2, -1, dtype=jnp.int32),
        )
        new_env_state = env_state.replace(
            step_count=new_step, agent_choices=choices, messages=new_messages)
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
        n = self.num_cards * self.num_cards if self.communication else self.num_cards
        return {agent: jnp.ones(n) for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step
