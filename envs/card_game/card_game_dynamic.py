"""Card Game environment with dynamic card positions (10 colors, max 5 shown).

Two agents observe colored cards placed at random grid positions. During
deliberation steps they can only "do nothing" (or send messages if
communication is enabled). On the decision step they pick a color.
Reward +1 if both pick the same color, 0 otherwise.

10 possible colors, min_cards to max_cards shown per episode (default 2-5),
placed at random free cells on a 3x5 grid. Agents fixed at (0,2) and (2,2).
Actions are color-based (pick by color identity). Unavailable colors masked.

Actions (no communication): Discrete(C+1)
  0..(C-1): pick color i (decision step, only if present)
  C: do nothing (deliberation only)

Actions (with communication): Discrete(C*C + C)
  0..(C*C-1): pick color (a//C) + message (a%C) — decision only
  C*C..(C*C+C-1): message only (a - C*C) — deliberation only
"""
from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv, WrappedEnvState
from envs.card_game.rendering_dynamic import (
    render_card_game,
    TILE_PIXELS,
    GRID_ROWS,
    GRID_COLS,
    NUM_COLORS,
)

_EGO_HIGHLIGHT_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)

# Fixed agent grid positions (center of top and bottom rows)
_AGENT_POSITIONS = jnp.array([
    [0, GRID_COLS // 2],              # agent 0: top center
    [GRID_ROWS - 1, GRID_COLS // 2],  # agent 1: bottom center
], dtype=jnp.int32)

# All grid cells as flat indices (0..14)
_ALL_CELLS = jnp.arange(GRID_ROWS * GRID_COLS)
# Mask: 1 for cells not occupied by agents
_AGENT_FLAT = _AGENT_POSITIONS[:, 0] * GRID_COLS + _AGENT_POSITIONS[:, 1]
_FREE_MASK = jnp.ones(GRID_ROWS * GRID_COLS).at[_AGENT_FLAT].set(0.0)


@dataclass
class CardGameState:
    card_positions: chex.Array   # (NUM_COLORS, 2) — [row, col] per color slot
    card_present: chex.Array     # (NUM_COLORS,) bool — which colors exist
    step_count: chex.Array       # scalar int32
    agent_choices: chex.Array    # (2,) chosen color indices, -1 until decision
    messages: chex.Array         # (2,) last message per agent, int32


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
    """Card coordination game with dynamic card positions and color-based actions.

    Exposes grid_height, grid_width, tile_size for compatibility with
    the JA-IPPO agent initialization pipeline.
    """

    def __init__(self, max_steps: int = 10, shuffle: bool = True,
                 fixed_partner_pos: int = -1, communication: bool = False,
                 max_cards: int = 5, min_cards: int = 2, **kwargs):
        self.max_steps = max_steps
        self.shuffle = shuffle
        self.fixed_partner_pos = fixed_partner_pos
        self.communication = communication
        self.num_colors = NUM_COLORS
        self.max_cards = min(max_cards, NUM_COLORS)
        self.min_cards = min_cards
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
            self._obs_dim += self.num_colors  # partner's message as one-hot

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

    # For backward compatibility with code that reads num_cards
    @property
    def num_cards(self):
        return self.num_colors

    def observation_space(self, agent: str):
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str):
        if self.communication:
            # 0..(C*C-1): pick color (a//C) + message (a%C) — decision
            # C*C..(C*C+C-1): message only (a - C*C) — deliberation
            return jaxmarl_spaces.Discrete(
                num_categories=self.num_colors * self.num_colors + self.num_colors)
        # 0..(C-1): pick color, C: do nothing
        return jaxmarl_spaces.Discrete(num_categories=self.num_colors + 1)

    def _make_obs(self, env_state: CardGameState) -> Dict[str, jnp.ndarray]:
        """Render image with per-agent ego highlight.

        When communication is enabled, appends the partner's last message
        as a one-hot vector to the flat observation.
        """
        img = render_card_game(env_state.card_positions, env_state.card_present)

        obs = {}
        for i in range(self.num_agents):
            row, col = _AGENT_POSITIONS[i]
            agent_img = _draw_border(
                img, row, col, self.tile_size, _EGO_HIGHLIGHT_COLOR
            )
            flat = agent_img.flatten().astype(jnp.float32) / 255.0
            if self.communication:
                partner_msg = env_state.messages[1 - i]
                msg_onehot = jax.nn.one_hot(partner_msg, self.num_colors)
                flat = jnp.concatenate([flat, msg_onehot])
            obs[self.agents[i]] = flat
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        key1, key2, key3 = jax.random.split(key, 3)

        # How many cards this episode: uniform [min_cards, max_cards]
        num_cards = jax.random.randint(key1, (), self.min_cards, self.max_cards + 1)

        # Which colors are present (first num_cards of a shuffled order)
        if self.shuffle:
            order = jax.random.permutation(key2, self.num_colors)
        else:
            order = jnp.arange(self.num_colors)
        card_present = jnp.arange(self.num_colors) < num_cards
        # Unshuffle: card_present[original_color] = True if that color was selected
        card_present_by_color = jnp.zeros(self.num_colors, dtype=jnp.bool_)
        card_present_by_color = card_present_by_color.at[order].set(card_present)

        # Random positions on free cells — sample max_cards positions
        # (only the first num_cards are used, rest masked by card_present)
        pos_indices = jax.random.choice(
            key3, _ALL_CELLS, shape=(self.max_cards,),
            p=_FREE_MASK, replace=False,
        )
        # Assign positions to present colors: each present color gets a unique position
        # Colors not present get a dummy position (0,0) — masked in rendering
        all_positions = jnp.zeros((self.num_colors, 2), dtype=jnp.int32)
        present_idx = 0
        # Use scan to assign positions to present colors in order
        def assign_pos(carry, i):
            positions, pos_idx = carry
            is_present = card_present_by_color[i]
            row = pos_indices[pos_idx] // GRID_COLS
            col = pos_indices[pos_idx] % GRID_COLS
            positions = jnp.where(
                is_present,
                positions.at[i].set(jnp.array([row, col])),
                positions,
            )
            pos_idx = jnp.where(is_present, pos_idx + 1, pos_idx)
            return (positions, pos_idx), None

        (card_positions, _), _ = jax.lax.scan(
            assign_pos, (all_positions, jnp.int32(0)), jnp.arange(self.num_colors)
        )

        env_state = CardGameState(
            card_positions=card_positions,
            card_present=card_present_by_color,
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

        is_decision = new_step >= self.max_steps
        raw_a0 = actions["agent_0"]
        raw_a1 = actions["agent_1"]

        # Decode actions based on communication mode
        if self.communication:
            n_pick_msg = self.num_colors * self.num_colors  # 25
            is_pick_0 = raw_a0 < n_pick_msg
            color_0 = jnp.where(is_pick_0, raw_a0 // self.num_colors, jnp.int32(-1))
            msg0 = jnp.where(is_pick_0, raw_a0 % self.num_colors, raw_a0 - n_pick_msg)

            is_pick_1 = raw_a1 < n_pick_msg
            color_1 = jnp.where(is_pick_1, raw_a1 // self.num_colors, jnp.int32(-1))
            msg1 = jnp.where(is_pick_1, raw_a1 % self.num_colors, raw_a1 - n_pick_msg)

            new_messages = jnp.array([msg0, msg1], dtype=jnp.int32)
        else:
            # Actions 0-4: pick color, action 5: do nothing
            color_0 = jnp.where(raw_a0 < self.num_colors, raw_a0, jnp.int32(-1))
            color_1 = jnp.where(raw_a1 < self.num_colors, raw_a1, jnp.int32(-1))
            new_messages = env_state.messages

        # Override agent 1's action if fixed partner is set
        if self.fixed_partner_pos >= 0:
            color_1 = jnp.int32(self.fixed_partner_pos)

        # Reward: both agents must pick the same color
        match = jnp.equal(color_0, color_1)
        reward_val = jnp.where(is_decision & match, 1.0, 0.0)

        reward = {agent: reward_val for agent in self.agents}
        done = is_decision
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        # Store choices on decision step
        choices = jnp.where(
            is_decision,
            jnp.array([color_0, color_1], dtype=jnp.int32),
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
        next_step = state.env_state.step_count + 1
        is_decision = next_step >= self.max_steps
        present = state.env_state.card_present  # (NUM_COLORS,) bool

        if self.communication:
            n_pick_msg = self.num_colors * self.num_colors  # 25
            n_msg_only = self.num_colors  # 5

            # Decision: pick+msg actions (0-24), only for present colors
            # For each of the 25 actions, check if the color (a//5) is present
            pick_msg_mask = present[jnp.arange(n_pick_msg) // self.num_colors]
            pick_msg_avail = jnp.where(is_decision, pick_msg_mask, jnp.zeros(n_pick_msg))

            # Deliberation: message-only actions (25-29)
            msg_only_avail = jnp.where(is_decision, jnp.zeros(n_msg_only), jnp.ones(n_msg_only))

            mask = jnp.concatenate([pick_msg_avail, msg_only_avail])
        else:
            # Decision: pick color (0-4) if present, no do-nothing
            color_avail = jnp.where(is_decision, present.astype(jnp.float32), jnp.zeros(self.num_colors))
            # Deliberation: only do-nothing (5)
            noop_avail = jnp.where(is_decision, jnp.zeros(1), jnp.ones(1))
            mask = jnp.concatenate([color_avail, noop_avail])

        return {agent: mask for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step
