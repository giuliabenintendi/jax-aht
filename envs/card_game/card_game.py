"""Card Game environment for testing joint attention (static positions).

Two agents observe 5 shuffled colored cards at fixed positions (row 1)
for several deliberation steps, then simultaneously pick a color.
Reward +1 if both pick the same color, 0 otherwise.

Layout (3×5 grid, TILE_PIXELS=7 → 21×35 px):
  Row 0: [ ] [ ] [agent_0 ▽] [ ] [ ]
  Row 1: [card] [card] [card] [card] [card]
  Row 2: [ ] [ ] [agent_1 △] [ ] [ ]

Actions (no communication):
  0-4: pick color i (only legal on decision step)
  5: do nothing (only legal during deliberation)

Actions (with communication):
  0-24: pick color (a//5) + send message (a%5) — decision only
  25-29: send message (a-25) — deliberation only
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
    AGENT_0_COLOR,
    AGENT_1_COLOR,
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
                 communication: bool = False, comm_reward_coef: float = 0.0, **kwargs):
        self.max_steps = max_steps
        self.shuffle = shuffle
        self.fixed_partner_pos = fixed_partner_pos
        self.communication = communication
        self.comm_reward_coef = comm_reward_coef
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
        self.num_scalar_obs = 0
        self._obs_dim = self._img_h * self._img_w * 3

        self.observation_spaces = {a: self.observation_space(a) for a in self.agents}
        self.action_spaces = {a: self.action_space(a) for a in self.agents}

    def observation_space(self, agent: str):
        return jaxmarl_spaces.Box(0.0, 1.0, (self._obs_dim,))

    def action_space(self, agent: str):
        if self.communication:
            # 0-24: pick card (a//5) + send message (a%5) — decision only
            # 25-29: send message (a-25) — deliberation only
            return jaxmarl_spaces.Discrete(
                num_categories=self.num_cards * self.num_cards + self.num_cards)
        # 0-4: pick color, 5: do nothing
        return jaxmarl_spaces.Discrete(num_categories=self.num_cards + 1)

    def _make_obs(self, env_state: CardGameState) -> Dict[str, jnp.ndarray]:
        """Render image observation for each agent.

        Each agent sees: ego border (white) + a colored dot (partner color)
        at the center of the card tile the partner messaged about.
        """
        img = render_card_game(env_state.card_permutation)

        obs = {}
        partner_colors = [AGENT_1_COLOR, AGENT_0_COLOR]  # agent i sees partner's color
        is_decision = (env_state.step_count + 1) >= self.max_steps
        white = jnp.array([255, 255, 255], dtype=jnp.uint8)
        for i in range(self.num_agents):
            row, col = _AGENT_POSITIONS[i]
            agent_img = _draw_border(
                img, row, col, self.tile_size, _EGO_HIGHLIGHT_COLOR
            )
            if self.communication:
                partner_msg = env_state.messages[1 - i]
                # Find the position of the messaged color via card_permutation
                # card_permutation[pos] = color, so we need pos where color == msg
                # Use argmin on |perm - msg| to find the position (exact match = 0)
                msg_pos = jnp.argmin(jnp.abs(env_state.card_permutation - partner_msg))
                # Draw dot only if partner has sent a valid message (>= 0)
                has_msg = partner_msg >= 0
                # Draw dot at center of messaged card tile
                card_row = 1
                dot_y = card_row * self.tile_size + self.tile_size // 2
                dot_x = msg_pos * self.tile_size + self.tile_size // 2
                color_2x2 = jnp.broadcast_to(partner_colors[i], (2, 2, 3))
                agent_img = jax.lax.cond(
                    has_msg,
                    lambda img: jax.lax.dynamic_update_slice(
                        img, color_2x2, (dot_y, dot_x, 0)),
                    lambda img: img,
                    agent_img,
                )
            # Draw white 4x4 square at top-left when decision time
            decision_img = agent_img.at[0:4, 0:4, :].set(white[None, None, :])
            agent_img = jnp.where(is_decision, decision_img, agent_img)
            flat = agent_img.flatten().astype(jnp.float32) / 255.0
            obs[self.agents[i]] = flat
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        perm = jax.random.permutation(key, self.num_cards) if self.shuffle else jnp.arange(self.num_cards)
        env_state = CardGameState(
            card_permutation=perm,
            step_count=jnp.int32(0),
            agent_choices=jnp.full(2, -1, dtype=jnp.int32),
            messages=jnp.full(2, -1, dtype=jnp.int32),
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

        # Decode action into card choice and message
        if self.communication:
            n_card_msg = self.num_cards * self.num_cards  # 25
            is_card_action = raw_a0 < n_card_msg
            # Card+message (0-24): card = a//5, msg = a%5
            # Message-only (25-29): no card, msg = a-25
            a0 = jnp.where(is_card_action, raw_a0 // self.num_cards, jnp.int32(-1))
            msg0 = jnp.where(is_card_action, raw_a0 % self.num_cards, raw_a0 - n_card_msg)

            is_card_action_1 = raw_a1 < n_card_msg
            a1 = jnp.where(is_card_action_1, raw_a1 // self.num_cards, jnp.int32(-1))
            msg1 = jnp.where(is_card_action_1, raw_a1 % self.num_cards, raw_a1 - n_card_msg)

            new_messages = jnp.array([msg0, msg1], dtype=jnp.int32)
        else:
            # 0-4: pick color, 5: do nothing → -1
            a0 = jnp.where(raw_a0 < self.num_cards, raw_a0, jnp.int32(-1))
            a1 = jnp.where(raw_a1 < self.num_cards, raw_a1, jnp.int32(-1))
            new_messages = env_state.messages

        # Override agent 1's action if fixed partner is set
        if self.fixed_partner_pos >= 0:
            a1 = jnp.int32(self.fixed_partner_pos)
        # Reward based on color match
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

        # Per-agent communication reward (added to training reward, not to logged return)
        if self.communication and self.comm_reward_coef > 0:
            partner_msg_0 = env_state.messages[1]  # agent_1's last message
            partner_msg_1 = env_state.messages[0]  # agent_0's last message
            comm_r0 = jnp.where(
                is_decision & (partner_msg_0 >= 0) & jnp.equal(a0, partner_msg_0),
                self.comm_reward_coef, 0.0)
            comm_r1 = jnp.where(
                is_decision & (partner_msg_1 >= 0) & jnp.equal(a1, partner_msg_1),
                self.comm_reward_coef, 0.0)
            comm_reward_arr = jnp.array([comm_r0, comm_r1])
        else:
            comm_reward_arr = jnp.zeros(self.num_agents)

        info = {
            "base_reward": base_reward_arr,
            "base_return": base_return,
            "comm_reward": comm_reward_arr,
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

        if self.communication:
            n_pick_msg = self.num_cards * self.num_cards  # 25
            n_msg_only = self.num_cards  # 5
            # Decision: pick+msg (0-24), Deliberation: msg only (25-29)
            pick_msg_avail = jnp.where(is_decision, jnp.ones(n_pick_msg), jnp.zeros(n_pick_msg))
            msg_only_avail = jnp.where(is_decision, jnp.zeros(n_msg_only), jnp.ones(n_msg_only))
            mask = jnp.concatenate([pick_msg_avail, msg_only_avail])
        else:
            # Decision: pick positions 0-4, no do-nothing
            pick_avail = jnp.where(is_decision, jnp.ones(self.num_cards), jnp.zeros(self.num_cards))
            # Deliberation: only do-nothing (action 5)
            noop_avail = jnp.where(is_decision, jnp.zeros(1), jnp.ones(1))
            mask = jnp.concatenate([pick_avail, noop_avail])

        return {agent: mask for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step
