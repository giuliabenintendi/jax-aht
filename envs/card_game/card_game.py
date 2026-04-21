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
  0-4: pick color i — decision only
  5-9: send message (a-5) — deliberation only

Optional diagnostic payoff asymmetry:
  one designated focal color pays `focal_card_reward` when coordinated on,
  while every other coordinated color pays `default_match_reward`.
"""
from functools import partial
from typing import Dict, Tuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv, WrappedEnvState
from envs.card_game.action_utils import (
    COMM_ACTION_DIM,
    decode_comm_action,
    decode_pick_or_noop,
    get_action_mask,
)
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
    target_color: chex.Array      # scalar int32, odd-card color for diagnostic mode; -1 otherwise


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

    def __init__(self, max_steps: int = 10, shuffle: bool = True,
                 communication: bool = False,
                 match_coef: float = 0.0,
                 stability_coef: float = 0.0,
                 follow_coef: float = 0.0,
                 odd_one_out_task: bool = False,
                 focal_card_idx: int = -1,
                 focal_card_reward: float = 1.0,
                 default_match_reward: float = 1.0,
                 **kwargs):
        self.max_steps = max_steps
        self.shuffle = shuffle
        self.communication = communication
        self.match_coef = match_coef
        self.stability_coef = stability_coef
        self.follow_coef = follow_coef
        self.odd_one_out_task = odd_one_out_task
        self.focal_card_idx = int(focal_card_idx)
        self.focal_card_reward = float(focal_card_reward)
        self.default_match_reward = float(default_match_reward)
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
            return jaxmarl_spaces.Discrete(num_categories=COMM_ACTION_DIM)
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
                # Draw 2×2 dot at center of messaged card tile
                card_row = 1
                dot_size = 2
                dot_y = card_row * self.tile_size + (self.tile_size - dot_size) // 2
                dot_x = msg_pos * self.tile_size + (self.tile_size - dot_size) // 2
                color_dot = jnp.broadcast_to(partner_colors[i], (dot_size, dot_size, 3))
                agent_img = jax.lax.cond(
                    has_msg,
                    lambda img: jax.lax.dynamic_update_slice(
                        img, color_dot, (dot_y, dot_x, 0)),
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
        if self.odd_one_out_task:
            key_colors, key_shuffle = jax.random.split(key)
            color_order = jax.random.permutation(key_colors, self.num_cards)
            odd_color = color_order[0]
            majority_color = color_order[1]
            cards = jnp.array(
                [odd_color, majority_color, majority_color, majority_color, majority_color],
                dtype=jnp.int32,
            )
            if self.shuffle:
                shuffle_idx = jax.random.permutation(key_shuffle, self.num_cards)
                perm = cards[shuffle_idx]
            else:
                perm = cards
            target_color = odd_color
        else:
            perm = jax.random.permutation(key, self.num_cards) if self.shuffle else jnp.arange(self.num_cards)
            target_color = jnp.int32(-1)
        env_state = CardGameState(
            card_permutation=perm,
            step_count=jnp.int32(0),
            agent_choices=jnp.full(2, -1, dtype=jnp.int32),
            messages=jnp.full(2, -1, dtype=jnp.int32),
            target_color=target_color,
        )
        obs = self._make_obs(env_state)
        return obs, WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=jnp.zeros(self.num_agents),
            step=jnp.int32(0),
        )

    def _decode_actions(self, raw_a0, raw_a1, prev_messages):
        """Return (pick_0, pick_1, new_messages) with -1 for inactive channels."""
        if self.communication:
            pick_0, msg_0 = decode_comm_action(raw_a0)
            pick_1, msg_1 = decode_comm_action(raw_a1)
            new_messages = jnp.array([msg_0, msg_1], dtype=jnp.int32)
        else:
            pick_0 = decode_pick_or_noop(raw_a0)
            pick_1 = decode_pick_or_noop(raw_a1)
            new_messages = prev_messages
        return pick_0, pick_1, new_messages

    def _base_reward(self, pick_0, pick_1, is_decision, target_color):
        """+1 on decision step for coordination (or odd-one-out) success, 0 otherwise.

        Requires both picks to be valid (>= 0): a non-pick action (e.g. a message
        leaking through a broken mask) must not satisfy `-1 == -1` and score.
        """
        valid = (pick_0 >= 0) & (pick_1 >= 0)
        if self.odd_one_out_task:
            success = valid & jnp.equal(pick_0, target_color) & jnp.equal(pick_1, target_color)
        elif self.focal_card_idx >= 0:
            success = valid & jnp.equal(pick_0, pick_1)
            coord_reward = jnp.where(
                jnp.equal(pick_0, jnp.int32(self.focal_card_idx)),
                self.focal_card_reward,
                self.default_match_reward,
            )
            return jnp.where(is_decision & success, coord_reward, 0.0)
        else:
            success = valid & jnp.equal(pick_0, pick_1)
        return jnp.where(is_decision & success, 1.0, 0.0)

    def _comm_shaping(self, prev_messages, new_messages, picks, is_decision):
        """Comm-shaping rewards: match (shared), stability (shared), follow (per-agent).

        - match: +match_coef per non-decision step when new messages agree (shared).
        - stability: +stability_coef per non-decision step when prev msgs matched, new msgs
          still match, and both agents kept their own message (shared; rewards committed
          persistence, discriminates against oscillation or synchronised flips).
        - follow (per-agent): +follow_coef on decision step when prev msgs matched AND
          agent i's pick equals agent i's prev message. Credit is assigned individually
          — agent 0 and agent 1 each earn (or don't) independently, to give the shared
          policy a cleaner per-agent training signal.

        Returns (match_arr, stable_arr, follow_arr), all shape (num_agents,).
        """
        shaping_active = self.communication and (
            self.match_coef > 0 or self.stability_coef > 0 or self.follow_coef > 0
        )
        if not shaping_active:
            zeros = jnp.zeros(self.num_agents)
            return zeros, zeros, zeros

        prev_msg_0, prev_msg_1 = prev_messages[0], prev_messages[1]
        new_msg_0, new_msg_1 = new_messages[0], new_messages[1]
        pick_0, pick_1 = picks

        new_valid = (new_msg_0 >= 0) & (new_msg_1 >= 0)
        new_match = new_valid & jnp.equal(new_msg_0, new_msg_1)

        prev_valid = (prev_msg_0 >= 0) & (prev_msg_1 >= 0)
        prev_match = prev_valid & jnp.equal(prev_msg_0, prev_msg_1)

        both_held = (
            prev_valid
            & jnp.equal(new_msg_0, prev_msg_0)
            & jnp.equal(new_msg_1, prev_msg_1)
        )

        match_val = jnp.where(new_match & ~is_decision, self.match_coef, 0.0)
        stable_val = jnp.where(
            prev_match & new_match & both_held & ~is_decision,
            self.stability_coef,
            0.0,
        )

        follow_ok_0 = is_decision & prev_match & jnp.equal(pick_0, prev_msg_0)
        follow_ok_1 = is_decision & prev_match & jnp.equal(pick_1, prev_msg_1)
        follow_val_0 = jnp.where(follow_ok_0, self.follow_coef, 0.0)
        follow_val_1 = jnp.where(follow_ok_1, self.follow_coef, 0.0)

        broadcast = lambda x: jnp.array([x, x])
        return (
            broadcast(match_val),
            broadcast(stable_val),
            jnp.array([follow_val_0, follow_val_1]),
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
        done = is_decision

        pick_0, pick_1, new_messages = self._decode_actions(
            actions["agent_0"], actions["agent_1"], env_state.messages,
        )

        reward_val = self._base_reward(pick_0, pick_1, is_decision, env_state.target_color)
        reward = {agent: reward_val for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        choices = jnp.where(
            is_decision,
            jnp.array([pick_0, pick_1], dtype=jnp.int32),
            jnp.full(2, -1, dtype=jnp.int32),
        )
        match_arr, stable_arr, follow_arr = self._comm_shaping(
            env_state.messages, new_messages,
            (pick_0, pick_1), is_decision,
        )
        comm_reward_arr = match_arr + stable_arr + follow_arr

        new_env_state = env_state.replace(
            step_count=new_step, agent_choices=choices, messages=new_messages,
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
            "comm_reward": comm_reward_arr,
            "comm_reward_match": match_arr,
            "comm_reward_stable": stable_arr,
            "comm_reward_follow": follow_arr,
            "step_count": jnp.broadcast_to(new_step, (self.num_agents,)),
            "target_color": jnp.broadcast_to(env_state.target_color, (self.num_agents,)),
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
        mask = get_action_mask(is_decision, self.communication)
        return {agent: mask for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step
