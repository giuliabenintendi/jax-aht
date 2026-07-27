"""Card Game environment for testing joint attention (static positions).

Two agents observe 5 shuffled colored cards for several deliberation steps,
then simultaneously pick a color. Reward +1 if both pick the same color,
0 otherwise.

Current policy observations are cards-only image renders:
  - five card rectangles centred in the frame
  - a 2-digit timestep counter in the top-left
  - optionally a white partner-message dot on the referenced card

The older full-scene render with agent triangles still exists only as a
debug/eval helper in `rendering.py`; it is not the observation emitted by
this environment.

Actions: a single Discrete(NUM_CARDS) at every step. The emitted card is
interpreted as a deliberation message when not on the decision step and as a
pick on the decision step (intent-expression layout).

The `communication` flag now toggles only whether the partner's last
emitted card is rendered into each agent's obs as a coloured dot — the
action layout is unified either way.
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
    render_card_game_minimal,
    TILE_PIXELS,
    GRID_ROWS,
    GRID_COLS,
    NUM_CARDS,
    CARD_RECT_W,
    CARD_RECT_H,
    CARD_RECT_Y,
    WHITE_COLOR,
)


@dataclass
class CardGameState:
    card_permutation: chex.Array  # (NUM_CARDS,) card identity at each position
    step_count: chex.Array        # scalar int32
    agent_choices: chex.Array     # (2,) chosen positions, -1 until decision step
    messages: chex.Array          # (2,) last message per agent, int32


class CardGameEnv(BaseEnv):
    """Card coordination game with image observations.

    Exposes grid_height, grid_width, tile_size for compatibility with
    the JA-IPPO agent initialization pipeline. Observations contain only the
    card strip, timestep counter, and optional partner-message dot.
    """

    def __init__(self, max_steps: int = 10, shuffle: bool = True,
                 communication: bool = False,
                 match_coef: float = 0.0,
                 stability_coef: float = 0.0,
                 follow_coef: float = 0.0,
                 scramble_partner_msg: bool = False,
                 gaze_mode: bool = False,
                 **kwargs):
        self.max_steps = max_steps
        self.shuffle = shuffle
        self.communication = communication
        self.match_coef = match_coef
        self.stability_coef = stability_coef
        self.follow_coef = follow_coef
        # gaze_mode (forced-noop deliberation) was removed 2026-07-27; the
        # parameter is still accepted because stored run configs carry
        # `gaze_mode: false` in ENV_KWARGS, but enabling it is an error.
        if gaze_mode:
            raise ValueError("gaze_mode was removed; retrain on the live-deliberation task")
        # When True, the partner-message dot rendered into each agent's obs is
        # drawn at a uniformly random card color instead of the color the
        # partner actually sent. env_state.messages is left untouched, so the
        # speaker's own reward shaping (match/stability/follow) is unaffected.
        self.scramble_partner_msg = scramble_partner_msg
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
        return jaxmarl_spaces.Discrete(num_categories=self.num_cards)

    @property
    def action_dim(self) -> int:
        return self.num_cards

    def _make_obs(
        self, env_state: CardGameState, key: chex.PRNGKey,
    ) -> Dict[str, jnp.ndarray]:
        """Render image observation for each agent.

        Each agent sees: 5 colored card rectangles + a 2-digit timestep counter
        in the top-left (1-indexed: reset shows 01, decision step shows
        max_steps). When `communication=True`, a 2x2 white dot is drawn at the
        centre of the card the partner last messaged about. The dot is white
        for both agents — there is no per-agent identification.

        `key` is used only when `self.scramble_partner_msg` is True, to
        independently resample each agent's perceived partner message. When
        the flag is False the key is unused.
        """
        # Display step + 1 so a max_steps=8 episode shows 01..08 instead of 00..07.
        img = render_card_game_minimal(
            env_state.card_permutation, env_state.step_count + 1
        )

        obs = {}
        agent_keys = jax.random.split(key, self.num_agents)
        dot_size = 2
        dot_y = CARD_RECT_Y + (CARD_RECT_H - dot_size) // 2
        for i in range(self.num_agents):
            agent_img = img
            if self.communication:
                real_msg = env_state.messages[1 - i]
                # has_msg is driven by the *real* message so step 0 (no message
                # yet) still produces an empty frame even when scrambling is on.
                has_msg = real_msg >= 0
                if self.scramble_partner_msg:
                    scrambled_msg = jax.random.randint(
                        agent_keys[i], (), 0, self.num_cards
                    )
                    partner_msg = jnp.where(has_msg, scrambled_msg, real_msg)
                else:
                    partner_msg = real_msg
                # Find the position of the messaged color via card_permutation
                # card_permutation[pos] = color, so we need pos where color == msg
                # Use argmin on |perm - msg| to find the position (exact match = 0)
                msg_pos = jnp.argmin(jnp.abs(env_state.card_permutation - partner_msg))
                card_x_origin = 1 + msg_pos * (CARD_RECT_W + 2)
                dot_x = card_x_origin + (CARD_RECT_W - dot_size) // 2
                white_dot = jnp.broadcast_to(
                    WHITE_COLOR, (dot_size, dot_size, 3)
                )
                agent_img = jax.lax.cond(
                    has_msg,
                    lambda img: jax.lax.dynamic_update_slice(
                        img, white_dot, (dot_y, dot_x, 0)),
                    lambda img: img,
                    agent_img,
                )
            flat = agent_img.flatten().astype(jnp.float32) / 255.0
            obs[self.agents[i]] = flat
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        key, key_obs = jax.random.split(key)
        perm = jax.random.permutation(key, self.num_cards) if self.shuffle else jnp.arange(self.num_cards)
        env_state = CardGameState(
            card_permutation=perm,
            step_count=jnp.int32(0),
            agent_choices=jnp.full(2, -1, dtype=jnp.int32),
            messages=jnp.full(2, -1, dtype=jnp.int32),
        )
        obs = self._make_obs(env_state, key_obs)
        return obs, WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=jnp.zeros(self.num_agents),
            step=jnp.int32(0),
        )

    def _decode_actions(self, raw_a0, raw_a1, prev_messages, is_decision):
        """Return (pick_0, pick_1, new_messages) with -1 for inactive channels.

        Unified action layout: each action is a card identity. On deliberation
        steps the action populates `messages`; on the decision step it
        populates the pick and `messages` is held at its previous value.

        """
        a0 = jnp.asarray(raw_a0, dtype=jnp.int32)
        a1 = jnp.asarray(raw_a1, dtype=jnp.int32)

        pick_0 = jnp.where(is_decision, a0, jnp.int32(-1))
        pick_1 = jnp.where(is_decision, a1, jnp.int32(-1))
        new_messages = jnp.where(
            is_decision,
            prev_messages,
            jnp.array([a0, a1], dtype=jnp.int32),
        )
        return pick_0, pick_1, new_messages

    def _base_reward(self, pick_0, pick_1, is_decision):
        """+1 on decision step for coordination success, 0 otherwise.

        Requires both picks to be valid (>= 0): a non-pick action (e.g. a message
        leaking through a broken mask) must not satisfy `-1 == -1` and score.
        """
        valid = (pick_0 >= 0) & (pick_1 >= 0)
        success = valid & jnp.equal(pick_0, pick_1)
        return jnp.where(is_decision & success, 1.0, 0.0)

    def _comm_shaping(self, prev_messages, new_messages, picks, is_decision):
        """Comm-shaping rewards: match (per-agent), stability (shared), follow (per-agent).

        - match (per-agent): +match_coef per non-decision step when agent i's current
          message equals the partner's previous message. Mirrors the JA match
          mechanism — alignment to the partner's previous-step signal, since an
          agent cannot observe the partner's current message. No reward on the
          first step, where no previous partner message exists.
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

        match_ok_0 = ~is_decision & (prev_msg_1 >= 0) & jnp.equal(new_msg_0, prev_msg_1)
        match_ok_1 = ~is_decision & (prev_msg_0 >= 0) & jnp.equal(new_msg_1, prev_msg_0)
        match_val_0 = jnp.where(match_ok_0, self.match_coef, 0.0)
        match_val_1 = jnp.where(match_ok_1, self.match_coef, 0.0)
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
            jnp.array([match_val_0, match_val_1]),
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
        key, key_reset, key_obs = jax.random.split(key, 3)
        env_state = state.env_state
        new_step = env_state.step_count + 1
        is_decision = new_step >= self.max_steps
        done = is_decision

        pick_0, pick_1, new_messages = self._decode_actions(
            actions["agent_0"], actions["agent_1"], env_state.messages,
            is_decision,
        )

        reward_val = self._base_reward(pick_0, pick_1, is_decision)
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
        obs_st = self._make_obs(new_env_state, key_obs)

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
        mask = jnp.ones(self.num_cards, dtype=jnp.float32)
        return {agent: mask for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step
