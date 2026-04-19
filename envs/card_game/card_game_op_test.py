"""Card Game OP-Test: minimal diagnostic for Other-Play wrapper correctness.

Layout (3×5 grid, TILE_PIXELS=7 → 21×35 px):
  Row 0: [ ] [ ] [agent_0 ▽] [ ] [ ]
  Row 1: [card] [card] [card] [card] [card]   # 4 whites + 1 red
  Row 2: [ ] [ ] [agent_1 △] [ ] [ ]

Actions:
  0-4: pick position i (only legal on decision step)
  5  : noop          (only legal during deliberation)

Reward (decision step only):
  +0.9 if both agents pick the red card,
  +1.0 if both agents pick the same non-red position,
   0.0 otherwise.

Diagnostic prediction:
  Without OP, stable positions allow a "pick position N" convention →
    self-play converges near 1.0; cross-play between independently trained
    seeds collapses (different seeds, different N).
  With OPTestPositionShuffleWrapper, per-agent random position perm breaks
    that convention → only OP-invariant signal is "pick the unique (red)
    card" → 0.9 in both self- and cross-play.
"""
from functools import partial
from typing import Any, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jaxmarl.environments import spaces as jaxmarl_spaces

from envs.base_env import BaseEnv, WrappedEnvState
from envs.card_game.action_utils import (
    NO_COMM_ACTION_DIM,
    decode_pick_or_noop,
    get_action_mask,
)
from envs.card_game.rendering import (
    AGENT_0_COLOR,
    AGENT_1_COLOR,
    GRID_COLS,
    GRID_ROWS,
    NUM_CARDS,
    TILE_PIXELS,
    _CARD_MASK,
    _DOWN_TRI_MASK,
    _UP_TRI_MASK,
    _render_tile,
)


_EGO_HIGHLIGHT_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)
RED_COLOR = jnp.array([220, 50, 50], dtype=jnp.uint8)
WHITE_COLOR = jnp.array([220, 220, 220], dtype=jnp.uint8)

_AGENT_POSITIONS = jnp.array([
    [0, 2],  # agent 0: top center
    [2, 2],  # agent 1: bottom center
], dtype=jnp.int32)


@dataclass
class CardGameOPTestState:
    red_position: chex.Array      # int32, 0..NUM_CARDS-1
    step_count: chex.Array        # int32
    agent_choices: chex.Array     # (2,) int32, -1 until decision step


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


def render_op_test_card_game(red_position: jnp.ndarray) -> jnp.ndarray:
    """Render the scene: 4 white cards + 1 red card at red_position."""
    h_px = GRID_ROWS * TILE_PIXELS
    w_px = GRID_COLS * TILE_PIXELS
    img = jnp.zeros((h_px, w_px, 3), dtype=jnp.uint8)

    # Agents (fixed positions, never shuffled)
    agent0_tile = _render_tile(_DOWN_TRI_MASK, AGENT_0_COLOR)
    img = jax.lax.dynamic_update_slice(img, agent0_tile, (0, 2 * TILE_PIXELS, 0))

    agent1_tile = _render_tile(_UP_TRI_MASK, AGENT_1_COLOR)
    img = jax.lax.dynamic_update_slice(
        img, agent1_tile, (2 * TILE_PIXELS, 2 * TILE_PIXELS, 0))

    # Card row: red at red_position, white elsewhere
    def draw_card(img, i):
        is_red = jnp.equal(i, red_position)
        color = jnp.where(is_red, RED_COLOR, WHITE_COLOR)
        card_tile = _render_tile(_CARD_MASK, color)
        img = jax.lax.dynamic_update_slice(
            img, card_tile, (TILE_PIXELS, i * TILE_PIXELS, 0))
        return img, None

    img, _ = jax.lax.scan(draw_card, img, jnp.arange(NUM_CARDS))
    return img


def _unwrap_op_test_state(state):
    """Walk .env_state chain to the base CardGameOPTestState (has red_position)."""
    s = state
    while hasattr(s, 'env_state') and not hasattr(s, 'red_position'):
        s = s.env_state
    return s


def _draw_choice_border_np(img, card_pos, color, thickness):
    """Draw a thick border around the card tile at GT col card_pos on upscaled img."""
    h, w = img.shape[:2]
    tile_h = h // GRID_ROWS
    tile_w = w // GRID_COLS
    y0 = tile_h  # card row = 1
    x0 = card_pos * tile_w
    img[y0:y0 + thickness, x0:x0 + tile_w] = color
    img[y0 + tile_h - thickness:y0 + tile_h, x0:x0 + tile_w] = color
    img[y0:y0 + tile_h, x0:x0 + thickness] = color
    img[y0:y0 + tile_h, x0 + tile_w - thickness:x0 + tile_w] = color
    return img


def render_op_test_eval_frames(ep_states, ep_actions=None, scale: int = 32):
    """Upscaled RGB frames from a list of episode states.

    Shows the GROUND TRUTH view (not per-agent OP-shuffled views). On the
    final step, draws a colored border around the card each agent picked
    (agent 0 = orange, agent 1 = magenta). Both borders on the red tile means
    the agents coordinated on the focal card.

    Args:
        ep_states: list of WrappedEnvState. The env auto-resets on done, so
            agent_choices in state is always -1 at the timestep we record —
            we can't read picks from state.
        ep_actions: list of (act_0, act_1) tuples collected by
            run_episode_with_states. These are GT-space picks (after any OP
            wrapper inversion). Used to draw the decision-step borders.
    """
    import numpy as np
    from PIL import Image

    agent0_border = np.array([255, 140, 0], dtype=np.uint8)
    agent1_border = np.array([255, 0, 255], dtype=np.uint8)
    thickness = max(2, scale // 8)

    frames = []
    last_action = ep_actions[-1] if ep_actions else None
    for t, state in enumerate(ep_states):
        inner = _unwrap_op_test_state(state)
        img = np.array(render_op_test_card_game(inner.red_position))
        h, w = img.shape[:2]
        frame = np.array(Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST))

        # Draw borders only on the final recorded frame (decision step)
        is_final = (t == len(ep_states) - 1)
        if is_final and last_action is not None:
            a0, a1 = int(last_action[0]), int(last_action[1])
            if 0 <= a0 < NUM_CARDS:
                _draw_choice_border_np(frame, a0, agent0_border, thickness)
            if 0 <= a1 < NUM_CARDS:
                _draw_choice_border_np(frame, a1, agent1_border, thickness)

        frames.append(frame)
    return frames


class CardGameOPTestEnv(BaseEnv):
    """OP-test diagnostic env: 4 white + 1 red card, position-based picks.

    No communication channel. Reward shape rewards the asymmetric option
    (red, 0.9) less than the symmetric option (any white, 1.0) so the OP
    cross-play test is meaningful.
    """

    def __init__(self, max_steps: int = 8, **kwargs):
        self.max_steps = max_steps
        self.communication = False  # diagnostic only; no comm
        self.num_cards = NUM_CARDS
        self.num_agents = 2
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self.name = "CardGameOPTest"

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
        return jaxmarl_spaces.Discrete(num_categories=NO_COMM_ACTION_DIM)

    def _make_obs(self, env_state: CardGameOPTestState) -> Dict[str, jnp.ndarray]:
        """Per-agent obs: scene + ego highlight + decision indicator."""
        img = render_op_test_card_game(env_state.red_position)
        is_decision = (env_state.step_count + 1) >= self.max_steps
        white = jnp.array([255, 255, 255], dtype=jnp.uint8)

        obs = {}
        for i in range(self.num_agents):
            row, col = _AGENT_POSITIONS[i]
            agent_img = _draw_border(
                img, row, col, self.tile_size, _EGO_HIGHLIGHT_COLOR
            )
            decision_img = agent_img.at[0:4, 0:4, :].set(white[None, None, :])
            agent_img = jnp.where(is_decision, decision_img, agent_img)
            obs[self.agents[i]] = agent_img.flatten().astype(jnp.float32) / 255.0
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], WrappedEnvState]:
        red_position = jax.random.randint(key, (), 0, NUM_CARDS)
        env_state = CardGameOPTestState(
            red_position=red_position,
            step_count=jnp.int32(0),
            agent_choices=jnp.full(2, -1, dtype=jnp.int32),
        )
        obs = self._make_obs(env_state)
        return obs, WrappedEnvState(
            env_state=env_state,
            base_return_so_far=jnp.zeros(self.num_agents),
            avail_actions=jnp.zeros(self.num_agents),
            step=jnp.int32(0),
        )

    def _base_reward(self, pick_0, pick_1, is_decision, red_position):
        """0.9 for matched red, 1.0 for matched non-red, 0 otherwise.

        Requires both picks valid (>= 0) so non-pick actions never satisfy
        `-1 == -1` and score.
        """
        valid = (pick_0 >= 0) & (pick_1 >= 0)
        match = valid & jnp.equal(pick_0, pick_1)
        is_red_pick = jnp.equal(pick_0, red_position)
        coord_reward = jnp.where(is_red_pick, 0.9, 1.0)
        return jnp.where(is_decision & match, coord_reward, 0.0)

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

        pick_0 = decode_pick_or_noop(actions["agent_0"])
        pick_1 = decode_pick_or_noop(actions["agent_1"])

        reward_val = self._base_reward(
            pick_0, pick_1, is_decision, env_state.red_position
        )
        reward = {agent: reward_val for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones["__all__"] = done

        choices = jnp.where(
            is_decision,
            jnp.array([pick_0, pick_1], dtype=jnp.int32),
            jnp.full(2, -1, dtype=jnp.int32),
        )
        new_env_state = env_state.replace(
            step_count=new_step, agent_choices=choices
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

        # Diagnostic flags (decision-step only). Aggregate across rollouts to
        # see what cross-play picks: red_match → focal lock; white_match →
        # symmetric coordination.
        valid = (pick_0 >= 0) & (pick_1 >= 0)
        match = valid & jnp.equal(pick_0, pick_1)
        is_red = jnp.equal(pick_0, env_state.red_position)
        red_match_val = jnp.where(
            is_decision, (match & is_red).astype(jnp.float32), 0.0
        )
        white_match_val = jnp.where(
            is_decision, (match & ~is_red).astype(jnp.float32), 0.0
        )

        info = {
            "base_reward": base_reward_arr,
            "base_return": base_return,
            "red_match": jnp.array([red_match_val, red_match_val]),
            "white_match": jnp.array([white_match_val, white_match_val]),
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
        mask = get_action_mask(is_decision, communication=False)
        return {agent: mask for agent in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: WrappedEnvState) -> jnp.array:
        return state.step


# ---------------------------------------------------------------------------
#  Position-shuffle wrapper (the only OP transform meaningful here)
# ---------------------------------------------------------------------------

@dataclass
class OPTestPositionShuffleState:
    env_state: Any
    per_agent_perm: Dict[str, chex.Array]  # {agent_name: (NUM_CARDS,) int32}


class OPTestPositionShuffleWrapper:
    """Per-agent independent card position shuffle.

    Obs: card row tiles permuted per agent; tile shown at view position j is
    the GT tile at position perm[j].
    Action inversion: agent's pick of view position j → GT position perm[j].
    """

    def __init__(self, env: CardGameOPTestEnv):
        self._env = env

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        env_key, wrap_key = jax.random.split(key)
        obs, env_state = self._env.reset(env_key)

        keys = jax.random.split(wrap_key, self._env.num_agents)
        per_agent_perm = {
            a: jax.random.permutation(keys[i], NUM_CARDS)
            for i, a in enumerate(self._env.agents)
        }

        state = OPTestPositionShuffleState(
            env_state=env_state, per_agent_perm=per_agent_perm,
        )
        new_obs = {
            a: self._shuffle_obs(obs[a], per_agent_perm[a])
            for a in self._env.agents
        }
        return new_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, reset_state=None):
        env_key, wrap_key = jax.random.split(key)
        true_action = self._invert_actions(action, state)

        obs, env_state, reward, done, info = self._env.step(
            env_key, state.env_state, true_action,
        )

        # Sample new perms (used only on auto-reset)
        keys = jax.random.split(wrap_key, self._env.num_agents)
        new_perms = {
            a: jax.random.permutation(keys[i], NUM_CARDS)
            for i, a in enumerate(self._env.agents)
        }
        is_done = done["__all__"]
        current_perms = {
            a: jnp.where(is_done, new_perms[a], state.per_agent_perm[a])
            for a in self._env.agents
        }

        new_obs = {
            a: self._shuffle_obs(obs[a], current_perms[a])
            for a in self._env.agents
        }
        new_state = OPTestPositionShuffleState(
            env_state=env_state, per_agent_perm=current_perms,
        )
        return new_obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: OPTestPositionShuffleState):
        return self._env.get_avail_actions(state.env_state)

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: OPTestPositionShuffleState):
        return self._env.get_step_count(state.env_state)

    # -- transforms ----------------------------------------------------------

    def _shuffle_obs(self, flat_obs, perm):
        """Rearrange card-row tiles: view col j ← GT col perm[j]."""
        img = (flat_obs * 255.0).astype(jnp.uint8).reshape(
            self._env._img_h, self._env._img_w, 3,
        )
        TP = TILE_PIXELS
        card_row = img[TP:2 * TP, :, :]
        tiles = card_row.reshape(TP, NUM_CARDS, TP, 3)
        shuffled = tiles[:, perm, :, :]
        new_card_row = shuffled.reshape(TP, NUM_CARDS * TP, 3)
        img = img.at[TP:2 * TP, :, :].set(new_card_row)
        return img.flatten().astype(jnp.float32) / 255.0

    def _invert_actions(self, action, state):
        """Map view picks → GT picks via per-agent perm."""
        true_action = {}
        for a in self._env.agents:
            perm = state.per_agent_perm[a]
            view_pick = jnp.asarray(action[a], dtype=jnp.int32)
            is_pick = view_pick < NUM_CARDS
            safe_idx = jnp.clip(view_pick, 0, NUM_CARDS - 1)
            gt_pick = perm[safe_idx]
            true_action[a] = jnp.where(is_pick, gt_pick, view_pick)
        return true_action

    # -- pass-through --------------------------------------------------------

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def __getattr__(self, name):
        return getattr(self._env, name)
