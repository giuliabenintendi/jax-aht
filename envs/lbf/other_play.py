"""Other-Play wrapper for the image-based LBF environment.

Implements Other-Play (Hu et al., 2020) over geometric symmetries of the square
LBF grid. LBF has no label/colour symmetry to permute (food is unlabelled), so
the symmetries that create arbitrary conventions between independently-trained
agents are geometric reflections (handedness).

The wrapper mirrors the card-game OP design:

- `LBFMirrorOtherPlayWrapper` — per-agent reflection group V4 = {identity,
  horizontal flip, vertical flip, both (= 180-degree rotation)}. Breaks
  left-right and up-down handedness conventions.

Each agent independently samples one group element per episode. Its observation
(the rendered grid image) is transformed by that element, and the action it
emits in that transformed frame is inverse-mapped back to the ground-truth frame
before reaching the env; the action mask is transformed into the agent's frame so
the policy masks the correct actions. NOOP and LOAD are orientation-invariant;
the four movement actions permute. The ground-truth MDP runs inside the base env.

Usage:
    env = LBFImageWrapper(jumanji_env, ...)
    env = LBFMirrorOtherPlayWrapper(env)
"""
from __future__ import annotations

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import dataclass

# jumanji LBF action layout (constants.py): NOOP, UP, DOWN, LEFT, RIGHT, LOAD
# with new_position = position + MOVES[action] in (row, col) where row grows
# downward and col grows rightward.
_MOVES = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]])
_NUM_ACTIONS = 6
_FIXED_ACTIONS = (0, 5)  # NOOP, LOAD: orientation-invariant

# Group elements as (horizontal_flip, vertical_flip, num_quarter_turns_cw),
# applied in that order. Public LBF OP uses only the reflection group V4.
MIRROR_V4 = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]   # id, H, V, H+V(=rot180)


def _dir_after(vec: np.ndarray, h_flip: int, v_flip: int, k: int) -> tuple[int, int]:
    """Apply a group element to a (dr, dc) displacement: flips, then k CW turns."""
    dr, dc = int(vec[0]), int(vec[1])
    if h_flip:
        dc = -dc
    if v_flip:
        dr = -dr
    for _ in range(k):
        dr, dc = dc, -dr  # one 90-degree clockwise turn
    return dr, dc


def _forward_action_perm(h_flip: int, v_flip: int, k: int) -> np.ndarray:
    """F_g: true action index -> the action index it becomes in the g-transformed frame."""
    perm = np.zeros(_NUM_ACTIONS, dtype=np.int32)
    for a in range(_NUM_ACTIONS):
        if a in _FIXED_ACTIONS:
            perm[a] = a
            continue
        dr, dc = _dir_after(_MOVES[a], h_flip, v_flip, k)
        for b in range(_NUM_ACTIONS):
            if b in _FIXED_ACTIONS:
                continue
            if _MOVES[b][0] == dr and _MOVES[b][1] == dc:
                perm[a] = b
                break
    return perm


def _build_tables(img_h: int, img_w: int, elements: list[tuple[int, int, int]]):
    """Precompute per-element pixel-gather and inverse-action tables.

    pix_perm[e] (img_h*img_w,): source flat-pixel index for each output pixel, so
    transformed = obs_pixels[pix_perm[e]] applies element e.
    act_perm[e] (6,): P_g, mapping a view-frame action back to the ground-truth
    action (the inverse of the forward direction permutation).
    """
    base = np.arange(img_h * img_w, dtype=np.int32).reshape(img_h, img_w)
    pix_perm = np.zeros((len(elements), img_h * img_w), dtype=np.int32)
    act_perm = np.zeros((len(elements), _NUM_ACTIONS), dtype=np.int32)
    for e, (h_flip, v_flip, k) in enumerate(elements):
        t = base
        if h_flip:
            t = np.fliplr(t)
        if v_flip:
            t = np.flipud(t)
        t = np.rot90(t, k=-k)  # k quarter-turns clockwise
        pix_perm[e] = t.reshape(-1)
        fwd = _forward_action_perm(h_flip, v_flip, k)
        act_perm[e] = np.argsort(fwd)  # inverse permutation: view-frame -> ground truth
    return pix_perm, act_perm


@dataclass
class OPGeometricState:
    env_state: Any
    per_agent_elem: dict[str, chex.Array]  # {agent_name: scalar int32 group element index}


class _GeometricOtherPlayWrapper:
    """Per-agent independent geometric Other-Play for image-based LBF.

    Subclasses set `elements`, a list of (h_flip, v_flip, k_rot) group elements.
    """

    elements: list[tuple[int, int, int]] = []

    def __init__(self, env):
        self._env = env
        self._img_h = env._img_h
        self._img_w = env._img_w
        self._n = len(self.elements)
        pix_perm, act_perm = _build_tables(self._img_h, self._img_w, self.elements)
        self._pix_perm = jnp.asarray(pix_perm)
        self._act_perm = jnp.asarray(act_perm)

    # -- core env interface --------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        env_key, wrap_key = jax.random.split(key)
        obs, env_state = self._env.reset(env_key)

        keys = jax.random.split(wrap_key, self._env.num_agents)
        per_agent_elem = {
            a: jax.random.randint(keys[i], (), 0, self._n)
            for i, a in enumerate(self._env.agents)
        }
        state = OPGeometricState(env_state=env_state, per_agent_elem=per_agent_elem)
        new_obs = {a: self._transform_obs(obs[a], per_agent_elem[a]) for a in self._env.agents}
        return new_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, reset_state=None):
        true_action = {
            a: self._act_perm[state.per_agent_elem[a]][action[a]]
            for a in self._env.agents
        }

        env_key, wrap_key = jax.random.split(key)
        obs, env_state, reward, done, info = self._env.step(env_key, state.env_state, true_action)

        # On auto-reset, resample each agent's element for the new episode.
        keys = jax.random.split(wrap_key, self._env.num_agents)
        new_elems = {
            a: jax.random.randint(keys[i], (), 0, self._n)
            for i, a in enumerate(self._env.agents)
        }
        is_done = done["__all__"]
        current_elems = {
            a: jnp.where(is_done, new_elems[a], state.per_agent_elem[a])
            for a in self._env.agents
        }
        new_obs = {a: self._transform_obs(obs[a], current_elems[a]) for a in self._env.agents}
        new_state = OPGeometricState(env_state=env_state, per_agent_elem=current_elems)
        return new_obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: OPGeometricState):
        true_avail = self._env.get_avail_actions(state.env_state)
        # Action v is available in the agent's frame iff the ground-truth action
        # it maps to (act_perm[e][v]) is available.
        return {
            a: true_avail[a][self._act_perm[state.per_agent_elem[a]]]
            for a in self._env.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: OPGeometricState):
        return self._env.get_step_count(state.env_state)

    # -- observation transform -----------------------------------------------

    def _transform_obs(self, flat_obs, elem):
        pixels = flat_obs.reshape(self._img_h * self._img_w, 3)
        transformed = pixels[self._pix_perm[elem]]
        return transformed.reshape(-1)

    # -- pass-through --------------------------------------------------------

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def __getattr__(self, name):
        return getattr(self._env, name)


class LBFMirrorOtherPlayWrapper(_GeometricOtherPlayWrapper):
    """Per-agent reflection Other-Play: V4 = {identity, H-flip, V-flip, both}."""

    elements = MIRROR_V4
