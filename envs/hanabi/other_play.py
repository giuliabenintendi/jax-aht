"""Other-Play wrapper for the Hanabi image environment.

Per-agent independent colour permutation (S_5, 120 permutations) sampled per
episode. The agent sees the rendered image with cards / fireworks / hint
stripes redrawn in the permuted colours. Three things are inverse-mapped
back to ground truth before reaching the inner env:

  - HINT_COLOR_X actions (indices 10-14): the visual-colour-X the agent
    chose corresponds to a different ground-truth colour. All other actions
    (discard, play, hint-rank, noop) are colour-agnostic and pass through.
  - The legal-action mask's HINT_COLOR_X section: legality depends on the
    partner's actual hand, which is in ground-truth colours, so the mask
    bits 10-14 are permuted to match the agent's visual frame.

No position-shuffle analogue for Hanabi — slot positions in your own hand
carry semantic information (the policy reasons about "slot 2" specifically),
so shuffling them per agent would corrupt the protocol rather than impose a
benign symmetry. Colour permutation alone is what Hu et al. 2020 used.
"""
from functools import partial
from itertools import permutations
from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp
from flax.struct import dataclass

from envs.hanabi.rendering import HANABI_COLORS, IMG_H, IMG_W


NUM_COLORS = 5
# Hanabi action layout (2-player defaults):
#   0-4   discard slot N
#   5-9   play slot N
#   10-14 hint colour X to partner   <- inverse-mapped by this wrapper
#   15-19 hint rank R to partner
#   20    noop
HINT_COLOR_START = 10
HINT_COLOR_END = 14  # inclusive


def remap_recoloured_action(action, inv_recolouring):
    """Map an action from recoloured space back to ground truth.

    Only HINT_COLOR_X actions (indices 10-14) are affected. For one of those,
    the agent wants to hint the colour visible at index X in their recoloured
    frame, which corresponds to ground-truth colour `inv_recolouring[X]`. The
    action becomes `HINT_COLOR_START + inv_recolouring[X]`.

    All other actions (discard, play, hint-rank, noop) are colour-agnostic and
    pass through unchanged.
    """
    action = jnp.asarray(action, dtype=jnp.int32)
    is_colour_hint = (action >= HINT_COLOR_START) & (action <= HINT_COLOR_END)
    visual_colour = action - HINT_COLOR_START
    # Clamp before indexing so non-hint actions don't OOB into inv_recolouring.
    safe_idx = jnp.clip(visual_colour, 0, NUM_COLORS - 1)
    true_colour = inv_recolouring[safe_idx]
    remapped = HINT_COLOR_START + true_colour
    return jnp.where(is_colour_hint, remapped, action)


@dataclass
class OPHanabiRecolouringState:
    env_state: Any
    per_agent_recolouring: Dict[str, chex.Array]      # forward: true colour -> visual colour
    per_agent_inv_recolouring: Dict[str, chex.Array]  # inverse: visual colour -> true colour


class HanabiColourPermutationWrapper:
    """Per-agent S_5 colour permutation, sampled per episode.

    Wraps a `HanabiImageWrapper`-style env emitting flat float32 obs in
    `[0, 1]`. The image is recoloured pixel-wise against `HANABI_COLORS`;
    the five auxiliary palette constants (card-back, info, life, rank-hint,
    background) are pixel-distinct from `HANABI_COLORS` so they pass
    through untouched. This is the invariant the renderer's palette
    purity test pins down.
    """

    def __init__(self, env):
        self._env = env
        self.all_perms = jnp.array(
            list(permutations(range(NUM_COLORS))), dtype=jnp.int32,
        )
        self.num_perms = self.all_perms.shape[0]  # 120 for n=5

    # -- helpers -------------------------------------------------------------

    def _sample_recolouring(self, key):
        idx = jax.random.randint(key, (), 0, self.num_perms)
        return self.all_perms[idx]

    @staticmethod
    def _invert_perm(perm):
        return jnp.zeros(NUM_COLORS, dtype=jnp.int32).at[perm].set(
            jnp.arange(NUM_COLORS, dtype=jnp.int32),
        )

    # -- core env interface --------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        env_key, wrap_key = jax.random.split(key)
        obs, env_state = self._env.reset(env_key)

        keys = jax.random.split(wrap_key, self._env.num_agents)
        per_agent_recolouring = {
            a: self._sample_recolouring(keys[i])
            for i, a in enumerate(self._env.agents)
        }
        per_agent_inv = {
            a: self._invert_perm(per_agent_recolouring[a])
            for a in self._env.agents
        }

        state = OPHanabiRecolouringState(
            env_state=env_state,
            per_agent_recolouring=per_agent_recolouring,
            per_agent_inv_recolouring=per_agent_inv,
        )
        new_obs = {
            a: self._recolour_obs(obs[a], per_agent_recolouring[a])
            for a in self._env.agents
        }
        return new_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, reset_state=None):
        # Inverse-map agent actions from recoloured space -> ground truth
        true_action = self._invert_actions(action, state)

        env_key, wrap_key = jax.random.split(key)
        obs, env_state, reward, done, info = self._env.step(
            env_key, state.env_state, true_action,
        )

        # On auto-reset (episode done), sample new per-agent recolourings.
        keys = jax.random.split(wrap_key, self._env.num_agents)
        new_recolourings = {
            a: self._sample_recolouring(keys[i])
            for i, a in enumerate(self._env.agents)
        }
        is_done = done["__all__"]
        current_recolourings = {
            a: jnp.where(is_done, new_recolourings[a],
                         state.per_agent_recolouring[a])
            for a in self._env.agents
        }
        current_inv = {
            a: self._invert_perm(current_recolourings[a])
            for a in self._env.agents
        }

        new_obs = {
            a: self._recolour_obs(obs[a], current_recolourings[a])
            for a in self._env.agents
        }
        new_state = OPHanabiRecolouringState(
            env_state=env_state,
            per_agent_recolouring=current_recolourings,
            per_agent_inv_recolouring=current_inv,
        )
        return new_obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: OPHanabiRecolouringState):
        """Permute the HINT_COLOR_X section of the mask per agent.

        Other action classes (discard/play/hint-rank/noop) are colour-agnostic
        and pass through. For colour-hint legality the ground-truth mask
        depends on the partner's actual cards, which the recoloured agent
        can't see in ground-truth colour space — so bit X of the visual mask
        must read bit inv_recolouring[X] of the ground-truth mask.
        """
        gt_legal = self._env.get_avail_actions(state.env_state)
        permuted = {}
        for a in self._env.agents:
            inv = state.per_agent_inv_recolouring[a]
            mask = gt_legal[a]
            colour_section = jax.lax.dynamic_slice(
                mask, (HINT_COLOR_START,), (NUM_COLORS,),
            )
            permuted_colour = colour_section[inv]
            permuted[a] = jax.lax.dynamic_update_slice(
                mask, permuted_colour, (HINT_COLOR_START,),
            )
        return permuted

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: OPHanabiRecolouringState):
        return self._env.get_step_count(state.env_state)

    # -- action inverse-mapping ----------------------------------------------

    def _invert_actions(self, action, state):
        true_action = {}
        for a in self._env.agents:
            inv = state.per_agent_inv_recolouring[a]
            true_action[a] = remap_recoloured_action(action[a], inv)
        return true_action

    # -- observation transform -----------------------------------------------

    def _recolour_obs(self, flat_obs, recolouring):
        """Pixel-match the obs image against HANABI_COLORS and swap colours.

        The image wrapper emits flat float32 in [0, 1]; convert to uint8 to
        compare exactly against HANABI_COLORS (no anti-aliasing in the
        renderer, so equality matches), then back to float32 at the end.
        """
        img = (flat_obs * 255.0).astype(jnp.uint8).reshape(IMG_H, IMG_W, 3)

        # Match each pixel against all 5 card colours: (H, W, 5)
        matches = jnp.all(
            img[:, :, None, :] == HANABI_COLORS[None, None, :, :],
            axis=-1,
        )

        new_colors = HANABI_COLORS[recolouring]  # (5, 3) - visual palette
        recoloured = jnp.einsum(
            "hws,sc->hwc",
            matches.astype(jnp.float32),
            new_colors.astype(jnp.float32),
        ).astype(jnp.uint8)

        any_match = jnp.any(matches, axis=-1, keepdims=True)
        new_img = jnp.where(any_match, recoloured, img)
        return new_img.flatten().astype(jnp.float32) / 255.0

    # -- pass-through --------------------------------------------------------

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def __getattr__(self, name):
        return getattr(self._env, name)
