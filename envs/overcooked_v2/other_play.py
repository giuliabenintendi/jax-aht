"""Other-Play wrapper for the image-based Overcooked V2 environment.

Implements Other-Play (Hu et al., 2020) over the ingredient-permutation symmetry
of Overcooked V2, matching the paper (`op_ingredient_permutations`). Each agent
independently sees the recipe ingredients relabeled by a per-episode permutation,
so no fixed colour convention can be relied on between independently-trained
agents. The permutation is spatially identity (recolour only), so the shared
allocentric frame the joint-attention machinery relies on is preserved. There is
no action remap — ingredients are not actions.

Unlike the LBF/card OP wrappers, the *sampling* is done by the base env
(`OvercookedV2` with `op_ingredient_permutations` set stores a per-agent
permutation in `state.ingredient_permutations` — verified byte-identical to the
paper's `_sample_op_ingredient_permutations`). This wrapper only reproduces the
env's symbolic `get_obs` relabeling in image space: it recolours the rendered
ingredient pixels of each agent's observation by that agent's permutation.

Usage:
    env = OvercookedV2ImageWrapper(..., op_ingredient_permutations=[0, 1])
    env = OvercookedV2OtherPlayWrapper(env)
"""
from __future__ import annotations

from functools import partial
from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp
import numpy as np

from envs.overcooked_v2.rendering import AGENT_COLORS, COLORS, INGREDIENT_COLORS

# Magenta ego border drawn by the image wrapper; must not be recoloured.
_EGO_HIGHLIGHT_COLOR = np.array([255, 0, 255], dtype=np.uint8)
# Colours the renderer uses for NON-ingredient elements (walls/pot, goal, plates,
# pot lid, recipe-indicator base). An ingredient colour colliding with any of
# these would make the pixel-wise recolour corrupt non-ingredient pixels.
_STRUCTURAL_COLOR_NAMES = ("grey", "green", "white", "black", "brown", "red")


class OvercookedV2OtherPlayWrapper:
    """Per-agent ingredient-permutation Other-Play for image-obs Overcooked V2.

    Reads the base env's per-agent permutation from
    `state.env_state.ingredient_permutations` and recolours each agent's rendered
    observation accordingly, reproducing the env's symbolic OP in image space.
    Stateless: the wrapped image env's state is passed through unchanged.
    """

    def __init__(self, env):
        self._env = env
        base = env.env  # OvercookedV2
        if not base.op_ingredient_permutations:
            raise ValueError(
                "OvercookedV2OtherPlayWrapper requires the base env to be built with "
                "op_ingredient_permutations set (so state.ingredient_permutations is sampled)."
            )
        self._n_ing = int(base.layout.num_ingredients)
        self._op_indices = [int(i) for i in base.op_ingredient_permutations]

        colors = np.asarray(INGREDIENT_COLORS)[: self._n_ing]  # (n_ing, 3) uint8
        self._assert_palette_unique(colors, env.num_agents)
        # Normalised ingredient palette, matching the image wrapper's obs scale.
        self._colors = jnp.asarray(colors, dtype=jnp.float32) / 255.0

    def _assert_palette_unique(self, colors: np.ndarray, num_agents: int) -> None:
        """Fail fast if a permuted ingredient's colour is not unique in the palette."""
        structural = np.stack([np.asarray(COLORS[c]) for c in _STRUCTURAL_COLOR_NAMES])
        agents_used = np.asarray(AGENT_COLORS)[:num_agents]
        forbidden = np.concatenate(
            [structural, agents_used, _EGO_HIGHLIGHT_COLOR[None]], axis=0
        )
        for j in self._op_indices:
            c = colors[j]
            if np.any(np.all(forbidden == c, axis=-1)):
                raise ValueError(
                    f"OP ingredient {j} colour {tuple(int(v) for v in c)} collides with a "
                    "non-ingredient render colour; the pixel recolour would corrupt it."
                )
            if int(np.sum(np.all(colors == c, axis=-1))) > 1:
                raise ValueError(
                    f"OP ingredient {j} colour {tuple(int(v) for v in c)} is not unique "
                    "among ingredient colours."
                )

    def _recolour(self, obs_flat: jnp.ndarray, perm: jnp.ndarray) -> jnp.ndarray:
        """Recolour ingredient pixels of one agent's flat obs by its permutation.

        Mirrors the env's symbolic `get_obs`: true ingredient i is shown with the
        colour of ingredient `perm^{-1}[i]`, so the agent perceives it as the
        ingredient occupying slot i under its permutation.
        """
        px = obs_flat.reshape(-1, 3)
        inv = jnp.zeros_like(perm).at[perm].set(
            jnp.arange(perm.shape[0], dtype=perm.dtype)
        )
        matches = jnp.all(px[:, None, :] == self._colors[None, :, :], axis=-1)  # (N, n_ing)
        new_colors = self._colors[inv]  # (n_ing, 3)
        recoloured = jnp.einsum("ni,ic->nc", matches.astype(jnp.float32), new_colors)
        any_match = jnp.any(matches, axis=-1, keepdims=True)
        return jnp.where(any_match, recoloured, px).reshape(-1)

    def _apply_op(self, obs: Dict[str, chex.Array], state) -> Dict[str, chex.Array]:
        perms = state.env_state.ingredient_permutations  # (num_agents, n_ing)
        return {
            a: self._recolour(obs[a], perms[i])
            for i, a in enumerate(self._env.agents)
        }

    def reset(self, key: chex.PRNGKey):
        obs, state = self._env.reset(key)
        return self._apply_op(obs, state), state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions, reset_state=None):
        obs, new_state, rewards, dones, info = self._env.step(
            key, state, actions, reset_state
        )
        return self._apply_op(obs, new_state), new_state, rewards, dones, info

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state):
        return self._env.get_avail_actions(state)

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state):
        return self._env.get_step_count(state)

    def observation_space(self, agent: str = ""):
        return self._env.observation_space(agent)

    def action_space(self, agent: str = ""):
        return self._env.action_space(agent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)
