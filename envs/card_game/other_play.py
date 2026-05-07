"""Other-Play wrappers for the Card Game environment.

Two composable wrappers implementing the Other-Play algorithm (Hu et al., 2020):

1. CardGamePositionShuffleWrapper — per-agent independent card position shuffling.
   Disables the env's internal shuffle and handles all spatial randomization.
   Actions are color-based, so no action remapping is needed.

2. CardGameRecolouringWrapper — per-agent independent color permutation (S5).
   Swaps card RGB values in the observation and inverse-maps color-based actions
   back to ground truth before passing to the env.

Usage:
    env = CardGameEnv(shuffle=False, ...)
    env = CardGamePositionShuffleWrapper(env)
    env = CardGameRecolouringWrapper(env)

The ground truth MDP runs inside the base env. Each wrapper transforms the
observation (apply symmetry) and inverse-transforms the action (undo symmetry)
so the env always sees ground truth actions and produces ground truth rewards.
"""
from itertools import permutations
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import chex
from flax.struct import dataclass

from envs.card_game.rendering import TILE_PIXELS, NUM_CARDS, CARD_COLORS


def remap_recoloured_action(action, inv_recolouring):
    """Map a recoloured-space action back to ground-truth color identity."""
    action = jnp.asarray(action, dtype=jnp.int32)
    return inv_recolouring[action]


# ---------------------------------------------------------------------------
#  Wrapper states
# ---------------------------------------------------------------------------

@dataclass
class OPPositionShuffleState:
    env_state: Any
    per_agent_perm: Dict[str, chex.Array]  # {agent_name: (NUM_CARDS,) int32}


@dataclass
class OPRecolouringState:
    env_state: Any
    per_agent_recolouring: Dict[str, chex.Array]      # forward: true color → visual color
    per_agent_inv_recolouring: Dict[str, chex.Array]   # inverse: visual color → true color


# ---------------------------------------------------------------------------
#  Position shuffle wrapper
# ---------------------------------------------------------------------------

class CardGamePositionShuffleWrapper:
    """Per-agent independent card position shuffling.

    Disables the inner env's shuffle and applies an independent random
    permutation of card tile positions for each agent per episode.
    Actions are color-based so no action remapping is needed.
    """

    def __init__(self, env):
        self._env = env
        self._env.shuffle = False

    # -- core env interface --------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        env_key, wrap_key = jax.random.split(key)
        obs, env_state = self._env.reset(env_key)

        keys = jax.random.split(wrap_key, self._env.num_agents)
        per_agent_perm = {
            a: jax.random.permutation(keys[i], NUM_CARDS)
            for i, a in enumerate(self._env.agents)
        }

        state = OPPositionShuffleState(
            env_state=env_state,
            per_agent_perm=per_agent_perm,
        )
        new_obs = {
            a: self._shuffle_obs(obs[a], per_agent_perm[a])
            for a in self._env.agents
        }
        return new_obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, reset_state=None):
        env_key, wrap_key = jax.random.split(key)
        obs, env_state, reward, done, info = self._env.step(
            env_key, state.env_state, action,
        )

        # On auto-reset (episode done), sample new permutations
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
        new_state = OPPositionShuffleState(
            env_state=env_state,
            per_agent_perm=current_perms,
        )
        return new_obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: OPPositionShuffleState):
        return self._env.get_avail_actions(state.env_state)

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: OPPositionShuffleState):
        return self._env.get_step_count(state.env_state)

    # -- observation transform -----------------------------------------------

    def _shuffle_obs(self, flat_obs, perm):
        """Rearrange card tiles in the observation image according to perm."""
        img = (flat_obs * 255.0).astype(jnp.uint8).reshape(
            self._env._img_h, self._env._img_w, 3,
        )
        TP = TILE_PIXELS
        card_row = img[TP:2 * TP, :, :]                     # (TP, 5*TP, 3)
        tiles = card_row.reshape(TP, NUM_CARDS, TP, 3)       # (TP, 5, TP, 3)
        shuffled = tiles[:, perm, :, :]
        new_card_row = shuffled.reshape(TP, NUM_CARDS * TP, 3)
        img = img.at[TP:2 * TP, :, :].set(new_card_row)
        return img.flatten().astype(jnp.float32) / 255.0

    # -- pass-through --------------------------------------------------------

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def __getattr__(self, name):
        return getattr(self._env, name)


# ---------------------------------------------------------------------------
#  Recolouring wrapper
# ---------------------------------------------------------------------------

class CardGameRecolouringWrapper:
    """Per-agent independent color permutation.

    Each agent sees a recoloured version of the cards (S5 = 120 permutations).
    Actions (color indices, and messages when communication is enabled) are
    inverse-mapped back to ground truth before reaching the env.
    """

    def __init__(self, env):
        self._env = env
        self.all_perms = jnp.array(
            list(permutations(range(NUM_CARDS))), dtype=jnp.int32,
        )
        self.num_perms = self.all_perms.shape[0]  # 120

    # -- helpers -------------------------------------------------------------

    def _sample_recolouring(self, key):
        idx = jax.random.randint(key, (), 0, self.num_perms)
        return self.all_perms[idx]

    @staticmethod
    def _invert_perm(perm):
        return jnp.zeros(NUM_CARDS, dtype=jnp.int32).at[perm].set(
            jnp.arange(NUM_CARDS, dtype=jnp.int32),
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

        state = OPRecolouringState(
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
        # Inverse-map agent actions from recoloured space → ground truth
        true_action = self._invert_actions(action, state)

        env_key, wrap_key = jax.random.split(key)
        obs, env_state, reward, done, info = self._env.step(
            env_key, state.env_state, true_action,
        )

        # On auto-reset, sample new recolourings
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
        new_state = OPRecolouringState(
            env_state=env_state,
            per_agent_recolouring=current_recolourings,
            per_agent_inv_recolouring=current_inv,
        )
        return new_obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: OPRecolouringState):
        return self._env.get_avail_actions(state.env_state)

    @partial(jax.jit, static_argnums=(0,))
    def get_step_count(self, state: OPRecolouringState):
        return self._env.get_step_count(state.env_state)

    # -- action inverse-mapping ----------------------------------------------

    def _invert_actions(self, action, state):
        """Map agent actions from recoloured space back to ground truth."""
        true_action = {}

        for a in self._env.agents:
            inv = state.per_agent_inv_recolouring[a]
            true_action[a] = remap_recoloured_action(action[a], inv)

        return true_action

    # -- observation transform -----------------------------------------------

    def _recolour_obs(self, flat_obs, recolouring):
        """Swap card RGB values in the card row of the observation image.

        Matches each pixel against all 5 CARD_COLORS simultaneously and
        replaces with the recoloured version. Non-card pixels (agent colors,
        communication dots, ego borders, decision indicator) are untouched.
        """
        img = (flat_obs * 255.0).astype(jnp.uint8).reshape(
            self._env._img_h, self._env._img_w, 3,
        )
        TP = TILE_PIXELS
        card_row = img[TP:2 * TP, :, :]  # (TP, 5*TP, 3)

        # Match each pixel against all 5 card colors: (TP, 5*TP, 5)
        matches = jnp.all(
            card_row[:, :, None, :] == CARD_COLORS[None, None, :, :],
            axis=-1,
        )

        # Recoloured palette
        new_colors = CARD_COLORS[recolouring]  # (5, 3)
        recoloured = jnp.einsum(
            'hws,sc->hwc',
            matches.astype(jnp.float32),
            new_colors.astype(jnp.float32),
        ).astype(jnp.uint8)

        any_match = jnp.any(matches, axis=-1, keepdims=True)  # (TP, 5*TP, 1)
        new_card_row = jnp.where(any_match, recoloured, card_row)
        img = img.at[TP:2 * TP, :, :].set(new_card_row)

        return img.flatten().astype(jnp.float32) / 255.0

    # -- pass-through --------------------------------------------------------

    def observation_space(self, agent):
        return self._env.observation_space(agent)

    def action_space(self, agent):
        return self._env.action_space(agent)

    def __getattr__(self, name):
        return getattr(self._env, name)
