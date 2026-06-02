"""Policy wrapper for the Image Actor-Critic (no attention baseline).

Same hstate packing as JAActorCriticPolicy but without partner_hstate
or attention map returns.
"""
from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.image_actor_critic import ImageActorCritic


class ImageActorCriticPolicy(AgentPolicy):
    """Policy wrapper for ImageActorCritic.

    Hidden state layout:
      hstate shape: (1, batch, lstm_hidden_dim * 2)
      Packs the shared LSTM carry `(h, c)` into a single array.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        img_height: int,
        img_width: int,
        conv_filters: int = 32,
        conv_num_blocks: int = 4,
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: str = "SAME",
        fc_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
    ):
        super().__init__(action_dim, obs_dim)
        self.img_height = img_height
        self.img_width = img_width
        self.network = ImageActorCritic(
            action_dim=action_dim,
            img_height=img_height,
            img_width=img_width,
            conv_filters=conv_filters,
            conv_num_blocks=conv_num_blocks,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            fc_hidden_dim=fc_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
        )
        self.lstm_hidden_dim = lstm_hidden_dim

    def _pack_hstate(self, lstm_state):
        """Pack shared `(h, c)` LSTM state -> (1, batch, 2*dim)."""
        h, c = lstm_state
        packed = jnp.concatenate([h, c], axis=-1)
        return packed[None, ...]

    def _unpack_hstate(self, hstate):
        """Unpack (1, batch, 2*dim) -> shared `(h, c)` LSTM state."""
        flat = hstate.squeeze(0)
        d = self.lstm_hidden_dim
        h = flat[..., 0:d]
        c = flat[..., d:2*d]
        return (h, c)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, greedy=False):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, _ = self.network.apply(
            params, hidden, (obs, done, avail_actions)
        )
        action = jax.lax.cond(
            greedy,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        new_hstate = self._pack_hstate(new_hidden)
        return action, new_hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, val = self.network.apply(
            params, hidden, (obs, done, avail_actions)
        )
        action = pi.sample(seed=rng)
        new_hstate = self._pack_hstate(new_hidden)
        return action, val, pi, new_hstate

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize packed hidden state: (1, batch, 2*lstm_hidden_dim)."""
        d = self.lstm_hidden_dim
        return jnp.zeros((1, batch_size, 2 * d))

    def init_params(self, rng):
        batch_size = 1
        init_hstate = self.init_hstate(batch_size)
        hidden = self._unpack_hstate(init_hstate)

        seq_len = 1
        dummy_obs = jnp.zeros((seq_len, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((seq_len, batch_size))
        dummy_avail = jnp.ones((seq_len, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail)

        return self.network.init(rng, hidden, dummy_x)
