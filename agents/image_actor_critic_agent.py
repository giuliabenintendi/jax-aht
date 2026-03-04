"""Policy wrapper for the Image Actor-Critic (no attention baseline).

Same hstate packing as JAActorCriticPolicy but without partner_hstate
or attention map returns.
"""
from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.image_actor_critic import ImageActorCritic, ImageScannedLSTM


class ImageActorCriticPolicy(AgentPolicy):
    """Policy wrapper for ImageActorCritic.

    Hidden state layout:
      hstate shape: (1, batch, lstm_hidden_dim * 4)
      Packs (actor_h, actor_c, critic_h, critic_c) into a single array.
    """

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        img_height: int,
        img_width: int,
        num_scalars: int = 6,
        activation: str = "relu",
        conv_filters: int = 64,
        fc_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        scalar_embed_dim: int = 5,
    ):
        super().__init__(action_dim, obs_dim)
        self.img_height = img_height
        self.img_width = img_width
        self.network = ImageActorCritic(
            action_dim=action_dim,
            img_height=img_height,
            img_width=img_width,
            num_scalars=num_scalars,
            conv_filters=conv_filters,
            fc_hidden_dim=fc_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            scalar_embed_dim=scalar_embed_dim,
            activation=activation,
        )
        self.lstm_hidden_dim = lstm_hidden_dim

    def _pack_hstate(self, actor_lstm_state, critic_lstm_state):
        """Pack ((actor_h, actor_c), (critic_h, critic_c)) -> (1, batch, 4*dim)."""
        actor_h, actor_c = actor_lstm_state
        critic_h, critic_c = critic_lstm_state
        packed = jnp.concatenate([actor_h, actor_c, critic_h, critic_c], axis=-1)
        return packed[None, ...]

    def _unpack_hstate(self, hstate):
        """Unpack (1, batch, 4*dim) -> ((actor_h, actor_c), (critic_h, critic_c))."""
        flat = hstate.squeeze(0)
        d = self.lstm_hidden_dim
        actor_h = flat[..., 0:d]
        actor_c = flat[..., d:2*d]
        critic_h = flat[..., 2*d:3*d]
        critic_c = flat[..., 3*d:4*d]
        return (actor_h, actor_c), (critic_h, critic_c)

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, _ = self.network.apply(
            params, hidden, (obs, done, avail_actions)
        )
        action = jax.lax.cond(
            test_mode,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        new_hstate = self._pack_hstate(*new_hidden)
        return action, new_hstate

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, val = self.network.apply(
            params, hidden, (obs, done, avail_actions)
        )
        action = pi.sample(seed=rng)
        new_hstate = self._pack_hstate(*new_hidden)
        return action, val, pi, new_hstate

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize packed hidden state: (1, batch, 4*lstm_hidden_dim)."""
        d = self.lstm_hidden_dim
        return jnp.zeros((1, batch_size, 4 * d))

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
