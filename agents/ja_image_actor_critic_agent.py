"""Policy wrapper for the Joint Attention Actor-Critic with image observations.

Extends JAActorCriticPolicy with per-agent head support. During evaluation,
single-agent inputs are padded to batch_size=2 so the network can route
to the correct agent's head (position 0 = agent 0, position 1 = agent 1).
"""
from functools import partial

import jax
import jax.numpy as jnp

from agents.agent_interface import AgentPolicy
from agents.ja_actor_critic_agent import JAActorCriticPolicy
from agents.ja_image_actor_critic import JAImageActorCritic


class JAImageActorCriticPolicy(JAActorCriticPolicy):

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
        num_heads: int = 4,
        head_features: int = 16,
        fc_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        spatial_basis_depth: int = 8,
    ):
        # Skip JAActorCriticPolicy.__init__ — we set self.network directly
        # but still call AgentPolicy.__init__ for action_dim/obs_dim
        AgentPolicy.__init__(self, action_dim, obs_dim)

        self.img_height = img_height
        self.img_width = img_width
        self.network = JAImageActorCritic(
            action_dim=action_dim,
            img_height=img_height,
            img_width=img_width,
            conv_filters=conv_filters,
            conv_num_blocks=conv_num_blocks,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            num_heads=num_heads,
            head_features=head_features,
            fc_hidden_dim=fc_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            spatial_basis_depth=spatial_basis_depth,
        )
        self.lstm_hidden_dim = lstm_hidden_dim

    def _pad_for_eval(self, obs, done, avail_actions, hstate, agent_id):
        """Pad single-agent eval inputs to batch_size=2 for per-agent routing.

        The network splits the batch at midpoint: position 0 -> agent 0 head,
        position 1 -> agent 1 head. Places real inputs at the correct position,
        fills the other with zeros (ones for avail_actions).
        """
        # Broadcast avail_actions to (seq, 1, n_actions) if needed
        avail_actions = jnp.broadcast_to(
            avail_actions, obs.shape[:-1] + (avail_actions.shape[-1],)
        )

        def _pad(real, fill):
            parts = [fill, fill]
            parts[agent_id] = real
            return jnp.concatenate(parts, axis=1)

        return (
            _pad(obs, jnp.zeros_like(obs)),
            _pad(done, jnp.zeros_like(done)),
            _pad(avail_actions, jnp.ones_like(avail_actions)),
            _pad(hstate, jnp.zeros_like(hstate)),
        )

    @partial(jax.jit, static_argnums=(0, 10))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, test_mode=False, agent_id=0):
        batch_obs, batch_done, batch_avail, batch_hstate = self._pad_for_eval(
            obs, done, avail_actions, hstate, agent_id)

        hidden = self._unpack_hstate(batch_hstate)
        new_hidden, pi, _, _ = self.network.apply(
            params, hidden, (batch_obs, batch_done, batch_avail)
        )
        action = jax.lax.cond(
            test_mode,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        # Extract this agent's outputs
        action = action[..., agent_id:agent_id+1]
        new_hstate = self._pack_hstate(*new_hidden)
        new_hstate = new_hstate[:, agent_id:agent_id+1]
        return action, new_hstate

    @partial(jax.jit, static_argnums=(0, 8))
    def get_action_and_attention(self, params, obs, done, avail_actions, hstate, rng,
                                 test_mode=False, agent_id=0):
        batch_obs, batch_done, batch_avail, batch_hstate = self._pad_for_eval(
            obs, done, avail_actions, hstate, agent_id)

        hidden = self._unpack_hstate(batch_hstate)
        new_hidden, pi, _, attn_map = self.network.apply(
            params, hidden, (batch_obs, batch_done, batch_avail)
        )
        action = jax.lax.cond(
            test_mode,
            lambda: pi.mode(),
            lambda: pi.sample(seed=rng),
        )
        action = action[..., agent_id:agent_id+1]
        new_hstate = self._pack_hstate(*new_hidden)
        new_hstate = new_hstate[:, agent_id:agent_id+1]
        attn = attn_map[..., agent_id:agent_id+1, :, :]
        return action, new_hstate, attn

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        """Full-batch training inference. No padding needed."""
        hidden = self._unpack_hstate(hstate)
        new_hidden, pi, val, attn_map = self.network.apply(
            params, hidden, (obs, done, avail_actions)
        )
        action = pi.sample(seed=rng)
        new_hstate = self._pack_hstate(*new_hidden)
        return action, val, pi, new_hstate, attn_map

    def init_hstate(self, batch_size, aux_info=None):
        """Initialize packed hidden state: (1, batch, 4*lstm_hidden_dim)."""
        d = self.lstm_hidden_dim
        return jnp.zeros((1, batch_size, 4 * d))

    def init_params(self, rng):
        """Initialize parameters. Batch size must be >= 2 for the per-agent split."""
        batch_size = 2
        init_hstate = self.init_hstate(batch_size)
        hidden = self._unpack_hstate(init_hstate)

        seq_len = 1
        dummy_obs = jnp.zeros((seq_len, batch_size, self.obs_dim))
        dummy_done = jnp.zeros((seq_len, batch_size))
        dummy_avail = jnp.ones((seq_len, batch_size, self.action_dim))
        dummy_x = (dummy_obs, dummy_done, dummy_avail)

        return self.network.init(rng, hidden, dummy_x)
