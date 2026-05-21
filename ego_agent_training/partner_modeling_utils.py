"""Utilities for ego training with an auxiliary partner-modeling loss."""
from functools import partial
import functools
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from agents.agent_interface import AgentPolicy
from agents.rnn_actor_critic import ScannedRNN


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    prev_action_onehot: jnp.ndarray
    partner_obs: jnp.ndarray
    partner_action_onehot: jnp.ndarray


class ScannedResetLSTM(nn.Module):
    """LSTM scanned over time with internal reset on done."""

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        lstm_state = carry
        ins, dones = x
        lstm_state_0 = jnp.where(
            dones[:, np.newaxis],
            self.initialize_carry(*lstm_state[0].shape)[0],
            lstm_state[0],
        )
        lstm_state_1 = jnp.where(
            dones[:, np.newaxis],
            self.initialize_carry(*lstm_state[1].shape)[1],
            lstm_state[1],
        )
        new_lstm_state, y = nn.OptimizedLSTMCell(features=ins.shape[1])(
            (lstm_state_0, lstm_state_1),
            ins,
        )
        return new_lstm_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.OptimizedLSTMCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActionConditionedLSTMEncoderNetwork(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        hidden, embedding = ScannedResetLSTM()(hidden, (obs, dones))
        embedding = nn.Dense(self.hidden_dim)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(self.output_dim)(embedding)
        return hidden, embedding


class ActionConditionedRNNEncoderNetwork(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        hidden, embedding = ScannedRNN()(hidden, (obs, dones))
        embedding = nn.Dense(self.hidden_dim)(embedding)
        embedding = nn.relu(embedding)
        embedding = nn.Dense(self.output_dim)(embedding)
        return hidden, embedding


class PartnerReconstructionDecoderNetwork(nn.Module):
    hidden_dim: int
    output_dim1: int
    output_dim2: int

    @nn.compact
    def __call__(self, x):
        obs_hidden = nn.Dense(self.hidden_dim)(x)
        obs_hidden = nn.relu(obs_hidden)
        obs_hidden = nn.Dense(self.hidden_dim)(obs_hidden)
        obs_hidden = nn.relu(obs_hidden)
        obs_reconstruction = nn.Dense(self.output_dim1)(obs_hidden)

        act_hidden = nn.Dense(self.hidden_dim)(x)
        act_hidden = nn.relu(act_hidden)
        act_hidden = nn.Dense(self.hidden_dim)(act_hidden)
        act_hidden = nn.relu(act_hidden)
        act_distribution = nn.Dense(self.output_dim2)(act_hidden)
        act_distribution = nn.softmax(act_distribution, axis=-1)
        return obs_reconstruction, act_distribution


class ActionConditionedLSTMEncoder:
    def __init__(self, action_dim, obs_dim, hidden_dim, output_dim):
        self.model = ActionConditionedLSTMEncoderNetwork(hidden_dim, output_dim)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

    def init_hstate(self, batch_size=1, aux_info=None):
        hstate = ScannedResetLSTM.initialize_carry(batch_size, self.hidden_dim)
        return (
            hstate[0].reshape(1, batch_size, self.hidden_dim),
            hstate[1].reshape(1, batch_size, self.hidden_dim),
        )

    def init_params(self, rng):
        batch_size = 1
        init_hstate = self.init_hstate(batch_size=batch_size)
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_act = jnp.zeros((1, batch_size, self.action_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (jnp.concatenate((dummy_obs, dummy_act), axis=-1), dummy_done)
        init_hstate = (
            init_hstate[0].reshape(batch_size, -1),
            init_hstate[1].reshape(batch_size, -1),
        )
        return self.model.init(rng, init_hstate, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, obs, done):
        batch_size = obs.shape[1]
        hstate = (hstate[0].squeeze(0), hstate[1].squeeze(0))
        new_hstate, embedding = self.model.apply(params, hstate, (obs, done))
        new_hstate = (
            new_hstate[0].reshape(1, batch_size, -1),
            new_hstate[1].reshape(1, batch_size, -1),
        )
        return embedding, new_hstate


class ActionConditionedRNNEncoder:
    def __init__(self, action_dim, obs_dim, hidden_dim, output_dim):
        self.model = ActionConditionedRNNEncoderNetwork(hidden_dim, output_dim)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

    def init_hstate(self, batch_size=1, aux_info=None):
        hstate = ScannedRNN.initialize_carry(batch_size, self.hidden_dim)
        return hstate.reshape(1, batch_size, self.hidden_dim)

    def init_params(self, rng):
        batch_size = 1
        init_hstate = self.init_hstate(batch_size=batch_size)
        dummy_obs = jnp.zeros((1, batch_size, self.obs_dim))
        dummy_act = jnp.zeros((1, batch_size, self.action_dim))
        dummy_done = jnp.zeros((1, batch_size))
        dummy_x = (jnp.concatenate((dummy_obs, dummy_act), axis=-1), dummy_done)
        return self.model.init(rng, init_hstate.reshape(batch_size, -1), dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_embedding(self, params, hstate, obs, done):
        batch_size = obs.shape[1]
        new_hstate, embedding = self.model.apply(params, hstate.squeeze(0), (obs, done))
        return embedding, new_hstate.reshape(1, batch_size, -1)


class PartnerReconstructionDecoder:
    def __init__(self, action_dim, obs_dim, embedding_dim, hidden_dim, output_dim1, output_dim2):
        self.model = PartnerReconstructionDecoderNetwork(hidden_dim, output_dim1, output_dim2)
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embedding_dim = embedding_dim

    def init_params(self, rng):
        dummy_x = jnp.zeros((1, 1, self.embedding_dim))
        return self.model.init(rng, dummy_x)

    @functools.partial(jax.jit, static_argnums=(0,))
    def compute_losses(self, params, embedding, modelled_agent_obs, modelled_agent_act):
        mean, prob1 = self.model.apply(params, embedding)
        recon_loss_obs = 0.5 * ((modelled_agent_obs - mean) ** 2).sum(-1)
        recon_loss_act = -jnp.log(jnp.sum(prob1 * modelled_agent_act, axis=-1))
        return recon_loss_obs.mean(), recon_loss_act.mean()


def initialize_partner_modeling_auxiliaries(config, env, rng):
    """Initialize the auxiliary encoder/decoder used by partner-modeling ego training."""
    encoder_type = config.get("ENCODER_TYPE", "lstm")
    if encoder_type == "lstm":
        encoder = ActionConditionedLSTMEncoder(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            hidden_dim=config.get("ENCODER_HIDDEN_DIM", 64),
            output_dim=config.get("ENCODER_OUTPUT_DIM", 64),
        )
    elif encoder_type == "rnn":
        encoder = ActionConditionedRNNEncoder(
            action_dim=env.action_space(env.agents[0]).n,
            obs_dim=env.observation_space(env.agents[0]).shape[0],
            hidden_dim=config.get("ENCODER_HIDDEN_DIM", 64),
            output_dim=config.get("ENCODER_OUTPUT_DIM", 64),
        )
    else:
        raise ValueError(f"Unknown ENCODER_TYPE: {encoder_type}")

    decoder = PartnerReconstructionDecoder(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        embedding_dim=config.get("ENCODER_OUTPUT_DIM", 64),
        hidden_dim=config.get("DECODER_HIDDEN_DIM", 64),
        output_dim1=env.observation_space(env.agents[1]).shape[0],
        output_dim2=env.action_space(env.agents[1]).n,
    )

    rng, init_rng_encoder, init_rng_decoder = jax.random.split(rng, 3)
    init_params_encoder = encoder.init_params(init_rng_encoder)
    init_params_decoder = decoder.init_params(init_rng_decoder)
    return encoder, decoder, {"encoder": init_params_encoder, "decoder": init_params_decoder}


class PartnerModelingPolicy(AgentPolicy):
    """Policy wrapper for a base ego policy plus partner-modeling auxiliaries."""

    def __init__(self, policy, encoder, decoder):
        super().__init__(action_dim=policy.action_dim, obs_dim=policy.obs_dim)
        self.policy = policy
        self.encoder = encoder
        self.decoder = decoder

    def init_hstate(self, batch_size=1, aux_info=None):
        encoder_hstate = self.encoder.init_hstate(batch_size=batch_size, aux_info=aux_info)
        policy_hstate = self.policy.init_hstate(batch_size=batch_size, aux_info=aux_info)
        return encoder_hstate, policy_hstate

    def init_params(self, rng):
        rng, init_rng_encoder, init_rng_decoder, init_rng_policy = jax.random.split(rng, 4)
        encoder_params = self.encoder.init_params(init_rng_encoder)
        decoder_params = self.decoder.init_params(init_rng_decoder)
        policy_params = self.policy.init_params(init_rng_policy)
        return {"encoder": encoder_params, "decoder": decoder_params, "policy": policy_params}

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, params, obs, done, avail_actions, hstate, rng,
                   aux_obs=None, env_state=None, greedy=False):
        prev_action, _, _ = aux_obs
        embedding, new_encoder_hstate = self.encoder.compute_embedding(
            params=params["encoder"],
            hstate=hstate[0],
            obs=jnp.concatenate((obs, prev_action), axis=-1),
            done=done,
        )
        action, new_policy_hstate = self.policy.get_action(
            params=params["policy"],
            obs=jnp.concatenate((obs, embedding), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1],
            rng=rng,
            aux_obs=aux_obs,
            env_state=env_state,
            greedy=greedy,
        )
        return action, (new_encoder_hstate, new_policy_hstate)

    @partial(jax.jit, static_argnums=(0,))
    def get_action_value_policy(self, params, obs, done, avail_actions, hstate, rng,
                                aux_obs=None, env_state=None):
        prev_action, _, _ = aux_obs
        embedding, new_encoder_hstate = self.encoder.compute_embedding(
            params=params["encoder"],
            hstate=hstate[0],
            obs=jnp.concatenate((obs, prev_action), axis=-1),
            done=done,
        )
        action, val, pi, new_policy_hstate = self.policy.get_action_value_policy(
            params=params["policy"],
            obs=jnp.concatenate((obs, embedding), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1],
            rng=rng,
            aux_obs=aux_obs,
            env_state=env_state,
        )
        return action, val, pi, (new_encoder_hstate, new_policy_hstate)

    @partial(jax.jit, static_argnums=(0,))
    def compute_decoder_losses(self, params, obs, done, avail_actions, hstate, rng,
                               modelled_agent_obs, modelled_agent_act,
                               aux_obs=None, env_state=None):
        prev_action, _, _ = aux_obs
        embedding, new_encoder_hstate = self.encoder.compute_embedding(
            params=params["encoder"],
            hstate=hstate[0],
            obs=jnp.concatenate((obs, prev_action), axis=-1),
            done=done,
        )
        action, val, pi, new_policy_hstate = self.policy.get_action_value_policy(
            params=params["policy"],
            obs=jnp.concatenate((obs, jax.lax.stop_gradient(embedding)), axis=-1),
            done=done,
            avail_actions=avail_actions,
            hstate=hstate[1],
            rng=rng,
            aux_obs=aux_obs,
            env_state=env_state,
        )
        recon_loss_obs, recon_loss_act = self.decoder.compute_losses(
            params=params["decoder"],
            embedding=embedding,
            modelled_agent_obs=modelled_agent_obs,
            modelled_agent_act=modelled_agent_act,
        )
        return (
            action,
            val,
            pi,
            recon_loss_obs,
            recon_loss_act,
            (new_encoder_hstate, new_policy_hstate),
        )
