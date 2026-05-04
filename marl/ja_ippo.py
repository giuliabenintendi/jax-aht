'''
JA-IPPO: Joint Attention IPPO with shared parameters (Lee et al. 2021).

Both agents share a single network and optimizer (parameter sharing). Cross-agent
coordination comes from:

  1. JA intrinsic reward: r_JA = -JSD(attn_agent_0, attn_agent_1), penalty
     for attention divergence, scaled by beta ramping from 0 to JA_BETA_MAX
     over JA_WARMUP_ENV_STEPS. Combined with env reward before normalization.
'''
import functools
import os
import shutil
from typing import NamedTuple

import hydra
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent, _get_image_dims
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import jsd_divergence, build_card_masks
from common.plot_utils import get_stats, get_metric_names, plot_seed_aggregate
from common.save_load_utils import save_train_run, REPO_PATH
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import Transition, batchify, unbatchify, _create_minibatches


_build_card_masks = build_card_masks  # backward compat alias
from marl.eval_logging import log_greedy_eval, log_eval_video


class JATransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    ja_reward: jnp.ndarray       # (NUM_ACTORS,) -- raw JA intrinsic reward (unscaled)
    partner_embed_actor: jnp.ndarray   # (NUM_ACTORS, positions, feat_dim) or scalar 0 when disabled
    partner_embed_critic: jnp.ndarray  # (NUM_ACTORS, positions, feat_dim) or scalar 0 when disabled
    plh_actor: jnp.ndarray             # (NUM_ACTORS, lstm_dim) or scalar 0 when disabled
    plh_critic: jnp.ndarray            # (NUM_ACTORS, lstm_dim) or scalar 0 when disabled


class RewardNormState(NamedTuple):
    """Running mean/variance for streaming reward normalization (Welford's algorithm)."""
    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def reward_norm_init() -> RewardNormState:
    return RewardNormState(
        mean=jnp.zeros(()),
        var=jnp.ones(()),
        count=jnp.zeros(()),
    )


def reward_norm_update(state: RewardNormState, batch: jnp.ndarray) -> RewardNormState:
    """Update running stats with a new batch of rewards."""
    batch_mean = batch.mean()
    batch_var = batch.var()
    batch_count = jnp.array(batch.size, dtype=jnp.float32)

    delta = batch_mean - state.mean
    total_count = state.count + batch_count
    new_mean = state.mean + delta * batch_count / jnp.maximum(total_count, 1.0)
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta ** 2 * state.count * batch_count / jnp.maximum(total_count, 1.0)
    new_var = m2 / jnp.maximum(total_count, 1.0)

    return RewardNormState(mean=new_mean, var=new_var, count=total_count)


def reward_norm_apply(state: RewardNormState, rewards: jnp.ndarray, clip: float = 10.0) -> jnp.ndarray:
    """Normalize rewards using running stats, clip to [-clip, clip]."""
    std = jnp.sqrt(state.var + 1e-8)
    normalized = (rewards - state.mean) / std
    return jnp.clip(normalized, -clip, clip)


def _get_obs_type(config):
    return config.get("OBS_TYPE", config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))



def make_train_loop(config, env):
    """Build init and step functions for JA-IPPO with Python-loop training.

    Used for image observations where memory requires sequential stepping.
    Returns (init_fn, make_step_fn) for the Python loop in run_ja_ippo.
    """
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )

    num_envs = config["NUM_ENVS"]
    num_actors = config["NUM_ACTORS"]
    ja_beta_max = config.get("JA_BETA_MAX", 0.01)
    ja_warmup_env_steps = config.get("JA_WARMUP_ENV_STEPS", 200_000)
    comm_warmup_env_steps = config.get("COMM_WARMUP_ENV_STEPS", 0)
    comm_reward_start_scale = config.get("COMM_REWARD_START_SCALE", 0.5)
    normalize_rewards = config.get("NORMALIZE_REWARDS", True)
    env_steps_per_update = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
    ja_warmup_updates = ja_warmup_env_steps / env_steps_per_update
    comm_warmup_updates = comm_warmup_env_steps / env_steps_per_update
    feed_other_attn = config.get("FEED_OTHER_ATTN", False)
    cross_agent_attn = config.get("CROSS_AGENT_ATTN", False)
    if cross_agent_attn and feed_other_attn:
        raise ValueError("CROSS_AGENT_ATTN and FEED_OTHER_ATTN are mutually exclusive")
    filter_attn_top1 = config.get("FILTER_ATTN_TOP1", False)
    query_partner_lstm = config.get("QUERY_PARTNER_LSTM", False)
    lstm_hidden_dim = config.get("LSTM_HIDDEN_DIM", 128)
    attn_msg_coef = config.get("ATTN_MSG_REWARD_COEF", 0.0)
    ja_card_attn = config.get("JA_CARD_ATTN", False)
    ja_card_conc_coef = config.get("JA_CARD_CONC_COEF", 0.1)
    ja_card_align_coef = config.get("JA_CARD_ALIGN_COEF", 0.1)
    ja_card_follow_coef = config.get("JA_CARD_FOLLOW_COEF", 0.5)
    ja_card_jsd_coef = config.get("JA_CARD_JSD_COEF", 0.0)
    card_cross_attn = config.get("CARD_CROSS_ATTN", False)
    if ja_card_attn and (feed_other_attn or cross_agent_attn):
        raise ValueError("JA_CARD_ATTN is mutually exclusive with FEED_OTHER_ATTN and CROSS_AGENT_ATTN")
    if card_cross_attn and not ja_card_attn:
        raise ValueError("CARD_CROSS_ATTN requires JA_CARD_ATTN=True")
    if card_cross_attn and (feed_other_attn or cross_agent_attn):
        raise ValueError("CARD_CROSS_ATTN is mutually exclusive with FEED_OTHER_ATTN and CROSS_AGENT_ATTN")
    # CARD_CROSS_ATTN is observation-only: disable JA reward shaping
    if card_cross_attn:
        ja_card_conc_coef = 0.0
        ja_card_align_coef = 0.0
        ja_card_follow_coef = 0.0
    if ja_card_attn:
        env_kwargs = config.get("ENV_KWARGS", {})
        if not (env_kwargs.get("other_play_position_shuffle") and env_kwargs.get("other_play_recolouring")):
            raise ValueError("JA_CARD_ATTN requires both other_play_position_shuffle and other_play_recolouring")

    # Precompute image and feature-map dimensions (only needed for image obs)
    obs_type = _get_obs_type(config)
    if obs_type in ("image", "fov"):
        img_h, img_w, _ = _get_image_dims(env)
        feat_h, feat_w = _compute_resnet_output_dims(
            img_h, img_w,
            stride=config.get("CONV_STRIDE", 2),
            kernel_size=config.get("CONV_KERNEL_SIZE", 3),
            padding=config.get("CONV_PADDING", "SAME"),
            num_blocks=config.get("CONV_NUM_BLOCKS", 4),
        )
    else:
        img_h = img_w = feat_h = feat_w = 0

    xattn_num_positions = feat_h * feat_w
    xattn_feat_dim = config.get("CONV_FILTERS", 32) + config.get("JA_SPATIAL_BASIS_DEPTH", 8)
    _env_max_steps = int(config.get("ENV_KWARGS", {}).get("max_steps", 8))

    # Precompute card tile masks (shared by attn-msg reward and JA card attention)
    _need_card_masks = (attn_msg_coef > 0 or ja_card_attn) and feat_h > 0
    if _need_card_masks:
        from envs.card_game.rendering import (
            TILE_PIXELS as _TP_AM, NUM_CARDS as _NC_AM, CARD_COLORS as _CC_AM,
        )
        _card_masks = _build_card_masks(img_h, img_w, feat_h, feat_w)
        _card_colors_f32 = _CC_AM.astype(jnp.float32) / 255.0  # (5, 3)
        # Safe pixel indices: row 12, cols ci*7+1 through ci*7+5 (outside message dot region)
        import numpy as _np
        _safe_pixel_rows = _np.full((_NC_AM, 5), 12, dtype=_np.int32)
        _safe_pixel_cols = _np.array([[ci * _TP_AM + 1 + k for k in range(5)]
                                      for ci in range(_NC_AM)], dtype=_np.int32)
        # Flat obs index for each (row, col): (row * img_w + col) * 3
        _safe_flat_indices = jnp.array(
            (_safe_pixel_rows * img_w + _safe_pixel_cols) * 3, dtype=jnp.int32)  # (5, 5)
        del _np, _safe_pixel_rows, _safe_pixel_cols
    else:
        _card_masks = _card_colors_f32 = _safe_flat_indices = None

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    agent_init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent

    def init_policy(rng):
        """Create the policy object (called once, not vmapped)."""
        rng, init_rng = jax.random.split(rng)
        policy, _ = agent_init_fn(config, env, init_rng)
        return policy

    def init_state(rng, policy):
        """Initialize runner state for a single seed (vmappable over rng)."""
        rng, init_rng = jax.random.split(rng)
        _, init_params = agent_init_fn(config, env, init_rng)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=policy.network.apply, params=init_params, tx=tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_hstate = policy.init_hstate(num_actors)
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}

        if feed_other_attn:
            init_other_attn = jnp.ones((num_actors, feat_h, feat_w)) / (feat_h * feat_w)
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng, init_other_attn)
        elif cross_agent_attn:
            init_pe_actor = jnp.zeros((num_actors, xattn_num_positions, xattn_feat_dim))
            init_pe_critic = jnp.zeros((num_actors, xattn_num_positions, xattn_feat_dim))
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng,
                            init_pe_actor, init_pe_critic)
        elif ja_card_attn:
            init_partner_card_attn = jnp.zeros((num_actors, 5))
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng,
                            init_partner_card_attn)
        else:
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng)
        if query_partner_lstm:
            init_plh_a = jnp.zeros((num_actors, lstm_hidden_dim))
            init_plh_c = jnp.zeros((num_actors, lstm_hidden_dim))
            runner_state = runner_state + (init_plh_a, init_plh_c)

        return runner_state

    def init(rng):
        """Legacy init: returns (runner_state, policy) for single-seed use."""
        policy = init_policy(rng)
        runner_state = init_state(rng, policy)
        return runner_state, policy

    def make_step_fn(policy):

        def _ppo_update(train_state, traj_batch, advantages, targets, rng):
            policy_loss_type = config.get("POLICY_LOSS_TYPE", "ppo")

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        loss_plh = dict(plh_actor=traj_batch.plh_actor, plh_critic=traj_batch.plh_critic) if query_partner_lstm else {}
                        if cross_agent_attn:
                            _, value, pi, _, _, _, _ = policy.get_action_value_policy(
                                params=params,
                                obs=traj_batch.obs,
                                done=traj_batch.done,
                                avail_actions=traj_batch.avail_actions,
                                hstate=init_hstate,
                                rng=jax.random.PRNGKey(0),
                                partner_embed_actor=traj_batch.partner_embed_actor,
                                partner_embed_critic=traj_batch.partner_embed_critic,
                                **loss_plh,
                            )
                        else:
                            _, value, pi, _, _ = policy.get_action_value_policy(
                                params=params,
                                obs=traj_batch.obs,
                                done=traj_batch.done,
                                avail_actions=traj_batch.avail_actions,
                                hstate=init_hstate,
                                rng=jax.random.PRNGKey(0),
                                **loss_plh,
                            )
                        log_prob = pi.log_prob(traj_batch.action)
                        entropy = pi.entropy().mean()

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        if policy_loss_type == "spo":
                            # SPO (Simple Policy Optimization):
                            # loss = -(ratio*A - |A|*(ratio-1)^2 / (2*eps))
                            # Smooth quadratic penalty around ratio=1 replaces
                            # PPO's clipped min. Optimum at ratio = 1 + eps*sign(A);
                            # beyond that the penalty pulls the ratio back, so
                            # the update can't drift far in a single step.
                            spo_penalty = (
                                jnp.abs(gae) * jnp.square(ratio - 1.0)
                                / (2.0 * config["CLIP_EPS"])
                            )
                            loss_actor = -(ratio * gae - spo_penalty).mean()
                        else:
                            loss_actor1 = ratio * gae
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae
                            )
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        # PPO diagnostics
                        approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
                        clip_frac = (jnp.abs(ratio - 1.0) > config["CLIP_EPS"]).mean()
                        return total_loss, (value_loss, loss_actor, entropy,
                                            approx_kl, clip_frac, ratio.mean(), ratio.std())

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    grad_norm = jnp.sqrt(
                        sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads))
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, grad_norm)

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate,
                                                  num_actors, config["NUM_MINIBATCHES"], perm_rng)

                train_state, minibatch_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, minibatch_info

            init_hstate = policy.init_hstate(num_actors)
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            return update_state[0], loss_info

        def _augment_obs_with_attn(obs_batch, prev_other_attn):
            """Append upsampled other-agent attention as 4th image channel."""
            rgb = obs_batch.reshape(num_actors, img_h, img_w, 3)
            upsampled = jax.image.resize(
                prev_other_attn, (num_actors, img_h, img_w), method='nearest',
            )
            # Normalize to [0, 1] so attention channel matches RGB scale
            attn_max = jnp.max(upsampled, axis=(-2, -1), keepdims=True)
            upsampled = upsampled / jnp.maximum(attn_max, 1e-8)
            augmented = jnp.concatenate([rgb, upsampled[..., None]], axis=-1)
            return augmented.reshape(num_actors, -1)

        def _swap_and_reset_attn(attn_map, done_batch):
            """Swap attention maps between agents, reset to uniform on done."""
            attn = attn_map.squeeze(0)  # (num_actors, feat_h, feat_w)
            attn_0 = attn[:num_envs]
            attn_1 = attn[num_envs:]
            # Each agent gets the other's attention
            swapped = jnp.concatenate([attn_1, attn_0], axis=0)
            uniform = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)
            return jnp.where(done_batch[:, None, None], uniform[None], swapped)

        def _single_step(runner_state, update_steps, rew_norm_state):
            ja_beta = jnp.minimum(
                ja_beta_max,
                ja_beta_max * update_steps / jnp.maximum(ja_warmup_updates, 1.0),
            )
            comm_scale = jnp.where(
                comm_warmup_env_steps > 0,
                comm_reward_start_scale
                + (1.0 - comm_reward_start_scale)
                * jnp.minimum(1.0, update_steps / jnp.maximum(comm_warmup_updates, 1.0)),
                1.0,
            )

            def _env_step(runner_state, unused):
                # query_partner_lstm appends (prev_plh_actor, prev_plh_critic) at the end
                if query_partner_lstm:
                    *rest, prev_plh_actor, prev_plh_critic = runner_state
                    runner_state_core = tuple(rest)
                else:
                    runner_state_core = runner_state
                    prev_plh_actor = prev_plh_critic = None
                if feed_other_attn:
                    (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn) = runner_state_core
                elif cross_agent_attn:
                    (train_state, env_state, last_obs, last_done, hstate, rng,
                     prev_pe_actor, prev_pe_critic) = runner_state_core
                elif ja_card_attn:
                    (train_state, env_state, last_obs, last_done, hstate, rng,
                     prev_partner_card_attn) = runner_state_core
                else:
                    (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state_core

                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, num_actors)
                raw_obs_batch = last_obs_batch  # keep pre-augmentation obs for attn-msg reward
                last_done_batch = batchify(last_done, env.agents, num_actors)

                # Augment obs with other agent's previous attention as 4th channel
                if feed_other_attn:
                    last_obs_batch = _augment_obs_with_attn(last_obs_batch, prev_other_attn)

                # Append partner's translated card attention as scalar suffix
                if ja_card_attn:
                    last_obs_batch = jnp.concatenate(
                        [last_obs_batch, prev_partner_card_attn], axis=-1)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions_batch = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors).astype(jnp.float32))

                plh_kwarg = dict(
                    plh_actor=prev_plh_actor.reshape(1, num_actors, -1),
                    plh_critic=prev_plh_critic.reshape(1, num_actors, -1),
                ) if query_partner_lstm else {}
                if cross_agent_attn:
                    action, value, pi, new_hstate, attn_map, actor_own_embed, critic_own_embed = \
                        policy.get_action_value_policy(
                            params=train_state.params,
                            obs=last_obs_batch.reshape(1, num_actors, -1),
                            done=last_done_batch.reshape(1, num_actors),
                            avail_actions=avail_actions_batch.reshape(1, num_actors, -1),
                            hstate=hstate,
                            rng=act_rng,
                            partner_embed_actor=prev_pe_actor[None],
                            partner_embed_critic=prev_pe_critic[None],
                            **plh_kwarg,
                        )
                else:
                    action, value, pi, new_hstate, attn_map = policy.get_action_value_policy(
                        params=train_state.params,
                        obs=last_obs_batch.reshape(1, num_actors, -1),
                        done=last_done_batch.reshape(1, num_actors),
                        avail_actions=avail_actions_batch.reshape(1, num_actors, -1),
                        hstate=hstate,
                        rng=act_rng,
                        **plh_kwarg,
                    )

                log_prob = pi.log_prob(action)
                action = action.squeeze()
                log_prob = log_prob.squeeze()
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                # Extract per-agent communication reward components before
                # interleaving reshape.
                comm_reward_raw = info.pop("comm_reward", jnp.zeros((num_envs, env.num_agents)))
                comm_reward_batch = comm_reward_raw.transpose(1, 0).reshape(-1)
                comm_match_raw = info.pop("comm_reward_match", jnp.zeros((num_envs, env.num_agents)))
                comm_match_batch = comm_match_raw.transpose(1, 0).reshape(-1)
                comm_stable_raw = info.pop("comm_reward_stable", jnp.zeros((num_envs, env.num_agents)))
                comm_stable_batch = comm_stable_raw.transpose(1, 0).reshape(-1)
                comm_follow_raw = info.pop("comm_reward_follow", jnp.zeros((num_envs, env.num_agents)))
                comm_follow_batch = comm_follow_raw.transpose(1, 0).reshape(-1)

                # Extract step count for attn-msg reward gating
                step_count_raw = info.pop("step_count", jnp.zeros((num_envs, env.num_agents)))
                step_count_batch = step_count_raw.transpose(1, 0).reshape(-1)

                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                # Filter attention to global argmax
                if filter_attn_top1:
                    flat = attn_map.reshape(*attn_map.shape[:2], -1)
                    top_idx = jnp.argmax(flat, axis=-1, keepdims=True)
                    attn_map = jnp.zeros_like(flat).at[jnp.arange(flat.shape[0])[:, None], jnp.arange(flat.shape[1])[None, :], top_idx].set(1.0).reshape(attn_map.shape)

                attn_0 = attn_map[:, :num_envs, ...]
                attn_1 = attn_map[:, num_envs:, ...]
                r_ja = -jsd_divergence(attn_0.squeeze(0), attn_1.squeeze(0))
                r_ja = jax.lax.stop_gradient(r_ja)

                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                r_ja_batch = jnp.concatenate([r_ja, r_ja])

                intrinsic = ja_beta * r_ja_batch

                # Attention-message consistency reward: reward for messaging the
                # card the agent is attending to (within-agent, OP-invariant)
                if attn_msg_coef > 0:
                    num_cards = 5
                    msg_color_idx = jnp.asarray(action, dtype=jnp.int32)
                    # In the unified action layout every action is a card
                    # identity; treat it as a message only on deliberation
                    # steps (i.e. step_count < max_steps means the action
                    # just taken was not the decision-step pick).
                    is_msg_action = step_count_batch < _env_max_steps
                    safe_msg_color_idx = jnp.where(is_msg_action, msg_color_idx, 0)

                    # Read card colors from raw (pre-augmentation) obs at safe pixels.
                    # _safe_flat_indices: (5, 5) — 5 cards x 5 sample pixels, each
                    # is the flat obs index of the R channel. Average across 5 pixels
                    # per card for robustness. → (num_actors, 5, 3)
                    card_rgb_sum = jnp.zeros((num_actors, num_cards, 3))
                    for k in range(5):
                        r_vals = raw_obs_batch[:, _safe_flat_indices[:, k]]
                        g_vals = raw_obs_batch[:, _safe_flat_indices[:, k] + 1]
                        b_vals = raw_obs_batch[:, _safe_flat_indices[:, k] + 2]
                        card_rgb_sum = card_rgb_sum + jnp.stack([r_vals, g_vals, b_vals], axis=-1)
                    card_rgb = card_rgb_sum / 5.0

                    # Find which card position has the messaged color
                    msg_color_rgb = _card_colors_f32[safe_msg_color_idx]  # (num_actors, 3)
                    color_dist = jnp.sum(jnp.abs(card_rgb - msg_color_rgb[:, None, :]), axis=-1)
                    msg_card_pos = jnp.argmin(color_dist, axis=-1)  # (num_actors,)

                    # Attention mass on the messaged card (soft overlap mask)
                    actor_card_mask = _card_masks[msg_card_pos]  # (num_actors, feat_h, feat_w)
                    attn = attn_map.squeeze(0)  # (num_actors, feat_h, feat_w)
                    attn_mass = jnp.sum(attn * actor_card_mask, axis=(-2, -1))

                    attn_msg_valid = is_msg_action & (step_count_batch > 2)
                    attn_msg_reward = jnp.where(attn_msg_valid, attn_mass, 0.0)
                    attn_msg_reward = jax.lax.stop_gradient(attn_msg_reward)
                else:
                    attn_msg_reward = jnp.zeros(num_actors)

                # OP-corrected card-level joint attention rewards
                if ja_card_attn:
                    attn = attn_map.squeeze(0)  # (num_actors, feat_h, feat_w)
                    card_pos_attn = jnp.einsum("ahw,chw->ac", attn, _card_masks)  # (num_actors, 5)
                    card_pos_attn_0 = card_pos_attn[:num_envs]   # (num_envs, 5)
                    card_pos_attn_1 = card_pos_attn[num_envs:]   # (num_envs, 5)

                    # Concentration reward: m * (log5 - H(q)) / log5
                    _eps = 1e-8
                    _log5 = jnp.log(5.0)
                    m = card_pos_attn.sum(axis=-1)  # (num_actors,)
                    q = card_pos_attn / (m[:, None] + _eps)
                    H = -jnp.sum(q * jnp.log(q + _eps), axis=-1)
                    r_conc = ja_card_conc_coef * m * (_log5 - H) / _log5  # (num_actors,)

                    # Map to physical card space via OP permutations
                    perm_0 = env_state.env_state.env_state.per_agent_perm["agent_0"]  # (num_envs, 5)
                    perm_1 = env_state.env_state.env_state.per_agent_perm["agent_1"]  # (num_envs, 5)
                    batch_idx = jnp.arange(num_envs)[:, None]
                    phys_0 = jnp.zeros((num_envs, 5)).at[batch_idx, perm_0].set(card_pos_attn_0)
                    phys_1 = jnp.zeros((num_envs, 5)).at[batch_idx, perm_1].set(card_pos_attn_1)

                    # Alignment reward: dot product in physical card space
                    alignment = jnp.sum(phys_0 * phys_1, axis=-1)  # (num_envs,)
                    r_align_env = ja_card_align_coef * alignment
                    r_align_valid = step_count_batch[:num_envs] > 1
                    r_align_env = jnp.where(r_align_valid, r_align_env, 0.0)
                    r_align = jnp.concatenate([r_align_env, r_align_env])  # (num_actors,)

                    # Mass-gated card attention divergence penalty (L2 on normalized card distributions)
                    if ja_card_jsd_coef > 0:
                        m_0 = card_pos_attn_0.sum(axis=-1)  # (num_envs,)
                        m_1 = card_pos_attn_1.sum(axis=-1)
                        q_phys_0 = phys_0 / (m_0[:, None] + _eps)
                        q_phys_1 = phys_1 / (m_1[:, None] + _eps)
                        l2_div = jnp.sum((q_phys_0 - q_phys_1) ** 2, axis=-1)
                        r_card_jsd_env = -ja_card_jsd_coef * jnp.minimum(m_0, m_1) * l2_div
                        r_card_jsd_valid = step_count_batch[:num_envs] > 1
                        r_card_jsd_env = jnp.where(r_card_jsd_valid, r_card_jsd_env, 0.0)
                        r_card_jsd = jnp.concatenate([r_card_jsd_env, r_card_jsd_env])
                    else:
                        r_card_jsd = jnp.zeros(num_actors)

                    # Follow-through bonus: match * mean attention on picked card
                    pick_0 = jnp.clip(action[:num_envs], 0, 4)
                    pick_1 = jnp.clip(action[num_envs:], 0, 4)
                    mass_on_pick_0 = jnp.take_along_axis(
                        card_pos_attn_0, pick_0[:, None], axis=1).squeeze(-1)
                    mass_on_pick_1 = jnp.take_along_axis(
                        card_pos_attn_1, pick_1[:, None], axis=1).squeeze(-1)
                    match = reward_batch[:num_envs] > 0  # env reward > 0 means match
                    r_follow_env = ja_card_follow_coef * match * 0.5 * (mass_on_pick_0 + mass_on_pick_1)
                    r_follow = jnp.concatenate([r_follow_env, r_follow_env])

                    ja_card_reward = jax.lax.stop_gradient(r_conc + r_align + r_follow + r_card_jsd)
                    dbg_card_attn_0 = jax.lax.stop_gradient(card_pos_attn_0)
                    dbg_card_attn_1 = jax.lax.stop_gradient(card_pos_attn_1)

                    # Translate partner attention to receiver's frame for next step obs
                    translated_for_0 = jnp.take_along_axis(phys_1, perm_0, axis=1)
                    translated_for_1 = jnp.take_along_axis(phys_0, perm_1, axis=1)
                    dbg_partner_card_attn_0 = jax.lax.stop_gradient(translated_for_0)
                    dbg_partner_card_attn_1 = jax.lax.stop_gradient(translated_for_1)
                    new_partner_card_attn = jnp.concatenate(
                        [translated_for_0, translated_for_1], axis=0)  # (num_actors, 5)
                    # Reset on episode boundaries
                    new_done_batch_ja = batchify(new_done, env.agents, num_actors).squeeze()
                    new_partner_card_attn = jnp.where(
                        new_done_batch_ja[:, None], 0.0, new_partner_card_attn)
                else:
                    ja_card_reward = jnp.zeros(num_actors)
                    dbg_card_attn_0 = jnp.zeros((num_envs, 5))
                    dbg_card_attn_1 = jnp.zeros((num_envs, 5))
                    dbg_partner_card_attn_0 = jnp.zeros((num_envs, 5))
                    dbg_partner_card_attn_1 = jnp.zeros((num_envs, 5))

                if cross_agent_attn:
                    pe_actor_stored = prev_pe_actor
                    pe_critic_stored = prev_pe_critic
                else:
                    pe_actor_stored = jnp.zeros((num_actors,))
                    pe_critic_stored = jnp.zeros((num_actors,))

                plh_a_stored = prev_plh_actor if query_partner_lstm else jnp.zeros((num_actors,))
                plh_c_stored = prev_plh_critic if query_partner_lstm else jnp.zeros((num_actors,))

                transition = JATransition(
                    done=batchify(new_done, env.agents, num_actors).squeeze(),
                    action=action,
                    value=value,
                    reward=reward_batch,
                    log_prob=log_prob,
                    obs=last_obs_batch,
                    info=info,
                    avail_actions=avail_actions_batch,
                    ja_reward=r_ja_batch,
                    partner_embed_actor=pe_actor_stored,
                    partner_embed_critic=pe_critic_stored,
                    plh_actor=plh_a_stored,
                    plh_critic=plh_c_stored,
                )

                if feed_other_attn:
                    new_done_batch = batchify(new_done, env.agents, num_actors).squeeze()
                    new_other_attn = _swap_and_reset_attn(attn_map, new_done_batch)
                    runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng, new_other_attn)
                elif cross_agent_attn:
                    # Swap halves: each agent gets the other's embedding
                    a_own = actor_own_embed.squeeze(0)
                    c_own = critic_own_embed.squeeze(0)
                    new_pe_actor = jnp.concatenate([a_own[num_envs:], a_own[:num_envs]], axis=0)
                    new_pe_critic = jnp.concatenate([c_own[num_envs:], c_own[:num_envs]], axis=0)
                    # Reset to zeros on episode boundaries
                    new_done_batch = batchify(new_done, env.agents, num_actors).squeeze()
                    pe_done_mask = new_done_batch.reshape(-1, *([1] * (new_pe_actor.ndim - 1)))
                    new_pe_actor = jnp.where(pe_done_mask, 0.0, new_pe_actor)
                    new_pe_critic = jnp.where(pe_done_mask, 0.0, new_pe_critic)
                    runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng,
                                    new_pe_actor, new_pe_critic)
                elif ja_card_attn:
                    runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng,
                                    new_partner_card_attn)
                else:
                    runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                if query_partner_lstm:
                    # Extract actor h and critic h from packed hstate, swap halves
                    d = lstm_hidden_dim
                    actor_h = new_hstate[:, :, :d].squeeze(0)
                    critic_h = new_hstate[:, :, 2*d:3*d].squeeze(0)
                    new_plh_a = jnp.concatenate([actor_h[num_envs:], actor_h[:num_envs]], axis=0)
                    new_plh_c = jnp.concatenate([critic_h[num_envs:], critic_h[:num_envs]], axis=0)
                    new_done_batch = batchify(new_done, env.agents, num_actors).squeeze()
                    new_plh_a = jnp.where(new_done_batch[:, None], 0.0, new_plh_a)
                    new_plh_c = jnp.where(new_done_batch[:, None], 0.0, new_plh_c)
                    runner_state = runner_state + (new_plh_a, new_plh_c)
                return runner_state, (transition, intrinsic, comm_reward_batch,
                                     comm_match_batch, comm_stable_batch, comm_follow_batch,
                                     attn_msg_reward,
                                     ja_card_reward,
                                     dbg_card_attn_0, dbg_card_attn_1,
                                     dbg_partner_card_attn_0, dbg_partner_card_attn_1)

            runner_state, (traj_batch, intrinsic_batch, comm_reward_batch,
                           comm_match_batch, comm_stable_batch, comm_follow_batch,
                           attn_msg_reward_batch,
                           ja_card_reward_batch,
                           dbg_card_attn_0_batch, dbg_card_attn_1_batch,
                           dbg_partner_card_attn_0_batch, dbg_partner_card_attn_1_batch) = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            if query_partner_lstm:
                *rest, prev_plh_actor, prev_plh_critic = runner_state
                runner_state = tuple(rest)
            if feed_other_attn:
                (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn) = runner_state
            elif cross_agent_attn:
                (train_state, env_state, last_obs, last_done, hstate, rng,
                 prev_pe_actor, prev_pe_critic) = runner_state
            elif ja_card_attn:
                (train_state, env_state, last_obs, last_done, hstate, rng,
                 prev_partner_card_attn) = runner_state
            else:
                (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state

            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            if feed_other_attn:
                last_obs_batch = _augment_obs_with_attn(last_obs_batch, prev_other_attn)
            if ja_card_attn:
                last_obs_batch = jnp.concatenate(
                    [last_obs_batch, prev_partner_card_attn], axis=-1)
            last_avail = jax.vmap(env.get_avail_actions)(env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32))

            last_plh_kwarg = dict(
                plh_actor=prev_plh_actor.reshape(1, num_actors, -1),
                plh_critic=prev_plh_critic.reshape(1, num_actors, -1),
            ) if query_partner_lstm else {}
            if cross_agent_attn:
                _, last_val, _, _, _, _, _ = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=last_obs_batch.reshape(1, num_actors, -1),
                    done=last_done_batch.reshape(1, num_actors),
                    avail_actions=last_avail_batch.reshape(1, num_actors, -1),
                    hstate=hstate,
                    rng=jax.random.PRNGKey(0),
                    partner_embed_actor=prev_pe_actor[None],
                    partner_embed_critic=prev_pe_critic[None],
                    **last_plh_kwarg,
                )
            else:
                _, last_val, _, _, _ = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=last_obs_batch.reshape(1, num_actors, -1),
                    done=last_done_batch.reshape(1, num_actors),
                    avail_actions=last_avail_batch.reshape(1, num_actors, -1),
                    hstate=hstate,
                    rng=jax.random.PRNGKey(0),
                    **last_plh_kwarg,
                )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            # Save raw env reward before combining
            raw_env_reward = traj_batch.reward
            scaled_comm_reward = comm_scale * comm_reward_batch
            scaled_comm_match = comm_scale * comm_match_batch
            scaled_comm_stable = comm_scale * comm_stable_batch
            scaled_comm_follow = comm_scale * comm_follow_batch
            scaled_attn_msg_reward = attn_msg_coef * attn_msg_reward_batch

            combined_raw = raw_env_reward + intrinsic_batch + scaled_comm_reward + scaled_attn_msg_reward + ja_card_reward_batch
            if normalize_rewards:
                rew_norm_state = reward_norm_update(rew_norm_state, combined_raw)
                combined = reward_norm_apply(rew_norm_state, combined_raw)
                traj_batch = traj_batch._replace(reward=combined)
            else:
                traj_batch = traj_batch._replace(reward=combined_raw)

            advantages, targets = _calculate_gae(traj_batch, last_val)
            adv_std = jnp.std(advantages)

            rng, ppo_rng = jax.random.split(rng)
            train_state, loss_info = _ppo_update(
                train_state, traj_batch, advantages, targets, ppo_rng)

            (total_loss, (value_loss, policy_loss, entropy,
                         approx_kl, clip_frac, ratio_mean, ratio_std)), grad_norm = loss_info

            jsd_values = -traj_batch.ja_reward[:, :num_envs]

            metric = traj_batch.info
            metric["update_steps"] = update_steps
            metric["ja_beta"] = ja_beta
            metric["comm_scale"] = comm_scale
            metric["jsd_mean"] = jsd_values.mean()
            metric["loss_total"] = total_loss[0].mean()
            metric["loss_value"] = value_loss[0].mean()
            metric["loss_policy"] = policy_loss[0].mean()
            metric["entropy"] = entropy.mean()
            metric["grad_norm"] = grad_norm.mean()
            metric["approx_kl"] = approx_kl.mean()
            metric["approx_kl_all"] = approx_kl.mean()
            metric["approx_kl_max"] = approx_kl.max()
            metric["clip_frac"] = clip_frac.mean()
            metric["ratio_mean"] = ratio_mean.mean()
            metric["ratio_std"] = ratio_std.mean()
            # Explained variance: how well value function predicts returns
            ev_var_ret = jnp.var(targets)
            ev_var_resid = jnp.var(targets - traj_batch.value)
            metric["explained_var"] = 1.0 - ev_var_resid / (ev_var_ret + 1e-8)
            metric["advantage_std"] = adv_std
            metric["raw_env_reward_mean"] = raw_env_reward[:, :num_envs].mean()
            metric["comm_reward_mean"] = scaled_comm_reward[:, :num_envs].mean()
            metric["comm_match_bonus_mean"] = scaled_comm_match[:, :num_envs].mean()
            metric["comm_stability_bonus_mean"] = scaled_comm_stable[:, :num_envs].mean()
            # follow is per-agent now; log both agents' means separately
            metric["comm_follow_bonus_agent0_mean"] = scaled_comm_follow[:, :num_envs].mean()
            metric["comm_follow_bonus_agent1_mean"] = scaled_comm_follow[:, num_envs:].mean()
            metric["combined_reward_mean"] = combined_raw[:, :num_envs].mean()
            metric["value_mean"] = traj_batch.value.mean()
            if card_cross_attn:
                # Log the learned gate values to track mechanism usage
                actor_params = train_state.params["params"]["actor_lstm"]
                card_xattn_params = actor_params["card_cross_attn"]
                metric["card_xattn_attn_gate"] = jnp.tanh(card_xattn_params["card_attn_gate"]).squeeze()
                metric["card_xattn_ff_gate"] = jnp.tanh(card_xattn_params["card_ff_gate"]).squeeze()
            for card_idx in range(5):
                metric[f"debug_card_attn_agent0_card{card_idx}"] = dbg_card_attn_0_batch[:, :, card_idx].mean()
                metric[f"debug_card_attn_agent1_card{card_idx}"] = dbg_card_attn_1_batch[:, :, card_idx].mean()
                metric[f"debug_partner_card_attn_for_agent0_card{card_idx}"] = dbg_partner_card_attn_0_batch[:, :, card_idx].mean()
                metric[f"debug_partner_card_attn_for_agent1_card{card_idx}"] = dbg_partner_card_attn_1_batch[:, :, card_idx].mean()

            if feed_other_attn:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn)
            elif cross_agent_attn:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng,
                                prev_pe_actor, prev_pe_critic)
            elif ja_card_attn:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng,
                                prev_partner_card_attn)
            else:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            if query_partner_lstm:
                runner_state = runner_state + (prev_plh_actor, prev_plh_critic)
            return runner_state, update_steps + 1, rew_norm_state, metric

        @functools.partial(jax.jit, donate_argnums=(0, 2))
        def step_fn(runner_state, update_steps, rew_norm_state):
            """Single-step wrapper (fallback, same as before)."""
            return _single_step(runner_state, update_steps, rew_norm_state)

        @functools.partial(jax.jit, static_argnums=(3,), donate_argnums=(0, 2))
        def chunked_step_fn(runner_state, update_steps, rew_norm_state, chunk_size):
            """Run chunk_size updates in a single JIT call via lax.scan.

            chunk_size is static -- JAX recompiles per distinct value, but there
            are typically only 2-3 distinct sizes (regular + remainder at ckpts).
            """
            def _scan_body(carry, _):
                rs, us, rns = carry
                rs, us, rns, metric = _single_step(rs, us, rns)
                return (rs, us, rns), metric

            (runner_state, update_steps, rew_norm_state), metrics = jax.lax.scan(
                _scan_body,
                (runner_state, update_steps, rew_norm_state),
                None,
                length=chunk_size,
            )
            return runner_state, update_steps, rew_norm_state, metrics

        return step_fn, chunked_step_fn, _single_step

    return init, make_step_fn, init_policy, init_state


def _push_chunk_to_wandb(chunk_metrics, env_step, seed_idx, logger):
    """Push aggregated chunk metrics to wandb live, once per training chunk.

    The training loop is sequential per seed, so live curves grow seed-by-seed under
    `LiveTrain/seed_<i>/...`. Step axis is env_step; chunk-mean for scalars; mask-weighted
    mean for episodic returns (skips chunks with zero completed episodes).
    """
    if logger is None or getattr(logger, "run", None) is None:
        return

    m = chunk_metrics
    returned = np.asarray(m["returned_episode"])
    returns = np.asarray(m["returned_episode_returns"])
    n_ep = float(returned.sum())
    if n_ep > 0:
        mean = float((returns * returned).sum() / n_ep)
        sq = float(((returns - mean) ** 2 * returned).sum() / n_ep)
        std = float(np.sqrt(max(sq, 0.0)))
    else:
        mean = float("nan")
        std = float("nan")

    data: dict[str, float | int] = {
        f"LiveTrain/seed_{seed_idx}/return_mean": mean,
        f"LiveTrain/seed_{seed_idx}/return_std": std,
        f"LiveTrain/seed_{seed_idx}/n_episodes": int(n_ep),
        "env_step": int(env_step),
    }

    scalar_keys = [
        ("ja_beta",              "JA/beta"),
        ("comm_scale",           "Comm/scale"),
        ("jsd_mean",             "JA/jsd"),
        ("loss_total",           "Loss/total"),
        ("loss_value",           "Loss/value"),
        ("loss_policy",          "Loss/policy"),
        ("entropy",              "Loss/entropy"),
        ("grad_norm",            "Loss/grad_norm"),
        ("approx_kl",            "Loss/approx_kl"),
        ("clip_frac",            "Loss/clip_frac"),
        ("explained_var",        "Loss/explained_var"),
        ("raw_env_reward_mean",  "Reward/env_raw"),
        ("comm_reward_mean",     "Reward/comm"),
    ]
    for key, name in scalar_keys:
        if key in m:
            data[f"LiveTrain/seed_{seed_idx}/{name}"] = float(np.asarray(m[key]).mean())

    logger.log(data, commit=True)


def _select_best_per_seed_ckpt(out, chunk_boundaries):
    """Score each saved checkpoint by mean episodic return over the chunk that produced it.

    Returns (best_params, best_idx, per_ckpt_chunk_return).
    `best_params` has the same tree structure as `out["final_params"]` (leading dim = num_seeds).
    `best_idx` is shape (num_seeds,). `per_ckpt_chunk_return` is shape (num_seeds, num_ckpts).

    Picking based on the chunk that *produced* a checkpoint approximates eval-time return
    cheaply (no extra rollouts). The argmax is per seed: each seed contributes its own peak.
    """
    metrics = out["metrics"]
    stacked_ckpts = out["checkpoints"]
    num_seeds, num_ckpts = jax.tree.leaves(stacked_ckpts)[0].shape[:2]

    returned = np.asarray(metrics["returned_episode"])         # (S, U, ...)
    returns = np.asarray(metrics["returned_episode_returns"])  # (S, U, ...)

    # Use only chunks whose params were actually saved (guards the +1 boundary edge case).
    n_chunks = min(num_ckpts, len(chunk_boundaries))
    los = [0] + list(chunk_boundaries[:n_chunks - 1])
    his = list(chunk_boundaries[:n_chunks])

    per_ckpt_returns = np.zeros((num_seeds, num_ckpts), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(los, his)):
        m = returned[:, lo:hi]
        v = returns[:, lo:hi]
        reduce_axes = tuple(range(1, m.ndim))
        denom = np.maximum(m.sum(axis=reduce_axes), 1)
        numer = (v * m).sum(axis=reduce_axes)
        per_ckpt_returns[:, i] = numer / denom

    best_idx = per_ckpt_returns.argmax(axis=1).astype(np.int32)  # (num_seeds,)
    seed_arange = np.arange(num_seeds)
    best_params = jax.tree.map(lambda c: c[seed_arange, best_idx], stacked_ckpts)
    return best_params, best_idx, per_ckpt_returns


def run_ja_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    # Propagate COMMUNICATION flag into ENV_KWARGS so the env is created with it
    if algorithm_config.get("COMMUNICATION", False):
        env_kwargs = dict(algorithm_config["ENV_KWARGS"])
        env_kwargs["communication"] = True
        algorithm_config["ENV_KWARGS"] = env_kwargs
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    num_seeds = algorithm_config["NUM_SEEDS"]
    num_updates = int(algorithm_config["TOTAL_TIMESTEPS"] // algorithm_config["ROLLOUT_LENGTH"] // algorithm_config["NUM_ENVS"])

    obs_type = _get_obs_type(algorithm_config)

    print(f"[ja_ippo] NUM_UPDATES={num_updates}, NUM_SEEDS={num_seeds}, "
          f"NUM_ENVS={algorithm_config['NUM_ENVS']}, obs_type={obs_type}")

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, num_seeds)

    init_fn, make_step_fn, init_policy_fn, init_state_fn = make_train_loop(algorithm_config, env)

    env_steps_per_update = int(algorithm_config["ROLLOUT_LENGTH"]) * int(algorithm_config["NUM_ENVS"])
    freq_timesteps = float(algorithm_config.get("CHECKPOINT_FREQ_TIMESTEPS", 0) or 0)
    if freq_timesteps > 0:
        # Frequency-based: derive num_ckpts from desired env-step interval. Each chunk
        # spans freq_updates updates; final chunk may be shorter to land exactly on num_updates.
        freq_updates = max(1, int(round(freq_timesteps / env_steps_per_update)))
        chunk_boundaries: list[int] = []
        b = 0
        while b < num_updates:
            b = min(b + freq_updates, num_updates)
            chunk_boundaries.append(b)
        num_ckpts = len(chunk_boundaries)
        print(f"[ja_ippo] Checkpoint cadence: every {freq_timesteps:.0f} env steps "
              f"({freq_updates} updates) -> {num_ckpts} checkpoints")
    else:
        num_ckpts = algorithm_config.get("NUM_CHECKPOINTS", 5)
        ckpt_interval = num_updates // max(1, num_ckpts - 1)
        chunk_boundaries = [min((i + 1) * ckpt_interval, num_updates) for i in range(num_ckpts)]
        if chunk_boundaries[-1] < num_updates:
            chunk_boundaries.append(num_updates)
        print(f"[ja_ippo] Checkpoint cadence: NUM_CHECKPOINTS={num_ckpts} evenly spaced")

    # Compile once for a single seed, then loop over seeds sequentially.
    # This reuses the same compiled step_fn for every seed -- no vmap, no
    # per-seed-count recompilation, constant memory regardless of NUM_SEEDS.
    print(f"[ja_ippo] Initializing policy and {num_seeds} seeds...")
    policy = init_policy_fn(rngs[0])
    step_fn, chunked_step_fn, _ = make_step_fn(policy)

    live_wandb = bool(algorithm_config.get("LIVE_WANDB_LOGGING", True))

    all_seed_metrics = []
    all_seed_ckpts = []
    all_seed_final_params = []

    for seed_idx in range(num_seeds):
        runner_state = init_state_fn(rngs[seed_idx], policy)
        update_steps = jnp.zeros((), dtype=jnp.int32)
        rew_norm_state = reward_norm_init()

        seed_metrics = []
        seed_ckpts = []
        steps_done = 0

        print(f"[ja_ippo] Seed {seed_idx}/{num_seeds}: training {num_updates} steps...")
        for chunk_end in chunk_boundaries:
            chunk_size = chunk_end - steps_done
            if chunk_size <= 0:
                continue

            runner_state, update_steps, rew_norm_state, chunk_metrics = chunked_step_fn(
                runner_state, update_steps, rew_norm_state, chunk_size)
            seed_metrics.append(chunk_metrics)
            steps_done = chunk_end

            # Checkpoint after each chunk
            if len(seed_ckpts) < num_ckpts:
                seed_ckpts.append(jax.tree.map(jnp.copy, runner_state[0].params))

            print(f"[ja_ippo]   step {steps_done}/{num_updates}")

            if live_wandb:
                _push_chunk_to_wandb(
                    chunk_metrics,
                    env_step=steps_done * env_steps_per_update,
                    seed_idx=seed_idx,
                    logger=logger,
                )

        all_seed_final_params.append(runner_state[0].params)
        # Concatenate chunk metrics along the update axis (axis 0)
        all_seed_metrics.append(jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *seed_metrics))
        all_seed_ckpts.append(jax.tree.map(lambda *xs: jnp.stack(xs), *seed_ckpts))

    # Stack across seeds: (num_seeds, ...)
    stacked_params = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_final_params)
    stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_metrics)
    stacked_ckpts = jax.tree.map(lambda *xs: jnp.stack(xs), *all_seed_ckpts)

    print("[ja_ippo] Training complete.")
    out = {
        "final_params": stacked_params,
        "metrics": stacked_metrics,
        "checkpoints": stacked_ckpts,
        "final_ckpt_idx": num_ckpts,
    }

    use_best = bool(algorithm_config.get("USE_BEST_CKPT_FOR_EVAL", True))
    best_params, best_idx, per_ckpt_returns = _select_best_per_seed_ckpt(out, chunk_boundaries)
    ckpt_env_steps = [int(b) * env_steps_per_update for b in chunk_boundaries[:num_ckpts]]
    out["best_params"] = best_params
    out["best_ckpt_idx"] = best_idx
    out["per_ckpt_chunk_return"] = per_ckpt_returns
    out["ckpt_env_steps"] = np.asarray(ckpt_env_steps, dtype=np.int64)

    best_env_steps = [ckpt_env_steps[int(i)] for i in best_idx]
    print(f"[ja_ippo] best_ckpt_idx per seed: {best_idx.tolist()} (of {num_ckpts}); "
          f"best env_step per seed: {best_env_steps}; "
          f"chunk-return at best: "
          f"{[round(float(per_ckpt_returns[s, best_idx[s]]), 3) for s in range(num_seeds)]}")

    savedir_for_scores = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Per-checkpoint folder: lives OUTSIDE the hydra output dir for easy enumeration across runs.
    # Resolves to {CHECKPOINT_ROOT}/{run_name}/ where CHECKPOINT_ROOT is repo-root-relative unless absolute.
    # Structure: <ckpt_root>/ckpt_{i:02d}_step_{env_steps}/  (params, shape (num_seeds, ...))
    #            <ckpt_root>/chunk_scores.json
    # `best`/`final` aliases are not written: best params flow into XP/greedy/video eval via eval_out;
    # final params and the full stacked checkpoints are still in saved_train_run.
    ckpt_root_setting = str(algorithm_config.get("CHECKPOINT_ROOT", "checkpoints"))
    if not os.path.isabs(ckpt_root_setting):
        ckpt_root_setting = os.path.join(REPO_PATH, ckpt_root_setting)
    run_name = None
    if logger is not None and getattr(logger, "run", None) is not None:
        run_name = getattr(logger.run, "name", None)
    if not run_name:
        run_name = os.path.basename(savedir_for_scores.rstrip("/")) or "unnamed_run"
    ckpt_root = os.path.join(ckpt_root_setting, run_name)

    ckpt_folder_paths: list[str] = []
    if bool(algorithm_config.get("SAVE_CHECKPOINT_FOLDER", True)):
        os.makedirs(ckpt_root, exist_ok=True)
        for i in range(num_ckpts):
            params_i = jax.tree.map(lambda c, _i=i: c[:, _i], stacked_ckpts)  # (num_seeds, ...)
            ret_mean = float(per_ckpt_returns[:, i].mean())
            ckpt_name = f"ckpt_{i:02d}_step_{ckpt_env_steps[i]}_ret_{ret_mean:.2f}"
            save_train_run(params_i, ckpt_root, ckpt_name)
            ckpt_folder_paths.append(os.path.join(ckpt_root, ckpt_name))
        print(f"[ja_ippo] Checkpoint folder: {ckpt_root} ({num_ckpts} ckpts)")

    # Sidecar JSON: lives next to the per-ckpt folders so everything checkpoint-related is in one place.
    scores_dir = ckpt_root if bool(algorithm_config.get("SAVE_CHECKPOINT_FOLDER", True)) else savedir_for_scores
    os.makedirs(scores_dir, exist_ok=True)
    scores_path = os.path.join(scores_dir, "chunk_scores.json")
    import json as _json
    with open(scores_path, "w") as _fh:
        _json.dump({
            "run_name": run_name,
            "num_seeds": int(num_seeds),
            "num_ckpts": int(num_ckpts),
            "ckpt_root": ckpt_root,
            "ckpt_env_steps": ckpt_env_steps,
            "ckpt_update_boundaries": [int(b) for b in chunk_boundaries[:num_ckpts]],
            "ckpt_folder_paths": ckpt_folder_paths,
            "per_seed_per_ckpt_return": per_ckpt_returns.tolist(),
            "best_ckpt_idx_per_seed": best_idx.tolist(),
            "best_env_step_per_seed": best_env_steps,
            "best_chunk_return_per_seed": [float(per_ckpt_returns[s, best_idx[s]]) for s in range(num_seeds)],
        }, _fh, indent=2)
    print(f"[ja_ippo] Wrote per-checkpoint scores: {scores_path}")

    log_metrics(config, out, logger)

    eval_out = {**out, "final_params": best_params} if use_best else out
    log_greedy_eval(algorithm_config, env, eval_out, logger)
    log_eval_video(algorithm_config, env, eval_out, logger)

    if num_seeds > 1:
        log_xp_eval(algorithm_config, env, eval_out)

    return out


def log_xp_eval(algorithm_config, env, out):
    """Run greedy cross-play evaluation when NUM_SEEDS > 1."""
    import wandb
    from evaluation.run_xp_seeds import run_xp_from_params

    obs_type = _get_obs_type(algorithm_config)
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algorithm_config, env, rng)

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Greedy XP
    print("[xp_eval] Running greedy XP...")
    run_xp_from_params(
        env, policy, out["final_params"], algorithm_config,
        savedir=savedir,
        task_name=algorithm_config.get("ENV_NAME"),
        wb_run=wandb.run,
        greedy_eval=True,
        wb_prefix="XP",
    )


def log_metrics(config, out, logger):
    '''Save train run output, export CSV, and log mean+/-std to wandb.'''
    import csv

    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    algorithm_config = dict(config.algorithm)
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    rollout_length = int(algorithm_config["ROLLOUT_LENGTH"])
    num_envs = int(algorithm_config["NUM_ENVS"])

    # Save mean+/-std training curve PNGs
    plot_seed_aggregate(
        train_stats,
        num_rollout_steps=rollout_length,
        num_envs=num_envs,
        savedir=savedir,
        savename="train_curve",
    )
    import wandb as _wandb
    for name in train_stats:
        png_path = os.path.join(savedir, f"train_curve_{name}.png")
        if os.path.exists(png_path):
            logger.log_item(f"Plots/train_curve_{name}", _wandb.Image(png_path), commit=False)

    num_seeds = train_metrics["returned_episode"].shape[0]
    num_updates = train_metrics["returned_episode"].shape[1]

    # Compute cross-seed mean and std from per-seed means
    # train_stats[k] shape: (num_seeds, num_updates, 2) where [:,:,0] = per-seed mean
    episode_stats_mean = {}
    for k, v in train_stats.items():
        v_arr = np.array(v)
        seed_means = v_arr[:, :, 0]
        episode_stats_mean[k] = np.stack([
            seed_means.mean(axis=0),
            seed_means.std(axis=0),
        ], axis=-1)

    # Scalar metrics: (metric_key, wandb_name)
    scalar_keys = [
        ("ja_beta",              "JA/beta"),
        ("comm_scale",           "Comm/scale"),
        ("jsd_mean",             "JA/jsd"),
        ("raw_env_reward_mean",  "Reward/env_raw"),
        ("combined_reward_mean", "Reward/combined_raw"),
        ("comm_reward_mean",                     "Reward/comm"),
        ("comm_match_bonus_mean",               "Reward/comm_match_bonus"),
        ("comm_stability_bonus_mean",           "Reward/comm_stability_bonus"),
        ("comm_follow_bonus_agent0_mean",       "Reward/comm_follow_bonus_agent0"),
        ("comm_follow_bonus_agent1_mean",       "Reward/comm_follow_bonus_agent1"),
        ("loss_total",           "Loss/total"),
        ("loss_value",           "Loss/value"),
        ("loss_policy",          "Loss/policy"),
        ("entropy",              "Loss/entropy"),
        ("grad_norm",            "Loss/grad_norm"),
        ("approx_kl",           "Loss/approx_kl"),
        ("approx_kl_all",       "Loss/approx_kl_all"),
        ("approx_kl_max",       "Loss/approx_kl_max"),
        ("clip_frac",           "Loss/clip_frac"),
        ("ratio_mean",          "Loss/ratio_mean"),
        ("ratio_std",           "Loss/ratio_std"),
        ("explained_var",       "Loss/explained_var"),
        ("advantage_std",       "Loss/advantage_std"),
        ("value_mean",           "Value/mean"),
    ]
    scalar_keys.extend([
        ("card_xattn_attn_gate", "Debug/card_xattn_attn_gate"),
        ("card_xattn_ff_gate", "Debug/card_xattn_ff_gate"),
    ])
    scalar_keys.extend(
        [(f"debug_card_attn_agent0_card{i}", f"Debug/card_attn_agent0_card{i}") for i in range(5)]
    )
    scalar_keys.extend(
        [(f"debug_card_attn_agent1_card{i}", f"Debug/card_attn_agent1_card{i}") for i in range(5)]
    )
    scalar_keys.extend(
        [(f"debug_partner_card_attn_for_agent0_card{i}", f"Debug/partner_card_attn_for_agent0_card{i}") for i in range(5)]
    )
    scalar_keys.extend(
        [(f"debug_partner_card_attn_for_agent1_card{i}", f"Debug/partner_card_attn_for_agent1_card{i}") for i in range(5)]
    )

    scalar_mean = {}
    scalar_std = {}
    for key, _ in scalar_keys:
        if key in train_metrics:
            vals = np.array(train_metrics[key])
            scalar_mean[key] = np.mean(vals, axis=0)
            scalar_std[key] = np.std(vals, axis=0)

    # --- Export CSV ---
    csv_header = ["update", "timestep"]
    for name in metric_names:
        csv_header.extend([f"{name}_mean", f"{name}_std"])
    if config.task["ENV_NAME"] == "overcooked-v1" and "base_return" in metric_names:
        csv_header.append("soups_delivered")
    for key, _ in scalar_keys:
        if key in scalar_mean:
            csv_header.extend([f"{key}_mean", f"{key}_std"])

    csv_path = os.path.join(savedir, "train_stats.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        for step in range(num_updates):
            row = [step, (step + 1) * rollout_length * num_envs]
            for name in metric_names:
                stat_data = np.array(train_stats[name])
                seed_means = stat_data[:, step, 0]
                row.extend([float(seed_means.mean()), float(seed_means.std())])
            if config.task["ENV_NAME"] == "overcooked-v1" and "base_return" in metric_names:
                base_data = np.array(train_stats["base_return"])
                row.append(float(base_data[:, step, 0].mean()) / 20.0)
            for key, _ in scalar_keys:
                if key in scalar_mean:
                    row.extend([float(scalar_mean[key][step]), float(scalar_std[key][step])])
            writer.writerow(row)

    print(f"[log_metrics] CSV: {csv_path} ({num_updates} updates, {num_seeds} seeds, {len(csv_header)} cols)")
    import wandb as _wandb
    _wandb.save(csv_path, base_path=savedir)

    # --- Log to wandb ---
    print_interval = max(1, num_updates // 20)

    for step in range(num_updates):
        step_data = {"train_step": step}

        # Episode metrics
        for stat_name, stat_data in episode_stats_mean.items():
            step_data[f"Train/{stat_name}_mean"] = float(stat_data[step, 0])
            if num_seeds > 1:
                step_data[f"Train/{stat_name}_std"] = float(stat_data[step, 1])
        if "base_return" in episode_stats_mean and config.task["ENV_NAME"] == "overcooked-v1":
            step_data["Train/soups_delivered"] = float(episode_stats_mean["base_return"][step, 0] / 20.0)

        # Scalar metrics
        for key, wandb_name in scalar_keys:
            if key in scalar_mean:
                step_data[f"{wandb_name}/mean"] = float(scalar_mean[key][step])
                if num_seeds > 1:
                    step_data[f"{wandb_name}/std"] = float(scalar_std[key][step])

        # Per-seed curves for cross-seed analysis
        for stat_name in train_stats:
            stat_data = np.array(train_stats[stat_name])
            for seed_idx in range(num_seeds):
                step_data[f"Seeds/{stat_name}/seed_{seed_idx}"] = float(stat_data[seed_idx, step, 0])

        logger.log(step_data, commit=True)

        if step % print_interval == 0 or step == num_updates - 1:
            env_steps = (step + 1) * rollout_length * num_envs
            pct = (step + 1) / num_updates * 100
            ret_str = "  ".join(f"{sn}={sd[step, 0]:.2f}" for sn, sd in episode_stats_mean.items())
            jsd = float(scalar_mean.get("jsd_mean", np.zeros(num_updates))[step])
            beta = float(scalar_mean.get("ja_beta", np.zeros(num_updates))[step])
            loss = float(scalar_mean.get("loss_total", np.zeros(num_updates))[step])
            grad = float(scalar_mean.get("grad_norm", np.zeros(num_updates))[step])
            extra = ""
            if "base_return" in episode_stats_mean and config.task["ENV_NAME"] == "overcooked-v1":
                soups = episode_stats_mean["base_return"][step, 0] / 20.0
                extra = f"  soups={soups:.1f}"
            print(f"[{pct:5.1f}%] step={step}/{num_updates}  env_steps={env_steps}  "
                  f"{ret_str}{extra}  jsd={jsd:.4f}  beta={beta:.4f}  loss={loss:.4f}  grad={grad:.3f}")

    logger.commit()

    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
