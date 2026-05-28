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
from typing import NamedTuple

import hydra
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent, _get_image_dims
from agents.ja_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import jsd_divergence, build_card_masks
from common.save_load_utils import REPO_PATH, save_train_run
from common.train_logging import log_live_chunk_metrics, report_ja_training_outputs
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import batchify, unbatchify, _create_minibatches


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
    plh_actor: jnp.ndarray             # (NUM_ACTORS, lstm_dim) or scalar 0 when disabled
    plh_critic: jnp.ndarray            # (NUM_ACTORS, lstm_dim) or scalar 0 when disabled
    partner_argmax: jnp.ndarray        # (NUM_ACTORS,) int — aux target aligned to this observation
    partner_argmax_valid: jnp.ndarray  # (NUM_ACTORS,) bool — False when no causal target exists yet
    partner_argmax_weight: jnp.ndarray # (NUM_ACTORS,) float — mass-based reliability weight for aux target


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
    query_partner_lstm = config.get("QUERY_PARTNER_LSTM", False)
    lstm_hidden_dim = config.get("LSTM_HIDDEN_DIM", 128)
    ja_card_attn = config.get("JA_CARD_ATTN", False)
    # Whether to feed the partner's translated card-attention back into the
    # next step's obs. Defaults to True. When False, the JSD reward still
    # fires but the policy receives no partner info beyond what the message
    # channel and rendered dot already provide.
    ja_card_partner_feed = ja_card_attn and config.get("JA_CARD_PARTNER_FEED", True)
    partner_feed_dim = 5
    ja_card_jsd_coef = config.get("JA_CARD_JSD_COEF", 0.0)
    # Two-part JA attention shaping (parallel to comm shaping):
    #   match: per-step lagged similarity between each agent's current
    #          canonical card-attention distribution and the partner's
    #          previous-step canonical card-attention distribution.
    #   self: per-step own-attention/action consistency. The emitted action is
    #         mapped to canonical card identity and rewarded by the mass that
    #         the agent's current canonical attention assigns to that card.
    # Both terms reuse phys_0 / phys_1 computed in the JA_CARD_METRIC path.
    ja_attn_match_coef = config.get("JA_ATTN_MATCH_COEF", 0.0)
    ja_attn_self_coef = config.get("JA_ATTN_SELF_COEF", 0.0)
    # Optional direct action shaping: reward the emitted canonical action
    # according to the partner's PREVIOUS-step canonical card-attention
    # distribution.
    ja_gaze_pick_coef = config.get("JA_GAZE_PICK_COEF", 0.0)
    ja_gaze_pick_active = ja_gaze_pick_coef > 0
    ja_attn_shaping_active = (
        ja_attn_match_coef > 0 or ja_attn_self_coef > 0
    )
    ja_prev_partner_phys_active = ja_attn_match_coef > 0 or ja_gaze_pick_active
    # Auxiliary NLL loss read from the attention pool directly: forces the
    # per-card pooled attention to peak at the partner's argmax view-slot in
    # the agent's own frame. When JA_CARD_PARTNER_FEED is on, the target is
    # shifted by one step to match the causal partner-feed suffix. The loss is
    # also weighted by partner/ego card mass so negligible card attention does
    # not create arbitrary supervision.
    ja_aux_partner_argmax_coef = config.get("JA_AUX_PARTNER_ARGMAX_COEF", 0.0)
    ja_aux_partner_argmax_active = ja_aux_partner_argmax_coef > 0
    # When True, compute the OP-corrected card-level JSD as a diagnostic metric
    # without feeding partner attention back into the obs and without applying
    # any shaping reward. Same OP requirements as JA_CARD_ATTN.
    # JA_CARD_ATTN implies this. So does JA_ATTN_*_COEF being positive.
    ja_card_metric = (
        ja_card_attn or config.get("JA_CARD_METRIC", False) or ja_attn_shaping_active
        or ja_aux_partner_argmax_active or ja_gaze_pick_active
    )
    if ja_card_attn and feed_other_attn:
        raise ValueError("JA_CARD_ATTN is mutually exclusive with FEED_OTHER_ATTN")
    if ja_card_metric:
        env_kwargs = config.get("ENV_KWARGS", {})
        if not (env_kwargs.get("other_play_position_shuffle") and env_kwargs.get("other_play_recolouring")):
            raise ValueError(
                "JA_CARD_ATTN/JA_CARD_METRIC require both "
                "other_play_position_shuffle and other_play_recolouring"
            )

    # Precompute image and feature-map dimensions (only needed for image obs)
    obs_type = _get_obs_type(config)
    if obs_type == "image":
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

    _env_max_steps = int(config.get("ENV_KWARGS", {}).get("max_steps", 8))

    # Precompute card tile masks (used by JA card attention and the diagnostic-
    # only JA_CARD_METRIC path).
    _need_card_masks = ja_card_metric and feat_h > 0
    if _need_card_masks:
        _card_masks = _build_card_masks(img_h, img_w, feat_h, feat_w)
    else:
        _card_masks = None
    _num_cards = int(_card_masks.shape[0]) if _card_masks is not None else 5

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    agent_init_fn = initialize_ja_image_agent if obs_type == "image" else initialize_ja_agent

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
        elif ja_card_partner_feed:
            init_partner_card_attn = jnp.zeros((num_actors, partner_feed_dim))
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng,
                            init_partner_card_attn)
        else:
            runner_state = (train_state, env_state, obsv, init_done, init_hstate, _rng)
        if ja_prev_partner_phys_active:
            init_prev_partner_phys = jnp.zeros((num_actors, 5), dtype=jnp.float32)
            init_prev_partner_valid = jnp.zeros((num_actors,), dtype=bool)
            runner_state = runner_state + (
                init_prev_partner_phys,
                init_prev_partner_valid,
            )
        if ja_aux_partner_argmax_active and ja_card_partner_feed:
            init_partner_argmax = jnp.zeros((num_actors,), dtype=jnp.int32)
            init_partner_argmax_valid = jnp.zeros((num_actors,), dtype=bool)
            init_partner_argmax_weight = jnp.zeros((num_actors,), dtype=jnp.float32)
            runner_state = runner_state + (
                init_partner_argmax,
                init_partner_argmax_valid,
                init_partner_argmax_weight,
            )
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
                        # Build network inputs (shared by policy + aux pass).
                        inputs_apply = (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.avail_actions,
                        )
                        if query_partner_lstm:
                            inputs_apply = inputs_apply + (
                                traj_batch.plh_actor.reshape(1, num_actors, -1),
                                traj_batch.plh_critic.reshape(1, num_actors, -1),
                            )
                        hidden = policy._unpack_hstate(init_hstate)

                        # Single forward pass. When aux is active, read the
                        # attention map directly and compute a mass-weighted
                        # NLL against the partner argmax target stored in the
                        # transition. The target is already time-aligned to
                        # the observation: when partner-feed is enabled it
                        # refers to the previous step (the only causal signal
                        # present in the obs suffix), otherwise it refers to
                        # the current step. The loss can only descend if
                        # attention itself peaks on the target view-slot — no
                        # Dense shortcut from actor_out.
                        _, pi, value, attn_map_apply = policy.network.apply(
                            params, hidden, inputs_apply,
                        )
                        if ja_aux_partner_argmax_active:
                            attn_2d = attn_map_apply                            # (T, num_actors, fh, fw)
                            # Take the agent's attention map, multiply it
                            # element-wise by card c's binary mask, and sum
                            # the result. Repeated for every card c.
                            per_card_attn = jnp.einsum(
                                "tahw,chw->tac", attn_2d, _card_masks,
                            )                                                   # (T, num_actors, 5)
                            own_card_mass = per_card_attn.sum(axis=-1)
                            per_card_norm = per_card_attn / (
                                own_card_mass[..., None] + 1e-8
                            )
                            log_probs = jnp.log(per_card_norm + 1e-8)
                            partner_target_flat = traj_batch.partner_argmax.reshape(-1)
                            log_probs_flat = log_probs.reshape(-1, log_probs.shape[-1])
                            nll_flat = -jnp.take_along_axis(
                                log_probs_flat, partner_target_flat[:, None], axis=-1,
                            ).squeeze(-1)
                            aux_weight_flat = (
                                traj_batch.partner_argmax_valid.reshape(-1).astype(jnp.float32)
                                * traj_batch.partner_argmax_weight.reshape(-1)
                                * own_card_mass.reshape(-1)
                            )
                            aux_weight_flat = jax.lax.stop_gradient(aux_weight_flat)
                            aux_denom = jnp.maximum(aux_weight_flat.sum(), 1e-8)
                            aux_loss = (nll_flat * aux_weight_flat).sum() / aux_denom
                        else:
                            aux_loss = jnp.float32(0.0)

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
                            + ja_aux_partner_argmax_coef * aux_loss
                        )
                        # PPO diagnostics
                        approx_kl = ((ratio - 1) - jnp.log(ratio)).mean()
                        clip_frac = (jnp.abs(ratio - 1.0) > config["CLIP_EPS"]).mean()
                        return total_loss, (value_loss, loss_actor, entropy,
                                            approx_kl, clip_frac, ratio.mean(), ratio.std(),
                                            aux_loss)

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

        def _extract_comm_reward_components(info):
            """Pop communication reward diagnostics and flatten them per actor."""
            def _pop_metric(key):
                raw = info.pop(key, jnp.zeros((num_envs, env.num_agents)))
                return raw.transpose(1, 0).reshape(-1)

            return (
                _pop_metric("comm_reward"),
                _pop_metric("comm_reward_match"),
                _pop_metric("comm_reward_stable"),
                _pop_metric("comm_reward_follow"),
            )

        def _project_card_attention(attn_map, env_state):
            """Project averaged attention into canonical card space."""
            attn = attn_map.squeeze(0)  # (num_actors, feat_h, feat_w)
            card_pos_attn = jnp.einsum("ahw,chw->ac", attn, _card_masks)
            card_pos_attn_0 = card_pos_attn[:num_envs]
            card_pos_attn_1 = card_pos_attn[num_envs:]

            eps = 1e-8
            perm_0 = env_state.env_state.env_state.per_agent_perm["agent_0"]
            perm_1 = env_state.env_state.env_state.per_agent_perm["agent_1"]
            batch_idx = jnp.arange(num_envs)[:, None]
            phys_0 = jnp.zeros((num_envs, _num_cards)).at[batch_idx, perm_0].set(card_pos_attn_0)
            phys_1 = jnp.zeros((num_envs, _num_cards)).at[batch_idx, perm_1].set(card_pos_attn_1)

            m_0 = card_pos_attn_0.sum(axis=-1)
            m_1 = card_pos_attn_1.sum(axis=-1)
            q_phys_0 = phys_0 / (m_0[:, None] + eps)
            q_phys_1 = phys_1 / (m_1[:, None] + eps)
            return perm_0, perm_1, phys_0, phys_1, q_phys_0, q_phys_1, m_0, m_1

        def _compute_card_jsd_reward(q_phys_0, q_phys_1, m_0, m_1, step_count_batch):
            """Compute diagnostic card JSD and optional shaping reward."""
            card_jsd_per_env = jsd_divergence(
                q_phys_0[:, None, :], q_phys_1[:, None, :]
            )
            card_jsd_step_mean = card_jsd_per_env.mean()

            if ja_card_attn and ja_card_jsd_coef > 0:
                l2_div = jnp.sum((q_phys_0 - q_phys_1) ** 2, axis=-1)
                r_card_jsd_env = -ja_card_jsd_coef * jnp.minimum(m_0, m_1) * l2_div
                r_card_jsd_valid = step_count_batch[:num_envs] > 1
                r_card_jsd_env = jnp.where(r_card_jsd_valid, r_card_jsd_env, 0.0)
                r_card_jsd = jnp.concatenate([r_card_jsd_env, r_card_jsd_env])
            else:
                r_card_jsd = jnp.zeros(num_actors)
            return card_jsd_step_mean, jax.lax.stop_gradient(r_card_jsd)

        def _build_partner_card_attention_feed(attn_map, new_done_batch, perm_0, perm_1, phys_0, phys_1):
            """Translate partner attention into each agent's own view frame."""
            translated_for_0 = jnp.take_along_axis(phys_1, perm_0, axis=1)
            translated_for_1 = jnp.take_along_axis(phys_0, perm_1, axis=1)

            partner_card_attn = jnp.concatenate([translated_for_0, translated_for_1], axis=0)
            return jnp.where(new_done_batch[:, None], 0.0, partner_card_attn)

        def _compute_partner_argmax_targets(phys_0, phys_1, q_phys_0, q_phys_1, perm_0, perm_1, m_0, m_1):
            """Compute partner attention targets in each agent's own view frame."""
            partner_canon_argmax_0 = phys_1.argmax(axis=-1)
            partner_canon_argmax_1 = phys_0.argmax(axis=-1)
            inv_perm_0 = jnp.argsort(perm_0, axis=-1)
            inv_perm_1 = jnp.argsort(perm_1, axis=-1)
            partner_argmax_in_ego_view_0 = jnp.take_along_axis(
                inv_perm_0, partner_canon_argmax_0[:, None], axis=1,
            ).squeeze(-1)
            partner_argmax_in_ego_view_1 = jnp.take_along_axis(
                inv_perm_1, partner_canon_argmax_1[:, None], axis=1,
            ).squeeze(-1)
            partner_argmax_per_actor = jnp.concatenate(
                [partner_argmax_in_ego_view_0, partner_argmax_in_ego_view_1]
            ).astype(jnp.int32)
            current_partner_phys_for_actor = jnp.concatenate([q_phys_1, q_phys_0], axis=0)
            current_partner_mass_per_actor = jnp.concatenate([m_1, m_0])
            return (
                jax.lax.stop_gradient(partner_argmax_per_actor),
                jax.lax.stop_gradient(current_partner_phys_for_actor.astype(jnp.float32)),
                jax.lax.stop_gradient(current_partner_mass_per_actor.astype(jnp.float32)),
            )

        def _compute_attention_shaping_rewards(
            q_phys_0,
            q_phys_1,
            prev_partner_phys_for_actor,
            prev_partner_valid_for_actor,
            action_0_gt,
            action_1_gt,
            pick_0_is_card,
            pick_1_is_card,
        ):
            """Compute attention shaping and gaze-pick rewards."""
            if ja_card_metric and ja_attn_shaping_active:
                if ja_attn_match_coef > 0:
                    log2 = jnp.log(jnp.asarray(2.0))
                    prev_partner_phys_0 = prev_partner_phys_for_actor[:num_envs]
                    prev_partner_phys_1 = prev_partner_phys_for_actor[num_envs:]
                    prev_partner_valid_0 = prev_partner_valid_for_actor[:num_envs]
                    prev_partner_valid_1 = prev_partner_valid_for_actor[num_envs:]
                    jsd_match_0 = jsd_divergence(
                        q_phys_0[:, None, :], prev_partner_phys_0[:, None, :]
                    ).reshape(num_envs)
                    jsd_match_1 = jsd_divergence(
                        q_phys_1[:, None, :], prev_partner_phys_1[:, None, :]
                    ).reshape(num_envs)
                    match_score_0 = 1.0 - jnp.clip(jsd_match_0, 0.0, log2) / log2
                    match_score_1 = 1.0 - jnp.clip(jsd_match_1, 0.0, log2) / log2
                    r_attn_match_per_actor = jnp.concatenate([
                        jnp.where(prev_partner_valid_0, ja_attn_match_coef * match_score_0, 0.0),
                        jnp.where(prev_partner_valid_1, ja_attn_match_coef * match_score_1, 0.0),
                    ])
                else:
                    r_attn_match_per_actor = jnp.zeros(num_actors)

                if ja_attn_self_coef > 0:
                    own_action_mass_0 = jnp.take_along_axis(
                        q_phys_0, action_0_gt[:, None], axis=1,
                    ).squeeze(-1)
                    own_action_mass_1 = jnp.take_along_axis(
                        q_phys_1, action_1_gt[:, None], axis=1,
                    ).squeeze(-1)
                    own_action_mass_0 = jnp.where(pick_0_is_card, own_action_mass_0, 0.0)
                    own_action_mass_1 = jnp.where(pick_1_is_card, own_action_mass_1, 0.0)
                    r_attn_self_per_actor = ja_attn_self_coef * jnp.concatenate([
                        own_action_mass_0,
                        own_action_mass_1,
                    ])
                else:
                    r_attn_self_per_actor = jnp.zeros(num_actors)

                r_attn_shaping = r_attn_match_per_actor + r_attn_self_per_actor
            else:
                r_attn_shaping = jnp.zeros(num_actors)
                r_attn_match_per_actor = jnp.zeros(num_actors)
                r_attn_self_per_actor = jnp.zeros(num_actors)

            if ja_gaze_pick_active:
                action_canonical_per_actor = jnp.concatenate([action_0_gt, action_1_gt])
                picked_partner_mass = jnp.take_along_axis(
                    prev_partner_phys_for_actor,
                    action_canonical_per_actor[:, None],
                    axis=1,
                ).squeeze(-1)
                action_is_card = jnp.concatenate([pick_0_is_card, pick_1_is_card])
                gaze_pick_ok = prev_partner_valid_for_actor & action_is_card
                r_gaze_pick_per_actor = jnp.where(
                    gaze_pick_ok,
                    ja_gaze_pick_coef * picked_partner_mass,
                    0.0,
                )
            else:
                r_gaze_pick_per_actor = jnp.zeros(num_actors)

            return (
                jax.lax.stop_gradient(r_attn_shaping),
                jax.lax.stop_gradient(r_attn_match_per_actor),
                jax.lax.stop_gradient(r_attn_self_per_actor),
                jax.lax.stop_gradient(r_gaze_pick_per_actor),
            )

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
                if ja_aux_partner_argmax_active and ja_card_partner_feed:
                    (
                        *rest,
                        prev_partner_argmax,
                        prev_partner_argmax_valid,
                        prev_partner_argmax_weight,
                    ) = runner_state_core
                    runner_state_core = tuple(rest)
                else:
                    prev_partner_argmax = jnp.zeros((num_actors,), dtype=jnp.int32)
                    prev_partner_argmax_valid = jnp.zeros((num_actors,), dtype=bool)
                    prev_partner_argmax_weight = jnp.zeros((num_actors,), dtype=jnp.float32)
                if ja_prev_partner_phys_active:
                    (
                        *rest,
                        prev_partner_phys_for_actor,
                        prev_partner_valid_for_actor,
                    ) = runner_state_core
                    runner_state_core = tuple(rest)
                else:
                    prev_partner_phys_for_actor = jnp.zeros((num_actors, 5), dtype=jnp.float32)
                    prev_partner_valid_for_actor = jnp.zeros((num_actors,), dtype=bool)
                if feed_other_attn:
                    (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn) = runner_state_core
                elif ja_card_partner_feed:
                    (train_state, env_state, last_obs, last_done, hstate, rng,
                     prev_partner_card_attn) = runner_state_core
                else:
                    (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state_core

                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, num_actors)
                last_done_batch = batchify(last_done, env.agents, num_actors)

                # Augment obs with other agent's previous attention as 4th channel
                if feed_other_attn:
                    last_obs_batch = _augment_obs_with_attn(last_obs_batch, prev_other_attn)

                # Append partner's translated card attention as scalar suffix
                if ja_card_partner_feed:
                    last_obs_batch = jnp.concatenate(
                        [last_obs_batch, prev_partner_card_attn], axis=-1)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions_batch = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors).astype(jnp.float32))

                plh_kwarg = dict(
                    plh_actor=prev_plh_actor.reshape(1, num_actors, -1),
                    plh_critic=prev_plh_critic.reshape(1, num_actors, -1),
                ) if query_partner_lstm else {}
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

                (
                    comm_reward_batch,
                    comm_match_batch,
                    comm_stable_batch,
                    comm_follow_batch,
                ) = _extract_comm_reward_components(info)

                # Extract step count for attn-msg reward gating
                step_count_raw = info.pop("step_count", jnp.zeros((num_envs, env.num_agents)))
                step_count_batch = step_count_raw.transpose(1, 0).reshape(-1)

                info = jax.tree.map(lambda x: x.reshape((num_actors,)), info)

                # attn_map is (1, num_actors, fh, fw).
                attn_0 = attn_map[:, :num_envs, ...]
                attn_1 = attn_map[:, num_envs:, ...]
                r_ja = -jsd_divergence(
                    attn_0.squeeze(0),
                    attn_1.squeeze(0),
                )
                r_ja = jax.lax.stop_gradient(r_ja)

                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                r_ja_batch = jnp.concatenate([r_ja, r_ja])

                intrinsic = ja_beta * r_ja_batch
                new_done_batch = batchify(new_done, env.agents, num_actors).squeeze()
                num_cards = _num_cards

                # OP-corrected card-level joint attention. Under JA_CARD_METRIC we
                # always compute the JSD as a diagnostic; only JA_CARD_ATTN +
                # JA_CARD_JSD_COEF>0 turns it into an actual shaping reward, and
                # only JA_CARD_ATTN feeds the partner's translated card-attention
                # back into the next obs.
                if ja_card_metric:
                    (
                        perm_0,
                        perm_1,
                        phys_0,
                        phys_1,
                        q_phys_0,
                        q_phys_1,
                        m_0,
                        m_1,
                    ) = _project_card_attention(attn_map, env_state)
                    card_jsd_step_mean, ja_card_reward = _compute_card_jsd_reward(
                        q_phys_0, q_phys_1, m_0, m_1, step_count_batch,
                    )
                    if ja_card_attn:
                        new_partner_card_attn = _build_partner_card_attention_feed(
                            attn_map, new_done_batch, perm_0, perm_1, phys_0, phys_1,
                        )
                else:
                    ja_card_reward = jnp.zeros(num_actors)
                    card_jsd_step_mean = jnp.float32(0.0)
                    m_0 = jnp.zeros((num_envs,), dtype=jnp.float32)
                    m_1 = jnp.zeros((num_envs,), dtype=jnp.float32)
                    num_cards = _num_cards
                    q_phys_0 = jnp.zeros((num_envs, num_cards), dtype=jnp.float32)
                    q_phys_1 = jnp.zeros((num_envs, num_cards), dtype=jnp.float32)
                    phys_0 = jnp.zeros((num_envs, num_cards), dtype=jnp.float32)
                    phys_1 = jnp.zeros((num_envs, num_cards), dtype=jnp.float32)
                    perm_0 = jnp.zeros((num_envs, num_cards), dtype=jnp.int32)
                    perm_1 = jnp.zeros((num_envs, num_cards), dtype=jnp.int32)
                    new_partner_card_attn = jnp.zeros((num_actors, partner_feed_dim), dtype=jnp.float32)

                # Aux target: view-slot in the agent's own frame where the
                # partner is attending most. When partner-feed is enabled, the
                # scalar suffix at step t reflects partner attention from step
                # t-1, so the causal aux label must be shifted the same way.
                # Otherwise fall back to current-step supervision.
                if ja_card_metric:
                    (
                        partner_argmax_per_actor,
                        current_partner_phys_for_actor,
                        current_partner_mass_per_actor,
                    ) = _compute_partner_argmax_targets(
                        phys_0, phys_1, q_phys_0, q_phys_1, perm_0, perm_1, m_0, m_1,
                    )
                else:
                    partner_argmax_per_actor = jnp.zeros(num_actors, dtype=jnp.int32)
                    current_partner_phys_for_actor = jnp.zeros((num_actors, num_cards), dtype=jnp.float32)
                    current_partner_mass_per_actor = jnp.zeros(num_actors, dtype=jnp.float32)

                if ja_aux_partner_argmax_active and ja_card_partner_feed:
                    aux_partner_argmax = prev_partner_argmax
                    aux_partner_argmax_valid = prev_partner_argmax_valid
                    aux_partner_argmax_weight = prev_partner_argmax_weight
                else:
                    aux_partner_argmax = partner_argmax_per_actor
                    aux_partner_argmax_valid = jnp.ones((num_actors,), dtype=bool)
                    aux_partner_argmax_weight = current_partner_mass_per_actor

                env_idx = jnp.arange(num_envs)
                pick_0_view = action[:num_envs]
                pick_1_view = action[num_envs:]
                pick_0_is_card = pick_0_view < num_cards
                pick_1_is_card = pick_1_view < num_cards
                if ja_card_metric:
                    inv_recol_0 = env_state.env_state.per_agent_inv_recolouring["agent_0"]
                    inv_recol_1 = env_state.env_state.per_agent_inv_recolouring["agent_1"]
                    action_0_gt = inv_recol_0[env_idx, jnp.minimum(pick_0_view, num_cards - 1)]
                    action_1_gt = inv_recol_1[env_idx, jnp.minimum(pick_1_view, num_cards - 1)]
                else:
                    action_0_gt = jnp.zeros((num_envs,), dtype=jnp.int32)
                    action_1_gt = jnp.zeros((num_envs,), dtype=jnp.int32)

                # JA attention shaping. Reuses phys_0 / phys_1 from the
                # JA_CARD_METRIC path; only fires when at least one coef is
                # positive.
                (
                    r_attn_shaping,
                    r_attn_match_per_actor,
                    r_attn_self_per_actor,
                    r_gaze_pick_per_actor,
                ) = _compute_attention_shaping_rewards(
                    q_phys_0,
                    q_phys_1,
                    prev_partner_phys_for_actor,
                    prev_partner_valid_for_actor,
                    action_0_gt,
                    action_1_gt,
                    pick_0_is_card,
                    pick_1_is_card,
                )

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
                    plh_actor=plh_a_stored,
                    plh_critic=plh_c_stored,
                    partner_argmax=aux_partner_argmax,
                    partner_argmax_valid=aux_partner_argmax_valid,
                    partner_argmax_weight=aux_partner_argmax_weight,
                )

                if feed_other_attn:
                    new_other_attn = _swap_and_reset_attn(attn_map, new_done_batch)
                    runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng, new_other_attn)
                elif ja_card_partner_feed:
                    runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng,
                                    new_partner_card_attn)
                else:
                    runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                if ja_prev_partner_phys_active:
                    new_done_batch_gaze = batchify(new_done, env.agents, num_actors).squeeze()
                    new_partner_phys_for_actor = jnp.where(
                        new_done_batch_gaze[:, None], 0.0, current_partner_phys_for_actor
                    ).astype(jnp.float32)
                    new_gaze_pick_valid = ~new_done_batch_gaze
                    runner_state = runner_state + (
                        new_partner_phys_for_actor,
                        new_gaze_pick_valid,
                    )
                if ja_aux_partner_argmax_active and ja_card_partner_feed:
                    new_done_batch_aux = batchify(new_done, env.agents, num_actors).squeeze()
                    new_partner_argmax = jnp.where(
                        new_done_batch_aux, 0, partner_argmax_per_actor
                    ).astype(jnp.int32)
                    new_partner_argmax_valid = ~new_done_batch_aux
                    new_partner_argmax_weight = jnp.where(
                        new_done_batch_aux, 0.0, current_partner_mass_per_actor
                    ).astype(jnp.float32)
                    runner_state = runner_state + (
                        new_partner_argmax,
                        new_partner_argmax_valid,
                        new_partner_argmax_weight,
                    )
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
                                     ja_card_reward, card_jsd_step_mean,
                                     r_attn_shaping,
                                     r_attn_match_per_actor,
                                     r_attn_self_per_actor, r_gaze_pick_per_actor)

            runner_state, (traj_batch, intrinsic_batch, comm_reward_batch,
                           comm_match_batch, comm_stable_batch, comm_follow_batch,
                           ja_card_reward_batch, card_jsd_batch,
                           r_attn_shaping_batch,
                           r_attn_match_batch,
                           r_attn_self_batch, r_gaze_pick_batch) = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            if query_partner_lstm:
                *rest, prev_plh_actor, prev_plh_critic = runner_state
                runner_state = tuple(rest)
            if ja_aux_partner_argmax_active and ja_card_partner_feed:
                (
                    *rest,
                    prev_partner_argmax,
                    prev_partner_argmax_valid,
                    prev_partner_argmax_weight,
                ) = runner_state
                runner_state = tuple(rest)
            if ja_prev_partner_phys_active:
                (
                    *rest,
                    prev_partner_phys_for_actor,
                    prev_partner_valid_for_actor,
                ) = runner_state
                runner_state = tuple(rest)
            if feed_other_attn:
                (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn) = runner_state
            elif ja_card_partner_feed:
                (train_state, env_state, last_obs, last_done, hstate, rng,
                 prev_partner_card_attn) = runner_state
            else:
                (train_state, env_state, last_obs, last_done, hstate, rng) = runner_state

            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            if feed_other_attn:
                last_obs_batch = _augment_obs_with_attn(last_obs_batch, prev_other_attn)
            if ja_card_partner_feed:
                last_obs_batch = jnp.concatenate(
                    [last_obs_batch, prev_partner_card_attn], axis=-1)
            last_avail = jax.vmap(env.get_avail_actions)(env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32))

            last_plh_kwarg = dict(
                plh_actor=prev_plh_actor.reshape(1, num_actors, -1),
                plh_critic=prev_plh_critic.reshape(1, num_actors, -1),
            ) if query_partner_lstm else {}
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
            combined_raw = (
                raw_env_reward + intrinsic_batch + scaled_comm_reward
                + ja_card_reward_batch + r_attn_shaping_batch + r_gaze_pick_batch
            )
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
                         approx_kl, clip_frac, ratio_mean, ratio_std,
                         aux_partner_argmax_loss)), grad_norm = loss_info

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
            metric["card_jsd_mean"] = card_jsd_batch.mean()
            # Average across both agents — under parameter sharing they converge
            # to the same value in expectation, so a single number suffices.
            metric["ja_attn_shaping_mean"] = r_attn_shaping_batch.mean()
            metric["ja_attn_match_mean"] = r_attn_match_batch.mean()
            metric["ja_attn_self_mean"] = r_attn_self_batch.mean()
            metric["aux_partner_argmax_loss"] = aux_partner_argmax_loss[0].mean()
            valid_prev_partner_count = jnp.maximum(
                jnp.asarray(r_gaze_pick_batch.size, dtype=jnp.float32), 1.0
            )
            metric["ja_gaze_pick_mean"] = r_gaze_pick_batch.sum() / valid_prev_partner_count

            if feed_other_attn:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng, prev_other_attn)
            elif ja_card_partner_feed:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng,
                                prev_partner_card_attn)
            else:
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            if ja_prev_partner_phys_active:
                runner_state = runner_state + (
                    prev_partner_phys_for_actor,
                    prev_partner_valid_for_actor,
                )
            if ja_aux_partner_argmax_active and ja_card_partner_feed:
                runner_state = runner_state + (
                    prev_partner_argmax,
                    prev_partner_argmax_valid,
                    prev_partner_argmax_weight,
                )
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
                log_live_chunk_metrics(
                    chunk_metrics,
                    env_step=steps_done * env_steps_per_update,
                    seed_idx=seed_idx,
                    logger=logger,
                )

        # Pull per-seed outputs to the host as each seed finishes -- holding
        # all seeds' metrics/checkpoints on-device and stacking them at the
        # end OOMs the GPU once NUM_SEEDS is large.
        all_seed_final_params.append(jax.device_get(runner_state[0].params))
        # Concatenate chunk metrics along the update axis (axis 0)
        all_seed_metrics.append(jax.device_get(
            jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *seed_metrics)))
        all_seed_ckpts.append(jax.device_get(
            jax.tree.map(lambda *xs: jnp.stack(xs), *seed_ckpts)))

    # Stack across seeds on the host: (num_seeds, ...). np.stack, not jnp --
    # the per-seed leaves are already host arrays (device_get above) and a
    # large device stack of full metrics/checkpoints exhausts GPU memory.
    stacked_params = jax.tree.map(lambda *xs: np.stack(xs), *all_seed_final_params)
    stacked_metrics = jax.tree.map(lambda *xs: np.stack(xs), *all_seed_metrics)
    stacked_ckpts = jax.tree.map(lambda *xs: np.stack(xs), *all_seed_ckpts)

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
    # Structure: <ckpt_root>/ckpt_{i:02d}_ret_{seed_mean_return}/  (params, shape (num_seeds, ...))
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
            ckpt_name = f"ckpt_{i:02d}_ret_{ret_mean:.2f}"
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

    report_ja_training_outputs(config, out, logger)

    eval_out = {**out, "final_params": best_params} if use_best else out
    log_greedy_eval(algorithm_config, env, eval_out, logger)
    log_eval_video(algorithm_config, env, eval_out, logger)

    if num_seeds > 1:
        log_xp_eval(algorithm_config, env, eval_out)

    return out


def log_xp_eval(algorithm_config, env, out):
    """Run greedy cross-play evaluation when NUM_SEEDS > 1.

    Evaluates under the training OP regime (the wrappers active during
    training). Results are logged to wandb under the `XP/` prefix.
    """
    import wandb
    from evaluation.run_xp_seeds import run_xp_from_params

    obs_type = _get_obs_type(algorithm_config)
    init_fn = initialize_ja_image_agent if obs_type == "image" else initialize_ja_agent
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algorithm_config, env, rng)

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    print("[xp_eval] Running greedy XP (training regime)...")
    run_xp_from_params(
        env, policy, out["final_params"], algorithm_config,
        savedir=savedir,
        task_name=algorithm_config.get("ENV_NAME"),
        wb_run=wandb.run,
        greedy_eval=True,
        wb_prefix="XP",
    )
