"""JA-IPPO for LBF.

Per-fruit attention pooling with a soft cross-entropy aux loss against the
partner's per-fruit attention, plus a distance-progress shaping term toward
the agent's own argmax fruit (r_self; card-game r_attn_self analog). Parallel
to ja_ippo.py (the OP-corrected card-game JA path). No Other-Play. Fruits are
lex-sorted each step; positions are fixed per episode so slot k is a stable
fruit identity.
"""
from __future__ import annotations

import os
from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_ja_image_agent
from agents.lbf.ja_lbf_attention import (
    agent_positions_from_log_state,
    as_spatial_attention,
    food_state_from_log_state,
    lbf_attention_ctx,
    lex_sort_food,
    per_fruit_attn,
    swap_partner,
)
from common.train_logging import (
    IMAGE_IPPO_SCALAR_KEYS,
    log_live_chunk_metrics,
    report_basic_training_outputs,
)
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.eval_lbf import _render_lbf_eval_frames
from marl.ippo_core import (
    calculate_gae,
    configure_training_dims,
    make_optimizer,
)
from marl.ja_ppo_core import (
    compute_last_value_ja,
    global_grad_norm,
    ppo_actor_critic_losses,
)
from marl.ppo_utils import _create_minibatches, batchify, unbatchify


JA_LBF_SCALAR_KEYS = list(IMAGE_IPPO_SCALAR_KEYS) + [
    ("aux_partner_argmax_loss", "Losses"),
    ("r_self_mean", "JA"),
]


class TransitionJA(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray  # = env reward + r_self_coef*r_self
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    # JA-specific fields, all per-actor. Fruit slot ordering is lex by (row, col).
    partner_prev_fruit_attn: jnp.ndarray  # (num_actors, N) float — per-fruit attention at t-1; aux target + obs feed
    partner_prev_valid: jnp.ndarray     # (num_actors,) bool — mask aux on invalid steps
    food_pos: jnp.ndarray               # (num_actors, N, 2) int — LEX SORTED, tiled per actor
    r_self: jnp.ndarray                 # (num_actors,) float — unscaled self shaping bonus, for logging
    food_eaten: jnp.ndarray             # (num_actors, N) bool — LEX SORTED, tiled per actor


def make_train(config, env):
    configure_training_dims(config, env)
    num_envs = config["NUM_ENVS"]
    num_actors = config["NUM_ACTORS"]
    num_agents = env.num_agents
    rollout_length = config["ROLLOUT_LENGTH"]

    if num_agents != 2:
        raise ValueError(
            f"ja_ippo_lbf assumes 2 agents (parameter-shared), got {num_agents}"
        )

    # Image / feature-map geometry
    ctx = lbf_attention_ctx(config, env)
    img_h, img_w = ctx["img_h"], ctx["img_w"]
    feat_h, feat_w = ctx["feat_h"], ctx["feat_w"]
    tile_size = ctx["tile_size"]
    num_fruits = ctx["num_fruits"]

    aux_coef = float(config.get("JA_AUX_PARTNER_ARGMAX_COEF", 0.0))
    r_self_coef = float(config.get("JA_FRUIT_R_SELF_COEF", 0.0))
    aux_active = aux_coef > 0.0
    # When true, partner's previous per-fruit attention vector (length N,
    # lex-sorted) is appended to the agent's obs as a scalar suffix. When
    # false, the agent receives no online partner-attention info; aux loss
    # can still operate (its target is stored in the trajectory regardless).
    partner_feed_active = bool(config.get("JA_FRUIT_PARTNER_FEED", True))

    config = dict(config)
    if partner_feed_active:
        config["JA_ENTITY_FEED_DIM"] = num_fruits
    else:
        config["JA_ENTITY_FEED_DIM"] = 0

    print(
        f"[ja_ippo_lbf] grid={img_h}x{img_w} feat={feat_h}x{feat_w} "
        f"N_fruits={num_fruits} tile={tile_size} "
        f"aux_coef={aux_coef} r_self_coef={r_self_coef} "
        f"partner_feed={partner_feed_active}",
        flush=True,
    )

    def _tile_to_actors(x_env):
        """Replicate an env-level tensor across the agent axis to actor order.

        x_env: (num_envs, ...) -> (num_actors, ...) with both agents in env e
        getting the same value.
        """
        reps = (num_agents,) + (1,) * (x_env.ndim - 1)
        return jnp.tile(x_env, reps)

    def _augment_obs_with_partner_attn(obs_batch_2d, partner_fruit_attn):
        """Append partner's per-fruit attention vector (length N) as scalar suffix.

        When JA_FRUIT_PARTNER_FEED is false, returns the obs unchanged so the
        policy sees only the image (and the policy was initialised with
        scalar_dim=0, so shapes match).
        """
        if not partner_feed_active:
            return obs_batch_2d
        return jnp.concatenate([obs_batch_2d, partner_fruit_attn], axis=-1)

    def init(rng):
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_ja_image_agent(config, env, init_rng)

        train_state = TrainState.create(
            apply_fn=policy.network.apply, params=init_params, tx=make_optimizer(config),
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_hstate = policy.init_hstate(num_actors)
        init_done = {k: jnp.zeros((num_envs,), dtype=bool) for k in env.agents + ["__all__"]}

        # Partner's previous-step per-fruit attention vector (length N, lex-sorted).
        # Used as scalar suffix to obs (online) and soft CE aux target. Reset to
        # uniform on episode boundaries.
        init_partner_fruit_attn = (
            jnp.ones((num_actors, num_fruits), dtype=jnp.float32) / float(num_fruits)
        )
        init_partner_valid = jnp.zeros((num_actors,), dtype=bool)

        runner_state = (
            train_state, env_state, obsv, init_done, init_hstate, _rng,
            init_partner_fruit_attn, init_partner_valid,
        )
        return runner_state, policy

    def make_step_fn(policy):
        @jax.jit
        def step_fn(runner_state, update_steps):
            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done, hstate, rng,
                 prev_partner_fruit_attn, prev_partner_valid) = runner_state

                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, num_actors)
                last_done_batch = batchify(last_done, env.agents, num_actors)

                # Append partner's previous per-fruit attention (lex-sorted N values)
                # as a scalar suffix to the flat obs.
                obs_with_partner = _augment_obs_with_partner_attn(
                    last_obs_batch, prev_partner_fruit_attn,
                )

                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions_batch = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, num_actors).astype(jnp.float32),
                )

                action, value, pi, new_hstate, attn_map = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=obs_with_partner.reshape(1, num_actors, -1),
                    done=last_done_batch.reshape(1, num_actors),
                    avail_actions=avail_actions_batch.reshape(1, num_actors, -1),
                    hstate=hstate,
                    rng=act_rng,
                )

                log_prob = pi.log_prob(action).squeeze()
                action = action.squeeze()
                value = value.squeeze()

                # --- Per-fruit attention from current step ---
                # attn_map: (1, actors, fh, fw).
                attn_2d = as_spatial_attention(attn_map).squeeze(0)

                # Pre-step food state (LBF state shared by both agents in each env).
                # Lex-sort fruits so slot k is always the k-th fruit in (row, col)
                # reading order. Agent can identify each slot's fruit from its image.
                food_pos_env_raw, food_eaten_env_raw = food_state_from_log_state(env_state)
                food_pos_env, food_eaten_env = lex_sort_food(food_pos_env_raw, food_eaten_env_raw)
                food_pos_actors = _tile_to_actors(food_pos_env)         # (num_actors, N, 2)
                food_eaten_actors = _tile_to_actors(food_eaten_env)     # (num_actors, N)

                per_fruit_norm, _on_mass = per_fruit_attn(
                    attn_2d, food_pos_actors, food_eaten_actors,
                    tile_size, feat_h, feat_w, img_h, img_w,
                )
                own_argmax = jnp.argmax(per_fruit_norm, axis=-1).astype(jnp.int32)  # (num_actors,)

                # --- Pre-step positions (per actor) ---
                # pos_env shape: (num_envs, num_agents, 2) -> swapaxes -> (num_agents, num_envs, 2) -> (num_actors, 2)
                pos_pre = agent_positions_from_log_state(env_state)
                pos_pre_actors = jnp.swapaxes(pos_pre, 0, 1).reshape(num_actors, 2)

                # --- Step env ---
                env_act = unbatchify(action, env.agents, num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                new_obs, new_env_state, reward, new_done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0),
                )(rng_step, env_state, env_act)

                pos_post = agent_positions_from_log_state(new_env_state)
                pos_post_actors = jnp.swapaxes(pos_post, 0, 1).reshape(num_actors, 2)
                done_actors = batchify(new_done, env.agents, num_actors).squeeze().astype(bool)

                actor_idx = jnp.arange(num_actors)

                # --- Distance progress toward the agent's OWN current-argmax fruit ---
                # Confidence-weighted one-sided shaping (card-game r_attn_self
                # analog): bonus = own peak mass x max(0, d_pre - d_post). Zeroed
                # on the terminal transition, where the env auto-resets and
                # pos_post belongs to a fresh episode.
                own_target_pos = food_pos_actors[actor_idx, own_argmax]  # (num_actors, 2)
                d_pre_self = jnp.sum(jnp.abs(pos_pre_actors - own_target_pos), axis=-1).astype(jnp.float32)
                d_post_self = jnp.sum(jnp.abs(pos_post_actors - own_target_pos), axis=-1).astype(jnp.float32)
                own_peak_mass = per_fruit_norm[actor_idx, own_argmax]  # (num_actors,)
                r_self = jnp.where(
                    ~done_actors,
                    own_peak_mass * jnp.maximum(d_pre_self - d_post_self, 0.0),
                    0.0,
                )

                env_reward = batchify(reward, env.agents, num_actors).squeeze()
                shaped_reward = env_reward + r_self_coef * r_self

                transition = TransitionJA(
                    done=done_actors,
                    action=action,
                    value=value,
                    reward=shaped_reward,
                    log_prob=log_prob,
                    obs=obs_with_partner,
                    info=jax.tree.map(lambda x: x.reshape((num_actors,)), info),
                    avail_actions=avail_actions_batch,
                    partner_prev_fruit_attn=prev_partner_fruit_attn,
                    partner_prev_valid=prev_partner_valid,
                    food_pos=food_pos_actors.astype(jnp.int32),
                    food_eaten=food_eaten_actors,
                    r_self=r_self,
                )

                # --- Update partner-attention state for next step ---
                # Swap own per-fruit vector across agent halves so each agent gets
                # its partner's per-fruit attention. Reset to uniform on done.
                swapped_fruit_attn = swap_partner(per_fruit_norm, num_agents)
                uniform_fruit = jnp.ones((num_fruits,), dtype=jnp.float32) / float(num_fruits)
                new_partner_fruit_attn = jnp.where(
                    done_actors[:, None], uniform_fruit[None], swapped_fruit_attn,
                )
                new_partner_valid = ~done_actors

                runner_state = (
                    train_state, new_env_state, new_obs, new_done, new_hstate, rng,
                    new_partner_fruit_attn, new_partner_valid,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, rollout_length,
            )

            (train_state, env_state, last_obs, last_done, hstate, rng,
             prev_partner_fruit_attn, prev_partner_valid) = runner_state

            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_done_batch = batchify(last_done, env.agents, num_actors)
            last_avail = jax.vmap(env.get_avail_actions)(env_state)
            last_avail_batch = jax.lax.stop_gradient(
                batchify(last_avail, env.agents, num_actors).astype(jnp.float32),
            )
            # Bootstrap value uses the same augmented obs format the rollout/loss use.
            last_obs_with_partner = _augment_obs_with_partner_attn(
                last_obs_batch, prev_partner_fruit_attn,
            )
            last_val = compute_last_value_ja(
                policy, train_state.params,
                last_obs_with_partner, last_done_batch, last_avail_batch, hstate, num_actors,
            )
            advantages, targets = calculate_gae(config, traj_batch, last_val)

            train_state, loss_info, rng = _run_ppo_aux_epochs(
                config, policy, train_state, traj_batch, advantages, targets,
                rng, num_actors,
                tile_size=tile_size, feat_h=feat_h, feat_w=feat_w,
                img_h=img_h, img_w=img_w,
                aux_coef=aux_coef, aux_active=aux_active,
            )

            # --- Rollout-level metrics ---
            metric = traj_batch.info
            metric["update_steps"] = update_steps
            metric["loss_total"] = loss_info.total_loss.mean()
            metric["loss_value"] = loss_info.value_loss.mean()
            metric["loss_policy"] = loss_info.policy_loss.mean()
            metric["entropy"] = loss_info.entropy.mean()
            metric["grad_norm"] = loss_info.grad_norm.mean()
            metric["value_mean"] = traj_batch.value.mean()
            # Use the canonical key name so log_live_chunk_metrics picks it up
            # (JA_LIVE_SCALAR_KEYS in common/train_logging.py expects this exact name).
            metric["aux_partner_argmax_loss"] = loss_info.aux_loss.mean()
            metric["r_self_mean"] = traj_batch.r_self.mean()

            runner_state = (
                train_state, env_state, last_obs, last_done, hstate, rng,
                prev_partner_fruit_attn, prev_partner_valid,
            )
            return runner_state, update_steps + 1, metric

        return step_fn

    return init, make_step_fn


def _run_ppo_aux_epochs(
    config, policy, train_state, traj_batch, advantages, targets, rng, num_actors,
    *, tile_size, feat_h, feat_w, img_h, img_w,
    aux_coef, aux_active,
):
    """PPO update with optional per-fruit soft cross-entropy aux loss.

    Aux: soft cross-entropy between agent's per-fruit attention (point-gathered
    from its spatial attention at each lex-sorted fruit's centre cell) and
    partner's per-fruit attention (stored directly in the trajectory). Gradient
    on agent's attention flows through only N fruit cells — keeps cuDNN happy
    at large NUM_ENVS. Partner's per-fruit vector is fed online as scalar suffix.
    """

    def _update_epoch(update_state, unused):
        def _update_minbatch(train_state, batch_info):
            init_hstate, traj_batch, advantages, targets = batch_info

            def _loss_fn(params, traj_batch, gae, targets):
                hidden = policy._unpack_hstate(init_hstate)
                inputs_apply = (
                    traj_batch.obs,
                    traj_batch.done,
                    traj_batch.avail_actions,
                )
                new_hidden, pi, value, attn_map_apply = policy.network.apply(
                    params, hidden, inputs_apply,
                )

                # --- Aux loss: per-fruit SOFT cross-entropy ---
                # Agent's per-fruit attention via point-gather of its spatial
                # attention at each (lex-sorted) fruit's centre cell. Partner's
                # per-fruit attention vector is already stored in the trajectory.
                # Gradient on agent's attn_2d flows only through N fruit cells.
                if aux_active:
                    attn_2d = as_spatial_attention(attn_map_apply)     # (T, A, fh, fw)
                    agent_per_fruit, _agent_on_mass = per_fruit_attn(
                        attn_2d, traj_batch.food_pos, traj_batch.food_eaten,
                        tile_size, feat_h, feat_w, img_h, img_w,
                    )                                                  # (T, A, N)
                    target_soft = jax.lax.stop_gradient(
                        traj_batch.partner_prev_fruit_attn,
                    )                                                  # (T, A, N)

                    log_probs = jnp.log(agent_per_fruit + 1e-8)        # (T, A, N)
                    nll_per_step = -(target_soft * log_probs).sum(axis=-1)  # (T, A)
                    nll_flat = nll_per_step.reshape(-1)
                    # Partner-side confidence weight: mirrors card-game's
                    # current_partner_mass_per_actor. Flat partner gives ~1/N,
                    # confident partner gives ~peak mass.
                    partner_peak = target_soft.max(axis=-1)            # (T, A)
                    aux_weight = (
                        traj_batch.partner_prev_valid.reshape(-1).astype(jnp.float32)
                        * partner_peak.reshape(-1)
                    )
                    aux_weight = jax.lax.stop_gradient(aux_weight)
                    aux_denom = jnp.maximum(aux_weight.sum(), 1e-8)
                    aux_loss = (nll_flat * aux_weight).sum() / aux_denom
                else:
                    aux_loss = jnp.float32(0.0)

                terms = ppo_actor_critic_losses(
                    pi, value,
                    actions=traj_batch.action,
                    value_old=traj_batch.value,
                    log_prob_old=traj_batch.log_prob,
                    gae=gae, targets=targets,
                    clip_eps=config["CLIP_EPS"],
                )
                total_loss = (
                    terms.policy_loss
                    + config["VF_COEF"] * terms.value_loss
                    - config["ENT_COEF"] * terms.entropy
                    + aux_coef * aux_loss
                )
                return total_loss, (terms.value_loss, terms.policy_loss, terms.entropy, aux_loss)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, (value_loss, policy_loss, entropy, aux_loss)), grads = grad_fn(
                train_state.params, traj_batch, advantages, targets,
            )
            grad_norm = global_grad_norm(grads)
            train_state = train_state.apply_gradients(grads=grads)
            stats = _PPOAuxStats(
                total_loss=total_loss,
                value_loss=value_loss,
                policy_loss=policy_loss,
                entropy=entropy,
                grad_norm=grad_norm,
                aux_loss=aux_loss,
            )
            return train_state, stats

        train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
        rng, perm_rng = jax.random.split(rng)
        minibatches = _create_minibatches(
            traj_batch, advantages, targets, init_hstate,
            num_actors, config["NUM_MINIBATCHES"], perm_rng,
        )
        train_state, stats = jax.lax.scan(_update_minbatch, train_state, minibatches)
        update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
        return update_state, stats

    init_hstate = policy.init_hstate(num_actors)
    update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
    update_state, loss_info = jax.lax.scan(
        _update_epoch, update_state, None, config["UPDATE_EPOCHS"],
    )
    return update_state[0], loss_info, update_state[-1]


class _PPOAuxStats(NamedTuple):
    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    entropy: jnp.ndarray
    grad_norm: jnp.ndarray
    aux_loss: jnp.ndarray


def run_ja_ippo_lbf(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    num_seeds = algorithm_config["NUM_SEEDS"]
    num_updates = int(
        algorithm_config["TOTAL_TIMESTEPS"]
        // algorithm_config["ROLLOUT_LENGTH"]
        // algorithm_config["NUM_ENVS"]
    )
    num_ckpts = algorithm_config.get("NUM_CHECKPOINTS", 5)
    ckpt_interval = max(1, num_updates // max(1, num_ckpts - 1))

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, num_seeds)

    init_fn, make_step_fn = make_train(algorithm_config, env)

    print(
        f"[ja_ippo_lbf] NUM_UPDATES={num_updates}, NUM_SEEDS={num_seeds}, "
        f"NUM_ENVS={algorithm_config['NUM_ENVS']}",
        flush=True,
    )

    live_wandb = bool(algorithm_config.get("LIVE_WANDB_LOGGING", True))
    env_steps_per_update = algorithm_config["ROLLOUT_LENGTH"] * algorithm_config["NUM_ENVS"]
    freq_timesteps = float(algorithm_config.get("CHECKPOINT_FREQ_TIMESTEPS", 0) or 0)
    if freq_timesteps > 0:
        live_log_interval = max(1, int(freq_timesteps / env_steps_per_update))
    else:
        live_log_interval = max(1, ckpt_interval)

    seed_outputs = []
    for s in range(num_seeds):
        print(f"[ja_ippo_lbf] Seed {s+1}/{num_seeds}: initializing...", flush=True)
        runner_state, policy = init_fn(rngs[s])
        step_fn = make_step_fn(policy)

        checkpoints = []
        all_metrics = []
        chunk_buffer = []
        update_steps = jnp.int32(0)

        print(f"[ja_ippo_lbf] Seed {s+1}/{num_seeds}: compiling step fn...", flush=True)
        for step in range(num_updates):
            runner_state, update_steps, metric = step_fn(runner_state, update_steps)
            all_metrics.append(metric)
            chunk_buffer.append(metric)

            should_ckpt = (step % ckpt_interval == 0) or (step == num_updates - 1)
            if should_ckpt and len(checkpoints) < num_ckpts:
                checkpoints.append(runner_state[0].params)

            if step % max(1, num_updates // 10) == 0 or step == num_updates - 1:
                print(
                    f"[ja_ippo_lbf] Seed {s+1}/{num_seeds}: step {step+1}/{num_updates}",
                    flush=True,
                )

            is_chunk_end = ((step + 1) % live_log_interval == 0) or (step == num_updates - 1)
            if live_wandb and is_chunk_end and chunk_buffer:
                chunk_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *chunk_buffer)
                log_live_chunk_metrics(
                    chunk_metrics,
                    env_step=(step + 1) * env_steps_per_update,
                    seed_idx=s,
                    logger=logger,
                )
                chunk_buffer = []

        stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *all_metrics)
        stacked_ckpts = jax.tree.map(lambda *xs: jnp.stack(xs), *checkpoints)

        # Pull per-seed state off the GPU so the next seed has headroom for its
        # rollout/PPO activations. Without this, NUM_SEEDS >= ~6 OOMs at
        # NUM_ENVS=256 on 12x12-8food because prior seeds' params/metrics/
        # checkpoints stay resident on device.
        seed_outputs.append({
            "final_params": jax.device_get(runner_state[0].params),
            "metrics": jax.device_get(stacked_metrics),
            "checkpoints": jax.device_get(stacked_ckpts),
            "final_ckpt_idx": len(checkpoints),
        })

    print("[ja_ippo_lbf] Training complete.", flush=True)

    # Stack across seeds in host memory; jnp.stack would push everything back
    # to the GPU and recreate the OOM we just avoided.
    out = jax.tree.map(lambda *xs: np.stack(xs), *seed_outputs)

    try:
        _log_eval_video(algorithm_config, env, out, logger)
    except Exception as e:
        # Safety net: a rendering/moviepy failure should not discard the trained
        # params and metrics. The per-fruit obs suffix is handled via lbf_ctx in
        # _log_eval_video.
        print(f"[ja_ippo_lbf] WARN: eval video failed ({e}); continuing.", flush=True)

    # XP evaluation — only meaningful with multiple seeds. run_xp_from_params
    # runs in-memory from the just-trained params (no checkpoint reload),
    # NUM_EVAL_EPISODES episodes per pair, with the disjoint-pairing SEM. The
    # eval policy is rebuilt from the config; initialize_ja_image_agent
    # auto-resolves JA_ENTITY_FEED_DIM from the env when partner-feed is on.
    if num_seeds > 1:
        try:
            from evaluation.run_xp_seeds import run_xp_from_params

            xp_policy, _ = initialize_ja_image_agent(
                algorithm_config, env, jax.random.PRNGKey(0),
            )
            savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            run_xp_from_params(
                env, xp_policy, out["final_params"], algorithm_config,
                savedir=savedir,
                task_name=algorithm_config.get("ENV_NAME"),
                wb_run=getattr(logger, "run", None),
                greedy_eval=True,
                wb_prefix="XP",
            )
        except Exception as e:
            print(f"[ja_ippo_lbf] WARN: XP eval failed ({e}); continuing.", flush=True)

    report_basic_training_outputs(
        config, out, logger,
        scalar_keys=JA_LBF_SCALAR_KEYS,
        print_prefix="ja_ippo_lbf",
    )
    return out


def _log_eval_video(algorithm_config, env, out, logger):
    """Run one eval episode with final params (seed 0), log base + per-agent
    attention-overlay videos to wandb."""
    from evaluation.vis_episodes import make_attention_video, run_episode_with_states

    rng = jax.random.PRNGKey(0)
    policy, _ = initialize_ja_image_agent(algorithm_config, env, rng)
    final_params = jax.tree.map(lambda x: x[0], out["final_params"])
    inner_env = env._env

    # Build lbf_ctx mirroring the training-time per-fruit partner-feed wiring,
    # so the eval rollout sees the same obs the trained policy expects.
    lbf_ctx = None
    if bool(algorithm_config.get("JA_FRUIT_PARTNER_FEED", True)):
        lbf_ctx = lbf_attention_ctx(algorithm_config, env)

    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    ep_states, attn_data, _ep_actions, _ep_messages = run_episode_with_states(
        jax.random.PRNGKey(42), inner_env, final_params, policy,
        final_params, policy, max_steps,
        collect_attention=True, lbf_ctx=lbf_ctx,
    )
    print(f"[ja_ippo_lbf] Eval episode: {len(ep_states)} frames collected", flush=True)

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    video_dir = f"{savedir}/videos"
    os.makedirs(video_dir, exist_ok=True)

    frames = _render_lbf_eval_frames(inner_env, ep_states)

    from moviepy import ImageSequenceClip
    base_path = f"{video_dir}/eval_final.mp4"
    ImageSequenceClip(frames, fps=10).write_videofile(
        base_path, fps=10, codec="libx264", audio=False,
        bitrate="8000k", preset="slow",
    )
    logger.log_video("Eval/episode_video", base_path, commit=False)

    attn_base = f"{video_dir}/eval_attention.mp4"
    make_attention_video(frames, attn_data, filename=attn_base, fps=10)
    for suffix in ("agent0", "agent1", "combined"):
        p = f"{video_dir}/eval_attention_{suffix}.mp4"
        if os.path.exists(p):
            logger.log_video(f"Eval/attention_{suffix}", p, commit=False)
