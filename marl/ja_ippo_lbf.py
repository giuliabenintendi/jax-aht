"""JA-IPPO for LBF.

Per-fruit attention pooling, cross-entropy aux loss against partner's
previous-step argmax fruit, distance-progress shaping toward partner-attended
fruit. Fresh implementation for LBF specifically; parallel to ja_ippo.py
(which serves the OP-corrected card-game JA path). No Other-Play. Dynamic
fruit masks computed each step from current uneaten-fruit positions.

v1 ships: aux loss + r_shape. r_self, partner-attn-in-obs, soft target, and
eaten-fruit aux masking edge cases are deferred to v2.
"""
from __future__ import annotations

import os
from typing import NamedTuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from agents.initialize_agents import _get_image_dims, initialize_ja_image_agent
from agents.ja_actor_critic import _compute_resnet_output_dims
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
from marl.ppo_utils import _create_minibatches, batchify, unbatchify


JA_LBF_SCALAR_KEYS = list(IMAGE_IPPO_SCALAR_KEYS) + [
    ("aux_partner_argmax_loss", "Losses"),
]


class TransitionJA(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray  # = env reward + r_shape_coef * r_shape
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    # JA-specific fields, all per-actor. Fruit slot ordering is lex by (row, col).
    partner_prev_fruit_attn: jnp.ndarray  # (num_actors, N) float — per-fruit attention at t-1; aux target + obs feed
    partner_prev_argmax: jnp.ndarray    # (num_actors,) int32 — argmax fruit slot, drives r_shape
    partner_prev_valid: jnp.ndarray     # (num_actors,) bool — mask aux on invalid steps
    food_pos: jnp.ndarray               # (num_actors, N, 2) int — LEX SORTED, tiled per actor
    food_eaten: jnp.ndarray             # (num_actors, N) bool — LEX SORTED, tiled per actor


def _per_fruit_attn(
    attn_2d, food_pos, food_eaten,
    tile_size, feat_h, feat_w, img_h, img_w,
):
    """Point-gather attention at each fruit's centre feature cell.

    Map each fruit's tile centre pixel to a single (fr, fc) feature cell and
    take attn_2d at that cell. Eaten fruits get zero. Renormalise over
    on-fruit mass.

    Avoids materialising the (..., N, feat_h, feat_w) dense mask tensor used
    in earlier versions — both the memory and the 5-D einsum that XLA was
    routing into a cuDNN kernel that failed at larger batch sizes.

    attn_2d:    (..., feat_h, feat_w)        spatial attention
    food_pos:   (..., N, 2)  int             grid positions of each fruit
    food_eaten: (..., N)     bool

    Returns:
        per_fruit_norm: (..., N) — distribution over fruits given on-mass
        on_mass:        (...,)   — total attention mass that landed on fruits
    """
    centre_r = food_pos[..., 0] * tile_size + tile_size // 2  # (..., N)
    centre_c = food_pos[..., 1] * tile_size + tile_size // 2
    fr = jnp.clip(centre_r * feat_h // img_h, 0, feat_h - 1).astype(jnp.int32)
    fc = jnp.clip(centre_c * feat_w // img_w, 0, feat_w - 1).astype(jnp.int32)

    # Flatten spatial dims and gather. attn_2d shape (..., feat_h, feat_w)
    # → (..., feat_h*feat_w). Same leading dims as flat_idx so take_along_axis works.
    attn_flat = attn_2d.reshape(*attn_2d.shape[:-2], feat_h * feat_w)
    flat_idx = fr * feat_w + fc                                 # (..., N)
    per_fruit = jnp.take_along_axis(attn_flat, flat_idx, axis=-1)

    alive = 1.0 - food_eaten.astype(jnp.float32)
    per_fruit = per_fruit * alive
    on_mass = per_fruit.sum(axis=-1)
    per_fruit_norm = per_fruit / (on_mass[..., None] + 1e-8)
    return per_fruit_norm, on_mass


def _lex_sort_food(food_pos, food_eaten):
    """Sort fruits by lex (row, col) so slot k = k-th fruit in reading order.

    food_pos: (num_envs, N, 2); food_eaten: (num_envs, N).
    Returns the same shapes, reordered per env.
    """
    def _single(pos, eaten):
        # lexsort orders by the LAST key as primary -> row primary, col secondary.
        idx = jnp.lexsort((pos[:, 1], pos[:, 0]))
        return pos[idx], eaten[idx]
    return jax.vmap(_single)(food_pos, food_eaten)


def _attention_2d(attn_map):
    """Return spatial attention with shape (..., feat_h, feat_w)."""
    if attn_map.ndim == 4:
        return attn_map
    raise ValueError(f"Unexpected attention map rank: {attn_map.ndim}")


def _swap_partner(x, num_agents):
    """Swap the agent block in an actor-ordered tensor.

    Actor order: [agent_0 envs, agent_1 envs, ...]. For 2 agents, this swaps
    halves so position i is now occupied by the partner's value.
    """
    if num_agents != 2:
        raise NotImplementedError("ja_ippo_lbf assumes 2 agents (parameter-shared).")
    half = x.shape[0] // 2
    return jnp.concatenate([x[half:], x[:half]], axis=0)


def _compute_last_value_ja(policy, params, last_obs_batch, last_done_batch, last_avail_batch, hstate, num_actors):
    """JA-aware variant of compute_last_value: unpacks the 5-tuple from get_action_value_policy."""
    _, last_val, _, _, _ = policy.get_action_value_policy(
        params=params,
        obs=last_obs_batch.reshape(1, num_actors, -1),
        done=last_done_batch.reshape(1, num_actors),
        avail_actions=last_avail_batch.reshape(1, num_actors, -1),
        hstate=hstate,
        rng=jax.random.PRNGKey(0),
    )
    return last_val.squeeze()


def _agent_positions_from_log_state(log_state):
    """Extract (num_envs, num_agents, 2) agent positions from LogWrapper-wrapped state."""
    return log_state.env_state.env_state.agents.position


def _food_state_from_log_state(log_state):
    """Extract (food_pos, food_eaten) from LogWrapper-wrapped state.

    food_pos: (num_envs, N, 2); food_eaten: (num_envs, N).
    """
    food = log_state.env_state.env_state.food_items
    return food.position, food.eaten


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
    img_h, img_w, _ = _get_image_dims(env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=config.get("CONV_STRIDE", 2),
        kernel_size=config.get("CONV_KERNEL_SIZE", 3),
        padding=config.get("CONV_PADDING", "SAME"),
        num_blocks=config.get("CONV_NUM_BLOCKS", 4),
    )
    inner = env._env if hasattr(env, "_env") else env
    tile_size = inner.tile_size
    num_fruits = inner._num_food

    aux_coef = float(config.get("JA_AUX_PARTNER_ARGMAX_COEF", 0.0))
    r_shape_coef = float(config.get("JA_FRUIT_R_SHAPE_COEF", 0.0))
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
        f"aux_coef={aux_coef} r_shape_coef={r_shape_coef} "
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
        # Used as: scalar suffix to obs (online), soft CE aux target, and r_shape
        # argmax target. Reset to uniform on episode boundaries.
        init_partner_fruit_attn = (
            jnp.ones((num_actors, num_fruits), dtype=jnp.float32) / float(num_fruits)
        )
        init_partner_argmax = jnp.zeros((num_actors,), dtype=jnp.int32)
        init_partner_valid = jnp.zeros((num_actors,), dtype=bool)

        runner_state = (
            train_state, env_state, obsv, init_done, init_hstate, _rng,
            init_partner_fruit_attn, init_partner_argmax, init_partner_valid,
        )
        return runner_state, policy

    def make_step_fn(policy):
        @jax.jit
        def step_fn(runner_state, update_steps):
            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, last_done, hstate, rng,
                 prev_partner_fruit_attn, prev_partner_argmax, prev_partner_valid) = runner_state

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
                attn_2d = _attention_2d(attn_map).squeeze(0)

                # Pre-step food state (LBF state shared by both agents in each env).
                # Lex-sort fruits so slot k is always the k-th fruit in (row, col)
                # reading order. Agent can identify each slot's fruit from its image.
                food_pos_env_raw, food_eaten_env_raw = _food_state_from_log_state(env_state)
                food_pos_env, food_eaten_env = _lex_sort_food(food_pos_env_raw, food_eaten_env_raw)
                food_pos_actors = _tile_to_actors(food_pos_env)         # (num_actors, N, 2)
                food_eaten_actors = _tile_to_actors(food_eaten_env)     # (num_actors, N)

                per_fruit_norm, _on_mass = _per_fruit_attn(
                    attn_2d, food_pos_actors, food_eaten_actors,
                    tile_size, feat_h, feat_w, img_h, img_w,
                )
                own_argmax = jnp.argmax(per_fruit_norm, axis=-1).astype(jnp.int32)  # (num_actors,)

                # --- Pre-step positions (per actor) ---
                # pos_env shape: (num_envs, num_agents, 2) -> swapaxes -> (num_agents, num_envs, 2) -> (num_actors, 2)
                pos_pre = _agent_positions_from_log_state(env_state)
                pos_pre_actors = jnp.swapaxes(pos_pre, 0, 1).reshape(num_actors, 2)

                # --- Step env ---
                env_act = unbatchify(action, env.agents, num_envs, env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                new_obs, new_env_state, reward, new_done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0),
                )(rng_step, env_state, env_act)

                # --- Distance progress toward partner's previous-argmax fruit ---
                pos_post = _agent_positions_from_log_state(new_env_state)
                pos_post_actors = jnp.swapaxes(pos_post, 0, 1).reshape(num_actors, 2)

                actor_idx = jnp.arange(num_actors)
                target_fruit_pos = food_pos_actors[actor_idx, prev_partner_argmax]  # (num_actors, 2)

                d_pre = jnp.sum(jnp.abs(pos_pre_actors - target_fruit_pos), axis=-1).astype(jnp.float32)
                d_post = jnp.sum(jnp.abs(pos_post_actors - target_fruit_pos), axis=-1).astype(jnp.float32)
                r_shape = jnp.where(prev_partner_valid, d_pre - d_post, 0.0)

                env_reward = batchify(reward, env.agents, num_actors).squeeze()
                shaped_reward = env_reward + r_shape_coef * r_shape

                done_actors = batchify(new_done, env.agents, num_actors).squeeze().astype(bool)

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
                    partner_prev_argmax=prev_partner_argmax,
                    partner_prev_valid=prev_partner_valid,
                    food_pos=food_pos_actors.astype(jnp.int32),
                    food_eaten=food_eaten_actors,
                )

                # --- Update partner-attention state for next step ---
                # Swap own per-fruit vector across agent halves so each agent gets
                # its partner's per-fruit attention. Reset to uniform on done.
                swapped_fruit_attn = _swap_partner(per_fruit_norm, num_agents)
                uniform_fruit = jnp.ones((num_fruits,), dtype=jnp.float32) / float(num_fruits)
                new_partner_fruit_attn = jnp.where(
                    done_actors[:, None], uniform_fruit[None], swapped_fruit_attn,
                )
                new_partner_argmax = _swap_partner(own_argmax, num_agents).astype(jnp.int32)
                new_partner_valid = ~done_actors

                runner_state = (
                    train_state, new_env_state, new_obs, new_done, new_hstate, rng,
                    new_partner_fruit_attn, new_partner_argmax, new_partner_valid,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, rollout_length,
            )

            (train_state, env_state, last_obs, last_done, hstate, rng,
             prev_partner_fruit_attn, prev_partner_argmax, prev_partner_valid) = runner_state

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
            last_val = _compute_last_value_ja(
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

            runner_state = (
                train_state, env_state, last_obs, last_done, hstate, rng,
                prev_partner_fruit_attn, prev_partner_argmax, prev_partner_valid,
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

                log_prob = pi.log_prob(traj_batch.action)
                value_pred_clipped = traj_batch.value + (
                    value - traj_batch.value
                ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                ratio = jnp.exp(log_prob - traj_batch.log_prob)
                gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae_norm
                loss_actor2 = jnp.clip(
                    ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"],
                ) * gae_norm
                policy_loss = -jnp.minimum(loss_actor1, loss_actor2).mean()
                entropy = pi.entropy().mean()

                # --- Aux loss: per-fruit SOFT cross-entropy ---
                # Agent's per-fruit attention via point-gather of its spatial
                # attention at each (lex-sorted) fruit's centre cell. Partner's
                # per-fruit attention vector is already stored in the trajectory.
                # Gradient on agent's attn_2d flows only through N fruit cells.
                if aux_active:
                    attn_2d = _attention_2d(attn_map_apply)            # (T, A, fh, fw)
                    agent_per_fruit, agent_on_mass = _per_fruit_attn(
                        attn_2d, traj_batch.food_pos, traj_batch.food_eaten,
                        tile_size, feat_h, feat_w, img_h, img_w,
                    )                                                  # (T, A, N), (T, A)
                    target_soft = jax.lax.stop_gradient(
                        traj_batch.partner_prev_fruit_attn,
                    )                                                  # (T, A, N)

                    log_probs = jnp.log(agent_per_fruit + 1e-8)        # (T, A, N)
                    nll_per_step = -(target_soft * log_probs).sum(axis=-1)  # (T, A)
                    nll_flat = nll_per_step.reshape(-1)
                    aux_weight = (
                        traj_batch.partner_prev_valid.reshape(-1).astype(jnp.float32)
                        * agent_on_mass.reshape(-1)
                    )
                    aux_weight = jax.lax.stop_gradient(aux_weight)
                    aux_denom = jnp.maximum(aux_weight.sum(), 1e-8)
                    aux_loss = (nll_flat * aux_weight).sum() / aux_denom
                else:
                    aux_loss = jnp.float32(0.0)

                total_loss = (
                    policy_loss
                    + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
                    + aux_coef * aux_loss
                )
                return total_loss, (value_loss, policy_loss, entropy, aux_loss)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            (total_loss, (value_loss, policy_loss, entropy, aux_loss)), grads = grad_fn(
                train_state.params, traj_batch, advantages, targets,
            )
            grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
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

        seed_outputs.append({
            "final_params": runner_state[0].params,
            "metrics": stacked_metrics,
            "checkpoints": stacked_ckpts,
            "final_ckpt_idx": len(checkpoints),
        })

    print("[ja_ippo_lbf] Training complete.", flush=True)

    out = jax.tree.map(lambda *xs: jnp.stack(xs), *seed_outputs)

    try:
        _log_eval_video(algorithm_config, env, out, logger)
    except Exception as e:
        # The eval-video helper reconstructs the policy from the yaml config and
        # calls run_episode_with_states, which doesn't know about the per-fruit
        # scalar suffix we append during training. A clean fix requires augmenting
        # obs at eval too; for now we just log the error so training output is
        # preserved.
        print(f"[ja_ippo_lbf] WARN: eval video failed ({e}); continuing.", flush=True)

    # XP evaluation — only meaningful with multiple seeds.
    if num_seeds > 1:
        try:
            _log_xp_eval(algorithm_config, env, out, logger)
        except Exception as e:
            print(f"[ja_ippo_lbf] WARN: XP eval failed ({e}); continuing.", flush=True)

    report_basic_training_outputs(
        config, out, logger,
        scalar_keys=JA_LBF_SCALAR_KEYS,
        print_prefix="ja_ippo_lbf",
    )
    return out


def _jsd_spatial(p, q):
    """Per-row Jensen-Shannon divergence between two (..., H, W) attention maps.

    Returns a tensor with shape (...,) — the JSD averaged over spatial cells.
    Inputs are assumed to be probability distributions (rows summing to 1
    along (-2, -1)).
    """
    p_flat = p.reshape(p.shape[:-2] + (-1,))
    q_flat = q.reshape(q.shape[:-2] + (-1,))
    m = 0.5 * (p_flat + q_flat)
    eps = 1e-12

    def _kl(a, b):
        return (a * (jnp.log(a + eps) - jnp.log(b + eps))).sum(axis=-1)

    return 0.5 * _kl(p_flat, m) + 0.5 * _kl(q_flat, m)


def _log_xp_eval(algorithm_config, env, out, logger):
    """Greedy cross-play eval between training seeds.

    Builds NxN matrices for:
      - mean per-pair episode return (the unshaped task reward)
      - mean per-pair JSD between agent-0 and agent-1 spatial attention maps
    Aggregates SP (diagonal) and XP (off-diagonal) scalars, plus the full
    matrices as wandb tables/images.

    Skipped automatically when NUM_SEEDS < 2 (caller already gates this).
    """
    import wandb

    stacked_params = out["final_params"]
    num_seeds = jax.tree.leaves(stacked_params)[0].shape[0]

    inner = env._env if hasattr(env, "_env") else env
    tile_size = inner.tile_size
    num_fruits = inner._num_food
    img_h, img_w, _ = _get_image_dims(env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=algorithm_config.get("CONV_STRIDE", 2),
        kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
        padding=algorithm_config.get("CONV_PADDING", "SAME"),
        num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
    )
    partner_feed_active = bool(algorithm_config.get("JA_FRUIT_PARTNER_FEED", True))

    # Inject the entity-feed dim before init so the eval policy's input shape
    # matches the trained params.
    cfg = dict(algorithm_config)
    cfg["JA_ENTITY_FEED_DIM"] = num_fruits if partner_feed_active else 0
    policy, _ = initialize_ja_image_agent(cfg, env, jax.random.PRNGKey(0))

    # Eval knobs
    n_envs = int(algorithm_config.get("XP_NUM_ENVS", 16))
    max_steps = int(algorithm_config.get("ROLLOUT_LENGTH", 128))
    n_actors = n_envs * 2  # 2 agents

    def _augment(obs_batch_2d, partner_fruit_attn):
        if not partner_feed_active:
            return obs_batch_2d
        return jnp.concatenate([obs_batch_2d, partner_fruit_attn], axis=-1)

    def _eval_pair(rng_pair, params_0, params_1):
        """Run n_envs episodes for max_steps with agent_0=params_0, agent_1=params_1.

        Returns: (mean_return, mean_jsd) — both scalars.
        """
        rng_pair, reset_rng = jax.random.split(rng_pair)
        reset_rngs = jax.random.split(reset_rng, n_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
        hstate = policy.init_hstate(n_actors)
        done = {k: jnp.zeros((n_envs,), dtype=bool) for k in env.agents + ["__all__"]}
        partner_attn = (
            jnp.ones((n_actors, num_fruits), dtype=jnp.float32) / float(num_fruits)
        )

        # Accumulators
        ret_sum = jnp.zeros((n_actors,), dtype=jnp.float32)
        ret_count = jnp.zeros((n_actors,), dtype=jnp.float32)
        jsd_sum = jnp.zeros((), dtype=jnp.float32)
        jsd_count = jnp.zeros((), dtype=jnp.float32)

        def _xp_step(carry, _):
            (env_state, last_obs, last_done, hstate, partner_attn, rng,
             ret_sum, ret_count, jsd_sum, jsd_count) = carry

            rng, act_rng = jax.random.split(rng)

            obs_batch = batchify(last_obs, env.agents, n_actors)
            done_batch = batchify(last_done, env.agents, n_actors)
            obs_aug = _augment(obs_batch, partner_attn)

            avail = jax.vmap(env.get_avail_actions)(env_state)
            avail_batch = batchify(avail, env.agents, n_actors).astype(jnp.float32)

            # Split per agent
            obs0 = obs_aug[:n_envs].reshape(1, n_envs, -1)
            obs1 = obs_aug[n_envs:].reshape(1, n_envs, -1)
            done0 = done_batch[:n_envs].reshape(1, n_envs)
            done1 = done_batch[n_envs:].reshape(1, n_envs)
            avail0 = avail_batch[:n_envs].reshape(1, n_envs, -1)
            avail1 = avail_batch[n_envs:].reshape(1, n_envs, -1)
            hs0 = hstate[:, :n_envs]
            hs1 = hstate[:, n_envs:]

            act0, hs0_new, attn0 = policy.get_action_and_attention(
                params=params_0, obs=obs0, done=done0,
                avail_actions=avail0, hstate=hs0, rng=act_rng, greedy=True,
            )
            act1, hs1_new, attn1 = policy.get_action_and_attention(
                params=params_1, obs=obs1, done=done1,
                avail_actions=avail1, hstate=hs1, rng=act_rng, greedy=True,
            )

            new_hstate = jnp.concatenate([hs0_new, hs1_new], axis=1)

            # Pool per-fruit attention for next step's partner-feed
            attn2d_0 = _attention_2d(attn0).squeeze(0)      # (n_envs, fh, fw)
            attn2d_1 = _attention_2d(attn1).squeeze(0)
            attn_all = jnp.concatenate([attn2d_0, attn2d_1], axis=0)  # (n_actors, fh, fw)

            food_pos_env_raw, food_eaten_env_raw = _food_state_from_log_state(env_state)
            food_pos_env, food_eaten_env = _lex_sort_food(
                food_pos_env_raw, food_eaten_env_raw,
            )
            reps = (2,) + (1,) * (food_pos_env.ndim - 1)
            food_pos_actors = jnp.tile(food_pos_env, reps)
            food_eaten_actors = jnp.tile(food_eaten_env, (2,) + (1,) * (food_eaten_env.ndim - 1))

            per_fruit, _ = _per_fruit_attn(
                attn_all, food_pos_actors, food_eaten_actors,
                tile_size, feat_h, feat_w, img_h, img_w,
            )

            # Step env
            actions = jnp.concatenate([act0.squeeze(0), act1.squeeze(0)], axis=0)
            env_act = unbatchify(actions, env.agents, n_envs, env.num_agents)
            env_act = {k: v.flatten() for k, v in env_act.items()}
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, n_envs)
            new_obs, new_env_state, reward, new_done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0),
            )(rng_step, env_state, env_act)

            # Update partner per-fruit for next step (swap, reset on done)
            done_actors = batchify(new_done, env.agents, n_actors).squeeze().astype(bool)
            new_partner_attn = _swap_partner(per_fruit, env.num_agents)
            uniform_fruit = jnp.ones((num_fruits,), dtype=jnp.float32) / float(num_fruits)
            new_partner_attn = jnp.where(
                done_actors[:, None], uniform_fruit[None], new_partner_attn,
            )

            # Accumulate episode return at episode end. LogWrapper exposes
            # returned_episode (bool) and returned_episode_returns (float) as
            # (num_envs, num_agents) arrays. Convert to actor order
            # [agent_0_envs..., agent_1_envs...] via swap+reshape.
            returned_arr = info["returned_episode"].astype(jnp.float32)        # (num_envs, num_agents)
            ep_returns_arr = info["returned_episode_returns"].astype(jnp.float32)
            returned_actors = jnp.swapaxes(returned_arr, 0, 1).reshape(n_actors)
            ep_ret_actors = jnp.swapaxes(ep_returns_arr, 0, 1).reshape(n_actors)
            ret_sum = ret_sum + ep_ret_actors * returned_actors
            ret_count = ret_count + returned_actors

            # JSD between agent-0 and agent-1 attention this step.
            jsd_step = _jsd_spatial(attn2d_0, attn2d_1)  # (n_envs,)
            jsd_sum = jsd_sum + jsd_step.sum()
            jsd_count = jsd_count + jnp.float32(n_envs)

            new_carry = (
                new_env_state, new_obs, new_done, new_hstate, new_partner_attn, rng,
                ret_sum, ret_count, jsd_sum, jsd_count,
            )
            return new_carry, None

        carry = (
            env_state, obsv, done, hstate, partner_attn, rng_pair,
            ret_sum, ret_count, jsd_sum, jsd_count,
        )
        (carry_final, _) = jax.lax.scan(_xp_step, carry, None, length=max_steps)
        (_, _, _, _, _, _, ret_sum_f, ret_count_f, jsd_sum_f, jsd_count_f) = carry_final

        mean_return = ret_sum_f.sum() / jnp.maximum(ret_count_f.sum(), 1.0)
        mean_jsd = jsd_sum_f / jnp.maximum(jsd_count_f, 1.0)
        return mean_return, mean_jsd

    eval_pair_jit = jax.jit(_eval_pair)

    print(f"[ja_ippo_lbf] XP eval: {num_seeds}x{num_seeds} pairs, {n_envs} envs each", flush=True)
    ret_matrix = np.zeros((num_seeds, num_seeds), dtype=np.float32)
    jsd_matrix = np.zeros((num_seeds, num_seeds), dtype=np.float32)

    seed_params = [jax.tree.map(lambda x: x[i], stacked_params) for i in range(num_seeds)]
    for i in range(num_seeds):
        for j in range(num_seeds):
            mean_ret, mean_jsd = eval_pair_jit(
                jax.random.PRNGKey(1000 * (i + 1) + (j + 1)),
                seed_params[i], seed_params[j],
            )
            ret_matrix[i, j] = float(mean_ret)
            jsd_matrix[i, j] = float(mean_jsd)
            tag = "SP" if i == j else "XP"
            print(f"  [{tag}] {i}x{j}: return={ret_matrix[i,j]:.4f} jsd={jsd_matrix[i,j]:.4f}", flush=True)

    sp_returns = np.diag(ret_matrix)
    xp_mask = ~np.eye(num_seeds, dtype=bool)
    xp_returns = ret_matrix[xp_mask]
    sp_jsd = np.diag(jsd_matrix)
    xp_jsd = jsd_matrix[xp_mask]

    summary = {
        "XP/return/sp_mean": float(sp_returns.mean()),
        "XP/return/xp_mean": float(xp_returns.mean()),
        "XP/return/sp_minus_xp": float(sp_returns.mean() - xp_returns.mean()),
        "XP/jsd/sp_mean": float(sp_jsd.mean()),
        "XP/jsd/xp_mean": float(xp_jsd.mean()),
    }
    print(f"[ja_ippo_lbf] XP summary: {summary}", flush=True)

    # wandb logging — matrices only (as tables). Scalar SP/XP summaries print
    # to stdout above; logging them via run.log() creates spurious 1-point
    # line plots, so we omit them here.
    run = getattr(logger, "run", None)
    if run is not None:
        ret_table = wandb.Table(
            columns=["seed_i"] + [f"seed_{j}" for j in range(num_seeds)],
            data=[[i] + ret_matrix[i].tolist() for i in range(num_seeds)],
        )
        jsd_table = wandb.Table(
            columns=["seed_i"] + [f"seed_{j}" for j in range(num_seeds)],
            data=[[i] + jsd_matrix[i].tolist() for i in range(num_seeds)],
        )
        run.log({
            "XP/return_matrix": ret_table,
            "XP/jsd_matrix": jsd_table,
        })


def _log_eval_video(algorithm_config, env, out, logger):
    """Run one eval episode with final params (seed 0), log video to wandb."""
    from evaluation.vis_episodes import run_episode_with_states

    rng = jax.random.PRNGKey(0)
    policy, _ = initialize_ja_image_agent(algorithm_config, env, rng)
    final_params = jax.tree.map(lambda x: x[0], out["final_params"])
    inner_env = env._env

    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    ep_states, _, _ = run_episode_with_states(
        jax.random.PRNGKey(42), inner_env, final_params, policy,
        final_params, policy, max_steps,
    )
    print(f"[ja_ippo_lbf] Eval episode: {len(ep_states)} frames collected", flush=True)

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    video_dir = f"{savedir}/videos"
    os.makedirs(video_dir, exist_ok=True)
    video_path = f"{video_dir}/eval_final.mp4"

    frames = _render_lbf_eval_frames(inner_env, ep_states)
    from moviepy import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=10)
    clip.write_videofile(
        video_path, fps=10, codec="libx264", audio=False,
        bitrate="8000k", preset="slow",
    )
    logger.log_video("Eval/episode_video", video_path, commit=False)
