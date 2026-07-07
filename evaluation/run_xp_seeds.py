"""Cross-play evaluation for multi-seed training runs.

Loads best_params (shape: num_seeds, ...) from a single checkpoint when
available, falling back to final_params for older checkpoints. Builds the NxN
cross-play matrix, and reports SP/XP with proper SEM following the pairing
scheme from the ZSC literature.

Reports the cross-play game-score matrix. For non-card-game envs also
reports a raw-spatial JSD matrix between attention maps; this is skipped
for card-game runs because Other-Play recolouring/shuffling makes raw
spatial JSD meaningless across agents.

Usage:
    uv run python -m evaluation.run_xp_seeds \
        --task overcooked-v1-image/cramped_room \
        --checkpoint results/.../saved_train_run
"""
import argparse
import csv
import itertools
import os
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from agents.initialize_agents import initialize_ja_image_agent
from agents.ja_utils import jsd_divergence
from agents.lbf.op_equivariance import (
    feat_transform_table,
    read_op_elems,
    transform_feat_maps,
)
from common.plot_utils import get_metric_names
from common.save_load_utils import load_train_run
from common.tree_utils import tree_stack
from envs import make_env
from envs.card_game.rendering import NUM_CARDS
from envs.log_wrapper import LogWrapper
from evaluation.card_game.action_distributions import (
    generate_action_distribution_artifacts,
)
from evaluation.card_game.xp_stats import xp_mean_se
from marl.eval_card_game import _log_card_game_xp_videos


EVAL_SEED = 34957
NUM_EVAL_EPISODES = 256
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs", "task")
ALGO_BASE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "marl", "configs", "algorithm", "ja_ippo", "_base_.yaml"
)


def _select_xp_params(run_data, *, prefer_best: bool = True):
    """Return params for XP eval, preferring best checkpoint params when present."""
    if prefer_best and "best_params" in run_data:
        return run_data["best_params"], "best_params"
    if "final_params" in run_data:
        return run_data["final_params"], "final_params"
    raise KeyError(
        "Neither 'best_params' nor 'final_params' found in checkpoint; "
        f"keys: {list(run_data.keys())}"
    )


def load_task_config(task_name: str) -> dict:
    config_path = os.path.join(CONFIGS_DIR, f"{task_name}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_algo_config() -> dict:
    with open(ALGO_BASE_CONFIG) as f:
        return yaml.safe_load(f)


def _get_card_game_position_perm(state, agent_name: str):
    """Return view-position -> GT-card map for this agent.

    With position OP this is the per-agent shuffle. Without position OP, the
    base card game can still have a shared random card_permutation (notably in
    recolour-only OP), so use that shared layout before falling back to identity.
    """
    s = state
    while s is not None:
        if hasattr(s, "per_agent_perm"):
            return s.per_agent_perm[agent_name]
        s = getattr(s, "env_state", None)
    s = state
    while s is not None:
        if hasattr(s, "card_permutation"):
            return s.card_permutation
        s = getattr(s, "env_state", None)
    return jnp.arange(NUM_CARDS, dtype=jnp.int32)


def _get_card_game_inv_recolouring(state, agent_name: str):
    """Return per-agent OP inverse-recolouring perm, or identity when off.

    Used to map an agent's raw policy action (in its recoloured colour
    space) back to GT colour space before comparing with the partner's
    action. This matches what `CardGameRecolouringWrapper._invert_actions`
    does to actions before they reach the inner env.
    """
    s = state
    while s is not None:
        if hasattr(s, "per_agent_inv_recolouring"):
            return s.per_agent_inv_recolouring[agent_name]
        s = getattr(s, "env_state", None)
    return jnp.arange(NUM_CARDS, dtype=jnp.int32)


def _attn_to_gt(attn_0_2d, attn_1_2d, env_state, agents):
    """Un-mirror each agent's (h, w) spatial attention into the GT frame.

    Under LBF Other-Play each agent sees a mirror-transformed observation, so its
    attention lives in that agent's mirror frame; JSD between the two raw maps then
    measures the frame mismatch, not real disagreement. `read_op_elems` returns the
    per-agent V4 element (None when OP is off, or for the card game's recolouring
    wrapper, which uses `per_agent_perm` not `per_agent_elem`), and the self-inverse
    transform brings both maps into the shared GT frame before the JSD is taken.
    """
    op_elems = read_op_elems(env_state, agents)
    if op_elems is None:
        return attn_0_2d, attn_1_2d
    h, w = attn_0_2d.shape[-2], attn_0_2d.shape[-1]
    table = feat_transform_table(h, w)
    g0 = jnp.reshape(op_elems[agents[0]], (1,))
    g1 = jnp.reshape(op_elems[agents[1]], (1,))
    a0 = transform_feat_maps(attn_0_2d.reshape(1, h, w), g0, table)[0]
    a1 = transform_feat_maps(attn_1_2d.reshape(1, h, w), g1, table)[0]
    return a0, a1


def run_single_episode_with_jsd(rng, env, agent_0_param, agent_0_policy,
                                agent_1_param, agent_1_policy,
                                max_episode_steps, action_sizes,
                                feed_attn_dims=None, ja_card_masks=None,
                                greedy_eval=True, partner_feed_dim=5,
                                lbf_ctx=None, feed_mask_fn=None):
    """Run one eval episode, returning LogWrapper info + mean JSD between attention maps.

    Args:
        feed_attn_dims: if not None, (img_h, img_w, feat_h, feat_w) for obs augmentation
            with the other agent's previous attention map (4th channel).
        feed_mask_fn: optional visibility hook `state -> (mask_0, mask_1)` applied to
            the feed channel per receiver (train-time gating parity, e.g. ocv2
            JA_VISIBILITY_GATING).
        lbf_ctx: if not None, dict with num_fruits/tile_size/feat_h/feat_w/img_h/img_w
            describing the LBF env so each agent's obs is augmented with the
            partner's previous per-fruit attention vector (length num_fruits,
            lex-sorted) — mirrors the training-time augmentation in
            ja_ippo_lbf.make_train.
    """
    from agents.ja_utils import augment_obs_for_eval
    from agents.lbf.ja_lbf_attention import (
        food_state_from_log_state,
        per_fruit_attn,
    )

    _lbf = lbf_ctx is not None
    if _lbf:
        partner_feed_dim = int(lbf_ctx["num_fruits"])

    def _lbf_per_fruit_single(attn_2d, env_state):
        """Compute one agent's per-fruit attention (length num_fruits, lex-sorted)
        from its spatial attention map and the current env food state."""
        food_pos, food_eaten = food_state_from_log_state(env_state)
        idx = jnp.lexsort((food_pos[:, 1], food_pos[:, 0]))
        food_pos = food_pos[idx]
        food_eaten = food_eaten[idx]
        per_fruit, _ = per_fruit_attn(
            attn_2d, food_pos, food_eaten,
            lbf_ctx["tile_size"], lbf_ctx["feat_h"], lbf_ctx["feat_w"],
            lbf_ctx["img_h"], lbf_ctx["img_w"],
        )
        return per_fruit

    def _call_attn(policy, params, obs, done, avail, hstate, rng,
                   pe_a=None, pe_c=None, prev_rew=None, prev_act=None):
        """Wrapper returning (act, hstate, attn, own_a, own_c) uniformly."""
        act, hs, attn = policy.get_action_and_attention(
            params=params, obs=obs, done=done, avail_actions=avail,
            hstate=hstate, rng=rng, greedy=greedy_eval,
            prev_reward=prev_rew, prev_action=prev_act,
        )
        z = jnp.zeros((1, 1, 1))  # dummy
        return act, hs, attn, z, z

    rng, reset_rng = jax.random.split(rng)
    init_obs, init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}

    init_hstate_0 = agent_0_policy.init_hstate(1, aux_info={"agent_id": 0})
    init_hstate_1 = agent_1_policy.init_hstate(1, aux_info={"agent_id": 1})
    use_prev_io = (
        getattr(agent_0_policy, "uses_prev_reward_action", False)
        and getattr(agent_1_policy, "uses_prev_reward_action", False)
    )
    if use_prev_io:
        prev_reward_0 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_reward_1 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_action_0 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_action_1 = jnp.zeros((1, 1), dtype=jnp.float32)

    _ja_card = ja_card_masks is not None
    # LBF and card both use the prev_pca_X scalar suffix; initial value differs.
    if _ja_card:
        prev_pca_0 = jnp.zeros(partner_feed_dim)
        prev_pca_1 = jnp.zeros(partner_feed_dim)
    elif _lbf:
        prev_pca_0 = jnp.ones(partner_feed_dim) / float(partner_feed_dim)
        prev_pca_1 = jnp.ones(partner_feed_dim) / float(partner_feed_dim)

    # Initialize uniform attention maps for feed_other_attn
    if feed_attn_dims is not None:
        _img_h, _img_w, _feat_h, _feat_w = feed_attn_dims
        prev_attn_0 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)
        prev_attn_1 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)

    avail_actions = env.get_avail_actions(init_env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

    # First step
    rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)

    obs_0 = init_obs["agent_0"]
    obs_1 = init_obs["agent_1"]
    if feed_attn_dims is not None:
        _fm_0 = _fm_1 = None
        if feed_mask_fn is not None:
            _fm_0, _fm_1 = feed_mask_fn(init_env_state)
        obs_0 = augment_obs_for_eval(obs_0, prev_attn_1, _img_h, _img_w, mask=_fm_0)
        obs_1 = augment_obs_for_eval(obs_1, prev_attn_0, _img_h, _img_w, mask=_fm_1)
    if _ja_card or _lbf:
        obs_0 = jnp.concatenate([obs_0, prev_pca_0])
        obs_1 = jnp.concatenate([obs_1, prev_pca_1])

    act_0, hstate_0, attn_0, _, _ = _call_attn(
        agent_0_policy, agent_0_param,
        obs_0.reshape(1, 1, -1), init_done["agent_0"].reshape(1, 1),
        avail_actions_0, init_hstate_0, act0_rng,
        prev_rew=prev_reward_0 if use_prev_io else None,
        prev_act=prev_action_0 if use_prev_io else None,
    )
    act_0 = act_0.squeeze()

    act_1, hstate_1, attn_1, _, _ = _call_attn(
        agent_1_policy, agent_1_param,
        obs_1.reshape(1, 1, -1), init_done["agent_1"].reshape(1, 1),
        avail_actions_1, init_hstate_1, act1_rng,
        prev_rew=prev_reward_1 if use_prev_io else None,
        prev_act=prev_action_1 if use_prev_io else None,
    )
    act_1 = act_1.squeeze()

    attn_0_2d, attn_1_2d = _attn_to_gt(attn_0, attn_1, init_env_state, env.agents)
    attn_0_flat = attn_0_2d.reshape(-1)
    attn_1_flat = attn_1_2d.reshape(-1)
    attn_0_dist = attn_0_flat / (attn_0_flat.sum() + 1e-8)
    attn_1_dist = attn_1_flat / (attn_1_flat.sum() + 1e-8)
    h, w = attn_0_2d.shape[-2], attn_0_2d.shape[-1]
    step_jsd = jsd_divergence(attn_0_dist.reshape(h, w), attn_1_dist.reshape(h, w))
    jsd_sum = step_jsd
    jsd_count = jnp.array(1.0)

    # Per-step token-match buffer: stores at index t whether the two agents'
    # actions agreed at step t in GT colour space (deliberation messages and
    # the decision pick). Under CardGameRecolouringWrapper each agent emits
    # in its own recoloured space, so we apply the per-agent inverse-
    # recolouring before comparison — same transform the wrapper does
    # internally before reaching the inner env. Aggregated over episodes
    # this yields one (i, j) matrix per step.
    inv_0 = _get_card_game_inv_recolouring(init_env_state, "agent_0")
    inv_1 = _get_card_game_inv_recolouring(init_env_state, "agent_1")
    first_match = (inv_0[act_0] == inv_1[act_1]).astype(jnp.float32)
    match_per_step = jnp.zeros(max_episode_steps, dtype=jnp.float32).at[0].set(first_match)

    both_actions = [act_0, act_1]
    env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
    env_act_onehot = {k: jax.nn.one_hot(both_actions[i], action_sizes[k])
                      for i, k in enumerate(env.agents)}
    obs, env_state, reward, done, dummy_info = env.step(step_rng, init_env_state, env_act)
    if use_prev_io:
        prev_reward_0 = reward["agent_0"].reshape(1, 1).astype(jnp.float32)
        prev_reward_1 = reward["agent_1"].reshape(1, 1).astype(jnp.float32)
        prev_action_0 = act_0.reshape(1, 1).astype(jnp.float32)
        prev_action_1 = act_1.reshape(1, 1).astype(jnp.float32)

    # Include prev attention in carry for feed_other_attn
    if feed_attn_dims is not None:
        prev_attn_0 = attn_0.squeeze()
        prev_attn_1 = attn_1.squeeze()

    card_jsd_sum = jnp.float32(0.0)
    card_jsd_count = jnp.float32(0.0)
    if _ja_card:
        a0_sq = attn_0.squeeze()
        a1_sq = attn_1.squeeze()
        perm_0 = _get_card_game_position_perm(init_env_state, "agent_0")
        perm_1 = _get_card_game_position_perm(init_env_state, "agent_1")
        ca_0 = jnp.einsum("hw,chw->c", a0_sq, ja_card_masks)
        ca_1 = jnp.einsum("hw,chw->c", a1_sq, ja_card_masks)
        ph_0 = jnp.zeros(5).at[perm_0].set(ca_0)
        ph_1 = jnp.zeros(5).at[perm_1].set(ca_1)
        prev_pca_0 = ph_1[perm_0]
        prev_pca_1 = ph_0[perm_1]
        _m0 = ca_0.sum()
        _m1 = ca_1.sum()
        _q0 = ph_0 / (_m0 + 1e-8)
        _q1 = ph_1 / (_m1 + 1e-8)
        card_jsd_sum = jsd_divergence(_q0[None, :], _q1[None, :])
        card_jsd_count = jnp.float32(1.0)
    elif _lbf:
        # Compute each agent's per-fruit attention from its spatial map and the
        # pre-step env state (the state the agent actually attended to). Swap
        # so prev_pca_X (the obs suffix for agent X on the next step) holds the
        # partner's attention vector — matching the training-time wiring.
        per_fruit_0 = _lbf_per_fruit_single(attn_0.squeeze(), init_env_state)
        per_fruit_1 = _lbf_per_fruit_single(attn_1.squeeze(), init_env_state)
        prev_pca_0 = per_fruit_1
        prev_pca_1 = per_fruit_0

    ep_ts = 1
    # Placeholder partner-LSTM hiddens (carried through unchanged; not wired
    # into XP eval). Keep them in the carry so the take_step return shape
    # matches what scan expects.
    pe_a0 = pe_c0 = pe_a1 = pe_c1 = jnp.zeros((1,))
    init_carry = (ep_ts, env_state, obs, rng, done, reward, env_act_onehot,
                  hstate_0, hstate_1, dummy_info, jsd_sum, jsd_count,
                  prev_reward_0 if use_prev_io else None,
                  prev_reward_1 if use_prev_io else None,
                  prev_action_0 if use_prev_io else None,
                  prev_action_1 if use_prev_io else None,
                  attn_0.squeeze(), attn_1.squeeze(),
                  pe_a0, pe_c0, pe_a1, pe_c1,
                  prev_pca_0 if (_ja_card or _lbf) else jnp.zeros(5),
                  prev_pca_1 if (_ja_card or _lbf) else jnp.zeros(5),
                  card_jsd_sum, card_jsd_count,
                  match_per_step)

    def scan_step(carry, _):
        def take_step(carry_step):
            (ep_ts, env_state, obs, rng, done, reward, act_onehot,
             hstate_0, hstate_1, last_info, jsd_sum, jsd_count,
             prev_reward_0, prev_reward_1, prev_action_0, prev_action_1,
             prev_a0, prev_a1,
             pe_a0, pe_c0, pe_a1, pe_c1,
             prev_pca_0, prev_pca_1,
             card_jsd_sum, card_jsd_count,
             match_per_step) = carry_step

            avail_actions = env.get_avail_actions(env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)

            obs_0 = obs["agent_0"]
            obs_1 = obs["agent_1"]
            if feed_attn_dims is not None:
                _fm_0 = _fm_1 = None
                if feed_mask_fn is not None:
                    _fm_0, _fm_1 = feed_mask_fn(env_state)
                obs_0 = augment_obs_for_eval(obs_0, prev_a1, _img_h, _img_w, mask=_fm_0)
                obs_1 = augment_obs_for_eval(obs_1, prev_a0, _img_h, _img_w, mask=_fm_1)
            if _ja_card or _lbf:
                obs_0 = jnp.concatenate([obs_0, prev_pca_0])
                obs_1 = jnp.concatenate([obs_1, prev_pca_1])

            act_0, hstate_0_next, attn_0, _, _ = _call_attn(
                agent_0_policy, agent_0_param,
                obs_0.reshape(1, 1, -1), done["agent_0"].reshape(1, 1),
                avail_actions_0, hstate_0, act0_rng,
                prev_rew=prev_reward_0 if use_prev_io else None,
                prev_act=prev_action_0 if use_prev_io else None,
            )
            act_0 = act_0.squeeze()

            act_1, hstate_1_next, attn_1, _, _ = _call_attn(
                agent_1_policy, agent_1_param,
                obs_1.reshape(1, 1, -1), done["agent_1"].reshape(1, 1),
                avail_actions_1, hstate_1, act1_rng,
                prev_rew=prev_reward_1 if use_prev_io else None,
                prev_act=prev_action_1 if use_prev_io else None,
            )
            act_1 = act_1.squeeze()

            a0_2d, a1_2d = _attn_to_gt(attn_0, attn_1, env_state, env.agents)
            a0 = a0_2d.reshape(-1)
            a1 = a1_2d.reshape(-1)
            a0 = a0 / (a0.sum() + 1e-8)
            a1 = a1 / (a1.sum() + 1e-8)
            step_jsd = jsd_divergence(a0.reshape(h, w), a1.reshape(h, w))
            jsd_sum_next = jsd_sum + step_jsd
            jsd_count_next = jsd_count + 1.0

            both_actions = [act_0, act_1]
            env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
            env_act_onehot = {k: jax.nn.one_hot(both_actions[i], action_sizes[k])
                              for i, k in enumerate(env.agents)}
            obs_next, env_state_next, reward, done_next, info_next = env.step(step_rng, env_state, env_act)
            if use_prev_io:
                next_prev_reward_0 = reward["agent_0"].reshape(1, 1).astype(jnp.float32)
                next_prev_reward_1 = reward["agent_1"].reshape(1, 1).astype(jnp.float32)
                next_prev_action_0 = act_0.reshape(1, 1).astype(jnp.float32)
                next_prev_action_1 = act_1.reshape(1, 1).astype(jnp.float32)
            else:
                next_prev_reward_0 = next_prev_reward_1 = None
                next_prev_action_0 = next_prev_action_1 = None

            # Update JA card partner attention + accumulate card-level JSD.
            if _ja_card:
                p0 = _get_card_game_position_perm(env_state, "agent_0")
                p1 = _get_card_game_position_perm(env_state, "agent_1")
                a0_sq = attn_0.squeeze()
                a1_sq = attn_1.squeeze()
                ca0 = jnp.einsum("hw,chw->c", a0_sq, ja_card_masks)
                ca1 = jnp.einsum("hw,chw->c", a1_sq, ja_card_masks)
                ph0 = jnp.zeros(5).at[p0].set(ca0)
                ph1 = jnp.zeros(5).at[p1].set(ca1)
                next_pca_0 = ph1[p0]
                next_pca_1 = ph0[p1]
                m0 = ca0.sum()
                m1 = ca1.sum()
                q0 = ph0 / (m0 + 1e-8)
                q1 = ph1 / (m1 + 1e-8)
                step_card_jsd = jsd_divergence(q0[None, :], q1[None, :])
                card_jsd_sum_next = card_jsd_sum + step_card_jsd
                card_jsd_count_next = card_jsd_count + 1.0
            elif _lbf:
                # Use pre-step env_state to match attention against the food
                # state the agent actually attended to.
                pf_0 = _lbf_per_fruit_single(attn_0.squeeze(), env_state)
                pf_1 = _lbf_per_fruit_single(attn_1.squeeze(), env_state)
                # Reset partner feed to uniform on episode boundary (matches train).
                uniform_fruit = jnp.ones(partner_feed_dim) / float(partner_feed_dim)
                ep_over = done_next["__all__"].squeeze().astype(bool)
                next_pca_0 = jnp.where(ep_over, uniform_fruit, pf_1)
                next_pca_1 = jnp.where(ep_over, uniform_fruit, pf_0)
                card_jsd_sum_next = card_jsd_sum
                card_jsd_count_next = card_jsd_count
            else:
                next_pca_0 = jnp.zeros(partner_feed_dim)
                next_pca_1 = jnp.zeros(partner_feed_dim)
                card_jsd_sum_next = card_jsd_sum
                card_jsd_count_next = card_jsd_count

            inv_0_step = _get_card_game_inv_recolouring(env_state, "agent_0")
            inv_1_step = _get_card_game_inv_recolouring(env_state, "agent_1")
            gt_a0_step = inv_0_step[act_0]
            gt_a1_step = inv_1_step[act_1]
            step_match = (gt_a0_step == gt_a1_step).astype(jnp.float32)
            match_per_step_next = match_per_step.at[ep_ts].set(step_match)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, reward, env_act_onehot,
                    hstate_0_next, hstate_1_next, info_next, jsd_sum_next, jsd_count_next,
                    next_prev_reward_0, next_prev_reward_1, next_prev_action_0, next_prev_action_1,
                    attn_0.squeeze(), attn_1.squeeze(),
                    pe_a0, pe_c0, pe_a1, pe_c1,
                    next_pca_0, next_pca_1,
                    card_jsd_sum_next, card_jsd_count_next,
                    match_per_step_next)

        (ep_ts, env_state, obs, rng, done, reward, act_onehot,
         hstate_0, hstate_1, last_info, jsd_sum, jsd_count,
         prev_reward_0, prev_reward_1, prev_action_0, prev_action_1,
         prev_a0, prev_a1,
         pe_a0, pe_c0, pe_a1, pe_c1,
         prev_pca_0, prev_pca_1,
         card_jsd_sum, card_jsd_count,
         match_per_step) = carry
        new_carry = jax.lax.cond(
            done["__all__"],
            lambda curr_carry: curr_carry,
            take_step,
            operand=carry,
        )
        return new_carry, None

    final_carry, _ = jax.lax.scan(scan_step, init_carry, None, length=max_episode_steps)
    info = final_carry[9]
    mean_jsd = final_carry[10] / final_carry[11]
    mean_card_jsd = final_carry[24] / (final_carry[25] + 1e-8)
    match_per_step_out = final_carry[26]  # shape (max_episode_steps,)
    return info, mean_jsd, mean_card_jsd, match_per_step_out


def run_episodes_with_jsd(rng, env, agent_0_param, agent_0_policy,
                          agent_1_param, agent_1_policy,
                          max_episode_steps, num_eps, action_sizes,
                          feed_attn_dims=None, ja_card_masks=None,
                          greedy_eval=True, partner_feed_dim=5,
                          lbf_ctx=None, feed_mask_fn=None):
    """Run num_eps episodes in parallel, returning LogWrapper info + per-episode mean JSD."""
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    vmap_fn = jax.vmap(
        lambda ep_rng: run_single_episode_with_jsd(
            ep_rng, env, agent_0_param, agent_0_policy,
            agent_1_param, agent_1_policy, max_episode_steps, action_sizes,
            feed_attn_dims=feed_attn_dims, ja_card_masks=ja_card_masks,
            greedy_eval=greedy_eval, partner_feed_dim=partner_feed_dim,
            lbf_ctx=lbf_ctx, feed_mask_fn=feed_mask_fn,
        )
    )
    all_info, all_jsd, all_card_jsd, all_match = vmap_fn(ep_rngs)
    # all_jsd, all_card_jsd: (num_eps,); all_match: (num_eps, max_episode_steps).
    return all_info, all_jsd, all_card_jsd, all_match


def run_row_with_jsd(rng, env, agent_0_param, agent_0_policy,
                     all_agent_1_params, agent_1_policy,
                     max_episode_steps, num_eps, action_sizes,
                     feed_attn_dims=None, ja_card_masks=None,
                     greedy_eval=True, partner_feed_dim=5,
                     lbf_ctx=None, feed_mask_fn=None):
    """Run one row of the XP matrix: agent_0 vs all partners, vmapped over partners and episodes."""
    num_partners = jax.tree.leaves(all_agent_1_params)[0].shape[0]
    partner_rngs = jax.random.split(rng, num_partners)

    # vmap over partners (j dimension)
    def eval_one_partner(partner_rng, agent_1_param):
        return run_episodes_with_jsd(
            partner_rng, env, agent_0_param, agent_0_policy,
            agent_1_param, agent_1_policy, max_episode_steps, num_eps, action_sizes,
            feed_attn_dims=feed_attn_dims, ja_card_masks=ja_card_masks,
            greedy_eval=greedy_eval, partner_feed_dim=partner_feed_dim,
            lbf_ctx=lbf_ctx, feed_mask_fn=feed_mask_fn,
        )

    return jax.vmap(eval_one_partner)(partner_rngs, all_agent_1_params)


def xp_mean_and_sem(xp_matrix):
    """XP mean and standard error via the disjoint-pair estimator.

    Delegates to `xp_stats.xp_mean_se` (Forkel et al. 2511.22581 eq 30/31): the
    role-symmetric mean over m = floor(N/2) disjoint seed pairs and its
    std(ddof=1)/sqrt(m). Single XP estimator shared across the codebase.

    Args:
        xp_matrix: (n, n) array where entry (i,j) is the mean metric
                   when seed i is agent 0 and seed j is agent 1.
    Returns:
        (mean, se) over the disjoint-pair samples.
    """
    mean, se, _ = xp_mean_se(xp_matrix)
    return mean, se


def _score_color_range(env_name: str, num_agents: int) -> tuple[float, float | None]:
    """Colour range for the XP episode-return heatmap.

    Pin to `[0, max-possible-return]` so a matrix whose cells fall in a narrow band
    (e.g. all ~0.2) is not rainbow-stretched over its own min/max. card-game's
    `base_return` is a decision-success rate in `[0, 1]`. LBF shares rewards as the
    mean over agents while Jumanji normalises the team return to 1.0, so each agent's
    episode return tops out at `1/num_agents` (0.5 for 2 agents). Envs whose max is
    unknown keep an auto ceiling but still pin the floor at 0.
    """
    if env_name == "card-game":
        return 0.0, 1.0
    if env_name == "lbf":
        return 0.0, 1.0 / num_agents
    return 0.0, None


def save_xp_heatmap(matrix_mean: np.ndarray, matrix_std,
                     title: str, filepath: str, fmt: str = ".2f",
                     cmap: str = "YlOrRd", vmin: float | None = None,
                     vmax: float | None = None):
    """Save an annotated NxN heatmap as PNG. Pass matrix_std=None to annotate
    cells with the mean only (no ±std line)."""
    n = matrix_mean.shape[0]
    fig, ax = plt.subplots(figsize=(1.5 + n * 1.2, 1.0 + n * 1.0))
    im = ax.imshow(matrix_mean, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    for i in range(n):
        for j in range(n):
            m = matrix_mean[i, j]
            if matrix_std is None:
                text = f"{m:{fmt}}"
            else:
                s = matrix_std[i, j]
                text = f"{m:{fmt}}\n±{s:{fmt}}"
            color = "white" if matrix_mean[i, j] > (im.norm.vmax + im.norm.vmin) / 2 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"seed_{i}" for i in range(n)], fontsize=9)
    ax.set_yticklabels([f"seed_{i}" for i in range(n)], fontsize=9)
    ax.set_xlabel("Agent 1")
    ax.set_ylabel("Agent 0")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[xp_seeds] heatmap saved: {filepath}")


def save_xp_csv(matrix_mean: np.ndarray, matrix_std: np.ndarray,
                 filepath: str, label: str = "value"):
    """Save NxN mean and std matrices as CSV."""
    n = matrix_mean.shape[0]
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["agent_0 \\ agent_1"] + [f"seed_{j}" for j in range(n)]
        writer.writerow([f"{label}_mean"] + header[1:])
        for i in range(n):
            writer.writerow([f"seed_{i}"] + [f"{matrix_mean[i, j]:.4f}" for j in range(n)])
        writer.writerow([])
        writer.writerow([f"{label}_std"] + header[1:])
        for i in range(n):
            writer.writerow([f"seed_{i}"] + [f"{matrix_std[i, j]:.4f}" for j in range(n)])
    print(f"[xp_seeds] CSV saved: {filepath}")


def _load_hydra_config(checkpoint_path: str) -> dict | None:
    """Load resolved Hydra config from the run directory, if available."""
    from omegaconf import OmegaConf
    run_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        return None
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def _build_run_label(algo_cfg: dict, task_name: str) -> str:
    """Build a human-readable label from config, e.g. 'cramped_room / BETA=0.1 / 5M'."""
    # Extract layout name from task (e.g. "overcooked-v1/cramped_room" -> "cramped_room")
    layout = task_name.split("/")[-1] if "/" in task_name else task_name
    parts = [layout]
    beta = algo_cfg.get("JA_BETA_MAX")
    if beta is not None:
        parts.append(f"BETA={beta}")
    total = algo_cfg.get("TOTAL_TIMESTEPS")
    if total is not None:
        total = float(total)
        parts.append(f"{total / 1e6:.0f}M" if total >= 1e6 else f"{total:.0f}")
    return " / ".join(parts)


def _build_xp_name(algo_cfg: dict, layout: str) -> str:
    """Build descriptive XP run name from config.

    Mirrors _build_run_string in wandb_visualizations.py but without
    alg prefix (already in the wandb name as XP_) and without date.
    """
    parts = [layout]
    beta = algo_cfg.get("JA_BETA_MAX", 0)
    parts.append(f"b{beta}")
    ent = algo_cfg.get("ENT_COEF", 0.01)
    if ent != 0.01:
        parts.append(f"ent{ent}")
    total = algo_cfg.get("TOTAL_TIMESTEPS")
    if total is not None:
        total = float(total)
        parts.append(f"{total/1e6:.0f}M" if total >= 1e6 else f"{total:.0f}")
    if algo_cfg.get("COMMUNICATION", False):
        parts.append("comm")
    if algo_cfg.get("FEED_OTHER_ATTN", False):
        parts.append("feed_attn")
    seeds = algo_cfg.get("NUM_SEEDS", 1)
    if seeds > 1:
        parts.append(f"s{seeds}")
    return "_".join(parts)


def _init_xp_wandb_run(algo_cfg: dict, task_name: str, run_dir: str, wb_prefix: str):
    """Create a dedicated wandb run for XP-style eval artifacts."""
    import wandb

    layout = task_name.split("/")[-1] if "/" in task_name else task_name
    extra_tags = [wb_prefix.lower()] if wb_prefix and wb_prefix != "XP" else []
    return wandb.init(
        project="aht-benchmark",
        entity="g-benintendi-university-of-brescia",
        config=algo_cfg,
        tags=[
            str(algo_cfg.get("ALG", "")),
            f"{task_name}" if "/" in task_name else layout,
            f"beta={algo_cfg.get('JA_BETA_MAX', 0)}",
            f"ent={algo_cfg.get('ENT_COEF', 0.01)}",
            "xp_eval",
        ] + extra_tags,
        group=f"{task_name}/{algo_cfg.get('ALG', '')}",
        name=f"{wb_prefix}_{_build_xp_name(algo_cfg, layout)}",
        dir=run_dir,
    )


def _log_xp_to_wandb(jsd_matrix, score_mean, xp_dir, algo_cfg,
                      task_name, run_dir, wb_run=None, wb_prefix="XP",
                      card_jsd_matrix=None, match_per_step_matrix=None):
    """Log XP results to wandb. Creates a new run if `wb_run` is None.

    `jsd_matrix=None` skips all JSD-related logging (used for card-game runs
    where raw spatial JSD is meaningless under Other-Play).
    """
    import wandb

    created_run = False
    if wb_run is None:
        wb_run = _init_xp_wandb_run(algo_cfg, task_name, run_dir, wb_prefix)
        created_run = True

    if score_mean is not None:
        wb_run.log(
            {f"{wb_prefix}/score_matrix": wandb.Image(os.path.join(xp_dir, "xp_score_matrix.png"))},
            commit=False,
        )
    if jsd_matrix is not None:
        wb_run.log(
            {f"{wb_prefix}/jsd_matrix": wandb.Image(os.path.join(xp_dir, "xp_jsd_matrix.png"))},
            commit=False,
        )

        jsd_ep_means = jsd_matrix.mean(axis=-1)
        sp_jsd = np.diag(jsd_ep_means).mean()
        xp_jsd_m, xp_jsd_s = xp_mean_and_sem(jsd_ep_means)
        wb_run.summary[f"{wb_prefix}/sp_jsd"] = sp_jsd
        wb_run.summary[f"{wb_prefix}/xp_jsd_mean"] = xp_jsd_m
        wb_run.summary[f"{wb_prefix}/xp_jsd_sem"] = xp_jsd_s
        sp_jsd_diag = np.diag(jsd_ep_means)
        wb_run.summary[f"{wb_prefix}/sp_jsd_sem"] = np.std(sp_jsd_diag) / np.sqrt(len(sp_jsd_diag))
    if score_mean is not None:
        sp_score_diag = np.diag(score_mean)
        sp_score = sp_score_diag.mean()
        sp_score_sem = np.std(sp_score_diag) / np.sqrt(len(sp_score_diag))
        xp_score_m, xp_score_s = xp_mean_and_sem(score_mean)
        wb_run.summary[f"{wb_prefix}/sp_score"] = sp_score
        wb_run.summary[f"{wb_prefix}/sp_score_sem"] = sp_score_sem
        wb_run.summary[f"{wb_prefix}/xp_score_mean"] = xp_score_m
        wb_run.summary[f"{wb_prefix}/xp_score_sem"] = xp_score_s

    wandb.save(os.path.join(xp_dir, "xp_score_matrix.csv"), base_path=xp_dir)
    if jsd_matrix is not None:
        wandb.save(os.path.join(xp_dir, "xp_jsd_matrix.csv"), base_path=xp_dir)

    if card_jsd_matrix is not None:
        card_jsd_png = os.path.join(xp_dir, "xp_card_jsd_matrix.png")
        if os.path.exists(card_jsd_png):
            wb_run.log(
                {f"{wb_prefix}/card_jsd_matrix": wandb.Image(card_jsd_png)},
                commit=False,
            )
            card_jsd_ep = card_jsd_matrix.mean(axis=-1)
            sp_card = np.diag(card_jsd_ep).mean()
            xp_card_m, xp_card_s = xp_mean_and_sem(card_jsd_ep)
            wb_run.summary[f"{wb_prefix}/sp_card_jsd"] = float(sp_card)
            wb_run.summary[f"{wb_prefix}/xp_card_jsd_mean"] = float(xp_card_m)
            wb_run.summary[f"{wb_prefix}/xp_card_jsd_sem"] = float(xp_card_s)
        card_jsd_csv = os.path.join(xp_dir, "xp_card_jsd_matrix.csv")
        if os.path.exists(card_jsd_csv):
            wandb.save(card_jsd_csv, base_path=xp_dir)

    if match_per_step_matrix is not None:
        n_steps = match_per_step_matrix.shape[-1]
        for t in range(n_steps):
            png_path = os.path.join(xp_dir, f"xp_step_match_t{t}.png")
            csv_path = os.path.join(xp_dir, f"xp_step_match_t{t}.csv")
            if os.path.exists(png_path):
                wb_run.log(
                    {f"{wb_prefix}/step_match_t{t}": wandb.Image(png_path)},
                    commit=False,
                )
                step_mean = match_per_step_matrix[:, :, :, t].mean(axis=2)
                sp_match_t = float(np.diag(step_mean).mean())
                xp_m_t, xp_s_t = xp_mean_and_sem(step_mean)
                wb_run.summary[f"{wb_prefix}/sp_step_match_t{t}"] = sp_match_t
                wb_run.summary[f"{wb_prefix}/xp_step_match_t{t}_mean"] = float(xp_m_t)
                wb_run.summary[f"{wb_prefix}/xp_step_match_t{t}_sem"] = float(xp_s_t)
            if os.path.exists(csv_path):
                wandb.save(csv_path, base_path=xp_dir)

    if created_run:
        wb_run.finish()
        print(f"[xp_seeds] wandb run: {wb_run.url}")


def run_xp_from_params(env, policy, stacked_params, algo_cfg: dict,
                       savedir: str, task_name: str | None = None,
                       wb_run=None, greedy_eval=True, wb_prefix="XP"):
    """Run cross-play evaluation from pre-built objects.

    Called either from standalone CLI or from training loops after multi-seed runs.

    Args:
        env: the LogWrapper-wrapped environment
        policy: the shared policy (same for all seeds)
        stacked_params: pytree with leading dim = num_seeds
        algo_cfg: algorithm config dict
        savedir: directory to save XP results
        task_name: task name for labels (e.g. "overcooked-v1-image/cramped_room")
        wb_run: existing wandb run to log to. If None, creates a new one.
    """
    # greedy_eval flows from the caller: greedy (argmax) decoding by default,
    # or stochastic sampling when greedy_eval=False (CLI --sampled).
    num_seeds = jax.tree.leaves(stacked_params)[0].shape[0]
    if num_seeds < 2:
        print(f"[xp_seeds] SKIP: only {num_seeds} seed(s) — need at least 2 for cross-play")
        return

    env_name = algo_cfg.get("ENV_NAME", "")
    if task_name is None:
        task_name = env_name
    run_label = _build_run_label(algo_cfg, task_name)
    created_wb_run = False
    if wb_run is None:
        wb_run = _init_xp_wandb_run(algo_cfg, task_name, savedir, wb_prefix)
        created_wb_run = True

    print(f"[xp_seeds] task={task_name}, seeds={num_seeds}, episodes={NUM_EVAL_EPISODES}")

    # Extract per-seed params and check for NaN
    seed_params = []
    for i in range(num_seeds):
        params_i = jax.tree.map(lambda x: x[i], stacked_params)
        num_nan = sum(int(jnp.isnan(x).sum()) for x in jax.tree.leaves(params_i))
        num_params = sum(x.size for x in jax.tree.leaves(params_i))
        status = f"OK ({num_params} params)" if num_nan == 0 else f"WARNING: {num_nan}/{num_params} NaN params!"
        seed_params.append(params_i)
        print(f"  seed {i}: {status}")

    max_steps = int(algo_cfg.get("ENV_KWARGS", {}).get("max_steps") or algo_cfg.get("ROLLOUT_LENGTH", 400))
    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, eval_rng = jax.random.split(rng)
    outer_rngs = jax.random.split(eval_rng, num_seeds)

    action_sizes = {k: int(env.action_space(k).n) for k in env.agents}

    # Compute feed_attn_dims if needed
    feed_attn = algo_cfg.get("FEED_OTHER_ATTN", False)
    feed_attn_dims = None
    if feed_attn:
        from agents.initialize_agents import _get_image_dims
        from agents.ja_actor_critic import _compute_resnet_output_dims
        _img_h, _img_w, _ = _get_image_dims(env)
        _feat_h, _feat_w = _compute_resnet_output_dims(
            _img_h, _img_w,
            stride=algo_cfg.get("CONV_STRIDE", 2),
            kernel_size=algo_cfg.get("CONV_KERNEL_SIZE", 3),
            padding=algo_cfg.get("CONV_PADDING", "SAME"),
            num_blocks=algo_cfg.get("CONV_NUM_BLOCKS", 4),
        )
        feed_attn_dims = (_img_h, _img_w, _feat_h, _feat_w)
        print(f"[xp_seeds] feed_other_attn enabled: img=({_img_h},{_img_w}), feat=({_feat_h},{_feat_w})")

    ja_card_masks = None
    if algo_cfg.get("JA_CARD_ATTN", False) or algo_cfg.get("JA_CARD_METRIC", False):
        from agents.initialize_agents import _get_image_dims
        from agents.ja_actor_critic import _compute_resnet_output_dims
        from agents.ja_utils import build_card_masks
        _img_h, _img_w, _ = _get_image_dims(env)
        _feat_h, _feat_w = _compute_resnet_output_dims(
            _img_h, _img_w,
            stride=algo_cfg.get("CONV_STRIDE", 2),
            kernel_size=algo_cfg.get("CONV_KERNEL_SIZE", 3),
            padding=algo_cfg.get("CONV_PADDING", "SAME"),
            num_blocks=algo_cfg.get("CONV_NUM_BLOCKS", 4),
        )
        ja_card_masks = build_card_masks(_img_h, _img_w, _feat_h, _feat_w)

    # LBF per-fruit partner-feed context (mirrors training-time augmentation).
    lbf_ctx = None
    if env_name == "lbf" and algo_cfg.get("JA_FRUIT_PARTNER_FEED", False):
        from agents.lbf.ja_lbf_attention import lbf_attention_ctx
        lbf_ctx = lbf_attention_ctx(algo_cfg, env)
        print(f"[xp_seeds] lbf partner-feed: N_fruits={lbf_ctx['num_fruits']}")

    # Ocv2 visibility gating: mask the feed exactly as at train time.
    feed_mask_fn = None
    if (feed_attn_dims is not None and env_name == "overcooked-v2"
            and algo_cfg.get("JA_VISIBILITY_GATING", False)):
        from agents.overcooked_v2.ja_overcooked_v2_attention import (
            make_visibility_mask_fn, overcooked_v2_object_ctx,
        )
        feed_mask_fn = make_visibility_mask_fn(overcooked_v2_object_ctx(algo_cfg, env))
        print("[xp_seeds] ocv2 visibility gating enabled for the feed channel")

    xp_partner_feed_dim = 5

    row_fn = jax.jit(lambda rng_i, p0: run_row_with_jsd(
        rng_i, env, p0, policy, stacked_params, policy, max_steps, NUM_EVAL_EPISODES, action_sizes,
        feed_attn_dims=feed_attn_dims, ja_card_masks=ja_card_masks, greedy_eval=greedy_eval,
        partner_feed_dim=xp_partner_feed_dim, lbf_ctx=lbf_ctx, feed_mask_fn=feed_mask_fn,
    ))

    all_row_metrics = []
    jsd_matrix = np.zeros((num_seeds, num_seeds, NUM_EVAL_EPISODES))
    card_jsd_matrix = np.zeros((num_seeds, num_seeds, NUM_EVAL_EPISODES))
    # Per-step token match is card-game-specific (each scan step is a message
    # or the decision pick); for primitive-action envs the metric is not
    # meaningful and would explode (e.g. Overcooked max_steps=400 → 400
    # heatmaps). Allocate only for card-game.
    track_step_match = (env_name == "card-game")
    match_per_step_matrix = (
        np.zeros((num_seeds, num_seeds, NUM_EVAL_EPISODES, max_steps), dtype=np.float32)
        if track_step_match else None
    )
    start_time = time.time()
    for i in range(num_seeds):
        print(f"  row {i} (seed {i} vs all) ...", end=" ", flush=True)
        row_metrics, row_jsds, row_card_jsds, row_match_per_step = row_fn(
            outer_rngs[i], seed_params[i],
        )
        jsd_matrix[i] = np.array(row_jsds)
        card_jsd_matrix[i] = np.array(row_card_jsds)
        if track_step_match:
            match_per_step_matrix[i] = np.array(row_match_per_step)
        all_row_metrics.append(row_metrics)

        for j in range(num_seeds):
            ret = np.array(row_metrics["returned_episode_returns"][j]).mean()
            jsd_mean = float(row_jsds[j].mean())
            cjsd_mean = float(row_card_jsds[j].mean())
            label = "SP" if i == j else "XP"
            print(
                f"  [{label}] {i}x{j}: return={ret:.2f} "
                f"jsd={jsd_mean:.4f} card_jsd={cjsd_mean:.4f}",
                end="",
            )
        print()

    xp_metrics = tree_stack(all_row_metrics)
    elapsed = time.time() - start_time
    print(f"[xp_seeds] evaluation done in {elapsed:.1f}s")

    metric_names = get_metric_names(env_name)
    seed_names = [f"seed_{i}" for i in range(num_seeds)]
    for metric_name in metric_names:
        print_xp_table(xp_metrics, metric_name, seed_names)

    is_card_game = env_name == "card-game"
    if not is_card_game:
        print_jsd_table(jsd_matrix, seed_names)
    print_sp_vs_xp_summary(xp_metrics, metric_names,
                           None if is_card_game else jsd_matrix, num_seeds)

    # Save heatmaps and CSVs
    xp_dir = os.path.join(savedir, "xp_results")
    os.makedirs(xp_dir, exist_ok=True)

    score_mean = score_std = None
    score_key = "base_return" if "base_return" in xp_metrics else "returned_episode_returns"
    if score_key in xp_metrics:
        score_data = np.array(xp_metrics[score_key]).mean(axis=-1)
        score_mean = score_data.mean(axis=-1)
        score_std = score_data.std(axis=-1)
        # Pin the colour scale to [0, max-possible-return] (per-env max in
        # _score_color_range) so a matrix whose cells fall in a narrow band
        # (e.g. all ~0.2) isn't rainbow-stretched over its own min/max.
        score_vmin, score_vmax = _score_color_range(env_name, env.num_agents)
        save_xp_heatmap(score_mean, score_std,
                         f"XP Episode Return — {run_label}",
                         os.path.join(xp_dir, "xp_score_matrix.png"),
                         vmin=score_vmin, vmax=score_vmax)
        save_xp_csv(score_mean, score_std,
                     os.path.join(xp_dir, "xp_score_matrix.csv"), label="episode_return")

    if not is_card_game:
        jsd_mean = jsd_matrix.mean(axis=-1)
        jsd_std = jsd_matrix.std(axis=-1)
        save_xp_heatmap(jsd_mean, jsd_std,
                         f"XP JSD — {run_label}",
                         os.path.join(xp_dir, "xp_jsd_matrix.png"),
                         fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=0.693)
        save_xp_csv(jsd_mean, jsd_std,
                     os.path.join(xp_dir, "xp_jsd_matrix.csv"), label="jsd")

    # OP-corrected card-level JSD matrix (card-game only; meaningful when
    # JA_CARD_METRIC or JA_CARD_ATTN was on at training time).
    have_card_jsd = is_card_game and ja_card_masks is not None
    card_jsd_mean = card_jsd_std = None
    if have_card_jsd:
        card_jsd_mean = card_jsd_matrix.mean(axis=-1)
        card_jsd_std = card_jsd_matrix.std(axis=-1)
        save_xp_heatmap(
            card_jsd_mean, card_jsd_std,
            f"XP Card JSD — {run_label}",
            os.path.join(xp_dir, "xp_card_jsd_matrix.png"),
            fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=0.693,
        )
        save_xp_csv(
            card_jsd_mean, card_jsd_std,
            os.path.join(xp_dir, "xp_card_jsd_matrix.csv"),
            label="card_jsd",
        )

    # Per-step token-match matrices: one (i, j) heatmap per scan step,
    # showing GT-frame agreement at that step. Step max_steps-1 is the
    # decision pick — that matrix should agree with the score matrix.
    # Earlier steps (0..max_steps-2) are deliberation messages and reveal
    # how quickly each pair converges on a shared protocol.
    if match_per_step_matrix is not None:
        n_steps = match_per_step_matrix.shape[-1]
        match_mean_steps = match_per_step_matrix.mean(axis=2)  # (S, S, max_steps)
        match_std_steps = match_per_step_matrix.std(axis=2)    # (S, S, max_steps)
        for t in range(n_steps):
            is_decision = (t == n_steps - 1)
            title = (
                f"XP Decision pick-match (step {t}) — {run_label}"
                if is_decision
                else f"XP Message-match step {t} — {run_label}"
            )
            png_path = os.path.join(xp_dir, f"xp_step_match_t{t}.png")
            csv_path = os.path.join(xp_dir, f"xp_step_match_t{t}.csv")
            save_xp_heatmap(
                match_mean_steps[:, :, t], match_std_steps[:, :, t],
                title, png_path,
                fmt=".3f", cmap="YlOrRd", vmin=0.0, vmax=1.0,
            )
            save_xp_csv(
                match_mean_steps[:, :, t], match_std_steps[:, :, t],
                csv_path, label=f"step_match_t{t}",
            )

    print(f"[xp_seeds] results saved to {xp_dir}")

    if env_name == "card-game":
        action_dist_dir = os.path.join(xp_dir, "action_distributions")
        ad_num_eps = int(algo_cfg.get("XP_ACTION_DIST_NUM_EPISODES", 50))
        seed_indices = list(range(num_seeds))
        print(
            f"[xp_seeds] logging card-game action distributions "
            f"(greedy only, {ad_num_eps} eps/seed)"
        )
        generate_action_distribution_artifacts(
            inner_env=env._env,
            stacked_params=stacked_params,
            policy=policy,
            max_steps=max_steps,
            output_dir=os.path.join(action_dist_dir, "greedy"),
            seed_indices=seed_indices,
            num_episodes=ad_num_eps,
            feed_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
            greedy=True,
            wb_run=wb_run,
            wb_prefix=wb_prefix,
        )

        # XP videos are off by default (heavy on disk + wandb). Enable per-run
        # with algorithm.EVAL_VIDEO_LOG_XP=true; XP_VIDEO_MAX_PAIRS defaults
        # to 3 when XP videos are enabled.
        log_xp_videos = bool(algo_cfg.get("EVAL_VIDEO_LOG_XP", False))
        max_pairs = int(algo_cfg.get("XP_VIDEO_MAX_PAIRS", 3 if log_xp_videos else 0))
        xp_video_eps = int(algo_cfg.get("EVAL_VIDEO_XP_NUM_EPISODES", 3))
        seed_pairs = (
            [(i, j) for i in range(num_seeds) for j in range(i + 1, num_seeds)][:max_pairs]
            if log_xp_videos and max_pairs > 0
            else []
        )
        if seed_pairs:
            print(
                f"[xp_seeds] logging card-game XP videos for pairs {seed_pairs} "
                f"({xp_video_eps} eps/pair)"
            )

            class _WandbVideoLogger:
                def __init__(self, run):
                    self.run = run

                def log_video(self, tag, path, commit=True, caption=None):
                    import wandb

                    self.run.log({tag: wandb.Video(path, format="mp4", caption=caption)}, commit=commit)

            xp_video_dir = os.path.join(xp_dir, "videos")
            os.makedirs(xp_video_dir, exist_ok=True)
            _log_card_game_xp_videos(
                env._env, policy, stacked_params, max_steps,
                f"{wb_prefix}/videos", xp_video_dir, _WandbVideoLogger(wb_run),
                feed_attn_dims=feed_attn_dims,
                ja_card_masks=ja_card_masks,
                seed_pairs=seed_pairs,
                num_episodes=xp_video_eps,
                fps=3,
            )

    _log_xp_to_wandb(
        None if is_card_game else jsd_matrix, score_mean, xp_dir,
        algo_cfg, task_name, savedir, wb_run=wb_run, wb_prefix=wb_prefix,
        card_jsd_matrix=card_jsd_matrix if have_card_jsd else None,
        match_per_step_matrix=match_per_step_matrix,
    )

    if created_wb_run:
        wb_run.finish()
        print(f"[xp_seeds] wandb run: {wb_run.url}")


def _rollout_team_return(rng, inner_env, team_params, policy, max_steps, *, greedy=True):
    """One episode for a fixed team with per-slot params; returns the shared return.

    `team_params[k]` is the params pytree for agent k.
    """
    agents = inner_env.agents
    n = inner_env.num_agents
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = inner_env.reset(reset_rng)
    hstates = [policy.init_hstate(1) for _ in range(n)]
    done = [jnp.zeros((1, 1), dtype=bool) for _ in range(n)]
    total = 0.0
    for _ in range(max_steps):
        avail = inner_env.get_avail_actions(env_state)
        env_act = {}
        for k, a in enumerate(agents):
            rng, act_rng = jax.random.split(rng)
            act_k, hstates[k] = policy.get_action(
                team_params[k],
                obs[a].reshape(1, 1, -1),
                done[k],
                avail[a].astype(jnp.float32).reshape(1, 1, -1),
                hstates[k], act_rng, greedy=greedy,
            )
            env_act[a] = act_k.reshape(())
        rng, step_rng = jax.random.split(rng)
        obs, env_state, reward, dones, _ = inner_env.step(step_rng, env_state, env_act)
        total += float(reward[agents[0]])  # shared cooperative reward
        done = [dones[a].reshape(1, 1) for a in agents]
        if bool(dones["__all__"]):
            break
    return total


def _mixed_seed_team_params(seed_i_params, seed_j_params, num_agents: int):
    """All nontrivial i/j team compositions for an N-agent XP pairing.

    For four agents this yields the requested 1+3, 2+2, and 3+1 cases, with all
    slot assignments included so the result is not tied to agent_0 being special.
    """
    teams = []
    slots = range(num_agents)
    for n_i in range(1, num_agents):
        for i_slots in itertools.combinations(slots, n_i):
            i_slot_set = set(i_slots)
            teams.append((
                n_i,
                [
                    seed_i_params if slot in i_slot_set else seed_j_params
                    for slot in slots
                ],
            ))
    return teams


def run_xp_nagent_from_params(env, policy, stacked_params, algo_cfg, savedir,
                              task_name=None, num_episodes=32, wb_prefix="XP",
                              logger=None):
    """N-agent cross-play across all mixed seed-i/seed-j team compositions.

    Diagonal entries are self-play (all slots use seed i). Off-diagonal entries
    average every nontrivial mixed composition between seeds i and j. For four
    agents, that is 1+3, 2+2, and 3+1, across all slot assignments. Saves a
    heatmap + CSV, prints the SP/XP summary (XP = off-diagonal mean via
    `xp_mean_and_sem`), and — if `logger` is given — logs the SP/XP scalars to
    the run's wandb. Parameter-shared network, per-slot params — no Other-Play /
    JSD / card-game machinery.
    """
    del task_name
    from envs.base_env import get_inner_env
    inner_env = get_inner_env(env)
    num_seeds = int(jax.tree.leaves(stacked_params)[0].shape[0])
    max_steps = int(algo_cfg.get("ENV_KWARGS", {}).get("max_steps", 100))
    greedy_eval = True

    def seed_params(s):
        return jax.tree.map(lambda x: x[s], stacked_params)

    matrix = np.zeros((num_seeds, num_seeds))
    by_count = {
        n_i: np.full((num_seeds, num_seeds), np.nan, dtype=np.float64)
        for n_i in range(1, inner_env.num_agents)
    }
    num_mixed_teams = (2 ** inner_env.num_agents) - 2
    print(
        f"[xp_seeds:nagent] mixed teams per seed pair: {num_mixed_teams}",
        flush=True,
    )
    rng = jax.random.PRNGKey(EVAL_SEED)
    for i in range(num_seeds):
        params_i = seed_params(i)
        for j in range(num_seeds):
            params_j = seed_params(j)
            teams = (
                [(inner_env.num_agents, [params_i] * inner_env.num_agents)]
                if i == j
                else _mixed_seed_team_params(params_i, params_j, inner_env.num_agents)
            )
            rets = []
            count_rets: dict[int, list[float]] = {n_i: [] for n_i in by_count}
            for n_i, team in teams:
                for _ in range(num_episodes):
                    rng, ep_rng = jax.random.split(rng)
                    ret = _rollout_team_return(
                        ep_rng, inner_env, team, policy, max_steps,
                        greedy=greedy_eval,
                    )
                    rets.append(ret)
                    if n_i in count_rets:
                        count_rets[n_i].append(ret)
            matrix[i, j] = float(np.mean(rets))
            for n_i, vals in count_rets.items():
                if vals:
                    by_count[n_i][i, j] = float(np.mean(vals))
        print(f"[xp_seeds:nagent] seed-pair row {i}: {num_seeds} partner seeds done", flush=True)

    sp = float(np.mean(np.diag(matrix)))
    xp_mean, xp_sem = xp_mean_and_sem(matrix)
    os.makedirs(savedir, exist_ok=True)
    save_xp_heatmap(
        matrix, None,
        title=f"XP return (row/col=seed pair; off-diag=all mixed teams)\nSP={sp:.1f}  XP={xp_mean:.1f}+/-{xp_sem:.1f}",
        filepath=os.path.join(savedir, "xp_nagent_return.png"),
    )
    save_xp_csv(matrix, np.zeros_like(matrix),
                os.path.join(savedir, "xp_nagent_return.csv"), label="return")
    for n_i, count_matrix in by_count.items():
        save_xp_csv(
            count_matrix, np.zeros_like(count_matrix),
            os.path.join(savedir, f"xp_nagent_return_{n_i}v{inner_env.num_agents - n_i}.csv"),
            label="return",
        )
    print(f"[xp_seeds:nagent] SP (diag) = {sp:.2f}  |  XP (off-diag) = {xp_mean:.2f} +/- {xp_sem:.2f}")
    if logger is not None:
        try:
            import wandb
            wb_run = getattr(logger, "run", None)
            if wb_run is not None:
                png = os.path.join(savedir, "xp_nagent_return.png")
                wb_run.log({f"{wb_prefix}/return_matrix": wandb.Image(png)}, commit=False)
        except Exception as e:
            print(f"[xp_seeds:nagent] WARN: wandb log failed ({e}); continuing.", flush=True)
    return matrix, sp, (xp_mean, xp_sem)


def run_xp(env, policy, params, algo_cfg, savedir, logger=None, *, jsd=True,
           task_name=None, wb_prefix="XP"):
    """Single in-process XP-eval entry, run right after training.

    Dispatches by team size and trainer type:
      - 2-agent JA runs (`jsd=True`) get the Other-Play / JSD / disjoint-pairing
        matrix (`run_xp_from_params`).
      - >2-agent envs, or non-attention baselines (`jsd=False`), get the
        return-only ego/partner matrix (`run_xp_nagent_from_params`).

    `params` is the per-seed params to cross-play; callers pass best-checkpoint
    params (final params only as a fallback).
    """
    if env.num_agents == 2 and jsd:
        run_xp_from_params(
            env, policy, params, algo_cfg, savedir=savedir, task_name=task_name,
            wb_run=getattr(logger, "run", None), greedy_eval=True, wb_prefix=wb_prefix,
        )
    else:
        run_xp_nagent_from_params(
            env, policy, params, algo_cfg, savedir=savedir, task_name=task_name,
            wb_prefix=wb_prefix, logger=logger,
        )


def run_xp_best_from_run(rundir: str, scores_path: str | None = None,
                         wb_prefix: str = "XP_best", wandb_resume: str | None = None):
    """Post-hoc XP on each seed's BEST checkpoint, for an already-finished run.

    Reads `chunk_scores.json` (`best_ckpt_idx_per_seed` + absolute
    `ckpt_folder_paths`), reconstructs per-seed best params (each seed's best
    index may differ), and runs the XP matrix dispatched by team size. Writes the
    matrix PNG/CSV under `rundir`. `scores_path` defaults to
    `<rundir>/chunk_scores.json`; pass it explicitly when SAVE_CHECKPOINT_FOLDER
    placed it under CHECKPOINT_ROOT/<run_name>/.
    """
    import json

    hydra_cfg = _load_hydra_config(rundir)
    if hydra_cfg is None:
        raise ValueError(f"No .hydra/config.yaml found under {rundir}")
    algo_cfg = hydra_cfg["algorithm"]

    if scores_path is None:
        scores_path = os.path.join(rundir, "chunk_scores.json")
    with open(scores_path) as f:
        cs = json.load(f)
    best_idx = cs["best_ckpt_idx_per_seed"]
    ckpt_paths = cs["ckpt_folder_paths"]
    print(f"[xp_best] {len(best_idx)} seeds; best ckpt idx/seed = {best_idx}", flush=True)

    env = LogWrapper(make_env(algo_cfg["ENV_NAME"], dict(algo_cfg["ENV_KWARGS"])))
    rng = jax.random.PRNGKey(EVAL_SEED)
    alg = algo_cfg.get("ALG", "ja_ippo")
    if alg == "image_ippo":
        from agents.initialize_agents import initialize_image_agent
        policy, _ = initialize_image_agent(algo_cfg, env, rng)
    else:
        policy, _ = initialize_ja_image_agent(algo_cfg, env, rng)

    # Each best ckpt folder holds all seeds at that checkpoint; take this seed's
    # slice, then stack across seeds into per-seed best params.
    seed_best = []
    for s, b in enumerate(best_idx):
        ckpt_path = ckpt_paths[b]
        print(f"[xp_best] seed {s}: loading ckpt[{b}] {ckpt_path}", flush=True)
        ckpt_params = load_train_run(ckpt_path)
        seed_best.append(jax.tree.map(lambda x, _s=s: x[_s], ckpt_params))
    best_params = jax.tree.map(lambda *xs: jnp.stack(xs), *seed_best)

    logger = None
    if wandb_resume:
        import types

        import wandb
        wb = wandb.init(
            id=wandb_resume, resume="must",
            project=hydra_cfg["logger"]["project"],
            entity=hydra_cfg["logger"]["entity"],
        )
        logger = types.SimpleNamespace(run=wb)

    run_xp(env, policy, best_params, algo_cfg, rundir, logger,
           jsd=(alg != "image_ippo"), task_name=algo_cfg.get("ENV_NAME"),
           wb_prefix=wb_prefix)
    if logger is not None:
        logger.run.finish()
    print(f"[xp_best] XP matrix written under {rundir}", flush=True)


def run_xp_evaluation(task_name: str | None, checkpoint_path: str, greedy_eval: bool = True,
                      use_best: bool = True,
                      wb_prefix: str | None = None,
                      xp_video_max_pairs: int | None = None,
                      no_xp_videos: bool = False,
                      no_op: bool = False):
    """Standalone XP evaluation from a saved checkpoint.

    `use_best` selects `best_params` over `final_params` (per-seed best checkpoint).
    If `best_params` is missing, final params are used as a compatibility fallback.
    Results land under the run's `xp_results/` directory and are logged to a
    fresh wandb run with `wb_prefix` (default `XP`).

    `no_op` strips every `other_play_*` env kwarg before `make_env`, so an
    OP-trained run is cross-played in the shared identity frame (both agents see
    the true board) rather than under per-agent random relabelling. This is the
    deployment-condition XP; results go to a separate `rerun_noop/` dir.
    """
    hydra_cfg = _load_hydra_config(checkpoint_path)
    if task_name is not None:
        task_cfg = load_task_config(task_name)
        algo_cfg = load_algo_config()
    else:
        if hydra_cfg is None:
            raise ValueError("No --task provided and no .hydra/config.yaml found")
        algo_cfg = hydra_cfg["algorithm"]
        task_cfg = {"ENV_NAME": algo_cfg["ENV_NAME"],
                    "ENV_KWARGS": algo_cfg["ENV_KWARGS"],
                    "ROLLOUT_LENGTH": algo_cfg["ROLLOUT_LENGTH"]}
        task_name = hydra_cfg.get("TASK_NAME", algo_cfg["ENV_NAME"])

    label_cfg = hydra_cfg["algorithm"] if hydra_cfg else algo_cfg

    env_kwargs = dict(task_cfg["ENV_KWARGS"])
    if algo_cfg.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    if task_cfg["ENV_NAME"] == "card-game":
        env_kwargs["scramble_partner_msg"] = False
    if no_op:
        removed = [k for k in list(env_kwargs)
                   if k.startswith("other_play_") and env_kwargs.pop(k, None)]
        print(f"[xp_seeds] --no-op: identity-frame XP, removed OP kwargs {removed}")
    env = make_env(task_cfg["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    run_data = load_train_run(checkpoint_path)
    all_final_params, params_key = _select_xp_params(run_data, prefer_best=use_best)
    print(f"[xp_seeds] using {params_key} from checkpoint")

    run_dir = os.path.dirname(checkpoint_path)
    if wb_prefix is None:
        wb_prefix = "XP"

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, init_rng = jax.random.split(rng)
    policy, _init_params = initialize_ja_image_agent(algo_cfg, env, init_rng)
    # Avoid overwriting the original training run's xp_results/ when re-evaluating with overrides.
    savedir = run_dir
    if no_op:
        savedir = os.path.join(run_dir, "rerun_noop")
        os.makedirs(savedir, exist_ok=True)
        print(f"[xp_seeds] writing no-OP rerun outputs to {savedir}")
    elif params_key == "best_params":
        savedir = os.path.join(run_dir, "rerun_best")
        os.makedirs(savedir, exist_ok=True)
        print(f"[xp_seeds] writing rerun outputs to {savedir}")
    if no_xp_videos:
        label_cfg["EVAL_VIDEO_LOG_XP"] = False
        print("[xp_seeds] --no-xp-videos: XP video rendering disabled")
    if xp_video_max_pairs is not None:
        # 0 or negative => render every off-diagonal pair (will be clipped to N*(N-1)/2 by
        # the slice in run_xp_from_params).
        effective = xp_video_max_pairs if xp_video_max_pairs > 0 else 10**9
        label_cfg["XP_VIDEO_MAX_PAIRS"] = effective
        print(f"[xp_seeds] XP_VIDEO_MAX_PAIRS overridden to {effective}")
    if env.num_agents != 2:
        # >2-agent envs: ego/partner cross-play, no 2-agent/OP/JSD machinery.
        run_xp_nagent_from_params(env, policy, all_final_params, label_cfg,
                                  savedir=savedir, task_name=task_name, wb_prefix=wb_prefix)
        return
    run_xp_from_params(env, policy, all_final_params, label_cfg,
                       savedir=savedir, task_name=task_name,
                       greedy_eval=greedy_eval, wb_prefix=wb_prefix)


def print_xp_table(xp_metrics, metric_name, seed_names):
    from prettytable import PrettyTable

    # (N, N, num_episodes, num_agents) -> avg over agents
    data = np.array(xp_metrics[metric_name]).mean(axis=-1)
    n = len(seed_names)
    table = PrettyTable()
    table.field_names = ["agent_0 \\ agent_1"] + seed_names

    for i in range(n):
        row = [seed_names[i]]
        for j in range(n):
            mean = data[i, j].mean()
            std = data[i, j].std()
            row.append(f"{mean:.2f} +/- {std:.2f}")
        table.add_row(row)

    print(f"\n{metric_name} (mean +/- std over {data.shape[2]} episodes):")
    print(table)


def print_jsd_table(jsd_matrix, seed_names):
    """Print N×N JSD matrix (mean ± std over episodes)."""
    from prettytable import PrettyTable

    n = len(seed_names)
    table = PrettyTable()
    table.field_names = ["agent_0 \\ agent_1"] + seed_names

    for i in range(n):
        row = [seed_names[i]]
        for j in range(n):
            mean = jsd_matrix[i, j].mean()
            std = jsd_matrix[i, j].std()
            row.append(f"{mean:.4f} +/- {std:.4f}")
        table.add_row(row)

    print(f"\nJSD (mean +/- std over {jsd_matrix.shape[2]} episodes):")
    print(table)


def print_sp_vs_xp_summary(xp_metrics, metric_names, jsd_matrix, num_seeds):
    """Report SP and XP with proper SEM using the seed-pairing scheme.

    `jsd_matrix=None` suppresses the JSD line (card-game runs).
    """
    print("\n=== Self-Play vs Cross-Play Summary ===")
    m = num_seeds // 2
    print(f"  ({num_seeds} seeds -> {m} independent XP samples)")
    if num_seeds % 2 != 0:
        print("  WARNING: odd number of seeds, last seed excluded from SEM computation")

    for metric_name in metric_names:
        # (N, N, episodes, agents) -> avg over agents and episodes -> (N, N)
        data = np.array(xp_metrics[metric_name]).mean(axis=(-1, -2))

        # SP: diagonal entries
        sp_scores = np.diag(data)
        sp_mean = np.mean(sp_scores)
        sp_sem = np.std(sp_scores) / np.sqrt(len(sp_scores))

        # XP: proper SEM via seed pairing
        xp_mean, xp_sem = xp_mean_and_sem(data)

        print(f"  {metric_name}:  SP = {sp_mean:.2f} +/- {sp_sem:.2f}  |  XP = {xp_mean:.2f} +/- {xp_sem:.2f}")

    if jsd_matrix is not None:
        jsd_ep_means = jsd_matrix.mean(axis=-1)  # (N, N)
        sp_jsd = np.diag(jsd_ep_means)
        sp_jsd_mean = np.mean(sp_jsd)
        sp_jsd_sem = np.std(sp_jsd) / np.sqrt(len(sp_jsd))
        xp_jsd_mean, xp_jsd_sem = xp_mean_and_sem(jsd_ep_means)
        print(f"  JSD:  SP = {sp_jsd_mean:.4f} +/- {sp_jsd_sem:.4f}  |  XP = {xp_jsd_mean:.4f} +/- {xp_jsd_sem:.4f}")


def run_xp_multi_checkpoint(task_name: str | None, checkpoint_paths: list[str]):
    """Cross-play evaluation loading one seed from each of multiple checkpoints.

    Used for fixed-partner experiments where each seed was trained separately.
    """
    # Use first checkpoint for config
    hydra_cfg = _load_hydra_config(checkpoint_paths[0])
    if task_name is not None:
        task_cfg = load_task_config(task_name)
        algo_cfg = load_algo_config()
    else:
        if hydra_cfg is None:
            raise ValueError("No --task provided and no .hydra/config.yaml found")
        algo_cfg = hydra_cfg["algorithm"]
        task_cfg = {"ENV_NAME": algo_cfg["ENV_NAME"],
                    "ENV_KWARGS": algo_cfg["ENV_KWARGS"],
                    "ROLLOUT_LENGTH": algo_cfg["ROLLOUT_LENGTH"]}
        task_name = hydra_cfg.get("TASK_NAME", algo_cfg["ENV_NAME"])

    eval_env_kwargs = dict(task_cfg["ENV_KWARGS"])
    if algo_cfg.get("COMMUNICATION", False):
        eval_env_kwargs["communication"] = True
    if task_cfg["ENV_NAME"] == "card-game":
        eval_env_kwargs["scramble_partner_msg"] = False
    env = make_env(task_cfg["ENV_NAME"], eval_env_kwargs)
    env = LogWrapper(env)

    # Initialize policy
    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, init_rng = jax.random.split(rng)
    policy, init_params = initialize_ja_image_agent(algo_cfg, env, init_rng)

    # Load one seed from each checkpoint
    seed_params = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        run_data = load_train_run(ckpt_path)
        params, params_key = _select_xp_params(run_data)
        # Take seed 0 from each checkpoint (each has 1 seed)
        params_0 = jax.tree.map(lambda x: x[0], params)
        num_params = sum(x.size for x in jax.tree.leaves(params_0))
        print(f"  checkpoint {i}: {ckpt_path} ({params_key}, {num_params} params)")
        seed_params.append(params_0)

    num_seeds = len(seed_params)
    print(f"[xp_seeds] multi-checkpoint mode: {num_seeds} seeds from {num_seeds} checkpoints")

    stacked_params = jax.tree.map(lambda *xs: jnp.stack(xs), *seed_params)

    # Eval over full episodes: ROLLOUT_LENGTH is the training chunk, which can be
    # shorter than an episode (ocv2: 256 < 400) so no episode completes and the
    # return metric stays empty. Use the env's episode length.
    max_steps = int(task_cfg.get("ENV_KWARGS", {}).get("max_steps") or task_cfg["ROLLOUT_LENGTH"])
    rng, eval_rng = jax.random.split(rng)
    outer_rngs = jax.random.split(eval_rng, num_seeds)
    action_sizes = {k: int(env.action_space(k).n) for k in env.agents}

    # Compute feed_attn_dims
    feed_attn = algo_cfg.get("FEED_OTHER_ATTN", False)
    feed_attn_dims = None
    if feed_attn:
        from agents.initialize_agents import _get_image_dims
        from agents.ja_actor_critic import _compute_resnet_output_dims
        _img_h, _img_w, _ = _get_image_dims(env)
        _feat_h, _feat_w = _compute_resnet_output_dims(
            _img_h, _img_w,
            stride=algo_cfg.get("CONV_STRIDE", 2),
            kernel_size=algo_cfg.get("CONV_KERNEL_SIZE", 3),
            padding=algo_cfg.get("CONV_PADDING", "SAME"),
            num_blocks=algo_cfg.get("CONV_NUM_BLOCKS", 4),
        )
        feed_attn_dims = (_img_h, _img_w, _feat_h, _feat_w)
        print("[xp_seeds] feed_other_attn enabled")

    ja_card_masks = None
    if algo_cfg.get("JA_CARD_ATTN", False) or algo_cfg.get("JA_CARD_METRIC", False):
        from agents.initialize_agents import _get_image_dims
        from agents.ja_actor_critic import _compute_resnet_output_dims
        from agents.ja_utils import build_card_masks
        _img_h, _img_w, _ = _get_image_dims(env)
        _feat_h, _feat_w = _compute_resnet_output_dims(
            _img_h, _img_w,
            stride=algo_cfg.get("CONV_STRIDE", 2),
            kernel_size=algo_cfg.get("CONV_KERNEL_SIZE", 3),
            padding=algo_cfg.get("CONV_PADDING", "SAME"),
            num_blocks=algo_cfg.get("CONV_NUM_BLOCKS", 4),
        )
        ja_card_masks = build_card_masks(_img_h, _img_w, _feat_h, _feat_w)

    # Ocv2 visibility gating: mask the feed exactly as at train time.
    feed_mask_fn = None
    if (feed_attn_dims is not None and task_cfg["ENV_NAME"] == "overcooked-v2"
            and algo_cfg.get("JA_VISIBILITY_GATING", False)):
        from agents.overcooked_v2.ja_overcooked_v2_attention import (
            make_visibility_mask_fn, overcooked_v2_object_ctx,
        )
        feed_mask_fn = make_visibility_mask_fn(overcooked_v2_object_ctx(algo_cfg, env))
        print("[xp_seeds] ocv2 visibility gating enabled for the feed channel")

    xp_partner_feed_dim = 5
    greedy_eval = True

    row_fn = jax.jit(lambda rng_i, p0: run_row_with_jsd(
        rng_i, env, p0, policy, stacked_params, policy, max_steps, NUM_EVAL_EPISODES, action_sizes,
        feed_attn_dims=feed_attn_dims, ja_card_masks=ja_card_masks, greedy_eval=greedy_eval,
        partner_feed_dim=xp_partner_feed_dim, feed_mask_fn=feed_mask_fn,
    ))

    all_row_metrics = []
    jsd_matrix = np.zeros((num_seeds, num_seeds, NUM_EVAL_EPISODES))
    start_time = time.time()
    # Prefer the env's sparse-return metric where present (ocv2 exposes
    # `base_return`; card/LBF fall back to `returned_episode_returns`).
    score_key = None
    for i in range(num_seeds):
        print(f"  row {i} (seed {i} vs all) ...", end=" ", flush=True)
        row_metrics, row_jsds, _row_card_jsds, _row_matches = row_fn(outer_rngs[i], seed_params[i])
        jsd_matrix[i] = np.array(row_jsds)
        all_row_metrics.append(row_metrics)
        if score_key is None:
            score_key = "base_return" if "base_return" in row_metrics else "returned_episode_returns"
        for j in range(num_seeds):
            ret = np.array(row_metrics[score_key][j]).mean()
            jsd_val = np.array(row_jsds[j]).mean()
            tag = "SP" if i == j else "XP"
            print(f"  [{tag}] {i}x{j}: return={ret:.2f} jsd={jsd_val:.4f}", end="")
        print()

    elapsed = time.time() - start_time
    print(f"[xp_seeds] evaluation done in {elapsed:.1f}s")

    # Build score matrix; reduce any trailing per-agent axis (ocv2 base_return
    # is per-agent, and the team reward makes both agents identical).
    score_matrix = np.zeros((num_seeds, num_seeds, NUM_EVAL_EPISODES))
    for i in range(num_seeds):
        arr = np.array(all_row_metrics[i][score_key])
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        score_matrix[i] = arr

    score_mean = score_matrix.mean(axis=-1)
    jsd_ep_means = jsd_matrix.mean(axis=-1)

    # Print summary
    sp_scores = np.diag(score_mean)
    sp_mean = np.mean(sp_scores)
    sp_sem = np.std(sp_scores) / np.sqrt(len(sp_scores))
    xp_mean, xp_sem = xp_mean_and_sem(score_mean)
    print(f"  Score: SP = {sp_mean:.4f} +/- {sp_sem:.4f}  |  XP = {xp_mean:.4f} +/- {xp_sem:.4f}")

    if task_cfg["ENV_NAME"] != "card-game":
        sp_jsd = np.diag(jsd_ep_means)
        sp_jsd_mean = np.mean(sp_jsd)
        sp_jsd_sem = np.std(sp_jsd) / np.sqrt(len(sp_jsd))
        xp_jsd_mean, xp_jsd_sem = xp_mean_and_sem(jsd_ep_means)
        print(f"  JSD:   SP = {sp_jsd_mean:.4f} +/- {sp_jsd_sem:.4f}  |  XP = {xp_jsd_mean:.4f} +/- {xp_jsd_sem:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-play evaluation across seeds")
    parser.add_argument("--task", default=None,
                        help="Task config name (default: inferred from Hydra config)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to saved_train_run directory (single multi-seed checkpoint)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Paths to multiple 1-seed checkpoints for multi-checkpoint XP")
    parser.add_argument("--use-best", action="store_true", default=True,
                        help="Use best_params when available (default; final_params is fallback)")
    parser.add_argument("--use-final", dest="use_best", action="store_false",
                        help="Force final_params instead of best_params")
    parser.add_argument("--sampled", action="store_true",
                        help="Stochastic (non-greedy) action selection for XP/SP "
                             "instead of the default argmax decoding.")
    parser.add_argument("--xp-video-max-pairs", type=int, default=None,
                        help="Override the cap on number of XP video pairs (default 3 from "
                             "config). Pass 0 (or any non-positive) to render every "
                             "off-diagonal pair.")
    parser.add_argument("--xp-video-all-pairs", action="store_true",
                        help="Convenience flag: render XP videos for every off-diagonal "
                             "pair. Equivalent to --xp-video-max-pairs 0.")
    parser.add_argument("--no-xp-videos", action="store_true",
                        help="Skip XP video rendering entirely (overrides "
                             "EVAL_VIDEO_LOG_XP from the saved Hydra config).")
    parser.add_argument("--no-op", action="store_true",
                        help="Remove Other-Play env wrappers at eval: cross-play in "
                             "the shared identity frame (deployment condition), not "
                             "under per-agent random relabelling. Writes to rerun_noop/.")
    parser.add_argument("--best-from-run", default=None,
                        help="Rundir (with .hydra/config.yaml): post-hoc XP on each "
                             "seed's best checkpoint, from chunk_scores.json + per-ckpt folders")
    parser.add_argument("--scores", default=None,
                        help="Path to chunk_scores.json (default: <best-from-run>/chunk_scores.json)")
    parser.add_argument("--wandb-resume", default=None,
                        help="wandb run id to resume and log XP_best/return_matrix into "
                             "(uses logger.project/entity from the run's config)")
    args = parser.parse_args()

    if args.best_from_run:
        run_xp_best_from_run(args.best_from_run, scores_path=args.scores,
                             wandb_resume=args.wandb_resume)
    elif args.checkpoints:
        run_xp_multi_checkpoint(args.task, args.checkpoints)
    elif args.checkpoint:
        max_pairs = 0 if args.xp_video_all_pairs else args.xp_video_max_pairs
        run_xp_evaluation(args.task, args.checkpoint, greedy_eval=not args.sampled,
                          use_best=args.use_best,
                          xp_video_max_pairs=max_pairs,
                          no_xp_videos=args.no_xp_videos,
                          no_op=args.no_op)
    else:
        parser.error("Either --checkpoint or --checkpoints is required")
