"""Cross-play evaluation for multi-seed training runs.

Loads final_params (shape: num_seeds, ...) from a single checkpoint,
builds the NxN cross-play matrix, and reports SP/XP with proper SEM
following the pairing scheme from the ZSC literature.

Reports two matrices: game score and JSD between attention maps.

Usage:
    uv run python -m evaluation.run_xp_seeds \
        --task overcooked-v1-image/cramped_room \
        --checkpoint results/.../saved_train_run
"""
import argparse
import csv
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
from common.plot_utils import get_metric_names
from common.save_load_utils import load_train_run
from common.tree_utils import tree_stack
from envs import make_env
from envs.card_game.rendering import NUM_CARDS
from envs.log_wrapper import LogWrapper
from evaluation.action_distributions import generate_action_distribution_artifacts
from marl.eval_card_game import _log_card_game_xp_videos


EVAL_SEED = 34957
NUM_EVAL_EPISODES = 1024
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs", "task")
ALGO_BASE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "marl", "configs", "algorithm", "ja_ippo", "_base_.yaml"
)


def load_task_config(task_name: str) -> dict:
    config_path = os.path.join(CONFIGS_DIR, f"{task_name}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_algo_config() -> dict:
    with open(ALGO_BASE_CONFIG) as f:
        return yaml.safe_load(f)


def _get_card_game_position_perm(state, agent_name: str):
    """Return per-agent OP position perm, or identity when OP shuffle is off."""
    s = state
    while s is not None:
        if hasattr(s, "per_agent_perm"):
            return s.per_agent_perm[agent_name]
        s = getattr(s, "env_state", None)
    return jnp.arange(NUM_CARDS, dtype=jnp.int32)


def run_single_episode_with_jsd(rng, env, agent_0_param, agent_0_policy,
                                agent_1_param, agent_1_policy,
                                max_episode_steps, action_sizes,
                                feed_attn_dims=None, ja_card_masks=None,
                                greedy_eval=True):
    """Run one eval episode, returning LogWrapper info + mean JSD between attention maps.

    Args:
        feed_attn_dims: if not None, (img_h, img_w, feat_h, feat_w) for obs augmentation
            with the other agent's previous attention map (4th channel).
    """
    from agents.ja_utils import augment_obs_for_eval

    _xattn = getattr(agent_0_policy, 'cross_agent_attn', False)
    _xnpos = getattr(agent_0_policy, 'xattn_num_positions', 0) if _xattn else 0
    _xfdim = getattr(agent_0_policy, 'xattn_feat_dim', 0) if _xattn else 0

    def _call_attn(policy, params, obs, done, avail, hstate, rng,
                   pe_a=None, pe_c=None, prev_rew=None, prev_act=None):
        """Wrapper returning (act, hstate, attn, own_a, own_c) uniformly."""
        if _xattn:
            act, hs, attn, oa, oc = policy.get_action_and_attention(
                params=params, obs=obs, done=done, avail_actions=avail,
                hstate=hstate, rng=rng, greedy=greedy_eval,
                partner_embed_actor=pe_a, partner_embed_critic=pe_c,
            )
            return act, hs, attn, oa, oc
        else:
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
    init_act_onehot = {k: jnp.zeros((action_sizes[k],))
                       for k in env.agents}

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
    if _ja_card:
        prev_pca_0 = jnp.zeros(5)  # partner card attention for agent 0
        prev_pca_1 = jnp.zeros(5)

    # Initialize uniform attention maps for feed_other_attn
    if feed_attn_dims is not None:
        _img_h, _img_w, _feat_h, _feat_w = feed_attn_dims
        prev_attn_0 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)
        prev_attn_1 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)

    # Initialize partner embeddings for cross-agent attention
    if _xattn:
        pe_a0 = jnp.zeros((1, 1, _xnpos, _xfdim))
        pe_c0 = jnp.zeros((1, 1, _xnpos, _xfdim))
        pe_a1 = jnp.zeros((1, 1, _xnpos, _xfdim))
        pe_c1 = jnp.zeros((1, 1, _xnpos, _xfdim))
    else:
        pe_a0 = pe_c0 = pe_a1 = pe_c1 = jnp.zeros((1, 1, 1))

    avail_actions = env.get_avail_actions(init_env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
    avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

    # First step
    rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)

    obs_0 = init_obs["agent_0"]
    obs_1 = init_obs["agent_1"]
    if feed_attn_dims is not None:
        obs_0 = augment_obs_for_eval(obs_0, prev_attn_1, _img_h, _img_w)
        obs_1 = augment_obs_for_eval(obs_1, prev_attn_0, _img_h, _img_w)
    if _ja_card:
        obs_0 = jnp.concatenate([obs_0, prev_pca_0])
        obs_1 = jnp.concatenate([obs_1, prev_pca_1])

    act_0, hstate_0, attn_0, oa0, oc0 = _call_attn(
        agent_0_policy, agent_0_param,
        obs_0.reshape(1, 1, -1), init_done["agent_0"].reshape(1, 1),
        avail_actions_0, init_hstate_0, act0_rng,
        pe_a=pe_a0, pe_c=pe_c0,
        prev_rew=prev_reward_0 if use_prev_io else None,
        prev_act=prev_action_0 if use_prev_io else None,
    )
    act_0 = act_0.squeeze()

    act_1, hstate_1, attn_1, oa1, oc1 = _call_attn(
        agent_1_policy, agent_1_param,
        obs_1.reshape(1, 1, -1), init_done["agent_1"].reshape(1, 1),
        avail_actions_1, init_hstate_1, act1_rng,
        pe_a=pe_a1, pe_c=pe_c1,
        prev_rew=prev_reward_1 if use_prev_io else None,
        prev_act=prev_action_1 if use_prev_io else None,
    )
    act_1 = act_1.squeeze()
    # Swap partner embeddings
    pe_a0, pe_a1 = oa1, oa0
    pe_c0, pe_c1 = oc1, oc0

    # Flatten attention maps to distributions for JSD
    attn_0_flat = attn_0.reshape(-1)
    attn_1_flat = attn_1.reshape(-1)
    attn_0_dist = attn_0_flat / (attn_0_flat.sum() + 1e-8)
    attn_1_dist = attn_1_flat / (attn_1_flat.sum() + 1e-8)
    # jsd_divergence expects (..., H, W) — reshape back
    h, w = attn_0.shape[-2], attn_0.shape[-1]
    step_jsd = jsd_divergence(attn_0_dist.reshape(h, w), attn_1_dist.reshape(h, w))
    jsd_sum = step_jsd
    jsd_count = jnp.array(1.0)

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

    if _ja_card:
        a0_sq = attn_0.squeeze()
        a1_sq = attn_1.squeeze()
        ca_0 = jnp.einsum("hw,chw->c", a0_sq, ja_card_masks)
        ca_1 = jnp.einsum("hw,chw->c", a1_sq, ja_card_masks)
        perm_0 = _get_card_game_position_perm(init_env_state, "agent_0")
        perm_1 = _get_card_game_position_perm(init_env_state, "agent_1")
        ph_0 = jnp.zeros(5).at[perm_0].set(ca_0)
        ph_1 = jnp.zeros(5).at[perm_1].set(ca_1)
        prev_pca_0 = ph_1[perm_0]
        prev_pca_1 = ph_0[perm_1]

    ep_ts = 1
    init_carry = (ep_ts, env_state, obs, rng, done, reward, env_act_onehot,
                  hstate_0, hstate_1, dummy_info, jsd_sum, jsd_count,
                  prev_reward_0 if use_prev_io else None,
                  prev_reward_1 if use_prev_io else None,
                  prev_action_0 if use_prev_io else None,
                  prev_action_1 if use_prev_io else None,
                  attn_0.squeeze(), attn_1.squeeze(),
                  pe_a0, pe_c0, pe_a1, pe_c1,
                  prev_pca_0 if _ja_card else jnp.zeros(5),
                  prev_pca_1 if _ja_card else jnp.zeros(5))

    def scan_step(carry, _):
        def take_step(carry_step):
            (ep_ts, env_state, obs, rng, done, reward, act_onehot,
             hstate_0, hstate_1, last_info, jsd_sum, jsd_count,
             prev_reward_0, prev_reward_1, prev_action_0, prev_action_1,
             prev_a0, prev_a1,
             pe_a0, pe_c0, pe_a1, pe_c1,
             prev_pca_0, prev_pca_1) = carry_step

            avail_actions = env.get_avail_actions(env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            rng, act0_rng, act1_rng, step_rng = jax.random.split(rng, 4)

            obs_0 = obs["agent_0"]
            obs_1 = obs["agent_1"]
            if feed_attn_dims is not None:
                obs_0 = augment_obs_for_eval(obs_0, prev_a1, _img_h, _img_w)
                obs_1 = augment_obs_for_eval(obs_1, prev_a0, _img_h, _img_w)
            if _ja_card:
                obs_0 = jnp.concatenate([obs_0, prev_pca_0])
                obs_1 = jnp.concatenate([obs_1, prev_pca_1])

            act_0, hstate_0_next, attn_0, oa0, oc0 = _call_attn(
                agent_0_policy, agent_0_param,
                obs_0.reshape(1, 1, -1), done["agent_0"].reshape(1, 1),
                avail_actions_0, hstate_0, act0_rng,
                pe_a=pe_a0, pe_c=pe_c0,
                prev_rew=prev_reward_0 if use_prev_io else None,
                prev_act=prev_action_0 if use_prev_io else None,
            )
            act_0 = act_0.squeeze()

            act_1, hstate_1_next, attn_1, oa1, oc1 = _call_attn(
                agent_1_policy, agent_1_param,
                obs_1.reshape(1, 1, -1), done["agent_1"].reshape(1, 1),
                avail_actions_1, hstate_1, act1_rng,
                pe_a=pe_a1, pe_c=pe_c1,
                prev_rew=prev_reward_1 if use_prev_io else None,
                prev_act=prev_action_1 if use_prev_io else None,
            )
            act_1 = act_1.squeeze()

            # Compute JSD between attention maps
            a0 = attn_0.reshape(-1)
            a1 = attn_1.reshape(-1)
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

            # Update JA card partner attention
            if _ja_card:
                ca0 = jnp.einsum("hw,chw->c", attn_0.squeeze(), ja_card_masks)
                ca1 = jnp.einsum("hw,chw->c", attn_1.squeeze(), ja_card_masks)
                p0 = _get_card_game_position_perm(env_state, "agent_0")
                p1 = _get_card_game_position_perm(env_state, "agent_1")
                ph0 = jnp.zeros(5).at[p0].set(ca0)
                ph1 = jnp.zeros(5).at[p1].set(ca1)
                next_pca_0 = ph1[p0]
                next_pca_1 = ph0[p1]
            else:
                next_pca_0 = jnp.zeros(5)
                next_pca_1 = jnp.zeros(5)

            return (ep_ts + 1, env_state_next, obs_next, rng, done_next, reward, env_act_onehot,
                    hstate_0_next, hstate_1_next, info_next, jsd_sum_next, jsd_count_next,
                    next_prev_reward_0, next_prev_reward_1, next_prev_action_0, next_prev_action_1,
                    attn_0.squeeze(), attn_1.squeeze(),
                    oa1, oc1, oa0, oc0,
                    next_pca_0, next_pca_1)

        (ep_ts, env_state, obs, rng, done, reward, act_onehot,
         hstate_0, hstate_1, last_info, jsd_sum, jsd_count,
         prev_reward_0, prev_reward_1, prev_action_0, prev_action_1,
         prev_a0, prev_a1,
         pe_a0, pe_c0, pe_a1, pe_c1,
         prev_pca_0, prev_pca_1) = carry
        new_carry = jax.lax.cond(
            done["__all__"],
            lambda curr_carry: curr_carry,
            take_step,
            operand=carry,
        )
        return new_carry, None

    final_carry, _ = jax.lax.scan(scan_step, init_carry, None, length=max_episode_steps)
    # Carry layout: (ep_ts, env_state, obs, rng, done, reward, act_onehot,
    #   hstate_0, hstate_1, last_info, jsd_sum, jsd_count, ...)
    info = final_carry[9]       # last_info
    mean_jsd = final_carry[10] / final_carry[11]  # jsd_sum / jsd_count
    return info, mean_jsd


def run_episodes_with_jsd(rng, env, agent_0_param, agent_0_policy,
                          agent_1_param, agent_1_policy,
                          max_episode_steps, num_eps, action_sizes,
                          feed_attn_dims=None, ja_card_masks=None,
                          greedy_eval=True):
    """Run num_eps episodes in parallel, returning LogWrapper info + per-episode mean JSD."""
    rngs = jax.random.split(rng, num_eps + 1)
    ep_rngs = rngs[1:]

    vmap_fn = jax.vmap(
        lambda ep_rng: run_single_episode_with_jsd(
            ep_rng, env, agent_0_param, agent_0_policy,
            agent_1_param, agent_1_policy, max_episode_steps, action_sizes,
            feed_attn_dims=feed_attn_dims, ja_card_masks=ja_card_masks,
            greedy_eval=greedy_eval,
        )
    )
    all_info, all_jsd = vmap_fn(ep_rngs)
    return all_info, all_jsd  # all_jsd shape: (num_eps,)


def run_row_with_jsd(rng, env, agent_0_param, agent_0_policy,
                     all_agent_1_params, agent_1_policy,
                     max_episode_steps, num_eps, action_sizes,
                     feed_attn_dims=None, ja_card_masks=None,
                     greedy_eval=True):
    """Run one row of the XP matrix: agent_0 vs all partners, vmapped over partners and episodes."""
    num_partners = jax.tree.leaves(all_agent_1_params)[0].shape[0]
    partner_rngs = jax.random.split(rng, num_partners)

    # vmap over partners (j dimension)
    def eval_one_partner(partner_rng, agent_1_param):
        return run_episodes_with_jsd(
            partner_rng, env, agent_0_param, agent_0_policy,
            agent_1_param, agent_1_policy, max_episode_steps, num_eps, action_sizes,
            feed_attn_dims=feed_attn_dims, ja_card_masks=ja_card_masks,
            greedy_eval=greedy_eval,
        )

    return jax.vmap(eval_one_partner)(partner_rngs, all_agent_1_params)


def xp_mean_and_sem(xp_matrix):
    """Compute XP mean and SEM over all off-diagonal entries.

    Uses std of all off-diagonal values but divides by sqrt(n) (number of
    independent seeds) rather than sqrt(n*(n-1)) since entries sharing a
    seed are correlated.

    Args:
        xp_matrix: (n, n) array where entry (i,j) is the mean metric
                   when seed i is agent 0 and seed j is agent 1.
    Returns:
        (mean, sem) over off-diagonal entries with n-based SEM.
    """
    n = xp_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = xp_matrix[mask]
    return np.mean(off_diag), np.std(off_diag) / np.sqrt(n)


def save_xp_heatmap(matrix_mean: np.ndarray, matrix_std: np.ndarray,
                     title: str, filepath: str, fmt: str = ".2f",
                     cmap: str = "YlOrRd", vmin: float | None = None,
                     vmax: float | None = None):
    """Save an annotated NxN heatmap as PNG."""
    n = matrix_mean.shape[0]
    fig, ax = plt.subplots(figsize=(1.5 + n * 1.2, 1.0 + n * 1.0))
    im = ax.imshow(matrix_mean, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    # Annotate cells with mean ± std
    for i in range(n):
        for j in range(n):
            m, s = matrix_mean[i, j], matrix_std[i, j]
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
    if algo_cfg.get("FILTER_ATTN_TOP1", False):
        parts.append("top1")
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
                      task_name, run_dir, wb_run=None, wb_prefix="XP"):
    """Log XP results to wandb. Creates a new run if `wb_run` is None."""
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
    wandb.save(os.path.join(xp_dir, "xp_jsd_matrix.csv"), base_path=xp_dir)

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
    # Stochastic XP has been retired; keep the parameter for compatibility.
    greedy_eval = True

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

    max_steps = int(algo_cfg.get("ROLLOUT_LENGTH", algo_cfg.get("ENV_KWARGS", {}).get("max_steps", 400)))
    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, eval_rng = jax.random.split(rng)
    outer_rngs = jax.random.split(eval_rng, num_seeds)

    action_sizes = {k: int(env.action_space(k).n) for k in env.agents}

    # Compute feed_attn_dims if needed
    feed_attn = algo_cfg.get("FEED_OTHER_ATTN", False)
    feed_attn_dims = None
    if feed_attn:
        from agents.initialize_agents import _get_image_dims
        from agents.ja_image_actor_critic import _compute_resnet_output_dims
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
    if algo_cfg.get("JA_CARD_ATTN", False):
        from agents.initialize_agents import _get_image_dims
        from agents.ja_image_actor_critic import _compute_resnet_output_dims
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

    row_fn = jax.jit(lambda rng_i, p0: run_row_with_jsd(
        rng_i, env, p0, policy, stacked_params, policy, max_steps, NUM_EVAL_EPISODES, action_sizes,
        feed_attn_dims=feed_attn_dims, ja_card_masks=ja_card_masks, greedy_eval=greedy_eval,
    ))

    all_row_metrics = []
    jsd_matrix = np.zeros((num_seeds, num_seeds, NUM_EVAL_EPISODES))
    start_time = time.time()
    for i in range(num_seeds):
        print(f"  row {i} (seed {i} vs all) ...", end=" ", flush=True)
        row_metrics, row_jsds = row_fn(outer_rngs[i], seed_params[i])
        jsd_matrix[i] = np.array(row_jsds)
        all_row_metrics.append(row_metrics)

        for j in range(num_seeds):
            ret = np.array(row_metrics["returned_episode_returns"][j]).mean()
            jsd_mean = float(row_jsds[j].mean())
            label = "SP" if i == j else "XP"
            print(f"  [{label}] {i}x{j}: return={ret:.2f} jsd={jsd_mean:.4f}", end="")
        print()

    xp_metrics = tree_stack(all_row_metrics)
    elapsed = time.time() - start_time
    print(f"[xp_seeds] evaluation done in {elapsed:.1f}s")

    metric_names = get_metric_names(env_name)
    seed_names = [f"seed_{i}" for i in range(num_seeds)]
    for metric_name in metric_names:
        print_xp_table(xp_metrics, metric_name, seed_names)

    print_jsd_table(jsd_matrix, seed_names)
    print_sp_vs_xp_summary(xp_metrics, metric_names, jsd_matrix, num_seeds)

    # Save heatmaps and CSVs
    beta = algo_cfg.get("JA_BETA_MAX", "unknown")
    beta_prefix = f"BETA{beta}"

    xp_dir = os.path.join(savedir, "xp_results")
    os.makedirs(xp_dir, exist_ok=True)

    central_xp_dir = os.path.join(savedir, "..", "xp_results")
    os.makedirs(central_xp_dir, exist_ok=True)

    score_mean = score_std = None
    if "base_return" in xp_metrics:
        score_data = np.array(xp_metrics["base_return"]).mean(axis=-1)
        score_mean = score_data.mean(axis=-1)
        score_std = score_data.std(axis=-1)
        for d in (xp_dir, central_xp_dir):
            prefix = "" if d == xp_dir else f"{beta_prefix}_"
            save_xp_heatmap(score_mean, score_std,
                             f"XP Episode Return — {run_label}",
                             os.path.join(d, f"{prefix}xp_score_matrix.png"))
            save_xp_csv(score_mean, score_std,
                         os.path.join(d, f"{prefix}xp_score_matrix.csv"), label="episode_return")

    jsd_mean = jsd_matrix.mean(axis=-1)
    jsd_std = jsd_matrix.std(axis=-1)
    for d in (xp_dir, central_xp_dir):
        prefix = "" if d == xp_dir else f"{beta_prefix}_"
        save_xp_heatmap(jsd_mean, jsd_std,
                         f"XP JSD — {run_label}",
                         os.path.join(d, f"{prefix}xp_jsd_matrix.png"),
                         fmt=".4f", cmap="YlGnBu", vmin=0.0, vmax=0.693)
        save_xp_csv(jsd_mean, jsd_std,
                     os.path.join(d, f"{prefix}xp_jsd_matrix.csv"), label="jsd")

    print(f"[xp_seeds] results saved to {xp_dir} and {central_xp_dir}")

    if env_name == "card-game":
        action_dist_dir = os.path.join(xp_dir, "action_distributions")
        ad_num_eps = int(algo_cfg.get("XP_ACTION_DIST_NUM_EPISODES", 50))
        seed_indices = list(range(num_seeds))
        print(
            f"[xp_seeds] logging card-game action distributions "
            f"(greedy + sampled, {ad_num_eps} eps/seed)"
        )
        for greedy_mode, mode_name in ((True, "greedy"), (False, "sampled")):
            generate_action_distribution_artifacts(
                inner_env=env._env,
                stacked_params=stacked_params,
                policy=policy,
                max_steps=max_steps,
                output_dir=os.path.join(action_dist_dir, mode_name),
                seed_indices=seed_indices,
                num_episodes=ad_num_eps,
                feed_attn_dims=feed_attn_dims,
                ja_card_masks=ja_card_masks,
                greedy=greedy_mode,
                wb_run=wb_run,
                wb_prefix=wb_prefix,
            )

        max_pairs = int(algo_cfg.get("XP_VIDEO_MAX_PAIRS", 3))
        xp_video_eps = int(algo_cfg.get("EVAL_VIDEO_XP_NUM_EPISODES", 3))
        seed_pairs = [
            (i, j) for i in range(num_seeds) for j in range(i + 1, num_seeds)
        ][:max_pairs]
        if seed_pairs:
            print(
                f"[xp_seeds] logging card-game XP videos for pairs {seed_pairs} "
                f"({xp_video_eps} eps/pair)"
            )

            class _WandbVideoLogger:
                def __init__(self, run):
                    self.run = run

                def log_video(self, tag, path, commit=True):
                    import wandb

                    self.run.log({tag: wandb.Video(path, format="mp4")}, commit=commit)

            xp_video_dir = os.path.join(xp_dir, "videos")
            os.makedirs(xp_video_dir, exist_ok=True)
            _log_card_game_xp_videos(
                env._env, policy, stacked_params, max_steps,
                f"{wb_prefix}/videos", xp_video_dir, _WandbVideoLogger(wb_run),
                feed_attn_dims=feed_attn_dims,
                ja_card_masks=ja_card_masks,
                filter_top1=bool(algo_cfg.get("FILTER_ATTN_TOP1", False)),
                seed_pairs=seed_pairs,
                num_episodes=xp_video_eps,
                fps=3,
            )

    _log_xp_to_wandb(jsd_matrix, score_mean, xp_dir, algo_cfg,
                      task_name, savedir, wb_run=wb_run, wb_prefix=wb_prefix)

    if created_wb_run:
        wb_run.finish()
        print(f"[xp_seeds] wandb run: {wb_run.url}")


def run_xp_evaluation(task_name: str | None, checkpoint_path: str, greedy_eval: bool = True,
                      use_best: bool = False, drop_op: bool = False,
                      wb_prefix: str | None = None,
                      xp_video_max_pairs: int | None = None):
    """Standalone XP evaluation from a saved checkpoint.

    `use_best` selects `best_params` over `final_params` (per-seed best checkpoint).
    `drop_op` forces `other_play_*` env_kwargs off so OP wrappers are not applied at eval —
    useful for probing whether agents trained under OP actually generalize, or just collapsed
    to a fixed convention in the recoloured/permuted frame. A fresh wandb run is created with
    `wb_prefix` (default `XP_NO_OP` when `drop_op` else `XP`); the original training run is
    untouched.
    """
    greedy_eval = True
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
    if drop_op:
        env_kwargs["other_play_position_shuffle"] = False
        env_kwargs["other_play_recolouring"] = False
        # Keep env-internal shuffle on so SP/XP without OP still varies card layout per episode
        # (matches what the env would have done in a no-OP training run).
        env_kwargs["shuffle"] = True
        print("[xp_seeds] --drop-op: OP wrappers disabled at eval; env shuffle=True")
    env = make_env(task_cfg["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    run_data = load_train_run(checkpoint_path)
    params_key = "best_params" if use_best else "final_params"
    if params_key not in run_data:
        raise KeyError(f"{params_key!r} not found in checkpoint; keys: {list(run_data.keys())}")
    all_final_params = run_data[params_key]
    print(f"[xp_seeds] using {params_key} from checkpoint")

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, init_rng = jax.random.split(rng)
    policy, _init_params = initialize_ja_image_agent(algo_cfg, env, init_rng)

    run_dir = os.path.dirname(checkpoint_path)
    if wb_prefix is None:
        wb_prefix = "XP_NO_OP" if drop_op else "XP"
    # Avoid overwriting the original training run's xp_results/ when re-evaluating with overrides.
    savedir = run_dir
    if drop_op or use_best:
        suffix_parts = []
        if drop_op:
            suffix_parts.append("no_op")
        if use_best:
            suffix_parts.append("best")
        savedir = os.path.join(run_dir, "rerun_" + "_".join(suffix_parts))
        os.makedirs(savedir, exist_ok=True)
        print(f"[xp_seeds] writing rerun outputs to {savedir}")
    if xp_video_max_pairs is not None:
        # 0 or negative => render every off-diagonal pair (will be clipped to N*(N-1)/2 by
        # the slice in run_xp_from_params).
        effective = xp_video_max_pairs if xp_video_max_pairs > 0 else 10**9
        label_cfg["XP_VIDEO_MAX_PAIRS"] = effective
        print(f"[xp_seeds] XP_VIDEO_MAX_PAIRS overridden to {effective}")
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
    """Report SP and XP with proper SEM using the seed-pairing scheme."""
    print("\n=== Self-Play vs Cross-Play Summary ===")
    m = num_seeds // 2
    print(f"  ({num_seeds} seeds -> {m} independent XP samples)")
    if num_seeds % 2 != 0:
        print(f"  WARNING: odd number of seeds, last seed excluded from SEM computation")

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

    # JSD summary
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
        params = run_data["final_params"]
        # Take seed 0 from each checkpoint (each has 1 seed)
        params_0 = jax.tree.map(lambda x: x[0], params)
        num_params = sum(x.size for x in jax.tree.leaves(params_0))
        print(f"  checkpoint {i}: {ckpt_path} ({num_params} params)")
        seed_params.append(params_0)

    num_seeds = len(seed_params)
    print(f"[xp_seeds] multi-checkpoint mode: {num_seeds} seeds from {num_seeds} checkpoints")

    stacked_params = jax.tree.map(lambda *xs: jnp.stack(xs), *seed_params)

    max_steps = task_cfg["ROLLOUT_LENGTH"]
    rng, eval_rng = jax.random.split(rng)
    outer_rngs = jax.random.split(eval_rng, num_seeds)
    action_sizes = {k: int(env.action_space(k).n) for k in env.agents}

    # Compute feed_attn_dims
    feed_attn = algo_cfg.get("FEED_OTHER_ATTN", False)
    feed_attn_dims = None
    if feed_attn:
        from agents.initialize_agents import _get_image_dims
        from agents.ja_image_actor_critic import _compute_resnet_output_dims
        _img_h, _img_w, _ = _get_image_dims(env)
        _feat_h, _feat_w = _compute_resnet_output_dims(
            _img_h, _img_w,
            stride=algo_cfg.get("CONV_STRIDE", 2),
            kernel_size=algo_cfg.get("CONV_KERNEL_SIZE", 3),
            padding=algo_cfg.get("CONV_PADDING", "SAME"),
            num_blocks=algo_cfg.get("CONV_NUM_BLOCKS", 4),
        )
        feed_attn_dims = (_img_h, _img_w, _feat_h, _feat_w)
        print(f"[xp_seeds] feed_other_attn enabled")

    ja_card_masks = None
    if algo_cfg.get("JA_CARD_ATTN", False):
        from agents.initialize_agents import _get_image_dims
        from agents.ja_image_actor_critic import _compute_resnet_output_dims
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

    row_fn = jax.jit(lambda rng_i, p0: run_row_with_jsd(
        rng_i, env, p0, policy, stacked_params, policy, max_steps, NUM_EVAL_EPISODES, action_sizes,
        feed_attn_dims=feed_attn_dims, ja_card_masks=ja_card_masks, greedy_eval=greedy_eval,
    ))

    all_row_metrics = []
    jsd_matrix = np.zeros((num_seeds, num_seeds, NUM_EVAL_EPISODES))
    start_time = time.time()
    for i in range(num_seeds):
        print(f"  row {i} (seed {i} vs all) ...", end=" ", flush=True)
        row_metrics, row_jsds = row_fn(outer_rngs[i], seed_params[i])
        jsd_matrix[i] = np.array(row_jsds)
        all_row_metrics.append(row_metrics)
        for j in range(num_seeds):
            ret = np.array(row_metrics["returned_episode_returns"][j]).mean()
            jsd_val = np.array(row_jsds[j]).mean()
            tag = "SP" if i == j else "XP"
            print(f"  [{tag}] {i}x{j}: return={ret:.2f} jsd={jsd_val:.4f}", end="")
        print()

    elapsed = time.time() - start_time
    print(f"[xp_seeds] evaluation done in {elapsed:.1f}s")

    # Build score matrix
    score_matrix = np.zeros((num_seeds, num_seeds, NUM_EVAL_EPISODES))
    for i in range(num_seeds):
        score_matrix[i] = np.array(all_row_metrics[i]["returned_episode_returns"])

    score_mean = score_matrix.mean(axis=-1)
    score_std = score_matrix.std(axis=-1)
    jsd_ep_means = jsd_matrix.mean(axis=-1)

    # Print summary
    sp_scores = np.diag(score_mean)
    sp_mean = np.mean(sp_scores)
    sp_sem = np.std(sp_scores) / np.sqrt(len(sp_scores))
    xp_mean, xp_sem = xp_mean_and_sem(score_mean)
    print(f"  Score: SP = {sp_mean:.4f} +/- {sp_sem:.4f}  |  XP = {xp_mean:.4f} +/- {xp_sem:.4f}")

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
    parser.add_argument("--use-best", action="store_true",
                        help="Use best_params (per-seed best checkpoint) instead of final_params")
    parser.add_argument("--drop-op", action="store_true",
                        help="Disable Other-Play wrappers at eval (overrides ENV_KWARGS)")
    parser.add_argument("--xp-video-max-pairs", type=int, default=None,
                        help="Override the cap on number of XP video pairs (default 3 from "
                             "config). Pass 0 (or any non-positive) to render every "
                             "off-diagonal pair.")
    parser.add_argument("--xp-video-all-pairs", action="store_true",
                        help="Convenience flag: render XP videos for every off-diagonal "
                             "pair. Equivalent to --xp-video-max-pairs 0.")
    args = parser.parse_args()

    if args.checkpoints:
        if args.use_best or args.drop_op:
            parser.error("--use-best/--drop-op are only supported with --checkpoint (single multi-seed run)")
        run_xp_multi_checkpoint(args.task, args.checkpoints)
    elif args.checkpoint:
        max_pairs = 0 if args.xp_video_all_pairs else args.xp_video_max_pairs
        run_xp_evaluation(args.task, args.checkpoint, greedy_eval=True,
                          use_best=args.use_best, drop_op=args.drop_op,
                          xp_video_max_pairs=max_pairs)
    else:
        parser.error("Either --checkpoint or --checkpoints is required")
