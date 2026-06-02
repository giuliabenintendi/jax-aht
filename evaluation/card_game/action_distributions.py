"""Per-agent action-distribution diagnostic for the card game.

Loads a saved training checkpoint, runs N self-play eval episodes per seed, and
tallies — separately for messages (deliberation steps) and picks (decision step)
— the distribution of each agent's actions across:

  - view-position: column 0..NUM_CARDS-1 in the agent's own OP-shuffled view.
    Concentration here means the agent has learned a positional convention
    (always pick column 2 in my view, regardless of the cards there).
  - view-color: the appearance color CARD_COLORS[recolouring[gt]] the agent sees
    for that action — i.e. the colour as it appears in the agent's own
    obs after OP recolouring. Concentration means the agent picks based on a
    fixed apparent colour (always pick the orange card I see), not on identity.
  - gt-color: canonical card identity 0..NUM_CARDS-1 (the physical card colour
    after stripping OP recolouring). Diagnostic: under OP this should be near-
    uniform regardless of policy — `recolouring` is sampled independently each
    episode so any agent (even a constant one in view-frame) maps to a uniform
    GT distribution. A spike here while OP is supposedly on means the inverse-
    map is broken or the wrapper isn't applied somewhere.

Under OP, view-position and view-color concentration flags a trivial convention;
gt-color concentration flags a wiring bug.

Usage:
    ./run_gpu.sh <gpu> evaluation.card_game.action_distributions \\
        --checkpoint /path/to/saved_train_run \\
        --all-seeds \\
        --num-episodes 50 \\
        --output-dir plots/card_game/<run_name>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_agent,
    initialize_ja_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import CARD_COLORS, NUM_CARDS
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import run_episode_with_states


def _find_state_with_attr(state, attr: str):
    s = state
    while s is not None:
        if hasattr(s, attr):
            return s
        s = getattr(s, "env_state", None)
    return None


def _per_agent_perm(state, agent_idx: int) -> np.ndarray:
    name = f"agent_{agent_idx}"
    perm_state = _find_state_with_attr(state, "per_agent_perm")
    if perm_state is None:
        return np.arange(NUM_CARDS, dtype=np.int32)
    return np.asarray(perm_state.per_agent_perm[name])


def _per_agent_recolouring(state, agent_idx: int) -> np.ndarray:
    """recolouring[gt_idx] = appearance index the agent sees for GT card gt_idx."""
    name = f"agent_{agent_idx}"
    recol_state = _find_state_with_attr(state, "per_agent_recolouring")
    if recol_state is None:
        return np.arange(NUM_CARDS, dtype=np.int32)
    return np.asarray(recol_state.per_agent_recolouring[name])


def _card_permutation(state) -> np.ndarray:
    card_state = _find_state_with_attr(state, "card_permutation")
    if card_state is None:
        return np.arange(NUM_CARDS, dtype=np.int32)
    return np.asarray(card_state.card_permutation)


def _gt_to_view_col(state, pos_perm: np.ndarray, gt_value: int):
    if gt_value < 0:
        return None
    perm_state = _find_state_with_attr(state, "per_agent_perm")
    if perm_state is None:
        card_perm = _card_permutation(state)
        matches = np.where(card_perm == gt_value)[0]
        return int(matches[0]) if len(matches) else None
    matches = np.where(pos_perm == gt_value)[0]
    return int(matches[0]) if len(matches) else None


_META_KEYS = ("__match_count__", "__episode_count__")


def _empty_dist():
    d = {(a, t): {"view_pos": [], "view_color": [], "gt_color": []}
         for a in (0, 1) for t in ("msg", "pick")}
    for k in _META_KEYS:
        d[k] = 0
    return d


def _state_for_recorded_action(ep_states, t: int, is_decision: bool):
    """Return the wrapper state matching the action recorded at step ``t``.

    ``run_episode_with_states`` appends ``ep_states`` after ``env.step``. On the
    terminal decision step, the OP wrappers auto-reset and resample the next
    episode's position/recolouring transforms. The final pick therefore must be
    interpreted using the previous state's transforms, which are still current
    at decision time.
    """
    if is_decision and t > 0:
        return ep_states[t - 1]
    return ep_states[t]


def _collect_one_seed(rng_key, inner_env, params, policy, max_steps,
                      num_episodes, feed_attn_dims, ja_card_masks,
                      greedy=True, params_partner=None):
    """Collect action-distribution samples.

    `params_partner` (optional) lets the caller pair `params` (agent 0) with a
    different partner (agent 1) for cross-play collection; defaults to `params`
    for self-play.
    """
    if params_partner is None:
        params_partner = params
    out = _empty_dist()
    rngs = jax.random.split(rng_key, num_episodes)
    for ep in range(num_episodes):
        ep_states, _, ep_actions, ep_messages = run_episode_with_states(
            rngs[ep], inner_env, params, policy, params_partner, policy, max_steps,
            collect_attention=True,  # required by run_episode_with_states API
            greedy=greedy,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
        )
        # `ep_messages` is only populated when the env has `communication=True`
        # (the dot-rendering flag). In delib-actions mode comm is off but
        # deliberation actions still exist — they are recorded in `ep_actions`
        # at every non-final step, with the final step being the pick.
        has_messages = len(ep_messages) > 0
        n_steps = len(ep_actions)
        for t in range(n_steps):
            is_decision = (t == n_steps - 1)
            action_state = _state_for_recorded_action(ep_states, t, is_decision)
            for ai in (0, 1):
                if is_decision:
                    gt = int(ep_actions[t][ai])
                elif has_messages:
                    gt = int(ep_messages[t][ai])
                else:
                    gt = int(ep_actions[t][ai])
                if gt < 0:
                    continue
                pos_perm = _per_agent_perm(action_state, ai)
                vc_pos = _gt_to_view_col(action_state, pos_perm, gt)
                if vc_pos is None:
                    continue
                recol = _per_agent_recolouring(action_state, ai)
                vc_color = int(recol[gt])
                key = (ai, "pick" if is_decision else "msg")
                out[key]["view_pos"].append(vc_pos)
                out[key]["view_color"].append(vc_color)
                out[key]["gt_color"].append(int(gt))
        # Track per-episode GT-frame match (= env's success condition).
        if n_steps > 0:
            pick_0 = int(ep_actions[n_steps - 1][0])
            pick_1 = int(ep_actions[n_steps - 1][1])
            if pick_0 >= 0 and pick_1 >= 0:
                out["__episode_count__"] += 1
                if pick_0 == pick_1:
                    out["__match_count__"] += 1
    return out


def _hist_norm(arr) -> np.ndarray:
    if not len(arr):
        return np.zeros(NUM_CARDS)
    h = np.bincount(np.asarray(arr, dtype=int), minlength=NUM_CARDS).astype(float)
    return h / h.sum()


def _merge(*dicts):
    out = _empty_dist()
    for d in dicts:
        for k, v in d.items():
            if k in _META_KEYS:
                out[k] += v
                continue
            out[k]["view_pos"].extend(v["view_pos"])
            out[k]["view_color"].extend(v["view_color"])
            out[k]["gt_color"].extend(v["gt_color"])
    return out


def _plot(out_path: Path, dists, title: str):
    """Plot per-agent action distributions side-by-side: A0 | A1, single row.
    Each panel shows the combined msg + pick distribution in view-color
    (agent-frame). GT-frame and position breakdowns are intentionally
    omitted; msg and pick are merged into one distribution per agent."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    rgb_card_colors = np.asarray(CARD_COLORS) / 255.0
    xs = np.arange(NUM_CARDS)
    for ai in (0, 1):
        ax = axes[ai]
        msg_arr = list(dists[(ai, "msg")]["view_color"])
        pick_arr = list(dists[(ai, "pick")]["view_color"])
        combined = msg_arr + pick_arr
        h = _hist_norm(combined)
        ax.bar(xs, h, color=rgb_card_colors, edgecolor="black", linewidth=0.4)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(xs)
        ax.set_title(f"agent_{ai}", fontsize=10)
        if ai == 0:
            ax.set_ylabel("fraction", fontsize=10)
        ax.text(0.98, 0.97, f"n={len(combined)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.axhline(1.0 / NUM_CARDS, color="gray", linewidth=0.5, linestyle="--")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _write_csv(out_path: Path, seed_indices, all_dists):
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "agent", "action_type", "value_type", "value", "count"])
        for sidx, d in zip(seed_indices, all_dists):
            for ai in (0, 1):
                for atype in ("msg", "pick"):
                    for vtype in ("view_pos", "view_color", "gt_color"):
                        arr = np.asarray(d[(ai, atype)][vtype], dtype=int)
                        h = np.bincount(arr, minlength=NUM_CARDS)
                        for k, c in enumerate(h):
                            w.writerow([sidx, ai, atype, vtype, k, int(c)])


def generate_action_distribution_artifacts(
    *,
    inner_env,
    stacked_params,
    policy,
    max_steps: int,
    output_dir: str | Path,
    seed_indices,
    num_episodes: int,
    feed_attn_dims=None,
    ja_card_masks=None,
    episode_rng_base: int = 200,
    greedy: bool = True,
    wb_run=None,
    wb_prefix: str = "XP",
):
    """Save per-seed/aggregate action-distribution artifacts and optionally log them."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dists = []
    for seed_idx in seed_indices:
        params = jax.tree.map(lambda x, _i=seed_idx: x[_i], stacked_params)
        rng_key = jax.random.PRNGKey(episode_rng_base + seed_idx)
        d = _collect_one_seed(
            rng_key, inner_env, params, policy, max_steps, num_episodes,
            feed_attn_dims, ja_card_masks, greedy=greedy,
        )
        all_dists.append(d)
        _plot(
            output_dir / f"action_dist_seed_{seed_idx}.png",
            d,
            title=f"seed {seed_idx} — {num_episodes} eps",
        )

    if len(all_dists) > 1:
        merged = _merge(*all_dists)
        _plot(
            output_dir / "action_dist_aggregate.png",
            merged,
            title=f"aggregate over {len(seed_indices)} seeds × {num_episodes} eps",
        )

    csv_path = output_dir / "action_dist_data.csv"
    _write_csv(csv_path, seed_indices, all_dists)

    if wb_run is not None:
        import wandb

        mode = "greedy" if greedy else "sampled"
        for seed_idx in seed_indices:
            png_path = output_dir / f"action_dist_seed_{seed_idx}.png"
            wb_run.log(
                {
                    f"{wb_prefix}/action_dist/{mode}/seed_{seed_idx}": wandb.Image(
                        str(png_path)
                    )
                },
                commit=False,
            )
        # Per-seed plots only — the aggregate plot was found redundant and is
        # no longer logged to wandb.
        for path in output_dir.glob("action_dist_*.png"):
            wandb.save(str(path), base_path=str(output_dir))
        wandb.save(str(csv_path), base_path=str(output_dir))

    return all_dists


def _entropy_peak(arr) -> tuple[float, float, int]:
    """Return (entropy, peak, argmax) for a list of int values in [0, NUM_CARDS)."""
    a = np.asarray(arr, dtype=int)
    if len(a) == 0:
        return float("nan"), float("nan"), -1
    h = np.bincount(a, minlength=NUM_CARDS).astype(float)
    p = h / h.sum()
    H = float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))
    return H, float(p.max()), int(np.argmax(h))


def _print_dist_stats(label: str, dists, sample_note: str = "") -> None:
    """Print peak/argmax/entropy summary per (agent, action_type, value_type).

    Max entropy for NUM_CARDS=5 is log(5) ≈ 1.61. Peak ≈ 0.20 + H ≈ 1.61 means
    uniform; peak → 1 + H → 0 means degenerate.
    """
    print(f"\n=== {label} ===")
    if sample_note:
        print(f"    {sample_note}")
    header = f"    {'who':12s}  {'view_pos':>20s}  {'view_color':>20s}  {'gt_color':>20s}"
    print(header)
    for ai in (0, 1):
        for atype in ("msg", "pick"):
            cells = []
            for vtype in ("view_pos", "view_color", "gt_color"):
                arr = np.asarray(dists[(ai, atype)][vtype], dtype=int)
                if len(arr) == 0:
                    cells.append("(n=0)")
                    continue
                H, peak, argmax_idx = _entropy_peak(arr)
                cells.append(f"peak={peak:.2f}@{argmax_idx} H={H:.2f}")
            who = f"agent_{ai} {atype}"
            print(f"    {who:12s}  {cells[0]:>20s}  {cells[1]:>20s}  {cells[2]:>20s}")


def _print_per_seed_equivariance_sp(label: str, all_dists, seed_indices) -> None:
    """Per-seed view_color H by role (SP).

    Under OP-on with an equivariant policy, all four H values should be near
    log(5) ≈ 1.61. Concentration (low H, peak high) flags role-specific
    non-equivariance.
    """
    import math as _math
    Hmax = _math.log(NUM_CARDS)
    print(f"\n--- {label}: per-seed view_color (Hmax=log({NUM_CARDS})={Hmax:.2f}) ---")
    print(f"  {'seed':>4}  {'msg a0':>16}  {'msg a1':>16}  {'pick a0':>16}  {'pick a1':>16}")
    for seed_idx, d in zip(seed_indices, all_dists):
        cells = []
        for atype, ai in [("msg", 0), ("msg", 1), ("pick", 0), ("pick", 1)]:
            H, peak, argmax_idx = _entropy_peak(d[(ai, atype)]["view_color"])
            if argmax_idx < 0:
                cells.append("(n=0)")
            else:
                cells.append(f"H={H:.2f} p={peak:.2f}@{argmax_idx}")
        print(f"  {seed_idx:>4}  {cells[0]:>16}  {cells[1]:>16}  {cells[2]:>16}  {cells[3]:>16}")


def _vocab_size(arr, threshold: float = 0.05) -> int:
    """Number of categories with marginal probability > threshold (vocabulary size)."""
    a = np.asarray(arr, dtype=int)
    if len(a) == 0:
        return 0
    h = np.bincount(a, minlength=NUM_CARDS).astype(float)
    p = h / h.sum()
    return int((p > threshold).sum())


def _effective_vocab(arr) -> float:
    """exp(H) — perplexity; equals true vocab size for uniform-on-subset distributions."""
    H, _, _ = _entropy_peak(arr)
    return float(np.exp(H)) if not np.isnan(H) else float("nan")


def _pearson(xs, ys) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _print_vocab_match_table(label: str, all_dists, index_labels, threshold: float = 0.05) -> None:
    """Per-seed/per-pair vocabulary size + match rate (= GT pick agreement = env score).

    `index_labels` are display strings (e.g. "0", "0v1") aligned with `all_dists`.
    Reports `vocab` (count of view_color bins above threshold) and `eff_vocab` (exp(H))
    for each agent's pick distribution, plus the per-row match rate. Closes with
    Pearson correlations between vocabulary measures and match.
    """
    print(f"\n--- {label}: vocab size (p>{threshold}) + match rate ---")
    header = (f"  {'idx':>6}  {'vocab a0':>8}  {'vocab a1':>8}  "
              f"{'eff a0':>7}  {'eff a1':>7}  "
              f"{'match':>8}  {'eps':>6}")
    print(header)
    rows = []
    for idx, d in zip(index_labels, all_dists):
        a0 = d[(0, "pick")]["view_color"]
        a1 = d[(1, "pick")]["view_color"]
        v0, v1 = _vocab_size(a0, threshold), _vocab_size(a1, threshold)
        e0, e1 = _effective_vocab(a0), _effective_vocab(a1)
        eps = d.get("__episode_count__", 0)
        match = (d.get("__match_count__", 0) / eps) if eps > 0 else float("nan")
        rows.append((str(idx), v0, v1, e0, e1, match, eps))
        print(f"  {str(idx):>6}  {v0:>8d}  {v1:>8d}  {e0:>7.2f}  {e1:>7.2f}  "
              f"{match:>8.3f}  {eps:>6d}")

    if len(rows) >= 2:
        v_sum = [r[1] + r[2] for r in rows]
        v_min = [min(r[1], r[2]) for r in rows]
        e_sum = [r[3] + r[4] for r in rows]
        e_min = [min(r[3], r[4]) for r in rows]
        m = [r[5] for r in rows]
        print(f"  Pearson r  match vs (vocab a0 + vocab a1)     = {_pearson(v_sum, m):+.3f}")
        print(f"  Pearson r  match vs min(vocab a0, vocab a1)  = {_pearson(v_min, m):+.3f}")
        print(f"  Pearson r  match vs (eff a0 + eff a1)         = {_pearson(e_sum, m):+.3f}")
        print(f"  Pearson r  match vs min(eff a0, eff a1)      = {_pearson(e_min, m):+.3f}")
        # Set-theoretic prediction: vocab_a0 + vocab_a1 > NUM_CARDS guarantees
        # |G_0 ∩ G_1| ≥ 1 (always have a coordinatable GT). This should correlate
        # with high match.
        guarantee = [int(r[1] + r[2] > NUM_CARDS) for r in rows]
        n_guaranteed = sum(guarantee)
        if 0 < n_guaranteed < len(rows):
            m_guar = [r[5] for r, g in zip(rows, guarantee) if g]
            m_unguar = [r[5] for r, g in zip(rows, guarantee) if not g]
            print(f"  vocab a0 + a1 > {NUM_CARDS} (intersection ≥ 1 guaranteed): "
                  f"{n_guaranteed}/{len(rows)} rows; "
                  f"mean match guaranteed = {np.mean(m_guar):.3f}; "
                  f"not = {np.mean(m_unguar):.3f}")
        elif n_guaranteed == len(rows):
            print(f"  vocab a0 + a1 > {NUM_CARDS} on all {len(rows)} rows; "
                  f"mean match = {np.mean(m):.3f}")
        else:
            print(f"  vocab a0 + a1 ≤ {NUM_CARDS} on all rows; intersection not guaranteed")


def _print_per_seed_equivariance_xp(label: str, pairs, all_dists, num_seeds) -> None:
    """Per-seed view_color H aggregated by role (XP).

    For each seed k, pool the view_color samples from every pair where k played
    agent_0 (resp. agent_1) and report H, peak, argmax. Same equivariance test:
    H ≈ log(5) ≈ 1.61 = equivariant in that role; low H = non-equivariant.
    """
    import math as _math
    Hmax = _math.log(NUM_CARDS)
    pooled: dict = {
        (k, atype, role): [] for k in range(num_seeds)
        for atype in ("msg", "pick") for role in (0, 1)
    }
    for (i, j), d in zip(pairs, all_dists):
        for atype in ("msg", "pick"):
            pooled[(i, atype, 0)].extend(d[(0, atype)]["view_color"])
            pooled[(j, atype, 1)].extend(d[(1, atype)]["view_color"])

    print(f"\n--- {label}: per-seed view_color pooled by role (Hmax=log({NUM_CARDS})={Hmax:.2f}) ---")
    print(f"  {'seed':>4}  {'msg as a0':>20}  {'msg as a1':>20}  {'pick as a0':>20}  {'pick as a1':>20}")
    for k in range(num_seeds):
        cells = []
        for atype, role in [("msg", 0), ("msg", 1), ("pick", 0), ("pick", 1)]:
            H, peak, argmax_idx = _entropy_peak(pooled[(k, atype, role)])
            n = len(pooled[(k, atype, role)])
            if argmax_idx < 0:
                cells.append("(n=0)")
            else:
                cells.append(f"H={H:.2f} p={peak:.2f}@{argmax_idx} n={n}")
        print(f"  {k:>4}  {cells[0]:>20}  {cells[1]:>20}  {cells[2]:>20}  {cells[3]:>20}")


def _build_env_and_policy(alg_config_template: dict):
    """Construct env and policy under the training OP regime."""
    alg_config = dict(alg_config_template)
    env_kwargs = dict(alg_config_template["ENV_KWARGS"])
    if alg_config.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    if alg_config["ENV_NAME"] == "card-game":
        env_kwargs["scramble_partner_msg"] = False
    alg_config["ENV_KWARGS"] = env_kwargs

    env = make_env(alg_config["ENV_NAME"], alg_config["ENV_KWARGS"])
    env = LogWrapper(env)
    inner_env = env._env
    max_steps = env_kwargs.get("max_steps", 8)

    obs_type = alg_config.get("OBS_TYPE",
                              env_kwargs.get("obs_type", "symbolic"))
    if obs_type in ("image", "fov"):
        init_fn = initialize_ja_image_agent
    else:
        init_fn = initialize_ja_agent
    policy, _ = init_fn(alg_config, env, jax.random.PRNGKey(0))

    feed_attn = alg_config.get("FEED_OTHER_ATTN", False)
    ja_card_attn = alg_config.get("JA_CARD_ATTN", False)
    ja_card_partner_feed = ja_card_attn and alg_config.get("JA_CARD_PARTNER_FEED", True)
    feed_attn_dims = None
    ja_card_masks = None
    if feed_attn or ja_card_partner_feed:
        from agents.ja_actor_critic import _compute_resnet_output_dims
        from agents.ja_utils import build_card_masks
        img_h = inner_env.grid_height * inner_env.tile_size
        img_w = inner_env.grid_width * inner_env.tile_size
        feat_h, feat_w = _compute_resnet_output_dims(
            img_h, img_w,
            stride=alg_config.get("CONV_STRIDE", 2),
            kernel_size=alg_config.get("CONV_KERNEL_SIZE", 3),
            padding=alg_config.get("CONV_PADDING", "SAME"),
            num_blocks=alg_config.get("CONV_NUM_BLOCKS", 4),
        )
        if feed_attn:
            feed_attn_dims = (img_h, img_w, feat_h, feat_w)
        if ja_card_partner_feed:
            ja_card_masks = build_card_masks(img_h, img_w, feat_h, feat_w)

    return inner_env, policy, max_steps, feed_attn_dims, ja_card_masks


def _run_mode(args, alg_config_template, final_params, num_seeds, seed_indices,
              greedy, xp_mode, output_dir, mode_label):
    """Build env+policy and run analysis under the training OP regime."""
    output_dir.mkdir(parents=True, exist_ok=True)
    inner_env, policy, max_steps, feed_attn_dims, ja_card_masks = _build_env_and_policy(
        alg_config_template,
    )
    eval_label = "XP" if xp_mode else "SP"
    print(f"\n[action_distributions] mode: {eval_label}  "
          f"(out: {output_dir.resolve()})")

    if xp_mode:
        if args.xp_pairs:
            pairs = []
            for tok in args.xp_pairs:
                i_str, j_str = tok.split(",")
                pairs.append((int(i_str), int(j_str)))
        else:
            pairs = [(i, j) for i in range(num_seeds) for j in range(i + 1, num_seeds)]
        print(f"  XP: {len(pairs)} pair(s); episodes/pair: {args.num_episodes}")
        all_dists = []
        for seed_i, seed_j in pairs:
            params_i = jax.tree.map(lambda x, _i=seed_i: x[_i], final_params)
            params_j = jax.tree.map(lambda x, _j=seed_j: x[_j], final_params)
            rng_key = jax.random.PRNGKey(
                args.episode_rng_base + seed_i * 1000 + seed_j * 100,
            )
            d = _collect_one_seed(
                rng_key, inner_env, params_i, policy, max_steps,
                args.num_episodes, feed_attn_dims, ja_card_masks,
                greedy=greedy, params_partner=params_j,
            )
            all_dists.append(d)
            _plot(
                output_dir / f"action_dist_xp_s{seed_i}_vs_s{seed_j}.png",
                d,
                title=f"XP s{seed_i} (agent 0) vs s{seed_j} (agent 1) — "
                      f"{args.num_episodes} eps, op={op_label}",
            )
        merged = _merge(*all_dists) if len(all_dists) > 1 else all_dists[0]
        if len(all_dists) > 1:
            _plot(
                output_dir / "action_dist_xp_aggregate.png",
                merged,
                title=f"XP aggregate over {len(pairs)} pair(s) × {args.num_episodes} eps "
                      f"— op={op_label}",
            )
        csv_path = output_dir / "action_dist_xp_data.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "agent", "action_type", "value_type", "value", "count"])
            for (i, j), d in zip(pairs, all_dists):
                tag = f"{i}v{j}"
                for ai in (0, 1):
                    for atype in ("msg", "pick"):
                        for vtype in ("view_pos", "view_color", "gt_color"):
                            arr = np.asarray(d[(ai, atype)][vtype], dtype=int)
                            h = np.bincount(arr, minlength=NUM_CARDS)
                            for k, c in enumerate(h):
                                w.writerow([tag, ai, atype, vtype, k, int(c)])
        sample_note = (f"aggregated over {len(pairs)} pairs × "
                       f"{args.num_episodes} eps; mode={'greedy' if greedy else 'sampled'}")
        stats_label = f"{mode_label or eval_label + '+OP_' + op_label}"
        _print_dist_stats(stats_label, merged, sample_note)
        # Per-seed view_color equivariance check, aggregated by role across partners
        _print_per_seed_equivariance_xp(stats_label, pairs, all_dists, num_seeds)
        # Vocabulary size vs match-rate correlation per pair
        pair_labels = [f"{i}v{j}" for (i, j) in pairs]
        _print_vocab_match_table(stats_label, all_dists, pair_labels)
        return

    all_dists = generate_action_distribution_artifacts(
        inner_env=inner_env,
        stacked_params=final_params,
        policy=policy,
        max_steps=max_steps,
        output_dir=output_dir,
        seed_indices=seed_indices,
        num_episodes=args.num_episodes,
        feed_attn_dims=feed_attn_dims,
        ja_card_masks=ja_card_masks,
        episode_rng_base=args.episode_rng_base,
        greedy=greedy,
    )
    merged = _merge(*all_dists) if len(all_dists) > 1 else all_dists[0]
    sample_note = (f"aggregated over {len(seed_indices)} seeds × "
                   f"{args.num_episodes} eps; mode={'greedy' if greedy else 'sampled'}")
    stats_label = f"{mode_label or eval_label + '+OP_' + op_label}"
    _print_dist_stats(stats_label, merged, sample_note)
    # Per-seed view_color equivariance check
    _print_per_seed_equivariance_sp(stats_label, all_dists, seed_indices)
    # Vocabulary size vs match-rate correlation per seed
    _print_vocab_match_table(stats_label, all_dists, [str(s) for s in seed_indices])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved_train_run directory")
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--use-best", action="store_true",
                        help="Use best_params instead of final_params when available")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--output-dir", default="plots/card_game")
    parser.add_argument("--episode-rng-base", type=int, default=200)
    parser.add_argument("--sampled", action="store_true",
                        help="Sample actions instead of using greedy argmax")
    parser.add_argument("--xp-mode", action="store_true",
                        help="Cross-play: collect distributions over (i,j) seed pairs "
                             "(agent 0 = seed_i, agent 1 = seed_j) instead of self-play. "
                             "Default: upper-triangle pairs (i<j); override with --xp-pairs.")
    parser.add_argument("--xp-pairs", nargs="+", default=None,
                        help='Explicit XP pairs as "i,j" tokens, e.g. `--xp-pairs 0,1 0,5`. '
                             "Implies --xp-mode.")
    parser.add_argument("--all-modes", action="store_true",
                        help="Run both SP and XP modes in one go. Outputs go under "
                             "{output-dir}/{mode}/. Overrides --xp-mode for this invocation. "
                             "Implies --all-seeds for SP.")
    args = parser.parse_args()
    if args.xp_pairs:
        args.xp_mode = True

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    cfg_dir = run_dir
    config_path = None
    for _ in range(4):
        cand = cfg_dir / ".hydra" / "config.yaml"
        if cand.exists():
            config_path = cand
            break
        cfg_dir = cfg_dir.parent
    if config_path is None:
        raise FileNotFoundError(f"No .hydra/config.yaml found near {run_dir}")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config_template = cfg["algorithm"]

    run_data = load_train_run(str(ckpt_path))
    params_key = "best_params" if args.use_best else "final_params"
    if params_key not in run_data:
        raise KeyError(f"{params_key!r} not found in checkpoint; keys: {list(run_data.keys())}")
    final_params = run_data[params_key]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    if args.all_modes:
        args.all_seeds = True
    seed_indices = list(range(num_seeds)) if args.all_seeds else [args.seed_idx]
    greedy = not args.sampled

    base_output_dir = Path(args.output_dir)
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  params_key: {params_key}; mode: {'greedy' if greedy else 'sampled'}; "
          f"episodes: {args.num_episodes}")
    print(f"  base output: {base_output_dir.resolve()}")

    if args.all_modes:
        modes = [
            ("sp", False),
            ("xp", True),
        ]
    else:
        modes = [(None, args.xp_mode)]

    for mode_label, xp_mode in modes:
        sub_dir = base_output_dir / mode_label if mode_label else base_output_dir
        _run_mode(
            args, alg_config_template, final_params, num_seeds, seed_indices,
            greedy, xp_mode, sub_dir, mode_label,
        )

    print(f"\nDone. Outputs under {base_output_dir.resolve()}/")


if __name__ == "__main__":
    main()
