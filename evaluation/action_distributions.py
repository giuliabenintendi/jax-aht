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
    ./run_gpu.sh <gpu> evaluation.action_distributions \\
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
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_agent,
    initialize_ja_dual_image_agent,
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


def _empty_dist():
    return {(a, t): {"view_pos": [], "view_color": [], "gt_color": []}
            for a in (0, 1) for t in ("msg", "pick")}


def _collect_one_seed(rng_key, inner_env, params, policy, max_steps,
                      num_episodes, feed_attn_dims, ja_card_masks,
                      greedy=True):
    out = _empty_dist()
    rngs = jax.random.split(rng_key, num_episodes)
    for ep in range(num_episodes):
        ep_states, _, ep_actions, ep_messages = run_episode_with_states(
            rngs[ep], inner_env, params, policy, params, policy, max_steps,
            collect_attention=True,  # required by run_episode_with_states API
            greedy=greedy,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
        )
        n_steps = min(len(ep_messages), len(ep_actions))
        for t in range(n_steps):
            is_decision = (t == n_steps - 1)
            for ai in (0, 1):
                gt = (int(ep_actions[t][ai]) if is_decision
                      else int(ep_messages[t][ai]))
                if gt < 0:
                    continue
                pos_perm = _per_agent_perm(ep_states[t], ai)
                vc_pos = _gt_to_view_col(ep_states[t], pos_perm, gt)
                if vc_pos is None:
                    continue
                recol = _per_agent_recolouring(ep_states[t], ai)
                vc_color = int(recol[gt])
                key = (ai, "pick" if is_decision else "msg")
                out[key]["view_pos"].append(vc_pos)
                out[key]["view_color"].append(vc_color)
                out[key]["gt_color"].append(int(gt))
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
            out[k]["view_pos"].extend(v["view_pos"])
            out[k]["view_color"].extend(v["view_color"])
            out[k]["gt_color"].extend(v["gt_color"])
    return out


def _plot(out_path: Path, dists, title: str):
    fig, axes = plt.subplots(2, 6, figsize=(19, 5.5), sharey=True)
    palettes = ["Oranges", "RdPu"]
    rgb_card_colors = np.asarray(CARD_COLORS) / 255.0
    col_titles = [
        "msg by view-position", "msg by view-color", "msg by gt-color",
        "pick by view-position", "pick by view-color", "pick by gt-color",
    ]
    var_for_col = [
        ("msg", "view_pos"), ("msg", "view_color"), ("msg", "gt_color"),
        ("pick", "view_pos"), ("pick", "view_color"), ("pick", "gt_color"),
    ]
    for ai in (0, 1):
        for ci, (atype, vtype) in enumerate(var_for_col):
            ax = axes[ai, ci]
            arr = dists[(ai, atype)][vtype]
            h = _hist_norm(arr)
            xs = np.arange(NUM_CARDS)
            if vtype in ("view_color", "gt_color"):
                ax.bar(xs, h, color=rgb_card_colors, edgecolor="black", linewidth=0.4)
            else:
                cmap = plt.colormaps[palettes[ai]]
                ax.bar(xs, h, color=cmap(0.7), edgecolor="black", linewidth=0.4)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(xs)
            if ai == 0:
                ax.set_title(col_titles[ci], fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"agent_{ai}\nfraction", fontsize=10)
            ax.text(0.98, 0.97, f"n={len(arr)}",
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
        agg_path = output_dir / "action_dist_aggregate.png"
        if agg_path.exists():
            wb_run.log(
                {
                    f"{wb_prefix}/action_dist/{mode}/aggregate": wandb.Image(
                        str(agg_path)
                    )
                },
                commit=False,
            )
        for path in output_dir.glob("action_dist_*.png"):
            wandb.save(str(path), base_path=str(output_dir))
        wandb.save(str(csv_path), base_path=str(output_dir))

    return all_dists


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
    args = parser.parse_args()

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
    alg_config = cfg["algorithm"]

    env_kwargs = dict(alg_config["ENV_KWARGS"])
    if alg_config.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    if alg_config["ENV_NAME"] == "card-game":
        env_kwargs["scramble_partner_msg"] = False
    alg_config["ENV_KWARGS"] = env_kwargs

    env = make_env(alg_config["ENV_NAME"], alg_config["ENV_KWARGS"])
    env = LogWrapper(env)
    inner_env = env._env
    max_steps = alg_config["ENV_KWARGS"].get("max_steps", 8)

    obs_type = alg_config.get("OBS_TYPE",
                              alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))
    if obs_type in ("image", "fov"):
        init_fn = (
            initialize_ja_dual_image_agent
            if alg_config.get("USE_DUAL_CRITIC", False)
            else initialize_ja_image_agent
        )
    else:
        init_fn = initialize_ja_agent
    policy, _ = init_fn(alg_config, env, jax.random.PRNGKey(0))

    feed_attn = alg_config.get("FEED_OTHER_ATTN", False)
    ja_card_attn = alg_config.get("JA_CARD_ATTN", False)
    ja_card_partner_feed = ja_card_attn and alg_config.get("JA_CARD_PARTNER_FEED", True)
    feed_attn_dims = None
    ja_card_masks = None
    if feed_attn or ja_card_partner_feed:
        from agents.ja_image_actor_critic import _compute_resnet_output_dims
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

    run_data = load_train_run(str(ckpt_path))
    params_key = "best_params" if args.use_best else "final_params"
    if params_key not in run_data:
        raise KeyError(f"{params_key!r} not found in checkpoint; keys: {list(run_data.keys())}")
    final_params = run_data[params_key]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    seed_indices = list(range(num_seeds)) if args.all_seeds else [args.seed_idx]
    greedy = not args.sampled

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  seeds analyzing: {seed_indices}; episodes/seed: {args.num_episodes}; "
          f"max_steps: {max_steps}")
    print(f"  params_key: {params_key}; mode: {'greedy' if greedy else 'sampled'}")
    print(f"  saving to: {output_dir.resolve()}")

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
    for seed_idx, d in zip(seed_indices, all_dists):
        n_msg = len(d[(0, "msg")]["view_pos"]) + len(d[(1, "msg")]["view_pos"])
        n_pick = len(d[(0, "pick")]["view_pos"]) + len(d[(1, "pick")]["view_pos"])
        print(f"  seed {seed_idx}: msg actions={n_msg}, pick actions={n_pick}")
    print(f"Done. Figures + CSV in {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
