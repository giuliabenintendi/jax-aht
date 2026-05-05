"""Per-step message-pair matrices averaged across seeds (card game).

Loads a saved training checkpoint, runs N greedy eval episodes per seed on the
communication card game, and produces one 5x5 row-normalized heatmap per step:
  - Deliberation steps t=0..max_steps-2: P(agent_1 msg | agent_0 msg).
  - Decision step t=max_steps-1: P(agent_1 pick | agent_0 pick).
Each seed's per-step 5x5 count matrix is row-normalized, then averaged
element-wise across seeds. Labels are in ground-truth card space via
`_invert_actions` on the wrapper chain.

Outputs inside --output-dir:
  step_{t}.png   — 5x5 heatmap for step t
  all_steps.png  — all steps side-by-side, shared colorbar

Usage:
    ./run_gpu.sh <gpu> evaluation.msg_pair_over_steps \\
        --checkpoint /path/to/saved/train_run \\
        --num-episodes 128 \\
        --output-dir plots/card_game/<run_name>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import NUM_CARDS
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import run_episode_with_states


def _get_obs_type(alg_config: dict) -> str:
    return alg_config.get(
        "OBS_TYPE",
        alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"),
    )


def _row_normalize(counts: np.ndarray) -> np.ndarray:
    """Row-normalize a (num_cards, num_cards) count matrix; empty rows stay 0.
    Matches eval_logging.py's per-seed normalization convention.
    """
    out = counts.astype(np.float32)
    row_sums = out.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    if np.any(nonzero):
        out[nonzero] = out[nonzero] / row_sums[nonzero]
    return out


def _plot_heatmap(ax, mat: np.ndarray, title: str, vmin: float, vmax: float,
                  show_ylabel: bool, show_xlabel: bool, annotate: bool):
    num_cards = mat.shape[0]
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(num_cards))
    ax.set_yticks(range(num_cards))
    ax.set_xticklabels([str(c) for c in range(num_cards)])
    ax.set_yticklabels([str(c) for c in range(num_cards)])
    if show_xlabel:
        ax.set_xlabel("agent_1 card", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("agent_0 card", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
    if annotate:
        thresh = vmax * 0.6
        for r in range(num_cards):
            for c in range(num_cards):
                val = float(mat[r, c])
                color = "white" if val > thresh else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved train run checkpoint directory")
    parser.add_argument("--num-episodes", type=int, default=64,
                        help="Greedy eval episodes per seed")
    parser.add_argument("--output-dir", default="plots/card_game")
    parser.add_argument("--episode-rng-base", type=int, default=2000,
                        help="Base seed; per-episode key = base + seed*10000 + ep")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Agreement rate at which to mark 'confidence reached' on agreement_line.png")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    config_path = None
    cur = run_dir
    for _ in range(4):
        candidate = cur / ".hydra" / "config.yaml"
        if candidate.exists():
            config_path = candidate
            break
        cur = cur.parent
    if config_path is None:
        raise FileNotFoundError(
            f"Could not locate .hydra/config.yaml near {run_dir}"
        )
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]

    if not alg_config.get("COMMUNICATION", False):
        raise ValueError(
            "msg_pair_over_steps requires a communication-enabled checkpoint "
            "(algorithm.COMMUNICATION=True)."
        )
    env_kwargs = dict(alg_config["ENV_KWARGS"])
    env_kwargs["communication"] = True
    alg_config["ENV_KWARGS"] = env_kwargs

    env = make_env(alg_config["ENV_NAME"], alg_config["ENV_KWARGS"])
    env = LogWrapper(env)
    inner_env = env._env
    max_steps = int(alg_config["ENV_KWARGS"].get("max_steps", 8))
    num_cards = getattr(inner_env, "num_cards", NUM_CARDS)

    obs_type = _get_obs_type(alg_config)
    init_fn = (
        initialize_ja_image_agent if obs_type in ("image", "fov")
        else initialize_ja_agent
    )
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env, rng)

    run_data = load_train_run(str(ckpt_path))
    final_params = run_data["final_params"]
    num_seeds = int(jax.tree.leaves(final_params)[0].shape[0])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  seeds: {num_seeds}, episodes per seed: {args.num_episodes}")
    print(f"  max_steps: {max_steps}, num_cards: {num_cards}")
    print(f"  saving to: {output_dir.resolve()}")

    # counts[step, seed] -> (num_cards, num_cards) in GT space.
    # Steps 0..max_steps-2 = message pair; step max_steps-1 = pick pair.
    counts = np.zeros((max_steps, num_seeds, num_cards, num_cards), dtype=np.int64)

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)
        for ep in range(args.num_episodes):
            ep_rng = jax.random.PRNGKey(
                args.episode_rng_base + seed_idx * 10000 + ep
            )
            _, ep_actions, ep_messages = run_episode_with_states(
                ep_rng, inner_env, params, policy, params, policy, max_steps,
                collect_attention=False, greedy=True,
            )
            for t in range(max_steps - 1):
                if t >= len(ep_messages):
                    break
                m0, m1 = ep_messages[t]
                if 0 <= m0 < num_cards and 0 <= m1 < num_cards:
                    counts[t, seed_idx, int(m0), int(m1)] += 1
            decision_t = max_steps - 1
            if decision_t < len(ep_actions):
                p0, p1 = ep_actions[decision_t]
                if 0 <= p0 < num_cards and 0 <= p1 < num_cards:
                    counts[decision_t, seed_idx, int(p0), int(p1)] += 1
        print(f"  [msg_pair] seed {seed_idx}/{num_seeds - 1} done")

    # Per-seed row-normalized, then mean across seeds.
    avg_row = np.zeros((max_steps, num_cards, num_cards), dtype=np.float32)
    for t in range(max_steps):
        row_stack = np.stack([_row_normalize(counts[t, s]) for s in range(num_seeds)])
        avg_row[t] = row_stack.mean(axis=0)

    # Per-step agreement rates for the console log, straight from counts.
    print("\nPer-step raw agreement (trace/total, mean across seeds):")
    for t in range(max_steps):
        per_seed_agree = []
        for s in range(num_seeds):
            tot = counts[t, s].sum()
            if tot > 0:
                per_seed_agree.append(np.trace(counts[t, s]) / tot)
        kind = "pick" if t == max_steps - 1 else "msg"
        if per_seed_agree:
            arr = np.asarray(per_seed_agree)
            print(f"  step {t} ({kind}): {arr.mean():.3f} ± {arr.std(ddof=1) / np.sqrt(len(arr)):.3f}  "
                  f"(seeds used: {len(per_seed_agree)}/{num_seeds})")
        else:
            print(f"  step {t} ({kind}): no data")

    def _save_all_steps(avg: np.ndarray, filename: str, vmax: float):
        fig, axes = plt.subplots(
            1, max_steps, figsize=(max_steps * 1.9, 2.6), sharey=True,
        )
        if max_steps == 1:
            axes = [axes]
        im = None
        for t, ax in enumerate(axes):
            is_decision = t == max_steps - 1
            kind = "pick" if is_decision else "msg"
            im = _plot_heatmap(
                ax, avg[t], f"step {t} ({kind})",
                0.0, vmax,
                show_ylabel=(t == 0), show_xlabel=True, annotate=False,
            )
        fig.subplots_adjust(right=0.92, wspace=0.25)
        cbar_ax = fig.add_axes([0.935, 0.15, 0.010, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)

    # Row-normalized outputs (default for per-step pngs, matches existing convention).
    for t in range(max_steps):
        is_decision = t == max_steps - 1
        kind = "pick" if is_decision else "msg"
        fig, ax = plt.subplots(figsize=(3.2, 3.0))
        im = _plot_heatmap(
            ax, avg_row[t], f"step {t} ({kind})",
            0.0, 1.0, show_ylabel=True, show_xlabel=True, annotate=True,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"step_{t}.png", bbox_inches="tight")
        plt.close(fig)

    _save_all_steps(avg_row, "all_steps.png", vmax=1.0)

    # Confidence-over-time line: per-step agreement rate (trace / total) computed
    # straight from the count matrix. Faded line per seed plus across-seed mean.
    per_seed_agreement = np.full((num_seeds, max_steps), np.nan, dtype=np.float32)
    for s in range(num_seeds):
        for t in range(max_steps):
            tot = counts[t, s].sum()
            if tot > 0:
                per_seed_agreement[s, t] = float(np.trace(counts[t, s])) / float(tot)
    mean_agreement = np.nanmean(per_seed_agreement, axis=0)

    fig, ax = plt.subplots(figsize=(max_steps * 0.7 + 1.0, 2.8))
    xs = np.arange(max_steps)
    for s in range(num_seeds):
        label = "per seed" if s == 0 else None
        ax.plot(xs, per_seed_agreement[s], color="gray", alpha=0.35,
                linewidth=0.8, marker="o", markersize=3, label=label)
    ax.plot(xs, mean_agreement, color="C0", linewidth=2.0,
            marker="o", markersize=5, label="mean across seeds")

    # "Confidence reached" indicator: dashed horizontal at threshold + dotted
    # vertical at the first step where the across-seed mean crosses it.
    thr = float(args.confidence_threshold)
    ax.axhline(thr, color="crimson", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"threshold = {thr:.2f}")
    crossing = next((int(t) for t, v in enumerate(mean_agreement) if not np.isnan(v) and v >= thr), None)
    if crossing is not None:
        ax.axvline(crossing, color="crimson", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.annotate(
            f"first crossing\nstep {crossing}",
            xy=(crossing, thr), xytext=(crossing + 0.15, thr - 0.18),
            fontsize=7, color="crimson",
            arrowprops=dict(arrowstyle="-", color="crimson", lw=0.7, alpha=0.7),
        )
        print(f"[confidence] mean agreement first reaches {thr:.2f} at step {crossing}")
    else:
        print(f"[confidence] mean agreement never reaches {thr:.2f} within {max_steps} steps")

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"step {t}\n({'pick' if t == max_steps - 1 else 'msg'})" for t in xs],
        fontsize=7,
    )
    ax.set_ylabel("agreement rate (diagonal sum)", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "agreement_line.png", bbox_inches="tight")
    plt.close(fig)

    # Per-seed combined figures.
    per_seed_dir = output_dir / "per_seed"
    per_seed_dir.mkdir(exist_ok=True)
    print("\nPer-seed raw agreement (trace/total) per step:")
    header = "seed " + " ".join(
        f"{'pick' if t == max_steps - 1 else 'msg':>6s}{t}" for t in range(max_steps)
    )
    print("  " + header)
    for s in range(num_seeds):
        row_norm_per_step = np.stack([_row_normalize(counts[t, s]) for t in range(max_steps)])

        fig, axes = plt.subplots(
            1, max_steps, figsize=(max_steps * 1.9, 2.8), sharey=True,
        )
        if max_steps == 1:
            axes = [axes]
        im = None
        for t, ax in enumerate(axes):
            kind = "pick" if t == max_steps - 1 else "msg"
            im = _plot_heatmap(
                ax, row_norm_per_step[t], f"step {t} ({kind})",
                0.0, 1.0,
                show_ylabel=(t == 0), show_xlabel=True, annotate=False,
            )
        fig.suptitle(f"seed {s}", fontsize=10)
        fig.subplots_adjust(right=0.92, wspace=0.25, top=0.85)
        cbar_ax = fig.add_axes([0.935, 0.15, 0.010, 0.65])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(per_seed_dir / f"seed_{s}_all_steps.png", bbox_inches="tight")
        plt.close(fig)

        agreements = []
        for t in range(max_steps):
            tot = counts[t, s].sum()
            agreements.append(
                f"{np.trace(counts[t, s]) / tot:6.3f}" if tot > 0 else "   nan"
            )
        print(f"  {s:>4d} " + " ".join(f"{a:>7s}" for a in agreements))

    print(f"Done. Figures in {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
