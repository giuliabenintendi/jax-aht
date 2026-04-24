"""Per-step message-pair agreement heatmaps across all seeds (card game).

Loads a saved training checkpoint, runs N greedy eval episodes per seed on the
communication card game, and produces aligned (num_seeds x num_cards) heatmaps
for every deliberation step (message pair at step t) and the final decision
step (pick pair). Columns are shifted so column 0 = "agent_1 matched agent_0";
the remaining columns are cyclic offsets. Action/message labels are mapped back
to ground-truth colour identity via `_invert_actions` on the wrapper chain.

Outputs inside --output-dir:
  step_{t}.png   — (num_seeds x num_cards) heatmap for step t
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


def _align_and_collapse(counts: np.ndarray) -> np.ndarray:
    """Row-normalize a (num_cards, num_cards) count matrix, cyclically shift
    each row r by -r so column 0 = diagonal, then mean across non-empty rows.
    Returns a (num_cards,) vector where index 0 is the agreement rate.
    """
    num_cards = counts.shape[0]
    row_sums = counts.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    if not np.any(nonzero):
        return np.zeros(num_cards, dtype=np.float32)
    norm = counts.astype(np.float32)
    norm[nonzero] = norm[nonzero] / row_sums[nonzero].astype(np.float32)
    aligned = np.zeros_like(norm)
    for r in range(num_cards):
        aligned[r] = np.roll(norm[r], -r)
    return aligned[nonzero].mean(axis=0)


def _plot_heatmap(ax, mat: np.ndarray, title: str, vmin: float, vmax: float,
                  show_ylabel: bool):
    num_seeds, num_cards = mat.shape
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(num_cards))
    ax.set_xticklabels([str(s) for s in range(num_cards)])
    ax.set_xlabel("offset (0 = match)", fontsize=8)
    if show_ylabel:
        ax.set_yticks(range(num_seeds))
        ax.set_yticklabels([str(s) for s in range(num_seeds)])
        ax.set_ylabel("seed", fontsize=8)
    else:
        ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis="both", labelsize=7)
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

    aligned = np.zeros((max_steps, num_seeds, num_cards), dtype=np.float32)
    for t in range(max_steps):
        for s in range(num_seeds):
            aligned[t, s] = _align_and_collapse(counts[t, s])

    vmax = float(aligned.max()) if aligned.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    vmin = 0.0

    for t in range(max_steps):
        is_decision = t == max_steps - 1
        kind = "pick" if is_decision else "msg"
        fig, ax = plt.subplots(figsize=(3.0, max(2.0, num_seeds * 0.15) + 0.3))
        im = _plot_heatmap(
            ax, aligned[t], f"step {t} ({kind})", vmin, vmax, show_ylabel=True,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / f"step_{t}.png", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(
        1, max_steps,
        figsize=(max_steps * 1.8, max(2.0, num_seeds * 0.15) + 0.8),
        sharey=True,
    )
    if max_steps == 1:
        axes = [axes]
    im = None
    for t, ax in enumerate(axes):
        is_decision = t == max_steps - 1
        kind = "pick" if is_decision else "msg"
        im = _plot_heatmap(
            ax, aligned[t], f"step {t} ({kind})",
            vmin, vmax, show_ylabel=(t == 0),
        )
    fig.subplots_adjust(right=0.92, wspace=0.15)
    cbar_ax = fig.add_axes([0.935, 0.15, 0.010, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.savefig(output_dir / "all_steps.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Done. Figures in {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
