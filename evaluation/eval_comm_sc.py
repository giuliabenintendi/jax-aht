"""Compute Speaker Consistency (SC) for a trained card-game checkpoint.

Per-seed and aggregate SC_i^(k) for each agent `i` and each deliberation
slot `k`, using paired `(M_{i,k}, A_i)` samples from N rollouts. A_i is the
decision-step pick; messages are decoded from the Discrete(10) comm action
space. See `evaluation.comm_metrics` for the underlying MI formula.

Usage:
    ./run_gpu.sh 0 evaluation.eval_comm_sc \
        --checkpoint <path_to_saved_train_run> \
        --num-episodes 512
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from envs.card_game.rendering import NUM_CARDS
from evaluation._card_game_utils import load_card_game_eval
from evaluation.comm_metrics import compute_sc_slotwise
from evaluation.vis_episodes import run_episode_with_states


def _collect_rollouts(
    inner_env, policy, params, num_episodes: int, max_steps: int,
    seed_offset: int, greedy: bool,
) -> tuple[list, list]:
    """Run `num_episodes` SP episodes and return `(ep_messages, ep_actions)`."""
    all_msgs: list[list[tuple[int, int]]] = []
    all_acts: list[list[tuple[int, int]]] = []
    for ep in range(num_episodes):
        rng = jax.random.PRNGKey(7_000 + seed_offset * 10_000 + ep)
        ep_states, ep_actions, ep_messages = run_episode_with_states(
            rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=False, greedy=greedy,
        )
        all_msgs.append(ep_messages)
        all_acts.append(ep_actions)
    return all_msgs, all_acts


def _plot_sc(all_sc: np.ndarray, output_path: Path, label: str) -> None:
    """Plot per-agent SC across slots, with per-seed lines and mean ± SEM band.

    Reading the figure: a curve that's flat-and-high from k=0 onward marks a
    "proposer" (committed message from the start). A curve that climbs from
    near 0 to high values marks a "follower" (commits late, after seeing the
    proposer). A bimodal pattern across seeds is a population mix.

    Args:
        all_sc: shape (num_seeds, 2, K) — per-seed SC grid in nats.
        output_path: PNG output path.
        label: run label for the title.
    """
    n_seeds, n_agents, K = all_sc.shape
    ks = np.arange(K)
    colors = ["#fb8500", "#9d4edd"]  # orange (agent 0), magenta-ish (agent 1)
    names = ["agent 0 (orange triangle)", "agent 1 (magenta triangle)"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for agent_idx in range(n_agents):
        per_seed = all_sc[:, agent_idx, :]
        mean = per_seed.mean(axis=0)
        sem = (per_seed.std(axis=0, ddof=1) / math.sqrt(n_seeds)
               if n_seeds > 1 else np.zeros_like(mean))

        # per-seed thin lines for population spread
        for s in range(n_seeds):
            ax.plot(ks, per_seed[s], color=colors[agent_idx], alpha=0.18,
                    linewidth=0.8)
        # mean and SEM band
        ax.plot(ks, mean, color=colors[agent_idx], linewidth=2.4,
                marker="o", markersize=5, label=f"{names[agent_idx]} mean")
        ax.fill_between(ks, mean - sem, mean + sem,
                        color=colors[agent_idx], alpha=0.22,
                        label=f"{names[agent_idx]} ±SEM")

    ceiling = math.log(NUM_CARDS)
    ax.axhline(ceiling, color="gray", linestyle="--",
               label=f"ceiling = log({NUM_CARDS}) = {ceiling:.3f}")

    ax.set_xticks(ks)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_xlabel("deliberation slot k")
    ax.set_ylabel("Speaker Consistency (nats)")
    ax.set_ylim(0, ceiling * 1.05)
    ax.set_title(
        f"Speaker Consistency — {label}  ({n_seeds} seeds, mean ± SEM)\n"
        f"flat-high curve = proposer; rising curve = follower",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _format_sc_grid(sc: np.ndarray) -> str:
    """Pretty-print a (2, K) SC grid in nats with 3 decimals."""
    K = sc.shape[1]
    header = "slot:  " + "  ".join(f"  k={k}" for k in range(K))
    a0 = "agent0 " + "  ".join(f"{v:6.3f}" for v in sc[0])
    a1 = "agent1 " + "  ".join(f"{v:6.3f}" for v in sc[1])
    return "\n".join((header, a0, a1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=512)
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use argmax actions. Default: sample from policy (matches Lowe et al.).",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    if not ev.env_kwargs.get("communication", False):
        raise SystemExit("Checkpoint was trained without communication; SC is undefined.")

    print(
        f"\nCheckpoint: {args.checkpoint}"
        f"\n  label={ev.label}  comm=on  best_per_seed=YES  best_idx={ev.best_idx.tolist()}"
        f"\n  seeds={ev.num_seeds}  episodes/seed={args.num_episodes}"
        f"  max_steps={ev.max_steps}  greedy={args.greedy}"
    )

    all_sc = []
    for seed_idx in range(ev.num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], ev.params)
        ep_messages, ep_actions = _collect_rollouts(
            ev.env, ev.policy, params, args.num_episodes, ev.max_steps,
            seed_offset=seed_idx, greedy=args.greedy,
        )
        sc = compute_sc_slotwise(ep_messages, ep_actions, NUM_CARDS, NUM_CARDS)
        all_sc.append(sc)
        print(f"\nSeed {seed_idx}  SC (nats, max={math.log(NUM_CARDS):.3f}):")
        print(_format_sc_grid(sc))

    all_sc_arr = np.stack(all_sc, axis=0)
    mean_sc = all_sc_arr.mean(axis=0)
    sem_sc = (all_sc_arr.std(axis=0, ddof=1) / math.sqrt(ev.num_seeds)
              if ev.num_seeds > 1 else np.zeros_like(mean_sc))

    print(f"\nAggregate across {ev.num_seeds} seeds (mean ± SEM):")
    K = mean_sc.shape[1]
    for agent_idx in range(2):
        parts = [
            f"{mean_sc[agent_idx, k]:6.3f}±{sem_sc[agent_idx, k]:.3f}"
            for k in range(K)
        ]
        print(f"  agent{agent_idx}: " + "  ".join(parts))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = ev.label.replace("/", "_").replace(" ", "_")
        np.save(out / f"sc_{slug}.npy", all_sc_arr)
        print(f"\nSaved {out / f'sc_{slug}.npy'}  shape={all_sc_arr.shape}")
        _plot_sc(all_sc_arr, out / f"sc_{slug}.png", ev.label)
        print(f"Saved {out / f'sc_{slug}.png'}")


if __name__ == "__main__":
    main()
