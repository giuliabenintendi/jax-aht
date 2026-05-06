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
    inner_env, policy, params_a, params_b, num_episodes: int, max_steps: int,
    seed_offset: int, greedy: bool,
) -> tuple[list, list]:
    """Run `num_episodes` episodes pairing agent 0 with `params_a` and agent
    1 with `params_b` (self-play when params_a is params_b).
    """
    all_msgs: list[list[tuple[int, int]]] = []
    all_acts: list[list[tuple[int, int]]] = []
    for ep in range(num_episodes):
        rng = jax.random.PRNGKey(7_000 + seed_offset * 10_000 + ep)
        ep_states, ep_actions, ep_messages = run_episode_with_states(
            rng, inner_env, params_a, policy, params_b, policy, max_steps,
            collect_attention=False, greedy=greedy,
        )
        all_msgs.append(ep_messages)
        all_acts.append(ep_actions)
    return all_msgs, all_acts


def _plot_sc_pairs_grid(
    pair_sc: np.ndarray,
    pairs: list[tuple[int, int]],
    output_path: Path,
    label: str,
) -> None:
    """Small-multiples: one panel per (agent_0_seed, agent_1_seed) pair.

    For self-play pairs (i == j), the panel matches `sc_per_seed_*` content;
    for cross-play pairs, the curves show how each agent's own message-pick
    consistency holds up when paired with a partner from a different seed.
    """
    n_pairs, _, K = pair_sc.shape
    n_cols = min(4, n_pairs)
    n_rows = math.ceil(n_pairs / n_cols)
    ks = np.arange(K)
    ceiling = math.log(NUM_CARDS)
    colors = ["#fb8500", "#9d4edd"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 2.6),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    for p, (seed_a, seed_b) in enumerate(pairs):
        r, c = divmod(p, n_cols)
        ax = axes[r, c]
        labels_p = [f"agent 0 (seed {seed_a})", f"agent 1 (seed {seed_b})"]
        for agent_idx in range(2):
            ax.plot(
                ks, pair_sc[p, agent_idx], color=colors[agent_idx],
                marker="o", markersize=3, linewidth=1.7,
                label=labels_p[agent_idx] if p == 0 else None,
            )
        ax.axhline(ceiling, color="gray", linestyle="--", linewidth=0.8)
        title = (f"SP: seed {seed_a}" if seed_a == seed_b
                 else f"XP: a0=seed{seed_a}, a1=seed{seed_b}")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.05, ceiling * 1.05)
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)

    for p in range(n_pairs, n_rows * n_cols):
        r, c = divmod(p, n_cols)
        axes[r, c].axis("off")

    handles, labels_h = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_h, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.supxlabel("deliberation slot k", fontsize=11)
    fig.supylabel("Speaker Consistency (nats)", fontsize=11)
    fig.suptitle(
        f"SC per pair — {label}  (ceiling = log({NUM_CARDS}) = {ceiling:.3f})",
        fontsize=12, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sc_compare_two_seeds(
    all_sc: np.ndarray,
    seed_a: int,
    seed_b: int,
    output_path: Path,
    label: str,
) -> None:
    """Side-by-side panel comparing SC curves for two specific seeds.

    Designed for paper / talk figures that highlight role-flip across seeds:
    e.g. agent 0 as proposer in one seed and agent 0 as follower in another.
    The visual mirror-image (one curve flat-high, the other low-rising)
    makes the population-level role asymmetry concrete.
    """
    n_seeds, _, K = all_sc.shape
    if not (0 <= seed_a < n_seeds and 0 <= seed_b < n_seeds):
        raise ValueError(
            f"seed indices out of range: got {(seed_a, seed_b)} but only {n_seeds} seeds"
        )
    ks = np.arange(K)
    ceiling = math.log(NUM_CARDS)
    colors = ["#fb8500", "#9d4edd"]

    def _role_label(seed_idx: int) -> str:
        agent0_sc0 = float(all_sc[seed_idx, 0, 0])
        agent1_sc0 = float(all_sc[seed_idx, 1, 0])
        if agent0_sc0 > agent1_sc0 + 0.5:
            return "agent 0 = proposer (committed at k=0)"
        if agent1_sc0 > agent0_sc0 + 0.5:
            return "agent 0 = follower (commits late)"
        return "symmetric protocol"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for i, seed in enumerate([seed_a, seed_b]):
        ax = axes[i]
        for agent_idx in range(2):
            final = float(all_sc[seed, agent_idx, -1])
            ax.plot(
                ks, all_sc[seed, agent_idx], color=colors[agent_idx],
                marker="o", markersize=5, linewidth=2.0,
                label=f"agent {agent_idx}  (final={final:.2f})",
            )
        ax.axhline(ceiling, color="gray", linestyle="--", linewidth=0.8,
                   label=f"ceiling = {ceiling:.2f}")
        ax.set_title(f"seed {seed} — {_role_label(seed)}", fontsize=11)
        ax.set_xticks(ks)
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_xlabel("deliberation slot k")
        ax.set_ylim(-0.05, ceiling * 1.05)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Speaker Consistency (nats)")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"SC seed comparison — {label}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sc_per_seed_grid(all_sc: np.ndarray, output_path: Path, label: str) -> None:
    """Small-multiples grid: one subplot per seed, both agents per panel.

    Lets you spot per-seed protocol patterns at a glance:
      - flat-high curve from k=0  -> agent committed to a fixed-frame strategy (bad under OP)
      - low-rising curve          -> agent conditions on obs/partner (healthy)
      - both rising symmetric     -> mutual gradual commitment
      - both flat-high            -> both fixed-frame (joint failure under OP)
      - both stuck low            -> undertrained
    """
    n_seeds, _, K = all_sc.shape
    n_cols = 4
    n_rows = math.ceil(n_seeds / n_cols)
    ks = np.arange(K)
    ceiling = math.log(NUM_CARDS)
    colors = ["#fb8500", "#9d4edd"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.4),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for s in range(n_seeds):
        r, c = divmod(s, n_cols)
        ax = axes[r, c]
        for agent_idx in range(2):
            ax.plot(ks, all_sc[s, agent_idx], color=colors[agent_idx],
                    marker="o", markersize=3, linewidth=1.6,
                    label=f"agent {agent_idx}" if s == 0 else None)
        ax.axhline(ceiling, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"seed {s}", fontsize=10)
        ax.set_ylim(-0.05, ceiling * 1.05)
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)

    # Hide unused panels
    for s in range(n_seeds, n_rows * n_cols):
        r, c = divmod(s, n_cols)
        axes[r, c].axis("off")

    # Single legend for the figure
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    # Common axis labels
    fig.supxlabel("deliberation slot k", fontsize=11)
    fig.supylabel("Speaker Consistency (nats)", fontsize=11)
    fig.suptitle(
        f"SC per seed — {label}  (ceiling = log({NUM_CARDS}) = {ceiling:.3f})",
        fontsize=12, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--compare-seeds", default=None,
        help=(
            "Two seed indices 'I,J' (e.g. '5,2'). Produces "
            "sc_compare_<I>_<J>.png with a side-by-side panel. Useful for "
            "highlighting role-flip across seeds (proposer vs follower)."
        ),
    )
    parser.add_argument(
        "--pairs", default=None,
        help=(
            "Cross-play pairs 'i1,j1;i2,j2;...' (e.g. '0,5;5,0;0,11;11,0'). "
            "For each pair, agent 0 uses seed i's params and agent 1 uses "
            "seed j's. Produces sc_pairs_<label>.png (small-multiples grid)."
        ),
    )
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
            ev.env, ev.policy, params, params, args.num_episodes, ev.max_steps,
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

    pair_sc_arr = None
    pairs_parsed: list[tuple[int, int]] = []
    if args.pairs:
        try:
            pairs_parsed = [
                tuple(int(x) for x in p.split(","))  # type: ignore[misc]
                for p in args.pairs.split(";") if p.strip()
            ]
        except ValueError as exc:
            raise SystemExit(f"--pairs must be 'i,j;i,j;...' of ints: {exc}") from exc
        for i, j in pairs_parsed:
            if not (0 <= i < ev.num_seeds and 0 <= j < ev.num_seeds):
                raise SystemExit(
                    f"pair ({i},{j}) out of range for {ev.num_seeds} seeds"
                )

        pair_sc_list = []
        for seed_a, seed_b in pairs_parsed:
            params_a = jax.tree.map(lambda x: x[seed_a], ev.params)
            params_b = jax.tree.map(lambda x: x[seed_b], ev.params)
            ep_messages, ep_actions = _collect_rollouts(
                ev.env, ev.policy, params_a, params_b,
                args.num_episodes, ev.max_steps,
                seed_offset=seed_a * 100 + seed_b,
                greedy=args.greedy,
            )
            sc = compute_sc_slotwise(ep_messages, ep_actions, NUM_CARDS, NUM_CARDS)
            pair_sc_list.append(sc)
            tag = "SP" if seed_a == seed_b else "XP"
            print(f"\n{tag} pair (a0=seed{seed_a}, a1=seed{seed_b})  SC:")
            print(_format_sc_grid(sc))
        pair_sc_arr = np.stack(pair_sc_list, axis=0)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = ev.label.replace("/", "_").replace(" ", "_")
        np.save(out / f"sc_{slug}.npy", all_sc_arr)
        print(f"\nSaved {out / f'sc_{slug}.npy'}  shape={all_sc_arr.shape}")
        _plot_sc(all_sc_arr, out / f"sc_{slug}.png", ev.label)
        print(f"Saved {out / f'sc_{slug}.png'}")
        _plot_sc_per_seed_grid(all_sc_arr, out / f"sc_per_seed_{slug}.png", ev.label)
        print(f"Saved {out / f'sc_per_seed_{slug}.png'}")
        if pair_sc_arr is not None:
            np.savez(
                out / f"sc_pairs_{slug}.npz",
                sc=pair_sc_arr, pairs=np.asarray(pairs_parsed, dtype=np.int32),
            )
            print(f"Saved {out / f'sc_pairs_{slug}.npz'}  shape={pair_sc_arr.shape}")
            _plot_sc_pairs_grid(
                pair_sc_arr, pairs_parsed,
                out / f"sc_pairs_{slug}.png", ev.label,
            )
            print(f"Saved {out / f'sc_pairs_{slug}.png'}")
        if args.compare_seeds:
            try:
                seed_a, seed_b = (int(s) for s in args.compare_seeds.split(","))
            except ValueError as exc:
                raise SystemExit(
                    f"--compare-seeds must be 'I,J' (two ints): {exc}"
                ) from exc
            cmp_path = out / f"sc_compare_{seed_a}_{seed_b}_{slug}.png"
            _plot_sc_compare_two_seeds(all_sc_arr, seed_a, seed_b, cmp_path, ev.label)
            print(f"Saved {cmp_path}")


if __name__ == "__main__":
    main()
