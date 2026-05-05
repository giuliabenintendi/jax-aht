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


if __name__ == "__main__":
    main()
