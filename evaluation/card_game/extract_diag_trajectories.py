"""Extract ground-truth trajectories + RNN hidden states for the paper-faithful
Dec-POMDP diagnostics (Tessera et al., AAMAS 2026).

Per agent per step we dump, in the GROUND-TRUTH (un-Other-Play) frame:
  - `intents`  : the canonical emitted card (the action),
  - `hstates`  : the RNN hidden state that produced it (UserData.hidden_states),
  - `gt_attn`  : each agent's canonical per-card attention (the JA channel;
                 zeros when the policy has no card-attention head).

The runner assembles the observation `[step, channel]` from these — the channel
is the partner's PREVIOUS canonical message (comm, derived from `intents`) or the
partner's previous canonical attention (JA, from `gt_attn`). Ground-truth frame
is required because Other-Play relabels per agent: only in the canonical frame
are obs and actions in one consistent frame and cross-agent coordination defined.

XP uses DISJOINT seed pairs (0,1),(2,3),... — the m=N/2 independent-units
protocol — not all-seed-0.

Hidden states are bulky: ~`num_seeds * episodes * T * 2 * hidden_dim` float16.
Default episodes are modest; use `--max-seeds` for a lighter first pass. The
scorer runs on the same box, so size is a disk concern, not a transfer one.

    ./run_gpu.sh 0 evaluation.card_game.extract_diag_trajectories \\
        --checkpoint <.../saved_train_run> --condition ja_shape --num-episodes 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from evaluation.card_game._card_game_utils import load_card_game_eval
from evaluation.card_game.eval_time_to_agree import _build_pair_runner, setup_card_feed


def _collect(runner, params_a, params_b, num_episodes: int, rng_base: int):
    rngs = jax.random.split(jax.random.PRNGKey(rng_base), num_episodes)
    intents, hstates, gt_attn = runner(rngs, params_a, params_b)
    return (np.asarray(intents),
            np.asarray(hstates, dtype=np.float16),
            np.asarray(gt_attn, dtype=np.float16))


def _decision_match(intents: np.ndarray) -> float:
    last = intents[:, -1, :]
    return float(np.mean((last[:, 0] == last[:, 1]) & (last[:, 0] >= 0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", required=True,
                        help="Condition label; output basename and diagnostics scenario id.")
    parser.add_argument("--num-episodes", type=int, default=300,
                        help="Episodes per SP seed and per XP pair. ~T*episodes rows feed the "
                             "MI estimators (their max_samples is 8000).")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Cap SP seeds (also caps the disjoint XP pairs) for a lighter pass.")
    parser.add_argument("--max-xp-pairs", type=int, default=None,
                        help="Cap on disjoint XP pairs (default: all N/2).")
    parser.add_argument("--greedy", action="store_true",
                        help="Greedy actions. Default SAMPLED — the MI estimators need entropy.")
    parser.add_argument("--use-latest", action="store_true",
                        help="Last chunk per seed instead of best-by-return (failed/partial runs).")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to evaluation/card_game/diag_data/")
    args = parser.parse_args()

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ev = load_card_game_eval(args.checkpoint, use_latest=args.use_latest)
    greedy = args.greedy
    T = ev.max_steps
    has_comm = bool(ev.env_kwargs.get("communication", False))
    partner_feed_dim, card_masks = setup_card_feed(ev)
    has_attn = card_masks is not None
    runner = _build_pair_runner(ev, greedy, partner_feed_dim, card_masks, return_diag=True)

    num_seeds = ev.num_seeds if args.max_seeds is None else min(args.max_seeds, ev.num_seeds)
    print(
        f"\nCondition {args.condition}: label={ev.label}  comm={has_comm}  attn={has_attn}  "
        f"seeds={num_seeds}/{ev.num_seeds}  T={T}  eps={args.num_episodes}  greedy={greedy}"
    )

    sp_intents, sp_hstates, sp_gt_attn = [], [], []
    for s in range(num_seeds):
        params = jax.tree.map(lambda x: x[s], ev.params)
        it, h, at = _collect(runner, params, params, args.num_episodes, 1_000_000 + s * 10_000)
        sp_intents.append(it)
        sp_hstates.append(h)
        sp_gt_attn.append(at)
        print(f"  SP seed {s:2d}: decision-match={_decision_match(it):.3f}")
    sp_intents = np.stack(sp_intents)
    sp_hstates = np.stack(sp_hstates)
    sp_gt_attn = np.stack(sp_gt_attn)

    pairs = [(2 * k, 2 * k + 1) for k in range(num_seeds // 2)]
    if args.max_xp_pairs is not None:
        pairs = pairs[: args.max_xp_pairs]
    xp_intents, xp_hstates, xp_gt_attn = [], [], []
    for idx, (i, j) in enumerate(pairs):
        pi = jax.tree.map(lambda x: x[i], ev.params)
        pj = jax.tree.map(lambda x: x[j], ev.params)
        it, h, at = _collect(runner, pi, pj, args.num_episodes, 2_000_000 + idx * 10_000)
        xp_intents.append(it)
        xp_hstates.append(h)
        xp_gt_attn.append(at)
        print(f"  XP pair {idx + 1:2d}/{len(pairs)} ({i},{j}): decision-match={_decision_match(it):.3f}")
    if pairs:
        xp_intents = np.stack(xp_intents)
        xp_hstates = np.stack(xp_hstates)
        xp_gt_attn = np.stack(xp_gt_attn)
        xp_pairs = np.array(pairs, dtype=np.int32)
    else:
        xp_intents = np.zeros((0, args.num_episodes, T, 2), np.int32)
        xp_hstates = np.zeros((0,), np.float16)
        xp_gt_attn = np.zeros((0,), np.float16)
        xp_pairs = np.zeros((0, 2), np.int32)

    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(__file__).parent / "diag_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.condition}.npz"
    np.savez(
        out_path,
        sp_intents=sp_intents, sp_hstates=sp_hstates, sp_gt_attn=sp_gt_attn,
        xp_intents=xp_intents, xp_hstates=xp_hstates, xp_gt_attn=xp_gt_attn,
        xp_pairs=xp_pairs, max_steps=T, has_comm=has_comm, has_attn=has_attn,
        greedy=greedy, condition=args.condition, label=ev.label,
    )
    print(f"\nSaved {out_path}  (sp_intents {sp_intents.shape}, sp_hstates {sp_hstates.shape})")


if __name__ == "__main__":
    main()
