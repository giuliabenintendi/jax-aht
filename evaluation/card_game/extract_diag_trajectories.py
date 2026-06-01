"""Extract ground-truth action trajectories for the Dec-POMDP diagnostics probe.

Rolls out N episodes per pairing (self-play per seed + cross-play per seed pair)
from a trained card-game checkpoint and saves the per-step canonical (ground
truth, un-OP'd) emitted card for both agents. This is the raw signal that
`run_dec_pomdp_diagnostics.py` turns into `dec_pomdp_diagnostics.UserData`.

Why only the emitted card and nothing about the board: under Other-Play the base
env shuffle is disabled and both the position-shuffle and recolouring wrappers
are per-agent and visual-only, so the ground-truth board layout is constant. The
only varying, cross-agent-commensurable signal is each agent's per-step emitted
card (a deliberation message, or a pick on the decision step), recovered in the
ground-truth frame by `_action_to_ground_truth`. All observation-feature
construction is deferred to the diagnostics runner so the observation definition
can change without re-running rollouts.

Runs on the GPU box (needs JAX + checkpoints):

    ./run_gpu.sh 0 evaluation.card_game.extract_diag_trajectories \\
        --checkpoint <path_to_saved_train_run> \\
        --condition op_comm_shape \\
        --num-episodes 1000 --max-xp-pairs 6
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import jax
import numpy as np

from evaluation.card_game._card_game_utils import load_card_game_eval
from evaluation.card_game.eval_time_to_agree import _build_pair_runner, setup_card_feed


def _collect_intents(runner, params_a, params_b, num_episodes: int, rng_base: int) -> np.ndarray:
    """Run `num_episodes` for one pairing; return canonical intents `(E, T, 2)`."""
    rngs = jax.random.split(jax.random.PRNGKey(rng_base), num_episodes)
    return np.asarray(runner(rngs, params_a, params_b))


def _decision_match(intents: np.ndarray) -> float:
    """Fraction of episodes where both decision-step picks agree (>= 0)."""
    last = intents[:, -1, :]
    return float(np.mean((last[:, 0] == last[:, 1]) & (last[:, 0] >= 0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition", required=True,
                        help="Condition label; used as output basename and scenario id.")
    parser.add_argument("--num-episodes", type=int, default=1000,
                        help="Episodes per SP seed and per XP pair (MI needs ~1k).")
    parser.add_argument("--max-xp-pairs", type=int, default=6,
                        help="Cap on XP seed pairs (default 6 = the m=N/2 disjoint budget).")
    parser.add_argument("--max-seeds", type=int, default=None,
                        help="Cap on SP seeds for a quick pilot.")
    parser.add_argument("--greedy", action="store_true",
                        help="Greedy actions. Default is SAMPLED: the MI estimators need "
                             "action entropy; a deterministic argmax policy makes I(O;A) "
                             "and AA degenerate.")
    parser.add_argument("--use-latest", action="store_true",
                        help="Take the last saved chunk per seed instead of best-by-return. "
                             "Use for runs that failed before metrics were fully written "
                             "(e.g. op_ja_shaped) where best-per-seed scoring would error.")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to evaluation/card_game/diag_data/")
    args = parser.parse_args()

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ev = load_card_game_eval(args.checkpoint, use_latest=args.use_latest)
    greedy = args.greedy
    T = ev.max_steps
    has_comm = bool(ev.env_kwargs.get("communication", False))
    partner_feed_dim, card_masks = setup_card_feed(ev)
    runner = _build_pair_runner(ev, greedy, partner_feed_dim, card_masks)

    num_seeds = ev.num_seeds if args.max_seeds is None else min(args.max_seeds, ev.num_seeds)
    print(
        f"\nCondition {args.condition}: label={ev.label}  comm={has_comm}  "
        f"seeds={num_seeds}/{ev.num_seeds}  T={T}  eps/pairing={args.num_episodes}  greedy={greedy}"
    )

    sp_intents = []
    for s in range(num_seeds):
        params = jax.tree.map(lambda x: x[s], ev.params)
        it = _collect_intents(runner, params, params, args.num_episodes,
                              rng_base=1_000_000 + s * 10_000)
        sp_intents.append(it)
        print(f"  SP seed {s:2d}: decision-match={_decision_match(it):.3f}")
    sp_intents = np.stack(sp_intents)  # (num_seeds, E, T, 2)

    pairs = list(itertools.combinations(range(num_seeds), 2))[: args.max_xp_pairs]
    xp_intents = []
    for idx, (i, j) in enumerate(pairs):
        pi = jax.tree.map(lambda x: x[i], ev.params)
        pj = jax.tree.map(lambda x: x[j], ev.params)
        it = _collect_intents(runner, pi, pj, args.num_episodes,
                              rng_base=2_000_000 + idx * 10_000)
        xp_intents.append(it)
        print(f"  XP pair {idx + 1:2d}/{len(pairs)} ({i},{j}): decision-match={_decision_match(it):.3f}")
    xp_intents = np.stack(xp_intents) if xp_intents else np.zeros((0, args.num_episodes, T, 2), np.int32)
    xp_pairs = np.array(pairs, dtype=np.int32) if pairs else np.zeros((0, 2), np.int32)

    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(__file__).parent / "diag_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.condition}.npz"
    np.savez(
        out_path,
        sp_intents=sp_intents, xp_intents=xp_intents, xp_pairs=xp_pairs,
        max_steps=T, has_comm=has_comm, greedy=greedy,
        condition=args.condition, label=ev.label,
    )
    print(f"\nSaved {out_path}  (sp {sp_intents.shape}, xp {xp_intents.shape})")


if __name__ == "__main__":
    main()
