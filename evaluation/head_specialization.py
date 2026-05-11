"""Quantitative head-specialization analysis for the JA image agent.

Tests the hypothesis: do different attention heads specialize for different
functional roles (e.g. one tracks partner's message, another tracks the
agent's own pick)?

For each (seed, agent, head), runs N episodes and accumulates:
  - P(head argmax == partner's most recent message view-slot)  — deliberation
  - P(head argmax == agent's own action view-slot this step)   — deliberation
  - P(head argmax == agent's pick view-slot at decision step)  — decision
  - mean total card-row mass for this head (how much it attends to cards)

Output: per (seed, agent) table of 4 heads × 4 alignment metrics, plus
average across seeds.

Usage:
    ./run_gpu.sh 5 evaluation.head_specialization \\
        --checkpoint <path>/saved_train_run \\
        --num-episodes 256
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.ja_utils import build_card_masks
from envs.card_game.rendering import NUM_CARDS
from evaluation._card_game_utils import load_card_game_eval
from evaluation.vis_episodes import run_episode_with_states


IMG_H, IMG_W = 21, 35
FEAT_H, FEAT_W = 6, 9


def _walk_to_card_state(state):
    s = state
    while hasattr(s, "env_state") and not hasattr(s, "card_permutation"):
        s = s.env_state
    return s


def _head_argmax_view_slot(head_attn_2d: np.ndarray, card_masks: np.ndarray) -> int:
    """For one head's (fh, fw) attention map, pool with card_masks to get a
    (5,) per-view-slot vector, return argmax slot.
    Returns -1 if total card-row mass is below a tiny threshold (head isn't
    really attending to cards at all)."""
    per_slot = np.einsum("hw,chw->c", head_attn_2d, card_masks)
    total = float(per_slot.sum())
    if total < 1e-4:
        return -1
    return int(per_slot.argmax())


def _analyze_episode(
    ep_states, attn_maps, ep_actions, ep_messages,
    agent_key: str, agent_idx: int, card_masks: np.ndarray,
    max_steps: int,
):
    """Walk through one episode, accumulate per-step per-head alignment data.

    Returns dict: head_idx -> dict of counts:
      'argmax_eq_partner_msg' (only counted on deliberation steps where partner_msg >= 0)
      'argmax_eq_own_action'  (deliberation steps only)
      'argmax_eq_decision_pick' (decision step only)
      'card_mass_sum'         (sum of phys.sum(-1) across all steps)
      'partner_msg_steps'     (number of steps where partner has emitted a msg)
      'deliberation_steps'    (number of deliberation steps)
      'decision_steps'        (always 1)
    """
    partner_idx = 1 - agent_idx
    accum = {h: {
        "argmax_eq_partner_msg": 0,
        "argmax_eq_own_action":  0,
        "argmax_eq_decision_pick": 0,
        "card_mass_sum": 0.0,
    } for h in range(4)}
    counts = {
        "partner_msg_steps": 0,
        "deliberation_steps": 0,
        "decision_steps": 0,
    }

    num_steps = len(attn_maps[agent_key])
    for t in range(num_steps):
        state_t = ep_states[t]
        pos_perm = np.asarray(state_t.env_state.per_agent_perm[agent_key])
        inv_recol = np.asarray(state_t.per_agent_inv_recolouring[agent_key])
        pos_perm_inv = np.argsort(pos_perm)

        # Own action at this step → view-slot
        action_t = int(ep_actions[t][agent_idx])
        if action_t >= 0:
            action_gt = int(inv_recol[action_t])
            own_view_slot = int(pos_perm_inv[action_gt])
        else:
            own_view_slot = -1

        # Partner's last-emitted GT message → view-slot
        inner = _walk_to_card_state(state_t)
        partner_msg_gt = int(np.asarray(inner.messages)[partner_idx])
        if partner_msg_gt >= 0:
            partner_view_slot = int(pos_perm_inv[partner_msg_gt])
        else:
            partner_view_slot = -1

        is_decision = (t == max_steps - 1)

        # Per-head argmax view-slot
        raw = np.asarray(attn_maps[agent_key][t]).squeeze()
        if raw.ndim == 2:
            # Old format, single head — replicate 4 times for consistency
            raw = np.stack([raw] * 4, axis=-1)
        # raw is (fh, fw, num_heads)
        for h_idx in range(raw.shape[-1]):
            head_attn = raw[..., h_idx]
            head_slot = _head_argmax_view_slot(head_attn, card_masks)

            # card mass = total mass on card row (regardless of slot)
            head_card_mass = float(head_attn.sum() * 0.0 + np.einsum("hw,chw->", head_attn, card_masks))
            accum[h_idx]["card_mass_sum"] += head_card_mass

            if head_slot < 0:
                continue

            if not is_decision:
                if partner_view_slot >= 0 and head_slot == partner_view_slot:
                    accum[h_idx]["argmax_eq_partner_msg"] += 1
                if own_view_slot >= 0 and head_slot == own_view_slot:
                    accum[h_idx]["argmax_eq_own_action"] += 1

            if is_decision:
                if own_view_slot >= 0 and head_slot == own_view_slot:
                    accum[h_idx]["argmax_eq_decision_pick"] += 1

        # Step counts (once per step, not per head)
        if not is_decision:
            counts["deliberation_steps"] += 1
            if partner_view_slot >= 0:
                counts["partner_msg_steps"] += 1
        else:
            counts["decision_steps"] += 1

    return accum, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--seed-idx", type=int, default=-1,
                        help="If >=0, only analyze that one seed. Else all seeds.")
    parser.add_argument("--episode-rng-base", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sampled", action="store_true",
                        help="Sample actions instead of greedy.")
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    greedy = not args.sampled

    run_dir = Path(args.checkpoint).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "head_specialization")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  label={ev.label}  seeds={ev.num_seeds}  eps={args.num_episodes}  greedy={greedy}")
    print(f"  output: {out_dir}\n")

    card_masks = np.asarray(build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W))

    seeds_to_run = (
        [args.seed_idx] if args.seed_idx >= 0
        else list(range(ev.num_seeds))
    )

    csv_path = out_dir / "head_specialization.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "seed", "agent", "head",
            "P(==partner_msg | delib & msg_present)",
            "P(==own_action | delib)",
            "P(==decision_pick)",
            "mean_card_mass",
        ])

        for seed_idx in seeds_to_run:
            params = jax.tree.map(lambda x: x[seed_idx], ev.params)

            for agent_idx, agent_key in enumerate(("agent_0", "agent_1")):
                head_accum = {h: {
                    "argmax_eq_partner_msg": 0,
                    "argmax_eq_own_action": 0,
                    "argmax_eq_decision_pick": 0,
                    "card_mass_sum": 0.0,
                } for h in range(4)}
                total_counts = {
                    "partner_msg_steps": 0,
                    "deliberation_steps": 0,
                    "decision_steps": 0,
                }

                for ep in range(args.num_episodes):
                    rng = jax.random.PRNGKey(
                        args.episode_rng_base + seed_idx * 1000 + ep
                    )
                    ep_states, attn_maps, ep_actions, ep_messages = run_episode_with_states(
                        rng, ev.env, params, ev.policy, params, ev.policy,
                        ev.max_steps, collect_attention=True, greedy=greedy,
                    )
                    ep_accum, ep_counts = _analyze_episode(
                        ep_states, attn_maps, ep_actions, ep_messages,
                        agent_key, agent_idx, card_masks, ev.max_steps,
                    )
                    for h in range(4):
                        for k in head_accum[h]:
                            head_accum[h][k] += ep_accum[h][k]
                    for k in total_counts:
                        total_counts[k] += ep_counts[k]

                total_steps = total_counts["deliberation_steps"] + total_counts["decision_steps"]

                print(f"=== seed {seed_idx}  {agent_key} "
                      f"({args.num_episodes} eps, {total_steps} steps) ===")
                print(
                    f"  {'head':<6} {'==msg|delib':>12} {'==action|delib':>15} "
                    f"{'==pick|dec':>12} {'mean_mass':>10}"
                )
                for h in range(4):
                    p_msg = (
                        head_accum[h]["argmax_eq_partner_msg"]
                        / max(total_counts["partner_msg_steps"], 1)
                    )
                    p_action = (
                        head_accum[h]["argmax_eq_own_action"]
                        / max(total_counts["deliberation_steps"], 1)
                    )
                    p_pick = (
                        head_accum[h]["argmax_eq_decision_pick"]
                        / max(total_counts["decision_steps"], 1)
                    )
                    mean_mass = head_accum[h]["card_mass_sum"] / max(total_steps, 1)
                    print(
                        f"  head {h:<2} {p_msg:>12.3f} {p_action:>15.3f} "
                        f"{p_pick:>12.3f} {mean_mass:>10.3f}"
                    )
                    w.writerow([
                        seed_idx, agent_key, h,
                        f"{p_msg:.4f}", f"{p_action:.4f}",
                        f"{p_pick:.4f}", f"{mean_mass:.4f}",
                    ])
                print()

    print(f"\nSaved CSV: {csv_path}")
    print(
        "\nReference: random argmax over 5 slots would give P ≈ 1/5 = 0.2 for any "
        "alignment target. Heads with P substantially above 0.2 are doing real "
        "alignment with that target. mean_card_mass is the fraction of attention "
        "the head puts on the card row (cells overlapping card pixels)."
    )


if __name__ == "__main__":
    main()
