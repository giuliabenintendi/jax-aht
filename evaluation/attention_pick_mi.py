"""Per-(head, step, target) mutual information analysis.

For each attention head h and each timestep t in [0..max_steps-1], compute the
mutual information between:
    head_h's argmax view-slot at step t
and four candidate targets:
    (a) own_action_view_slot at step t          — what the agent emits NOW
    (b) own_pick_view_slot at decision step     — what the agent eventually picks
    (c) partner_pick_view_slot at decision step — what partner eventually picks
    (d) partner_msg_view_slot at step t         — what partner is signalling NOW

A head with high MI(head_argmax, own_pick_at_decision) is "decision-predictive":
its attention at intermediate timesteps tells us where the agent will pick.
A head with high MI(head_argmax, partner_msg) is "message-tracking".

Outputs per seed × agent:
    - heatmap PNG: heads (rows) × steps (cols), one heatmap per target → 4 panels
    - CSV with per-(head, step, target) MI in nats

Usage:
    ./run_gpu.sh 5 evaluation.attention_pick_mi \
        --checkpoint <path>/saved_train_run \
        --num-episodes 256
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _mi_int(xs: np.ndarray, ys: np.ndarray, n_x: int = NUM_CARDS,
            n_y: int = NUM_CARDS) -> float:
    """Empirical MI between two integer-valued sequences, both in [0, n_*).

    Both -1 entries are dropped. MI in nats.
    """
    mask = (xs >= 0) & (ys >= 0)
    if mask.sum() < 2:
        return 0.0
    xs = xs[mask]
    ys = ys[mask]
    joint = np.zeros((n_x, n_y), dtype=np.float64)
    np.add.at(joint, (xs, ys), 1.0)
    joint /= joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    safe = (joint > 0) & (px > 0) & (py > 0)
    mi = float(np.sum(joint[safe] * np.log(joint[safe] / (px * py)[safe])))
    return max(0.0, mi)


def _head_argmax_view_slot(head_attn_2d: np.ndarray, card_masks: np.ndarray) -> int:
    per_slot = np.einsum("hw,chw->c", head_attn_2d, card_masks)
    if per_slot.sum() < 1e-4:
        return -1
    return int(per_slot.argmax())


def _collect_episode(ep_states, attn_maps, ep_actions, ep_messages,
                     agent_idx: int, card_masks: np.ndarray, max_steps: int,
                     num_heads: int):
    """Returns per-step arrays for one episode.

    Returns dict with keys:
        head_argmax: (T, H) int
        own_action:  (T,) int — view-slot of agent's own action at step t
        own_pick:    int — view-slot of own pick at decision step
        partner_pick: int — view-slot of partner's pick at decision (in OWN view-frame)
        partner_msg: (T,) int — view-slot of partner's last-emitted msg as seen this step
    """
    agent_key = f"agent_{agent_idx}"
    partner_idx = 1 - agent_idx
    partner_key = f"agent_{partner_idx}"
    T = len(attn_maps[agent_key])

    head_argmax = np.full((T, num_heads), -1, dtype=np.int32)
    own_action = np.full((T,), -1, dtype=np.int32)
    partner_msg = np.full((T,), -1, dtype=np.int32)

    own_pick_gt = -1
    partner_pick_gt = -1

    for t in range(T):
        state_t = ep_states[t]
        pos_perm = np.asarray(state_t.env_state.per_agent_perm[agent_key])
        inv_recol = np.asarray(state_t.per_agent_inv_recolouring[agent_key])
        pos_perm_inv = np.argsort(pos_perm)

        is_decision = (t == max_steps - 1)
        if is_decision:
            raw = int(ep_actions[t][agent_idx])
            partner_raw_pick = int(ep_actions[t][partner_idx])
            if raw >= 0:
                own_pick_gt = int(inv_recol[raw])
                own_action[t] = int(pos_perm_inv[own_pick_gt])
            if partner_raw_pick >= 0:
                partner_inv_recol = np.asarray(state_t.per_agent_inv_recolouring[partner_key])
                partner_pick_gt = int(partner_inv_recol[partner_raw_pick])
        else:
            msg_raw = int(ep_messages[t][agent_idx]) if len(ep_messages) > t else -1
            if msg_raw >= 0:
                msg_gt = int(inv_recol[msg_raw])
                own_action[t] = int(pos_perm_inv[msg_gt])

        inner = _walk_to_card_state(state_t)
        partner_msg_gt = int(np.asarray(inner.messages)[partner_idx])
        if partner_msg_gt >= 0:
            partner_msg[t] = int(pos_perm_inv[partner_msg_gt])

        raw_attn = np.asarray(attn_maps[agent_key][t]).squeeze()
        if raw_attn.ndim == 2:
            # Old single-head format; replicate to num_heads
            raw_attn = np.stack([raw_attn] * num_heads, axis=-1)
        for h in range(min(num_heads, raw_attn.shape[-1])):
            head_argmax[t, h] = _head_argmax_view_slot(raw_attn[..., h], card_masks)

    # own_pick / partner_pick to view-slot of THIS agent
    own_pick_slot = -1
    partner_pick_slot_own_view = -1
    if own_pick_gt >= 0:
        # Use the last step's pos_perm (assume stable through episode)
        state_last = ep_states[max_steps - 1]
        pos_perm_last = np.asarray(state_last.env_state.per_agent_perm[agent_key])
        pos_perm_inv_last = np.argsort(pos_perm_last)
        own_pick_slot = int(pos_perm_inv_last[own_pick_gt])
        if partner_pick_gt >= 0:
            # Partner's pick translated to this agent's view-slot frame
            partner_pick_slot_own_view = int(pos_perm_inv_last[partner_pick_gt])

    return {
        "head_argmax":            head_argmax,
        "own_action":             own_action,
        "own_pick":               own_pick_slot,
        "partner_pick":           partner_pick_slot_own_view,
        "partner_msg":            partner_msg,
    }


def _save_mi_panel(mi_grid: np.ndarray, head_labels: list, step_labels: list,
                    target_labels: list, out_path: Path, title: str) -> None:
    """mi_grid shape: (num_targets, num_heads, num_steps)."""
    n_targets = mi_grid.shape[0]
    fig, axes = plt.subplots(1, n_targets, figsize=(3.5 * n_targets + 1, 3.0))
    if n_targets == 1:
        axes = [axes]
    vmax = float(mi_grid.max()) if mi_grid.max() > 0 else 1.0
    log_n = math.log(NUM_CARDS)
    for i, ax in enumerate(axes):
        m = mi_grid[i]
        im = ax.imshow(m, cmap="viridis", vmin=0.0, vmax=min(vmax, log_n),
                       aspect="auto")
        for h_idx in range(m.shape[0]):
            for s_idx in range(m.shape[1]):
                v = m[h_idx, s_idx]
                tc = "white" if v < (vmax * 0.5) else "black"
                ax.text(s_idx, h_idx, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color=tc)
        ax.set_xticks(range(m.shape[1]))
        ax.set_yticks(range(m.shape[0]))
        ax.set_xticklabels(step_labels, fontsize=8)
        ax.set_yticklabels(head_labels, fontsize=8)
        ax.set_title(target_labels[i], fontsize=10)
        if i == 0:
            ax.set_ylabel("head", fontsize=9)
        ax.set_xlabel("step", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--seed-idx", type=int, default=-1,
                        help="If >=0, only analyze that one seed. Else all.")
    parser.add_argument("--episode-rng-base", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sampled", action="store_true")
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    greedy = not args.sampled

    run_dir = Path(args.checkpoint).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "attention_pick_mi")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  label={ev.label}  seeds={ev.num_seeds}  eps={args.num_episodes}  greedy={greedy}")
    print(f"  output: {out_dir}\n")

    card_masks = np.asarray(build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W))
    seeds_to_run = ([args.seed_idx] if args.seed_idx >= 0
                    else list(range(ev.num_seeds)))

    csv_path = out_dir / "attention_pick_mi.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "agent", "head", "step", "target", "MI_nats"])

        for seed_idx in seeds_to_run:
            params = jax.tree.map(lambda x: x[seed_idx], ev.params)

            for agent_idx, agent_key in enumerate(("agent_0", "agent_1")):
                # Accumulate per-(head, step) sequences across episodes.
                per_step_head_seq = None   # (E, T, H)
                per_step_own_action_seq = None  # (E, T)
                per_step_partner_msg_seq = None  # (E, T)
                own_pick_seq = np.zeros(args.num_episodes, dtype=np.int32)
                partner_pick_seq = np.zeros(args.num_episodes, dtype=np.int32)

                for ep in range(args.num_episodes):
                    rng = jax.random.PRNGKey(
                        args.episode_rng_base + seed_idx * 1000 + ep
                    )
                    ep_states, attn_maps, ep_actions, ep_messages = run_episode_with_states(
                        rng, ev.env, params, ev.policy, params, ev.policy,
                        ev.max_steps, collect_attention=True, greedy=greedy,
                    )
                    raw_first = np.asarray(attn_maps[agent_key][0]).squeeze()
                    num_heads = raw_first.shape[-1] if raw_first.ndim == 3 else 1
                    data = _collect_episode(
                        ep_states, attn_maps, ep_actions, ep_messages,
                        agent_idx, card_masks, ev.max_steps, num_heads,
                    )
                    if per_step_head_seq is None:
                        T = data["head_argmax"].shape[0]
                        per_step_head_seq = np.full((args.num_episodes, T, num_heads), -1, dtype=np.int32)
                        per_step_own_action_seq = np.full((args.num_episodes, T), -1, dtype=np.int32)
                        per_step_partner_msg_seq = np.full((args.num_episodes, T), -1, dtype=np.int32)
                    per_step_head_seq[ep] = data["head_argmax"]
                    per_step_own_action_seq[ep] = data["own_action"]
                    per_step_partner_msg_seq[ep] = data["partner_msg"]
                    own_pick_seq[ep] = data["own_pick"]
                    partner_pick_seq[ep] = data["partner_pick"]

                T_total = per_step_head_seq.shape[1]
                num_heads = per_step_head_seq.shape[2]
                target_labels = ["own_action_now", "own_pick@dec", "partner_pick@dec", "partner_msg_now"]
                mi_grid = np.zeros((4, num_heads, T_total), dtype=np.float64)

                for h in range(num_heads):
                    head_arg = per_step_head_seq[:, :, h]
                    for t in range(T_total):
                        mi_grid[0, h, t] = _mi_int(head_arg[:, t], per_step_own_action_seq[:, t])
                        mi_grid[1, h, t] = _mi_int(head_arg[:, t], own_pick_seq)
                        mi_grid[2, h, t] = _mi_int(head_arg[:, t], partner_pick_seq)
                        mi_grid[3, h, t] = _mi_int(head_arg[:, t], per_step_partner_msg_seq[:, t])

                # Persist CSV rows
                for h in range(num_heads):
                    for t in range(T_total):
                        for ti, name in enumerate(target_labels):
                            w.writerow([seed_idx, agent_key, h, t, name, f"{mi_grid[ti, h, t]:.4f}"])

                # Stdout per-head summary (mean over steps)
                print(f"=== seed {seed_idx}  {agent_key} ===")
                print(f"  {'head':<6} " + "  ".join(f"{tl:>16s}" for tl in target_labels))
                for h in range(num_heads):
                    means = [mi_grid[ti, h].mean() for ti in range(4)]
                    print(f"  head {h:<2} " +
                          "  ".join(f"{m:>16.3f}" for m in means))

                # Save PNG
                png_path = out_dir / f"mi_seed{seed_idx}_{agent_key}.png"
                head_labels = [f"h{h}" for h in range(num_heads)]
                step_labels = [f"t{t}" for t in range(T_total)]
                _save_mi_panel(mi_grid, head_labels, step_labels, target_labels,
                               png_path,
                               title=f"seed {seed_idx} {agent_key} — head×step MI (nats)")
                print(f"  [saved {png_path}]\n")

    print(f"\nDone. CSV: {csv_path}")
    print(f"Reference: MI > 0 means information; MI = log(5) ≈ 1.609 nats is the ceiling.")


if __name__ == "__main__":
    main()
