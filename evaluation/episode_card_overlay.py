"""Per-episode card-attention overlays for both agents, canonical frame.

For each of N episodes, produces a single PNG with two side-by-side panels:
  - Left:  agent_0 per-step per-card head-averaged attention (T rows × 5 cards)
  - Right: agent_1 per-step per-card head-averaged attention

Both panels are in CANONICAL frame (after applying each agent's view→canonical
permutation), so "card 0" means the same physical card in both panels. This
makes it possible to read by eye whether the two agents attend to the same
physical card.

Each row's argmax cell is annotated with an *. Decision-step row (t = T−1)
also marks each agent's picked card with [P] (white box) — comparing whether
the two [P]s land in the same column tells you whether the agents coordinated.

Why this answers the questions:
  (1) "Does partner's attention influence the agent?" — at step t, agent_0's
      obs contains agent_1's attention from step t−1 (per-head, canonical).
      So if agent_0's t-row pattern shifts to track agent_1's (t−1)-row
      pattern across episodes, the agent is using the feed.
  (2) "Are they coordinating via attention?" — at t = T−1, the two [P]s in
      the same column means they picked the same canonical card.

Usage:
    ./run_gpu.sh 5 evaluation.episode_card_overlay \\
        --checkpoint <path>/saved_train_run \\
        --seed-idx 3 --num-episodes 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from agents.ja_utils import build_card_masks
from evaluation._card_game_utils import load_card_game_eval
from evaluation.vis_episodes import (
    run_episode_with_states, _get_card_game_position_perm,
)


IMG_H, IMG_W = 21, 35
FEAT_H, FEAT_W = 6, 9


def _per_card_attention(attn_map_per_step, card_masks):
    """Pool per-head spatial attention onto 5 cards (head-averaged).

    Returns (T, 5).
    """
    out = []
    for a in attn_map_per_step:
        a_sq = np.asarray(a).squeeze()
        if a_sq.ndim == 3:
            a_2d = a_sq.mean(axis=-1)
        else:
            a_2d = a_sq
        per_card = np.einsum("hw,chw->c", a_2d, card_masks)
        out.append(per_card)
    return np.stack(out, axis=0)  # (T, 5)


def _canonicalize(per_card_view_frame, perms_per_step):
    """Translate each step's per-card attention from view→canonical.

    perms_per_step[t][k] = canonical_slot of card at view-slot k at step t.
    Returns (T, 5) in canonical frame.
    """
    T = per_card_view_frame.shape[0]
    out = np.zeros_like(per_card_view_frame)
    for t in range(T):
        p = np.asarray(perms_per_step[t])
        for k in range(5):
            out[t, p[k]] = per_card_view_frame[t, k]
    return out


def _draw_panel(ax, mat, title, pick_canon=None, max_steps=None):
    """Draw a (T, 5) heatmap. Annotate per-row argmax with *. Mark pick with box."""
    T = mat.shape[0]
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0,
                   vmax=max(0.05, float(mat.max())))
    ax.set_xticks(range(5))
    ax.set_xticklabels([f"c{c}" for c in range(5)])
    ax.set_yticks(range(T))
    ax.set_yticklabels([f"t{t}" for t in range(T)])
    ax.set_title(title, fontsize=10)
    for t in range(T):
        row = mat[t]
        if row.sum() < 1e-6:
            continue
        am = int(np.argmax(row))
        ax.text(am, t, "*", ha="center", va="center",
                color="white", fontsize=14, fontweight="bold")
    if pick_canon is not None and pick_canon >= 0:
        t = (max_steps - 1) if max_steps is not None else (T - 1)
        ax.add_patch(Rectangle(
            (pick_canon - 0.45, t - 0.45), 0.9, 0.9,
            fill=False, edgecolor="red", linewidth=2.5,
        ))
    return im


def _resolve_pick_canonical(ep_states, ep_actions, agent_key, max_steps):
    """Resolve agent's pick at decision step in canonical frame.

    The recorded action goes through OP recolouring + position shuffle, so the
    raw int needs both inverses applied to land in canonical card space.
    """
    state_last = ep_states[max_steps - 1]
    raw = int(ep_actions[max_steps - 1][0 if agent_key == "agent_0" else 1])
    if raw < 0:
        return -1
    inv_recol = np.asarray(state_last.per_agent_inv_recolouring[agent_key])
    return int(inv_recol[raw])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=6)
    parser.add_argument("--episode-rng-base", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sampled", action="store_true")
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    greedy = not args.sampled
    run_dir = Path(args.checkpoint).resolve().parent
    out_dir = (Path(args.output_dir) if args.output_dir
               else run_dir / f"episode_card_overlay_seed{args.seed_idx}")
    out_dir.mkdir(parents=True, exist_ok=True)

    card_masks = np.asarray(build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W))
    policy_scalar_dim = int(getattr(ev.policy.network, "scalar_dim", 0))
    partner_feed_dim = policy_scalar_dim if policy_scalar_dim > 0 else 5
    use_card_masks = jnp.asarray(card_masks) if policy_scalar_dim > 0 else None

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  seed_idx={args.seed_idx}  num_eps={args.num_episodes}  greedy={greedy}")
    print(f"  partner_feed_dim={partner_feed_dim}  output: {out_dir}\n")

    params = jax.tree.map(lambda x: x[args.seed_idx], ev.params)

    for ep in range(args.num_episodes):
        rng = jax.random.PRNGKey(
            args.episode_rng_base + args.seed_idx * 1000 + ep
        )
        ep_states, attn_maps, ep_actions, _ = run_episode_with_states(
            rng, ev.env, params, ev.policy, params, ev.policy,
            ev.max_steps, collect_attention=True, greedy=greedy,
            ja_card_masks=use_card_masks,
            partner_feed_dim=partner_feed_dim,
        )

        perms_0 = [_get_card_game_position_perm(s, "agent_0") for s in ep_states[:-1]]
        perms_1 = [_get_card_game_position_perm(s, "agent_1") for s in ep_states[:-1]]
        pc_0_view = _per_card_attention(attn_maps["agent_0"], card_masks)
        pc_1_view = _per_card_attention(attn_maps["agent_1"], card_masks)
        pc_0_canon = _canonicalize(pc_0_view, perms_0)
        pc_1_canon = _canonicalize(pc_1_view, perms_1)

        pick_0_canon = _resolve_pick_canonical(ep_states, ep_actions, "agent_0", ev.max_steps)
        pick_1_canon = _resolve_pick_canonical(ep_states, ep_actions, "agent_1", ev.max_steps)
        matched = pick_0_canon == pick_1_canon and pick_0_canon >= 0

        fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
        _draw_panel(axes[0], pc_0_canon, f"agent_0 (pick=c{pick_0_canon})",
                    pick_canon=pick_0_canon, max_steps=ev.max_steps)
        _draw_panel(axes[1], pc_1_canon, f"agent_1 (pick=c{pick_1_canon})",
                    pick_canon=pick_1_canon, max_steps=ev.max_steps)
        match_str = "MATCH" if matched else "MISS"
        fig.suptitle(
            f"seed {args.seed_idx} ep {ep} — canonical-frame per-card attn "
            f"({match_str}: a0=c{pick_0_canon}, a1=c{pick_1_canon})",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = out_dir / f"ep{ep:02d}_{'match' if matched else 'miss'}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  ep {ep}: a0=c{pick_0_canon}  a1=c{pick_1_canon}  -> {match_str}"
              f"   [saved {out_path.name}]")

    print(f"\nDone. {args.num_episodes} PNGs in {out_dir}")


if __name__ == "__main__":
    main()
