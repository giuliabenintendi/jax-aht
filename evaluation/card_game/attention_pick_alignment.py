"""Quick alignment test at the decision step:

For N self-play episodes, at the decision step (t=T-1) compute, per agent:
  1) P(own attention argmax  ==  own pick)
       — does the agent actually pick the card it was attending to?
  2) P(own attention argmax  ==  partner attention argmax in own view-slot frame)
       — do the two agents agree on which canonical card to attend to?

Both attention argmaxes are taken in the agent's own view-slot frame (the
partner's attention is translated into ego's frame via the OP perm). The pick
is the agent's raw action at the decision step, which is already in its own
view-slot frame.

Usage:
    ./run_gpu.sh <GPU> evaluation.card_game.attention_pick_alignment \\
        --checkpoint /path/to/saved_train_run \\
        --seed-idx 0 \\
        --num-episodes 1024
"""
from __future__ import annotations

import argparse
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from agents.ja_utils import build_card_masks
from envs.card_game.rendering import NUM_CARDS
from evaluation.card_game._card_game_utils import load_card_game_eval
from evaluation.vis_episodes import run_episode_with_states


IMG_H, IMG_W = 21, 35
FEAT_H, FEAT_W = 6, 9


def _head_average(raw):
    a = np.asarray(raw).squeeze()
    if a.ndim == 3:
        a = a.mean(axis=-1)
    return a


def _walk_perms(state):
    s = state
    while s is not None:
        if hasattr(s, "per_agent_perm"):
            return (
                np.asarray(s.per_agent_perm["agent_0"]),
                np.asarray(s.per_agent_perm["agent_1"]),
            )
        s = getattr(s, "env_state", None)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=1024)
    parser.add_argument("--rng-base", type=int, default=20260515)
    parser.add_argument("--sampled", action="store_true",
                        help="Sampled actions (default greedy).")
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    greedy = not args.sampled

    card_masks = np.asarray(build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W))  # (5, fh, fw)
    scalar_dim = int(getattr(ev.policy.network, "scalar_dim", 0))
    partner_feed_dim = scalar_dim if scalar_dim > 0 else 5
    use_card_masks = jnp.asarray(card_masks) if scalar_dim > 0 else None

    params = jax.tree.map(lambda x: x[args.seed_idx], ev.params)
    T = ev.max_steps

    print(f"checkpoint: {args.checkpoint}")
    print(f"label={ev.label}  seed_idx={args.seed_idx}  greedy={greedy}")
    print(f"running {args.num_episodes} episodes at decision step t={T-1} ...")

    # Counters
    own_attn_eq_pick = [0, 0]            # per agent
    attn_align_in_own_view = [0, 0]      # per agent (joint-attention agreement)
    coordinated_picks = 0                # both agents picked same canonical card
    # Diagnostic: distribution of own-pick view-slots (catch constant-action collapse)
    pick_hist = [Counter(), Counter()]

    for ep in range(args.num_episodes):
        rng = jax.random.PRNGKey(args.rng_base + args.seed_idx * 1000 + ep)
        ep_states, attn_data, ep_actions, _, _ = run_episode_with_states(
            rng, ev.env, params, ev.policy, params, ev.policy, T,
            collect_attention=True, greedy=greedy,
            ja_card_masks=use_card_masks,
            partner_feed_dim=partner_feed_dim,
        )
        # Decision-step attention maps (head-averaged)
        a0 = _head_average(attn_data["agent_0"][T - 1])      # (fh, fw)
        a1 = _head_average(attn_data["agent_1"][T - 1])
        # Per-card pooled mass in each agent's own view
        m0 = np.einsum("hw,chw->c", a0, card_masks)          # (5,) agent-0 view
        m1 = np.einsum("hw,chw->c", a1, card_masks)
        # Position permutations from the state at decision time
        perm_0, perm_1 = _walk_perms(ep_states[T - 1])
        if perm_0 is None or perm_1 is None:
            continue
        # Scatter to canonical frame: phys[perm[k]] = m[k]
        phys_0 = np.zeros(NUM_CARDS); phys_0[perm_0] = m0
        phys_1 = np.zeros(NUM_CARDS); phys_1[perm_1] = m1
        # Partner attention translated into ego view-slot frame
        partner_in_0 = phys_1[perm_0]    # agent 0's view of partner's attention
        partner_in_1 = phys_0[perm_1]
        # Argmaxes — each in its own agent's view-slot frame
        own_argmax_0 = int(np.argmax(m0))
        own_argmax_1 = int(np.argmax(m1))
        partner_argmax_in_0 = int(np.argmax(partner_in_0))
        partner_argmax_in_1 = int(np.argmax(partner_in_1))
        # Picks (raw actions at decision step; raw IS the agent's view-slot)
        pick_0 = int(ep_actions[T - 1][0])
        pick_1 = int(ep_actions[T - 1][1])

        if 0 <= pick_0 < NUM_CARDS:
            pick_hist[0][pick_0] += 1
            if own_argmax_0 == pick_0:
                own_attn_eq_pick[0] += 1
            if own_argmax_0 == partner_argmax_in_0:
                attn_align_in_own_view[0] += 1
        if 0 <= pick_1 < NUM_CARDS:
            pick_hist[1][pick_1] += 1
            if own_argmax_1 == pick_1:
                own_attn_eq_pick[1] += 1
            if own_argmax_1 == partner_argmax_in_1:
                attn_align_in_own_view[1] += 1

        # Coordinated picks: same CANONICAL card. Translate each pick to canonical.
        # raw action = view-slot, perm[view_slot] = canonical position.
        if 0 <= pick_0 < NUM_CARDS and 0 <= pick_1 < NUM_CARDS:
            if perm_0[pick_0] == perm_1[pick_1]:
                coordinated_picks += 1

    n = args.num_episodes
    chance = 1.0 / NUM_CARDS
    print(f"\nResults (N = {n}):\n")
    print(f"  P(own attention argmax  ==  own pick)")
    print(f"    agent 0: {own_attn_eq_pick[0]/n:.3f}   agent 1: {own_attn_eq_pick[1]/n:.3f}    (chance {chance:.2f})")
    print(f"  P(own attention argmax  ==  partner attention argmax in own view)")
    print(f"    agent 0: {attn_align_in_own_view[0]/n:.3f}   agent 1: {attn_align_in_own_view[1]/n:.3f}    (chance {chance:.2f})")
    print(f"\n  P(coordinated pick — both picked same canonical card) = {coordinated_picks/n:.3f}    (chance {chance:.2f})")
    print(f"\n  Pick distribution (agent 0 view-slots): {dict(pick_hist[0])}")
    print(f"  Pick distribution (agent 1 view-slots): {dict(pick_hist[1])}")


if __name__ == "__main__":
    main()
