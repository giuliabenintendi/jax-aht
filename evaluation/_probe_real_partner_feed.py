"""Supervised probe on REAL partner feeds from a trained checkpoint.

Loads a checkpoint, rolls out N episodes, collects at every step:
  - the actual 20-vector partner feed agent_0 receives (per-head, view-slot frame),
  - what partner's argmax view-slot was at the previous step (in agent_0's view).

Trains a tiny MLP to predict the second from the first. If accuracy is high,
the real feeds carry recoverable information about partner attention. If low,
the feeds are degenerate and the chicken-and-egg argument holds.

Also reports raw stats on the feed values (mean, max, std, fraction of mass
that's near zero) so we can see whether the feed is essentially a noise
floor or has structure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

from agents.ja_utils import build_card_masks
from evaluation._card_game_utils import load_card_game_eval
from evaluation.vis_episodes import (
    run_episode_with_states, _get_card_game_position_perm,
)


NUM_CARDS = 5
IMG_H, IMG_W = 21, 35
FEAT_H, FEAT_W = 6, 9


class MLP(nn.Module):
    hidden: int = 64
    out_dim: int = NUM_CARDS

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        return nn.Dense(self.out_dim)(x)


def _per_head_card_attn(attn_map, card_masks):
    """attn_map: (1, 1, fh, fw, num_heads) -> (5, num_heads)"""
    a = np.asarray(attn_map).squeeze()
    if a.ndim == 2:
        a = a[..., None]
    return np.einsum("hwk,chw->ck", a, card_masks)


def _build_partner_feed_for_ego(
    ego_attn_per_card, partner_attn_per_card, perm_ego, perm_partner
):
    """Compute the per-head partner feed that the ego agent receives.

    Mirrors marl/ja_ippo.py per-head branch exactly:
      partner attn (view-slot frame) -> scatter via perm_partner to canonical
      -> read via perm_ego back into ego view-slot frame.
    Returns (5*num_heads,) c-order flatten.
    """
    nh = partner_attn_per_card.shape[-1]
    phys = np.zeros((5, nh), dtype=np.float32)
    phys[np.asarray(perm_partner)] = partner_attn_per_card
    ego = phys[np.asarray(perm_ego)]            # (5, num_heads)
    return ego.reshape(-1)                       # (5*num_heads,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed-idx", type=int, default=0)
    p.add_argument("--num-episodes", type=int, default=128)
    p.add_argument("--mlp-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    print(f"[load] {args.checkpoint}")
    ev = load_card_game_eval(args.checkpoint)
    print(f"  label={ev.label}  seeds={ev.num_seeds}")
    if args.seed_idx >= ev.num_seeds:
        raise SystemExit(f"seed_idx {args.seed_idx} >= {ev.num_seeds}")
    params = jax.tree.map(lambda x: x[args.seed_idx], ev.params)
    card_masks = np.asarray(build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W))

    policy_scalar_dim = int(getattr(ev.policy.network, "scalar_dim", 0))
    partner_feed_dim = policy_scalar_dim if policy_scalar_dim > 0 else 5
    use_card_masks = jnp.asarray(card_masks) if policy_scalar_dim > 0 else None
    print(f"  partner_feed_dim={partner_feed_dim}")

    # Collect (feed_for_agent_0_at_step_t, partner_argmax_view_at_step_t-1) pairs.
    # Convention: feed at step t reflects partner's attention at step t-1, so
    # the natural label is partner_argmax(t-1) in agent_0's view-slot frame.
    feeds = []
    labels = []
    feed_stats = []
    for ep in range(args.num_episodes):
        rng = jax.random.PRNGKey(2000 + args.seed_idx * 10000 + ep)
        ep_states, attn_maps, ep_actions, _ = run_episode_with_states(
            rng, ev.env, params, ev.policy, params, ev.policy,
            ev.max_steps, collect_attention=True, greedy=True,
            ja_card_masks=use_card_masks,
            partner_feed_dim=partner_feed_dim,
        )
        T = len(attn_maps["agent_0"])
        for t in range(1, T):   # t >= 1; step 0's feed is the all-zeros init
            perm_0 = np.asarray(_get_card_game_position_perm(ep_states[t], "agent_0"))
            perm_1 = np.asarray(_get_card_game_position_perm(ep_states[t], "agent_1"))
            # Partner's attention at the PREVIOUS step (t-1) is what the feed encodes.
            a1_prev = attn_maps["agent_1"][t - 1]
            partner_per_card = _per_head_card_attn(a1_prev, card_masks)  # (5, num_heads)
            # Rebuild the feed agent_0 sees at step t.
            feed = _build_partner_feed_for_ego(
                None, partner_per_card, perm_0, perm_1,
            )  # (5*num_heads,)
            feeds.append(feed)
            # Label: partner's argmax view-slot at t-1, translated to ego view-slot
            # via canonical. argmax over per-card head-mean.
            partner_card_avg = partner_per_card.mean(axis=-1)  # (5,)
            partner_argmax_in_partner_view = int(partner_card_avg.argmax())
            partner_argmax_canonical = int(perm_1[partner_argmax_in_partner_view])
            # ego view-slot of that canonical card
            label_view_slot = int(np.argsort(perm_0)[partner_argmax_canonical])
            labels.append(label_view_slot)
            feed_stats.append((feed.min(), feed.max(), feed.mean(), (feed > 0.05).sum()))

    feeds = np.stack(feeds, axis=0).astype(np.float32)
    labels = np.array(labels, dtype=np.int32)
    print(f"\n[data] N = {len(labels)} feed examples (shape {feeds.shape})")
    print(f"[stats] feed value stats:")
    mins = np.array([s[0] for s in feed_stats])
    maxs = np.array([s[1] for s in feed_stats])
    means = np.array([s[2] for s in feed_stats])
    above = np.array([s[3] for s in feed_stats])
    print(f"  min     range [{mins.min():.4f}, {mins.max():.4f}]")
    print(f"  max     range [{maxs.min():.4f}, {maxs.max():.4f}],  mean {maxs.mean():.4f}")
    print(f"  mean    range [{means.min():.4f}, {means.max():.4f}], mean {means.mean():.4f}")
    print(f"  # entries > 0.05  per example: mean {above.mean():.2f}/20  (median {np.median(above):.0f})")

    # Label distribution
    print(f"[labels] distribution:")
    counts = np.bincount(labels, minlength=NUM_CARDS)
    for k in range(NUM_CARDS):
        print(f"  label={k}: {counts[k]} ({100*counts[k]/len(labels):.1f}%)")
    print(f"  majority-class baseline accuracy: {counts.max()/len(labels):.3f}")

    # 80/20 split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(labels))
    n_train = int(0.8 * len(labels))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    x_train, y_train = feeds[train_idx], labels[train_idx]
    x_test, y_test = feeds[test_idx], labels[test_idx]
    print(f"\n[train] {len(train_idx)} train, {len(test_idx)} test")

    # Train tiny MLP
    model = MLP(hidden=64, out_dim=NUM_CARDS)
    rng_init = jax.random.PRNGKey(0)
    params_mlp = model.init(rng_init, jnp.zeros((1, feeds.shape[1])))
    opt = optax.adam(args.lr)
    opt_state = opt.init(params_mlp)

    def loss_fn(p, x, y):
        logits = model.apply(p, x)
        oh = jax.nn.one_hot(y, NUM_CARDS)
        loss = -jnp.mean(jnp.sum(oh * jax.nn.log_softmax(logits), axis=-1))
        acc = jnp.mean(logits.argmax(-1) == y)
        return loss, acc

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    @jax.jit
    def step(p, os, x, y):
        (loss, acc), g = grad_fn(p, x, y)
        upd, os = opt.update(g, os, p)
        p = optax.apply_updates(p, upd)
        return p, os, loss, acc

    batch_size = 256
    print(f"\nMLP training: {args.mlp_steps} steps, batch {batch_size}, lr {args.lr}")
    print(f"{'step':>6} {'train_loss':>11} {'train_acc':>10} {'test_acc':>10}")
    log_every = max(1, args.mlp_steps // 25)
    rng_t = jax.random.PRNGKey(1)
    for s in range(args.mlp_steps):
        rng_t, br = jax.random.split(rng_t)
        bi = jax.random.randint(br, (batch_size,), 0, n_train)
        xb = jnp.asarray(x_train[np.asarray(bi)])
        yb = jnp.asarray(y_train[np.asarray(bi)])
        params_mlp, opt_state, loss, acc = step(params_mlp, opt_state, xb, yb)
        if s == 0 or (s + 1) % log_every == 0 or s == args.mlp_steps - 1:
            # Test accuracy
            test_logits = model.apply(params_mlp, jnp.asarray(x_test))
            test_acc = float(jnp.mean(test_logits.argmax(-1) == jnp.asarray(y_test)))
            print(f"{s+1:>6} {float(loss):>11.4f} {float(acc):>10.3f} {test_acc:>10.3f}")

    # Final test
    test_logits = model.apply(params_mlp, jnp.asarray(x_test))
    final_acc = float(jnp.mean(test_logits.argmax(-1) == jnp.asarray(y_test)))
    print(f"\nFinal test accuracy: {final_acc:.3f}")
    print(f"Majority-class baseline: {counts.max()/len(labels):.3f}")
    print(f"Random chance: {1/NUM_CARDS:.3f}")


if __name__ == "__main__":
    main()
