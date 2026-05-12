"""Supervised sanity check: can a tiny MLP map a 20-dim per-head partner feed
to the 'right' card?

Setup mirrors what agent_0 actually sees:
  Per example, sample a random target canonical card and a random per-agent
  position permutation. Build the 20-vector exactly as marl/ja_ippo.py does:
  one-hot mass at (target_view_slot, head k) for all 4 heads, optionally
  perturbed with Gaussian noise to make it less trivial. Label is the target
  view-slot.

Trains a small MLP for a few hundred steps and reports per-step accuracy.

If accuracy reaches ~100%: the input has enough info to identify the right
card, end-to-end. So the RL policy's failure to use this signal is a training-
dynamics problem (gradient too small, advantage too noisy, exploration too
weak), NOT a representation/capacity problem.

If accuracy plateaus low: the input itself is corrupted somehow.
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import numpy as np


NUM_CARDS = 5
NUM_HEADS = 4
FEED_DIM = NUM_CARDS * NUM_HEADS  # 20


class MLP(nn.Module):
    hidden: int = 64
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(NUM_CARDS)(x)
        return x


def make_batch(rng, batch_size: int, noise_std: float):
    """Build (input_20, label_view_slot) for a batch of synthetic examples.

    Mirrors marl/ja_ippo.py's per-head partner feed: for each example pick a
    random target canonical card and a random per_agent_perm; the input
    receives mass 1.0 at the target view-slot for all 4 heads, zero elsewhere.
    Add Gaussian noise to simulate non-delta attention.
    """
    rng_t, rng_perm, rng_noise = jax.random.split(rng, 3)
    target_canon = jax.random.randint(rng_t, (batch_size,), 0, NUM_CARDS)
    # Random per_agent_perm for each example: a random permutation of 0..NUM_CARDS-1
    perms = jax.vmap(lambda k: jax.random.permutation(k, NUM_CARDS))(
        jax.random.split(rng_perm, batch_size)
    )
    # Target view-slot = the v for which perms[v] == target_canon.
    # Equivalently, argsort(perms)[target_canon].
    pos_perm_inv = jax.vmap(jnp.argsort)(perms)
    target_view = jax.vmap(lambda inv, t: inv[t])(pos_perm_inv, target_canon)
    # Build per-card per-head feed: one-hot at target_view, repeated across heads.
    per_card_one_hot = jax.nn.one_hot(target_view, NUM_CARDS)  # (B, 5)
    feed = jnp.broadcast_to(per_card_one_hot[:, :, None], (batch_size, NUM_CARDS, NUM_HEADS))
    feed = feed.reshape(batch_size, FEED_DIM)
    # Optional noise to simulate non-perfect attention
    noise = jax.random.normal(rng_noise, feed.shape) * noise_std
    feed = feed + noise
    return feed, target_view


def loss_fn(params, model, x, y):
    logits = model.apply(params, x)
    one_hot = jax.nn.one_hot(y, NUM_CARDS)
    log_p = jax.nn.log_softmax(logits)
    loss = -jnp.mean(jnp.sum(one_hot * log_p, axis=-1))
    pred = logits.argmax(axis=-1)
    acc = jnp.mean(pred == y)
    return loss, acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise-std", type=float, default=0.1,
                   help="Gaussian noise std added to each input element.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = jax.random.PRNGKey(args.seed)
    model = MLP(hidden=64)
    init_rng, rng = jax.random.split(rng)
    dummy_x = jnp.zeros((args.batch_size, FEED_DIM))
    params = model.init(init_rng, dummy_x)

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    grad_fn = jax.jit(jax.value_and_grad(
        lambda p, x, y: loss_fn(p, model, x, y), has_aux=True,
    ))

    @jax.jit
    def step(params, opt_state, x, y):
        (loss, acc), grads = grad_fn(params, x, y)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, acc

    print(f"noise_std={args.noise_std}  batch={args.batch_size}  lr={args.lr}  steps={args.steps}")
    print(f"{'step':>6} {'loss':>8} {'acc':>6}")
    log_every = max(1, args.steps // 25)
    for s in range(args.steps):
        rng, b_rng = jax.random.split(rng)
        x, y = make_batch(b_rng, args.batch_size, args.noise_std)
        params, opt_state, loss, acc = step(params, opt_state, x, y)
        if s == 0 or (s + 1) % log_every == 0 or s == args.steps - 1:
            print(f"{s+1:>6} {float(loss):>8.4f} {float(acc):>6.3f}")

    # Final eval on a fresh batch
    rng, eval_rng = jax.random.split(rng)
    x, y = make_batch(eval_rng, args.batch_size * 4, args.noise_std)
    logits = model.apply(params, x)
    pred = logits.argmax(axis=-1)
    final_acc = float(jnp.mean(pred == y))
    print(f"\nFinal accuracy on fresh batch: {final_acc:.3f}")
    print(f"Random chance: {1/NUM_CARDS:.3f}")


if __name__ == "__main__":
    main()
