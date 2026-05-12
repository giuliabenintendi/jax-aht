"""Reproduce the load_card_game_eval → apply pipeline and print actual shapes.

Goal: identify where scalar_embed kernel becomes (0, 5) between disk (20, 5)
and apply.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from evaluation._card_game_utils import load_card_game_eval


def walk_shapes(tree, prefix="", needle="scalar_embed"):
    if hasattr(tree, "keys"):
        for k, v in tree.items():
            walk_shapes(v, f"{prefix}/{k}", needle)
    else:
        shape = getattr(tree, "shape", None)
        if shape is not None and needle in prefix:
            print(f"  {prefix}: shape={shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--seed-idx", type=int, default=3)
    args = p.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    print(f"label={ev.label} num_seeds={ev.num_seeds}")
    print(f"policy.obs_dim = {ev.policy.obs_dim}")
    print(f"network.scalar_dim = {ev.policy.network.scalar_dim}")
    print(f"network.message_dim = {ev.policy.network.message_dim}")

    print("\n=== ev.params (best_per_seed, leading axis = num_seeds) ===")
    walk_shapes(ev.params)

    seed_params = jax.tree.map(lambda x: x[args.seed_idx], ev.params)
    print(f"\n=== seed_params (seed {args.seed_idx}) ===")
    walk_shapes(seed_params)

    print("\n=== attempting apply ===")
    obs_dict, state = ev.env.reset(jax.random.PRNGKey(0))
    obs = obs_dict["agent_0"]
    print(f"env obs.shape = {obs.shape}")
    hstate = ev.policy.init_hstate(1)
    done = jnp.zeros((1, 1), dtype=bool)
    avail = ev.env.get_avail_actions(state)["agent_0"]
    avail = jnp.asarray(avail).reshape(1, 1, -1).astype(jnp.float32)
    obs_flat = jnp.asarray(obs).reshape(1, 1, -1)
    print(f"reshaped obs_flat.shape = {obs_flat.shape}")

    try:
        action, hstate_new, attn = ev.policy.get_action_and_attention(
            seed_params, obs_flat, done, avail, hstate,
            jax.random.PRNGKey(1), greedy=True,
        )
        print(f"apply OK; attn.shape={attn.shape}")
    except Exception as e:
        print(f"apply FAILED: {type(e).__name__}")
        print(f"  msg: {e}")
        raise


if __name__ == "__main__":
    main()
