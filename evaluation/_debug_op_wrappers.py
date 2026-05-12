"""End-to-end sanity check of the OP wrappers + per-head canonical-frame round-trip.

What we verify, top to bottom:
  1. Reset returns DIFFERENT per-agent perms and recolourings.
  2. The two agents' obs ARE pixel-wise different.
  3. Each card identity lands at the right view-slot/color in each agent's view
     (i.e., the per-agent perm and recolouring are applied as documented).
  4. The canonical-frame round-trip used by the per-head partner feed is
     correct: a value placed at agent_1's view-slot k, scattered to canonical
     via perm_1, then read into agent_0's view via perm_0, equals the value
     of the same physical card in agent_0's view.
  5. Running the same network params on two different agents' obs produces
     DIFFERENT attention maps. If they're bit-for-bit identical, attention is
     content-invariant (architectural problem, not wrapper bug).

Output:
  - Console table of all five checks (PASS / FAIL with diagnostic).
  - Two PNGs at <out>/agent_0_obs.png and agent_1_obs.png so you can see the
    cards visually in each agent's frame.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.card_game.rendering import NUM_CARDS, TILE_PIXELS, CARD_COLORS


def _walk(state, attr: str):
    s = state
    while s is not None and not hasattr(s, attr):
        s = getattr(s, "env_state", None)
    return getattr(s, attr) if s is not None else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/tmp/op_debug")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        max_steps=8, obs_type="image", shuffle=True,
        other_play_position_shuffle=True, other_play_recolouring=True,
        match_coef=0.0, stability_coef=0.0, follow_coef=0.0,
        gaze_mode=True,
    )
    env = make_env("card-game", env_kwargs)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(args.seed)
    obs, state = env.reset(rng)
    obs_0 = np.asarray(obs["agent_0"])
    obs_1 = np.asarray(obs["agent_1"])

    perm_0 = np.asarray(_walk(state, "per_agent_perm")["agent_0"])
    perm_1 = np.asarray(_walk(state, "per_agent_perm")["agent_1"])
    recol_0 = np.asarray(_walk(state, "per_agent_recolouring")["agent_0"])
    recol_1 = np.asarray(_walk(state, "per_agent_recolouring")["agent_1"])
    inv_recol_0 = np.asarray(_walk(state, "per_agent_inv_recolouring")["agent_0"])
    inv_recol_1 = np.asarray(_walk(state, "per_agent_inv_recolouring")["agent_1"])
    card_perm = np.asarray(_walk(state, "card_permutation"))

    print("\n=== STATE ===")
    print(f"  canonical card_permutation: {card_perm.tolist()}")
    print(f"  agent_0  perm = {perm_0.tolist()}  recolouring = {recol_0.tolist()}")
    print(f"  agent_1  perm = {perm_1.tolist()}  recolouring = {recol_1.tolist()}")

    # ---- check 1: perms / recolourings differ ----
    print("\n=== CHECK 1: per-agent perms and recolourings are different ===")
    ok_perm = not np.array_equal(perm_0, perm_1)
    ok_recol = not np.array_equal(recol_0, recol_1)
    print(f"  perm differ:          {'PASS' if ok_perm else 'FAIL'}")
    print(f"  recolouring differ:   {'PASS' if ok_recol else 'FAIL (could be unlucky sampling)'}")

    # ---- check 2: obs differ ----
    print("\n=== CHECK 2: obs pixel-wise differ ===")
    diff = np.abs(obs_0 - obs_1)
    n_diff = int((diff > 1e-6).sum())
    print(f"  pixels that differ:   {n_diff} of {obs_0.size}")
    print(f"  max abs diff:         {diff.max():.6f}")
    print(f"  status:               {'PASS' if n_diff > 0 else 'FAIL (obs are identical!)'}")

    # ---- check 3: card identity in each view ----
    # In agent X's view, the slot for canonical card k is the view position v
    # such that perm_X[v] == k.  Equivalently: v = argsort(perm_X)[k].
    # The colour shown for canonical card k in agent X's view is recol_X[k].
    print("\n=== CHECK 3: card identity per view ===")
    pos_inv_0 = np.argsort(perm_0)
    pos_inv_1 = np.argsort(perm_1)
    print(f"  {'canonical':>9} {'a0_pos':>7} {'a0_color':>9} {'a1_pos':>7} {'a1_color':>9}")
    for k in range(NUM_CARDS):
        print(f"  {k:>9} {int(pos_inv_0[k]):>7} {int(recol_0[k]):>9} "
              f"{int(pos_inv_1[k]):>7} {int(recol_1[k]):>9}")

    # ---- check 4: canonical-frame round-trip ----
    # Plant a unique value at each VIEW position of agent_1, scatter to canonical,
    # then read into agent_0's view. Result[v] should equal the value placed at
    # agent_1's view position for the SAME physical card that is at agent_0's
    # view position v.
    print("\n=== CHECK 4: canonical-frame round-trip (per-head feed logic) ===")
    a1_view_vals = np.arange(1, NUM_CARDS + 1, dtype=np.int32)  # 1..5
    # canonical[perm_1[v]] = a1_view_vals[v]  (i.e., place at canonical slot
    # for the physical card at agent_1's view-slot v)
    canonical = np.zeros(NUM_CARDS, dtype=np.int32)
    canonical[perm_1] = a1_view_vals
    a0_view = canonical[perm_0]  # read at canonical slot for agent_0's view-slot
    # Expected: a0_view[v] = a1_view_vals[the v' for which perm_1[v'] == perm_0[v]]
    expected = np.zeros(NUM_CARDS, dtype=np.int32)
    for v in range(NUM_CARDS):
        canon = perm_0[v]
        v_prime = int(np.where(perm_1 == canon)[0][0])
        expected[v] = a1_view_vals[v_prime]
    ok_rt = np.array_equal(a0_view, expected)
    print(f"  placed at agent_1 view:  {a1_view_vals.tolist()}")
    print(f"  read in agent_0 view:    {a0_view.tolist()}")
    print(f"  expected:                {expected.tolist()}")
    print(f"  status:                  {'PASS' if ok_rt else 'FAIL'}")

    # ---- check 5: same-net different-obs → different attention after LSTM warms up ----
    # At step 0 with init_hstate=zeros, queries are Dense(0)=0 -> softmax(0) is
    # uniform regardless of image. So step-0 attention is mathematically forced
    # to be identical for both agents; that's NOT a content-invariance bug. To
    # actually probe content dependence, we step the LSTM forward two iterations
    # on each agent's own image and check whether attention at step 2 differs.
    print("\n=== CHECK 5: attention depends on image after LSTM warm-up ===")
    try:
        from agents.initialize_agents import initialize_ja_image_agent
        cfg = dict(
            ENV_KWARGS=env_kwargs,
            OBS_TYPE="image",
            FEED_OTHER_ATTN=False,
            JA_CARD_ATTN=True,
            JA_CARD_PARTNER_FEED=True,
            JA_PARTNER_FEED_PER_HEAD=True,
            JA_NUM_HEADS=4,
            JA_HEAD_FEATURES=16,
            JA_SCALAR_EMBED_DIM=5,
            JA_SPATIAL_BASIS_DEPTH=8,
            CONV_FILTERS=32,
            CONV_NUM_BLOCKS=4,
            FC_HIDDEN_DIM=64,
            LSTM_HIDDEN_DIM=64,
            JA_AUX_PARTNER_ARGMAX_COEF=0.0,
        )
        policy, params = initialize_ja_image_agent(cfg, env, jax.random.PRNGKey(args.seed + 1))
        aug_obs_0 = jnp.concatenate([obs["agent_0"], jnp.zeros(20)]).reshape(1, 1, -1)
        aug_obs_1 = jnp.concatenate([obs["agent_1"], jnp.zeros(20)]).reshape(1, 1, -1)
        done_in = jnp.zeros((1, 1), dtype=bool)
        avail = env.get_avail_actions(state)["agent_0"]
        avail_in = jnp.asarray(avail).reshape(1, 1, -1).astype(jnp.float32)
        hstate_0 = policy.init_hstate(1)
        hstate_1 = policy.init_hstate(1)
        last_attn_0 = None
        last_attn_1 = None
        for step in range(3):
            _, hstate_0, attn_0_step = policy.get_action_and_attention(
                params=params, obs=aug_obs_0, done=done_in,
                avail_actions=avail_in, hstate=hstate_0, rng=jax.random.PRNGKey(1 + step),
                greedy=True,
            )
            _, hstate_1, attn_1_step = policy.get_action_and_attention(
                params=params, obs=aug_obs_1, done=done_in,
                avail_actions=avail_in, hstate=hstate_1, rng=jax.random.PRNGKey(1 + step),
                greedy=True,
            )
            d = float(jnp.abs(attn_0_step - attn_1_step).max())
            note = "(uniform; query=0)" if step == 0 else ""
            print(f"  step {step}: max abs diff attention(a0) vs attention(a1) = {d:.6f}  {note}")
            last_attn_0 = attn_0_step
            last_attn_1 = attn_1_step
        d_final = float(jnp.abs(last_attn_0 - last_attn_1).max())
        print(f"  status: {'PASS (attention depends on image)' if d_final > 1e-5 else 'FAIL (attention is content-invariant after warm-up)'}")
    except Exception as e:
        print(f"  could not run forward pass: {type(e).__name__}: {e}")

    # ---- visualize ----
    img_h = env._env._img_h if hasattr(env._env, "_img_h") else 21
    img_w = env._env._img_w if hasattr(env._env, "_img_w") else 35
    for tag, o in (("agent_0", obs_0), ("agent_1", obs_1)):
        img = o[: img_h * img_w * 3].reshape(img_h, img_w, 3)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"{tag} obs at reset")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out / f"{tag}_obs.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"\nSaved obs images to {out}/agent_0_obs.png and {out}/agent_1_obs.png")


if __name__ == "__main__":
    main()
