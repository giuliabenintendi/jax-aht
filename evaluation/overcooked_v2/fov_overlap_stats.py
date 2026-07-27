"""How often do the two agents' fields of view overlap, over many episodes?

Section 6.3 asserts that on demo_cook_simple "the agents' views overlap mainly around
the pot". That is a claim about how the trained agents actually move, not about the
layout alone, so it has to be measured per policy rather than derived from the map.

Reports, over `--episodes` rollouts:
  - the share of timesteps with a non-empty view intersection
  - the distribution of shared-tile counts
  - per-episode overlap fraction, mean +/- SEM across episodes (the number with an
    error bar, since episodes are the independent unit here, not timesteps)
  - P(pot inside the shared region | the views overlap), which is the actual
    "mainly around the pot" claim

Overlap depends only on agent positions, but positions depend on the policy, so the
rollout must reproduce the eval path exactly -- including the partner-attention feed,
without which the agents would receive the wrong observation and move differently.

GPU only. Usage:
    ./run_gpu.sh 2 evaluation.overcooked_v2.fov_overlap_stats \
        --checkpoint <ckpt> --episodes 1024 --out fov_stats.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import numpy as np

from agents.overcooked_v2.ja_overcooked_v2_attention import TaskObject
from evaluation.eval_attention_video import _ocv2_video_ctx
from evaluation.eval_first_insert import _base_state, build_feed_ctx
from evaluation.overcooked_v2.extract_mate_figure_data import _load
from evaluation.vis_episodes import run_episode_with_states


def _seed_stats(checkpoint, args, no_op):
    """Overlap statistics for one training seed."""
    conf, env, env_l, policy, params = _load(checkpoint, no_op)
    params_b = params
    if args.partner:
        from common.save_load_utils import load_train_run
        from evaluation.eval_attention_video import _find_ckpt
        pd = load_train_run(_find_ckpt(args.partner))
        key = "best_params" if "best_params" in pd else "final_params"
        params_b = jax.tree.map(
            lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x, pd[key])
        print(f"[fov] CROSS-PLAY agent_1 <- {os.path.basename(args.partner)}")

    ctx = _ocv2_video_ctx(conf, env_l)
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    view = int(ctx["agent_view_size"] or 0)
    n_static = int(ctx["static_num_objects"])
    pos = np.asarray(ctx["object_pos"])[:n_static]
    cat = np.asarray(ctx["object_cat"])[:n_static]
    pots = [(int(r), int(c)) for r, c in pos[cat == int(TaskObject.POT)]]
    fa, fm, fr = build_feed_ctx(conf, env_l)
    max_steps = int(conf["ENV_KWARGS"].get("max_steps", 400))
    print(f"[fov] grid {gh}x{gw} = {gh * gw} tiles, view radius {view} "
          f"({(2 * view + 1) ** 2} tiles per agent), pot at {pots}")

    rows, cols = np.arange(gh)[:, None], np.arange(gw)[None, :]
    hist = np.zeros(gh * gw + 1, dtype=np.int64)
    ep_frac, ep_pot_frac = [], []
    n_steps = n_overlap = n_pot_shared = 0
    t0 = time.time()
    tag = os.path.basename(checkpoint.rstrip("/"))

    for ep in range(args.episodes):
        ep_states, _attn, _a, _m = run_episode_with_states(
            jax.random.PRNGKey(ep), env, params, policy, params_b, policy, max_steps,
            collect_attention=True, feed_other_attn_dims=fa, feed_mask_fn=fm,
            feed_reframe_fn=fr,
        )
        n = len(ep_states)
        ov_here = pot_here = 0
        for t in range(n):
            ag = _base_state(ep_states[t]).agents
            m = np.ones((gh, gw), dtype=bool)
            for i in range(2):
                ar, ac = int(ag.pos.y[i]), int(ag.pos.x[i])
                m &= (np.abs(rows - ar) <= view) & (np.abs(cols - ac) <= view)
            k = int(m.sum())
            hist[k] += 1
            if k > 0:
                ov_here += 1
                if any(m[r, c] for r, c in pots):
                    pot_here += 1
        n_steps += n
        n_overlap += ov_here
        n_pot_shared += pot_here
        ep_frac.append(ov_here / n)
        ep_pot_frac.append(pot_here / ov_here if ov_here else np.nan)
        if (ep + 1) % args.report_every == 0:
            el = time.time() - t0
            print(f"[fov] {tag} {ep + 1}/{args.episodes} eps  {el:.0f}s  "
                  f"({el / (ep + 1):.2f}s/ep)  overlap so far "
                  f"{100 * n_overlap / n_steps:.1f}% of steps", flush=True)

    ks = np.arange(len(hist))
    tiles_total = int((hist * ks).sum())
    # Conditional on the views actually meeting: the average size of a non-empty
    # shared region, which is the quantity the caption quotes.
    mean_given = tiles_total / n_overlap if n_overlap else 0.0
    stats = dict(
        seed=tag, episodes=args.episodes, n_steps=n_steps, n_overlap=n_overlap,
        overlap_frac=n_overlap / n_steps,
        steps_per_ep=n_steps / args.episodes,
        mean_shared_all=tiles_total / n_steps,
        mean_shared_given_overlap=mean_given,
        max_shared=int(ks[hist > 0].max()),
        grid_tiles=gh * gw, view_tiles=(2 * view + 1) ** 2,
        hist=hist, ep_frac=np.array(ep_frac),
    )
    print(f"[fov] {tag}: overlap {100 * stats['overlap_frac']:.2f}% of steps | "
          f"mean|overlap {mean_given:.2f} tiles | max {stats['max_shared']} "
          f"| {time.time() - t0:.0f}s", flush=True)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True,
                   help="comma-separated checkpoint dirs, one per training seed")
    p.add_argument("--partner", default=None,
                   help="second checkpoint for cross-play (default: self-play)")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--keep-op", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--report-every", type=int, default=16)
    args = p.parse_args()

    no_op = not args.keep_op
    ckpts = [c for c in args.checkpoints.split(",") if c]
    per_seed = [_seed_stats(c, args, no_op) for c in ckpts]

    fr = np.array([s["overlap_frac"] for s in per_seed])
    mg = np.array([s["mean_shared_given_overlap"] for s in per_seed])
    spe = np.array([s["steps_per_ep"] for s in per_seed])
    mx = max(s["max_shared"] for s in per_seed)
    gt, vt = per_seed[0]["grid_tiles"], per_seed[0]["view_tiles"]

    def pm(x):
        return x.mean(), (x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)

    f_m, f_s = pm(fr)
    g_m, g_s = pm(mg)
    ep_len = spe.mean()

    print(f"\n[fov] ===== {len(ckpts)} seeds x {args.episodes} episodes "
          f"({'cross-play' if args.partner else 'self-play'}) =====")
    print(f"  episode length            : {ep_len:.0f} timesteps")
    print(f"  FOVs overlap on           : {f_m * ep_len:.0f} +/- {f_s * ep_len:.0f} "
          f"of {ep_len:.0f} timesteps   ({100 * f_m:.1f} +/- {100 * f_s:.1f} %)")
    print(f"  when overlapping, shared  : {g_m:.1f} +/- {g_s:.1f} tiles on average, "
          f"max {mx}")
    print(f"  kitchen / single view     : {gt} / {vt} tiles "
          f"(max overlap = {100 * mx / gt:.0f}% of kitchen, {100 * mx / vt:.0f}% of a view)")
    print("  per-seed overlap fraction :",
          ", ".join(f"{s['seed']} {100 * s['overlap_frac']:.1f}%" for s in per_seed))

    if args.out:
        np.savez(args.out,
                 seeds=np.array([s["seed"] for s in per_seed]),
                 overlap_frac=fr, mean_shared_given_overlap=mg, steps_per_ep=spe,
                 max_shared=np.array([s["max_shared"] for s in per_seed]),
                 hist=np.stack([s["hist"] for s in per_seed]),
                 ep_frac=np.stack([s["ep_frac"] for s in per_seed]),
                 episodes=args.episodes, grid_tiles=gt, view_tiles=vt)
        print(f"[fov] wrote {args.out}")


if __name__ == "__main__":
    main()
