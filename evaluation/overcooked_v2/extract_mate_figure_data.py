"""Dump the real per-timestep data behind the OvercookedV2 qualitative figure.

Runs one evaluation episode per checkpoint and writes an npz holding, for every
timestep: each agent's object-level attention distribution p_t^i(e), the FOV masks,
the agents' pose/inventory, and (optionally) rendered board frames for a window.
`plot_mate_figure.py` consumes the npz; keeping extraction separate means the figure
can be re-styled without re-running the policies.

Attention is projected from each agent's egocentric crop onto world tiles before any
pooling, so p_t^i(e) is expressed in world coordinates for both agents.

`--no-op` (default on) strips the Other-Play ingredient permutation, so the episode
runs in the identity frame and phi_A^i is the identity — p_t^i(e) is therefore already
in the unpermuted frame, matching every reported ocv2 number.

Object set: the layout's interactable static cells (pot, delivery, plate piles,
ingredient piles) PLUS the recipe indicators. Indicators are detected here explicitly
because MATE's aux deliberately excludes them (`detect_task_objects` docstring), so
attention landing on an indicator is emergent, never supervised.

GPU only -- JAX CPU inference is behaviourally broken for ocv2 policies.

    ./run_gpu.sh 2 evaluation.overcooked_v2.extract_mate_figure_data \
        --mate /scratch/.../MATE/s47 --op /scratch/.../OP/s47 \
        --episode 0 --frame-window 95:145 --out /scratch/.../mate_fig_data.npz
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_image_agent
from agents.overcooked_v2.ja_overcooked_v2_attention import (
    TaskObject,
    detect_task_objects,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.eval_attention_video import (
    _find_ckpt,
    _ocv2_video_ctx,
    _project_ego_attention,
)
from evaluation.eval_first_insert import _base_state, build_feed_ctx
from evaluation.run_xp_seeds import _strip_other_play_kwargs
from evaluation.vis_episodes import run_episode_with_states
from marl.ja_ippo import select_mechanism
from marl.ppo_core import configure_training_dims

_CAT_NAME = {
    int(TaskObject.POT): "pot",
    int(TaskObject.GOAL): "delivery",
    int(TaskObject.PLATE_PILE): "plate_pile",
    int(TaskObject.INGREDIENT_PILE): "ingredient_pile",
    int(TaskObject.RECIPE_INDICATOR): "recipe_indicator",
}


def _load(checkpoint: str, no_op: bool):
    cfg_path = os.path.join(checkpoint, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(checkpoint, "config.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    conf = dict(cfg["algorithm"])
    task = cfg["task"]
    conf["ENV_NAME"] = task["ENV_NAME"]
    conf["ENV_KWARGS"] = dict(task["ENV_KWARGS"])
    conf["ROLLOUT_LENGTH"] = task.get("ROLLOUT_LENGTH", conf.get("ROLLOUT_LENGTH", 256))
    if no_op:
        removed = _strip_other_play_kwargs(conf["ENV_KWARGS"])
        print(f"[extract] {os.path.basename(checkpoint)}: identity frame, removed {removed}")

    env = make_env(conf["ENV_NAME"], conf["ENV_KWARGS"])
    env_l = LogWrapper(env)
    mech = select_mechanism(conf, env_l)
    configure_training_dims(conf, env_l)
    conf["JA_ENTITY_FEED_DIM"] = mech.entity_feed_dim()
    policy, _ = initialize_ja_image_agent(conf, env_l, jax.random.PRNGKey(0))
    run_data = load_train_run(_find_ckpt(checkpoint))
    key = "best_params" if "best_params" in run_data else "final_params"
    params = jax.tree.map(
        lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x, run_data[key])
    print(f"[extract] loaded {key} from {checkpoint}")
    return conf, env, env_l, policy, params


def _figure_objects(conf, env_l):
    """Static task objects plus recipe indicators, with stable slot order."""
    from envs.base_env import get_inner_env

    raw = get_inner_env(env_l)
    pos, cat, _ing = detect_task_objects(
        raw.layout.static_objects, include_recipe_indicator=True)
    return np.asarray(pos, dtype=np.int32), np.asarray(cat, dtype=np.int32)


def _object_distribution(attn_map, obj_pos, subdiv):
    """p(e): attention mass per object tile, normalised over the object set.

    Normalising over objects (rather than the whole board) makes p a distribution on
    a support both agents share, so sum_e min(p1, p2) is a genuine histogram
    intersection in [0, 1]. The unnormalised object mass is returned too, since a
    frame where an agent puts almost nothing on any object should be visible as such.
    """
    a = np.asarray(attn_map).squeeze().astype(np.float64)
    s = subdiv
    mass = np.array([a[r * s:(r + 1) * s, c * s:(c + 1) * s].sum() for r, c in obj_pos])
    tot_board = float(a.sum())
    tot_obj = float(mass.sum())
    p = mass / tot_obj if tot_obj > 1e-12 else np.zeros_like(mass)
    return p, (tot_obj / tot_board if tot_board > 1e-12 else 0.0)


def _fov_mask(state, ctx):
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    view = int(ctx["agent_view_size"] or 0)
    ag = _base_state(state).agents
    rows, cols = np.arange(gh)[:, None], np.arange(gw)[None, :]
    masks = []
    for i in range(len(ag.pos.x)):
        ar, ac = int(ag.pos.y[i]), int(ag.pos.x[i])
        masks.append((np.abs(rows - ar) <= view) & (np.abs(cols - ac) <= view))
    return np.stack(masks)


def _run(checkpoint, episode, no_op):
    conf, env, env_l, policy, params = _load(checkpoint, no_op)
    ctx = _ocv2_video_ctx(conf, env_l)
    subdiv = 1
    if ctx and ctx.get("egocentric") and not ctx.get("rotate_obs", False):
        fh, afs = int(ctx["feat_h"]), int(ctx["agent_fov_size"])
        if afs > 0 and fh % afs == 0:
            subdiv = max(1, fh // afs)
    fa, fm, fr = build_feed_ctx(conf, env_l)
    max_steps = int(conf["ENV_KWARGS"].get("max_steps", 400))
    ep_states, attn, _a, _m = run_episode_with_states(
        jax.random.PRNGKey(episode), env, params, policy, params, policy, max_steps,
        collect_attention=True, feed_other_attn_dims=fa, feed_mask_fn=fm,
        feed_reframe_fn=fr,
    )
    attn = _project_ego_attention(attn, ep_states, ctx, subdiv=subdiv)
    print(f"[extract] {os.path.basename(checkpoint)}: {len(ep_states)} states, subdiv={subdiv}")
    return conf, env, env_l, ctx, subdiv, ep_states, attn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mate", required=True)
    p.add_argument("--op", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frame-window", default=None, metavar="T0:T1",
                   help="render board frames for this timestep range (inclusive)")
    p.add_argument("--tile", type=int, default=64)
    p.add_argument("--keep-op", action="store_true",
                   help="keep the Other-Play permutation (default strips it)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    no_op = not args.keep_op
    conf, env, env_l, ctx, subdiv, ep_states, attn = _run(args.mate, args.episode, no_op)
    obj_pos, obj_cat = _figure_objects(conf, env_l)
    names = [f"{_CAT_NAME.get(int(c), 'obj')}@{int(r)},{int(k)}"
             for (r, k), c in zip(obj_pos, obj_cat)]
    print(f"[extract] {len(obj_pos)} figure objects: {names}")

    n = min(len(attn["agent_0"]), len(attn["agent_1"]), len(ep_states))
    P = np.zeros((2, n, len(obj_pos)))
    onboard = np.zeros((2, n))
    fov = np.zeros((2, n, int(ctx["grid_h"]), int(ctx["grid_w"])), dtype=bool)
    apos = np.zeros((2, n, 2), dtype=np.int32)
    adir = np.zeros((2, n), dtype=np.int32)
    ainv = np.zeros((2, n), dtype=np.int64)
    for t in range(n):
        for i in (0, 1):
            P[i, t], onboard[i, t] = _object_distribution(
                attn[f"agent_{i}"][t], obj_pos, subdiv)
        fov[:, t] = _fov_mask(ep_states[t], ctx)
        raw = _base_state(ep_states[t]).agents
        for i in (0, 1):
            apos[i, t] = (int(raw.pos.y[i]), int(raw.pos.x[i]))
            adir[i, t] = int(raw.dir[i])
            ainv[i, t] = int(raw.inventory[i])

    align = np.minimum(P[0], P[1]).sum(axis=1)
    shared = (fov[0] & fov[1]).sum(axis=(1, 2))
    out = {
        "obj_pos": obj_pos, "obj_cat": obj_cat, "obj_names": np.array(names),
        "p_mate": P, "onboard_mate": onboard, "align_mate": align,
        "fov": fov, "shared_count": shared, "agent_pos": apos,
        "agent_dir": adir, "agent_inv": ainv, "episode": args.episode,
        "subdiv": subdiv, "grid_h": int(ctx["grid_h"]), "grid_w": int(ctx["grid_w"]),
        "view_size": int(ctx["agent_view_size"] or 0),
    }

    if args.op:
        _c2, _e2, _el2, ctx2, sub2, st2, at2 = _run(args.op, args.episode, no_op)
        n2 = min(len(at2["agent_0"]), len(at2["agent_1"]), len(st2), n)
        P2 = np.zeros((2, n2, len(obj_pos)))
        for t in range(n2):
            for i in (0, 1):
                P2[i, t], _ = _object_distribution(at2[f"agent_{i}"][t], obj_pos, sub2)
        out["p_op"] = P2
        out["align_op"] = np.minimum(P2[0], P2[1]).sum(axis=1)
        out["shared_count_op"] = np.array(
            [(_fov_mask(st2[t], ctx2)[0] & _fov_mask(st2[t], ctx2)[1]).sum()
             for t in range(n2)])

    if args.frame_window:
        from envs.overcooked_v2.rendering import render_eval_frames
        t0, t1 = (int(v) for v in args.frame_window.split(":"))
        ts = list(range(t0, min(t1 + 1, n)))
        frames = np.stack([np.asarray(f) for f in render_eval_frames(
            [ep_states[t] for t in ts], tile_size=args.tile, chunk=4)])
        out["frames"] = frames
        out["frame_ts"] = np.array(ts)
        # Symbolic grid too: a flat-colour redraw needs the raw cell codes, not the
        # rasterised board. grid[..., 0] = StaticObject, grid[..., 1] = DynamicObject.
        out["grids"] = np.stack([np.asarray(_base_state(ep_states[t]).grid) for t in ts])
        out["attn_a0"] = np.stack([np.asarray(attn["agent_0"][t]).squeeze() for t in ts])
        out["attn_a1"] = np.stack([np.asarray(attn["agent_1"][t]).squeeze() for t in ts])
        out["recipe"] = np.array([int(_base_state(ep_states[t]).recipe) for t in ts])
        print(f"[extract] rendered {len(ts)} frames at tile {args.tile}: {frames.shape}"
              f"  + symbolic grids {out['grids'].shape}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[extract] wrote {args.out}")

    ov = np.where(shared > 0)[0]
    print(f"\n[extract] episode {args.episode}: {n} steps, "
          f"overlap on {len(ov)} steps ({100 * len(ov) / n:.1f}%)")
    print(f"[extract] alignment MATE mean {align.mean():.3f} "
          f"(overlap {align[shared > 0].mean():.3f} / disjoint {align[shared == 0].mean():.3f})")
    if args.op:
        ao = out["align_op"]
        so = out["shared_count_op"]
        print(f"[extract] alignment OP   mean {ao.mean():.3f} "
              f"(overlap {ao[so > 0].mean():.3f} / disjoint {ao[so == 0].mean():.3f})")


if __name__ == "__main__":
    main()
