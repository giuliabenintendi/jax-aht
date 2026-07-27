"""Render cross-play best-checkpoint videos: agent 0 from run A, agent 1 from run B.

Companion to `render_best.py` (self-play): same config recompose, feed branch and
captions, but the two agents load params from DIFFERENT runs so convention
mismatches become watchable. Optional `--force-recipe {0,1}` pins the recipe
(0=onion, 1=broccoli) the way the XP matrix decomposition does, so the pairs that
score ~0 on broccoli can be inspected directly.

The feed branch mirrors training: full-map runs gate the partner's whole
attention map on partner visibility, while legacy egocentric runs also reframe
the partner map into the receiver crop.

Usage (box, CPU):
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run python -m evaluation.render_xp \
      <run_dir_a> <run_dir_b> <task> <algo> <out_dir> [n_eps] [--no-op] [--force-recipe N]
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
from moviepy import ImageSequenceClip

from agents.initialize_agents import initialize_ja_image_agent
from agents.overcooked_v2.ja_overcooked_v2_attention import (
    make_visibility_mask_fn,
    reframe_partner_attention_for_eval,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.base_env import get_inner_env
from envs.log_wrapper import LogWrapper
from envs.overcooked_v2.common import DynamicObject
from envs.render_registry import get_eval_frames
from evaluation.render_best import (
    _draw_recipe,
    _draw_score,
    _episode_return,
    _project_ocv2_ego_attention,
    _state_recipe,
    best_ckpt,
    build_conf,
)
from evaluation.vis_episodes import make_attention_video, run_episode_with_states
from marl.ja_ippo import select_mechanism
from marl.ppo_core import configure_training_dims


def _load_params(run_dir: str):
    ck = best_ckpt(run_dir)
    run_data = load_train_run(ck)
    # Chunked single-seed ckpts carry a leading seed dim of size 1 on every leaf.
    params = jax.tree.map(lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x, run_data)
    return params, ck


def render_xp(run_dir_a: str, run_dir_b: str, task: str, algo: str, out_dir: str,
              n_eps: int = 2, no_op: bool = False, force_recipe: int | None = None) -> None:
    conf = build_conf(task, algo)
    if no_op:
        ek = conf["ENV_KWARGS"]
        removed = [k for k in list(ek) if (k.startswith("other_play_") or k == "op_ingredient_permutations") and ek.pop(k, None) is not None]
        print(f"[render_xp] --no-op: stripped {removed}", flush=True)
    env = make_env(conf["ENV_NAME"], conf["ENV_KWARGS"])
    env_l = LogWrapper(env)
    mech = select_mechanism(conf, env_l)
    configure_training_dims(conf, env_l)
    conf["JA_ENTITY_FEED_DIM"] = mech.entity_feed_dim()
    policy, _ = initialize_ja_image_agent(conf, env_l, jax.random.PRNGKey(0))

    params_a, ck_a = _load_params(run_dir_a)
    params_b, ck_b = _load_params(run_dir_b)

    feed_dims = None
    feed_mask_fn = None
    feed_reframe_fn = None
    ocv2_video_ctx = None
    if getattr(mech, "feed_other_attn", False):
        feed_dims = (mech.img_h, mech.img_w, mech.feat_h, mech.feat_w)
        if getattr(mech, "visibility_gating", False):
            feed_mask_fn = make_visibility_mask_fn({
                "feat_h": mech.feat_h, "feat_w": mech.feat_w,
                "grid_h": mech.grid_h, "grid_w": mech.grid_w,
                "agent_view_size": mech.view_size,
                "egocentric": mech.egocentric,
            })
        if getattr(mech, "egocentric", False):
            feed_reframe_ctx = {
                "feat_h": mech.feat_h,
                "feat_w": mech.feat_w,
                "img_h": mech.img_h,
                "img_w": mech.img_w,
                "tile_size": mech.tile_size,
                "grid_h": mech.grid_h,
                "grid_w": mech.grid_w,
                "agent_view_size": mech.view_size,
                "agent_fov_size": mech.agent_fov_size,
                "rotate_obs": mech.rotate_obs,
                "egocentric": mech.egocentric,
                "feed_border_project": getattr(mech, "feed_border_project", False),
            }
            ocv2_video_ctx = feed_reframe_ctx

            def feed_reframe_fn(a0, a1, s0, s1):
                return reframe_partner_attention_for_eval(a0, a1, s0, s1, feed_reframe_ctx)

    inner_env = getattr(env, "_env", env)
    raw_env = get_inner_env(env)
    if force_recipe is not None:
        # Mirrors run_xp_seeds --force-recipe: possible_recipes is a layout field,
        # not an env kwarg, so it is mutated post-make_env.
        raw_env.possible_recipes = jnp.array([[int(force_recipe)] * 3], dtype=jnp.int32)
        print("[render_xp] --force-recipe: forced possible_recipes to", raw_env.possible_recipes.tolist(), flush=True)
    # Name by ingredient index, not list position: under --force-recipe the
    # possible_recipes list has a single entry.
    recipe_names = ["onion", "broccoli"]
    encs = {int(DynamicObject.get_recipe_encoding(r)): recipe_names[int(np.asarray(r)[0])]
            for r in raw_env.possible_recipes}
    max_steps = int(conf["ENV_KWARGS"].get("max_steps", 400))
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for e in range(n_eps):
        ep_states, attn_data, _a, _m, ep_infos = run_episode_with_states(
            jax.random.PRNGKey(42 + e), inner_env, params_a, policy, params_b, policy,
            max_steps, collect_attention=True, greedy=False,
            feed_other_attn_dims=feed_dims, feed_mask_fn=feed_mask_fn,
            feed_reframe_fn=feed_reframe_fn,
            collect_info=True,
        )
        score = _episode_return(ep_infos)
        steps = len(ep_infos)
        frames = get_eval_frames(conf["ENV_NAME"], inner_env, ep_states)
        n_attn = len(attn_data.get("agent_0", []))
        frames = list(frames[:n_attn] if n_attn else frames)
        recipes = [_state_recipe(s) for s in ep_states][:len(frames)]
        frames = [_draw_recipe(f, encs.get(recipes[t], "?")) for t, f in enumerate(frames)]
        frames = [_draw_score(f, score, steps) for f in frames]
        attn_for_video = _project_ocv2_ego_attention(attn_data, ep_states, ocv2_video_ctx)
        stem = os.path.join(out_dir, f"ep{e}")
        ImageSequenceClip(frames, fps=10).write_videofile(
            f"{stem}.mp4", fps=10, codec="libx264", audio=False, preset="ultrafast",
        )
        make_attention_video(frames, attn_for_video, filename=f"{stem}_attn.mp4", fps=10)
        rows.append({
            "episode": e,
            "return": score,
            "steps": steps,
            "video": f"ep{e}.mp4",
            "attention_video": f"ep{e}_attn_combined.mp4",
            "ckpt_agent0": os.path.basename(ck_a),
            "ckpt_agent1": os.path.basename(ck_b),
            "no_op": bool(no_op),
            "force_recipe": force_recipe if force_recipe is not None else "",
        })
        print(f"[render_xp] ep{e}: return={score:.1f} steps={steps}", flush=True)
    csv_path = os.path.join(out_dir, "scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"DONE {os.path.basename(run_dir_a)} x {os.path.basename(run_dir_b)} -> {out_dir} "
          f"({n_eps} eps, A={os.path.basename(ck_a)}, B={os.path.basename(ck_b)})", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    no_op = "--no-op" in args
    force_recipe = None
    if "--force-recipe" in args:
        i = args.index("--force-recipe")
        force_recipe = int(args[i + 1])
        del args[i:i + 2]
    args = [a for a in args if a != "--no-op"]
    rda, rdb, tk, al, od = args[0], args[1], args[2], args[3], args[4]
    neps = int(args[5]) if len(args) > 5 else 2
    render_xp(rda, rdb, tk, al, od, neps, no_op=no_op, force_recipe=force_recipe)
