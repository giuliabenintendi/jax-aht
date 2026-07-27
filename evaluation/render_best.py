"""Batch-render best-checkpoint self-play videos for the ocv2 8-12 seed sweep.

For each run's `checkpoints/<name>/` dir it picks the highest-return chunk
(`ckpt_NN_ret_XX`), reconstructs the Hydra config from the task + algorithm names,
and renders N episodes of god's-eye gameplay + attention overlay. Reproduces the
training `log_ckpt_video` feed branch so MATE's 4-channel (partner-attention) policy
gets the obs shape it was trained on. Uses sampled actions (greedy self-play under
parameter sharing deadlocks symmetrically and is not representative).

Usage (box, CPU):
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run python -m evaluation.render_best \
      <run_dir> <task> <algo> <out_dir> [n_eps]
"""
from __future__ import annotations

import glob
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import numpy as np
from hydra import compose, initialize_config_dir
from moviepy import ImageSequenceClip
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from agents.initialize_agents import initialize_ja_image_agent
from agents.overcooked_v2.ja_overcooked_v2_attention import (
    egocentric_attention_to_world_tiles,
    make_visibility_mask_fn,
    reframe_partner_attention_for_eval,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.base_env import get_inner_env
from envs.log_wrapper import LogWrapper
from envs.overcooked_v2.common import DynamicObject
from envs.render_registry import get_eval_frames
from evaluation.vis_episodes import make_attention_video, run_episode_with_states
from marl.ja_ippo import select_mechanism
from marl.ppo_core import configure_training_dims

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# demo_cook_simple: recipe 0=onion (yellow), 1=broccoli (dark green). Caption
# background is the ingredient colour so the recipe reads at a glance.
_RECIPE_STYLE = {
    "onion": ("ONION", (235, 210, 40), (0, 0, 0)),
    "broccoli": ("BROCCOLI", (40, 150, 40), (255, 255, 255)),
}


def _state_recipe(s) -> int:
    for _ in range(6):
        if hasattr(s, "recipe"):
            return int(np.asarray(s.recipe))
        s = getattr(s, "env_state", getattr(s, "_env_state", None))
        if s is None:
            break
    return -1


def _draw_recipe(img: np.ndarray, name: str) -> np.ndarray:
    label, bg, fg = _RECIPE_STYLE.get(name, (name.upper(), (128, 128, 128), (255, 255, 255)))
    txt = f"WANT: {label}"
    pic = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(pic)
    d.rectangle([0, 0, 7 * len(txt) + 6, 14], fill=bg)
    d.text((3, 3), txt, fill=fg)
    return np.asarray(pic)


def _episode_return(ep_infos) -> float:
    if not ep_infos:
        return float("nan")
    # base_return_so_far RESETS to 0 on the terminal step, so ep_infos[-1] reads 0.
    # Take the peak over the episode (the cumulative value just before reset).
    for key in ("base_return", "returned_episode_returns"):
        vals = [float(np.asarray(i[key]).mean()) for i in ep_infos if key in i]
        if vals:
            return max(vals)
    return float("nan")


def _draw_score(img: np.ndarray, score: float, steps: int) -> np.ndarray:
    txt = f"RETURN: {score:.1f}  STEPS: {steps}"
    pic = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(pic)
    x1 = pic.width - 6
    x0 = max(0, x1 - (7 * len(txt) + 6))
    d.rectangle([x0, 0, x1, 14], fill=(20, 20, 20))
    d.text((x0 + 3, 3), txt, fill=(255, 255, 255))
    return np.asarray(pic)


def _project_ocv2_ego_attention(attn_data, ep_states, ctx):
    if ctx is None or not ctx.get("egocentric", False):
        return attn_data
    out = {"agent_0": [], "agent_1": []}
    n = min(len(attn_data.get("agent_0", [])), len(attn_data.get("agent_1", [])), len(ep_states))
    for t in range(n):
        a0, a1 = egocentric_attention_to_world_tiles(
            jax.numpy.asarray(attn_data["agent_0"][t]).squeeze(),
            jax.numpy.asarray(attn_data["agent_1"][t]).squeeze(),
            ep_states[t],
            ctx,
        )
        out["agent_0"].append(np.asarray(a0))
        out["agent_1"].append(np.asarray(a1))
    return out


def build_conf(task: str, algo: str) -> dict:
    with initialize_config_dir(version_base=None, config_dir=os.path.join(REPO, "marl", "configs")):
        cfg = compose(config_name="base_config_ja_ippo", overrides=[f"task={task}", f"algorithm={algo}"])
    cfg = OmegaConf.to_container(cfg, resolve=True)
    conf = dict(cfg["algorithm"])
    tc = cfg["task"]
    conf["ENV_NAME"] = tc["ENV_NAME"]
    conf["ENV_KWARGS"] = dict(tc["ENV_KWARGS"])
    conf["ROLLOUT_LENGTH"] = tc.get("ROLLOUT_LENGTH", conf.get("ROLLOUT_LENGTH", 256))
    return conf


def best_ckpt(run_dir: str) -> str:
    cands = glob.glob(os.path.join(run_dir, "ckpt_*_ret_*"))
    if not cands:
        raise FileNotFoundError(f"no ckpt_*_ret_* under {run_dir}")

    def ret(d: str) -> float:
        m = re.search(r"_ret_(-?[0-9.]+)$", d)
        return float(m.group(1)) if m else -1e9

    return max(cands, key=ret)


def render(run_dir: str, task: str, algo: str, out_dir: str, n_eps: int = 2, no_op: bool = False,
           partner_run_dir: str | None = None) -> None:
    """`partner_run_dir` set = cross-play: agent 0 from run_dir, agent 1 from partner."""
    conf = build_conf(task, algo)
    if no_op:
        # Deployment frame: strip Other-Play relabeling so self-play agents share
        # one unpermuted world (matches the --no-op XP eval).
        ek = conf["ENV_KWARGS"]
        removed = [k for k in list(ek) if (k.startswith("other_play_") or k == "op_ingredient_permutations") and ek.pop(k, None) is not None]
        print(f"[render_best] --no-op: stripped {removed}", flush=True)
    env = make_env(conf["ENV_NAME"], conf["ENV_KWARGS"])
    env_l = LogWrapper(env)
    mech = select_mechanism(conf, env_l)
    configure_training_dims(conf, env_l)
    conf["JA_ENTITY_FEED_DIM"] = mech.entity_feed_dim()
    policy, _ = initialize_ja_image_agent(conf, env_l, jax.random.PRNGKey(0))

    ck = best_ckpt(run_dir)
    run_data = load_train_run(ck)
    # Chunked single-seed ckpts carry a leading seed dim of size 1 on every leaf.
    params = jax.tree.map(lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x, run_data)

    params_b = params
    if partner_run_dir is not None:
        pdata = load_train_run(best_ckpt(partner_run_dir))
        params_b = jax.tree.map(lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x, pdata)
        print(f"[render_best] CROSS-PLAY agent_1 <- {os.path.basename(partner_run_dir)}", flush=True)

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
    recipe_names = ["onion", "broccoli"]
    encs = {int(DynamicObject.get_recipe_encoding(r)): recipe_names[i]
            for i, r in enumerate(raw_env.possible_recipes)}
    max_steps = int(conf["ENV_KWARGS"].get("max_steps", 400))
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for e in range(n_eps):
        ep_states, attn_data, _a, _m, ep_infos = run_episode_with_states(
            jax.random.PRNGKey(42 + e), inner_env, params, policy, params_b, policy,
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
            "checkpoint": os.path.basename(ck),
            "no_op": bool(no_op),
        })
    csv_path = os.path.join(out_dir, "scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "episode", "return", "steps", "video", "attention_video", "checkpoint", "no_op",
        ])
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "scores.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"DONE {os.path.basename(run_dir)} -> {out_dir} ({n_eps} eps, best={os.path.basename(ck)}, scores={csv_path})", flush=True)


if __name__ == "__main__":
    partner = None
    if "--partner" in sys.argv:
        pi = sys.argv.index("--partner")
        partner = sys.argv[pi + 1]
        del sys.argv[pi:pi + 2]
    argv = [a for a in sys.argv[1:] if a != "--no-op"]
    no_op = "--no-op" in sys.argv
    rd, tk, al, od = argv[0], argv[1], argv[2], argv[3]
    neps = int(argv[4]) if len(argv) > 4 else 2
    render(rd, tk, al, od, neps, no_op=no_op, partner_run_dir=partner)
