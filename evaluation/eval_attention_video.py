"""Post-hoc attention-overlay videos for an OvercookedV2 checkpoint.

Reproduces the training-time `JSDMechanism.log_ckpt_video` attention branch on a
saved run: renders the god's-eye gameplay plus per-agent (Blues/Reds), combined
(jet), and additive red/blue attention overlays. Works on any run dir with a
saved checkpoint, independent of whether CKPT_VIDEO_ATTENTION was on during
training.

Usage (box, GPU only — JAX CPU inference is broken for ocv2):
  CUDA_VISIBLE_DEVICES=<gpu> uv run python -m evaluation.eval_attention_video <run_dir> [out_dir] [n_episodes]

With n_episodes > 1, rolls out that many episodes (PRNGKey(ep) each) into
out_dir/ep00, out_dir/ep01, ... emitting only the gameplay + additive red/blue
overlay, skipping the per-agent, combined and PO-explainer videos. With
n_episodes == 1 (default) it renders the full suite including the per-agent,
PO and egocentric-comparison videos.

Accepts either a hydra run dir (`.hydra/config.yaml`) or a submission-style
checkpoint dir (`config.yaml` beside `saved_train_run/`).
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
import numpy as np
from moviepy import ImageSequenceClip
from omegaconf import OmegaConf
from PIL import Image

from agents.initialize_agents import initialize_ja_image_agent
from agents.overcooked_v2.ja_overcooked_v2_attention import (
    TaskObject,
    egocentric_attention_to_world_tiles,
    overcooked_v2_object_ctx,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.render_registry import get_eval_frames
from evaluation.eval_first_insert import _base_state, build_feed_ctx
from evaluation.run_xp_seeds import _strip_other_play_kwargs
from evaluation.vis_episodes import make_attention_video, run_episode_with_states
from marl.ja_ippo import select_mechanism
from marl.ppo_core import configure_training_dims


_MAGENTA = np.array([255, 0, 255], dtype=np.uint8)


def _stamp_t(frame, step):
    """Pixel-art timestep, top-left on a dark plate, so a screenshot of any frame
    is self-identifying: frame index == timestep == row `t` of pot_attention.csv."""
    from envs.card_game.rendering import PIXEL_DIGITS, _stamp_label_np

    out = np.ascontiguousarray(frame)
    scale = max(2, out.shape[1] // 120)
    digits = [int(c) for c in str(int(step))]
    w = (4 * len(digits) + 1) * scale
    out[: 7 * scale, : w + scale] = (out[: 7 * scale, : w + scale] * 0.25).astype(np.uint8)
    _stamp_label_np(out, [np.asarray(PIXEL_DIGITS[d]) for d in digits],
                    y=1, x=1, color=(255, 255, 255), scale=scale)
    return out


def _box_tile(img, tx, ty, tsz):
    """Draw a magenta 1px border around grid tile (tx, ty) in-place."""
    x0, y0 = tx * tsz, ty * tsz
    img[y0, x0 : x0 + tsz] = _MAGENTA
    img[y0 + tsz - 1, x0 : x0 + tsz] = _MAGENTA
    img[y0 : y0 + tsz, x0] = _MAGENTA
    img[y0 : y0 + tsz, x0 + tsz - 1] = _MAGENTA
    return img


def _label(img, text):
    """Draw a black-boxed white caption in the top-left corner."""
    from PIL import ImageDraw

    pic = Image.fromarray(img)
    d = ImageDraw.Draw(pic)
    d.rectangle([0, 0, 7 * len(text) + 5, 13], fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 255))
    return np.asarray(pic)


def _redblue_overlay(frame, attn_r, attn_b, alpha=0.6):
    """Additive red/blue attention overlay on a GT frame.

    agent_0 attention -> red channel, agent_1 -> blue channel, so cells both
    agents attend to render magenta (joint attention made visible). Both maps are
    scaled by their shared max within the frame, preserving which agent is more
    peaked (unlike the per-agent min-max in `_overlay_attention`).
    """
    r = np.array(attn_r).squeeze().astype(np.float32)
    b = np.array(attn_b).squeeze().astype(np.float32)
    m = max(float(r.max()), float(b.max()))
    if m > 1e-8:
        r, b = r / m, b / m
    h_px, w_px = frame.shape[:2]

    def up(a):
        return np.array(
            Image.fromarray(a, mode="F").resize((w_px, h_px), resample=Image.NEAREST)
        )

    r, b = up(r), up(b)
    heat = np.zeros((h_px, w_px, 3), np.float32)
    heat[..., 0] = r
    heat[..., 2] = b
    a = (np.maximum(r, b) * alpha)[..., None]
    blended = (1 - a) * frame.astype(np.float32) + a * heat * 255
    return np.clip(blended, 0, 255).astype(np.uint8)


def _ocv2_video_ctx(conf, env):
    """Image/feature geometry plus task-object cells, or None off OvercookedV2.

    Derived from the env/config rather than the JA mechanism: OP and IPPO use a
    JSDMechanism that carries no geometry, so a mechanism-based ctx would leave
    their attention unprojected while MATE's is projected — not a fair pairing.
    """
    if conf.get("ENV_NAME") != "overcooked-v2":
        return None
    return overcooked_v2_object_ctx(conf, env)


def _pot_tiles(ctx):
    """(row, col) of every pot cell in the layout."""
    n_static = int(ctx["static_num_objects"])
    pos = np.asarray(ctx["object_pos"])[:n_static]
    cat = np.asarray(ctx["object_cat"])[:n_static]
    return pos[cat == int(TaskObject.POT)]


def _pot_attention_rows(attn_data, ep_states, ctx, ep_idx):
    """Per-timestep share of each agent's attention mass landing on the pot.

    `in_fov` marks whether the pot is inside that agent's +/-view_size window:
    an agent across the kitchen *cannot* attend the pot, so only in-FOV rows
    make "ignoring the pot" a meaningful statement.
    """
    pots = _pot_tiles(ctx)
    view = int(ctx["agent_view_size"] or 0)
    rows = []
    n = min(len(attn_data.get("agent_0", [])), len(attn_data.get("agent_1", [])), len(ep_states))
    for t in range(n):
        ag = _base_state(ep_states[t]).agents
        rec = {"ep": ep_idx, "t": t}
        for idx, name in ((0, "r"), (1, "b")):
            a = np.asarray(attn_data[f"agent_{idx}"][t]).squeeze().astype(np.float64)
            tot = float(a.sum())
            mass = float(sum(a[int(pr), int(pc)] for pr, pc in pots))
            ar, ac = int(ag.pos.y[idx]), int(ag.pos.x[idx])
            d = min(max(abs(ar - int(pr)), abs(ac - int(pc))) for pr, pc in pots)
            rec[f"{name}_pot"] = mass / tot if tot > 1e-8 else 0.0
            rec[f"{name}_row"], rec[f"{name}_col"] = ar, ac
            rec[f"{name}_pot_cheb"] = d
            rec[f"{name}_in_fov"] = int(d <= view)
        rows.append(rec)
    return rows


def _project_ego_attention(attn_data, ep_states, ctx, subdiv=1):
    """Map crop-local attention to world tiles; without this the crop corners
    would be drawn as global-kitchen corners. `subdiv>1` keeps sub-tile detail
    (finer heatmap instead of per-tile blocks) for figure/video rendering."""
    if ctx is None:
        return attn_data
    out = {"agent_0": [], "agent_1": []}
    n = min(len(attn_data.get("agent_0", [])), len(attn_data.get("agent_1", [])), len(ep_states))
    for t in range(n):
        a0, a1 = egocentric_attention_to_world_tiles(
            jnp.asarray(attn_data["agent_0"][t]).squeeze(),
            jnp.asarray(attn_data["agent_1"][t]).squeeze(),
            ep_states[t],
            ctx,
            subdiv=subdiv,
        )
        out["agent_0"].append(np.asarray(a0))
        out["agent_1"].append(np.asarray(a1))
    return out


def _make_redblue_video(frames, maps_r, maps_b, filename, fps=10):
    # attn[i] is computed from ep_states[i], which frames[i] renders, so the two
    # are paired index-for-index (a +1 offset would trail the agent by a tile).
    n = min(len(maps_r), len(maps_b), len(frames))
    out = []
    for i in range(n):
        out.append(_redblue_overlay(frames[i], maps_r[i], maps_b[i]))
    ImageSequenceClip(out, fps=fps).write_videofile(
        filename, fps=fps, codec="libx264", audio=False, preset="ultrafast"
    )
    print(f"[attn video] Saved {filename} ({len(out)} frames)")


def _find_ckpt(run_dir):
    direct = os.path.join(run_dir, "saved_train_run")
    if os.path.isdir(direct):
        return direct
    for cand in sorted(glob.glob(os.path.join(run_dir, "*"))):
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, "*METADATA*")):
            return cand
    raise FileNotFoundError(f"no orbax checkpoint under {run_dir}")


def render_episode(rng, env, conf, params, policy, feed_ctx, out_dir, full, video_ctx=None,
                   ep_idx=0, params_b=None, tile=0, stamp=False):
    """`params_b` set = cross-play: agent 0 from `params`, agent 1 from `params_b`."""
    feed_attn_dims, feed_mask_fn, feed_reframe_fn = feed_ctx
    ep_states, attn_data, _a, _m, ep_obs = run_episode_with_states(
        rng, env, params, policy, params if params_b is None else params_b, policy,
        int(conf["ENV_KWARGS"].get("max_steps", 400)),
        collect_attention=True, collect_obs=True,
        feed_other_attn_dims=feed_attn_dims, feed_mask_fn=feed_mask_fn,
        feed_reframe_fn=feed_reframe_fn,
    )
    attn_data = _project_ego_attention(attn_data, ep_states, video_ctx)
    pot_rows = _pot_attention_rows(attn_data, ep_states, video_ctx, ep_idx) if video_ctx else []
    os.makedirs(out_dir, exist_ok=True)
    if tile:
        # Render natively at `tile` (sprites are drawn at that size) rather than
        # upscaling the 32px board; chunked, since supersampling 400 frames OOMs.
        from envs.overcooked_v2.rendering import render_eval_frames
        frames = [np.asarray(f) for f in render_eval_frames(ep_states, tile_size=tile, chunk=8)]
    else:
        frames = get_eval_frames(conf["ENV_NAME"], env, ep_states)
    n_attn = len(attn_data.get("agent_0", []))
    frames = list(frames[:n_attn] if n_attn else frames)
    if stamp:
        frames = [_stamp_t(f, t) for t, f in enumerate(frames)]

    stem = os.path.join(out_dir, "eval")
    ImageSequenceClip(frames, fps=10).write_videofile(
        f"{stem}.mp4", fps=10, codec="libx264", audio=False, preset="ultrafast"
    )
    _make_redblue_video(
        frames, attn_data.get("agent_0", []), attn_data.get("agent_1", []),
        f"{stem}_redblue.mp4",
    )
    if not full:
        return pot_rows
    make_attention_video(frames, attn_data, filename=f"{stem}_attn.mp4", fps=10)

    # Partial observability: reshape each agent's flat obs back to the allocentric
    # masked RGB frame it actually fed the policy (out-of-view cells black, magenta
    # box on its own tile), shown beside the god's-eye state as a triptych
    # [full state | agent_0 sees | agent_1 sees]. The obs is rendered at a smaller
    # tile size than the god's-eye eval frames, so nearest-upsample it to match.
    hpx, wpx, _ = frames[0].shape
    obs_size = int(ep_obs[0]["agent_0"].size)
    # Ego obs is a square crop. collect_obs stores the RAW env obs (pre-feed),
    # so infer channels from the size: 3 unless only 4 fits a square.
    n_ch = 3 if round((obs_size / 3) ** 0.5) ** 2 * 3 == obs_size else 4
    obs_h = obs_w = int(round((obs_size / n_ch) ** 0.5))
    scale = max(1, hpx // obs_h)
    n_po = min(len(frames), len(ep_obs))
    po_frames = []
    for t in range(n_po):
        views = [frames[t]]
        for a in ("agent_0", "agent_1"):
            ob = (ep_obs[t][a].reshape(obs_h, obs_w, n_ch)[..., :3] * 255).astype(np.uint8)
            ob = np.repeat(np.repeat(ob, scale, axis=0), scale, axis=1)
            views.append(np.pad(ob, ((0, hpx - ob.shape[0]), (0, 0), (0, 0))))
        po_frames.append(np.concatenate(views, axis=1))
    ImageSequenceClip(po_frames, fps=10).write_videofile(
        f"{stem}_po.mp4", fps=10, codec="libx264", audio=False, preset="ultrafast"
    )

    # Observation-frame comparison for agent_0 on the SAME trajectory:
    #   [full state | current allocentric-masked | B: egocentric north-up
    #    | A: egocentric forward-up].
    # Only the frame changes. B recenters the world on the agent (a shift);
    # A additionally rotates it so the agent's facing is always up, so the
    # kitchen visibly spins as the agent turns (Carroll et al. 2019 rep, the
    # "flipping" view). B keeps one shared world orientation, A does not — the
    # crux for the joint-attention loss.
    tsz = hpx // env.grid_height
    fov = env.agent_view_size or 2
    win = 2 * fov + 1
    pad = fov * tsz
    fwd_k = (0, 2, 1, 3)  # dir(UP,DOWN,RIGHT,LEFT) -> #CCW quarter-turns for forward-up
    cmp_frames = []
    for t in range(n_po):
        ag = _base_state(ep_states[t]).agents
        x, y, d = int(ag.pos.x[0]), int(ag.pos.y[0]), int(ag.dir[0])
        boxed = _box_tile(frames[t].copy(), x, y, tsz)
        padded = np.pad(frames[t], ((pad, pad), (pad, pad), (0, 0)))
        crop = padded[y * tsz : y * tsz + win * tsz, x * tsz : x * tsz + win * tsz]
        crop_b = _box_tile(crop.copy(), fov, fov, tsz)
        crop_a = _box_tile(np.rot90(crop, fwd_k[d], axes=(0, 1)).copy(), fov, fov, tsz)
        masked = (ep_obs[t]["agent_0"].reshape(obs_h, obs_w, n_ch)[..., :3] * 255).astype(np.uint8)
        masked = np.repeat(np.repeat(masked, scale, axis=0), scale, axis=1)
        masked = np.pad(masked, ((0, hpx - masked.shape[0]), (0, 0), (0, 0)))
        cmp_frames.append(
            np.concatenate(
                [
                    _label(boxed, "full state"),
                    _label(masked, "current: allocentric masked"),
                    _label(crop_b, "B: egocentric north-up"),
                    _label(crop_a, "A: egocentric forward-up"),
                ],
                axis=1,
            )
        )
    ImageSequenceClip(cmp_frames, fps=10).write_videofile(
        f"{stem}_obs_compare.mp4", fps=10, codec="libx264", audio=False, preset="ultrafast"
    )
    return pot_rows


def main():
    partner = None
    if "--partner" in sys.argv:
        i = sys.argv.index("--partner")
        partner = os.path.abspath(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    tile = 0
    if "--tile" in sys.argv:
        i = sys.argv.index("--tile")
        tile = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
    stamp = "--stamp" in sys.argv
    argv = [a for a in sys.argv[1:] if a not in ("--no-op", "--stamp")]
    no_op = "--no-op" in sys.argv
    run_dir = os.path.abspath(argv[0])
    out_dir = os.path.abspath(argv[1]) if len(argv) > 1 else os.path.join(run_dir, "attn_videos")
    n_eps = int(argv[2]) if len(argv) > 2 else 1

    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(run_dir, "config.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    conf = dict(cfg["algorithm"])
    task = cfg["task"]
    conf["ENV_NAME"] = task["ENV_NAME"]
    conf["ENV_KWARGS"] = dict(task["ENV_KWARGS"])
    conf["ROLLOUT_LENGTH"] = task.get("ROLLOUT_LENGTH", conf.get("ROLLOUT_LENGTH", 256))

    # The reported SP/XP numbers are all --no-op (identity frame). With OP left on,
    # each agent sees a per-episode ingredient relabelling while the god's-eye frame
    # always draws TRUE colours, so a correct cook reads as the "wrong" one on screen.
    if no_op:
        removed = _strip_other_play_kwargs(conf["ENV_KWARGS"])
        print(f"--no-op: identity frame, removed OP kwargs {removed}")

    env = make_env(conf["ENV_NAME"], conf["ENV_KWARGS"])
    env_l = LogWrapper(env)
    mech = select_mechanism(conf, env_l)
    configure_training_dims(conf, env_l)
    conf["JA_ENTITY_FEED_DIM"] = mech.entity_feed_dim()
    policy, _ = initialize_ja_image_agent(conf, env_l, jax.random.PRNGKey(0))

    run_data = load_train_run(_find_ckpt(run_dir))
    key = "best_params" if "best_params" in run_data else "final_params"
    params = jax.tree.map(
        lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x, run_data[key]
    )  # stacked (num_seeds, ...) -> seed 0
    print(f"loaded {key} (seed 0) from {run_dir}")

    params_b = None
    if partner:
        pdata = load_train_run(_find_ckpt(partner))
        pkey = "best_params" if "best_params" in pdata else "final_params"
        params_b = jax.tree.map(
            lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x, pdata[pkey]
        )
        print(f"CROSS-PLAY: agent_0 <- {os.path.basename(run_dir)}, "
              f"agent_1 <- {os.path.basename(partner)} ({pkey})")

    # MATE checkpoints need the 4th feed channel wired exactly as in the XP
    # evals, or the policy reshape fails (4800 vs 6400).
    feed_ctx = build_feed_ctx(conf, env_l)
    video_ctx = _ocv2_video_ctx(conf, env_l)
    ego = bool(video_ctx and video_ctx["egocentric"])
    print(f"egocentric attention projection: {'ON' if ego else 'OFF (allocentric)'}")

    rows = []
    if n_eps == 1:
        rows += render_episode(jax.random.PRNGKey(42), env, conf, params, policy, feed_ctx,
                               out_dir, full=True, video_ctx=video_ctx,
                               params_b=params_b) or []
    else:
        for ep in range(n_eps):
            rows += render_episode(
                jax.random.PRNGKey(ep), env, conf, params, policy, feed_ctx,
                os.path.join(out_dir, f"ep{ep:02d}"), full=False, video_ctx=video_ctx,
                ep_idx=ep, params_b=params_b, tile=tile, stamp=stamp,
            ) or []
    if rows:
        csv_path = os.path.join(out_dir, "pot_attention.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {csv_path} ({len(rows)} rows)")

    print(f"wrote videos to {out_dir}:")
    for p in sorted(glob.glob(os.path.join(out_dir, "**", "*.mp4"), recursive=True)):
        print(f"  {os.path.relpath(p, out_dir)}")


if __name__ == "__main__":
    main()
