"""Combined-attention filmstrip for an OvercookedV2 checkpoint.

Renders ONE single-row strip: each cell is the god's-eye kitchen with the TEAM's
combined spatial attention overlaid as blocky colormap squares. Agent 0's and
agent 1's maps are each projected from their egocentric crop onto world tiles
(raw crop-local maps would paint local corners as global corners), then SUMMED,
so cells both agents attend read brightest while a cell only one attends still
lights up. Styling helpers are shared with the LBF paper filmstrip.

A pixel-art timestep counter is stamped top-left of every cell and a vertical
colorbar sits beside the strip; `--vmax` is shared across all cells so brightness
is comparable frame-to-frame (printed when auto-derived). `--grid NCOLS` wraps the
same cells over several rows instead (`--stack` is the one-column case), which is
what keeps a multi-frame figure legible at one LaTeX column width.

Render `--no-op`: every reported ocv2 number is identity-frame, and leaving the
training-time OP relabelling on makes each agent see a permuted palette.

Usage (GPU only — JAX CPU inference is broken for ocv2):
    ./run_gpu.sh 5 evaluation.eval_ocv2_filmstrip \
        --checkpoint /scratch/benintendi/checkpoints_AAAI2027/overcookedV2/MATE/s51 \
        --frames 4:380,4:385,4:390,4:395,4:399 --no-op \
        --out plots/overcooked-v2/mate_filmstrip.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from agents.initialize_agents import initialize_ja_image_agent
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import PIXEL_DIGITS, _stamp_label_np
from envs.log_wrapper import LogWrapper
from envs.render_registry import get_eval_frames
from evaluation.eval_attention_video import (
    _find_ckpt,
    _ocv2_video_ctx,
    _project_ego_attention,
)
from evaluation.eval_first_insert import _base_state, build_feed_ctx
from evaluation.lbf.eval_filmstrips import (
    WHITE,
    _assemble_row,
    _overlay_blocky,
    _render_colorbar,
)
from evaluation.run_xp_seeds import _strip_other_play_kwargs
from evaluation.vis_episodes import run_episode_with_states
from marl.ja_ippo import select_mechanism
from marl.ppo_core import configure_training_dims


_PINK = np.array([255, 105, 180], dtype=np.float32)        # agent 0 (red agent)
_LIGHTBLUE = np.array([120, 200, 255], dtype=np.float32)   # agent 1 (blue agent)

# Fusion modes x palettes swept by --batch. Per-agent uses sequential palettes
# keyed to each agent's body colour (warm = red agent, cool = blue agent).
BATCH_CMAPS = ["coolwarm", "inferno", "magma", "viridis"]
PER_AGENT_PALETTES = [("Oranges", "Blues"), ("YlOrRd", "PuBu"), ("autumn", "winter")]


def _overlay_two_cmap(frame, a0, a1, vmax, cmap0="Oranges", cmap1="Blues",
                      alpha=0.9, gamma=0.6):
    """Per-agent additive glow, each agent drawn through its own colormap.

    Sequential palettes (not two flat colours) keep the within-agent intensity
    readable; both share `vmax` so the agents stay comparable. Glows ADD, so a
    cell both attend saturates bright rather than becoming a third hue.
    """
    from matplotlib import colormaps

    h_px, w_px = frame.shape[:2]

    def glow(a, name):
        n = np.clip(a / max(vmax, 1e-8), 0.0, 1.0)
        full = np.asarray(Image.fromarray(n.astype(np.float32)).resize(
            (w_px, h_px), resample=Image.NEAREST))
        heat = colormaps[name](full)[..., :3] * 255.0
        return (alpha * full ** gamma)[..., None] * heat

    out = frame.astype(np.float32) + glow(a0, cmap0) + glow(a1, cmap1)
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_isolines(frame, attn, vmax, cmap_name="coolwarm", n_levels=4, lo=0.15,
                      lw=None, fill=0.0, labels=False, halo=True, label_fs=None,
                      cmap_lo=0.0, cmap_hi=1.0, line_alpha=1.0):
    """Draw attention as contour rings over an untouched board.

    The blocky wash paints colour on top of the kitchen; isolines leave every sprite
    visible and only mark where the mass sits. The coarse (grid*subdiv) map is
    bicubic-upsampled to pixel resolution before contouring, so rings are smooth
    curves rather than staircases around cell corners. Levels are fractions of the
    SHARED vmax, so a given ring means the same value in every cell, and each ring
    takes its colour from the same colormap the colorbar shows.
    """
    import matplotlib
    matplotlib.use("Agg")
    from io import BytesIO

    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    h_px, w_px = frame.shape[:2]
    n = np.clip(attn / max(vmax, 1e-8), 0.0, 1.0)
    up = np.asarray(Image.fromarray(n.astype(np.float32)).resize(
        (w_px, h_px), resample=Image.BICUBIC))
    levels = np.linspace(lo, 0.95, n_levels)
    if float(up.max()) <= levels[0]:
        return frame  # nothing above the lowest ring; contour() would raise
    cmap = colormaps[cmap_name]
    # Map through [cmap_lo, cmap_hi] rather than indexing the colormap raw: palettes
    # that start near black (inferno, magma) otherwise draw their lowest rings in a
    # colour indistinguishable from the board.
    colors = [cmap(cmap_lo + float(v) * (cmap_hi - cmap_lo)) for v in levels]
    lw = lw if lw else max(2.0, w_px / 500.0)

    dpi = 100
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(0, w_px)
    ax.set_ylim(h_px, 0)
    if fill > 0:
        ax.contourf(up, levels=list(levels) + [1.0], colors=colors, alpha=fill,
                    extent=(0, w_px, h_px, 0))
    cs = ax.contour(up, levels=levels, colors=colors, linewidths=lw,
                    extent=(0, w_px, h_px, 0))
    if halo:
        # A dark halo keeps a ring readable where it crosses a bright sprite or the
        # grey counters; without it the pale mid-colormap rings vanish there.
        fx = [pe.withStroke(linewidth=lw + 2.0, foreground="black", alpha=0.55)]
        try:
            cs.set_path_effects(fx)
        except AttributeError:
            for coll in cs.collections:
                coll.set_path_effects(fx)
    if labels:
        # Inline numeric labels break each line where the number sits, which is what
        # makes a full-board contour plot readable rather than a tangle of curves.
        fs = label_fs if label_fs else max(9.0, w_px / 110.0)
        txt = ax.clabel(cs, levels, inline=True, fmt="%.2f", fontsize=fs)
        if halo:
            for tt in txt:
                tt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black",
                                                   alpha=0.6)])
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=True, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    lines = np.asarray(Image.open(buf).convert("RGBA").resize((w_px, h_px),
                                                              resample=Image.LANCZOS))
    a = lines[..., 3:4].astype(np.float32) / 255.0 * float(line_alpha)
    out = frame.astype(np.float32) * (1 - a) + lines[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _stamp_step(cell: np.ndarray, step: int, color=WHITE) -> None:
    """Pixel-art timestep counter, top-left. Unlike the LBF strip's 2-digit
    stamp, ocv2 episodes run to 400 steps, so all digits are rendered. A dark
    plate is drawn behind it: the top wall is a counter agents park dishes on,
    and a bare stamp collides with them."""
    scale = max(2, cell.shape[1] // 90)
    digits = [int(c) for c in str(int(step))]
    w = (4 * len(digits) + 1) * scale
    cell[: 7 * scale, : w + scale] = (cell[: 7 * scale, : w + scale] * 0.25).astype(np.uint8)
    _stamp_label_np(cell, [np.asarray(PIXEL_DIGITS[d]) for d in digits],
                    y=1, x=1, color=color, scale=scale)


def _overlay_two_palette(frame: np.ndarray, a0: np.ndarray, a1: np.ndarray,
                         vmax: float, alpha: float = 0.9,
                         gamma: float = 0.6) -> np.ndarray:
    """Per-agent additive glow: agent 0 pink, agent 1 light blue.

    Both maps share `vmax`, so which agent is more peaked is preserved. The glows
    ADD, so a cell both agents attend saturates toward white — joint attention
    reads as brightness rather than a third hue nobody can name.
    """
    h_px, w_px = frame.shape[:2]

    def glow(a, colour):
        n = np.clip(a / max(vmax, 1e-8), 0.0, 1.0)
        full = np.asarray(Image.fromarray(n.astype(np.float32)).resize(
            (w_px, h_px), resample=Image.NEAREST))
        return (alpha * full ** gamma)[..., None] * colour[None, None, :]

    out = frame.astype(np.float32) + glow(a0, _PINK) + glow(a1, _LIGHTBLUE)
    return np.clip(out, 0, 255).astype(np.uint8)


def _parse_frames(spec: str) -> list[tuple[int, int]]:
    out = []
    for tok in spec.split(","):
        ep, t = tok.strip().split(":")
        out.append((int(ep), int(t)))
    return out


def _shadow_unseen(frame, state, ctx, alpha, style="lift"):
    """Dim world tiles outside BOTH agents' view windows (fog of war).

    Each agent sees a (2*view+1)^2 Chebyshev box around itself; everything
    outside the union is dimmed toward black. `style="lift"` adds a small
    blue-grey tint so unseen floor reads darker-but-different from the seen
    black floor (use on the bare board); `style="black"` is a pure
    semi-transparent black rectangle (use AFTER the attention overlay, where the
    floor is colour so the dim shows and does not clash with the palette).
    """
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    view = int(ctx["agent_view_size"] or 0)
    if view <= 0:
        return frame
    ag = _base_state(state).agents
    rows = np.arange(gh)[:, None]
    cols = np.arange(gw)[None, :]
    seen = np.zeros((gh, gw), dtype=bool)
    for i in range(len(ag.pos.x)):
        ar, ac = int(ag.pos.y[i]), int(ag.pos.x[i])
        seen |= (np.abs(rows - ar) <= view) & (np.abs(cols - ac) <= view)
    th, tw = frame.shape[0] // gh, frame.shape[1] // gw
    mask = np.kron(~seen, np.ones((th, tw), dtype=bool))
    out = frame.astype(np.float32)
    h = min(mask.shape[0], out.shape[0])
    w = min(mask.shape[1], out.shape[1])
    m = mask[:h, :w]
    lift = np.array([14.0, 14.0, 22.0], dtype=np.float32) if style == "lift" else 0.0
    out[:h, :w][m] = out[:h, :w][m] * (1.0 - alpha) + lift
    return np.clip(out, 0, 255).astype(np.uint8)


def _shadow_levels(frame, state, ctx, alpha_none, alpha_one):
    """3-level fog by how many agents can see each tile: seen by BOTH (the overlap /
    shared-visibility zone) stays full-bright, seen by ONE is dimmed by `alpha_one`,
    seen by NEITHER by `alpha_none` (darkest). Highlights the mutual-visibility region."""
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    view = int(ctx["agent_view_size"] or 0)
    if view <= 0:
        return frame
    ag = _base_state(state).agents
    rows = np.arange(gh)[:, None]
    cols = np.arange(gw)[None, :]
    count = np.zeros((gh, gw), dtype=np.int32)
    for i in range(len(ag.pos.x)):
        ar, ac = int(ag.pos.y[i]), int(ag.pos.x[i])
        count += ((np.abs(rows - ar) <= view) & (np.abs(cols - ac) <= view)).astype(np.int32)
    factor = np.where(count >= 2, 1.0,
                      np.where(count == 1, 1.0 - alpha_one, 1.0 - alpha_none))
    th, tw = frame.shape[0] // gh, frame.shape[1] // gw
    factor_px = np.kron(factor, np.ones((th, tw), dtype=np.float32))
    out = frame.astype(np.float32)
    h = min(factor_px.shape[0], out.shape[0])
    w = min(factor_px.shape[1], out.shape[1])
    out[:h, :w] *= factor_px[:h, :w, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _visibility_count(state, ctx):
    """Per-tile count of how many agents have the tile inside their view window."""
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    view = int(ctx["agent_view_size"] or 0)
    ag = _base_state(state).agents
    rows = np.arange(gh)[:, None]
    cols = np.arange(gw)[None, :]
    count = np.zeros((gh, gw), dtype=np.int32)
    for i in range(len(ag.pos.x)):
        ar, ac = int(ag.pos.y[i]), int(ag.pos.x[i])
        count += ((np.abs(rows - ar) <= view) & (np.abs(cols - ac) <= view)).astype(np.int32)
    return count


def _fov_counts(state, ctx):
    """Tiles seen by BOTH agents / exactly one / neither, over the same geometry the
    fog and the boxes draw. The both-count is the whole channel any signal has to pass
    through, so it is the number worth quoting beside a frame that shows it."""
    count = _visibility_count(state, ctx)
    return (int((count >= 2).sum()), int((count == 1).sum()), int((count == 0).sum()),
            count.size)


def _tint_shared(frame, state, ctx, alpha=0.22, colour=(255, 232, 130)):
    """Tint the tiles BOTH agents can see.

    Fog brightness alone cannot carry the shared zone: empty floor is near-black, so
    a dim-vs-bright difference there is invisible, and the one fact this figure exists
    to show disappears whenever the overlap lands off the grey counters.
    """
    mask = (_visibility_count(state, ctx) >= 2).astype(np.float32)
    if mask.sum() == 0:
        return frame
    gh, gw = mask.shape
    th, tw = frame.shape[0] // gh, frame.shape[1] // gw
    m = np.kron(mask, np.ones((th, tw), dtype=np.float32))[..., None] * alpha
    h, w = min(m.shape[0], frame.shape[0]), min(m.shape[1], frame.shape[1])
    out = frame.astype(np.float32)
    tint = np.array(colour, dtype=np.float32)
    out[:h, :w] = out[:h, :w] * (1 - m[:h, :w]) + tint * m[:h, :w]
    return np.clip(out, 0, 255).astype(np.uint8)


def _outline_shared(frame, state, ctx, colour=(255, 255, 255), thickness=None):
    """Outline the both-visible region instead of tinting it.

    A tint has to borrow a hue, and every warm hue is already spoken for by the
    attention colormap — a warm shared band reads as "attended". An outline marks the
    same tiles without competing for colour.
    """
    mask = _visibility_count(state, ctx) >= 2
    if not mask.any():
        return frame
    gh, gw = mask.shape
    th, tw = frame.shape[0] // gh, frame.shape[1] // gw
    t = thickness if thickness else max(3, th // 12)
    out = frame.copy()
    col = np.array(colour, dtype=np.uint8)
    for r, c in zip(*np.where(mask)):
        y0, y1 = r * th, min((r + 1) * th, out.shape[0])
        x0, x1 = c * tw, min((c + 1) * tw, out.shape[1])
        if r == 0 or not mask[r - 1, c]:
            out[y0:y0 + t, x0:x1] = col
        if r == gh - 1 or not mask[r + 1, c]:
            out[y1 - t:y1, x0:x1] = col
        if c == 0 or not mask[r, c - 1]:
            out[y0:y1, x0:x0 + t] = col
        if c == gw - 1 or not mask[r, c + 1]:
            out[y0:y1, x1 - t:x1] = col
    return out


_FOV_TINT_A0 = np.array([232, 150, 130], dtype=np.float32)   # warm, agent 0
_FOV_TINT_A1 = np.array([168, 174, 226], dtype=np.float32)   # cool, agent 1


def _tint_fov(frame, state, ctx, alpha=0.35):
    """Wash each agent's view window in its own colour, blending where they overlap.

    An alternative to `_draw_fov_box`: the boxes state the FOV boundary but leave the
    interior unmarked, so the shared region has to be inferred from where two outlines
    cross. Tinting makes each window a region and the intersection a third, blended
    colour that needs no explanation.
    """
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    view = int(ctx["agent_view_size"] or 0)
    if view <= 0:
        return frame
    ag = _base_state(state).agents
    rows, cols = np.arange(gh)[:, None], np.arange(gw)[None, :]
    th, tw = frame.shape[0] // gh, frame.shape[1] // gw
    out = frame.astype(np.float32)
    for i, colour in enumerate((_FOV_TINT_A0, _FOV_TINT_A1)):
        if i >= len(ag.pos.x):
            break
        ar, ac = int(ag.pos.y[i]), int(ag.pos.x[i])
        m = ((np.abs(rows - ar) <= view) & (np.abs(cols - ac) <= view)).astype(np.float32)
        mp = np.kron(m, np.ones((th, tw), dtype=np.float32))[..., None] * alpha
        h, w = min(mp.shape[0], out.shape[0]), min(mp.shape[1], out.shape[1])
        out[:h, :w] = out[:h, :w] * (1 - mp[:h, :w]) + colour * mp[:h, :w]
    return np.clip(out, 0, 255).astype(np.uint8)


def _hatch_fov(frame, state, ctx, alpha=0.55, tint=0.16, period_frac=0.42):
    """Hatch each agent's view window, the two agents at opposite diagonals.

    Colour alone is not print-safe: the warm and cool washes sit at nearly the same
    luminance, so a greyscale copy collapses them to one tone. Direction survives any
    conversion, and the intersection becomes cross-hatch -- the union of both patterns
    -- so "both agents see this" is legible with no colour at all. A light tint is
    kept underneath for colour readers.
    """
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    view = int(ctx["agent_view_size"] or 0)
    if view <= 0:
        return frame
    ag = _base_state(state).agents
    h_px, w_px = frame.shape[:2]
    th, tw = h_px // gh, w_px // gw
    rows, cols = np.arange(gh)[:, None], np.arange(gw)[None, :]
    yy, xx = np.mgrid[0:h_px, 0:w_px]
    period = max(6, int(th * period_frac))
    thick = max(2, period // 5)
    out = frame.astype(np.float32)
    for i, (colour, diag) in enumerate((
            (_FOV_TINT_A0, (xx + yy) % period), (_FOV_TINT_A1, (xx - yy) % period))):
        if i >= len(ag.pos.x):
            break
        ar, ac = int(ag.pos.y[i]), int(ag.pos.x[i])
        m = ((np.abs(rows - ar) <= view) & (np.abs(cols - ac) <= view)).astype(np.float32)
        mp = np.kron(m, np.ones((th, tw), dtype=np.float32))
        h, w = min(mp.shape[0], h_px), min(mp.shape[1], w_px)
        reg = np.zeros((h_px, w_px), dtype=np.float32)
        reg[:h, :w] = mp[:h, :w]
        out = out * (1 - (reg * tint)[..., None]) + colour * (reg * tint)[..., None]
        lines = (diag < thick).astype(np.float32) * reg * alpha
        out = out * (1 - lines[..., None]) + colour * lines[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


_FOV_BOX_A0 = np.array([255, 60, 60], dtype=np.uint8)     # agent 0 = red agent
_FOV_BOX_A1 = np.array([70, 130, 255], dtype=np.uint8)    # agent 1 = blue agent


def _draw_fov_box(frame, state, ctx, thickness=None):
    """Outline each agent's (2*view+1)^2 view window: red box for agent 0, blue for
    agent 1. Drawn last so the boundary stays crisp over attention/shadow."""
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    view = int(ctx["agent_view_size"] or 0)
    if view <= 0:
        return frame
    ag = _base_state(state).agents
    th, tw = frame.shape[0] // gh, frame.shape[1] // gw
    t = thickness if thickness else max(2, th // 16)
    out = frame.copy()
    for i, col in enumerate((_FOV_BOX_A0, _FOV_BOX_A1)):
        if i >= len(ag.pos.x):
            break
        ar, ac = int(ag.pos.y[i]), int(ag.pos.x[i])
        y0, y1 = max(0, ar - view) * th, min(gh, ar + view + 1) * th
        x0, x1 = max(0, ac - view) * tw, min(gw, ac + view + 1) * tw
        y1, x1 = min(y1, out.shape[0]), min(x1, out.shape[1])
        out[y0:y0 + t, x0:x1] = col
        out[y1 - t:y1, x0:x1] = col
        out[y0:y1, x0:x0 + t] = col
        out[y0:y1, x1 - t:x1] = col
    return out


def _add_black_margin(frame, ctx, view):
    """Wrap the board in a `view`-tile solid-black margin = the off-map cells the
    egocentric obs pads with black. A corner agent's whole (2*view+1)^2 window then
    fits inside the frame, with its off-grid part rendered as real black pad."""
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    th, tw = frame.shape[0] // gh, frame.shape[1] // gw
    mh, mw = view * th, view * tw
    canvas = np.zeros((frame.shape[0] + 2 * mh, frame.shape[1] + 2 * mw, 3), np.uint8)
    canvas[mh:mh + frame.shape[0], mw:mw + frame.shape[1]] = frame
    return canvas


def _draw_fov_box_margin(canvas, state, ctx, view, thickness=None):
    """FOV boxes on a margin-expanded canvas: world tile (ar,ac) sits at canvas tile
    (ar+view, ac+view), so the (2*view+1)^2 window is canvas tiles [ar, ar+2*view]."""
    gh, gw = int(ctx["grid_h"]), int(ctx["grid_w"])
    th, tw = canvas.shape[0] // (gh + 2 * view), canvas.shape[1] // (gw + 2 * view)
    ag = _base_state(state).agents
    t = thickness if thickness else max(2, th // 16)
    out = canvas.copy()
    for i, col in enumerate((_FOV_BOX_A0, _FOV_BOX_A1)):
        if i >= len(ag.pos.x):
            break
        ar, ac = int(ag.pos.y[i]), int(ag.pos.x[i])
        y0, y1 = ar * th, (ar + 2 * view + 1) * th
        x0, x1 = ac * tw, (ac + 2 * view + 1) * tw
        y1, x1 = min(y1, out.shape[0]), min(x1, out.shape[1])
        out[y0:y0 + t, x0:x1] = col
        out[y1 - t:y1, x0:x1] = col
        out[y0:y1, x0:x0 + t] = col
        out[y0:y1, x1 - t:x1] = col
    return out


def _compose_padmargin(cell, state, ctx, view, dim_alpha, fov_box):
    """Two-tier partial-obs composite: on-map tiles outside the FOV dimmed
    (transparent 'not visible'), then a solid-black off-map margin ('padding'),
    then red/blue FOV boxes. `cell` already has the attention overlaid."""
    cell = _shadow_unseen(cell, state, ctx, dim_alpha, style="black")
    cell = _add_black_margin(cell, ctx, view)
    if fov_box:
        cell = _draw_fov_box_margin(cell, state, ctx, view)
    return cell


def _mutual_series(attn, ep_states, ctx):
    """Per-timestep MUTUAL attention: the share of mass each agent puts on the
    PARTNER's own tile, and whether the partner is inside its +/-view_size window
    at all. Outside that window attending the partner is impossible rather than
    merely absent, so `vis` must gate any claim built on these numbers.

    Distinct from `_focus_series`: the pot is a shared OBJECT both can see, and it
    sits between the agents, so pot attention is not evidence of mutual attention.
    """
    view = int(ctx["agent_view_size"] or 0)
    rows = []
    n = min(len(attn["agent_0"]), len(attn["agent_1"]), len(ep_states))
    for t in range(n):
        ag = _base_state(ep_states[t]).agents
        p0 = (int(ag.pos.y[0]), int(ag.pos.x[0]))
        p1 = (int(ag.pos.y[1]), int(ag.pos.x[1]))
        d = max(abs(p0[0] - p1[0]), abs(p0[1] - p1[1]))
        a0 = np.asarray(attn["agent_0"][t]).squeeze()
        a1 = np.asarray(attn["agent_1"][t]).squeeze()
        s0, s1 = float(a0.sum()), float(a1.sum())
        rows.append({
            "t": t, "vis": int(d <= view), "cheb": d,
            "a0_on_a1": float(a0[p1]) / s0 if s0 > 1e-8 else 0.0,
            "a1_on_a0": float(a1[p0]) / s1 if s1 > 1e-8 else 0.0,
        })
    return rows


def _focus_series(attn, ep_states, ctx):
    """Per-timestep fraction of the team's attention mass landing on the pot or
    on either agent's own tile — i.e. how cleanly the frame reads as "attention
    on pot + agents only" rather than smeared over empty floor."""
    from evaluation.eval_attention_video import _pot_tiles

    gh = int(ctx["grid_h"])
    pots = [(int(r), int(c)) for r, c in _pot_tiles(ctx)]
    out = []
    n = min(len(attn["agent_0"]), len(attn["agent_1"]), len(ep_states))
    for t in range(n):
        a = (np.asarray(attn["agent_0"][t]).squeeze()
             + np.asarray(attn["agent_1"][t]).squeeze())
        # attention may be projected at sub-tile resolution (grid_h*s); sum the
        # s x s block per grid cell so pot/agent tiles are read at any resolution.
        s = max(1, a.shape[0] // gh)
        ag = _base_state(ep_states[t]).agents
        cells = set(pots)
        for i in (0, 1):
            cells.add((int(ag.pos.y[i]), int(ag.pos.x[i])))
        tot = float(a.sum())
        hit = float(sum(a[r * s:(r + 1) * s, c * s:(c + 1) * s].sum() for r, c in cells))
        out.append(hit / tot if tot > 1e-8 else 0.0)
    return out


def _recolour_cells(frame, attn, vmax, cmap_name="inferno", dark_thresh=60):
    """Recolour the board's dark cells by attention, Overcooked-challenge style.

    The additive glow lays colour ON the board and fights it; this REPLACES the
    near-black floor (and the pot's dark body) with the colormap, so the heat IS
    the cell's colour. Bright sprites (agents, piles, goal) and the grey walls sit
    above `dark_thresh` and survive untouched, which is what keeps the board
    readable — no alpha wash over everything.
    """
    from matplotlib import colormaps

    n = np.clip(attn / max(vmax, 1e-8), 0.0, 1.0)
    h_px, w_px = frame.shape[:2]
    full = np.asarray(Image.fromarray(n.astype(np.float32)).resize(
        (w_px, h_px), resample=Image.NEAREST))
    heat = colormaps[cmap_name](full)[..., :3] * 255.0
    dark = frame.max(axis=-1) < dark_thresh
    out = frame.astype(np.float32).copy()
    out[dark] = heat[dark]
    return np.clip(out, 0, 255).astype(np.uint8)


def _render_hbar(cmap_name, width_px, lo_label="Less attention",
                 hi_label="More attention", height_px=90):
    """Horizontal colorbar with word labels either side (reference-figure style)."""
    import matplotlib
    matplotlib.use("Agg")
    from io import BytesIO

    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import Normalize

    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi * 0.42, height_px / dpi), dpi=dpi)
    cb = ColorbarBase(ax, cmap=colormaps[cmap_name], norm=Normalize(0, 1),
                      orientation="horizontal", ticks=[])
    cb.outline.set_visible(False)
    fs = max(13, round(height_px * 0.30))
    ax.text(-0.04, 0.5, lo_label, ha="right", va="center", fontsize=fs,
            transform=ax.transAxes)
    ax.text(1.04, 0.5, hi_label, ha="left", va="center", fontsize=fs,
            transform=ax.transAxes)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def _render_numbered_hbar(cmap_name, lo, hi, width_px, height_px, tick_fs):
    """Horizontal colorbar with 0-1 number ticks, resized to exactly `width_px`.

    The colormap is truncated to [lo, hi] so the bar shows the same range the cells
    were painted through.
    """
    import matplotlib
    matplotlib.use("Agg")
    from io import BytesIO

    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    base = colormaps[cmap_name]
    aa = np.linspace(0, 1, 256)
    trunc = LinearSegmentedColormap.from_list("t", base(lo + aa * (hi - lo))[:, :3])
    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, max(1.0, height_px / dpi)), dpi=dpi)
    cb = ColorbarBase(ax, cmap=trunc, norm=Normalize(0, 1), orientation="horizontal",
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.outline.set_visible(False)
    ax.tick_params(labelsize=tick_fs)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    return np.asarray(im.resize((width_px, max(1, int(im.height * width_px / im.width)))))


def _render_minimal_cbar(cmap_name, lo, hi, height_px, width_frac=0.055,
                         labels=("1", "0")):
    """Slim vertical bar labelled only at its ends.

    Attention is normalised per figure, so intermediate tick values carry no absolute
    meaning -- the informative thing is where the mass sits. Ends-only labelling says
    "low to high" without implying the intermediate numbers are worth reading, and
    takes a fraction of the width a full numeric axis needs.
    """
    import matplotlib
    matplotlib.use("Agg")
    from io import BytesIO

    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    base = colormaps[cmap_name]
    aa = np.linspace(0, 1, 256)
    trunc = LinearSegmentedColormap.from_list("t", base(lo + aa * (hi - lo))[:, :3])
    dpi = 100
    w_px = max(24, int(height_px * width_frac))
    fs = max(15, round(height_px * 0.085))
    fig = plt.figure(figsize=(w_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes((0.0, 0.075, 1.0, 0.85))
    ax.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto", cmap=trunc,
              extent=(0, 1, 0, 1))
    ax.set_axis_off()
    ax.text(0.5, 1.012, labels[0], ha="center", va="bottom", fontsize=fs,
            transform=ax.transAxes)
    ax.text(0.5, -0.012, labels[1], ha="center", va="top", fontsize=fs,
            transform=ax.transAxes)
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    return np.asarray(im.resize((max(1, int(im.width * height_px / im.height)),
                                 height_px), resample=Image.LANCZOS))


def _render_minimal_hbar(cmap_name, lo, hi, width_px, height_px, labels=("0", "1")):
    """Horizontal counterpart of `_render_minimal_cbar`: a slim bar with its ends
    labelled, sitting under the frames instead of beside them."""
    import matplotlib
    matplotlib.use("Agg")
    from io import BytesIO

    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import LinearSegmentedColormap

    base = colormaps[cmap_name]
    aa = np.linspace(0, 1, 256)
    trunc = LinearSegmentedColormap.from_list("t", base(lo + aa * (hi - lo))[:, :3])
    dpi = 100
    fs = max(13, round(height_px * 0.62))
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    ax = fig.add_axes((0.06, 0.0, 0.88, 1.0))
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=trunc,
              extent=(0, 1, 0, 1))
    ax.set_axis_off()
    ax.text(-0.012, 0.5, labels[0], ha="right", va="center", fontsize=fs,
            transform=ax.transAxes)
    ax.text(1.012, 0.5, labels[1], ha="left", va="center", fontsize=fs,
            transform=ax.transAxes)
    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    return np.asarray(im.resize((width_px, max(1, int(im.height * width_px / im.width))),
                                resample=Image.LANCZOS))


def _assemble_grid(cells, ncols, hbar, out_path):
    """Lay cells out in `ncols` columns over as many rows as needed, legend below."""
    ch, cw = cells[0].shape[:2]
    pad = max(4, cw // 80)
    nrows = (len(cells) + ncols - 1) // ncols
    grid_w = ncols * cw + (ncols - 1) * pad
    grid_h = nrows * ch + (nrows - 1) * pad
    bar = None
    if hbar is not None:
        scale = grid_w / hbar.shape[1]
        bar = np.asarray(Image.fromarray(hbar).resize(
            (grid_w, max(1, int(hbar.shape[0] * scale))), resample=Image.LANCZOS))
    total_h = grid_h + (bar.shape[0] + pad * 3 if bar is not None else 0)
    out = np.full((total_h, grid_w, 3), 255, dtype=np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, ncols)
        y, x = r * (ch + pad), k * (cw + pad)
        out[y:y + ch, x:x + cw] = c
    if bar is not None:
        out[grid_h + pad * 3:, :] = bar
    Image.fromarray(out).save(out_path)
    print(f"[ocv2_filmstrip] saved {out_path}  ({grid_w}x{total_h} px, {nrows}x{ncols})")


def _best_window(focus, n_frames, stride):
    """Start of the highest mean-focus window spanning n_frames at `stride`."""
    span = (n_frames - 1) * stride + 1
    best, best_t = -1.0, 0
    for t0 in range(0, len(focus) - span):
        m = float(np.mean([focus[t0 + i * stride] for i in range(n_frames)]))
        if m > best:
            best, best_t = m, t0
    return best_t, best


def _strip(cells, cbar, out_path, stamp_steps):
    for cell, t in zip(cells, stamp_steps):
        _stamp_step(cell, t)
    _assemble_row(cells, len(cells), cbar, out_path, cbar_side="right")


def _load(checkpoint: str, no_op: bool, ckpt_dir: str | None = None):
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
        print(f"[ocv2_filmstrip] --no-op: identity frame, removed {removed}")

    env = make_env(conf["ENV_NAME"], conf["ENV_KWARGS"])
    env_l = LogWrapper(env)
    mech = select_mechanism(conf, env_l)
    configure_training_dims(conf, env_l)
    conf["JA_ENTITY_FEED_DIM"] = mech.entity_feed_dim()
    policy, _ = initialize_ja_image_agent(conf, env_l, jax.random.PRNGKey(0))
    run_data = load_train_run(ckpt_dir if ckpt_dir else _find_ckpt(checkpoint))
    if "best_params" in run_data or "final_params" in run_data:
        key = "best_params" if "best_params" in run_data else "final_params"
        run_data = run_data[key]
    else:
        # ckpt_*_ret_* saves store the raw params pytree (matches render_best)
        key = "raw"
    params = jax.tree.map(
        lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x, run_data)
    print(f"[ocv2_filmstrip] loaded {key} from {ckpt_dir if ckpt_dir else checkpoint}")
    return conf, env, env_l, policy, params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ckpt", default=None,
                   help="specific ckpt_*_ret_* dir to load raw params from instead of "
                        "saved_train_run best_params (use to replay render_best videos)")
    p.add_argument("--frames", default=None,
                   help="comma-separated ep:t, e.g. 4:380,4:385,4:390,4:395,4:399")
    p.add_argument("--scan", type=int, default=0, metavar="N_EPISODES",
                   help="roll out N episodes and print the windows where attention "
                        "sits most cleanly on pot+agents; renders nothing")
    p.add_argument("--scan-len", type=int, default=5)
    p.add_argument("--narrative", type=int, default=0, metavar="N_EPISODES",
                   help="scan N episodes for the release->collect->deliver story: an agent "
                        "holding an ingredient, an agent holding a plate at the pot, an "
                        "agent holding a cooked dish at the delivery. Prints candidate ep:t.")
    p.add_argument("--scan-mutual", type=int, default=0, metavar="N_EPISODES",
                   help="roll out N episodes and report MUTUAL attention (each agent's "
                        "mass on the PARTNER's tile), gated on partner visibility")
    p.add_argument("--batch", type=int, default=0, metavar="N_EPISODES",
                   help="one rollout per episode; auto-pick the best pot+agent window and "
                        "render every fusion mode x palette from it")
    p.add_argument("--n-frames", type=int, default=6)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--recolour", action="store_true",
                   help="recolour the board's dark cells by attention (2-row grid + "
                        "bottom legend) instead of the additive glow strip")
    p.add_argument("--cols", type=int, default=3, help="columns in the grid layout")
    p.add_argument("--muted", action="store_true",
                   help="LBF-paper look: additive glow, low --gamma, trimmed --cmap-hi, "
                        "2-row grid + bottom legend")
    p.add_argument("--muted-cmaps", default="coolwarm,inferno")
    p.add_argument("--muted-his", default="0.75,0.85,1.0")
    p.add_argument("--steps", default=None,
                   help="explicit comma-separated timesteps for the --window episode "
                        "(overrides --n-frames/--stride, e.g. 3,5,8,12,17,24,40)")
    p.add_argument("--window", default=None, metavar="EP:T0",
                   help="in --batch, render THIS episode at THIS start step instead of "
                        "auto-selecting the max-focus window (e.g. 3:18)")
    p.add_argument("--tile", type=int, default=96,
                   help="god's-eye tile size in px, rendered NATIVELY (sprites are drawn "
                        "at this size, so raise this instead of --scale for sharp edges)")
    p.add_argument("--crop", default=None, metavar="C0:C1",
                   help="keep only world columns [C0,C1) e.g. 2:11")
    p.add_argument("--partner", default=None,
                   help="checkpoint for agent_1 (cross-play); default = self-play")
    p.add_argument("--combine", default="sum",
                   choices=["sum", "intersection", "per-agent"],
                   help="sum: a0+a1 (union, both-attend brightest); intersection: "
                        "min(a0,a1) (only cells BOTH attend); per-agent: a0 pink, a1 light blue")
    p.add_argument("--out", default="plots/overcooked-v2/mate_filmstrip.png")
    p.add_argument("--no-op", action="store_true")
    p.add_argument("--vmax", type=float, default=None,
                   help="shared colour ceiling; default = max over selected cells")
    p.add_argument("--cmap", default="coolwarm")
    p.add_argument("--cmap-lo", type=float, default=0.0)
    p.add_argument("--cmap-hi", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--gamma", type=float, default=0.6)
    p.add_argument("--cbar-side", default="right", choices=["left", "right", "bottom", "none"])
    p.add_argument("--cbar-minimal", action="store_true",
                   help="slim colorbar labelled only 0 and 1 (attention is normalised, "
                        "so intermediate ticks carry no absolute meaning)")
    p.add_argument("--cbar-width", type=float, default=0.055,
                   help="minimal colorbar width as a fraction of cell height")
    p.add_argument("--scale", type=int, default=4, help="nearest-neighbour upscale")
    p.add_argument("--shadow", action="store_true",
                   help="dim world tiles outside both agents' view windows to "
                        "show partial observability (--batch path only)")
    p.add_argument("--shadow-alpha", type=float, default=0.7,
                   help="shadow dim strength in [0,1]; higher = darker unseen area")
    p.add_argument("--fov-levels", action="store_true",
                   help="3-level fog: overlap (both agents see) stays full-bright, "
                        "one-agent tiles dimmed by --shadow-alpha-mid, unseen by "
                        "--shadow-alpha (darkest)")
    p.add_argument("--shadow-alpha-mid", type=float, default=0.4,
                   help="dim strength for tiles seen by exactly one agent (--fov-levels)")
    p.add_argument("--shadow-style", default="lift", choices=["lift", "black"],
                   help="lift: darken + blue-grey tint (reads on the bare black floor); "
                        "black: pure semi-transparent black (use with --shadow-after so "
                        "it dims the attention-coloured floor + piles without a blue clash)")
    p.add_argument("--shadow-after", action="store_true",
                   help="apply the shadow AFTER the attention overlay (a black rectangle "
                        "over unseen tiles, dimming attention/piles/walls there too); "
                        "full-env only (no --crop)")
    p.add_argument("--video", type=int, default=0, metavar="N_EPISODES",
                   help="render N episodes as full attention VIDEOS in the exact filmstrip "
                        "style (coolwarm sum, full env, step stamp, +shadow) so picked "
                        "frames match the final figure; one mp4 per episode")
    p.add_argument("--video-tile", type=int, default=48,
                   help="tile size for --video (smaller than --tile; all frames render so "
                        "a big tile OOMs)")
    p.add_argument("--dump-oai", default=None, metavar="PATH.npz",
                   help="with --frames: dump per-frame grid/agents/recipe/attention to an "
                        "npz (for building the overcooked_ai sprite figure offline) and exit")
    p.add_argument("--shared-tint", type=float, default=0.0, metavar="ALPHA",
                   help="tint tiles both agents can see (0 = off); makes the shared "
                        "zone visible even where it lands on near-black floor")
    p.add_argument("--shared-outline", action="store_true",
                   help="outline the both-visible region in white instead of (or as "
                        "well as) tinting it; keeps warm hues free for the attention map")
    p.add_argument("--no-attn", action="store_true",
                   help="render the board with fog/FOV boxes only, no attention overlay "
                        "(environment figure rather than mechanism figure)")
    p.add_argument("--fov-scan", type=int, default=0, metavar="N_EPISODES",
                   help="report how many tiles both agents can see, per timestep, so a "
                        "frame can be picked on the measured overlap")
    p.add_argument("--isolines", action="store_true",
                   help="draw attention as contour rings over an unpainted board "
                        "instead of the blocky wash (--frames path)")
    p.add_argument("--iso-levels", default="4",
                   help="number of contour rings; a comma list sets it per frame "
                        "(e.g. 6,4 -- a frame with a sharply peaked core needs fewer, "
                        "more widely spaced rings for its labels to fit)")
    p.add_argument("--iso-lo", type=float, default=0.15,
                   help="lowest ring, as a fraction of the shared vmax")
    p.add_argument("--iso-lw", type=float, default=0.0,
                   help="ring line width in px (0 = auto from cell width)")
    p.add_argument("--iso-fill", type=float, default=0.0,
                   help="alpha of a soft fill between rings (0 = lines only)")
    p.add_argument("--iso-alpha", type=float, default=1.0,
                   help="opacity of the contour strokes (<1 lets the board read through)")
    p.add_argument("--board-dim", type=float, default=1.0,
                   help="scrim over the board before the overlay is drawn (<1 darkens "
                        "it so the contours carry the eye)")
    p.add_argument("--iso-labels", action="store_true",
                   help="inline numeric labels on each contour")
    p.add_argument("--iso-no-halo", action="store_true",
                   help="drop the dark stroke behind the contours (cleaner over a "
                        "board that is already dark)")
    p.add_argument("--fov-hatch", type=float, default=0.0, metavar="ALPHA",
                   help="hatch each agent's view at opposite diagonals (greyscale-safe; "
                        "the intersection reads as cross-hatch)")
    p.add_argument("--fov-hatch-period", type=float, default=0.42,
                   help="hatch line spacing as a fraction of tile size")
    p.add_argument("--fov-hatch-tint", type=float, default=0.16,
                   help="flat tint under the hatch (colour cue for colour readers)")
    p.add_argument("--fov-tint", type=float, default=0.0, metavar="ALPHA",
                   help="wash each agent's view window in its own colour instead of "
                        "outlining it; the intersection reads as a blend")
    p.add_argument("--grid", type=int, default=0, metavar="NCOLS",
                   help="lay --frames out in NCOLS columns over as many rows as needed "
                        "(--cbar-side bottom spans the grid, right spans its height)")
    p.add_argument("--stack", action="store_true",
                   help="one-column layout: stack the frames vertically (portrait) with the "
                        "colorbar on the right, to fit a single LaTeX column")
    p.add_argument("--fov-box", action="store_true",
                   help="draw a coloured box around each agent's view window (red=agent 0, "
                        "blue=agent 1) to make the FOV explicit")
    p.add_argument("--pad-margin", action="store_true",
                   help="two-tier partial-obs look: a solid-black off-map margin (the obs "
                        "black pad) plus semi-transparent dim for on-map non-FOV tiles "
                        "(--shadow-alpha sets the dim). Implies the FOV/shadow treatment.")
    args = p.parse_args()

    if (not args.frames and not args.scan and not args.scan_mutual and not args.batch
            and not args.video and not args.narrative and not args.fov_scan):
        p.error("pass --frames, --scan, --scan-mutual, --batch, --video, --narrative "
                "or --fov-scan")
    conf, env, env_l, policy, params = _load(args.checkpoint, args.no_op, args.ckpt)
    video_ctx = _ocv2_video_ctx(conf, env_l)
    # Sub-tile projection factor: the egocentric feature map (e.g. 10x10) tiles the
    # 5x5 crop 2x per tile, so binning to world tiles paints blocky per-tile squares.
    # subdiv = feat/fov keeps that sub-tile detail (finer heatmap, not rectangles).
    attn_subdiv = 1
    if video_ctx and video_ctx.get("egocentric") and not video_ctx.get("rotate_obs", False):
        _fh, _afs = int(video_ctx["feat_h"]), int(video_ctx["agent_fov_size"])
        if _afs > 0 and _fh % _afs == 0:
            attn_subdiv = max(1, _fh // _afs)
    print(f"[ocv2_filmstrip] attention sub-tile factor = {attn_subdiv}")
    feed_ctx = build_feed_ctx(conf, env_l)
    fa, fm, fr = feed_ctx
    max_steps = int(conf["ENV_KWARGS"].get("max_steps", 400))

    params_b = params
    if args.partner:
        pdata = load_train_run(_find_ckpt(args.partner))
        pkey = "best_params" if "best_params" in pdata else "final_params"
        params_b = jax.tree.map(
            lambda x: x[0] if hasattr(x, "shape") and x.ndim > 0 else x, pdata[pkey])
        print(f"[ocv2_filmstrip] CROSS-PLAY agent_1 <- {os.path.basename(args.partner)}")

    def rollout(ep):
        ep_states, attn, _a, _m = run_episode_with_states(
            jax.random.PRNGKey(ep), env, params, policy, params_b, policy, max_steps,
            collect_attention=True, feed_other_attn_dims=fa, feed_mask_fn=fm,
            feed_reframe_fn=fr,
        )
        return ep_states, _project_ego_attention(attn, ep_states, video_ctx, subdiv=attn_subdiv)

    if args.video:
        from envs.overcooked_v2.rendering import render_eval_frames
        from moviepy import ImageSequenceClip
        out_root = Path(args.out).parent if Path(args.out).suffix else Path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)
        for ep in range(args.video):
            ep_states, attn = rollout(ep)
            n = min(len(attn["agent_0"]), len(attn["agent_1"]), len(ep_states))
            a0 = [np.asarray(attn["agent_0"][t]).squeeze() for t in range(n)]
            a1 = [np.asarray(attn["agent_1"][t]).squeeze() for t in range(n)]
            fused = [x + y for x, y in zip(a0, a1)]
            vmax = args.vmax if args.vmax is not None else max(float(m.max()) for m in fused)
            raw = [np.asarray(f) for f in render_eval_frames(
                ep_states[:n], tile_size=args.video_tile, chunk=8)]
            cells = []
            for t in range(n):
                c = _overlay_blocky(raw[t], fused[t], vmax, args.cmap, args.cmap_lo,
                                    args.cmap_hi, args.alpha, args.gamma)
                if args.pad_margin:
                    _view = int(video_ctx["agent_view_size"] or 0)
                    c = _compose_padmargin(c, ep_states[t], video_ctx, _view,
                                           args.shadow_alpha, args.fov_box)
                else:
                    if args.shadow:
                        c = _shadow_unseen(c, ep_states[t], video_ctx, args.shadow_alpha,
                                           args.shadow_style)
                    if args.fov_box:
                        c = _draw_fov_box(c, ep_states[t], video_ctx)
                # No step-number stamp on videos (clean frames for picking).
                cells.append(np.ascontiguousarray(c))
            out = out_root / f"attn_ep{ep:02d}.mp4"
            ImageSequenceClip(cells, fps=10).write_videofile(
                str(out), fps=10, codec="libx264", audio=False, preset="ultrafast")
            print(f"[ocv2_filmstrip] video ep{ep} vmax={vmax:.3f} ({n} frames) -> {out}")
        print(f"[ocv2_filmstrip] wrote {args.video} attention videos to {out_root}")
        return

    if args.narrative:
        from agents.overcooked_v2.ja_overcooked_v2_attention import TaskObject
        from envs.overcooked_v2.common import MAX_INGREDIENTS
        n_static = int(video_ctx["static_num_objects"])
        pos = np.asarray(video_ctx["object_pos"])[:n_static]
        cat = np.asarray(video_ctx["object_cat"])[:n_static]
        pots = [(int(r), int(c)) for r, c in pos[cat == int(TaskObject.POT)]]
        goals = [(int(r), int(c)) for r, c in pos[cat == int(TaskObject.GOAL)]]
        pr, pc = pots[0]
        print(f"[narrative] pot={pots} delivery={goals} MAX_ING={MAX_INGREDIENTS}")

        def _klass(inv):
            inv = int(inv)
            plate, ing = (inv & 1) != 0, (inv >> 2) != 0
            return "dish" if (plate and ing) else "plate" if plate else \
                   "ingredient" if ing else "empty"

        def _adj(p, targets):
            return any(abs(p[0] - r) + abs(p[1] - c) <= 1 for r, c in targets)

        def _potcount(o):  # ingredients currently in the pot
            o = int(o) >> 2
            c = 0
            for _ in range(MAX_INGREDIENTS):
                c += o & 0x3
                o >>= 2
            return c

        for ep in range(args.narrative):
            ep_states, attn = rollout(ep)
            focus = _focus_series(attn, ep_states, video_ctx)

            def _f(t):
                return round(float(focus[t]), 2) if t < len(focus) else 0.0

            third, plate_pot, deliver = [], [], []
            for t in range(len(ep_states)):
                raw = _base_state(ep_states[t])
                pcount = _potcount(raw.grid[pr, pc, 1])
                ag = raw.agents
                for i in range(len(ag.pos.x)):
                    p = (int(ag.pos.y[i]), int(ag.pos.x[i]))
                    k = _klass(ag.inventory[i])
                    if k == "ingredient" and _adj(p, pots) and pcount == 2:
                        third.append((t, i))
                    if k == "plate" and _adj(p, pots):
                        plate_pot.append((t, i))
                    if k == "dish" and _adj(p, goals):
                        deliver.append((t, i))
            print(f"\n[narrative] ep{ep}:")
            print(f"  3rd-ingredient@pot (t,agent,focus): {[(t, i, _f(t)) for t, i in third[:8]]}")
            print(f"  plate@pot         : {[(t, i, _f(t)) for t, i in plate_pot[:8]]}")
            print(f"  dish@delivery     : {[(t, i, _f(t)) for t, i in deliver[:8]]}")
            for t1, _i1 in sorted(third, key=lambda x: -focus[x[0]] if x[0] < len(focus) else 0)[:3]:
                t2 = next((t for t, _i in plate_pot if t > t1), None)
                t3 = next((t for t, _i in deliver if t2 is not None and t > t2), None)
                if t2 is not None and t3 is not None:
                    print(f"  -> triple {ep}:{t1},{ep}:{t2},{ep}:{t3}  "
                          f"focus={_f(t1)}/{_f(t2)}/{_f(t3)}")
        return

    if args.batch:
        out_root = Path(args.out).parent if Path(args.out).suffix else Path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)
        force_ep = force_t0 = None
        if args.window:
            force_ep, force_t0 = (int(v) for v in args.window.split(":"))
        eps = [force_ep] if force_ep is not None else range(args.batch)
        for ep in eps:
            ep_states, attn = rollout(ep)
            if force_t0 is not None:
                t0, score = force_t0, float("nan")
            else:
                focus = _focus_series(attn, ep_states, video_ctx)
                t0, score = _best_window(focus, args.n_frames, args.stride)
            if args.steps is not None:
                steps = [int(v) for v in args.steps.split(",")]
                t0 = steps[0]
            else:
                steps = [t0 + i * args.stride for i in range(args.n_frames)]
            # Render natively at --tile rather than upscaling the default 32px board:
            # the sprites are drawn AT tile_size (with subdiv supersampling), so a big
            # tile gives genuinely sharp edges where an upscale only magnifies mush.
            # Render ONLY the selected states — supersampling all 400 at a big tile OOMs.
            from envs.overcooked_v2.rendering import render_eval_frames
            a0s = [np.asarray(attn["agent_0"][t]).squeeze() for t in steps]
            a1s = [np.asarray(attn["agent_1"][t]).squeeze() for t in steps]
            raw = [np.asarray(f) for f in render_eval_frames(
                [ep_states[t] for t in steps], tile_size=args.tile, chunk=2)]
            if args.shadow and not args.shadow_after:
                raw = [_shadow_unseen(f, ep_states[t], video_ctx, args.shadow_alpha,
                                      args.shadow_style)
                       for f, t in zip(raw, steps)]
            if args.crop:
                c0, c1 = (int(v) for v in args.crop.split(":"))
                tw = raw[0].shape[1] // int(video_ctx["grid_w"])
                raw = [f[:, c0 * tw:c1 * tw] for f in raw]
                a0s = [a[:, c0:c1] for a in a0s]
                a1s = [a[:, c0:c1] for a in a1s]
            bigs = [np.asarray(Image.fromarray(f).resize(
                (f.shape[1] * args.scale, f.shape[0] * args.scale),
                resample=Image.NEAREST)) for f in raw]
            d = out_root / f"ep{ep:02d}"
            d.mkdir(parents=True, exist_ok=True)
            print(f"\n[ocv2_filmstrip] ep{ep} window t{steps[0]}-{steps[-1]} "
                  f"(focus {score:.3f}) -> {d}")

            if args.recolour or args.muted:
                fused_r = {"sum": [x + y for x, y in zip(a0s, a1s)],
                           "intersection": [np.minimum(x, y) for x, y in zip(a0s, a1s)]}
                for mode, maps in fused_r.items():
                    vmax = max(float(m.max()) for m in maps)
                    if vmax < 1e-6:
                        print(f"    {mode}: all-zero, skipped")
                        continue
                    # --muted mirrors the LBF paper strip: additive glow, low gamma so
                    # weak attention stays soft, cmap-hi trimmed so peaks read salmon
                    # rather than saturated red. --recolour replaces the cell instead.
                    variants = ([(f"muted_{cm}_hi{hi}", cm, hi) for cm in args.muted_cmaps.split(",")
                                 for hi in (float(x) for x in args.muted_his.split(","))]
                                if args.muted else [(f"recolour_{cm}", cm, 1.0)
                                                    for cm in BATCH_CMAPS])
                    for tag, cm, hi in variants:
                        cells = []
                        for b, m, t in zip(bigs, maps, steps):
                            if args.muted:
                                c = _overlay_blocky(b, m, vmax, cm, args.cmap_lo, hi,
                                                    args.alpha, args.gamma)
                            else:
                                c = _recolour_cells(b, m, vmax, cm)
                            if args.pad_margin:
                                _view = int(video_ctx["agent_view_size"] or 0)
                                c = _compose_padmargin(c, ep_states[t], video_ctx, _view,
                                                       args.shadow_alpha, args.fov_box)
                            else:
                                if args.shadow and args.shadow_after:
                                    c = _shadow_unseen(c, ep_states[t], video_ctx,
                                                       args.shadow_alpha, args.shadow_style)
                                if args.fov_box:
                                    c = _draw_fov_box(c, ep_states[t], video_ctx)
                            c = np.ascontiguousarray(c)
                            _stamp_step(c, t)
                            cells.append(c)
                        bar = _render_hbar(cm, cells[0].shape[1] * args.cols)
                        _assemble_grid(cells, args.cols, bar, d / f"{tag}_{mode}.png")
                    print(f"    {mode}: vmax={vmax:.3f} ({len(variants)} variants)")
                continue

            fused = {"sum": [x + y for x, y in zip(a0s, a1s)],
                     "intersection": [np.minimum(x, y) for x, y in zip(a0s, a1s)]}
            for mode, maps in fused.items():
                vmax = max(float(m.max()) for m in maps)
                if vmax < 1e-6:
                    print(f"    {mode}: all-zero, skipped")
                    continue
                for cm in BATCH_CMAPS:
                    cells = [np.ascontiguousarray(
                        _overlay_blocky(b, m, vmax, cm, args.cmap_lo, args.cmap_hi,
                                        args.alpha, args.gamma))
                             for b, m in zip(bigs, maps)]
                    cbar = _render_colorbar(cm, args.cmap_lo, args.cmap_hi, cells[0].shape[0])
                    _strip(cells, cbar, d / f"{mode}_{cm}.png", steps)
                print(f"    {mode}: vmax={vmax:.3f}  ({len(BATCH_CMAPS)} palettes)")

            pk = max(max(float(x.max()), float(y.max())) for x, y in zip(a0s, a1s))
            for c0, c1 in PER_AGENT_PALETTES:
                cells = [np.ascontiguousarray(
                    _overlay_two_cmap(b, x, y, pk, c0, c1, args.alpha, args.gamma))
                         for b, x, y in zip(bigs, a0s, a1s)]
                _strip(cells, None, d / f"peragent_{c0}_{c1}.png", steps)
            print(f"    per-agent: vmax={pk:.3f}  ({len(PER_AGENT_PALETTES)} palettes)")
        print(f"\n[ocv2_filmstrip] wrote {args.batch} episodes to {out_root}")
        return

    if args.fov_scan:
        from collections import Counter
        for ep in range(args.fov_scan):
            ep_states, _ = rollout(ep)
            hist, frames = Counter(), {}
            for t, st in enumerate(ep_states):
                both, _one, _none, tot = _fov_counts(st, video_ctx)
                hist[both] += 1
                frames.setdefault(both, []).append(t)
            print(f"\n[fov_scan] ep{ep}: shared-visibility tiles out of {tot}")
            for v in sorted(hist):
                pct = 100 * hist[v] / len(ep_states)
                print(f"  both={v:3d}  {hist[v]:4d} frames ({pct:4.1f}%)  e.g. {frames[v][:10]}")
        return

    if args.scan_mutual:
        allr = []
        for ep in range(args.scan_mutual):
            ep_states, attn = rollout(ep)
            for r in _mutual_series(attn, ep_states, video_ctx):
                r["ep"] = ep
                allr.append(r)
        vis = [r for r in allr if r["vis"]]
        print(f"\n[ocv2_filmstrip] MUTUAL attention over {args.scan_mutual} episodes")
        print(f"  frames total            : {len(allr)}")
        print(f"  partner VISIBLE (cheb<=2): {len(vis)} ({100*len(vis)/len(allr):.1f}%)")
        if vis:
            both = [r for r in vis if r["a0_on_a1"] > 1e-6 and r["a1_on_a0"] > 1e-6]
            any0 = [r for r in vis if r["a0_on_a1"] > 1e-6]
            any1 = [r for r in vis if r["a1_on_a0"] > 1e-6]
            print(f"  | a0 attends a1's tile   : {100*len(any0)/len(vis):.1f}% of visible frames "
                  f"(mean share {np.mean([r['a0_on_a1'] for r in vis]):.3f})")
            print(f"  | a1 attends a0's tile   : {100*len(any1)/len(vis):.1f}% of visible frames "
                  f"(mean share {np.mean([r['a1_on_a0'] for r in vis]):.3f})")
            print(f"  | MUTUAL (both at once)  : {100*len(both)/len(vis):.1f}% of visible frames "
                  f"= {100*len(both)/len(allr):.1f}% of all frames")
            both.sort(key=lambda r: -min(r["a0_on_a1"], r["a1_on_a0"]))
            print("  best mutual frames:")
            for r in both[:10]:
                print(f"    ep{r['ep']} t{r['t']}  a0->a1={r['a0_on_a1']:.3f} "
                      f"a1->a0={r['a1_on_a0']:.3f} cheb={r['cheb']}")
        return

    if args.scan:
        best = []
        for ep in range(args.scan):
            ep_states, attn = rollout(ep)
            f = _focus_series(attn, ep_states, video_ctx)
            L = args.scan_len
            for t0 in range(0, len(f) - L):
                best.append((float(np.mean(f[t0:t0 + L])), ep, t0))
        best.sort(reverse=True)
        print(f"\n[ocv2_filmstrip] top windows by attention-on-pot+agents "
              f"(len {args.scan_len}, fraction of team mass):")
        seen = []
        for score, ep, t0 in best:
            if any(e == ep and abs(t0 - s) < args.scan_len for _, e, s in seen):
                continue
            seen.append((score, ep, t0))
            step = max(1, (args.scan_len - 1) // 4)
            spec = ",".join(f"{ep}:{t0 + i * step}" for i in range(5))
            print(f"  focus={score:.3f}  ep{ep} t{t0}-{t0 + args.scan_len - 1}   --frames {spec}")
            if len(seen) >= 12:
                break
        return

    want = _parse_frames(args.frames)
    from envs.overcooked_v2.rendering import render_eval_frames

    # PRNGKey(ep) matches evaluation.eval_attention_video, so an ep:t here is the
    # same state as frame t of that episode's video / row t of pot_attention.csv.
    # Render the wanted states natively at --tile (like --batch) so pad-margin/
    # shadow/box get a sharp board; keep the per-frame state for those overlays.
    cells_raw = []
    for ep in sorted({e for e, _ in want}):
        ep_states, attn = rollout(ep)
        ts = [t for e, t in want if e == ep]
        boards = [np.asarray(f) for f in render_eval_frames(
            [ep_states[t] for t in ts], tile_size=args.tile, chunk=2)]
        for t, board in zip(ts, boards):
            a0 = np.asarray(attn["agent_0"][t]).squeeze()
            a1 = np.asarray(attn["agent_1"][t]).squeeze()
            cells_raw.append((ep, t, board, a0, a1, ep_states[t]))
    cells_raw.sort(key=lambda r: want.index((r[0], r[1])))

    if args.dump_oai:
        data = {"frames": np.array([t for _e, t, *_ in cells_raw])}
        for e, t, board, a0, a1, st in cells_raw:
            raw = _base_state(st)
            data[f"grid_{t}"] = np.asarray(raw.grid)
            data[f"ay_{t}"] = np.asarray(raw.agents.pos.y)
            data[f"ax_{t}"] = np.asarray(raw.agents.pos.x)
            data[f"dir_{t}"] = np.asarray(raw.agents.dir)
            data[f"inv_{t}"] = np.asarray(raw.agents.inventory)
            data[f"recipe_{t}"] = np.asarray(raw.recipe)
            data[f"a0_{t}"] = np.asarray(a0)
            data[f"a1_{t}"] = np.asarray(a1)
        np.savez(args.dump_oai, **data)
        print(f"[ocv2_filmstrip] dumped OAI-render data for {len(cells_raw)} frames -> {args.dump_oai}")
        return

    def fuse(a0, a1):
        if args.combine == "sum":
            return a0 + a1
        if args.combine == "intersection":
            return np.minimum(a0, a1)
        return None  # per-agent keeps the maps separate

    if args.combine == "per-agent":
        peak = max(max(float(a0.max()), float(a1.max())) for _e, _t, _b, a0, a1, _s in cells_raw)
    else:
        peak = max(float(fuse(a0, a1).max()) for _e, _t, _b, a0, a1, _s in cells_raw)
    vmax = args.vmax if args.vmax is not None else peak
    print(f"[ocv2_filmstrip] combine={args.combine}  shared vmax = {vmax:.3f}")

    iso_levels = [int(v) for v in str(args.iso_levels).split(",")]
    cells = []
    for k_cell, (e, t, board, a0, a1, st) in enumerate(cells_raw):
        big = np.asarray(Image.fromarray(board).resize(
            (board.shape[1] * args.scale, board.shape[0] * args.scale),
            resample=Image.NEAREST))
        if args.board_dim < 1.0:
            big = np.clip(big.astype(np.float32) * args.board_dim, 0, 255).astype(np.uint8)
        both, one, none, tot = _fov_counts(st, video_ctx)
        print(f"[fov] ep{e} t{t}: shared={both} single={one} unseen={none} of {tot} tiles")
        if args.no_attn:
            cell = big
        elif args.combine == "per-agent":
            cell = _overlay_two_palette(big, a0, a1, vmax, args.alpha, args.gamma)
        elif args.isolines:
            nl = iso_levels[k_cell] if k_cell < len(iso_levels) else iso_levels[-1]
            cell = _overlay_isolines(big, fuse(a0, a1), vmax, args.cmap,
                                     nl, args.iso_lo, args.iso_lw,
                                     args.iso_fill, args.iso_labels,
                                     not args.iso_no_halo, None,
                                     args.cmap_lo, args.cmap_hi, args.iso_alpha)
        else:
            cell = _overlay_blocky(big, fuse(a0, a1), vmax, args.cmap, args.cmap_lo,
                                   args.cmap_hi, args.alpha, args.gamma)
        if args.pad_margin:
            _view = int(video_ctx["agent_view_size"] or 0)
            cell = _compose_padmargin(cell, st, video_ctx, _view, args.shadow_alpha,
                                      args.fov_box)
        else:
            if args.fov_levels:
                cell = _shadow_levels(cell, st, video_ctx, args.shadow_alpha,
                                      args.shadow_alpha_mid)
            elif args.shadow:
                cell = _shadow_unseen(cell, st, video_ctx, args.shadow_alpha,
                                      args.shadow_style)
            if args.fov_hatch > 0:
                cell = _hatch_fov(cell, st, video_ctx, args.fov_hatch,
                                  args.fov_hatch_tint, args.fov_hatch_period)
            if args.fov_tint > 0:
                cell = _tint_fov(cell, st, video_ctx, args.fov_tint)
            if args.shared_tint > 0:
                cell = _tint_shared(cell, st, video_ctx, args.shared_tint)
            if args.shared_outline:
                cell = _outline_shared(cell, st, video_ctx)
            if args.fov_box:
                cell = _draw_fov_box(cell, st, video_ctx)
        cell = np.ascontiguousarray(cell)
        _stamp_step(cell, t)
        cells.append(cell)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.stack:
        # One-column (portrait) layout: frames stacked top-to-bottom, colorbar on the
        # right spanning the full stack height. Fits a single LaTeX column.
        ch, cw = cells[0].shape[:2]
        pad = max(4, ch // 30)
        total_h = len(cells) * ch + (len(cells) - 1) * pad
        cbar = None
        if args.cbar_side != "none" and args.combine != "per-agent":
            cbar = _render_colorbar(args.cmap, args.cmap_lo, args.cmap_hi, total_h)
        gap = pad * 3
        cbar_w = cbar.shape[1] if cbar is not None else 0
        total_w = cw + (gap + cbar_w if cbar is not None else 0)
        canvas = np.full((total_h, total_w, 3), 255, np.uint8)
        for i, c in enumerate(cells):
            y = i * (ch + pad)
            canvas[y:y + ch, :cw] = c
        if cbar is not None:
            canvas[:cbar.shape[0], cw + gap:cw + gap + cbar_w] = cbar
        Image.fromarray(canvas).save(out)
        print(f"[ocv2_filmstrip] wrote {out} ({total_w}x{total_h}, {len(cells)}x1 stacked one-column)")
        return
    if args.grid:
        # Multi-row layout for a single LaTeX column: `--grid` cells per row, reading
        # order left-to-right then top-to-bottom, rows as needed. A row of N frames is
        # N times wider than one frame, so at \columnwidth each frame shrinks by N;
        # wrapping trades height for frame size.
        ncols = min(args.grid, len(cells))
        ch, cw = cells[0].shape[:2]
        pad = max(4, cw // 80)
        nrows = (len(cells) + ncols - 1) // ncols
        grid_w = ncols * cw + (ncols - 1) * pad
        grid_h = nrows * ch + (nrows - 1) * pad
        gap = pad * 3
        cbar = hbar = None
        if args.cbar_side != "none" and args.combine != "per-agent":
            if args.cbar_side == "bottom":
                hbar = (_render_minimal_hbar(args.cmap, args.cmap_lo, args.cmap_hi,
                                             grid_w, ch * 0.085)
                        if args.cbar_minimal
                        else _render_numbered_hbar(args.cmap, args.cmap_lo, args.cmap_hi,
                                                   grid_w, ch * 0.14,
                                                   max(16, round(ch * 0.05))))
            elif args.cbar_minimal:
                cbar = _render_minimal_cbar(args.cmap, args.cmap_lo, args.cmap_hi,
                                            grid_h, args.cbar_width)
            else:
                cbar = _render_colorbar(args.cmap, args.cmap_lo, args.cmap_hi, grid_h)
        total_w = grid_w + (gap + cbar.shape[1] if cbar is not None else 0)
        total_h = grid_h + (gap + hbar.shape[0] if hbar is not None else 0)
        canvas = np.full((total_h, total_w, 3), 255, np.uint8)
        for i, c in enumerate(cells):
            r, k = divmod(i, ncols)
            canvas[r * (ch + pad):r * (ch + pad) + ch, k * (cw + pad):k * (cw + pad) + cw] = c
        if cbar is not None:
            canvas[:cbar.shape[0], grid_w + gap:grid_w + gap + cbar.shape[1]] = cbar
        if hbar is not None:
            canvas[grid_h + gap:grid_h + gap + hbar.shape[0], :grid_w] = hbar
        Image.fromarray(canvas).save(out)
        print(f"[ocv2_filmstrip] wrote {out} ({total_w}x{total_h}, {nrows}x{ncols} grid, "
              f"cbar {args.cbar_side})")
        return
    if args.cbar_side == "bottom":
        # Row of frames + a horizontal NUMBERED colorbar underneath (keeps the figure
        # no wider than the row itself).
        ch, cw = cells[0].shape[:2]
        pad = max(4, cw // 80)
        row_w = len(cells) * cw + (len(cells) - 1) * pad
        row = np.full((ch, row_w, 3), 255, np.uint8)
        for i, c in enumerate(cells):
            row[:, i * (cw + pad):i * (cw + pad) + cw] = c
        hbar = _render_numbered_hbar(args.cmap, args.cmap_lo, args.cmap_hi, row_w,
                                     ch * 0.14, max(16, round(ch * 0.05)))
        gap = pad * 3
        canvas = np.full((ch + gap + hbar.shape[0], row_w, 3), 255, np.uint8)
        canvas[:ch] = row
        canvas[ch + gap:ch + gap + hbar.shape[0]] = hbar
        Image.fromarray(canvas).save(out)
        print(f"[ocv2_filmstrip] wrote {out} ({row_w}x{canvas.shape[0]}, row + bottom colorbar)")
        return
    cbar = None
    if args.cbar_side != "none" and args.combine != "per-agent":
        cbar = (_render_minimal_cbar(args.cmap, args.cmap_lo, args.cmap_hi,
                                     cells[0].shape[0], args.cbar_width)
                if args.cbar_minimal
                else _render_colorbar(args.cmap, args.cmap_lo, args.cmap_hi,
                                      cells[0].shape[0]))
    _assemble_row(cells, len(cells), cbar, out,
                  cbar_side=("left" if args.cbar_side == "left" else "right"))
    print(f"[ocv2_filmstrip] wrote {out}")


if __name__ == "__main__":
    main()
