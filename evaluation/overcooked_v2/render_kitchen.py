"""Flat-colour redraw of an OvercookedV2 board from its symbolic state.

The env's own renderer draws a near-black board, which prints as an unreadable dark
slab at figure scale. This draws the same state in a light kitchen palette with the
agents' view windows as translucent washes, so the shared region appears as a blended
colour rather than something to infer from two crossing outlines.

Vector output (matplotlib patches, not a raster), so the board stays sharp at any
column width. Nothing here is loaded from disk -- no external sprite assets.

Cell codes follow `envs/overcooked_v2/common.py`: grid[..., 0] is a `StaticObject`,
grid[..., 1] a `DynamicObject` bitfield (bit 0 plate, bit 1 cooked, 2 bits per
ingredient from bit 2 up).
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

FLOOR = "#C6963C"        # walkable tile
COUNTER = "#E6D6B8"      # wall / counter surface
GRIDLINE = "#5A4A32"
POT_BODY = "#4A4A4A"
POT_RIM = "#2E2E2E"
DELIVERY = "#5C9A5C"
PLATE = "#FFFFFF"
PLATE_EDGE = "#B8B8B8"
INDICATOR_BG = "#8A6A4A"
A0_TINT = "#E8907A"      # informed agent view
A1_TINT = "#A9AEE2"      # partner view
A0_HAT = "#D93A2B"
A1_HAT = "#2E6FD9"
SKIN = "#F2D2B0"
COAT = "#F7F2EA"

# demo_cook_simple has 3 ingredients; index 1 is the recipe ingredient in the episodes
# used for the figure. Colours are chosen to stay distinct from the FOV washes.
ING_COLS = ["#B85CC8", "#3FA34D", "#E8C33A"]

STATIC_EMPTY, STATIC_WALL = 0, 1
STATIC_GOAL, STATIC_POT = 4, 5
STATIC_RECIPE_IND, STATIC_BUTTON_IND = 6, 7
STATIC_PLATE_PILE, STATIC_ING_BASE = 9, 10

DYN_PLATE, DYN_COOKED, DYN_ING_BASE = 1, 2, 4


def _ingredient_list(dyn: int) -> list[int]:
    """Ingredient indices held in a DynamicObject bitfield (2 bits per ingredient)."""
    out, obj, idx = [], int(dyn) >> 2, 0
    while obj > 0:
        out += [idx] * (obj & 0x3)
        obj >>= 2
        idx += 1
    return out


def _cluster(ax, cx, cy, colours, r=0.085, spread=0.13, zorder=6):
    """Small rosette of dots, the way piles and multi-ingredient soups read."""
    n = len(colours)
    if n == 0:
        return
    if n == 1:
        offs = [(0, 0)]
    else:
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
        offs = [(spread * np.cos(a), spread * np.sin(a)) for a in ang]
    for (dx, dy), col in zip(offs, colours):
        ax.add_patch(Circle((cx + dx, cy + dy), r, facecolor=col,
                            edgecolor="#00000055", linewidth=0.3, zorder=zorder))


def _chef(ax, cx, cy, direction, hat, held=None):
    """Simple chef: body, hat band, facing nub, plus whatever is carried."""
    ax.add_patch(FancyBboxPatch((cx - 0.26, cy - 0.20), 0.52, 0.42,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor=COAT, edgecolor="#00000066",
                                linewidth=0.5, zorder=7))
    ax.add_patch(FancyBboxPatch((cx - 0.26, cy - 0.30), 0.52, 0.17,
                                boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=hat, edgecolor="#00000066",
                                linewidth=0.5, zorder=8))
    ax.add_patch(Circle((cx, cy - 0.02), 0.115, facecolor=SKIN,
                        edgecolor="#00000055", linewidth=0.4, zorder=8))
    # Facing nub: dir 0/1/2/3 = N/S/E/W.
    dx, dy = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0)}.get(int(direction), (0, 1))
    ax.add_patch(Polygon([[cx + 0.30 * dx - 0.07 * dy, cy + 0.30 * dy - 0.07 * dx],
                          [cx + 0.30 * dx + 0.07 * dy, cy + 0.30 * dy + 0.07 * dx],
                          [cx + 0.42 * dx, cy + 0.42 * dy]],
                         closed=True, facecolor=hat, edgecolor="none", zorder=8))
    if held:
        _cluster(ax, cx + 0.26, cy + 0.26, held, r=0.062, spread=0.085, zorder=9)


def draw_board(ax, grid, agent_pos, agent_dir, agent_inv, fov=None, attn=None,
               vmax=1.0, fov_alpha=0.42, attn_cmap="Reds", attn_alpha=0.85):
    """Draw one frame. `fov` is (2, H, W) bool; `attn` an (H*s, W*s) attention map."""
    static, dyn = np.asarray(grid)[..., 0], np.asarray(grid)[..., 1]
    gh, gw = static.shape

    for r in range(gh):
        for c in range(gw):
            s = int(static[r, c])
            base = COUNTER if s != STATIC_EMPTY else FLOOR
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=base, edgecolor=GRIDLINE,
                                   linewidth=0.45, zorder=1))
            cx, cy = c + 0.5, r + 0.5
            if s == STATIC_POT:
                ax.add_patch(FancyBboxPatch((cx - 0.30, cy - 0.22), 0.60, 0.44,
                                            boxstyle="round,pad=0,rounding_size=0.08",
                                            facecolor=POT_BODY, edgecolor=POT_RIM,
                                            linewidth=0.6, zorder=4))
                ing = _ingredient_list(dyn[r, c])
                if ing:
                    _cluster(ax, cx, cy - 0.02, [ING_COLS[i % len(ING_COLS)] for i in ing],
                             r=0.065, spread=0.10, zorder=5)
            elif s == STATIC_GOAL:
                ax.add_patch(Rectangle((c + 0.12, r + 0.12), 0.76, 0.76,
                                       facecolor=DELIVERY, edgecolor="#3E6E3E",
                                       linewidth=0.6, zorder=4))
            elif s == STATIC_PLATE_PILE:
                for k, (dx, dy) in enumerate(((-0.13, -0.10), (0.14, -0.12), (0.0, 0.15))):
                    ax.add_patch(Circle((cx + dx, cy + dy), 0.135, facecolor=PLATE,
                                        edgecolor=PLATE_EDGE, linewidth=0.5,
                                        zorder=4 + k * 0.01))
            elif s in (STATIC_RECIPE_IND, STATIC_BUTTON_IND):
                ax.add_patch(Rectangle((c + 0.14, r + 0.14), 0.72, 0.72,
                                       facecolor=INDICATOR_BG, edgecolor="#5E4632",
                                       linewidth=0.6, zorder=4))
            elif s >= STATIC_ING_BASE:
                idx = s - STATIC_ING_BASE
                _cluster(ax, cx, cy, [ING_COLS[idx % len(ING_COLS)]] * 3,
                         r=0.105, spread=0.15, zorder=4)
            elif s == STATIC_WALL:
                d = int(dyn[r, c])
                if d:
                    if d & DYN_PLATE:
                        ax.add_patch(Circle((cx, cy), 0.19, facecolor=PLATE,
                                            edgecolor=PLATE_EDGE, linewidth=0.5,
                                            zorder=4))
                    ing = _ingredient_list(d)
                    if ing:
                        _cluster(ax, cx, cy, [ING_COLS[i % len(ING_COLS)] for i in ing],
                                 r=0.062, spread=0.095, zorder=5)

    if attn is not None:
        from matplotlib import colormaps
        a = np.asarray(attn).squeeze()
        s = max(1, a.shape[0] // gh)
        cmap = colormaps[attn_cmap]
        for r in range(gh):
            for c in range(gw):
                v = float(a[r * s:(r + 1) * s, c * s:(c + 1) * s].sum())
                f = min(v / max(vmax, 1e-9), 1.0)
                if f > 0.02:
                    ax.add_patch(Rectangle((c, r), 1, 1, facecolor=cmap(f),
                                           edgecolor="none", alpha=attn_alpha * f,
                                           zorder=2))

    if fov is not None:
        for m, col in zip(np.asarray(fov), (A0_TINT, A1_TINT)):
            for r, c in zip(*np.where(m)):
                ax.add_patch(Rectangle((c, r), 1, 1, facecolor=col, edgecolor="none",
                                       alpha=fov_alpha, zorder=3))

    for i, hat in enumerate((A0_HAT, A1_HAT)):
        r, c = int(agent_pos[i][0]), int(agent_pos[i][1])
        held = [ING_COLS[k % len(ING_COLS)] for k in _ingredient_list(agent_inv[i])]
        if int(agent_inv[i]) & DYN_PLATE:
            held = held or [PLATE]
        _chef(ax, c + 0.5, r + 0.5, agent_dir[i], hat, held)

    ax.set_xlim(0, gw)
    ax.set_ylim(gh, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
