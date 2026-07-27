"""Build the OvercookedV2 qualitative figure from real logged episode data.

Three environment frames showing the handover, plus a peak-object agreement curve.

Metric note. The obvious choice, the histogram intersection sum_e min(p1, p2), is not
usable here for two reasons measured on this data:
  1. Egocentric attention maps have disjoint spatial support when the FOVs are
     disjoint, so the intersection is identically 0 off the overlap band -- a rise at
     the band edge is forced by geometry and evidences nothing.
  2. The intersection is maximised by DIFFUSE distributions, and MATE's attention is
     measurably more diffuse than OP's (entropy 0.67 vs 0.43 over 8 episodes), so a
     MATE-over-OP gap in it is confounded with peakedness.
This panel therefore plots MUTUAL FOCUS: the mean mass each agent puts on the object
the OTHER is peaked on, 0.5 * (p1(argmax p2) + p2(argmax p1)). It is continuous (so a
per-frame marker is meaningful), and a merely diffuse pair scores at the uniform
baseline 1/M = 0.083 rather than high, so it does not reward spreading attention.
Point 1 still applies to any such metric, so the caption must say the off-band zero is
structural. The claim rests on the printed null control: pairing agent 0 at t with
agent 1 at t+17 preserves both marginals and destroys only the synchrony, and MATE
scores 0.493 on overlap frames against a 0.177 null, while OP scores 0.000 against
0.000 -- OP's uninformed agent never places any mass on the pot at all.

Reads the npz from `extract_mate_figure_data.py`. Runs anywhere numpy/matplotlib do
(no GPU, no JAX).

    uv run python -m evaluation.overcooked_v2.plot_mate_figure \
        --data mate_fig_ep0.npz --frames 138,162,166 --out-dir plots/overcooked-v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

A_COL = "#E4572E"   # agent A = informed
B_COL = "#3E92CC"   # agent B = uninformed partner
SHARED_COL = "#D6D3D1"
OP_COL = "#A8A29E"
BAND_COL = "#F5F1E8"

TITLES = ["disjoint views", "focus enters shared view", "partner follows and acts"]


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _fov_rect(ax, r, c, view, gh, gw, colour, lw=1.4, dashed=False):
    y0, y1 = max(0, r - view), min(gh, r + view + 1)
    x0, x1 = max(0, c - view), min(gw, c + view + 1)
    return _rect(ax, x0, y0, x1, y1, colour, lw, dashed)


def _rect(ax, x0, y0, x1, y1, colour, lw, dashed=False, pad=0.06, zorder=4):
    box = FancyBboxPatch(
        (x0 + pad, y0 + pad), (x1 - x0) - 2 * pad, (y1 - y0) - 2 * pad,
        boxstyle="round,pad=0,rounding_size=0.18",
        linewidth=lw, edgecolor=colour, facecolor="none",
        linestyle=(0, (2.5, 2.5)) if dashed else "solid", zorder=zorder)
    ax.add_patch(box)
    return box


def _ring(ax, r, c, p, colour, radius):
    """Object-level focus ring: white halo underneath, agent-coloured ring on top."""
    cx, cy = c + 0.5, r + 0.5
    ax.add_patch(Circle((cx, cy), radius, fill=False, edgecolor="white",
                        linewidth=1.2 + 3.2 * p, alpha=0.55 * p, zorder=5))
    ax.add_patch(Circle((cx, cy), radius, fill=False, edgecolor=colour,
                        linewidth=0.6 + 2.2 * p, alpha=0.35 + 0.65 * p, zorder=6))


def _chip(ax, t):
    ax.text(0.035, 0.94, f"t = {t}", transform=ax.transAxes, ha="left", va="top",
            family="monospace", fontsize=7, color="white", zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="black", alpha=0.62,
                      edgecolor="none"))


def _shared_extent(fov0, fov1):
    inter = fov0 & fov1
    if not inter.any():
        return None
    rs, cs = np.where(inter)
    return cs.min(), rs.min(), cs.max() + 1, rs.max() + 1


def _mutual_focus(P):
    """0.5 * (p1(argmax p2) + p2(argmax p1)) per timestep."""
    j0, j1 = P[0].argmax(1), P[1].argmax(1)
    idx = np.arange(P.shape[1])
    return 0.5 * (P[0][idx, j1] + P[1][idx, j0])


def _rolling(x, w):
    if w <= 1:
        return x.astype(float)
    pad = w // 2
    xp = np.pad(x.astype(float), (pad, pad), mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")[:len(x)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", required=True, help="three timesteps, e.g. 138,162,166")
    ap.add_argument("--window", default=None, metavar="T0:T1",
                    help="x-range of the mini-plot (default: the rendered frame window)")
    ap.add_argument("--smooth", type=int, default=1,
                    help="rolling window; 1 = raw per-frame values")
    ap.add_argument("--cells", default=None,
                    help="three PNGs pre-rendered by eval_ocv2_filmstrip (blocky "
                         "attention wash + 3-level fog + FOV boxes + step stamp), used "
                         "as the frame panels instead of redrawing the board here")
    ap.add_argument("--out-dir", default="plots/overcooked-v2")
    ap.add_argument("--name", default="figure_ocv2")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    ts = [int(v) for v in args.frames.split(",")]
    assert len(ts) == 3
    names = list(d["obj_names"])
    P, fov, obj_pos = d["p_mate"], d["fov"], d["obj_pos"]
    shared, gh, gw = d["shared_count"], int(d["grid_h"]), int(d["grid_w"])
    view, apos = int(d["view_size"]), d["agent_pos"]
    frames, frame_ts = d["frames"], list(d["frame_ts"])

    t0, t1 = ((int(v) for v in args.window.split(":")) if args.window
              else (frame_ts[0], frame_ts[-1]))
    span = np.arange(t0, t1 + 1)

    _style()
    cells = args.cells.split(",") if args.cells else None
    # Explicit axes rather than a grid: the board is 2.2:1, so a grid row sized for the
    # mini-plot leaves a band of dead space above and below every frame.
    FW, GAP, X0, TOP = 0.228, 0.012, 0.006, 0.875
    H = 2.0
    fig = plt.figure(figsize=(6.9, H))
    fh = (FW * 6.9) / 2.2 / H
    frame_axes = []
    for k, t in enumerate(ts):
        ax = fig.add_axes((X0 + k * (FW + GAP), TOP - fh, FW, fh))
        frame_axes.append(ax)
        if cells:
            # The cell already carries the wash, fog, FOV boxes and step stamp at full
            # render resolution -- drawing anything over it would only fight it.
            ax.imshow(plt.imread(cells[k]), interpolation="lanczos", zorder=1)
        else:
            img = frames[frame_ts.index(t)].astype(np.float32) * 0.55
            ax.imshow(np.clip(img, 0, 255).astype(np.uint8),
                      extent=(0, gw, gh, 0), interpolation="nearest", zorder=1)
            ax.set_xlim(0, gw)
            ax.set_ylim(gh, 0)
            for i, col in ((0, A_COL), (1, B_COL)):
                _fov_rect(ax, int(apos[i, t, 0]), int(apos[i, t, 1]), view, gh, gw, col)
            ext = _shared_extent(fov[0, t], fov[1, t])
            if ext is not None:
                _rect(ax, *ext, SHARED_COL, 0.9, dashed=True, pad=0.19, zorder=7)
            j0, j1 = int(P[0, t].argmax()), int(P[1, t].argmax())
            _ring(ax, *obj_pos[j0], float(P[0, t, j0]), A_COL, 0.42)
            _ring(ax, *obj_pos[j1], float(P[1, t, j1]),
                  B_COL, 0.34 if j0 == j1 else 0.42)
            _chip(ax, t)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(TITLES[k], fontsize=8, pad=3)

    x_lo, x_hi = X0, X0 + 3 * FW + 2 * GAP
    cax = fig.add_axes((x_lo + 0.030, 0.275, (x_hi - x_lo) - 0.060, 0.052))
    grad = np.linspace(0, 1, 512)[None, :]
    cax.imshow(grad, aspect="auto", cmap="coolwarm", extent=(0, 1, 0, 1))
    cax.set_yticks([])
    cax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    cax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    cax.tick_params(length=2, labelsize=6.5, pad=1.5)
    for sp in cax.spines.values():
        sp.set_visible(False)
    cax.set_xlabel("team attention (shared scale across frames)", fontsize=7,
                   labelpad=2)

    ax = fig.add_axes((x_hi + 0.085, 0.235, 0.975 - (x_hi + 0.085), 0.63))
    curves = [(_rolling(_mutual_focus(P), args.smooth)[span], A_COL, 1.5, "MATE")]
    if "p_op" in d:
        Q = d["p_op"]
        n = min(Q.shape[1], P.shape[1])
        curves.append((_rolling(_mutual_focus(Q), args.smooth)[span[span < n]],
                       OP_COL, 1.2, "OP"))

    # Overlap is NOT one contiguous stretch -- panel (a)'s disjoint moment sits in a
    # gap between runs, so shading first-to-last would paint over the very frame the
    # figure turns on. Shade each run separately.
    ov = shared[span] > 0
    runs, start = [], None
    for k, flag in enumerate(ov):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            runs.append((span[start], span[k - 1]))
            start = None
    if start is not None:
        runs.append((span[start], span[len(ov) - 1]))
    for r0, r1 in runs:
        ax.axvspan(r0, r1 + 1, color=BAND_COL, zorder=0, lw=0)
    if runs:
        main_run = max(runs, key=lambda r: r[1] - r[0])
        ax.text(main_run[0] + 0.6, 0.965, "views overlap", fontsize=6.5, style="italic",
                color="#78716C", ha="left", va="top", zorder=2)

    # Both curves finish near zero, so labelling at the right end stacks the words.
    # Anchor at the last panel timestep, where MATE is on its plateau and OP is flat.
    k_lab = int(list(span).index(ts[-1]))
    for y, col, lw, lab in curves:
        x = span[:len(y)]
        ax.plot(x, y, color=col, lw=lw, zorder=3, solid_capstyle="round")
        k = min(k_lab, len(y) - 1)
        ax.text(x[k] + 1.8, max(float(y[k]), 0.0) + 0.06, lab, color=col, fontsize=7,
                va="center", ha="left", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])

    ym = curves[0][0]
    ax.plot(ts, [ym[list(span).index(t)] for t in ts], "o", ms=3.4, mfc="white",
            mec=A_COL, mew=1.0, ls="none", zorder=7)

    ax.set_xlim(t0, t1 + 4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2, labelsize=6.5)
    ax.set_xlabel("timestep", fontsize=7.5, labelpad=1.5)
    ax.set_ylabel("mutual focus", fontsize=7.5, labelpad=1)

    if cells:
        handles = [
            Line2D([], [], color="#FF3C3C", lw=1.6, label="informed agent view"),
            Line2D([], [], color="#4682FF", lw=1.6, label="partner view"),
        ]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x_lo + 0.030, 0.005),
                   ncol=2, frameon=False, fontsize=6.8, handlelength=1.6,
                   columnspacing=1.6, handletextpad=0.5)
    else:
        handles = [
            Line2D([], [], color=A_COL, lw=1.4, label="agent A view"),
            Line2D([], [], color=B_COL, lw=1.4, label="agent B view"),
            Line2D([], [], color=SHARED_COL, lw=1.0, ls=(0, (2.5, 2.5)),
                   label="shared view"),
            Line2D([], [], color="#444444", lw=1.1, marker="o", mfc="none", ms=6,
                   ls="none", label=r"object-level focus (ring width $\propto p_t(e)$)"),
        ]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.006, 0.005),
                   ncol=4, frameon=False, fontsize=6.8, handlelength=1.6,
                   columnspacing=1.5, handletextpad=0.5)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{args.name}.pdf")
    fig.savefig(out / f"{args.name}.png", dpi=300)
    plt.close(fig)

    print(f"[figure] wrote {out / args.name}.pdf and .png")
    print(f"[caption] episode {int(d['episode'])}, timesteps {ts}, window {t0}-{t1}")
    print(f"[caption] overlap runs in window: {runs}")
    for k, t in enumerate(ts):
        j0, j1 = int(P[0, t].argmax()), int(P[1, t].argmax())
        print(f"[caption] t={t} ({TITLES[k]}): shared {int(shared[t])} tiles | "
              f"A peak {names[j0]} p={P[0, t, j0]:.2f} | B peak {names[j1]} p={P[1, t, j1]:.2f}")


if __name__ == "__main__":
    main()
