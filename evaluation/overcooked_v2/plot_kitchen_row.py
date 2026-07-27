"""Three redrawn OvercookedV2 frames in a row, with subcaptions.

Uses `render_kitchen.draw_board` (flat-colour vector redraw) on the symbolic states
dumped by `extract_mate_figure_data.py`. Agent view windows are translucent washes, so
the shared region reads as a blend of the two.

    uv run python -m evaluation.overcooked_v2.plot_kitchen_row \
        --data mate_fig_ep0.npz --frames 138,152,166 --out-dir plots/overcooked-v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluation.overcooked_v2.render_kitchen import A0_TINT, A1_TINT, draw_board

CAPTIONS = ["(a) disjoint views", "(b) focus enters shared view",
            "(c) partner follows and acts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--attn", action="store_true", help="overlay the team attention map")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--fov-alpha", type=float, default=0.30)
    ap.add_argument("--captions", default=None, help="comma-separated, overrides default")
    ap.add_argument("--width", type=float, default=6.9)
    ap.add_argument("--out-dir", default="plots/overcooked-v2")
    ap.add_argument("--name", default="kitchen_row")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    ts = [int(v) for v in args.frames.split(",")]
    frame_ts = list(d["frame_ts"])
    grids, fov = d["grids"], d["fov"]
    apos, adir, ainv = d["agent_pos"], d["agent_dir"], d["agent_inv"]
    caps = args.captions.split(",") if args.captions else CAPTIONS

    attn_maps = None
    if args.attn:
        # Team attention = the two agents' world-projected maps summed, exactly as the
        # filmstrip fuses them, so the two figures show the same quantity.
        a0, a1 = d["attn_a0"], d["attn_a1"]
        attn_maps = {t: a0[frame_ts.index(t)] + a1[frame_ts.index(t)] for t in ts}
        vmax = args.vmax or max(float(m.max()) for m in attn_maps.values())
        print(f"[kitchen] attention vmax = {vmax:.3f}")
    else:
        vmax = 1.0

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    gh, gw = grids.shape[1], grids.shape[2]
    n = len(ts)
    pw = (args.width - 0.06 * (n - 1)) / n
    ph = pw * gh / gw
    # Inches reserved below the boards for the caption and the view key, plus a strip
    # above for the timestep title -- laying the axes flush to the figure clips it.
    BOT, TOPGAP = 0.46, 0.17
    H = ph + BOT + TOPGAP
    fig = plt.figure(figsize=(args.width, H))
    for k, t in enumerate(ts):
        i = frame_ts.index(t)
        x0 = (k * (pw + 0.06)) / args.width
        ax = fig.add_axes((x0, BOT / H, pw / args.width, ph / H))
        draw_board(ax, grids[i], apos[:, t], adir[:, t], ainv[:, t],
                   fov=fov[:, t], attn=attn_maps[t] if attn_maps else None,
                   vmax=vmax, fov_alpha=args.fov_alpha)
        ax.set_title(f"t = {t}", fontsize=7.5, pad=2)
        fig.text(x0 + (pw / args.width) / 2, 0.25 / H,
                 caps[k] if k < len(caps) else "", ha="center", va="bottom", fontsize=8)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=A0_TINT, edgecolor="#00000033", lw=0.4,
                     label="informed agent view"),
               Patch(facecolor=A1_TINT, edgecolor="#00000033", lw=0.4,
                     label="partner view")]
    if attn_maps:
        handles.append(Patch(facecolor="#C0392B", edgecolor="none",
                             label="team attention"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02 / H),
               ncol=len(handles), frameon=False, fontsize=7, handlelength=1.1,
               handleheight=0.9, columnspacing=1.8, handletextpad=0.5)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{args.name}.pdf")
    fig.savefig(out / f"{args.name}.png", dpi=300)
    plt.close(fig)
    print(f"[kitchen] wrote {out / args.name}.pdf and .png ({args.width}x{H:.2f} in)")
    for t in ts:
        print(f"[kitchen] t={t}: shared {int((fov[0, t] & fov[1, t]).sum())} tiles")


if __name__ == "__main__":
    main()
