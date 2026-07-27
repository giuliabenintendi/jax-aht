"""Single-panel card-game ablation bars: baselines + MATE ablations + MATE."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# matplotlib tab10 palette (order: orange, blue, pink, brown, lilac, green, red)
ORANGE = "#ff7f0e"
BLUE   = "#1f77b4"
PINK   = "#e377c2"
BROWN  = "#7D5A3C"
LILAC  = "#9467bd"
GREEN  = "#2ca02c"
RED    = "#d62728"
LEEGREEN  = "#8EBB69"   # Lee et al.
DARKGREEN = "#3E8347"   # MATE-aux (medium green: clearly lighter than the dark swatch, still distinct from Lee's light green)
RANDOM_COLOR, RANDOM_BASELINE, SP_HATCH = "#8B0000", 0.20, "//"

# (label, colour, SP, SPsem, XP, XPsem)  -- HE = highest all-pairs XP over the live-deliberation 48-seed sweep (alpha=1.75); IPPO = delib task
COND = [
    ("IPPO",       BROWN,     1.000, 0.000, 0.208, 0.084),
    ("OP",         BLUE,      0.200, 0.002, 0.201, 0.002),
    ("HE IPPO",    PINK,      1.000, 0.000, 0.214, 0.008),
    ("Lee et al.", LEEGREEN,  1.000, 0.000, 0.195, 0.002),
    ("MATE$-$OP",  ORANGE,    1.000, 0.000, 0.218, 0.028),
    ("MATE$-$feed", LILAC,    0.203, 0.058, 0.200, 0.004),
    ("MATE$-$aux", DARKGREEN, 0.219, 0.060, 0.203, 0.005),
    ("MATE",       RED,       0.872, 0.008, 0.823, 0.010),
]
W = 0.40
EKW = dict(ecolor="black", elinewidth=1.4, capsize=5, capthick=1.4)


def draw_underbrace(ax, xmin, xmax, y_top, depth, transform, beta=20.0, lw=1.2):
    """Horizontal curly underbrace (central tip pointing down) grouping xmin..xmax."""
    n = 201
    x = np.linspace(xmin, xmax, n)
    xh = x[: n // 2 + 1]
    yh = 1 / (1 + np.exp(-beta * (xh - xh[0]))) + 1 / (1 + np.exp(-beta * (xh - xh[-1])))
    yfull = np.concatenate((yh, yh[-2::-1]))  # 0.5 at ends, 1.5 at centre
    y = y_top - depth * (yfull - 0.5)
    ax.plot(x, y, color="black", lw=lw, transform=transform, clip_on=False)

labels = [c[0] for c in COND]
x = np.arange(len(labels))
plt.rcParams["hatch.linewidth"] = 1.4
fig, ax = plt.subplots(figsize=(13.5, 5.8))
for i, (_, base, spm, spe, xpm, xpe) in enumerate(COND):
    ax.bar(x[i] - W / 2, spm, W, color="white", hatch=SP_HATCH,
           edgecolor=base, linewidth=1.3, yerr=spe, error_kw=EKW, zorder=3)
    ax.bar(x[i] + W / 2, xpm, W, color=base, edgecolor="none",
           yerr=xpe, error_kw=EKW, zorder=3)
ax.axhline(RANDOM_BASELINE, ls="--", lw=1.8, color=RANDOM_COLOR, zorder=5)

# The three MATE ablations share a "MATE ablations" bracket, so they get short
# horizontal "w/o X" labels while the standalone conditions stay rotated.
ABLATION_IDX = (4, 5, 6)
tick_labels = list(labels)
tick_labels[4], tick_labels[5], tick_labels[6] = "w/o OP", "w/o feed", "w/o aux"

ax.set_xticks(x)
texts = ax.set_xticklabels(tick_labels, fontsize=25)
for i, t in enumerate(texts):
    if i in ABLATION_IDX:
        t.set_rotation(0)
        t.set_ha("center")
        t.set_fontsize(19)
    else:
        t.set_rotation(30)
        t.set_ha("right")
        t.set_rotation_mode("anchor")

xtrans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
lo, hi = ABLATION_IDX[0] - 0.42, ABLATION_IDX[-1] + 0.42
mid = (lo + hi) / 2
draw_underbrace(ax, lo, hi, -0.085, 0.055, xtrans)
ax.text(mid, -0.165, "MATE ablations", transform=xtrans, ha="center",
        va="top", fontsize=23)
ax.set_xlim(-0.7, len(labels) - 0.3)
ax.set_ylim(0, 1.12)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("Episode return", fontsize=28)
ax.tick_params(axis="y", labelsize=24)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [
    Patch(facecolor="white", hatch=SP_HATCH, edgecolor="#888888", label="Self-play (SP)"),
    Patch(facecolor="#888888", edgecolor="none", label="Cross-play (XP)"),
    Line2D([0], [0], color=RANDOM_COLOR, ls="--", lw=1.8, label="Random baseline"),
]
# Placed at width=0.98\linewidth in a single column (scaled down ~0.24x), so
# axis fonts read small. The single-row legend is sized up to span the full
# column width (tight handles/spacing let the text fill it), which is the
# largest a 3-entry one-row legend can be at this width.
fig.legend(handles=handles, frameon=False, fontsize=28, loc="upper center",
           bbox_to_anchor=(0.5, 1.05), ncol=3, handlelength=1.0,
           handletextpad=0.35, columnspacing=0.9)
fig.tight_layout(rect=(0, 0, 1, 0.9))
OUT = Path("plots/card_game/sp_xp_ablation_bars.png")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT.resolve())
