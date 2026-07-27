"""High-entropy curve on the 0.1-interval grid (0.05..0.75) — separate export.

Same house style as plot_high_entropy.py (render() is reused); reads the grid01
data file and writes to a NEW output so the original 0.05-step figure survives.
Caveats baked into the data file: the 0.35 greedy row is the worst-12 subset of
the 16-seed pool (per Giulia 2026-07-25); 0.65 is a 4-seed greedy prelim (no
sampled row yet); 0.75 has no data and stays a reserved tick.

Usage:
    uv run --no-project --with matplotlib --with numpy \\
        python evaluation/overcooked_v2/plot_high_entropy_grid01.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plot_high_entropy import render

DATA = Path("results/overcooked_v2/high_entropy_curve_grid01.json")
OUT = Path("plots/overcooked_v2/high_entropy_curve_ocv2_grid01.png")

GRID = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65]


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"missing {DATA}")
    raw = json.loads(DATA.read_text())
    alphas = [a for a in GRID if raw.get(f"{a:g}")]
    # Upper right is data-free on this grid (every curve decays past 0.35); lower
    # right would cover the sampled tails at 0.45-0.55.
    render(alphas, np.array(alphas), raw, grid=GRID, out=OUT, rotation=0,
           legend_loc="upper right")


if __name__ == "__main__":
    main()
