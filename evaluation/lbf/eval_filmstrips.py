"""Combined-attention filmstrip for LBF Other-Play / MATE checkpoints.

Renders ONE single-row strip per checkpoint (e.g. lbf_OP.png, lbf_MATE.png). Each
cell is the real LBF board (Jumanji viewer render) with the team's combined spatial
attention overlaid as blocky inferno squares: agent 0's and agent 1's maps are each
un-mirrored into the ground-truth frame (so an Other-Play run's per-agent mirror
views land on the true board), then fused per cell. `--combine sum` (default) adds
them so cells where BOTH agents attend read brightest — a union that also lights
cells only one attends. `--combine intersection` takes the element-wise min so ONLY
cells both agents attend light up — pure agreement on an apple; it saves to
lbf_<label>_intersection.png so it never overwrites the sum strip.

The six columns show two apple-collection events, three frames each (approach ->
converge -> grasp), auto-selected around real pickups (an apple's `eaten` flag
flipping). A pixel-art timestep counter is stamped top-left of every cell and a
vertical inferno colorbar sits on the left; the colour scale is shared across the
checkpoints so the strips are directly comparable.

Reuses the rollout + OP-reframe + board-render path behind the per-checkpoint
attention videos (`run_episode_with_states`, `read_op_elems`,
`_render_lbf_eval_frames`).

Usage (on the GPU box; JAX cannot run on macOS):
    ./run_gpu.sh 0 evaluation.lbf.eval_filmstrips \\
        --checkpoints <op_run>/saved_train_run <mate_run>/saved_train_run \\
        --labels OP MATE --seed 9 --rng-seed 1 --output-dir plots/lbf/filmstrips

AAAI-2027 paper figure (figures/LBF/lbf_sidebyside.png) — verified byte-identical.
Left block is MATE, right block is OP-only, and the two use DIFFERENT seed pairs, so
--row-seeds is required; shared vmax lands on 0.500. Add --light for the white-board
version. Both blocks are cross-play, despite the right one being seed-3's:

    MATE=results/lbf-image-12x12-8food-50steps/ja_ippo/lbf50_opja_12s_3M/2026-06-24_21-59-54/saved_train_run
    OP=results/lbf-image-12x12-8food-50steps/ja_ippo/lbf50_oponly_12s_3M/2026-06-25_05-38-22/saved_train_run
    ./run_gpu.sh 0 evaluation.lbf.eval_filmstrips \\
        --checkpoints $MATE $OP --labels MATE OP --row-seeds 4:8 3:8 \\
        --frames 0:9,0:11,0:13 3:11,3:13,3:15 \\
        --cmap coolwarm --no-grid --cbar-side right --stack \\
        --output-dir plots/lbf/final
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
from jumanji.environments.routing.lbf.viewer import LevelBasedForagingViewer
from matplotlib import colormaps
from PIL import Image
from typing_extensions import override

from agents.initialize_agents import initialize_ja_image_agent
from agents.lbf.ja_lbf_attention import food_state_from_log_state
from agents.lbf.op_equivariance import read_op_elems
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import PIXEL_DIGITS, _stamp_label_np
from envs.log_wrapper import LogWrapper
from evaluation.run_xp_seeds import _load_hydra_config, _select_xp_params
from evaluation.vis_episodes import _apply_v4, run_episode_with_states
from marl.eval_lbf import _lbf_grid_size, _render_lbf_eval_frames, _to_jumanji_state

CMAP_NAME = "inferno"
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRID_GREY = (0.80, 0.80, 0.80)
ALPHA_MAX = 0.9   # opacity at a=1; shared by the overlays and the as-drawn colorbar


def _feed_dims(algo_cfg: dict, env) -> tuple[int, int, int, int] | None:
    """(img_h, img_w, feat_h, feat_w) for FEED_OTHER_ATTN runs, else None.

    Mirrors the block in `run_xp_seeds.run_xp_from_params` so MATE's policy
    receives the 4-channel obs (RGB + partner attention) it was trained on.
    """
    if not algo_cfg.get("FEED_OTHER_ATTN", False):
        return None
    from agents.initialize_agents import _get_image_dims
    from agents.ja_actor_critic import _compute_resnet_output_dims

    img_h, img_w, _ = _get_image_dims(env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=algo_cfg.get("CONV_STRIDE", 2),
        kernel_size=algo_cfg.get("CONV_KERNEL_SIZE", 3),
        padding=algo_cfg.get("CONV_PADDING", "SAME"),
        num_blocks=algo_cfg.get("CONV_NUM_BLOCKS", 4),
    )
    return img_h, img_w, feat_h, feat_w


def _to_2d(attn) -> np.ndarray:
    """Squeeze a raw attention map to (feat_h, feat_w), head-averaging if needed."""
    a = np.asarray(attn).squeeze()
    while a.ndim > 2:
        a = a.mean(axis=-1)
    return a


def _combine(a0: np.ndarray, a1: np.ndarray, mode: str) -> np.ndarray:
    """Fuse the two agents' GT-frame attention into the team map drawn per cell.

    'sum' lights any cell either agent attends (brightest where both); 'intersection'
    is the fuzzy-set min, lighting ONLY cells both agents attend — pure agreement.
    """
    if mode == "sum":
        return a0 + a1
    if mode == "intersection":
        return np.minimum(a0, a1)
    raise ValueError(f"unknown combine mode: {mode}")


def _mode_tag(combine: str) -> str:
    """Filename suffix so a non-sum render never overwrites the default sum strip."""
    return "" if combine == "sum" else f"_{combine}"


def _load_condition(checkpoint: str, seed: tuple[int, int], label: str) -> dict:
    """Load one checkpoint's env/policy/params once so many rng can be rolled out
    without re-reading the checkpoint each time."""
    seed0, seed1 = seed
    hydra_cfg = _load_hydra_config(checkpoint)
    if hydra_cfg is None:
        raise ValueError(f"No .hydra/config.yaml under {Path(checkpoint).parent}")
    algo_cfg = hydra_cfg["algorithm"]
    env_kwargs = dict(algo_cfg["ENV_KWARGS"])

    base_env = make_env(algo_cfg["ENV_NAME"], env_kwargs)
    env = LogWrapper(base_env)
    inner_env = getattr(env, "_env", env)

    policy, _ = initialize_ja_image_agent(algo_cfg, env, jax.random.PRNGKey(0))

    run_data = load_train_run(checkpoint)
    all_params, params_key = _select_xp_params(run_data)
    num_seeds = jax.tree.leaves(all_params)[0].shape[0]
    if max(seed0, seed1) >= num_seeds:
        raise ValueError(f"seed {seed} out of range (run has {num_seeds} seeds)")

    feed_dims = _feed_dims(algo_cfg, env)
    max_steps = int(env_kwargs.get("time_limit", env_kwargs.get("max_steps", 50)))
    mode = "SP" if seed0 == seed1 else "XP"
    print(
        f"[lbf_filmstrip] {Path(checkpoint).parent.name}: {mode} seed0={seed0} "
        f"seed1={seed1} ({params_key}, {num_seeds} seeds), max_steps={max_steps}, "
        f"feed_other_attn={'yes' if feed_dims else 'no'}"
    )
    return {
        "inner_env": inner_env,
        "policy": policy,
        "params_0": jax.tree.map(lambda x: x[seed0], all_params),
        "params_1": jax.tree.map(lambda x: x[seed1], all_params),
        "feed_dims": feed_dims,
        "max_steps": max_steps,
        "label": label,
    }


def _rollout(loaded: dict, rng_seed: int) -> dict:
    """Roll out one episode from a pre-loaded checkpoint; returns board frames,
    combined GT-frame attention, live-apple counts, and the initial apple layout."""
    inner_env, policy = loaded["inner_env"], loaded["policy"]
    ep_states, attn_data, _acts, _msgs = run_episode_with_states(
        jax.random.PRNGKey(rng_seed), inner_env, loaded["params_0"], policy,
        loaded["params_1"], policy, loaded["max_steps"],
        collect_attention=True, feed_other_attn_dims=loaded["feed_dims"],
    )

    n = len(attn_data["agent_0"])
    op_elems = read_op_elems(ep_states[0], list(inner_env.agents))

    def _elem(name: str) -> int:
        return 0 if op_elems is None else int(np.asarray(op_elems[name]).reshape(-1)[0])

    e0, e1 = _elem("agent_0"), _elem("agent_1")
    attn_0 = [_apply_v4(_to_2d(attn_data["agent_0"][t]), e0) for t in range(n)]
    attn_1 = [_apply_v4(_to_2d(attn_data["agent_1"][t]), e1) for t in range(n)]
    # Fuse both agents' GT-frame maps; 'sum' = union, 'intersection' = both-attend.
    mode = loaded.get("combine", "sum")
    combined = [_combine(attn_0[t], attn_1[t], mode) for t in range(n)]
    light = loaded.get("light", False)
    frames = (_render_lbf_frames_rgba(inner_env, ep_states) if light
              else _render_lbf_eval_frames(inner_env, ep_states))[:n]
    alive = [int((~np.asarray(food_state_from_log_state(ep_states[t])[1])).sum())
             for t in range(n)]
    food0 = np.asarray(food_state_from_log_state(ep_states[0])[0])

    return {"frames": frames, "combined": combined, "attn_0": attn_0, "attn_1": attn_1,
            "alive": alive, "n": n, "food0": food0, "label": loaded["label"],
            "light": light}


def _rollout_condition(checkpoint: str, seed: tuple[int, int], rng_seed: int) -> dict:
    return _rollout(_load_condition(checkpoint, seed, ""), rng_seed)


def _apple_spread(food0: np.ndarray) -> int:
    """Bounding-box area (in cells) spanned by the initial apples; higher = more
    spread out across the board."""
    rows, cols = food0[:, 0], food0[:, 1]
    return int((rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1))


def _pickup_steps(alive: list[int]) -> list[int]:
    """Timesteps at which the live-apple count drops (an apple was eaten)."""
    return [t for t in range(1, len(alive)) if alive[t] < alive[t - 1]]


def _divergence_moments(cond: dict, top: int = 5) -> list[tuple[int, float, float]]:
    """Timesteps where the two agents' attention peaks land far apart (each on a
    different, well-attended cell) — i.e. divergent conventions.

    Score = feature-grid distance between the argmax cells, weighted by the weaker
    of the two peaks (so both must actually be attending). Returns the top-`top`
    (1-based timestep, distance, score) tuples, most divergent first.
    """
    scored = []
    for t, (a0, a1) in enumerate(zip(cond["attn_0"], cond["attn_1"])):
        r0, c0 = np.unravel_index(int(np.argmax(a0)), a0.shape)
        r1, c1 = np.unravel_index(int(np.argmax(a1)), a1.shape)
        dist = float(np.hypot(r0 - r1, c0 - c1))
        score = dist * min(float(a0.max()), float(a1.max()))
        scored.append((t + 1, dist, score))
    return sorted(scored, key=lambda x: -x[2])[:top]


def _event_frames(p: int, per: int, stride: int, n: int) -> list[int]:
    """`per` timesteps ending at pickup `p` (approach -> grab), strided."""
    return sorted(dict.fromkeys(
        max(0, min(n - 1, p - (per - 1 - i) * stride)) for i in range(per)))


def _select_two_events(cond: dict, per: int = 3, stride: int = 2) -> list[int]:
    """Two apple-collection events (early + late), `per` frames each.

    Each event is the strongest-attention pickup in its half of the episode, shown
    as `per` strided frames up to and including the grab. Falls back to evenly-
    spaced frames when fewer than two apples are eaten.
    """
    n = cond["n"]
    pickups = _pickup_steps(cond["alive"])

    def _runup(p: int) -> float:
        return max(float(cond["combined"][t].max())
                   for t in _event_frames(p, per, stride, n))

    if len(pickups) < 2:
        idxs = np.linspace(0, n - 1, per * 2).round().astype(int)
        return sorted(dict.fromkeys(int(i) for i in idxs))

    half = n // 2
    early = [p for p in pickups if p < half]
    late = [p for p in pickups if p >= half]
    if early and late:
        e, l = max(early, key=_runup), max(late, key=_runup)
    else:
        ranked = sorted(pickups, key=_runup, reverse=True)
        e = ranked[0]
        l = next((p for p in ranked[1:] if abs(p - e) >= per * stride), ranked[1])
        e, l = sorted((e, l))
    return _event_frames(e, per, stride, n) + _event_frames(l, per, stride, n)


def _overlay_blocky(frame: np.ndarray, attn: np.ndarray, vmax: float,
                    cmap_name: str = CMAP_NAME, cmap_lo: float = 0.0,
                    cmap_hi: float = 1.0, alpha: float = ALPHA_MAX,
                    gamma: float = 0.6) -> np.ndarray:
    """Additive colormap glow of summed attention over the LBF board.

    The board keeps full brightness (vivid apples/agents, crisp white grid); the
    attention is ADDED as translucent blocky squares (nearest-neighbour keeps the
    feature cells crisp) so it never darkens the board — near-zero attention adds
    nothing, peaks glow at the colormap top. `cmap_lo`/`cmap_hi` select the slice of
    the colormap that [0, 1] spans: raising `cmap_lo` truncates off a dark bottom so
    low attention reads as a visible colour on black instead of near-black purple,
    and lowering `cmap_hi` trades the deep top end for brighter mid-tones. Opacity
    still tracks the raw weight so empty cells stay dark.
    """
    a = np.clip(attn / max(vmax, 1e-8), 0.0, 1.0)
    h_px, w_px = frame.shape[:2]
    a_full = np.asarray(Image.fromarray(a.astype(np.float32)).resize(
        (w_px, h_px), resample=Image.NEAREST))
    heat = colormaps[cmap_name](cmap_lo + a_full * (cmap_hi - cmap_lo))[..., :3]
    glow = (alpha * a_full ** gamma)[..., None] * heat * 255.0
    return np.clip(frame.astype(np.float32) + glow, 0, 255).astype(np.uint8)


@lru_cache(maxsize=1)
def _black_agent_sprite() -> np.ndarray:
    """Jumanji's agent icon — a pure-white silhouette plus alpha — recoloured black."""
    import pkg_resources
    from matplotlib import pyplot as plt

    path = pkg_resources.resource_filename(
        "jumanji", "environments/routing/lbf/imgs/agent.png")
    img = plt.imread(path).copy()
    img[..., :3] = 0.0
    return img


class _LightLBFViewer(LevelBasedForagingViewer):
    """LBF board drawn on a TRANSPARENT canvas with BLACK agents, for light strips.

    Jumanji hard-codes a black board and a white agent sprite, neither of which
    survives a white background. Rendering the board transparent lets the caller
    paint the attention heat behind it, so apples and agents composite over the heat
    exactly as rendered instead of being washed out by an overlay drawn on top.
    The level badges are left alone (they read their dark fill from
    `constants._GRID_COLOR`), so apples stay pixel-identical to the dark strips.
    """

    def __init__(self, grid_size: int) -> None:
        # Distinct figure name: `_get_fig_ax` caches figures by name, so sharing the
        # default would hand a transparent figure to the dark renderer.
        super().__init__(grid_size=grid_size, name="LevelBasedForagingLight",
                         render_mode="rgb_array")

    @override
    def _get_fig_ax(self):
        fig, ax = super()._get_fig_ax()
        fig.patch.set_alpha(0.0)
        return fig, ax

    @override
    def _draw_grid(self, ax) -> None:
        # Jumanji's lattice is opaque white, invisible on a white board. Recolour the
        # LineCollection it just added rather than re-deriving the geometry. The
        # lattice spans the board's outer boundary too, so it doubles as the frame's
        # border, separating neighbouring cells across the strip's white gaps.
        super()._draw_grid(ax)
        ax.collections[-1].set_color(GRID_GREY)

    @override
    def _draw_agents(self, agents, ax) -> None:
        # Mirrors jumanji's `_draw_agents`, swapping in the black sprite.
        from jumanji.tree_utils import tree_slice
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage

        for i in range(len(agents.level)):
            agent = tree_slice(agents, i)
            cell_center = self._entity_position(agent)
            imagebox = OffsetImage(_black_agent_sprite(),
                                   zoom=self.icon_size / self.grid_size)
            ax.add_artist(AnnotationBbox(imagebox, cell_center, frameon=False,
                                         zorder=0))
            self.draw_badge(agent.level, cell_center, ax)


def _render_lbf_frames_rgba(inner_env, ep_states) -> list[np.ndarray]:
    """Board frames as RGBA with a transparent background and black agents."""
    import matplotlib
    matplotlib.use("Agg")

    viewer = _LightLBFViewer(grid_size=_lbf_grid_size(inner_env))
    # `render` hands back a view onto the reused canvas buffer, so copy per frame.
    frames = [np.asarray(viewer.render(_to_jumanji_state(s))).copy()
              for s in ep_states]
    viewer.close()
    return frames


def _overlay_light(board_rgba: np.ndarray, attn: np.ndarray, vmax: float,
                   cmap_name: str = CMAP_NAME, cmap_lo: float = 0.0,
                   cmap_hi: float = 1.0, alpha: float = ALPHA_MAX,
                   gamma: float = 0.6) -> np.ndarray:
    """Attention heat on a white board, with the board content composited on top.

    The dark strips ADD the heat to a black board; on white, adding is a no-op, so
    the heat is blended into the white background instead and the transparent-
    background board is laid over it — apples and agents therefore stay exactly as
    rendered rather than being tinted by the overlay. Same colormap and same opacity
    ramp as `_overlay_blocky`, so unattended cells stay white and peaks reach the
    colormap top.
    """
    a = np.clip(attn / max(vmax, 1e-8), 0.0, 1.0)
    h_px, w_px = board_rgba.shape[:2]
    a_full = np.asarray(Image.fromarray(a.astype(np.float32)).resize(
        (w_px, h_px), resample=Image.NEAREST))
    heat = colormaps[cmap_name](cmap_lo + a_full * (cmap_hi - cmap_lo))[..., :3] * 255.0
    w = (alpha * a_full ** gamma)[..., None]
    background = (1.0 - w) * 255.0 + w * heat
    board = board_rgba[..., :3].astype(np.float32)
    board_a = board_rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(board * board_a + background * (1.0 - board_a),
                   0, 255).astype(np.uint8)


def _strip_grid(frame: np.ndarray) -> np.ndarray:
    """Blacken the LBF board's white grid lattice, leaving apples/agents intact.

    Grid lines are the only near-white pixels that span most of a full row/column;
    apples and agents are localised, so their rows/columns stay well below the
    coverage threshold and are untouched.
    """
    white = frame.mean(axis=2) > 200
    grid_rows = white.mean(axis=1) > 0.4
    grid_cols = white.mean(axis=0) > 0.4
    out = frame.copy()
    out[grid_rows, :] = 0
    out[:, grid_cols] = 0
    return out


def _stamp_timestep(cell: np.ndarray, step_1indexed: int,
                    color: tuple[int, int, int] = WHITE) -> None:
    """Pixel-art 2-digit timestep counter, top-left (card-game glyphs)."""
    scale = max(3, cell.shape[1] // 56)
    tens, ones = (step_1indexed // 10) % 10, step_1indexed % 10
    _stamp_label_np(
        cell, (np.asarray(PIXEL_DIGITS[tens]), np.asarray(PIXEL_DIGITS[ones])),
        y=2, x=2, color=color, scale=scale,
    )


def _render_colorbar(cmap_name: str, cmap_lo: float, cmap_hi: float,
                     height_px: int, as_drawn: bool = False, gamma: float = 0.6,
                     alpha: float = ALPHA_MAX, light: bool = False) -> np.ndarray:
    """Vertical colorbar for the colormap slice [cmap_lo, cmap_hi], labelled 0..1.

    By default the bar shows `heat(a)` at full intensity, which is NOT what the board
    draws: cells are drawn at opacity `alpha*a**gamma`, so only a=1 matches and the
    bar's bottom end advertises a colour that never appears (an empty cell is the bare
    board, not `heat(0)`). `as_drawn=True` bakes that opacity ramp in against the
    board's background, so every swatch is the colour a cell of that value really gets
    — the legend then spans the full range actually used, with the ticks unchanged.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

    from matplotlib.colorbar import ColorbarBase
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    base = colormaps[cmap_name]
    a = np.linspace(0.0, 1.0, 256)
    cols = base(cmap_lo + a * (cmap_hi - cmap_lo))[:, :3]
    if as_drawn:
        w = (alpha * a ** gamma)[:, None]
        # Mirror the overlays: additive over a black board, alpha-blend over white.
        cols = (1.0 - w) + w * cols if light else w * cols
    trunc = LinearSegmentedColormap.from_list(f"{cmap_name}_t", cols)
    dpi = 100
    fig, ax = plt.subplots(figsize=(0.9, height_px / dpi), dpi=dpi)
    ColorbarBase(ax, cmap=trunc, norm=Normalize(0.0, 1.0), orientation="vertical",
                 ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=max(16, round(height_px * 0.09)))
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img = img.resize((max(1, img.width * height_px // img.height), height_px),
                     Image.LANCZOS)
    return np.array(img)


def _block_font(size: int):
    """A bold truetype font (matplotlib's bundled DejaVuSans-Bold), else default."""
    from PIL import ImageFont
    try:
        import matplotlib
        path = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf")
        return ImageFont.truetype(path, size)
    except (OSError, ImportError):
        return ImageFont.load_default()


def _assemble_row(cells: list[np.ndarray], per: int, cbar: np.ndarray | None,
                  out_path: Path, cbar_side: str = "left",
                  block_labels: list[str] | None = None) -> None:
    """Stitch pre-rendered cells into one row: colorbar on `cbar_side` (skipped when
    `cbar is None`), then cells with a wider gap after every `per`-th cell (separating
    collection events). When `block_labels` is given (one per `per`-cell block), a
    caption strip is added below the cells with each label centred under its block."""
    from PIL import ImageDraw

    cell_h, cell_w = cells[0].shape[:2]
    pad = max(4, cell_w // 60)
    group_gap = pad * 7
    cbar_gap = pad * 4
    cbar_w = cbar.shape[1] if cbar is not None else 0

    start_x = (cbar_w + cbar_gap) if (cbar is not None and cbar_side == "left") else 0
    xs, x = [], start_x
    for i in range(len(cells)):
        xs.append(x)
        x += cell_w + (group_gap if (i + 1) % per == 0 else pad)
    cells_end = xs[-1] + cell_w

    if cbar is None:
        total_w, cbar_x = cells_end, 0
    elif cbar_side == "left":
        total_w, cbar_x = cells_end, 0
    else:
        total_w, cbar_x = cells_end + cbar_gap + cbar_w, cells_end + cbar_gap

    # Report block centres as a fraction of total width, for aligning LaTeX
    # subcaptions under each block on the combined image.
    n_blocks = (len(cells) + per - 1) // per
    if n_blocks > 1:
        centres = [round(((xs[b * per] + xs[min(b * per + per - 1, len(cells) - 1)]
                           + cell_w) / 2) / total_w, 3) for b in range(n_blocks)]
        print(f"[lbf_filmstrip]   block x-centres (fraction of width): {centres}")

    label_h = int(cell_h * 0.16) if block_labels else 0
    final = np.full((cell_h + label_h, total_w, 3), np.array(WHITE, dtype=np.uint8),
                    dtype=np.uint8)
    if cbar is not None:
        final[:cell_h, cbar_x:cbar_x + cbar_w] = cbar
    for i, cell in enumerate(cells):
        final[:cell_h, xs[i]:xs[i] + cell_w] = cell

    if block_labels:
        img = Image.fromarray(final)
        draw = ImageDraw.Draw(img)
        font = _block_font(int(label_h * 0.55))
        cy = cell_h + label_h // 2
        for b, lab in enumerate(block_labels):
            c0, c1 = b * per, min(b * per + per - 1, len(cells) - 1)
            cx = (xs[c0] + xs[c1] + cell_w) // 2
            bb = draw.textbbox((0, 0), lab, font=font)
            draw.text((cx - (bb[2] - bb[0]) // 2, cy - (bb[3] - bb[1]) // 2 - bb[1]),
                      lab, fill=(0, 0, 0), font=font)
        final = np.array(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final).save(out_path)
    print(f"[lbf_filmstrip] saved {out_path}  ({total_w}x{final.shape[0]} px)")


def _render_cell(cond: dict, t: int, vmax: float, cmap_name: str,
                 cmap_lo: float, cmap_hi: float = 1.0,
                 gamma: float = 0.6) -> np.ndarray:
    """One overlaid + timestep-stamped board cell."""
    frame = cond["frames"][t]
    if cond.get("light"):
        cell = _overlay_light(frame, cond["combined"][t], vmax,
                              cmap_name=cmap_name, cmap_lo=cmap_lo,
                              cmap_hi=cmap_hi, gamma=gamma)
        _stamp_timestep(cell, t + 1, BLACK)
        return cell
    if cond.get("no_grid"):
        frame = _strip_grid(frame)
    cell = _overlay_blocky(frame, cond["combined"][t], vmax, cmap_name=cmap_name,
                           cmap_lo=cmap_lo, cmap_hi=cmap_hi, gamma=gamma).copy()
    _stamp_timestep(cell, t + 1)
    return cell


def _build_single_row(cond: dict, cols: list[int], vmax: float, per: int,
                      out_path: Path, cmap_name: str = CMAP_NAME,
                      cmap_lo: float = 0.0, cmap_hi: float = 1.0,
                      cbar_side: str = "left", gamma: float = 0.6,
                      as_drawn: bool = False) -> None:
    """One-row strip from a single episode: colorbar + board cells."""
    cells = [_render_cell(cond, t, vmax, cmap_name, cmap_lo, cmap_hi, gamma)
             for t in cols]
    _assemble_row(cells, per,
                  _render_colorbar(cmap_name, cmap_lo, cmap_hi, cells[0].shape[0],
                                   as_drawn=as_drawn, gamma=gamma,
                                   light=bool(cond.get("light"))),
                  out_path, cbar_side)


def _build_video(cond: dict, vmax: float, out_path: Path, fps: int,
                 cmap_name: str = CMAP_NAME, cmap_lo: float = 0.0,
                 cmap_hi: float = 1.0, gamma: float = 0.6) -> None:
    """Full-episode mp4 of the summed attention (same additive style as the strip),
    with a pixel-art timestep counter — for scrubbing to pick frames to extract."""
    from moviepy import ImageSequenceClip

    frames = [_render_cell(cond, t, vmax, cmap_name, cmap_lo, cmap_hi, gamma)
              for t in range(cond["n"])]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ImageSequenceClip(frames, fps=fps).write_videofile(
        str(out_path), fps=fps, codec="libx264", audio=False,
        preset="ultrafast", bitrate="6000k")
    print(f"[lbf_filmstrip] saved {out_path}  ({cond['n']} frames @ {fps}fps)")


def _parse_cols(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_frames(spec: str) -> list[tuple[int, int]]:
    """Parse 'rng:t,rng:t,...' (t is the 1-based timestep shown on the video) into
    a list of (rng, 0-based timestep) pairs."""
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        rng_s, t_s = tok.split(":")
        out.append((int(rng_s), int(t_s) - 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="One saved_train_run dir per output strip.")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="One label per checkpoint; output is lbf_<label>.png.")
    parser.add_argument("--seed", type=int, default=9,
                        help="Agent-0 seed index (healthy OP seed).")
    parser.add_argument("--seed2", type=int, default=None,
                        help="Agent-1 seed index; omit for self-play.")
    parser.add_argument("--frames-per-event", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1,
                        help="Steps between the frames within an event (1 = "
                             "consecutive frames).")
    parser.add_argument("--rng-seed", type=int, default=1)
    parser.add_argument("--cols", nargs="+", default=None,
                        help="Explicit 0-based timestep indices per checkpoint, one "
                             "comma-list each. Overrides the auto event selection.")
    parser.add_argument("--norm", choices=("shared", "strip"), default="shared",
                        help="'shared' = one colour scale across both strips "
                             "(comparable brightness; OP genuinely dimmer); 'strip' "
                             "= each strip to its own peak (shows OP's scatter).")
    parser.add_argument("--scan", type=int, default=0,
                        help="Diagnostic: roll out rng-seeds 0..N-1, print pickup "
                             "steps + attention peak per checkpoint, and exit.")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Batch: render rng-seeds 0..N-1 into <out>/episodes/ as "
                             "lbf_<label>_ep<rng>.png (for browsing), ranked by "
                             "apple-spread. Auto event selection per episode.")
    parser.add_argument("--rng-list", default=None,
                        help="Explicit comma-list of rng seeds to render (overrides "
                             "--episodes / --rng-seed).")
    parser.add_argument("--video", action="store_true",
                        help="Render the full episode as an mp4 (for scrubbing to "
                             "pick frames) instead of the strip.")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--frames", nargs="+", default=None,
                        help="Hand-picked frames per checkpoint as 'rng:t,rng:t,...' "
                             "(t = 1-based timestep shown on the video). Lets one "
                             "strip mix frames from different episodes. Overrides "
                             "auto/--cols/--rng.")
    parser.add_argument("--combine", choices=("sum", "intersection"), default="sum",
                        help="How to fuse the two agents' attention per cell: 'sum' "
                             "(union; the current default) or 'intersection' (min; "
                             "only cells BOTH agents attend). Intersection saves to "
                             "lbf_<label>_intersection.png so it never overwrites sum.")
    parser.add_argument("--cmap", default=CMAP_NAME,
                        help="Attention colormap (e.g. inferno, turbo, plasma).")
    parser.add_argument("--cmap-lo", type=float, default=0.0,
                        help="Truncate off the colormap's dark bottom (0..1); e.g. "
                             "0.35 makes low attention a visible orange on black "
                             "instead of near-invisible purple.")
    parser.add_argument("--cmap-hi", type=float, default=1.0,
                        help="Upper end of the colormap slice that [0,1] spans (0..1). "
                             "Lowering it drops the colormap's deep top end for "
                             "brighter mid-tones; e.g. --cmap-lo 0.2 --cmap-hi 0.7 "
                             "zooms coolwarm into its light blue -> salmon middle.")
    parser.add_argument("--cbar-as-drawn", action="store_true",
                        help="Draw the colorbar as the colours cells ACTUALLY get, i.e. "
                             "with the opacity ramp baked in over the board background. "
                             "Without it the bar shows full-intensity colours that only "
                             "match the board at a=1, and its low end advertises a "
                             "colour no cell ever has (an empty cell is bare board). "
                             "Ticks are unchanged either way.")
    parser.add_argument("--gamma", type=float, default=0.6,
                        help="Opacity ramp exponent: a cell is drawn at opacity "
                             "0.9*a**gamma. THIS, not the colormap, sets how visible "
                             "weak attention is — at gamma 0.6 a cell at a=0.1 caps at "
                             "58/255 however bright its colour. Lower (0.35, 0.25) "
                             "lifts the diffuse cloud and leaves the peak untouched, "
                             "at the cost of hazing up near-zero cells.")
    parser.add_argument("--cbar-side", choices=("left", "right"), default="left")
    parser.add_argument("--no-grid", action="store_true",
                        help="Blacken the LBF board's white grid lattice for a clean "
                             "black background (apples/agents kept).")
    parser.add_argument("--light", action="store_true",
                        help="White board with a light-grey grid, black agents and a "
                             "black timestep counter (apples and colormap unchanged). "
                             "The attention is blended into the white background UNDER "
                             "the board rather than added on top, since adding to white "
                             "is a no-op. Ignores --no-grid.")
    parser.add_argument("--row-seeds", nargs="+", default=None,
                        help="Per-checkpoint seed pair as 's0:s1' (overrides "
                             "--seed/--seed2), so each row of a --stack figure can use "
                             "a different pairing.")
    parser.add_argument("--stack", action="store_true",
                        help="With --frames: compose all checkpoints' frames into ONE "
                             "figure, each checkpoint a side-by-side block (wider gap "
                             "between blocks) sharing one colorbar on --cbar-side. "
                             "Shared normalisation.")
    parser.add_argument("--inline-labels", action="store_true",
                        help="Bake '(a) <label>' captions into the --stack image under "
                             "each block (default off: label in LaTeX instead).")
    parser.add_argument("--separate", action="store_true",
                        help="With --stack: emit one image per checkpoint block (no "
                             "colorbar) plus a standalone lbf_colorbar.png, all on the "
                             "shared scale — for LaTeX subfigures with \\subcaption.")
    parser.add_argument("--cbar-attach", action="store_true",
                        help="With --separate: attach the shared colorbar to the LAST "
                             "block (on --cbar-side) instead of emitting it standalone.")
    parser.add_argument("--output-dir", default="plots/lbf/filmstrips")
    args = parser.parse_args()

    if len(args.labels) != len(args.checkpoints):
        parser.error("--labels must have one entry per --checkpoints")
    manual = [_parse_cols(s) for s in args.cols] if args.cols else None
    if manual is not None and len(manual) != len(args.checkpoints):
        parser.error("--cols must have one comma-list per --checkpoints")

    seed_pair = (args.seed, args.seed if args.seed2 is None else args.seed2)
    per = args.frames_per_event
    if args.row_seeds:
        if len(args.row_seeds) != len(args.checkpoints):
            parser.error("--row-seeds must have one 's0:s1' per --checkpoints")
        pairs = [tuple(int(x) for x in s.split(":")) for s in args.row_seeds]
    else:
        pairs = [seed_pair] * len(args.checkpoints)
    loaded = [_load_condition(ckpt, sp, label)
              for ckpt, sp, label in zip(args.checkpoints, pairs, args.labels)]
    for L in loaded:
        L["combine"] = args.combine
        L["light"] = args.light
    tag = _mode_tag(args.combine)

    if args.scan:
        for rs in range(args.scan):
            for L in loaded:
                cond = _rollout(L, rs)
                peak = max(float(m.max()) for m in cond["combined"])
                print(f"[scan] rng={rs} {L['label']}: n={cond['n']} "
                      f"pickups={_pickup_steps(cond['alive'])} peak={peak:.3f} "
                      f"spread={_apple_spread(cond['food0'])}")
        return

    cmap_name, cmap_lo, cmap_hi = args.cmap, args.cmap_lo, args.cmap_hi
    gamma = args.gamma
    cbar_kw = dict(as_drawn=args.cbar_as_drawn, gamma=gamma, light=args.light)
    if not 0.0 <= cmap_lo < cmap_hi <= 1.0:
        parser.error("need 0 <= --cmap-lo < --cmap-hi <= 1")
    out_dir = Path(args.output_dir)

    # Hand-picked frames: each strip may mix frames from several episodes.
    if args.frames:
        if len(args.frames) != len(loaded):
            parser.error("--frames must have one spec per --checkpoints")
        specs = [_parse_frames(s) for s in args.frames]
        # Roll out only the episodes each checkpoint needs.
        rollouts = [
            {rng: _rollout(L, rng) for rng in {rs for rs, _ in spec}}
            for L, spec in zip(loaded, specs)
        ]
        for epmap in rollouts:
            for cond in epmap.values():
                cond["no_grid"] = args.no_grid
        shared_vmax = max(
            float(rollouts[i][rng]["combined"][t].max())
            for i, spec in enumerate(specs) for rng, t in spec
        )
        if args.stack:
            block_cells = []
            for L, spec, epmap in zip(loaded, specs, rollouts):
                cells = [_render_cell(epmap[rng], t, shared_vmax, cmap_name, cmap_lo,
                                      cmap_hi, gamma) for rng, t in spec]
                block_cells.append((L["label"], cells))
                print(f"[lbf_filmstrip]   {L['label']}: block frames "
                      f"{[(rng, t + 1) for rng, t in spec]}")
            print(f"[lbf_filmstrip]   shared vmax={shared_vmax:.3f}")
            cell_h = block_cells[0][1][0].shape[0]
            if args.separate:
                # One image per block, for LaTeX subfigures. The shared colorbar is
                # either standalone or attached to the last block (on --cbar-side).
                cbar = _render_colorbar(cmap_name, cmap_lo, cmap_hi, cell_h,
                                        **cbar_kw)
                last = len(block_cells) - 1
                for i, (label, cells) in enumerate(block_cells):
                    on_this = cbar if (args.cbar_attach and i == last) else None
                    _assemble_row(cells, per, on_this,
                                  out_dir / f"lbf_block_{label}{tag}.png",
                                  args.cbar_side)
                if not args.cbar_attach:
                    Image.fromarray(cbar).save(out_dir / "lbf_colorbar.png")
                    print(f"[lbf_filmstrip] saved {out_dir}/lbf_colorbar.png")
                return
            # Combined side-by-side: each block a group, one shared colorbar.
            all_cells = [c for _, cells in block_cells for c in cells]
            block_labels = ([f"({chr(97 + i)}) {label}"
                             for i, (label, _) in enumerate(block_cells)]
                            if args.inline_labels else None)
            _assemble_row(all_cells, per,
                          _render_colorbar(cmap_name, cmap_lo, cmap_hi, cell_h,
                                           **cbar_kw),
                          out_dir / f"lbf_sidebyside{tag}.png", args.cbar_side,
                          block_labels=block_labels)
            return
        for L, spec, epmap in zip(loaded, specs, rollouts):
            vmax = (shared_vmax if args.norm == "shared"
                    else max(float(epmap[rng]["combined"][t].max()) for rng, t in spec))
            cells = [_render_cell(epmap[rng], t, vmax, cmap_name, cmap_lo, cmap_hi,
                                  gamma) for rng, t in spec]
            print(f"[lbf_filmstrip]   {L['label']}: frames "
                  f"{[(rng, t + 1) for rng, t in spec]}  vmax={vmax:.3f}")
            _assemble_row(cells, per,
                          _render_colorbar(cmap_name, cmap_lo, cmap_hi,
                                           cells[0].shape[0], **cbar_kw),
                          out_dir / f"lbf_{L['label']}{tag}.png", args.cbar_side)
        return

    def _peak(cond, cols):
        return max(float(np.max(cond["combined"][t])) for t in cols)

    def _render_episode(rng_seed: int, out_dir: Path, suffix: str = "") -> int:
        conds = [_rollout(L, rng_seed) for L in loaded]
        for c in conds:
            c["no_grid"] = args.no_grid
        spread = _apple_spread(conds[0]["food0"])
        if args.video:
            for cond in conds:
                vmax = max(float(m.max()) for m in cond["combined"])  # whole episode
                div = _divergence_moments(cond)
                div_str = " ".join(f"t={t}(d={d:.0f})" for t, d, _ in div)
                print(f"[lbf_filmstrip]   ep{rng_seed} {cond['label']}: video "
                      f"n={cond['n']} spread={spread} vmax={vmax:.3f}")
                print(f"[lbf_filmstrip]     divergence moments: {div_str}")
                _build_video(cond, vmax, out_dir / f"lbf_{cond['label']}{tag}{suffix}.mp4",
                             args.fps, cmap_name, cmap_lo, cmap_hi, gamma)
            return spread
        cols_per = [
            manual[i] if manual else _select_two_events(c, per, args.stride)
            for i, c in enumerate(conds)
        ]
        shared_vmax = max(_peak(c, cols) for c, cols in zip(conds, cols_per))
        for cond, cols in zip(conds, cols_per):
            print(f"[lbf_filmstrip]   ep{rng_seed} {cond['label']}: spread={spread} "
                  f"pickups={_pickup_steps(cond['alive'])} cols={[t + 1 for t in cols]}")
            # 'shared' keeps brightness comparable (OP genuinely dimmer); 'strip'
            # scales each to its own peak so OP's diffuse scatter stays visible.
            vmax = _peak(cond, cols) if args.norm == "strip" else shared_vmax
            _build_single_row(cond, cols, vmax, per,
                              out_dir / f"lbf_{cond['label']}{tag}{suffix}.png",
                              cmap_name, cmap_lo, cmap_hi, args.cbar_side,
                              gamma=gamma, as_drawn=args.cbar_as_drawn)
        return spread

    if args.rng_list is not None:
        rng_list = [int(x) for x in args.rng_list.split(",") if x.strip()]
    elif args.episodes:
        rng_list = list(range(args.episodes))
    else:
        rng_list = [args.rng_seed]

    multi = len(rng_list) > 1
    sub = out_dir / "episodes" if multi else out_dir
    scored = [(rs, _render_episode(rs, sub, f"_ep{rs:02d}" if multi else ""))
              for rs in rng_list]
    if multi:
        print("\n[lbf_filmstrip] episodes ranked by apple-spread (higher = more spread):")
        for rs, sp in sorted(scored, key=lambda x: -x[1]):
            print(f"   ep{rs:02d}: spread={sp}")


if __name__ == "__main__":
    main()
