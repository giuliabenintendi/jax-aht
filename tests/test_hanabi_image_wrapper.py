"""Visual + invariant tests for the Hanabi image renderer and wrapper.

Generates side-by-side filmstrips of both agents' observations across a
short rollout so the per-agent egocentric view, the hint-stripe encoding,
and the fireworks/token strip can be eyeballed.

Outputs (when run as a script or via the visual test):
  - logs/hanabi_image/filmstrip.png — 2-column grid (agent_0 | agent_1)
      one row per captured turn, with the turn index labeled on the side.
  - logs/hanabi_image/palette.png — single frame upscaled for close inspection

The pytest invariants below run without producing files: shape, dtype,
egocentric divergence, and palette purity (every coloured pixel matches
`HANABI_COLORS` or one of the documented auxiliary constants).
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from envs.hanabi.hanabi_image_wrapper import HanabiImageWrapper
from envs.hanabi.rendering import (
    BACKGROUND_COLOR,
    CARD_BACK_COLOR,
    DECK_BAR_COLOR,
    HANABI_COLORS,
    IMG_H,
    IMG_W,
    INFO_TOKEN_COLOR,
    LIFE_TOKEN_COLOR,
    RANK_HINT_COLOR,
)

_UPSCALE = 8
_TURNS_TO_CAPTURE = (0, 3, 7, 15, 30, 60)


def _make_env() -> HanabiImageWrapper:
    return HanabiImageWrapper(
        num_agents=2,
        num_colors=5,
        num_ranks=5,
        max_info_tokens=8,
        max_life_tokens=3,
        num_cards_of_rank=np.array([3, 2, 2, 2, 1]),
    )


def _flat_to_img(flat: jnp.ndarray) -> np.ndarray:
    return np.asarray((flat.reshape(IMG_H, IMG_W, 3) * 255.0).astype(jnp.uint8))


def _upscale(img: np.ndarray, factor: int = _UPSCALE) -> np.ndarray:
    return np.kron(img, np.ones((factor, factor, 1), dtype=np.uint8))


def _palette() -> np.ndarray:
    """All RGB values the renderer is allowed to produce."""
    return np.concatenate([
        np.asarray(HANABI_COLORS),
        np.asarray(CARD_BACK_COLOR)[None],
        np.asarray(INFO_TOKEN_COLOR)[None],
        np.asarray(LIFE_TOKEN_COLOR)[None],
        np.asarray(RANK_HINT_COLOR)[None],
        np.asarray(DECK_BAR_COLOR)[None],
        np.asarray(BACKGROUND_COLOR)[None],
    ])


def test_obs_shape_and_dtype():
    env = _make_env()
    obs, _ = env.reset(jax.random.PRNGKey(0))
    assert obs["agent_0"].shape == (IMG_H * IMG_W * 3,)
    assert obs["agent_0"].dtype == jnp.float32
    assert float(obs["agent_0"].min()) >= 0.0
    assert float(obs["agent_0"].max()) <= 1.0


def test_egocentric_views_differ():
    """The two agents should never see byte-identical images — partner hand
    differs and own-hand differs.
    """
    env = _make_env()
    obs, _ = env.reset(jax.random.PRNGKey(0))
    assert not jnp.array_equal(obs["agent_0"], obs["agent_1"])


def test_palette_purity_at_reset():
    """Every pixel in the freshly-reset obs must match the documented palette.

    This is the invariant the OP recolouring wrapper will depend on:
    HANABI_COLORS plus the five auxiliary constants. If this fails, OP
    recolouring would silently leave some coloured pixels in the original
    palette.
    """
    env = _make_env()
    obs, _ = env.reset(jax.random.PRNGKey(0))
    img = _flat_to_img(obs["agent_0"])
    palette = _palette()
    flat = img.reshape(-1, 3)
    matches = np.any(np.all(flat[:, None, :] == palette[None, :, :], axis=-1), axis=-1)
    if not matches.all():
        bad = flat[~matches]
        unique = np.unique(bad, axis=0)
        raise AssertionError(
            f"{(~matches).sum()} pixels outside palette; unique offenders: {unique[:5].tolist()}"
        )


def test_palette_purity_during_rollout():
    """Palette purity must hold across the game, not just at reset.

    Walks a random-legal rollout for 80 turns (covers a full episode under
    most policies) and checks every observation produced.
    """
    env = _make_env()
    key = jax.random.PRNGKey(2026)
    obs, state = env.reset(key)
    palette = _palette()

    def check(obs):
        img = _flat_to_img(obs)
        flat = img.reshape(-1, 3)
        matches = np.any(np.all(flat[:, None, :] == palette[None, :, :], axis=-1), axis=-1)
        return bool(matches.all()), flat[~matches]

    for turn in range(80):
        for agent in env.agents:
            ok, bad = check(obs[agent])
            assert ok, f"palette broken at turn {turn} for {agent}: {np.unique(bad, axis=0)[:5].tolist()}"

        legal = env.get_avail_actions(state)
        key, k_a, k_s = jax.random.split(key, 3)
        ks = jax.random.split(k_a, env.num_agents)
        actions = {
            a: jax.random.categorical(ks[i], jnp.log(legal[a] + 1e-9))
            for i, a in enumerate(env.agents)
        }
        obs, state, _, _, _ = env.step(k_s, state, actions)


def test_render_visual_filmstrip():
    """Write a per-turn filmstrip PNG. Not a true assertion — produces a
    visual artifact for human inspection.

    Output: logs/hanabi_image/filmstrip.png
    """
    out_dir = Path("logs/hanabi_image")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env()
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    captures: list[tuple[int, np.ndarray, np.ndarray, int, int]] = []
    max_turns = max(_TURNS_TO_CAPTURE) + 1

    for turn in range(max_turns):
        if turn in _TURNS_TO_CAPTURE:
            captures.append((
                turn,
                _flat_to_img(obs["agent_0"]),
                _flat_to_img(obs["agent_1"]),
                int(state.env_state.score),
                int(state.env_state.life_tokens.sum()),
            ))
        legal = env.get_avail_actions(state)
        key, k_a, k_s = jax.random.split(key, 3)
        ks = jax.random.split(k_a, env.num_agents)
        actions = {
            a: jax.random.categorical(ks[i], jnp.log(legal[a] + 1e-9))
            for i, a in enumerate(env.agents)
        }
        obs, state, _, _, _ = env.step(k_s, state, actions)

    if not captures:
        return

    sep = 4
    tile_h, tile_w = IMG_H * _UPSCALE, IMG_W * _UPSCALE
    row_h = tile_h + 2 * sep
    label_w = 80
    total_w = label_w + 2 * tile_w + 3 * sep
    total_h = row_h * len(captures)

    canvas = np.full((total_h, total_w, 3), 40, dtype=np.uint8)
    for r, (turn, a0, a1, score, lives) in enumerate(captures):
        y0 = r * row_h + sep
        # paste agent_0 obs
        canvas[y0:y0 + tile_h, label_w + sep:label_w + sep + tile_w] = _upscale(a0)
        # paste agent_1 obs
        canvas[y0:y0 + tile_h, label_w + 2 * sep + tile_w:label_w + 2 * sep + 2 * tile_w] = _upscale(a1)

    img = Image.fromarray(canvas)
    # Draw text labels with PIL
    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
        except OSError:
            font = ImageFont.load_default()
        # header above first row
        for r, (turn, _, _, score, lives) in enumerate(captures):
            y0 = r * row_h + sep + tile_h // 2 - 12
            draw.text(
                (4, y0),
                f"turn {turn}\nscore {score}\nlives {lives}",
                fill=(220, 220, 220),
                font=font,
            )
        # column headers
        draw.text((label_w + sep + 4, 2), "agent_0", fill=(220, 220, 220), font=font)
        draw.text((label_w + 2 * sep + tile_w + 4, 2), "agent_1", fill=(220, 220, 220), font=font)
    except Exception:
        pass  # labels are nice-to-have, not required

    out_path = out_dir / "filmstrip.png"
    img.save(out_path)
    print(f"wrote {out_path}  size={img.size}")


def test_render_palette_swatch():
    """Write a single upscaled frame plus a palette swatch for close inspection.

    Output: logs/hanabi_image/palette.png
    """
    out_dir = Path("logs/hanabi_image")
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _make_env()
    obs, _ = env.reset(jax.random.PRNGKey(7))
    frame = _upscale(_flat_to_img(obs["agent_0"]), factor=16)

    palette = _palette()
    swatch_h = 24
    swatch_w = 32
    swatch = np.zeros((swatch_h, swatch_w * len(palette), 3), dtype=np.uint8)
    for i, c in enumerate(palette):
        swatch[:, i * swatch_w:(i + 1) * swatch_w] = c

    sep = 8
    canvas = np.full(
        (frame.shape[0] + swatch_h + sep, max(frame.shape[1], swatch.shape[1]), 3),
        40,
        dtype=np.uint8,
    )
    canvas[:frame.shape[0], :frame.shape[1]] = frame
    canvas[frame.shape[0] + sep:frame.shape[0] + sep + swatch_h, :swatch.shape[1]] = swatch

    out_path = out_dir / "palette.png"
    Image.fromarray(canvas).save(out_path)
    print(f"wrote {out_path}  frame_shape={frame.shape}  palette_swatch={swatch.shape}")
