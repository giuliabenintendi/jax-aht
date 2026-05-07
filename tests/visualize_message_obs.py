# /// script
# requires-python = ">=3.11"
# dependencies = ["jax", "numpy", "pillow"]
# ///
"""Quick standalone visualisation of message dots in card-game obs and the
stacked per-agent eval composite. Doesn't touch `make_env` so it works on a
plain Python environment (no jaxmarl / CUDA).

Usage:
    uv run --script tests/visualize_message_obs.py

Outputs go to `logs/message_obs_visual_quick/`:
    obs_a0_step{NN}.png      — what A0 sees (white-only, no agent colours)
    obs_a1_step{NN}.png      — what A1 sees
    composite_step{NN}.png   — stacked colour-coded eval-video frame

The pytest counterpart `tests/test_message_obs_visual.py` runs the real env
and asserts dot positions; this script is faster and is just for eye-balling.
"""
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
_rendering_path = ROOT / "envs" / "card_game" / "rendering.py"
_spec = importlib.util.spec_from_file_location("card_game_rendering", _rendering_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

render_card_game_minimal = _mod.render_card_game_minimal
render_card_game_eval_frames = _mod.render_card_game_eval_frames
CARD_RECT_W = _mod.CARD_RECT_W
CARD_RECT_H = _mod.CARD_RECT_H
CARD_RECT_Y = _mod.CARD_RECT_Y
WHITE_COLOR = _mod.WHITE_COLOR

OUT_DIR = ROOT / "logs" / "message_obs_visual_quick"
OUT_DIR.mkdir(parents=True, exist_ok=True)
for f in OUT_DIR.glob("*.png"):
    f.unlink()

# Fixed perm so the dot-column math is easy to check by eye.
CARD_PERM = jnp.array([4, 0, 2, 3, 1])
SCALE = 32


def make_obs(agent_idx, step_count, partner_msg):
    img = render_card_game_minimal(CARD_PERM, step_count + 1)
    img_np = np.array(img)
    if int(partner_msg) >= 0:
        msg_pos = int(np.argmin(np.abs(np.array(CARD_PERM) - int(partner_msg))))
        dot_size = 2
        card_x = 1 + msg_pos * (CARD_RECT_W + 2)
        dot_y = int(CARD_RECT_Y) + (CARD_RECT_H - dot_size) // 2
        dot_x = card_x + (CARD_RECT_W - dot_size) // 2
        img_np[dot_y:dot_y + dot_size, dot_x:dot_x + dot_size] = np.array(
            WHITE_COLOR, dtype=np.uint8,
        )
    return img_np


def upscale(img):
    return np.array(
        Image.fromarray(img).resize(
            (img.shape[1] * SCALE, img.shape[0] * SCALE), Image.NEAREST,
        )
    )


# Scripted episode: pairs of (a0_msg, a1_msg) for 7 deliberation steps,
# then a final decision pick.
DELIBERATION = [
    (0, 4),  # step 1 -> A0 sees dot at column where colour 4 lives,
             #            A1 sees dot at column where colour 0 lives
    (2, 2),  # step 2 -> both messages match (green), both see dot on green
    (3, 1),
    (4, 0),
    (1, 3),
    (2, 4),
    (0, 0),
]
DECISION = (2, 2)  # both pick green at decision step


def main():
    states = []
    prev_msgs = (-1, -1)
    states.append(SimpleNamespace(
        card_permutation=CARD_PERM,
        step_count=jnp.int32(0),
        agent_choices=jnp.array([-1, -1], dtype=jnp.int32),
        messages=jnp.array(prev_msgs, dtype=jnp.int32),
        target_color=jnp.int32(-1),
    ))
    for aidx in (0, 1):
        Image.fromarray(upscale(make_obs(aidx, 0, prev_msgs[1 - aidx]))).save(
            OUT_DIR / f"obs_a{aidx}_step00.png"
        )

    for t, (a0, a1) in enumerate(DELIBERATION, start=1):
        msgs = (a0, a1)
        states.append(SimpleNamespace(
            card_permutation=CARD_PERM,
            step_count=jnp.int32(t),
            agent_choices=jnp.array([-1, -1], dtype=jnp.int32),
            messages=jnp.array(msgs, dtype=jnp.int32),
            target_color=jnp.int32(-1),
        ))
        for aidx in (0, 1):
            Image.fromarray(upscale(make_obs(aidx, t, msgs[1 - aidx]))).save(
                OUT_DIR / f"obs_a{aidx}_step{t:02d}.png"
            )

    # Splice the decision picks into the last deliberation state for the
    # eval composite (the env auto-resets after the decision step, so we
    # can't observe a "real" picks-set state otherwise).
    last = states[-1]
    states[-1] = SimpleNamespace(
        card_permutation=last.card_permutation,
        step_count=last.step_count,
        agent_choices=jnp.array(DECISION, dtype=jnp.int32),
        messages=last.messages,
        target_color=last.target_color,
    )

    frames = render_card_game_eval_frames(states, scale=SCALE)
    for t, f in enumerate(frames):
        Image.fromarray(f).save(OUT_DIR / f"composite_step{t:02d}.png")

    print(f"saved {2 * len(states)} per-agent obs PNGs and {len(frames)} composite PNGs")
    print(f"dir: {OUT_DIR}")
    print(f"card_perm: {np.array(CARD_PERM).tolist()}")
    print("expected dot column for partner action a -> index where perm == a:")
    for a in range(5):
        pos = int(np.where(np.array(CARD_PERM) == a)[0][0])
        print(f"  partner sends colour {a} -> dot at column {pos}")


if __name__ == "__main__":
    main()
    sys.exit(0)
