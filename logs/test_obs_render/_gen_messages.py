# /// script
# requires-python = ">=3.11"
# dependencies = ["jax", "numpy", "pillow"]
# ///
"""Standalone visualisation of message dots in the per-agent obs.

Mimics the env's `_make_obs` directly using the rendering primitives so we
don't need the full `make_env` stack. Saves PNGs to the same folder.
"""
from pathlib import Path
import importlib.util
from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
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

out_dir = Path(__file__).resolve().parent / "messages"
out_dir.mkdir(exist_ok=True)
for f in out_dir.glob("*.png"):
    f.unlink()

# Fix the card permutation so dot-position math is easy to check by eye.
card_perm = jnp.array([4, 0, 2, 3, 1])  # cards at positions 0..4

def make_obs(agent_idx, card_perm, step_count, partner_msg):
    img = render_card_game_minimal(card_perm, step_count + 1)
    img_np = np.array(img)
    if int(partner_msg) >= 0:
        msg_pos = int(np.argmin(np.abs(np.array(card_perm) - int(partner_msg))))
        dot_size = 2
        card_x = 1 + msg_pos * (CARD_RECT_W + 2)
        dot_y = int(CARD_RECT_Y) + (CARD_RECT_H - dot_size) // 2
        dot_x = card_x + (CARD_RECT_W - dot_size) // 2
        img_np[dot_y:dot_y + dot_size, dot_x:dot_x + dot_size] = np.array(WHITE_COLOR, dtype=np.uint8)
    return img_np

def upscale(img, factor=32):
    return np.array(
        Image.fromarray(img).resize(
            (img.shape[1] * factor, img.shape[0] * factor), Image.NEAREST
        )
    )

# Per-agent obs at every step. Messages emitted at step t become visible
# in obs_{t+1}, mirroring the env semantics.
deliberation_actions = [
    (0, 4),
    (2, 2),
    (3, 1),
    (4, 0),
    (1, 3),
    (2, 4),
    (0, 0),
]

prev_msgs = (-1, -1)
states = [SimpleNamespace(
    card_permutation=card_perm,
    step_count=jnp.int32(0),
    agent_choices=jnp.array([-1, -1], dtype=jnp.int32),
    messages=jnp.array(prev_msgs, dtype=jnp.int32),
    target_color=jnp.int32(-1),
)]

# Save reset obs.
for aidx in (0, 1):
    obs = make_obs(aidx, card_perm, 0, prev_msgs[1 - aidx])
    Image.fromarray(upscale(obs)).save(out_dir / f"obs_a{aidx}_step00.png")

for t, (a0, a1) in enumerate(deliberation_actions, start=1):
    msgs = (a0, a1)
    states.append(SimpleNamespace(
        card_permutation=card_perm,
        step_count=jnp.int32(t),
        agent_choices=jnp.array([-1, -1], dtype=jnp.int32),
        messages=jnp.array(msgs, dtype=jnp.int32),
        target_color=jnp.int32(-1),
    ))
    for aidx in (0, 1):
        obs = make_obs(aidx, card_perm, t, msgs[1 - aidx])
        Image.fromarray(upscale(obs)).save(out_dir / f"obs_a{aidx}_step{t:02d}.png")

# Inject decision picks into the last state for the eval composite.
decision = (2, 2)
last = states[-1]
states[-1] = SimpleNamespace(
    card_permutation=last.card_permutation,
    step_count=last.step_count,
    agent_choices=jnp.array(decision, dtype=jnp.int32),
    messages=last.messages,
    target_color=last.target_color,
)

frames = render_card_game_eval_frames(states, scale=32)
for t, f in enumerate(frames):
    Image.fromarray(f).save(out_dir / f"composite_step{t:02d}.png")

print(f"saved {2 * len(states)} per-agent obs PNGs")
print(f"saved {len(frames)} composite PNGs")
print(f"dir: {out_dir}")
print(f"card_perm: {np.array(card_perm).tolist()}")
print("expected dot column for partner action a -> index where perm == a")
for a in range(5):
    pos = int(np.where(np.array(card_perm) == a)[0][0])
    print(f"  partner sends {a} -> dot at column {pos}")
