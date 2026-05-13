# /// script
# requires-python = ">=3.11"
# dependencies = ["jax", "numpy", "pillow"]
# ///
"""Render a single eval frame at scale=8 (clearer pixel structure for review)."""
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

render_card_game_eval_frames = _mod.render_card_game_eval_frames

out_dir = Path(__file__).resolve().parent
perm = jnp.array([0, 1, 2, 3, 4])

def make_state(step, choices=(-1, -1)):
    return SimpleNamespace(
        card_permutation=perm,
        step_count=jnp.int32(step),
        agent_choices=jnp.array(choices, dtype=jnp.int32),
        messages=jnp.full(2, -1, dtype=jnp.int32),
        target_color=jnp.int32(-1),
    )

ep_states = [
    make_state(0),
    make_state(7, choices=(2, 4)),  # different picks for clarity
]

# Generate at scale 64 for very crisp viewing
frames = render_card_game_eval_frames(ep_states, scale=64)
for i, f in enumerate(frames):
    Image.fromarray(f).save(out_dir / f"eval_frame_x64_{i}.png")
print("done", [f.shape for f in frames])
