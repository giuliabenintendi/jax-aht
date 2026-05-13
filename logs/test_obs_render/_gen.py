# /// script
# requires-python = ">=3.11"
# dependencies = ["jax", "numpy", "pillow"]
# ///
"""One-off: render a few card-game obs PNGs (no agents, with timestep)."""
from pathlib import Path
import importlib.util

import numpy as np
import jax.numpy as jnp
from PIL import Image

# Load rendering.py directly to avoid envs/__init__.py importing jumanji.
_rendering_path = (
    Path(__file__).resolve().parents[2] / "envs" / "card_game" / "rendering.py"
)
_spec = importlib.util.spec_from_file_location("card_game_rendering", _rendering_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
render_card_game_minimal = _mod.render_card_game_minimal

out_dir = Path(__file__).resolve().parent
perm = jnp.array([0, 1, 2, 3, 4])
scale = 32

steps = [0, 1, 5, 9, 10, 50, 99, 123, 999]
for t in steps:
    img = np.array(render_card_game_minimal(perm, jnp.int32(t)))
    big = Image.fromarray(img).resize(
        (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST
    )
    big.save(out_dir / f"obs_t{t:03d}_x{scale}.png")
    Image.fromarray(img).save(out_dir / f"obs_t{t:03d}_raw.png")
    print(f"saved t={t:03d}")
