# /// script
# requires-python = ">=3.11"
# dependencies = ["jax", "numpy", "pillow"]
# ///
"""Smoke-test render_card_game_eval_frames + new _make_obs.

Builds a fake mini-trajectory: 3 frames (deliberation, deliberation, decision-
with-picks) and saves both the obs PNG (as the agent sees it) and the eval-
frame PNG (with A0/A1 legend).
"""
from pathlib import Path
import importlib.util
from types import SimpleNamespace

import numpy as np
import jax
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

out_dir = Path(__file__).resolve().parent
perm = jnp.array([0, 1, 2, 3, 4])

# Synthetic states (the real WrappedEnvState wraps these via .env_state, but
# render_card_game_eval_frames unwraps via _unwrap_card_game_state which walks
# .env_state. We can pass plain SimpleNamespace with the fields it reads.)
def make_state(step, choices=(-1, -1)):
    return SimpleNamespace(
        card_permutation=perm,
        step_count=jnp.int32(step),
        agent_choices=jnp.array(choices, dtype=jnp.int32),
        messages=jnp.full(2, -1, dtype=jnp.int32),
        target_color=jnp.int32(-1),
    )

ep_states = [
    make_state(0),                       # 1st obs (display 01)
    make_state(3),                       # mid-episode (display 04)
    make_state(7, choices=(2, 4)),       # decision: A0 picks green, A1 picks purple
]

frames = render_card_game_eval_frames(ep_states, scale=32)
for i, f in enumerate(frames):
    Image.fromarray(f).save(out_dir / f"eval_composite_{i}.png")
print("eval composite frames:", [f.shape for f in frames])

# Also dump the matching obs image (no choice border, no legend) for the
# decision-step state.
obs_img = np.array(render_card_game_minimal(perm, jnp.int32(8)))
Image.fromarray(obs_img).resize(
    (obs_img.shape[1] * 32, obs_img.shape[0] * 32), Image.NEAREST
).save(out_dir / "obs_decision.png")
print("done")
