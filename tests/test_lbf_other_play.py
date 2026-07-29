"""Correctness checks for the LBF mirror Other-Play wrapper.

Validates the precomputed transform tables (pixel gather + inverse action map)
and that the mirror (V4) wrapper applies the intended geometric transform to real
LBF image observations without degenerating.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from envs import make_env
from envs.lbf.other_play import (
    MIRROR_V4,
    _FIXED_ACTIONS,
    _NUM_ACTIONS,
    LBFMirrorOtherPlayWrapper,
    _build_tables,
    _forward_action_perm,
)


def _make_env():
    return make_env(
        "lbf",
        {"obs_type": "image", "grid_size": 12, "num_food": 8,
         "max_agent_level": 3, "num_agents": 2},
    )


def test_tables_are_permutations():
    pix, act = _build_tables(7, 7, MIRROR_V4)
    for e in range(len(MIRROR_V4)):
        assert sorted(pix[e].tolist()) == list(range(7 * 7))
        assert sorted(act[e].tolist()) == list(range(_NUM_ACTIONS))


def test_action_map_is_inverse_with_fixed_points():
    for h_flip, v_flip, k in MIRROR_V4:
        fwd = _forward_action_perm(h_flip, v_flip, k)
        inv = np.argsort(fwd)
        for a in range(_NUM_ACTIONS):
            assert inv[fwd[a]] == a
        for a in _FIXED_ACTIONS:
            assert fwd[a] == a and inv[a] == a


def test_mirror_obs_transform_matches_numpy_reference():
    env = _make_env()
    op = LBFMirrorOtherPlayWrapper(env)
    obs, _ = env.reset(jax.random.PRNGKey(0))
    img = np.asarray(obs["agent_0"]).reshape(env._img_h, env._img_w, 3)
    for e, (h_flip, v_flip, k) in enumerate(MIRROR_V4):
        ref = img
        if h_flip:
            ref = np.fliplr(ref)
        if v_flip:
            ref = np.flipud(ref)
        ref = np.rot90(ref, k=-k)
        got = np.asarray(op._transform_obs(obs["agent_0"], jnp.int32(e)))
        got = got.reshape(env._img_h, env._img_w, 3)
        assert np.allclose(got, ref)


def test_both_flips_equals_rot180():
    # MIRROR_V4 element 3 = (H, V) should equal a 180-degree rotation.
    env = _make_env()
    op = LBFMirrorOtherPlayWrapper(env)
    obs, _ = env.reset(jax.random.PRNGKey(2))
    img = np.asarray(obs["agent_0"]).reshape(env._img_h, env._img_w, 3)
    both = np.asarray(op._transform_obs(obs["agent_0"], jnp.int32(3)))
    assert np.allclose(both.reshape(env._img_h, env._img_w, 3), np.rot90(img, k=2))


def test_wrappers_run_and_preserve_shapes():
    env = _make_env()
    obs_dim = env._img_h * env._img_w * 3
    op = LBFMirrorOtherPlayWrapper(env)
    key = jax.random.PRNGKey(1)
    obs, state = op.reset(key)
    assert obs["agent_0"].shape == (obs_dim,)
    assert op.get_avail_actions(state)["agent_0"].shape == (_NUM_ACTIONS,)
    for _ in range(5):
        key, sub = jax.random.split(key)
        action = {a: jnp.int32(0) for a in op.agents}  # NOOP, frame-invariant
        obs, state, reward, done, info = op.step(sub, state, action)
    assert bool(jnp.all(jnp.isfinite(obs["agent_0"])))
