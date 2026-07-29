"""Checks for the feature-resolution OP transforms used by equivariant LBF JA."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.lbf.op_equivariance import (
    feat_transform_table,
    read_op_elems,
    transform_feat_maps,
)
from envs import make_env
from envs.lbf.other_play import MIRROR_V4


def _lbf(op_mirror: bool):
    kw = {"obs_type": "image", "grid_size": 12, "num_food": 8,
          "max_agent_level": 3, "num_agents": 2}
    if op_mirror:
        kw["other_play_mirror"] = True
    return make_env("lbf", kw)


def test_feat_transforms_match_numpy_reference():
    fh = fw = 6
    table = feat_transform_table(fh, fw, MIRROR_V4)
    maps = np.arange(2 * fh * fw, dtype=np.float32).reshape(2, fh, fw)
    for e, (h_flip, v_flip, k) in enumerate(MIRROR_V4):
        got = np.asarray(transform_feat_maps(jnp.asarray(maps), jnp.array([e, e]), table))
        for a in range(2):
            ref = maps[a]
            if h_flip:
                ref = np.fliplr(ref)
            if v_flip:
                ref = np.flipud(ref)
            ref = np.rot90(ref, k=-k)
            assert np.allclose(got[a], ref)


def test_mirror_is_self_inverse():
    fh = fw = 6
    table = feat_transform_table(fh, fw, MIRROR_V4)
    maps = jnp.asarray(np.random.RandomState(0).rand(3, fh, fw).astype(np.float32))
    for e in range(len(MIRROR_V4)):
        elem = jnp.array([e, e, e])
        twice = transform_feat_maps(transform_feat_maps(maps, elem, table), elem, table)
        assert np.allclose(np.asarray(twice), np.asarray(maps))


def test_read_op_elems_present_under_op_and_none_without():
    env_op = _lbf(op_mirror=True)
    _, st = env_op.reset(jax.random.PRNGKey(0))
    elems = read_op_elems(st, env_op.agents)
    assert elems is not None and set(elems) == {"agent_0", "agent_1"}

    env_plain = _lbf(op_mirror=False)
    _, st2 = env_plain.reset(jax.random.PRNGKey(0))
    assert read_op_elems(st2, env_plain.agents) is None
