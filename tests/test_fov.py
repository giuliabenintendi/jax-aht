"""Tests for FOV observation pipeline: wrapper, cropping, rotation, networks, training."""
import math

import jax
import jax.numpy as jnp
import numpy as np

from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.overcooked.overcooked_v1 import OvercookedV1
from envs.overcooked.rendering import render_state
from envs.overcooked.rendering.overcooked_rendering import TILE_PIXELS
from envs.overcooked.overcooked_fov_wrapper import (
    _crop_fov, _rotate_forward_up, fov_observation, OvercookedFOVWrapper,
)


def test_crop_fov_shape():
    """_crop_fov returns (fov_px, fov_px, 3) regardless of agent position."""
    env = OvercookedV1()
    rng = jax.random.PRNGKey(0)
    _, state = env.reset(rng)
    img = render_state(state)

    fov_size = 7
    fov_px = fov_size * TILE_PIXELS
    for centered in (True, False):
        for agent_idx in range(2):
            crop = _crop_fov(img, state.agent_pos[agent_idx], fov_size, TILE_PIXELS, centered)
            assert crop.shape == (fov_px, fov_px, 3), f"agent {agent_idx}, centered={centered}"


def test_crop_fov_padding():
    """Crop near map edge should be zero-padded, not error."""
    # Create a tiny 3x3 image
    img = jnp.ones((21, 21, 3), dtype=jnp.uint8) * 128
    # Agent at corner (0, 0)
    pos = jnp.array([0, 0], dtype=jnp.uint32)
    crop = _crop_fov(img, pos, fov_size=7, tile_size=TILE_PIXELS, centered=True)
    assert crop.shape == (49, 49, 3)
    # Some pixels should be zero (padding)
    assert float(crop.min()) == 0.0


def test_rotate_forward_up():
    """Each direction produces a distinct rotation."""
    crop = jnp.arange(49 * 49 * 3).reshape(49, 49, 3).astype(jnp.float32)
    rotations = []
    for d in range(4):
        rotated = _rotate_forward_up(crop, jnp.int32(d))
        rotations.append(np.array(rotated))

    # N (dir=0) should be identity
    np.testing.assert_array_equal(rotations[0], np.array(crop))
    # S (dir=1) should be 180 rotation
    np.testing.assert_array_equal(rotations[1], np.rot90(np.array(crop), k=2))
    # E (dir=2) should be 90 CCW
    np.testing.assert_array_equal(rotations[2], np.rot90(np.array(crop), k=1))
    # W (dir=3) should be 270 CCW
    np.testing.assert_array_equal(rotations[3], np.rot90(np.array(crop), k=3))


def test_agents_see_different_crops():
    """Two agents at different positions should get different FOV observations."""
    env = OvercookedV1()
    rng = jax.random.PRNGKey(42)
    _, state = env.reset(rng)
    img = render_state(state)

    fov_0 = fov_observation(img, state.agent_pos[0], state.agent_dir_idx[0], 7, TILE_PIXELS, False)
    fov_1 = fov_observation(img, state.agent_pos[1], state.agent_dir_idx[1], 7, TILE_PIXELS, False)

    # Agents are at different positions, so crops should differ
    assert not jnp.array_equal(fov_0, fov_1)


def test_fov_wrapper_obs_shape():
    """OvercookedFOVWrapper produces correct obs shape and values in [0,1]."""
    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "fov"})
    rng = jax.random.PRNGKey(1)

    obs, state = env.reset(rng)
    expected_dim = 49 * 49 * 3
    for agent in env.agents:
        assert obs[agent].shape == (expected_dim,)
        assert float(obs[agent].min()) >= 0.0
        assert float(obs[agent].max()) <= 1.0

    # Step and check shapes persist
    actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
    rng, step_rng = jax.random.split(rng)
    obs2, state2, rewards, dones, info = env.step(step_rng, state, actions)
    for agent in env.agents:
        assert obs2[agent].shape == obs[agent].shape


def test_fov_wrapper_different_obs():
    """FOV wrapper gives different observations to each agent."""
    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "fov"})
    rng = jax.random.PRNGKey(42)
    obs, _ = env.reset(rng)
    assert not jnp.array_equal(obs["agent_0"], obs["agent_1"])


def test_make_env_fov():
    """make_env with obs_type='fov' returns a working env with smaller obs than symbolic."""
    fov_env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "fov"})
    rng = jax.random.PRNGKey(2)
    obs, _ = fov_env.reset(rng)
    fov_obs_dim = obs["agent_0"].shape[0]
    assert fov_obs_dim == 49 * 49 * 3


def test_ja_fov_forward_pass():
    """Init JA image policy with FOV dims, run forward pass, check shapes."""
    from agents.ja_image_actor_critic_agent import JAImageActorCriticPolicy

    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "fov"})
    fov_px = env.fov_px

    policy = JAImageActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        img_height=fov_px,
        img_width=fov_px,
        num_scalars=0,
        conv_filters=4,
        num_heads=2,
        head_features=4,
        fc_hidden_dim=4,
        lstm_hidden_dim=4,
        spatial_basis_depth=4,
        scalar_embed_dim=2,
    )

    rng = jax.random.PRNGKey(3)
    rng, init_rng, act_rng = jax.random.split(rng, 3)
    params = policy.init_params(init_rng)

    batch_size = 1
    seq_len = 1
    hstate = policy.init_hstate(batch_size)

    obs_dim = env.observation_space(env.agents[0]).shape[0]
    dummy_obs = jnp.zeros((seq_len, batch_size, obs_dim))
    dummy_done = jnp.zeros((seq_len, batch_size))
    dummy_avail = jnp.ones((seq_len, batch_size, env.action_space(env.agents[0]).n))

    action, val, pi, new_hstate, attn_map = policy.get_action_value_policy(
        params, dummy_obs, dummy_done, dummy_avail, hstate, act_rng,
    )
    assert action.shape == (seq_len, batch_size)
    assert val.shape == (seq_len, batch_size)
    assert new_hstate.shape == hstate.shape

    expected_feat_h = math.ceil(fov_px / 4)
    expected_feat_w = math.ceil(fov_px / 4)
    assert attn_map.shape == (seq_len, batch_size, expected_feat_h, expected_feat_w)


def test_image_fov_forward_pass():
    """Init Image (no attention) policy with FOV dims, run forward pass."""
    from agents.image_actor_critic_agent import ImageActorCriticPolicy

    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "fov"})
    fov_px = env.fov_px

    policy = ImageActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        img_height=fov_px,
        img_width=fov_px,
        num_scalars=0,
        conv_filters=4,
        fc_hidden_dim=4,
        lstm_hidden_dim=4,
        scalar_embed_dim=2,
    )

    rng = jax.random.PRNGKey(4)
    rng, init_rng, act_rng = jax.random.split(rng, 3)
    params = policy.init_params(init_rng)

    batch_size = 1
    seq_len = 1
    hstate = policy.init_hstate(batch_size)

    obs_dim = env.observation_space(env.agents[0]).shape[0]
    dummy_obs = jnp.zeros((seq_len, batch_size, obs_dim))
    dummy_done = jnp.zeros((seq_len, batch_size))
    dummy_avail = jnp.ones((seq_len, batch_size, env.action_space(env.agents[0]).n))

    action, val, pi, new_hstate = policy.get_action_value_policy(
        params, dummy_obs, dummy_done, dummy_avail, hstate, act_rng,
    )
    assert action.shape == (seq_len, batch_size)
    assert val.shape == (seq_len, batch_size)
    assert new_hstate.shape == hstate.shape


def test_ja_fov_train_loop():
    """Run make_train for 2 updates with FOV obs — catches shape mismatches."""
    from marl.ja_ippo import make_train

    env = make_env("overcooked-v1", {
        "layout": "cramped_room", "max_steps": 10, "obs_type": "fov",
    })
    env = LogWrapper(env)

    config = {
        "ENV_NAME": "overcooked-v1",
        "ENV_KWARGS": {"layout": "cramped_room", "max_steps": 10, "obs_type": "fov"},
        "OBS_TYPE": "fov",
        "NUM_ENVS": 2,
        "TOTAL_TIMESTEPS": 40,
        "ROLLOUT_LENGTH": 10,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 1,
        "NUM_CHECKPOINTS": 2,
        "LR": 1e-3,
        "ANNEAL_LR": False,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "relu",
        "JA_CONV_FILTERS": 4,
        "JA_NUM_HEADS": 2,
        "JA_HEAD_FEATURES": 4,
        "JA_SPATIAL_BASIS_DEPTH": 4,
        "JA_SCALAR_EMBED_DIM": 2,
        "FC_HIDDEN_DIM": 4,
        "LSTM_HIDDEN_DIM": 4,
        "JA_BETA_MAX": 0.01,
        "JA_WARMUP_ENV_STEPS": 100,
        "TRAIN_SEED": 0,
    }

    train_fn = make_train(config, env)
    rng = jax.random.PRNGKey(0)
    out = jax.jit(train_fn)(rng)

    assert "final_params" in out
    assert "metrics" in out
    assert "agent_0" in out["final_params"]
    assert "agent_1" in out["final_params"]
    assert jnp.all(out["metrics"]["jsd_mean"] >= 0)
