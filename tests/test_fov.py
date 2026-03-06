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
    _align_forward_up, fov_observation, OvercookedFOVWrapper,
)


def test_fov_observation_shape():
    """fov_observation returns (fov_px, fov_px, 3) regardless of agent position."""
    env = OvercookedV1()
    rng = jax.random.PRNGKey(0)
    _, state = env.reset(rng)
    img = render_state(state)

    fov_size = 7
    fov_px = fov_size * TILE_PIXELS
    for centered in (True, False):
        for agent_idx in range(2):
            obs = fov_observation(
                img, state.agent_pos[agent_idx], state.agent_dir_idx[agent_idx],
                fov_size, TILE_PIXELS, centered,
            )
            assert obs.shape == (fov_px, fov_px, 3), f"agent {agent_idx}, centered={centered}"


def test_fov_edge_padding():
    """Crop near map edge should be zero-padded, not error."""
    img = jnp.ones((21, 21, 3), dtype=jnp.uint8) * 128
    pos = jnp.array([0, 0], dtype=jnp.uint32)
    dir_idx = jnp.int32(0)
    obs = fov_observation(img, pos, dir_idx, fov_size=7, tile_size=TILE_PIXELS, centered=True)
    assert obs.shape == (49, 49, 3)
    # Some pixels should be zero (padding)
    assert float(obs.min()) == 0.0


def test_align_forward_up():
    """Rotation mapping matches OGC: k = [2, 1, 0, 3][dir_idx]."""
    crop = jnp.arange(49 * 49 * 3).reshape(49, 49, 3).astype(jnp.float32)
    rotations = []
    for d in range(4):
        rotated = _align_forward_up(crop, jnp.int32(d))
        rotations.append(np.array(rotated))

    # OGC mapping: N->k=2, S->k=1, E->k=0, W->k=3
    np.testing.assert_array_equal(rotations[0], np.rot90(np.array(crop), k=2))  # N
    np.testing.assert_array_equal(rotations[1], np.rot90(np.array(crop), k=1))  # S
    np.testing.assert_array_equal(rotations[2], np.array(crop))                  # E
    np.testing.assert_array_equal(rotations[3], np.rot90(np.array(crop), k=3))  # W


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
        conv_filters=4,
        conv_num_blocks=2,
        conv_kernel_size=3,
        conv_stride=2,
        conv_padding="SAME",
        num_heads=2,
        head_features=4,
        fc_hidden_dim=4,
        lstm_hidden_dim=4,
        spatial_basis_depth=4,
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
        conv_filters=4,
        conv_num_blocks=2,
        conv_kernel_size=3,
        conv_stride=2,
        conv_padding="SAME",
        fc_hidden_dim=4,
        lstm_hidden_dim=4,
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
        "CONV_FILTERS": 4,
        "CONV_NUM_BLOCKS": 2,
        "CONV_KERNEL_SIZE": 3,
        "CONV_STRIDE": 2,
        "CONV_PADDING": "SAME",
        "JA_NUM_HEADS": 2,
        "JA_HEAD_FEATURES": 4,
        "JA_SPATIAL_BASIS_DEPTH": 4,
        "FC_HIDDEN_DIM": 4,
        "LSTM_HIDDEN_DIM": 4,
        "JA_BETA_MAX": 0.01,
        "JA_WARMUP_ENV_STEPS": 100,
        "TRAIN_SEED": 0,
    }

    init_fn, make_step_fn = make_train(config, env)
    rng = jax.random.PRNGKey(0)
    runner_state, policy, lstm_dim = init_fn(rng)
    step_fn = make_step_fn(policy, lstm_dim)

    num_updates = int(config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"])
    update_steps = jnp.int32(0)
    all_metrics = []
    for _ in range(num_updates):
        runner_state, update_steps, metric = step_fn(runner_state, update_steps)
        all_metrics.append(metric)

    assert len(all_metrics) == num_updates
    assert jnp.all(all_metrics[-1]["jsd_mean"] >= 0)
