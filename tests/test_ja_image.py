"""Tests for image-based JA pipeline: rendering, wrapper, network, training."""
import jax
import jax.numpy as jnp

from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.overcooked.overcooked_v1 import OvercookedV1
from envs.overcooked.rendering import render_state
from envs.overcooked.rendering.overcooked_rendering import TILE_PIXELS


def test_render_state_shape():
    """render_state produces (H*7, W*7, 3) uint8 from an OvercookedV1 state."""
    env = OvercookedV1()
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    img = render_state(state)
    h, w = env.height, env.width
    assert img.shape == (h * TILE_PIXELS, w * TILE_PIXELS, 3)
    assert img.dtype == jnp.uint8


def test_image_wrapper():
    """OvercookedImageWrapper produces correct obs shape and values in [0,1]."""
    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "image"})
    rng = jax.random.PRNGKey(1)

    obs, state = env.reset(rng)
    for agent in env.agents:
        assert obs[agent].shape == (env.observation_space(agent).shape[0],)
        assert float(obs[agent].min()) >= 0.0
        assert float(obs[agent].max()) <= 1.0

    # Step and check auto-reset works
    actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
    rng, step_rng = jax.random.split(rng)
    obs2, state2, rewards, dones, info = env.step(step_rng, state, actions)
    for agent in env.agents:
        assert obs2[agent].shape == obs[agent].shape


def test_make_env_image():
    """make_env with obs_type='image' returns a working env; symbolic path unchanged."""
    # Image path
    img_env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "image"})
    rng = jax.random.PRNGKey(2)
    obs, _ = img_env.reset(rng)
    img_obs_dim = obs["agent_0"].shape[0]

    # Symbolic path (default)
    sym_env = make_env("overcooked-v1", {"layout": "cramped_room"})
    obs_sym, _ = sym_env.reset(rng)
    sym_obs_dim = obs_sym["agent_0"].shape[0]

    # Image obs should be much larger than symbolic
    assert img_obs_dim > sym_obs_dim


def test_ja_image_forward_pass():
    """Init JA image policy with tiny sizes, run forward pass, check shapes."""
    from agents.ja_image_actor_critic_agent import JAImageActorCriticPolicy

    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "image"})
    image_wrapper = env  # no LogWrapper here

    img_h = image_wrapper.grid_height * image_wrapper.tile_size
    img_w = image_wrapper.grid_width * image_wrapper.tile_size

    policy = JAImageActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        img_height=img_h,
        img_width=img_w,
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
    # Attention map should be at downsampled resolution (÷4)
    import math
    expected_ah = math.ceil(img_h / 4)
    expected_aw = math.ceil(img_w / 4)
    assert attn_map.shape == (seq_len, batch_size, expected_ah, expected_aw)


def test_ja_image_train_loop():
    """Run make_train for 2 updates with image obs — catches shape mismatches."""
    from marl.ja_ippo import make_train

    env = make_env("overcooked-v1", {
        "layout": "cramped_room", "max_steps": 10, "obs_type": "image",
    })
    env = LogWrapper(env)

    config = {
        "ENV_NAME": "overcooked-v1",
        "ENV_KWARGS": {"layout": "cramped_room", "max_steps": 10, "obs_type": "image"},
        "OBS_TYPE": "image",
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
