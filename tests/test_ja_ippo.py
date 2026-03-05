"""Smoke tests for JA-IPPO: policy forward pass, eval episode, and training loop."""
import jax
import jax.numpy as jnp

from envs import make_env
from envs.base_env import get_inner_env
from envs.log_wrapper import LogWrapper
from agents.ja_actor_critic_agent import JAActorCriticPolicy
from common.run_episodes import run_single_episode
from marl.ja_ippo import make_train


def _make_tiny_ja_policy(env):
    """Create a JA policy with minimal network sizes for fast testing."""
    inner_env = get_inner_env(env)
    obs_width = inner_env.obs_shape[0]
    obs_height = inner_env.obs_shape[1]
    obs_channels = inner_env.obs_shape[2]

    return JAActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        obs_height=obs_height,
        obs_width=obs_width,
        obs_channels=obs_channels,
        conv_filters=2,
        num_heads=2,
        head_features=4,
        fc_hidden_dim=4,
        lstm_hidden_dim=4,
        spatial_basis_depth=4,
        scalar_embed_dim=2,
    )


def test_ja_policy_forward_pass():
    """Init JA policy with tiny sizes, run get_action and get_action_value_policy, check shapes."""
    env = make_env("overcooked-v1", {"layout": "cramped_room", "max_steps": 10})
    policy = _make_tiny_ja_policy(env)

    rng = jax.random.PRNGKey(0)
    rng, init_rng, act_rng, avp_rng = jax.random.split(rng, 4)
    params = policy.init_params(init_rng)

    batch_size = 1
    seq_len = 1
    hstate = policy.init_hstate(batch_size)

    obs_dim = env.observation_space(env.agents[0]).shape[0]
    dummy_obs = jnp.zeros((seq_len, batch_size, obs_dim))
    dummy_done = jnp.zeros((seq_len, batch_size))
    dummy_avail = jnp.ones((seq_len, batch_size, env.action_space(env.agents[0]).n))

    # get_action
    action, new_hstate = policy.get_action(params, dummy_obs, dummy_done, dummy_avail, hstate, act_rng)
    assert action.shape == (seq_len, batch_size)
    assert new_hstate.shape == hstate.shape

    # get_action_value_policy
    action3, val, pi, new_hstate3, attn_map = policy.get_action_value_policy(
        params, dummy_obs, dummy_done, dummy_avail, hstate, avp_rng
    )
    assert action3.shape == (seq_len, batch_size)
    assert val.shape == (seq_len, batch_size)
    assert new_hstate3.shape == hstate.shape
    inner_env = get_inner_env(env)
    assert attn_map.shape == (seq_len, batch_size, inner_env.obs_shape[1], inner_env.obs_shape[0])

    # get_action_and_attention
    action4, new_hstate4, attn_map2 = policy.get_action_and_attention(
        params, dummy_obs, dummy_done, dummy_avail, hstate, act_rng
    )
    assert action4.shape == (seq_len, batch_size)
    assert new_hstate4.shape == hstate.shape
    assert attn_map2.shape == attn_map.shape


def test_ja_eval_episode():
    """Two JA agents run run_single_episode with partner-state routing end-to-end."""
    env = make_env("overcooked-v1", {"layout": "cramped_room", "max_steps": 10})

    rng = jax.random.PRNGKey(42)
    rng, init0_rng, init1_rng, ep_rng = jax.random.split(rng, 4)

    policy_0 = _make_tiny_ja_policy(env)
    params_0 = policy_0.init_params(init0_rng)

    policy_1 = _make_tiny_ja_policy(env)
    params_1 = policy_1.init_params(init1_rng)

    info = run_single_episode(
        ep_rng, env,
        agent_0_param=params_0, agent_0_policy=policy_0,
        agent_1_param=params_1, agent_1_policy=policy_1,
        max_episode_steps=10,
    )

    assert "base_return" in info


def test_ja_train_loop():
    """Run make_train for 2 updates with tiny config — catches shape mismatches in
    JSD computation, reward augmentation, and PPO loss.
    """
    env = make_env("overcooked-v1", {"layout": "cramped_room", "max_steps": 10})
    env = LogWrapper(env)

    config = {
        "ENV_NAME": "overcooked-v1",
        "ENV_KWARGS": {"layout": "cramped_room", "max_steps": 10},
        # Minimal sizes to keep the test fast
        "NUM_ENVS": 2,
        "TOTAL_TIMESTEPS": 40,  # 2 updates * 10 rollout * 2 envs
        "ROLLOUT_LENGTH": 10,
        "NUM_MINIBATCHES": 1,
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
        # Tiny JA network
        "ACTIVATION": "relu",
        "JA_CONV_FILTERS": 2,
        "JA_NUM_HEADS": 2,
        "JA_HEAD_FEATURES": 4,
        "JA_SPATIAL_BASIS_DEPTH": 4,
        "JA_SCALAR_EMBED_DIM": 2,
        "FC_HIDDEN_DIM": 4,
        "LSTM_HIDDEN_DIM": 4,
        # JA reward
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

    # Should have JA-specific metrics
    assert "ja_beta" in all_metrics[-1]
    assert "ja_reward_mean" in all_metrics[-1]
    assert "jsd_mean" in all_metrics[-1]

    # JSD should be non-negative
    assert jnp.all(all_metrics[-1]["jsd_mean"] >= 0)
