"""Tests for the gaze-based image PPO path."""
import jax
import jax.numpy as jnp

from envs import make_env
from envs.log_wrapper import LogWrapper


def test_gaze_policy_forward_pass():
    from agents.gaze_image_actor_critic_agent import GazeImageActorCriticPolicy

    env = make_env("card-game", {"max_steps": 4})
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    img_h = env.grid_height * env.tile_size
    img_w = env.grid_width * env.tile_size

    policy = GazeImageActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=obs_dim,
        img_height=img_h,
        img_width=img_w,
        conv_filters=4,
        conv_num_blocks=2,
        conv_kernel_size=3,
        conv_stride=2,
        fc_hidden_dim=8,
        lstm_hidden_dim=8,
        gaze_key_dim=8,
    )

    rng = jax.random.PRNGKey(0)
    rng, init_rng, act_rng = jax.random.split(rng, 3)
    params = policy.init_params(init_rng)
    hstate = policy.init_hstate(1)

    dummy_obs = jnp.zeros((1, 1, obs_dim))
    dummy_done = jnp.zeros((1, 1))
    dummy_avail = jnp.ones((1, 1, env.action_space(env.agents[0]).n))

    action, value, pi, new_hstate, aux = policy.get_action_value_policy(
        params, dummy_obs, dummy_done, dummy_avail, hstate, act_rng
    )

    assert action.shape == (1, 1)
    assert value.shape == (1, 1)
    assert new_hstate.shape == hstate.shape
    assert aux["attn_map"].shape[0] == 1
    assert aux["attn_map"].shape[1] == 1
    assert jnp.allclose(aux["attn_map"].sum(axis=(-2, -1)), 1.0, atol=1e-5)


def test_gaze_train_loop_card_game():
    from marl.gaze_ippo import make_train

    env = make_env("card-game", {"max_steps": 4})
    env = LogWrapper(env)

    config = {
        "ENV_NAME": "card-game",
        "ENV_KWARGS": {"max_steps": 4},
        "NUM_ENVS": 2,
        "TOTAL_TIMESTEPS": 16,
        "ROLLOUT_LENGTH": 4,
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
        "FC_HIDDEN_DIM": 8,
        "LSTM_HIDDEN_DIM": 8,
        "MOTT_KEY_DIM": 8,
        "CONV_FILTERS": 4,
        "CONV_NUM_BLOCKS": 2,
        "CONV_KERNEL_SIZE": 3,
        "CONV_STRIDE": 2,
        "CONV_PADDING": "SAME",
        "TRAIN_SEED": 0,
    }

    init_fn, make_step_fn = make_train(config, env)
    rng = jax.random.PRNGKey(0)
    runner_state, policy = init_fn(rng)
    step_fn = make_step_fn(policy)

    num_updates = int(config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"])
    update_steps = jnp.int32(0)
    all_metrics = []
    for _ in range(num_updates):
        runner_state, update_steps, metric = step_fn(runner_state, update_steps)
        all_metrics.append(metric)

    assert len(all_metrics) == num_updates
    assert jnp.isfinite(all_metrics[-1]["loss_total"])
