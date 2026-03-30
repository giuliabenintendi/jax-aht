"""Tests for the FEED_OTHER_ATTN feature.

Run with: uv run pytest -s tests/test_feed_other_attn.py
Requires GPU (JAX with CUDA).
"""
import jax
import jax.numpy as jnp
import numpy as np

from agents.ja_image_actor_critic import JAImageActorCritic, JAImageScannedLSTM, _compute_resnet_output_dims
from agents.ja_dual_image_actor_critic import JADualImageActorCritic
from agents.ja_image_actor_critic_agent import JAImageActorCriticPolicy
from agents.ja_dual_image_actor_critic_agent import JADualImageActorCriticPolicy
from agents.ja_utils import augment_obs_for_eval
from agents.initialize_agents import initialize_ja_image_agent, initialize_ja_dual_image_agent

# cramped_room dimensions
IMG_H, IMG_W = 28, 35
FEAT_H, FEAT_W = _compute_resnet_output_dims(IMG_H, IMG_W, stride=2, kernel_size=3, padding="SAME", num_blocks=4)
ACTION_DIM = 6


def test_resnet_output_dims():
    """Feature map dims for cramped_room (4x5 grid, TILE_PIXELS=7)."""
    assert FEAT_H == 7 and FEAT_W == 9


def test_scanned_lstm_num_channels_3():
    """Default 3-channel image: network init and forward pass."""
    rng = jax.random.PRNGKey(0)
    lstm = JAImageScannedLSTM(img_height=IMG_H, img_width=IMG_W, num_channels=3)

    batch_size = 4
    obs_dim = IMG_H * IMG_W * 3
    carry = JAImageScannedLSTM.initialize_carry(batch_size, 64)
    dummy_obs = jnp.zeros((1, batch_size, obs_dim))
    dummy_done = jnp.zeros((1, batch_size))

    params = lstm.init(rng, carry, (dummy_obs, dummy_done))
    (new_h, new_c), (lstm_out, attn_map) = lstm.apply(params, carry, (dummy_obs, dummy_done))

    assert lstm_out.shape == (1, batch_size, 64)
    assert attn_map.shape == (1, batch_size, FEAT_H, FEAT_W)


def test_scanned_lstm_num_channels_4():
    """4-channel image: network init and forward pass with attention channel."""
    rng = jax.random.PRNGKey(0)
    lstm = JAImageScannedLSTM(img_height=IMG_H, img_width=IMG_W, num_channels=4)

    batch_size = 4
    obs_dim = IMG_H * IMG_W * 4
    carry = JAImageScannedLSTM.initialize_carry(batch_size, 64)
    dummy_obs = jnp.zeros((1, batch_size, obs_dim))
    dummy_done = jnp.zeros((1, batch_size))

    params = lstm.init(rng, carry, (dummy_obs, dummy_done))
    (new_h, new_c), (lstm_out, attn_map) = lstm.apply(params, carry, (dummy_obs, dummy_done))

    assert lstm_out.shape == (1, batch_size, 64)
    assert attn_map.shape == (1, batch_size, FEAT_H, FEAT_W)


def test_scanned_lstm_with_scalar_suffix():
    """Image JA accepts extra scalar features appended after the image."""
    rng = jax.random.PRNGKey(0)
    lstm = JAImageScannedLSTM(
        img_height=IMG_H, img_width=IMG_W, num_channels=3, scalar_dim=2
    )

    batch_size = 4
    obs_dim = IMG_H * IMG_W * 3 + 2
    carry = JAImageScannedLSTM.initialize_carry(batch_size, 64)
    dummy_obs = jnp.zeros((1, batch_size, obs_dim))
    dummy_done = jnp.zeros((1, batch_size))

    params = lstm.init(rng, carry, (dummy_obs, dummy_done))
    (_, _), (lstm_out, attn_map) = lstm.apply(params, carry, (dummy_obs, dummy_done))

    assert lstm_out.shape == (1, batch_size, 64)
    assert attn_map.shape == (1, batch_size, FEAT_H, FEAT_W)


def test_scanned_lstm_gaussian_gaze_with_bypass():
    """Gaussian-gaze mode returns a normalized gaze map and valid LSTM output."""
    rng = jax.random.PRNGKey(0)
    lstm = JAImageScannedLSTM(
        img_height=IMG_H,
        img_width=IMG_W,
        num_channels=3,
        attn_mode="gaussian_gaze",
        use_global_bypass=True,
    )

    batch_size = 3
    obs_dim = IMG_H * IMG_W * 3
    carry = JAImageScannedLSTM.initialize_carry(batch_size, 64)
    dummy_obs = jnp.zeros((1, batch_size, obs_dim))
    dummy_done = jnp.zeros((1, batch_size))

    params = lstm.init(rng, carry, (dummy_obs, dummy_done))
    (_, _), (lstm_out, attn_map) = lstm.apply(params, carry, (dummy_obs, dummy_done))

    assert lstm_out.shape == (1, batch_size, 64)
    assert attn_map.shape == (1, batch_size, FEAT_H, FEAT_W)
    assert jnp.allclose(attn_map.sum(axis=(-2, -1)), 1.0, atol=1e-5)
    assert jnp.all(attn_map >= 0.0)


def test_single_critic_policy_4ch():
    """JAImageActorCriticPolicy with num_channels=4 inits and runs."""
    rng = jax.random.PRNGKey(0)
    obs_dim = IMG_H * IMG_W * 4

    policy = JAImageActorCriticPolicy(
        action_dim=ACTION_DIM, obs_dim=obs_dim,
        img_height=IMG_H, img_width=IMG_W, num_channels=4,
        lstm_hidden_dim=64,
    )
    params = policy.init_params(rng)
    hstate = policy.init_hstate(2)

    dummy_obs = jnp.zeros((1, 2, obs_dim))
    dummy_done = jnp.zeros((1, 2))
    dummy_avail = jnp.ones((1, 2, ACTION_DIM))

    action, value, pi, new_hstate, attn_map = policy.get_action_value_policy(
        params=params, obs=dummy_obs, done=dummy_done,
        avail_actions=dummy_avail, hstate=hstate, rng=rng,
    )
    assert action.shape == (1, 2)
    assert attn_map.shape == (1, 2, FEAT_H, FEAT_W)


def test_dual_critic_policy_4ch():
    """JADualImageActorCriticPolicy with num_channels=4 inits and runs."""
    rng = jax.random.PRNGKey(0)
    obs_dim = IMG_H * IMG_W * 4

    policy = JADualImageActorCriticPolicy(
        action_dim=ACTION_DIM, obs_dim=obs_dim,
        img_height=IMG_H, img_width=IMG_W, num_channels=4,
        lstm_hidden_dim=64,
    )
    params = policy.init_params(rng)
    hstate = policy.init_hstate(2)

    dummy_obs = jnp.zeros((1, 2, obs_dim))
    dummy_done = jnp.zeros((1, 2))
    dummy_avail = jnp.ones((1, 2, ACTION_DIM))

    action, (val_ext, val_int), pi, new_hstate, attn_map = policy.get_action_value_policy(
        params=params, obs=dummy_obs, done=dummy_done,
        avail_actions=dummy_avail, hstate=hstate, rng=rng,
    )
    assert action.shape == (1, 2)
    assert val_ext.shape == (1, 2)
    assert val_int.shape == (1, 2)
    assert attn_map.shape == (1, 2, FEAT_H, FEAT_W)


def test_augment_obs_for_eval():
    """augment_obs_for_eval produces correct shape and channel layout.

    Reshape to (H, W, 4) — the way the network unpacks it — and verify
    that channels 0-2 are RGB and channel 3 is the attention mask.
    """
    obs_flat = jnp.ones(IMG_H * IMG_W * 3) * 0.5
    attn = jnp.ones((FEAT_H, FEAT_W)) / (FEAT_H * FEAT_W)

    augmented = augment_obs_for_eval(obs_flat, attn, IMG_H, IMG_W)

    assert augmented.shape == (IMG_H * IMG_W * 4,)

    image = augmented.reshape(IMG_H, IMG_W, 4)
    assert jnp.allclose(image[:, :, :3], 0.5)

    expected_val = 1.0 / (FEAT_H * FEAT_W)
    assert jnp.allclose(image[:, :, 3], expected_val, atol=1e-6)


def test_augment_obs_nearest_neighbor():
    """Nearest-neighbor upsampling preserves spatial structure."""
    # Non-uniform attention: hot spot at (0,0)
    attn = jnp.zeros((FEAT_H, FEAT_W))
    attn = attn.at[0, 0].set(1.0)

    augmented = augment_obs_for_eval(jnp.zeros(IMG_H * IMG_W * 3), attn, IMG_H, IMG_W)
    image = augmented.reshape(IMG_H, IMG_W, 4)
    attn_channel = image[:, :, 3]

    # Top-left block should be 1.0, rest should be 0.0
    assert attn_channel[0, 0] == 1.0
    assert attn_channel[IMG_H - 1, IMG_W - 1] == 0.0
    # RGB channels should be untouched (all zeros)
    assert jnp.allclose(image[:, :, :3], 0.0)


def test_swap_and_reset_logic():
    """Verify attention swap between agents and reset on done."""
    num_envs = 4
    num_actors = num_envs * 2

    # Agent 0 attends to top-left, agent 1 attends to bottom-right
    attn_0 = jnp.zeros((num_envs, FEAT_H, FEAT_W))
    attn_0 = attn_0.at[:, 0, 0].set(1.0)
    attn_1 = jnp.zeros((num_envs, FEAT_H, FEAT_W))
    attn_1 = attn_1.at[:, -1, -1].set(1.0)

    # Batch as network output: (1, num_actors, feat_h, feat_w)
    attn_map = jnp.concatenate([attn_0, attn_1], axis=0)[None, ...]

    # No dones
    done_batch = jnp.zeros(num_actors)

    # Simulate _swap_and_reset_attn
    attn = attn_map.squeeze(0)
    a0 = attn[:num_envs]
    a1 = attn[num_envs:]
    swapped = jnp.concatenate([a1, a0], axis=0)
    uniform = jnp.ones((FEAT_H, FEAT_W)) / (FEAT_H * FEAT_W)
    result = jnp.where(done_batch[:, None, None], uniform[None], swapped)

    # Agent 0 should now see agent 1's attention (bottom-right hot)
    assert result[0, -1, -1] == 1.0
    assert result[0, 0, 0] == 0.0
    # Agent 1 should now see agent 0's attention (top-left hot)
    assert result[num_envs, 0, 0] == 1.0
    assert result[num_envs, -1, -1] == 0.0

    # With done on env 0 for both agents
    done_batch = done_batch.at[0].set(1.0).at[num_envs].set(1.0)
    result_done = jnp.where(done_batch[:, None, None], uniform[None], swapped)

    # Env 0 agents should have uniform attention
    expected_uniform = 1.0 / (FEAT_H * FEAT_W)
    assert jnp.allclose(result_done[0], expected_uniform)
    assert jnp.allclose(result_done[num_envs], expected_uniform)
    # Env 1 agents should still have swapped attention
    assert result_done[1, -1, -1] == 1.0


def test_initialize_agents_feed_other_attn():
    """initialize_ja_image_agent and initialize_ja_dual_image_agent
    set num_channels=4 and correct obs_dim when FEED_OTHER_ATTN=true."""
    from envs import make_env
    from envs.log_wrapper import LogWrapper

    config = {
        "ENV_NAME": "overcooked-v1",
        "ENV_KWARGS": {"layout": "cramped_room", "max_steps": 400, "obs_type": "image"},
        "FEED_OTHER_ATTN": True,
        "CONV_FILTERS": 32, "CONV_NUM_BLOCKS": 4,
        "CONV_KERNEL_SIZE": 3, "CONV_STRIDE": 2, "CONV_PADDING": "SAME",
        "JA_NUM_HEADS": 4, "JA_HEAD_FEATURES": 16,
        "FC_HIDDEN_DIM": 64, "LSTM_HIDDEN_DIM": 64,
        "JA_SPATIAL_BASIS_DEPTH": 8,
    }
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    rng = jax.random.PRNGKey(0)

    # Single critic
    policy, params = initialize_ja_image_agent(config, env, rng)
    assert policy.obs_dim == IMG_H * IMG_W * 4
    assert policy.network.num_channels == 4

    # Dual critic
    policy_dual, params_dual = initialize_ja_dual_image_agent(config, env, rng)
    assert policy_dual.obs_dim == IMG_H * IMG_W * 4
    assert policy_dual.network.num_channels == 4

    # Without FEED_OTHER_ATTN
    config["FEED_OTHER_ATTN"] = False
    policy_3ch, _ = initialize_ja_image_agent(config, env, rng)
    assert policy_3ch.obs_dim == IMG_H * IMG_W * 3
    assert policy_3ch.network.num_channels == 3


def test_initialize_agents_gaussian_gaze():
    """Image JA agent initialization propagates gaussian-gaze config knobs."""
    from envs import make_env
    from envs.log_wrapper import LogWrapper

    config = {
        "ENV_NAME": "overcooked-v1",
        "ENV_KWARGS": {"layout": "cramped_room", "max_steps": 400, "obs_type": "image"},
        "FEED_OTHER_ATTN": False,
        "CONV_FILTERS": 32,
        "CONV_NUM_BLOCKS": 4,
        "CONV_KERNEL_SIZE": 3,
        "CONV_STRIDE": 2,
        "CONV_PADDING": "SAME",
        "FC_HIDDEN_DIM": 64,
        "LSTM_HIDDEN_DIM": 64,
        "JA_ATTN_MODE": "gaussian_gaze",
        "JA_USE_GLOBAL_BYPASS": True,
        "JA_GAZE_MIN_SIGMA": 0.75,
    }
    env = make_env(config["ENV_NAME"], config["ENV_KWARGS"])
    env = LogWrapper(env)
    rng = jax.random.PRNGKey(0)

    policy, _ = initialize_ja_image_agent(config, env, rng)
    assert policy.network.attn_mode == "gaussian_gaze"
    assert policy.network.use_global_bypass is True
    assert np.isclose(policy.network.gaze_min_sigma, 0.75)


def test_visualize_4th_channel():
    """Visualize the 4-channel observation: RGB + attention heatmap.

    Saves a figure to tests/feed_other_attn_vis.png showing:
      - Left: 3-channel RGB image observation (agent 0)
      - Center: the attention map at feature-map resolution (what agent 1 produced)
      - Right: the 4th channel after nearest-neighbor upsampling to pixel resolution
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from envs import make_env

    env = make_env("overcooked-v1", {"layout": "cramped_room", "max_steps": 400, "obs_type": "image"})
    rng = jax.random.PRNGKey(42)
    rng, reset_rng = jax.random.split(rng)
    obs, _ = env.reset(reset_rng)

    # RGB obs for agent 0
    obs_0 = np.array(obs["agent_0"])
    rgb = obs_0.reshape(IMG_H, IMG_W, 3)

    # Get a real attention map from a randomly initialized network (untrained)
    policy = JAImageActorCriticPolicy(
        action_dim=ACTION_DIM, obs_dim=IMG_H * IMG_W * 3,
        img_height=IMG_H, img_width=IMG_W, num_channels=3, lstm_hidden_dim=64,
    )
    rng, init_rng, act_rng = jax.random.split(rng, 3)
    params = policy.init_params(init_rng)
    hstate = policy.init_hstate(1)

    # Forward pass with agent 1's obs to get its attention map
    obs_1 = obs["agent_1"]
    _, _, _, _, attn_map = policy.get_action_value_policy(
        params=params, obs=obs_1.reshape(1, 1, -1),
        done=jnp.zeros((1, 1)), avail_actions=jnp.ones((1, 1, ACTION_DIM)),
        hstate=hstate, rng=act_rng,
    )
    attn = np.array(attn_map.squeeze())  # (feat_h, feat_w)
    attn_jnp = jnp.array(attn)

    # Upsample to pixel resolution (same as what the training loop does)
    upsampled = np.array(jax.image.resize(attn_jnp, (IMG_H, IMG_W), method='nearest'))

    # Use gridspec for consistent panel sizes: RGB and upsampled share the same
    # pixel aspect (IMG_H x IMG_W), feature map is smaller (FEAT_H x FEAT_W)
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[IMG_W, IMG_W, IMG_W], wspace=0.35)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(rgb, aspect="equal")
    ax0.set_title(f"RGB obs (agent 0)\n{IMG_H}x{IMG_W}x3")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(attn, cmap="hot", interpolation="nearest", aspect="equal",
                      extent=[0, IMG_W, IMG_H, 0])
    ax1.set_title(f"Other's attention (feat map)\n{FEAT_H}x{FEAT_W}")
    # Annotate values on the feature-map grid
    cell_w = IMG_W / FEAT_W
    cell_h = IMG_H / FEAT_H
    for r in range(FEAT_H):
        for c in range(FEAT_W):
            v = attn[r, c]
            if v > 0.005:
                ax1.text(c * cell_w + cell_w / 2, r * cell_h + cell_h / 2,
                         f"{v:.3f}", ha="center", va="center",
                         fontsize=5, color="white" if v > 0.05 else "black")
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[2])
    im2 = ax2.imshow(upsampled, cmap="hot", interpolation="nearest", aspect="equal")
    ax2.set_title(f"4th channel (upsampled)\n{IMG_H}x{IMG_W}")
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    fig.suptitle("FEED_OTHER_ATTN: what the 4th image channel looks like\n"
                 "(untrained network — attention is near-uniform, will sharpen with training)",
                 fontweight="bold", fontsize=10)
    plt.tight_layout()

    out_path = "tests/feed_other_attn_vis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisualization saved to {out_path}")
