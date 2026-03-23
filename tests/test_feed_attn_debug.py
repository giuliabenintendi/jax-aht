"""Debug what the 4th channel (feed_attn) looks like for the card game.

Saves images showing:
1. The raw scene (RGB, 3 channels)
2. The 4th channel agent 0 receives (partner's attention upsampled)
3. The 4th channel agent 1 receives (agent 0's attention upsampled)

Tests both normal mode and fixed_partner mode.

Run: ./run_gpu.sh 0 pytest -s tests/test_feed_attn_debug.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from pathlib import Path

from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.card_game.rendering import TILE_PIXELS, GRID_ROWS, GRID_COLS, NUM_CARDS
from agents.initialize_agents import initialize_ja_image_agent, _get_image_dims
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import augment_obs_for_eval


def _save_channel_debug(out_dir, step, obs_flat, prev_attn_partner, attn_own, img_h, img_w, scale, label):
    """Save debug images for one agent at one step."""
    # RGB scene
    scene = np.array(obs_flat[:img_h * img_w * 3]).reshape(img_h, img_w, 3)
    scene_img = np.array(Image.fromarray((scene * 255).astype(np.uint8)).resize(
        (img_w * scale, img_h * scale), Image.NEAREST))

    # 4th channel (what this agent receives from partner)
    upsampled = np.array(jax.image.resize(prev_attn_partner, (img_h, img_w), method='nearest'))
    ch4_norm = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
    import matplotlib.cm as cm
    ch4_resized = np.array(Image.fromarray(ch4_norm.astype(np.float32), mode='F').resize(
        (img_w * scale, img_h * scale), resample=Image.NEAREST))
    ch4_img = (cm.hot(ch4_resized)[:, :, :3] * 255).astype(np.uint8)

    # Own attention output
    attn_np = np.array(attn_own)
    attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
    attn_resized = np.array(Image.fromarray(attn_norm.astype(np.float32), mode='F').resize(
        (img_w * scale, img_h * scale), resample=Image.NEAREST))
    attn_img = (cm.hot(attn_resized)[:, :, :3] * 255).astype(np.uint8)

    # Stack: scene | 4th channel received | own attention
    combined = np.concatenate([scene_img, ch4_img, attn_img], axis=1)
    path = out_dir / f"step{step}_{label}.png"
    Image.fromarray(combined).save(path)
    return path


def _run_debug(env_kwargs, out_dir_name, fixed_attn_map=None):
    out_dir = Path(f"tests/{out_dir_name}")
    out_dir.mkdir(exist_ok=True)

    env = make_env("card-game", env_kwargs)
    env = LogWrapper(env)

    img_h = GRID_ROWS * TILE_PIXELS
    img_w = GRID_COLS * TILE_PIXELS
    feat_h, feat_w = _compute_resnet_output_dims(img_h, img_w, stride=2, kernel_size=3, padding="SAME", num_blocks=4)

    algo_config = {
        "ALG": "ja_ippo", "NUM_ENVS": 1, "CONV_FILTERS": 32, "CONV_NUM_BLOCKS": 4,
        "CONV_KERNEL_SIZE": 3, "CONV_STRIDE": 2, "CONV_PADDING": "SAME",
        "FC_HIDDEN_DIM": 256, "LSTM_HIDDEN_DIM": 128,
        "JA_NUM_HEADS": 4, "JA_HEAD_FEATURES": 16, "JA_SPATIAL_BASIS_DEPTH": 8,
        "ACTIVATION": "relu", "ENV_NAME": "card-game", "ENV_KWARGS": env_kwargs,
        "FEED_OTHER_ATTN": True,
    }
    rng = jax.random.PRNGKey(42)
    policy, params = initialize_ja_image_agent(algo_config, env, rng)

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    hstate_0 = policy.init_hstate(1)
    hstate_1 = policy.init_hstate(1)

    # Initial attention: uniform or fixed
    prev_attn_0 = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)
    if fixed_attn_map is not None:
        prev_attn_1 = fixed_attn_map
    else:
        prev_attn_1 = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)

    max_steps = env_kwargs["max_steps"]
    scale = 20

    print(f"\n{'='*60}")
    print(f"Debug: {out_dir_name}")
    print(f"Image: {img_h}x{img_w}, Feature map: {feat_h}x{feat_w}")
    print(f"fixed_partner_pos: {env_kwargs.get('fixed_partner_pos', -1)}")
    print(f"{'='*60}")

    for step in range(max_steps):
        obs_0 = obs["agent_0"]
        obs_1 = obs["agent_1"]

        print(f"\nStep {step + 1}/{max_steps}:")
        print(f"  prev_attn_1 (what agent 0 receives): min={float(prev_attn_1.min()):.4f} max={float(prev_attn_1.max()):.4f} sum={float(prev_attn_1.sum()):.4f}")
        print(f"  prev_attn_0 (what agent 1 receives): min={float(prev_attn_0.min()):.4f} max={float(prev_attn_0.max()):.4f} sum={float(prev_attn_0.sum()):.4f}")

        obs_0_aug = augment_obs_for_eval(obs_0, prev_attn_1, img_h, img_w)
        obs_1_aug = augment_obs_for_eval(obs_1, prev_attn_0, img_h, img_w)

        rng, rng0, rng1, step_rng = jax.random.split(rng, 4)
        done = {k: jnp.zeros((1,), dtype=bool) for k in env.agents + ["__all__"]}
        avail = env.get_avail_actions(env_state.env_state)

        act_0, hstate_0, attn_0 = policy.get_action_and_attention(
            params=params, obs=obs_0_aug.reshape(1, 1, -1),
            done=done["agent_0"].reshape(1, 1),
            avail_actions=avail["agent_0"].astype(jnp.float32),
            hstate=hstate_0, rng=rng0, greedy=True,
        )
        act_1, hstate_1, attn_1 = policy.get_action_and_attention(
            params=params, obs=obs_1_aug.reshape(1, 1, -1),
            done=done["agent_1"].reshape(1, 1),
            avail_actions=avail["agent_1"].astype(jnp.float32),
            hstate=hstate_1, rng=rng1, greedy=True,
        )

        attn_0_sq = attn_0.squeeze()
        attn_1_sq = attn_1.squeeze()

        print(f"  agent 0 attn argmax: {np.unravel_index(np.array(attn_0_sq).argmax(), attn_0_sq.shape)}")
        print(f"  agent 1 attn argmax: {np.unravel_index(np.array(attn_1_sq).argmax(), attn_1_sq.shape)}")
        print(f"  agent 0 action: {int(act_0.squeeze())}")
        print(f"  agent 1 action: {int(act_1.squeeze())}")

        # Save debug images
        p0 = _save_channel_debug(out_dir, step, obs_0, prev_attn_1, attn_0_sq, img_h, img_w, scale, "agent0")
        p1 = _save_channel_debug(out_dir, step, obs_1, prev_attn_0, attn_1_sq, img_h, img_w, scale, "agent1")
        print(f"  Saved: {p0}, {p1}")

        # Update for next step
        prev_attn_0 = attn_0_sq
        if fixed_attn_map is not None:
            prev_attn_1 = fixed_attn_map  # Keep fixed
        else:
            prev_attn_1 = attn_1_sq

        # Override agent 1 action if fixed partner
        fp = env_kwargs.get("fixed_partner_pos", -1)
        act_1_final = jnp.int32(fp) if fp >= 0 else act_1.squeeze()

        env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1_final}
        obs, env_state, reward, dones, info = env.step(step_rng, env_state, env_act)
        print(f"  Reward: {float(reward['agent_0'])}")


def test_normal_mode():
    """Debug feed_attn in normal (no fixed partner) mode."""
    _run_debug(
        env_kwargs={"max_steps": 8, "obs_type": "image", "shuffle": True, "fixed_partner_pos": -1},
        out_dir_name="feed_attn_debug_normal",
    )


def test_fixed_partner_mode():
    """Debug feed_attn with fixed partner at position 0."""
    img_h = GRID_ROWS * TILE_PIXELS
    img_w = GRID_COLS * TILE_PIXELS
    feat_h, feat_w = _compute_resnet_output_dims(img_h, img_w, stride=2, kernel_size=3, padding="SAME", num_blocks=4)

    # Create fixed attention for position 0
    pixel_col = 0 * TILE_PIXELS + TILE_PIXELS // 2
    pixel_row = 1 * TILE_PIXELS + TILE_PIXELS // 2
    fc = min(int(pixel_col / (img_w / feat_w)), feat_w - 1)
    fr = min(int(pixel_row / (img_h / feat_h)), feat_h - 1)
    fixed_attn = jnp.zeros((feat_h, feat_w))
    fixed_attn = fixed_attn.at[fr, fc].set(1.0)
    print(f"Fixed attention at feature ({fr},{fc})")

    _run_debug(
        env_kwargs={"max_steps": 4, "obs_type": "image", "shuffle": True, "fixed_partner_pos": 0},
        out_dir_name="feed_attn_debug_fixed",
        fixed_attn_map=fixed_attn,
    )
