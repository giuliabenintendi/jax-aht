"""Debug the fixed partner card game setup.

Runs one episode with fixed_partner_pos=0, saves per-step debug images showing:
- Row 1: The scene as agent 0 sees it (RGB)
- Row 2: The 4th channel agent 0 receives (partner's attention, upsampled)
- Row 3: Agent 0's output attention
- Row 4: Agent 1's output attention (should be overridden to fixed)

Run: ./run_gpu.sh 0 pytest -s tests/test_fixed_partner_debug.py
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
from common.save_load_utils import load_train_run


def test_fixed_partner_debug():
    out_dir = Path("tests/fixed_partner_debug")
    out_dir.mkdir(exist_ok=True)

    # Create env with fixed partner
    env_kwargs = {"max_steps": 4, "obs_type": "image", "shuffle": True, "fixed_partner_pos": 0}
    env = make_env("card-game", env_kwargs)
    env = LogWrapper(env)

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
    print(f"Fixed attention: pos=0, feature=({fr},{fc}), sum={float(fixed_attn.sum())}")
    print(f"Fixed attention map:\n{np.array(fixed_attn)}")

    # Initialize a random policy (untrained)
    algo_config = {
        "ALG": "ja_ippo", "NUM_ENVS": 1, "CONV_FILTERS": 32, "CONV_NUM_BLOCKS": 4,
        "CONV_KERNEL_SIZE": 3, "CONV_STRIDE": 2, "CONV_PADDING": "SAME",
        "FC_HIDDEN_DIM": 256, "LSTM_HIDDEN_DIM": 128,
        "JA_NUM_HEADS": 4, "JA_HEAD_FEATURES": 16, "JA_SPATIAL_BASIS_DEPTH": 8,
        "ACTIVATION": "relu", "ENV_NAME": "card-game", "ENV_KWARGS": env_kwargs,
        "FEED_OTHER_ATTN": True,
    }
    rng = jax.random.PRNGKey(42)
    policy, init_params = initialize_ja_image_agent(algo_config, env, rng)

    # Run one episode
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    hstate_0 = policy.init_hstate(1)
    hstate_1 = policy.init_hstate(1)

    # Initialize uniform attention for feed_other_attn
    prev_attn_0 = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)
    prev_attn_1 = fixed_attn  # Agent 0 receives fixed attention from agent 1

    scale = 20
    max_steps = env_kwargs["max_steps"]

    for step in range(max_steps):
        print(f"\n=== Step {step + 1}/{max_steps} ===")

        obs_0 = obs["agent_0"]
        obs_1 = obs["agent_1"]

        # Show what agent 0 receives as 4th channel
        print(f"  Agent 0 receives partner attn (prev_attn_1):")
        print(f"    min={float(prev_attn_1.min()):.4f}, max={float(prev_attn_1.max()):.4f}, sum={float(prev_attn_1.sum()):.4f}")
        print(f"    map:\n{np.array(prev_attn_1).round(3)}")

        # Augment obs
        obs_0_aug = augment_obs_for_eval(obs_0, prev_attn_1, img_h, img_w)
        obs_1_aug = augment_obs_for_eval(obs_1, prev_attn_0, img_h, img_w)

        # Extract the 4th channel to visualize
        fourth_channel = np.array(obs_0_aug[img_h * img_w * 3:]).reshape(img_h, img_w)

        rng, rng0, rng1, step_rng = jax.random.split(rng, 4)
        done = {k: jnp.zeros((1,), dtype=bool) for k in env.agents + ["__all__"]}
        avail = env.get_avail_actions(env_state.env_state)

        act_0, hstate_0, attn_0 = policy.get_action_and_attention(
            params=init_params, obs=obs_0_aug.reshape(1, 1, -1),
            done=done["agent_0"].reshape(1, 1),
            avail_actions=avail["agent_0"].astype(jnp.float32),
            hstate=hstate_0, rng=rng0, greedy=True,
        )
        act_1, hstate_1, attn_1 = policy.get_action_and_attention(
            params=init_params, obs=obs_1_aug.reshape(1, 1, -1),
            done=done["agent_1"].reshape(1, 1),
            avail_actions=avail["agent_1"].astype(jnp.float32),
            hstate=hstate_1, rng=rng1, greedy=True,
        )

        act_0_val = int(act_0.squeeze())
        act_1_val = int(act_1.squeeze())
        # Override agent 1's action
        act_1_override = 0  # fixed_partner_pos=0

        attn_0_np = np.array(attn_0.squeeze())
        attn_1_np = np.array(attn_1.squeeze())

        print(f"  Agent 0 action: {act_0_val}, Agent 1 action: {act_1_val} (overridden to {act_1_override})")
        print(f"  Agent 0 attention max at: {np.unravel_index(attn_0_np.argmax(), attn_0_np.shape)}")
        print(f"  Agent 1 attention max at: {np.unravel_index(attn_1_np.argmax(), attn_1_np.shape)}")

        # Save debug image: scene | 4th channel | agent0 attn | fixed attn
        scene = np.array(obs_0[:img_h * img_w * 3]).reshape(img_h, img_w, 3)
        scene_up = np.array(Image.fromarray((scene * 255).astype(np.uint8)).resize(
            (img_w * scale, img_h * scale), Image.NEAREST))

        def attn_to_img(attn, cmap_name="Reds"):
            import matplotlib.cm as cm
            a = np.array(attn)
            a_norm = (a - a.min()) / (a.max() - a.min() + 1e-8)
            resized = np.array(Image.fromarray(a_norm.astype(np.float32), mode='F').resize(
                (img_w * scale, img_h * scale), resample=Image.NEAREST))
            cmap = getattr(cm, cmap_name)
            return (cmap(resized)[:, :, :3] * 255).astype(np.uint8)

        ch4_img = attn_to_img(fourth_channel, "Blues")
        attn0_img = attn_to_img(attn_0_np, "Oranges")
        fixed_img = attn_to_img(np.array(fixed_attn), "Reds")

        # Stack horizontally
        combined = np.concatenate([scene_up, ch4_img, attn0_img, fixed_img], axis=1)
        path = out_dir / f"step_{step}.png"
        Image.fromarray(combined).save(path)
        print(f"  Saved {path}")

        # Update attention for next step
        prev_attn_0 = attn_0.squeeze()
        prev_attn_1 = fixed_attn  # Always fixed

        # Step env
        env_act = {"agent_0": act_0.squeeze(), "agent_1": jnp.int32(act_1_override)}
        obs, env_state, reward, dones, info = env.step(step_rng, env_state, env_act)
        print(f"  Reward: {float(reward['agent_0'])}")
