"""Visualize attention filtering: raw vs card-only masked attention.

Shows what happens when we mask the attention to only consider
card positions in the feature map.

Run: ./run_gpu.sh 0 pytest -s tests/test_attn_filtering.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.cm as cm

from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.card_game.rendering import TILE_PIXELS, GRID_ROWS, GRID_COLS, NUM_CARDS
from agents.initialize_agents import initialize_ja_image_agent
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import augment_obs_for_eval


def _build_card_mask(feat_h, feat_w, img_h, img_w):
    """Build a binary mask over the feature map that covers only card positions."""
    mask = np.zeros((feat_h, feat_w), dtype=np.float32)
    for pos in range(NUM_CARDS):
        pixel_col = pos * TILE_PIXELS + TILE_PIXELS // 2
        pixel_row = 1 * TILE_PIXELS + TILE_PIXELS // 2
        fc = min(int(pixel_col / (img_w / feat_w)), feat_w - 1)
        fr = min(int(pixel_row / (img_h / feat_h)), feat_h - 1)
        mask[fr, fc] = 1.0
        # Also include the row above/below for better coverage
        if fr > 0:
            mask[fr - 1, fc] = 1.0
        if fr + 1 < feat_h:
            mask[fr + 1, fc] = 1.0
    return jnp.array(mask)


def _filter_attention(attn, card_mask):
    """Apply card mask and renormalize."""
    filtered = attn * card_mask
    total = filtered.sum()
    return jnp.where(total > 1e-8, filtered / total, filtered)


def _attn_to_img(attn, img_h, img_w, scale, cmap_name="hot"):
    """Convert attention map to upscaled colored image."""
    attn_np = np.array(attn)
    a_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)
    resized = np.array(Image.fromarray(a_norm.astype(np.float32), mode='F').resize(
        (img_w * scale, img_h * scale), resample=Image.NEAREST))
    cmap = getattr(cm, cmap_name)
    return (cmap(resized)[:, :, :3] * 255).astype(np.uint8)


def test_attention_filtering():
    out_dir = Path("tests/attn_filtering")
    out_dir.mkdir(exist_ok=True)

    env_kwargs = {"max_steps": 8, "obs_type": "image", "shuffle": True}
    env = make_env("card-game", env_kwargs)
    env = LogWrapper(env)

    img_h = GRID_ROWS * TILE_PIXELS
    img_w = GRID_COLS * TILE_PIXELS
    feat_h, feat_w = _compute_resnet_output_dims(img_h, img_w, stride=2, kernel_size=3, padding="SAME", num_blocks=4)

    # Build card mask
    card_mask = _build_card_mask(feat_h, feat_w, img_h, img_w)
    print(f"Feature map: {feat_h}x{feat_w}")
    print(f"Card mask ({int(card_mask.sum())} active cells out of {feat_h * feat_w}):")
    print(np.array(card_mask))

    # Save mask visualization
    scale = 20
    mask_img = _attn_to_img(card_mask, img_h, img_w, scale, "Greens")
    scene_perm = jnp.arange(NUM_CARDS)
    from envs.card_game.rendering import render_card_game
    scene = np.array(render_card_game(scene_perm))
    scene_up = np.array(Image.fromarray(scene).resize((img_w * scale, img_h * scale), Image.NEAREST))
    # Overlay mask on scene
    alpha = 0.5
    mask_resized = np.array(Image.fromarray(np.array(card_mask).astype(np.float32), mode='F').resize(
        (img_w * scale, img_h * scale), resample=Image.NEAREST))
    blended = ((1 - alpha * mask_resized[:, :, None]) * scene_up + alpha * mask_resized[:, :, None] * mask_img)
    mask_overlay = np.clip(blended, 0, 255).astype(np.uint8)
    Image.fromarray(mask_overlay).save(out_dir / "card_mask_overlay.png")
    print(f"Saved card mask overlay")

    # Initialize random policy
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

    # Run a few steps and compare raw vs filtered attention
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)
    hstate = policy.init_hstate(1)
    prev_attn = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)

    for step in range(4):
        obs_0 = obs["agent_0"]
        obs_aug = augment_obs_for_eval(obs_0, prev_attn, img_h, img_w)

        rng, rng0, step_rng = jax.random.split(rng, 3)
        done = {k: jnp.zeros((1,), dtype=bool) for k in env.agents + ["__all__"]}
        avail = env.get_avail_actions(env_state)

        _, hstate, attn = policy.get_action_and_attention(
            params=params, obs=obs_aug.reshape(1, 1, -1),
            done=done["agent_0"].reshape(1, 1),
            avail_actions=avail["agent_0"].astype(jnp.float32),
            hstate=hstate, rng=rng0, greedy=True,
        )
        attn_sq = attn.squeeze()

        # Filter
        attn_filtered = _filter_attention(attn_sq, card_mask)

        print(f"\nStep {step}:")
        print(f"  Raw attn: max={float(attn_sq.max()):.4f} at {np.unravel_index(np.array(attn_sq).argmax(), attn_sq.shape)}")
        print(f"  Filtered: max={float(attn_filtered.max()):.4f} at {np.unravel_index(np.array(attn_filtered).argmax(), attn_filtered.shape)}")
        print(f"  Mass on cards (raw): {float((attn_sq * card_mask).sum()):.4f}")
        print(f"  Mass on non-cards (raw): {float((attn_sq * (1 - card_mask)).sum()):.4f}")

        # Save comparison: scene | raw attn | filtered attn | raw upsampled | filtered upsampled
        raw_img = _attn_to_img(attn_sq, img_h, img_w, scale, "hot")
        filt_img = _attn_to_img(attn_filtered, img_h, img_w, scale, "hot")

        # Also show what feed_attn would look like
        raw_up = np.array(jax.image.resize(attn_sq, (img_h, img_w), method='nearest'))
        filt_up = np.array(jax.image.resize(attn_filtered, (img_h, img_w), method='nearest'))
        raw_ch4 = _attn_to_img(raw_up, img_h, img_w, scale, "Blues")
        filt_ch4 = _attn_to_img(filt_up, img_h, img_w, scale, "Blues")

        combined = np.concatenate([scene_up, raw_img, filt_img, raw_ch4, filt_ch4], axis=1)
        path = out_dir / f"step{step}_compare.png"
        Image.fromarray(combined).save(path)
        print(f"  Saved: {path} (scene | raw attn | filtered | raw 4th ch | filtered 4th ch)")

        prev_attn = attn_sq
        env_act = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
        obs, env_state, _, _, _ = env.step(step_rng, env_state, env_act)
