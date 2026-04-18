"""Debug what the 4th channel (feed_attn) looks like for the card game.

Saves images showing:
1. The raw scene (RGB, 3 channels)
2. The 4th channel agent 0 receives (partner's attention upsampled)
3. The 4th channel agent 1 receives (agent 0's attention upsampled)

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
    """Save debug figure with titles, colorbars, and proper labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene = np.array(obs_flat[:img_h * img_w * 3]).reshape(img_h, img_w, 3)
    scene_up = np.array(Image.fromarray((scene * 255).astype(np.uint8)).resize(
        (img_w * scale, img_h * scale), Image.NEAREST))

    # Upsample attention maps to image resolution for overlay
    ch4_up = np.array(jax.image.resize(prev_attn_partner, (img_h, img_w), method='nearest'))
    attn_up = np.array(jax.image.resize(attn_own, (img_h, img_w), method='nearest'))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Scene
    axes[0].imshow(scene_up)
    axes[0].set_title(f"Scene ({label})", fontsize=12)
    axes[0].axis("off")

    # Panel 2: 4th channel received (raw, not normalized)
    im1 = axes[1].imshow(np.array(prev_attn_partner), cmap="hot", interpolation="nearest")
    axes[1].set_title(f"4th channel received\n(feat map {prev_attn_partner.shape})", fontsize=11)
    axes[1].set_xlabel(f"sum={float(jnp.array(prev_attn_partner).sum()):.3f}")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Panel 3: 4th channel upsampled (what agent actually sees)
    im2 = axes[2].imshow(ch4_up, cmap="hot", interpolation="nearest")
    axes[2].set_title(f"4th channel upsampled\n({img_h}x{img_w})", fontsize=11)
    axes[2].set_xlabel(f"min={ch4_up.min():.4f} max={ch4_up.max():.4f}")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # Panel 4: Agent's own attention output
    im3 = axes[3].imshow(np.array(attn_own), cmap="hot", interpolation="nearest")
    axes[3].set_title(f"Own attention output\n(feat map {attn_own.shape})", fontsize=11)
    axes[3].set_xlabel(f"sum={float(jnp.array(attn_own).sum()):.3f}")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    fig.suptitle(f"Step {step} - {label}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = out_dir / f"step{step}_{label}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def _run_debug(env_kwargs, out_dir_name):
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

    prev_attn_0 = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)
    prev_attn_1 = jnp.ones((feat_h, feat_w)) / (feat_h * feat_w)

    max_steps = env_kwargs["max_steps"]
    scale = 20

    print(f"\n{'='*60}")
    print(f"Debug: {out_dir_name}")
    print(f"Image: {img_h}x{img_w}, Feature map: {feat_h}x{feat_w}")
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
        avail = env.get_avail_actions(env_state)

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

        prev_attn_0 = attn_0_sq
        prev_attn_1 = attn_1_sq

        env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1.squeeze()}
        obs, env_state, reward, dones, info = env.step(step_rng, env_state, env_act)
        print(f"  Reward: {float(reward['agent_0'])}")


def test_normal_mode():
    """Debug feed_attn."""
    _run_debug(
        env_kwargs={"max_steps": 8, "obs_type": "image", "shuffle": True},
        out_dir_name="feed_attn_debug_normal",
    )
