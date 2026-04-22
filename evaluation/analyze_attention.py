"""Post-hoc per-agent attention analysis for the card game.

Loads a saved training checkpoint, runs N eval episodes with a chosen seed,
and produces for each agent a 2-row figure showing:
  - Top row: the agent's own-frame observation at each step.
  - Bottom row: the agent's attention map at each step, rendered with the
    coolwarm palette and normalized to the episode's global max (no overlay
    on the obs image).

Usage:
    ./run_gpu.sh <gpu> evaluation.analyze_attention \\
        --checkpoint /path/to/saved/train_run \\
        --seed-idx 0 \\
        --num-episodes 5 \\
        --output-dir plots/card_game/<run_name>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_image_agent, initialize_ja_agent
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import GRID_COLS, GRID_ROWS, TILE_PIXELS
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import run_episode_with_states


_H = GRID_ROWS * TILE_PIXELS  # 21
_W = GRID_COLS * TILE_PIXELS  # 35


def _obs_to_image(flat_obs):
    """Flat obs → (H, W, 3) uint8 RGB image."""
    return (np.asarray(flat_obs) * 255).astype(np.uint8).reshape(_H, _W, 3)


def _render_agent_figure(ep_obs, attn_maps, agent_key: str, episode_idx: int,
                        output_path: Path, max_steps: int, cmap: str = "Oranges"):
    """Render one agent's episode: 2 rows × max_steps columns (obs + attn)."""
    num_steps = min(len(ep_obs), len(attn_maps[agent_key]))
    if num_steps == 0:
        return

    attn_stack = np.array([
        np.asarray(attn_maps[agent_key][t]).squeeze()
        for t in range(num_steps)
    ])
    attn_max = float(attn_stack.max())
    if attn_max <= 0:
        attn_max = 1.0

    fig, axes = plt.subplots(
        2, num_steps,
        figsize=(num_steps * 1.8, 4.0),
        gridspec_kw={"height_ratios": [1.2, 1.0]},
    )
    if num_steps == 1:
        axes = axes.reshape(2, 1)

    im = None
    obs_extent = (0, _W, _H, 0)  # obs coordinate space; reused for attention so shapes align
    for t in range(num_steps):
        is_decision = (t == num_steps - 1)

        # Top: obs image (own frame)
        obs_img = _obs_to_image(ep_obs[t][agent_key])
        axes[0, t].imshow(obs_img, interpolation="nearest", extent=obs_extent)
        label = f"t={t + 1}"
        if is_decision:
            label += " (D)"
        axes[0, t].set_title(label, fontsize=10)
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])
        if is_decision:
            for spine in axes[0, t].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)

        # Bottom: attention heatmap. Upscale via extent to share the obs
        # coordinate space so both panels have identical footprint.
        attn = np.asarray(attn_maps[agent_key][t]).squeeze()
        im = axes[1, t].imshow(
            attn, cmap=cmap, vmin=0.0, vmax=attn_max,
            interpolation="nearest", extent=obs_extent,
        )
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])
        if is_decision:
            for spine in axes[1, t].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)

    axes[0, 0].set_ylabel("obs", fontsize=10)
    axes[1, 0].set_ylabel("attention", fontsize=10)

    fig.subplots_adjust(left=0.03, right=0.9, top=0.90, bottom=0.05,
                        wspace=0.05, hspace=0.15)
    cbar_ax = fig.add_axes([0.91, 0.08, 0.012, 0.35])
    fig.colorbar(im, cax=cbar_ax)

    fig.suptitle(f"Episode {episode_idx} — {agent_key}  (vmax={attn_max:.3f})",
                 fontsize=11)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _get_obs_type(alg_config):
    return alg_config.get(
        "OBS_TYPE",
        alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved train run checkpoint directory")
    parser.add_argument("--seed-idx", type=int, default=0,
                        help="Which seed to analyze (0..NUM_SEEDS-1)")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--output-dir", default="plots/card_game")
    parser.add_argument("--episode-rng-base", type=int, default=100,
                        help="Base seed for per-episode RNGs: key = base + ep")
    args = parser.parse_args()

    # Load config from adjacent .hydra/config.yaml
    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    # walk up to find .hydra
    config_path = None
    cur = run_dir
    for _ in range(4):
        candidate = cur / ".hydra" / "config.yaml"
        if candidate.exists():
            config_path = candidate
            break
        cur = cur.parent
    if config_path is None:
        raise FileNotFoundError(
            f"Could not locate .hydra/config.yaml near {run_dir}"
        )
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]

    # Propagate COMMUNICATION flag into ENV_KWARGS (matches ja_ippo training path)
    if alg_config.get("COMMUNICATION", False):
        env_kwargs = dict(alg_config["ENV_KWARGS"])
        env_kwargs["communication"] = True
        alg_config["ENV_KWARGS"] = env_kwargs

    # Build env + policy
    env = make_env(alg_config["ENV_NAME"], alg_config["ENV_KWARGS"])
    env = LogWrapper(env)
    inner_env = env._env
    max_steps = alg_config["ENV_KWARGS"].get("max_steps", 8)

    obs_type = _get_obs_type(alg_config)
    init_fn = (
        initialize_ja_image_agent if obs_type in ("image", "fov")
        else initialize_ja_agent
    )
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env, rng)

    # Load params, pick the requested seed
    run_data = load_train_run(str(ckpt_path))
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    if args.seed_idx >= num_seeds:
        raise ValueError(
            f"seed_idx={args.seed_idx} out of range for {num_seeds} seeds"
        )
    params = jax.tree.map(lambda x: x[args.seed_idx], final_params)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  seeds in checkpoint: {num_seeds}, analyzing seed {args.seed_idx}")
    print(f"  max_steps: {max_steps}")
    print(f"  saving to: {output_dir.resolve()}")

    for ep in range(args.num_episodes):
        ep_rng = jax.random.PRNGKey(args.episode_rng_base + ep)
        result = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=True, collect_obs=True,
        )
        ep_states, attn_maps, ep_actions, ep_messages, ep_obs = result

        ep_dir = output_dir / f"episode_{ep}"
        ep_dir.mkdir(exist_ok=True)

        for agent_key, cmap in (("agent_0", "Oranges"), ("agent_1", "RdPu")):
            _render_agent_figure(
                ep_obs, attn_maps, agent_key, ep,
                output_path=ep_dir / f"{agent_key}_obs_and_attention.png",
                max_steps=max_steps, cmap=cmap,
            )

        # Small text summary per episode
        summary = ep_dir / "summary.txt"
        with summary.open("w") as f:
            f.write(f"Episode {ep}\n")
            f.write(f"  max_steps: {max_steps}\n")
            f.write(f"  ep_actions (GT): {ep_actions}\n")
            f.write(f"  ep_messages (GT): {ep_messages}\n")

        print(f"  episode {ep}: saved to {ep_dir}/")

    print(f"Done. Figures in {output_dir.resolve()}/")


if __name__ == "__main__":
    main()
