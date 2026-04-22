"""Post-hoc per-agent attention analysis for the card game.

Loads a saved training checkpoint, runs N eval episodes with a chosen seed,
and for each agent produces two separate figures per episode:
  - <agent>_obs.png: agent's own-frame view across the episode, with a dot
    highlighting the agent's own message (or pick, on the decision step) in
    its own colour (orange for agent 0, magenta for agent 1).
  - <agent>_attention.png: attention map per step rendered with the agent's
    palette (Oranges / RdPu), normalised to the episode's global max.

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
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import (
    AGENT_0_COLOR,
    AGENT_1_COLOR,
    CARD_COLORS,
    GRID_COLS,
    GRID_ROWS,
    NUM_CARDS,
    TILE_PIXELS,
    render_card_game,
)
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import run_episode_with_states


_H = GRID_ROWS * TILE_PIXELS
_W = GRID_COLS * TILE_PIXELS
_WHITE = np.array([255, 255, 255], dtype=np.uint8)
_AGENT_GRID_POSITIONS = [(0, 2), (2, 2)]  # (row, col) of agent 0, agent 1 tiles


def _draw_border_np(img, row, col, tile_size, color):
    """Draw 1-pixel border around the tile at (row, col)."""
    y = row * tile_size
    x = col * tile_size
    img[y, x:x + tile_size] = color
    img[y + tile_size - 1, x:x + tile_size] = color
    img[y:y + tile_size, x] = color
    img[y:y + tile_size, x + tile_size - 1] = color
    return img


def _walk_to_card_state(state):
    """Walk .env_state chain to the innermost CardGameState."""
    s = state
    while hasattr(s, "env_state") and not hasattr(s, "card_permutation"):
        s = s.env_state
    return s


def _get_per_agent_info(state, agent_idx: int):
    """Return (card_permutation, pos_perm, recolouring) for agent_idx from a wrapped state."""
    name = f"agent_{agent_idx}"
    recol = np.asarray(state.per_agent_recolouring[name])
    pos_perm = np.asarray(state.env_state.per_agent_perm[name])
    card_state = _walk_to_card_state(state)
    card_perm = np.asarray(card_state.card_permutation)
    return card_perm, pos_perm, recol


def _render_own_frame(card_perm, pos_perm, recolouring, agent_idx: int,
                     own_gt_value: int, is_decision: bool):
    """Render agent's own-frame view (post-OP), with agent's own-colour dot on
    the card it just acted on (messaged card, or picked card on decision step)."""
    # render_card_game uses jax.lax.scan, so input must be a JAX array
    base = np.asarray(render_card_game(jnp.asarray(card_perm))).copy()
    TP = TILE_PIXELS

    # Position shuffle of the card row
    card_row = base[TP:2 * TP, :, :].copy()
    tiles = card_row.reshape(TP, NUM_CARDS, TP, 3)
    shuffled = tiles[:, pos_perm, :, :]
    base[TP:2 * TP, :, :] = shuffled.reshape(TP, NUM_CARDS * TP, 3)

    # Recolouring of the card row: read from the original source, write to a
    # separate destination so permutation cycles don't cascade through each other.
    src = base[TP:2 * TP, :, :]
    dst = src.copy()
    card_colors_np = np.asarray(CARD_COLORS)
    for gt_idx in range(NUM_CARDS):
        original = card_colors_np[gt_idx]
        new_color = card_colors_np[int(recolouring[gt_idx])]
        mask = np.all(src == original, axis=-1)
        dst[mask] = new_color
    base[TP:2 * TP, :, :] = dst

    # Ego highlight — white border around this agent's tile
    agent_row, agent_col = _AGENT_GRID_POSITIONS[agent_idx]
    base = _draw_border_np(base, agent_row, agent_col, TP, _WHITE)

    # Decision indicator (white 4×4 top-left)
    if is_decision:
        base[:4, :4, :] = _WHITE

    # Agent's own-colour marker: dot on messaged card (deliberation), bounding
    # box around picked card (decision step).
    if own_gt_value >= 0:
        gt_positions = np.where(card_perm == own_gt_value)[0]
        if len(gt_positions):
            gt_col = int(gt_positions[0])
            view_positions = np.where(pos_perm == gt_col)[0]
            if len(view_positions):
                view_col = int(view_positions[0])
                agent_color = np.asarray(
                    AGENT_0_COLOR if agent_idx == 0 else AGENT_1_COLOR
                )
                if is_decision:
                    base = _draw_border_np(base, 1, view_col, TP, agent_color)
                else:
                    dot_size = 2
                    dy = TP + (TP - dot_size) // 2
                    dx = view_col * TP + (TP - dot_size) // 2
                    base[dy:dy + dot_size, dx:dx + dot_size] = agent_color

    return base


def _render_obs_sequence(ep_states, ep_actions, ep_messages, agent_idx: int,
                        output_path: Path, max_steps: int):
    """1 row × T columns: agent's own view with own-action dot per step."""
    num_steps = min(len(ep_states) - 1, max_steps)
    if num_steps == 0:
        return

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.8, 2.0))
    if num_steps == 1:
        axes = [axes]

    for t in range(num_steps):
        is_decision = (t == num_steps - 1)
        if is_decision:
            own_value = int(ep_actions[t][agent_idx]) if t < len(ep_actions) else -1
        else:
            own_value = int(ep_messages[t][agent_idx]) if t < len(ep_messages) else -1

        state = ep_states[t]
        card_perm, pos_perm, recolouring = _get_per_agent_info(state, agent_idx)

        img = _render_own_frame(
            card_perm, pos_perm, recolouring, agent_idx, own_value, is_decision,
        )
        axes[t].imshow(img, interpolation="nearest")
        axes[t].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.05)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _render_attention_sequence(attn_maps, ep_states, ep_actions, ep_messages,
                              agent_idx: int, agent_key: str, cmap: str,
                              output_path: Path, max_steps: int):
    """1 row × T columns: per-step attention + arrow to messaged card or pick bbox."""
    num_steps = min(len(attn_maps[agent_key]), max_steps)
    if num_steps == 0:
        return

    attn_stack = np.array([
        np.asarray(attn_maps[agent_key][t]).squeeze() for t in range(num_steps)
    ])
    attn_max = float(attn_stack.max())
    if attn_max <= 0:
        attn_max = 1.0

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 1.8, 2.0))
    if num_steps == 1:
        axes = [axes]

    extent = (0, _W, _H, 0)
    TP = TILE_PIXELS

    CARD_INSET = 1
    CARD_SIZE = TP - 2 * CARD_INSET  # 5

    agent_rgb = (
        np.asarray(AGENT_0_COLOR if agent_idx == 0 else AGENT_1_COLOR) / 255.0
    )
    # Arrow is always straight vertical: starts from the vertical centre of the
    # agent's row on the X-column of the messaged card, points toward the card.
    agent_row, _agent_col = _AGENT_GRID_POSITIONS[agent_idx]
    arrow_start_y = agent_row * TP + TP / 2.0

    def _add_card_outlines(ax):
        for card_col in range(NUM_CARDS):
            x = card_col * TP + CARD_INSET
            y = 1 * TP + CARD_INSET
            ax.add_patch(patches.Rectangle(
                (x, y), CARD_SIZE, CARD_SIZE,
                linewidth=0.8, edgecolor="black", facecolor="none",
            ))

    def _view_col_of(card_perm, pos_perm, gt_value):
        if gt_value < 0:
            return None
        gt_cols = np.where(card_perm == gt_value)[0]
        if len(gt_cols) == 0:
            return None
        gt_col = int(gt_cols[0])
        view_cols = np.where(pos_perm == gt_col)[0]
        if len(view_cols) == 0:
            return None
        return int(view_cols[0])

    for t in range(num_steps):
        is_decision = (t == num_steps - 1)
        attn = np.asarray(attn_maps[agent_key][t]).squeeze()
        axes[t].imshow(
            attn, cmap=cmap, vmin=0.0, vmax=attn_max,
            interpolation="nearest", extent=extent,
        )
        _add_card_outlines(axes[t])

        # Overlay action indicator
        state = ep_states[t]
        card_perm, pos_perm, _ = _get_per_agent_info(state, agent_idx)
        if is_decision:
            pick_gt = int(ep_actions[t][agent_idx]) if t < len(ep_actions) else -1
            view_col = _view_col_of(card_perm, pos_perm, pick_gt)
            if view_col is not None:
                # External to the 5x5 card — use full tile footprint (7x7)
                axes[t].add_patch(patches.Rectangle(
                    (view_col * TP, 1 * TP),
                    TP, TP,
                    linewidth=1.8, edgecolor=agent_rgb, facecolor="none",
                ))
        else:
            msg_gt = int(ep_messages[t][agent_idx]) if t < len(ep_messages) else -1
            view_col = _view_col_of(card_perm, pos_perm, msg_gt)
            if view_col is not None:
                # Straight vertical arrow on the card's X column; tip lands on
                # the card centre.
                card_center = (
                    view_col * TP + TP / 2.0,
                    1 * TP + TP / 2.0,
                )
                arrow_start = (card_center[0], arrow_start_y)
                axes[t].annotate(
                    "",
                    xy=card_center,
                    xytext=arrow_start,
                    arrowprops=dict(
                        arrowstyle="->",
                        color="red",
                        lw=1.8,
                        shrinkA=0.0,
                        shrinkB=0.0,
                        mutation_scale=14,
                    ),
                )

        axes[t].set_xlim(0, _W)
        axes[t].set_ylim(_H, 0)
        axes[t].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.05)
    fig.savefig(output_path, bbox_inches="tight")
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

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
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

    if alg_config.get("COMMUNICATION", False):
        env_kwargs = dict(alg_config["ENV_KWARGS"])
        env_kwargs["communication"] = True
        alg_config["ENV_KWARGS"] = env_kwargs

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
        ep_states, attn_maps, ep_actions, ep_messages = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=True,
        )

        ep_dir = output_dir / f"episode_{ep}"
        ep_dir.mkdir(exist_ok=True)

        for agent_idx, (agent_key, cmap) in enumerate([
            ("agent_0", "Oranges"),
            ("agent_1", "RdPu"),
        ]):
            _render_attention_sequence(
                attn_maps, ep_states, ep_actions, ep_messages,
                agent_idx, agent_key, cmap,
                output_path=ep_dir / f"{agent_key}_attention.png",
                max_steps=max_steps,
            )

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
