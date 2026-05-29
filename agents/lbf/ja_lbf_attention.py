"""Shared per-fruit attention geometry for LBF joint attention.

These helpers are used by the training loop (`marl/ja_ippo_lbf.py`) and by both
evaluation paths (`evaluation/vis_episodes.py`, `evaluation/run_xp_seeds.py`).
They live here, in the leaf `agents` package, so evaluation no longer has to
reach into the training module for them.

Fruit slot ordering is always lex by (row, col): fruit positions are fixed for
an episode (only `eaten` changes), so the lex order is stable across steps and
slot k refers to the same physical fruit throughout an episode.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def per_fruit_attn(
    attn_2d, food_pos, food_eaten,
    tile_size, feat_h, feat_w, img_h, img_w,
):
    """Point-gather attention at each fruit's centre feature cell.

    Map each fruit's tile centre pixel to a single (fr, fc) feature cell and
    take attn_2d at that cell. Eaten fruits get zero. Renormalise over
    on-fruit mass.

    Avoids materialising the (..., N, feat_h, feat_w) dense mask tensor used
    in earlier versions — both the memory and the 5-D einsum that XLA was
    routing into a cuDNN kernel that failed at larger batch sizes.

    attn_2d:    (..., feat_h, feat_w)        spatial attention
    food_pos:   (..., N, 2)  int             grid positions of each fruit
    food_eaten: (..., N)     bool

    Returns:
        per_fruit_norm: (..., N) — distribution over fruits given on-mass
        on_mass:        (...,)   — total attention mass that landed on fruits
    """
    centre_r = food_pos[..., 0] * tile_size + tile_size // 2  # (..., N)
    centre_c = food_pos[..., 1] * tile_size + tile_size // 2
    fr = jnp.clip(centre_r * feat_h // img_h, 0, feat_h - 1).astype(jnp.int32)
    fc = jnp.clip(centre_c * feat_w // img_w, 0, feat_w - 1).astype(jnp.int32)

    # Flatten spatial dims and gather. attn_2d shape (..., feat_h, feat_w)
    # → (..., feat_h*feat_w). Same leading dims as flat_idx so take_along_axis works.
    attn_flat = attn_2d.reshape(*attn_2d.shape[:-2], feat_h * feat_w)
    flat_idx = fr * feat_w + fc                                 # (..., N)
    per_fruit = jnp.take_along_axis(attn_flat, flat_idx, axis=-1)

    alive = 1.0 - food_eaten.astype(jnp.float32)
    per_fruit = per_fruit * alive
    on_mass = per_fruit.sum(axis=-1)
    per_fruit_norm = per_fruit / (on_mass[..., None] + 1e-8)
    return per_fruit_norm, on_mass


def lex_sort_food(food_pos, food_eaten):
    """Sort fruits by lex (row, col) so slot k = k-th fruit in reading order.

    food_pos: (num_envs, N, 2); food_eaten: (num_envs, N).
    Returns the same shapes, reordered per env.
    """
    def _single(pos, eaten):
        # lexsort orders by the LAST key as primary -> row primary, col secondary.
        idx = jnp.lexsort((pos[:, 1], pos[:, 0]))
        return pos[idx], eaten[idx]
    return jax.vmap(_single)(food_pos, food_eaten)


def as_spatial_attention(attn_map):
    """Return spatial attention with shape (..., feat_h, feat_w)."""
    if attn_map.ndim == 4:
        return attn_map
    raise ValueError(f"Unexpected attention map rank: {attn_map.ndim}")


def swap_partner(x, num_agents):
    """Swap the agent block in an actor-ordered tensor.

    Actor order: [agent_0 envs, agent_1 envs, ...]. For 2 agents, this swaps
    halves so position i is now occupied by the partner's value.
    """
    if num_agents != 2:
        raise NotImplementedError("ja_ippo_lbf assumes 2 agents (parameter-shared).")
    half = x.shape[0] // 2
    return jnp.concatenate([x[half:], x[:half]], axis=0)


def agent_positions_from_log_state(log_state):
    """Extract (num_envs, num_agents, 2) agent positions from LogWrapper-wrapped state."""
    return log_state.env_state.env_state.agents.position


def food_state_from_log_state(log_state):
    """Extract (food_pos, food_eaten) from LogWrapper-wrapped state.

    food_pos: (num_envs, N, 2); food_eaten: (num_envs, N).
    """
    food = log_state.env_state.env_state.food_items
    return food.position, food.eaten


def lbf_attention_ctx(config, env) -> dict:
    """Image/feature-map geometry needed to pool per-fruit attention.

    Consolidates the dimension computation the training loop, eval-video, and
    cross-play paths each used to repeat. The returned dict is the `lbf_ctx`
    consumed by `run_episode_with_states` / `run_single_episode_with_jsd` and
    feeds `per_fruit_attn` directly.
    """
    from agents.initialize_agents import _get_image_dims
    from agents.ja_actor_critic import _compute_resnet_output_dims

    img_h, img_w, _ = _get_image_dims(env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=config.get("CONV_STRIDE", 2),
        kernel_size=config.get("CONV_KERNEL_SIZE", 3),
        padding=config.get("CONV_PADDING", "SAME"),
        num_blocks=config.get("CONV_NUM_BLOCKS", 4),
    )
    inner = env._env if hasattr(env, "_env") else env
    return {
        "num_fruits": int(inner._num_food),
        "tile_size": int(inner.tile_size),
        "feat_h": feat_h,
        "feat_w": feat_w,
        "img_h": img_h,
        "img_w": img_w,
    }
