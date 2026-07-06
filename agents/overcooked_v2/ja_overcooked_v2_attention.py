"""Task-object detector and attention geometry for Overcooked V2 (MATE).

Analogous to `agents/lbf/ja_lbf_attention.py`. The MATE mechanism pools each
agent's spatial attention onto the task-relevant objects; this module is the
detector that says which cells those objects occupy and where they land on the
feature/attention grid.

Overcooked V2 task objects are the interactable STATIC cells of the layout —
defined mechanically as every cell that is neither wall nor empty: the pot, the
ingredient piles (distractor included), the delivery station (goal), the recipe
indicator, and the plate piles. Defining the set this way bakes in no
recipe-specific knowledge (dropping the distractor pile, say, would leak which
ingredient the current recipe needs). Unlike LBF apples these never move, are
never consumed, and are unaffected by `random_agent_positions` (only agents
move, and agents are not stored in the static grid channel). So their positions
are fixed for a layout and read once, at init, from the layout's static grid —
which is exactly `state.grid[:, :, 0]` (the `StaticObject` channel) for any
state of that layout.

Other-Play permutes only ingredient COLOURS per agent (`ingredient_permutations`),
never positions, so these object positions are OP-invariant and live in the
shared allocentric frame the joint-attention machinery relies on.
"""
from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
import numpy as np

from envs.base_env import get_inner_env
from envs.overcooked_v2.common import StaticObject

_INGREDIENT_BASE = int(StaticObject.INGREDIENT_PILE_BASE)


class TaskObject(IntEnum):
    """Coarse category of a detected Overcooked V2 task object.

    The object set is every interactable static cell — defined mechanically as
    non-wall, non-empty — so no per-episode or recipe-specific knowledge is baked
    in: the pot, the delivery station (goal), the recipe indicator, the plate
    piles, and the ingredient piles (distractor included).
    """

    POT = 0
    GOAL = 1
    RECIPE_INDICATOR = 2
    PLATE_PILE = 3
    INGREDIENT_PILE = 4


def _static_to_task(obj: int) -> int | None:
    """Map a `StaticObject` value to its `TaskObject` category, or None for
    non-interactable cells (walls, empty). The button recipe indicator is treated
    as a recipe indicator so button-variant layouts detect it too."""
    if obj == StaticObject.POT:
        return int(TaskObject.POT)
    if obj == StaticObject.GOAL:
        return int(TaskObject.GOAL)
    if obj in (StaticObject.RECIPE_INDICATOR, StaticObject.BUTTON_RECIPE_INDICATOR):
        return int(TaskObject.RECIPE_INDICATOR)
    if obj == StaticObject.PLATE_PILE:
        return int(TaskObject.PLATE_PILE)
    if obj >= _INGREDIENT_BASE:
        return int(TaskObject.INGREDIENT_PILE)
    return None


def detect_task_objects(
    static_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect task-object cells in a `StaticObject`-encoded `(H, W)` grid.

    Accepts either the layout's `static_objects` (host, at init) or a materialised
    `state.grid[:, :, 0]` — identical for a fixed layout. Cells are read in
    row-major order so slot `k` is stable across calls (positions never change).

    Returns `(positions, categories, ingredient_idx)`:
        positions:      `(M, 2)` int — `(row, col)` of each detected object.
        categories:     `(M,)`   int — `TaskObject` category.
        ingredient_idx: `(M,)`   int — ingredient index for ingredient piles,
                        else `-1`. Under Other-Play this is the TRUE ingredient
                        (positions are OP-invariant); the per-agent colour
                        relabeling does not change it.
    """
    grid = np.asarray(static_grid)
    positions: list[tuple[int, int]] = []
    categories: list[int] = []
    ingredients: list[int] = []
    height, width = grid.shape
    for row in range(height):
        for col in range(width):
            obj = int(grid[row, col])
            cat = _static_to_task(obj)
            if cat is None:
                continue
            positions.append((row, col))
            categories.append(cat)
            ingredients.append(
                obj - _INGREDIENT_BASE if cat == TaskObject.INGREDIENT_PILE else -1
            )
    return (
        np.array(positions, dtype=np.int32).reshape(-1, 2),
        np.array(categories, dtype=np.int32),
        np.array(ingredients, dtype=np.int32),
    )


def object_feature_coords(object_pos, tile_size, feat_h, feat_w, img_h, img_w):
    """Map object grid cells `(row, col)` to feature-grid cells `(fr, fc)`.

    Uses each object's tile-centre pixel, matching the LBF fruit-cell mapping
    (`ja_lbf_attention._food_feature_coords`) so the pooled-attention geometry is
    consistent across envs. `object_pos` is `(..., 2)` with `(row, col)` last.
    """
    centre_r = object_pos[..., 0] * tile_size + tile_size // 2
    centre_c = object_pos[..., 1] * tile_size + tile_size // 2
    fr = jnp.clip(centre_r * feat_h // img_h, 0, feat_h - 1).astype(jnp.int32)
    fc = jnp.clip(centre_c * feat_w // img_w, 0, feat_w - 1).astype(jnp.int32)
    return fr, fc


def overcooked_v2_object_ctx(config, env) -> dict:
    """Image/feature geometry plus the detected task objects, for pooling attention.

    Mirrors `agents/lbf/ja_lbf_attention.lbf_attention_ctx`: the object positions,
    categories and their feature-grid cells are resolved once from the layout and
    returned as constants for the JA mechanism and eval paths to consume.
    """
    from agents.initialize_agents import _get_image_dims
    from agents.ja_actor_critic import _compute_resnet_output_dims

    img_h, img_w, _ = _get_image_dims(env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h,
        img_w,
        stride=config.get("CONV_STRIDE", 2),
        kernel_size=config.get("CONV_KERNEL_SIZE", 3),
        padding=config.get("CONV_PADDING", "SAME"),
        num_blocks=config.get("CONV_NUM_BLOCKS", 4),
    )
    raw = get_inner_env(env)
    grid_h, grid_w = int(raw.height), int(raw.width)
    tile_size = img_h // grid_h
    obj_pos, obj_cat, obj_ing = detect_task_objects(raw.layout.static_objects)
    fr, fc = object_feature_coords(obj_pos, tile_size, feat_h, feat_w, img_h, img_w)
    return {
        "img_h": img_h,
        "img_w": img_w,
        "feat_h": feat_h,
        "feat_w": feat_w,
        "tile_size": tile_size,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "num_objects": int(obj_pos.shape[0]),
        "object_pos": jnp.asarray(obj_pos),  # (M, 2) (row, col)
        "object_cat": jnp.asarray(obj_cat),  # (M,)
        "object_ing": jnp.asarray(obj_ing),  # (M,)
        "object_feat_rc": jnp.stack([fr, fc], axis=-1),  # (M, 2) feature cell
    }


## Tests


def test_detect_task_objects_demo_cook_simple():
    from envs.overcooked_v2.layouts import overcooked_v2_layouts

    layout = overcooked_v2_layouts["demo_cook_simple"]
    pos, cat, ing = detect_task_objects(layout.static_objects)

    # Every interactable static cell: 1 pot, 1 goal, 2 recipe indicators,
    # 2 plate piles, 6 ingredient piles (distractor included).
    assert pos.shape == (12, 2)
    assert (cat == TaskObject.POT).sum() == 1
    assert (cat == TaskObject.GOAL).sum() == 1
    assert (cat == TaskObject.RECIPE_INDICATOR).sum() == 2
    assert (cat == TaskObject.PLATE_PILE).sum() == 2
    assert (cat == TaskObject.INGREDIENT_PILE).sum() == 6

    assert tuple(pos[cat == TaskObject.POT][0]) == (2, 7)
    assert tuple(pos[cat == TaskObject.GOAL][0]) == (2, 10)
    assert {tuple(p) for p in pos[cat == TaskObject.PLATE_PILE].tolist()} == {(1, 10), (3, 10)}

    # Ingredient piles carry their ingredient index (distractor 2 included);
    # every other category carries -1.
    assert set(ing[cat == TaskObject.INGREDIENT_PILE].tolist()) == {0, 1, 2}
    assert (ing[cat != TaskObject.INGREDIENT_PILE] == -1).all()
