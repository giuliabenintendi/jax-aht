"""Headless tile renderer for Overcooked V2.

Vendored from JaxMARL v0.1.0 `jaxmarl/viz/overcooked_v2_visualizer.py`
(commit 66f41e5a36131d86bf5791d6bbe501275ed2cd30), stripped of the `Window`
and `imageio` display dependencies. Only the pure-JAX tile-drawing logic is
kept so it can run inside a jitted observation pipeline.

`render_state` bakes the agents into the grid and returns the full god's-eye
RGB image. Per-agent view masking and the ego marker are applied by the image
wrapper, not here.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

import envs.overcooked_v2.grid_rendering_v2 as rendering
from envs.overcooked_v2.common import DynamicObject, StaticObject
from envs.overcooked_v2.settings import INDICATOR_ACTIVATION_TIME, POT_COOK_TIME

TILE_PIXELS = 8
SUBDIVS = 3

COLORS = {
    "red": jnp.array([255, 0, 0], dtype=jnp.uint8),
    "green": jnp.array([0, 255, 0], dtype=jnp.uint8),
    "blue": jnp.array([0, 0, 255], dtype=jnp.uint8),
    "purple": jnp.array([160, 32, 240], dtype=jnp.uint8),
    "yellow": jnp.array([255, 255, 0], dtype=jnp.uint8),
    "grey": jnp.array([100, 100, 100], dtype=jnp.uint8),
    "white": jnp.array([255, 255, 255], dtype=jnp.uint8),
    "black": jnp.array([25, 25, 25], dtype=jnp.uint8),
    "orange": jnp.array([230, 180, 0], dtype=jnp.uint8),
    "pink": jnp.array([255, 105, 180], dtype=jnp.uint8),
    "brown": jnp.array([139, 69, 19], dtype=jnp.uint8),
    "cyan": jnp.array([0, 255, 255], dtype=jnp.uint8),
    "light_blue": jnp.array([173, 216, 230], dtype=jnp.uint8),
    "dark_green": jnp.array([0, 150, 0], dtype=jnp.uint8),
}

INGREDIENT_COLORS = jnp.array(
    [
        COLORS["yellow"],
        COLORS["dark_green"],
        COLORS["purple"],
        COLORS["cyan"],
        COLORS["red"],
        COLORS["orange"],
        COLORS["purple"],
        COLORS["blue"],
        COLORS["pink"],
        COLORS["brown"],
    ]
)

AGENT_COLORS = jnp.array(
    [
        COLORS["red"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["purple"],
        COLORS["yellow"],
        COLORS["orange"],
    ]
)


def _encode_agent_extras(direction, idx):
    return direction | (idx << 2)


def _decode_agent_extras(extras):
    direction = extras & 0x3
    idx = extras >> 2
    return direction, idx


def _render_dynamic_item(
    ingredients,
    img,
    plate_fn=rendering.point_in_circle(0.5, 0.5, 0.3),
    ingredient_fn=rendering.point_in_circle(0.5, 0.5, 0.15),
    dish_positions=jnp.array([(0.5, 0.4), (0.4, 0.6), (0.6, 0.6)]),
):
    def _no_op(img, ingredients):
        return img

    def _render_plate(img, ingredients):
        return rendering.fill_coords(img, plate_fn, COLORS["white"])

    def _render_ingredient(img, ingredients):
        idx = DynamicObject.get_ingredient_idx(ingredients)
        return rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])

    def _render_dish(img, ingredients):
        img = rendering.fill_coords(img, plate_fn, COLORS["white"])
        ingredient_indices = DynamicObject.get_ingredient_idx_list_jit(ingredients)

        for idx, ingredient_idx in enumerate(ingredient_indices):
            color = INGREDIENT_COLORS[ingredient_idx]
            pos = dish_positions[idx]
            ing_fn = rendering.point_in_circle(pos[0], pos[1], 0.1)
            img_ing = rendering.fill_coords(img, ing_fn, color)

            img = jax.lax.select(ingredient_idx != -1, img_ing, img)

        return img

    branches = jnp.array(
        [
            ingredients == 0,
            ingredients == DynamicObject.PLATE,
            DynamicObject.is_ingredient(ingredients),
            ingredients & DynamicObject.COOKED,
        ]
    )
    branch_idx = jnp.argmax(branches)

    img = jax.lax.switch(
        branch_idx,
        [_no_op, _render_plate, _render_ingredient, _render_dish],
        img,
        ingredients,
    )

    return img


def _render_pot(cell, img):
    ingredients = cell[1]
    time_left = cell[2]

    is_cooking = time_left > 0
    is_cooked = (ingredients & DynamicObject.COOKED) != 0
    is_idle = ~is_cooking & ~is_cooked
    ingredients = DynamicObject.get_ingredient_idx_list_jit(ingredients)
    has_ingredients = ingredients[0] != -1

    img = rendering.fill_coords(img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"])

    ingredient_fns = [
        rendering.point_in_circle(*coord, 0.13)
        for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]
    ]

    for i, ingredient_idx in enumerate(ingredients):
        img_ing = rendering.fill_coords(
            img, ingredient_fns[i], INGREDIENT_COLORS[ingredient_idx]
        )
        img = jax.lax.select(ingredient_idx != -1, img_ing, img)

    pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
    lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
    handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

    lid_fn_open = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
    handle_fn_open = rendering.rotate_fn(handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
    pot_open = is_idle & has_ingredients

    img = rendering.fill_coords(img, pot_fn, COLORS["black"])

    img_closed = rendering.fill_coords(img, lid_fn, COLORS["black"])
    img_closed = rendering.fill_coords(img_closed, handle_fn, COLORS["black"])

    img_open = rendering.fill_coords(img, lid_fn_open, COLORS["black"])
    img_open = rendering.fill_coords(img_open, handle_fn_open, COLORS["black"])

    img = jax.lax.select(pot_open, img_open, img_closed)

    progress_fn = rendering.point_in_rect(
        0.1, 0.9 - (0.9 - 0.1) / POT_COOK_TIME * time_left, 0.83, 0.88
    )
    img_timer = rendering.fill_coords(img, progress_fn, COLORS["green"])
    img = jax.lax.select(is_cooking, img_timer, img)

    return img


def _render_cell(cell, img):
    static_object = cell[0]

    def _render_empty(cell, img):
        return img

    def _render_wall(cell, img):
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
        )
        img = _render_dynamic_item(cell[1], img)
        return img

    def _render_agent(cell, img):
        tri_fn = rendering.point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )

        direction, idx = _decode_agent_extras(cell[2])

        # A bit hacky, but needed so that actions order matches Overcooked-AI
        direction_reordering = jnp.array([3, 1, 0, 2])
        direction = direction_reordering[direction]

        agent_color = AGENT_COLORS[idx]

        tri_fn = rendering.rotate_fn(
            tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction
        )
        img = rendering.fill_coords(img, tri_fn, agent_color)

        img = _render_dynamic_item(
            cell[1],
            img,
            plate_fn=rendering.point_in_circle(0.75, 0.75, 0.2),
            ingredient_fn=rendering.point_in_circle(0.75, 0.75, 0.15),
            dish_positions=jnp.array([(0.65, 0.65), (0.85, 0.65), (0.75, 0.85)]),
        )

        return img

    def _render_agent_self(cell, img):
        return img

    def _render_goal(cell, img):
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
        )
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["green"]
        )
        return img

    def _render_pot_cell(cell, img):
        return _render_pot(cell, img)

    def _render_recipe_indicator(cell, img):
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
        )
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"]
        )
        img = _render_dynamic_item(cell[1], img)
        return img

    def _render_button_recipe_indicator(cell, img):
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
        )
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"]
        )
        img = _render_dynamic_item(cell[1], img)

        time_left = cell[2]
        progress_fn = rendering.point_in_rect(
            0.1,
            0.9 - (0.9 - 0.1) / INDICATOR_ACTIVATION_TIME * time_left,
            0.83,
            0.88,
        )
        img_timer = rendering.fill_coords(img, progress_fn, COLORS["green"])

        button_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
        img_button = rendering.fill_coords(img, button_fn, COLORS["red"])

        img = jax.lax.select(time_left > 0, img_timer, img_button)
        return img

    def _render_plate_pile(cell, img):
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
        )
        plate_fns = [
            rendering.point_in_circle(*coord, 0.2)
            for coord in [(0.3, 0.3), (0.75, 0.42), (0.4, 0.75)]
        ]
        for plate_fn in plate_fns:
            img = rendering.fill_coords(img, plate_fn, COLORS["white"])
        return img

    def _render_ingredient_pile(cell, img):
        ingredient_idx = cell[0] - StaticObject.INGREDIENT_PILE_BASE

        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
        )
        ingredient_fns = [
            rendering.point_in_circle(*coord, 0.15)
            for coord in [
                (0.5, 0.15),
                (0.3, 0.4),
                (0.8, 0.35),
                (0.4, 0.8),
                (0.75, 0.75),
            ]
        ]

        for ingredient_fn in ingredient_fns:
            img = rendering.fill_coords(
                img, ingredient_fn, INGREDIENT_COLORS[ingredient_idx]
            )

        return img

    render_fns_dict = {
        StaticObject.EMPTY: _render_empty,
        StaticObject.WALL: _render_wall,
        StaticObject.AGENT: _render_agent,
        StaticObject.SELF_AGENT: _render_agent_self,
        StaticObject.GOAL: _render_goal,
        StaticObject.POT: _render_pot_cell,
        StaticObject.RECIPE_INDICATOR: _render_recipe_indicator,
        StaticObject.BUTTON_RECIPE_INDICATOR: _render_button_recipe_indicator,
        StaticObject.PLATE_PILE: _render_plate_pile,
    }

    render_fns = [_render_empty] * (max(render_fns_dict.keys()) + 2)
    for key, value in render_fns_dict.items():
        render_fns[key] = value
    render_fns[-1] = _render_ingredient_pile

    branch_idx = jnp.clip(static_object, 0, len(render_fns) - 1)

    return jax.lax.switch(branch_idx, render_fns, cell, img)


def _render_tile(obj, tile_size, subdivs):
    img = jnp.zeros(
        shape=(tile_size * subdivs, tile_size * subdivs, 3),
        dtype=jnp.uint8,
    )

    # Grid lines (top and left edges)
    img = rendering.fill_coords(
        img, rendering.point_in_rect(0, 0.031, 0, 1), COLORS["grey"]
    )
    img = rendering.fill_coords(
        img, rendering.point_in_rect(0, 1, 0, 0.031), COLORS["grey"]
    )

    img = _render_cell(obj, img)

    img = rendering.downsample(img, subdivs)
    return img


def _bake_agents(grid, agents, recipe):
    """Place agents into the grid and reveal the recipe on indicator cells."""
    num_agents = agents.dir.shape[0]

    def _include_agents(grid, x):
        agent, idx = x
        pos = agent.pos
        inventory = agent.inventory
        direction = agent.dir
        extra_info = _encode_agent_extras(direction, idx)
        new_grid = grid.at[pos.y, pos.x].set([StaticObject.AGENT, inventory, extra_info])
        return new_grid, None

    grid, _ = jax.lax.scan(_include_agents, grid, (agents, jnp.arange(num_agents)))

    static_objects = grid[:, :, 0]
    ingredients = grid[:, :, 1]
    extra_info = grid[:, :, 2]

    recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
    button_recipe_indicator_mask = (
        static_objects == StaticObject.BUTTON_RECIPE_INDICATOR
    ) & (extra_info > 0)

    new_ingredients_layer = jnp.where(
        recipe_indicator_mask | button_recipe_indicator_mask,
        recipe | DynamicObject.COOKED | DynamicObject.PLATE,
        ingredients,
    )
    grid = grid.at[:, :, 1].set(new_ingredients_layer)
    return grid


def render_state(state, tile_size=TILE_PIXELS, subdivs=SUBDIVS):
    """Render the full god's-eye RGB image with agents baked in.

    Returns a `(height * tile_size, width * tile_size, 3)` uint8 array. View
    masking and the ego marker are the wrapper's responsibility.
    """
    grid = _bake_agents(state.grid, state.agents, state.recipe)

    img_grid = jax.vmap(jax.vmap(lambda obj: _render_tile(obj, tile_size, subdivs)))(grid)
    grid_rows, grid_cols, tile_h, tile_w, channels = img_grid.shape
    big_image = img_grid.transpose(0, 2, 1, 3, 4).reshape(
        grid_rows * tile_h, grid_cols * tile_w, channels
    )
    return big_image


def render_eval_frames(ep_states, tile_size=4 * TILE_PIXELS, chunk=64):
    """Render an eval rollout (list of WrappedEnvState) to god's-eye RGB frames.

    Used by the eval/checkpoint-video path; rendered at a larger tile than the
    training observation for legibility. The per-timestep states are stacked and
    rendered with a jitted `vmap` (in chunks to bound peak memory) — rendering
    each frame eagerly in a Python loop is ~100x slower.
    """
    env_states = [s.env_state for s in ep_states]
    batched = jax.tree.map(lambda *xs: jnp.stack(xs), *env_states)
    render_v = jax.jit(jax.vmap(lambda st: render_state(st, tile_size)))
    total = len(env_states)
    out = []
    for i in range(0, total, chunk):
        sl = jax.tree.map(lambda x: x[i:i + chunk], batched)
        out.append(np.asarray(render_v(sl)).astype(np.uint8))
    return list(np.concatenate(out, axis=0))
