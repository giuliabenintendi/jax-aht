"""Policy-observation renderer for Overcooked V2.

This intentionally sits beside `envs.overcooked_v2.rendering`: eval videos use
the visual renderer, while image policies consume this simpler tile observation.
The two renderers share state semantics and palettes, but not tile artwork.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp

import envs.overcooked_v2.grid_rendering_v2 as gr
from envs.overcooked_v2.common import DynamicObject, StaticObject
from envs.overcooked_v2.rendering import (
    AGENT_COLORS,
    COLORS,
    INGREDIENT_COLORS,
    TILE_PIXELS,
)
from envs.overcooked_v2.settings import INDICATOR_ACTIVATION_TIME, POT_COOK_TIME


def _encode_agent_extras(direction, idx):
    return direction | (idx << 2)


def _decode_agent_extras(extras):
    direction = extras & 0x3
    idx = extras >> 2
    return direction, idx


def _bake_agents(grid, agents, recipe):
    """Place agents and visible recipe indicators into the observation grid."""
    num_agents = agents.dir.shape[0]

    def _include_agent(grid, x):
        agent, idx = x
        extra_info = _encode_agent_extras(agent.dir, idx)
        new_grid = grid.at[agent.pos.y, agent.pos.x].set(
            [StaticObject.AGENT, agent.inventory, extra_info]
        )
        return new_grid, None

    grid, _ = jax.lax.scan(_include_agent, grid, (agents, jnp.arange(num_agents)))

    static_objects = grid[:, :, 0]
    ingredients = grid[:, :, 1]
    extra_info = grid[:, :, 2]

    recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
    button_recipe_indicator_mask = (
        static_objects == StaticObject.BUTTON_RECIPE_INDICATOR
    ) & (extra_info > 0)
    recipe_dish = recipe | DynamicObject.COOKED | DynamicObject.PLATE
    ingredients = jnp.where(
        recipe_indicator_mask | button_recipe_indicator_mask,
        recipe_dish,
        ingredients,
    )
    return grid.at[:, :, 1].set(ingredients)


def _paint_dynamic_item(img, ingredients, ingredient_colors=None, *, small=False):
    colors = INGREDIENT_COLORS if ingredient_colors is None else ingredient_colors
    plate_r = 0.22 if small else 0.30
    ingredient_r = 0.13 if small else 0.17
    centre = (0.72, 0.72) if small else (0.5, 0.5)
    dish_positions = (
        jnp.array([(0.63, 0.65), (0.82, 0.65), (0.72, 0.84)])
        if small
        else jnp.array([(0.42, 0.42), (0.58, 0.42), (0.50, 0.60)])
    )

    def _none(img, ingredients):
        return img

    def _plate(img, ingredients):
        return gr.fill_coords(img, gr.point_in_circle(*centre, plate_r), COLORS["white"])

    def _ingredient(img, ingredients):
        idx = DynamicObject.get_ingredient_idx(ingredients)
        return gr.fill_coords(
            img, gr.point_in_circle(*centre, ingredient_r), colors[idx]
        )

    def _dish(img, ingredients):
        img = gr.fill_coords(img, gr.point_in_circle(*centre, plate_r), COLORS["white"])
        ingredient_indices = DynamicObject.get_ingredient_idx_list_jit(ingredients)
        for i, ingredient_idx in enumerate(ingredient_indices):
            marker = gr.fill_coords(
                img,
                gr.point_in_circle(dish_positions[i, 0], dish_positions[i, 1], 0.09),
                colors[ingredient_idx],
            )
            img = jax.lax.select(ingredient_idx != -1, marker, img)
        return img

    branches = jnp.array(
        [
            ingredients == DynamicObject.EMPTY,
            ingredients == DynamicObject.PLATE,
            DynamicObject.is_ingredient(ingredients),
            (ingredients & DynamicObject.COOKED) != 0,
        ]
    )
    return jax.lax.switch(
        jnp.argmax(branches),
        [_none, _plate, _ingredient, _dish],
        img,
        ingredients,
    )


def _paint_pot(cell, img, ingredient_colors=None):
    colors = INGREDIENT_COLORS if ingredient_colors is None else ingredient_colors
    ingredients = cell[1]
    time_left = cell[2]
    ingredient_indices = DynamicObject.get_ingredient_idx_list_jit(ingredients)

    img = gr.fill_coords(img, gr.point_in_rect(0, 1, 0, 1), COLORS["grey"])
    img = gr.fill_coords(img, gr.point_in_rect(0.12, 0.88, 0.35, 0.86), COLORS["black"])
    img = gr.fill_coords(img, gr.point_in_rect(0.18, 0.82, 0.22, 0.30), COLORS["black"])

    spots = jnp.array([(0.28, 0.42), (0.50, 0.42), (0.72, 0.42)])
    for i, ingredient_idx in enumerate(ingredient_indices):
        marker = gr.fill_coords(
            img,
            gr.point_in_circle(spots[i, 0], spots[i, 1], 0.11),
            colors[ingredient_idx],
        )
        img = jax.lax.select(ingredient_idx != -1, marker, img)

    cooked = (ingredients & DynamicObject.COOKED) != 0
    cooking = time_left > 0
    cooked_img = gr.fill_coords(
        img, gr.point_in_rect(0.18, 0.82, 0.46, 0.78), COLORS["orange"]
    )
    img = jax.lax.select(cooked, cooked_img, img)

    progress = 0.88 - (0.88 - 0.12) / POT_COOK_TIME * time_left
    timer_img = gr.fill_coords(
        img, gr.point_in_rect(0.12, progress, 0.88, 0.94), COLORS["green"]
    )
    return jax.lax.select(cooking, timer_img, img)


def _paint_cell(cell, img, ingredient_colors=None):
    static_object = cell[0]

    def _empty(cell, img):
        return img

    def _wall(cell, img):
        img = gr.fill_coords(img, gr.point_in_rect(0, 1, 0, 1), COLORS["grey"])
        return _paint_dynamic_item(img, cell[1], ingredient_colors)

    def _agent(cell, img):
        direction, idx = _decode_agent_extras(cell[2])
        direction_reordering = jnp.array([3, 1, 0, 2])
        direction = direction_reordering[direction]
        tri_fn = gr.point_in_triangle((0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
        tri_fn = gr.rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction)
        img = gr.fill_coords(img, tri_fn, AGENT_COLORS[idx])
        return _paint_dynamic_item(img, cell[1], ingredient_colors, small=True)

    def _agent_self(cell, img):
        return img

    def _goal(cell, img):
        return gr.fill_coords(img, gr.point_in_rect(0, 1, 0, 1), COLORS["green"])

    def _pot(cell, img):
        return _paint_pot(cell, img, ingredient_colors)

    def _recipe_indicator(cell, img):
        img = gr.fill_coords(img, gr.point_in_rect(0, 1, 0, 1), COLORS["brown"])
        return _paint_dynamic_item(img, cell[1], ingredient_colors)

    def _button_recipe_indicator(cell, img):
        img = _recipe_indicator(cell, img)
        time_left = cell[2]
        progress = 0.88 - (0.88 - 0.12) / INDICATOR_ACTIVATION_TIME * time_left
        active = gr.fill_coords(
            img, gr.point_in_rect(0.12, progress, 0.88, 0.94), COLORS["green"]
        )
        inactive = gr.fill_coords(img, gr.point_in_circle(0.5, 0.5, 0.18), COLORS["red"])
        return jax.lax.select(time_left > 0, active, inactive)

    def _plate_pile(cell, img):
        img = gr.fill_coords(img, gr.point_in_rect(0, 1, 0, 1), COLORS["grey"])
        for coord in [(0.30, 0.30), (0.72, 0.42), (0.42, 0.75)]:
            img = gr.fill_coords(img, gr.point_in_circle(*coord, 0.18), COLORS["white"])
        return img

    def _ingredient_pile(cell, img):
        colors = INGREDIENT_COLORS if ingredient_colors is None else ingredient_colors
        ingredient_idx = cell[0] - StaticObject.INGREDIENT_PILE_BASE
        img = gr.fill_coords(img, gr.point_in_rect(0, 1, 0, 1), COLORS["grey"])
        for coord in [(0.5, 0.15), (0.3, 0.4), (0.8, 0.35), (0.4, 0.8), (0.75, 0.75)]:
            img = gr.fill_coords(img, gr.point_in_circle(*coord, 0.14), colors[ingredient_idx])
        return img

    render_fns_dict = {
        StaticObject.EMPTY: _empty,
        StaticObject.WALL: _wall,
        StaticObject.AGENT: _agent,
        StaticObject.SELF_AGENT: _agent_self,
        StaticObject.GOAL: _goal,
        StaticObject.POT: _pot,
        StaticObject.RECIPE_INDICATOR: _recipe_indicator,
        StaticObject.BUTTON_RECIPE_INDICATOR: _button_recipe_indicator,
        StaticObject.PLATE_PILE: _plate_pile,
    }
    render_fns = [_empty] * (max(render_fns_dict.keys()) + 2)
    for key, value in render_fns_dict.items():
        render_fns[key] = value
    render_fns[-1] = _ingredient_pile
    branch_idx = jnp.clip(static_object, 0, len(render_fns) - 1)
    return jax.lax.switch(branch_idx, render_fns, cell, img)


def _paint_tile(cell, tile_size, ingredient_colors=None):
    img = jnp.zeros((tile_size, tile_size, 3), dtype=jnp.uint8)
    img = gr.fill_coords(img, gr.point_in_rect(0, 0.04, 0, 1), COLORS["grey"])
    img = gr.fill_coords(img, gr.point_in_rect(0, 1, 0, 0.04), COLORS["grey"])
    return _paint_cell(cell, img, ingredient_colors)


def render_obs_state(state, tile_size=TILE_PIXELS, ingredient_colors=None):
    """Render the image policy observation canvas before per-agent FOV transforms."""
    grid = _bake_agents(state.grid, state.agents, state.recipe)
    img_grid = jax.vmap(
        jax.vmap(lambda obj: _paint_tile(obj, tile_size, ingredient_colors))
    )(grid)
    grid_rows, grid_cols, tile_h, tile_w, channels = img_grid.shape
    return img_grid.transpose(0, 2, 1, 3, 4).reshape(
        grid_rows * tile_h, grid_cols * tile_w, channels
    )
