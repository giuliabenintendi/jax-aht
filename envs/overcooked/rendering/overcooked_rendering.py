"""OGC tile-based renderer. Copied from ogc/overcooked_rendering.py.

Produces (H*7, W*7, 3) uint8 RGB images from OvercookedState.
The @jax.jit decorator on render_state is removed so it can be called
inside traced code (wrappers, training loops).
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from jaxmarl.environments.overcooked.common import (
    OBJECT_TO_INDEX,
    COLOR_TO_INDEX,
    COLORS,
)

TILE_PIXELS = 7
INDEX_TO_COLOR = [k for k, v in COLOR_TO_INDEX.items()]

JAX_COLORS = np.stack([COLORS[k] for k in INDEX_TO_COLOR])
JAX_COLORS = jnp.array(JAX_COLORS)

COLOR_TO_AGENT_INDEX = {0: 0, 2: 1}  # Hardcoded. Red is first, blue is second


# --- 3x5 digit font (rows=5, cols=3), 1 means "draw pixel" ---
_DIGITS_3x5 = jnp.array([
    # 0
    [[1,1,1],
     [1,0,1],
     [1,0,1],
     [1,0,1],
     [1,1,1]],
    # 1
    [[0,1,0],
     [1,1,0],
     [0,1,0],
     [0,1,0],
     [1,1,1]],
    # 2
    [[1,1,1],
     [0,0,1],
     [1,1,1],
     [1,0,0],
     [1,1,1]],
    # 3
    [[1,1,1],
     [0,0,1],
     [0,1,1],
     [0,0,1],
     [1,1,1]],
    # 4
    [[1,0,1],
     [1,0,1],
     [1,1,1],
     [0,0,1],
     [0,0,1]],
    # 5
    [[1,1,1],
     [1,0,0],
     [1,1,1],
     [0,0,1],
     [1,1,1]],
    # 6
    [[1,1,1],
     [1,0,0],
     [1,1,1],
     [1,0,1],
     [1,1,1]],
    # 7
    [[1,1,1],
     [0,0,1],
     [0,1,0],
     [0,1,0],
     [0,1,0]],
    # 8
    [[1,1,1],
     [1,0,1],
     [1,1,1],
     [1,0,1],
     [1,1,1]],
    # 9
    [[1,1,1],
     [1,0,1],
     [1,1,1],
     [0,0,1],
     [1,1,1]],
], dtype=jnp.bool_)  # (10,5,3)


def overlay_progress_4x5(img, k, color):
    # 7x7 indices: 0..6
    # Use a 4x5 block centered-ish: x=2..5 (4 cols), y=1..5 (5 rows) => 20 pixels.
    xs = jnp.array([2, 3, 4, 5], dtype=jnp.int32)
    ys = jnp.array([5, 4, 3, 2, 1], dtype=jnp.int32)  # bottom->top fill

    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")       # (5,4)
    coords = jnp.stack([yy, xx], axis=-1).reshape(-1, 2)  # (20,2) [y,x]

    k = jnp.clip(k, 0, coords.shape[0])
    mask = jnp.arange(coords.shape[0]) < k

    def body(i, im):
        y, x = coords[i, 0], coords[i, 1]
        return lax.cond(
            mask[i],
            lambda z: z.at[y, x, :].set(color),
            lambda z: z,
            im,
        )

    return lax.fori_loop(0, coords.shape[0], body, img)


def make_fill_coords(tile_size):
    y, x = jnp.meshgrid(
        jnp.arange(tile_size), jnp.arange(tile_size), indexing="ij"
    )
    yf = (y + 0.5) / tile_size
    xf = (x + 0.5) / tile_size

    def fc(img, fn, color):
        mask = fn(xf, yf)  # (tile_size, tile_size)
        return jnp.where(mask[:, :, None], color, img)

    return fc

FILL_COORDS_7 = make_fill_coords(TILE_PIXELS)


def render_fn(state):
    data = render_state(state, highlight=False)
    return data


def point_in_rect(xmin, xmax, ymin, ymax):
    def fn(x, y):
        return jnp.logical_and(
            jnp.logical_and(x >= xmin, x <= xmax), jnp.logical_and(
                y >= ymin, y <= ymax)
        )

    return fn


def point_in_circle(cx, cy, r):
    def fn(x, y):
        return (x - cx) ** 2 + (y - cy) ** 2 <= r**2

    return fn


def point_in_triangle(a, b, c):
    a, b, c = jnp.array(a), jnp.array(b), jnp.array(c)

    def fn(x, y):
        points = jnp.stack([x.ravel(), y.ravel()], axis=0)

        v0 = c - a
        v1 = b - a
        v2 = points - a[:, None]

        dot00 = jnp.dot(v0, v0)
        dot01 = jnp.dot(v0, v1)
        dot02 = jnp.sum(v0[:, None] * v2, axis=0)
        dot11 = jnp.dot(v1, v1)
        dot12 = jnp.sum(v1[:, None] * v2, axis=0)

        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        result = jnp.logical_and(jnp.logical_and(u >= 0, v >= 0), (u + v) < 1)

        return result.reshape(x.shape)

    return fn


def rotate_fn(fin, cx, cy, theta):
    def fout(x, y):
        x = x - cx
        y = y - cy
        x2 = cx + x * jnp.cos(-theta) - y * jnp.sin(-theta)
        y2 = cy + y * jnp.cos(-theta) + x * jnp.sin(-theta)
        return fin(x2, y2)

    return fout


def render_tile(obj, highlight, agent_dir_idx, agent_inv, tile_size=TILE_PIXELS):
    img = jnp.zeros((tile_size, tile_size, 3), dtype=jnp.uint8)

    # Render grid lines
    fn = point_in_rect(0, 0.031, 0, 1)
    img = FILL_COORDS_7(img, fn, jnp.array([100, 100, 100]))
    fn = point_in_rect(0, 1, 0, 0.031)
    img = FILL_COORDS_7(img, fn, jnp.array([100, 100, 100]))

    def render_obj(img, obj_type, color):
        def do_nothing(x):
            return x

        def render_wall(img):
            fn = point_in_rect(0, 1, 0, 1)
            img = FILL_COORDS_7(img, fn, color)
            return img

        def render_goal(img):
            fn = point_in_rect(0, 1, 0, 1)
            img = FILL_COORDS_7(img, fn, COLORS["grey"])
            fn = point_in_rect(0.1, 0.9, 0.1, 0.9)
            img = FILL_COORDS_7(img, fn, color)
            return img

        def render_agent(img):
            tri_fn = point_in_triangle(
                (0.12, 0.19), (0.87, 0.50), (0.12, 0.81))
            agent_idx_dir_map = jnp.array([1, 0, 3, 2], dtype=jnp.int32)
            tri_fn = rotate_fn(
                tri_fn, 0.5, 0.5, 0.5 * jnp.pi *
                agent_idx_dir_map[agent_dir_idx]
            )
            img = FILL_COORDS_7(img, tri_fn, color)

            def render_inventory(img):
                def _render_nothing(x):
                    return x

                def _render_mini_onion(img):
                    onion_fn = point_in_circle(0.75, 0.75, 0.15)
                    img = FILL_COORDS_7(img, onion_fn, COLORS["yellow"])
                    return img

                def _render_mini_plate(img):
                    plate_fn = point_in_circle(0.75, 0.75, 0.2)
                    img = FILL_COORDS_7(img, plate_fn, COLORS["white"])
                    return img

                def _render_mini_dish(img):
                    plate_fn = point_in_circle(0.75, 0.75, 0.2)
                    img = FILL_COORDS_7(img, plate_fn, COLORS["white"])
                    onion_fn = point_in_circle(0.75, 0.75, 0.13)
                    img = FILL_COORDS_7(img, onion_fn, COLORS["orange"])
                    return img

                img = lax.cond(
                    agent_inv == OBJECT_TO_INDEX["onion"],
                    _render_mini_onion,
                    _render_nothing,
                    img,
                )
                img = lax.cond(
                    agent_inv == OBJECT_TO_INDEX["plate"],
                    _render_mini_plate,
                    _render_nothing,
                    img,
                )
                img = lax.cond(
                    agent_inv == OBJECT_TO_INDEX["dish"],
                    _render_mini_dish,
                    _render_nothing,
                    img,
                )

                return img

            img = lax.cond(agent_inv != 1, render_inventory, lambda x: x, img)
            return img

        def render_empty(img):
            fn = point_in_rect(0, 1, 0, 1)
            img = FILL_COORDS_7(img, fn, COLORS["black"])
            return img

        def render_onion_pile(img):
            fn = point_in_rect(0, 1, 0, 1)
            img = FILL_COORDS_7(img, fn, COLORS["grey"])
            onion_coords = jnp.array(
                [(0.5, 0.15), (0.3, 0.4), (0.8, 0.35), (0.4, 0.8), (0.75, 0.75)]
            )

            def apply_onion(carry, coord_id):
                img, onion_coords = carry
                coord = onion_coords[coord_id]
                onion_fn = point_in_circle(*coord, 0.15)
                return (FILL_COORDS_7(img, onion_fn, color), onion_coords), None

            carry, _ = jax.lax.scan(
                apply_onion, (img, onion_coords), jnp.arange(len(onion_coords))
            )
            img = carry[0]
            return img

        def render_onion(img):
            fn = point_in_rect(0, 1, 0, 1)
            img = FILL_COORDS_7(img, fn, COLORS["grey"])
            onion_fn = point_in_circle(0.5, 0.5, 0.15)
            img = FILL_COORDS_7(img, onion_fn, color)
            return img

        def render_plate_pile(img):
            fn = point_in_rect(0, 1, 0, 1)
            img = FILL_COORDS_7(img, fn, COLORS["grey"])

            def apply_plate(carry, coord_id):
                img, plate_coords = carry
                coord = plate_coords[coord_id]
                plate_fn = point_in_circle(*coord, 0.2)
                return (FILL_COORDS_7(img, plate_fn, color), plate_coords), None

            plate_coords = jnp.array([(0.3, 0.3), (0.75, 0.42), (0.4, 0.75)])
            carry, _ = jax.lax.scan(
                apply_plate, (img, plate_coords), jnp.arange(len(plate_coords))
            )
            img = carry[0]
            return img

        def render_plate(img):
            fn = point_in_rect(0, 1, 0, 1)
            img = FILL_COORDS_7(img, fn, COLORS["grey"])
            plate_fn = point_in_circle(0.5, 0.5, 0.2)
            img = FILL_COORDS_7(img, plate_fn, color)
            return img

        def render_dish(img):
            fn = point_in_rect(0, 1, 0, 1)
            img = FILL_COORDS_7(img, fn, COLORS["grey"])
            plate_fn = point_in_circle(0.75, 0.75, 0.2)
            img = FILL_COORDS_7(img, plate_fn, COLORS["white"])
            onion_fn = point_in_circle(0.75, 0.75, 0.13)
            img = FILL_COORDS_7(img, onion_fn, COLORS["orange"])
            return img

        def render_pot(img):
            img = rendering_pot(obj, img)
            return img

        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["wall"], render_wall, do_nothing, img)

        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["goal"], render_goal, do_nothing, img)
        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["agent"], render_agent, do_nothing, img)
        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["empty"], render_empty, do_nothing, img)
        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["onion_pile"],
            render_onion_pile,
            do_nothing,
            img,
        )
        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["onion"], render_onion, do_nothing, img)
        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["plate_pile"],
            render_plate_pile,
            do_nothing,
            img,
        )
        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["plate"], render_plate, do_nothing, img)
        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["dish"], render_dish, do_nothing, img)
        img = lax.cond(
            obj_type == OBJECT_TO_INDEX["pot"], render_pot, do_nothing, img)

        return img

    img = render_obj(img, obj[0], JAX_COLORS[obj[1]])
    return img


def rendering_pot(obj, img):
    pot_status = obj[-1]
    num_onions = jnp.maximum(23 - pot_status, 0)
    is_cooking = (pot_status < 20) & (pot_status > 0)
    is_done = pot_status == 0
    steps_done = jnp.clip(20 - pot_status, 0, 20)

    pot_fn    = point_in_rect(0.1, 0.9, 0.33, 0.9)
    lid_fn    = point_in_rect(0.1, 0.9, 0.21, 0.25)
    com_fn    = point_in_rect(0.1, 0.9, 0.21, 0.9)
    handle_fn = point_in_rect(0.4, 0.6, 0.16, 0.21)

    interior_fn = point_in_rect(0.12, 0.88, 0.36, 0.88)

    # background
    img = FILL_COORDS_7(img, point_in_rect(0, 1, 0, 1), COLORS["grey"])

    # draw pot first
    img = FILL_COORDS_7(img, pot_fn, COLORS["black"])
    img = FILL_COORDS_7(img, lid_fn, COLORS["black"])
    img = FILL_COORDS_7(img, handle_fn, COLORS["black"])

    # onions / soup on top of pot
    def render_onions(img, num_onions):
        onion_fns = jnp.array([(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)])

        def render_onion(carry, i):
            img, coords, num_onions = carry
            coord = coords[i]
            onion_fn = point_in_circle(*coord, 0.13)
            img = jnp.where(
                i < num_onions, FILL_COORDS_7(
                    img, onion_fn, COLORS["yellow"]), img
            )  # show progress of onions filled
            return (img, coords, num_onions), None

        carry, _ = lax.scan(
            render_onion, (img, onion_fns, num_onions), jnp.arange(
                len(onion_fns))
        )
        img = carry[0]
        return img

    show_onions = (num_onions > 0) & (~is_cooking) & (~is_done)
    img = lax.cond(show_onions, render_onions, lambda x, y: x, img, num_onions)
    img = lax.cond(
        is_done,
        lambda x: FILL_COORDS_7(x, com_fn, COLORS["orange"]),
        lambda x: x,
        img,
    )

    img = lax.cond(
        is_cooking,
        lambda x: overlay_progress_4x5(x, steps_done, COLORS["purple"]),
        lambda x: x,
        img,
    )

    return img


def render_grid(grid, highlight_mask, agent_dir_idx, agent_inv, tile_size=TILE_PIXELS):
    """
    grid:           (H, W, ...)
    highlight_mask: (H, W) bool
    agent_dir_idx:  (num_agents,)
    agent_inv:      (num_agents,)
    """

    # Per-tile wrapper that keeps the agent_id logic local to the tile
    def render_tile_for_vmap(obj, highlight):
        # 2 is blue, 0 is red
        agent_id = jnp.where(obj[1] == 2, 1, 0)
        tile_img = render_tile(
            obj,
            highlight,
            agent_dir_idx[agent_id],
            agent_inv[agent_id],
            tile_size=tile_size,
        )
        return tile_img.astype(jnp.uint8)

    render_row = jax.vmap(render_tile_for_vmap, in_axes=(0, 0))
    tiles = jax.vmap(render_row, in_axes=(0, 0))(grid, highlight_mask)
    # tiles shape: (H, W, tile_size, tile_size, 3)

    H, W = grid.shape[:2]
    tiles = tiles.transpose(0, 2, 1, 3, 4)          # (H, tile, W, tile, 3)
    img = tiles.reshape(H * tile_size, W * tile_size, 3)

    return img


# No @jax.jit — called inside traced code
def render_state(state, highlight=False, tile_size=TILE_PIXELS, agent_view_size=5):
    padding = 4
    grid = state.maze_map[padding:-padding, padding:-padding, :]

    highlight_mask = jnp.zeros(grid.shape[:2], dtype=bool)

    img = render_grid(
        grid,
        highlight_mask,
        state.agent_dir_idx,
        state.agent_inv,
    )

    return img
