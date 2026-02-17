"""Grid geometry utilities for partial observability and attention.

Provides the world-to-agent coordinate transform used by both
the PO environment wrapper (FOV masking) and the JA mechanism
(inferred partner attention).
"""
import jax.numpy as jnp


def cone_forward_lateral(
    h: int, w: int, pos_xy: jnp.ndarray, dir_idx: jnp.ndarray
):
    """Transform grid displacements into agent-relative (forward, lateral).

    For every cell in an (H, W) grid, computes how far it is "ahead" and
    "sideways" relative to an agent at `pos_xy` facing `dir_idx`.

    Direction convention (matches OvercookedV1 DIR_TO_VEC):
        0=NORTH (0,-1), 1=SOUTH (0,1), 2=EAST (1,0), 3=WEST (-1,0)

    Args:
        h: grid height (rows)
        w: grid width (columns)
        pos_xy: agent position as (x, y)
        dir_idx: facing direction index

    Returns:
        (forward, lateral) arrays of shape (H, W).
        Positive forward = ahead of agent, negative = behind.
    """
    x0, y0 = pos_xy[0], pos_xy[1]
    xs = jnp.arange(w)[None, :]  # columns = x
    ys = jnp.arange(h)[:, None]  # rows = y
    dx = xs - x0
    dy = ys - y0

    f_n, l_n = -dy, dx
    f_s, l_s = dy, dx
    f_e, l_e = dx, dy
    f_w, l_w = -dx, dy

    forward = jnp.select(
        [dir_idx == 0, dir_idx == 1, dir_idx == 2, dir_idx == 3],
        [f_n, f_s, f_e, f_w],
        default=f_e,
    )
    lateral = jnp.select(
        [dir_idx == 0, dir_idx == 1, dir_idx == 2, dir_idx == 3],
        [l_n, l_s, l_e, l_w],
        default=l_e,
    )
    return forward, lateral


def fov_cone_mask(
    h: int, w: int, pos_xy: jnp.ndarray, dir_idx: jnp.ndarray,
    fov_range: int, fov_slope: float = 0.7,
) -> jnp.ndarray:
    """Boolean FOV mask (H, W) for the cone geometry.

    A cell is visible if it is in front, within range, and inside the cone angle.
    """
    forward, lateral = cone_forward_lateral(h, w, pos_xy, dir_idx)
    in_front = forward >= 0
    in_range = forward <= fov_range
    in_cone = jnp.abs(lateral) <= (fov_slope * forward + 1.0)
    return in_front & in_range & in_cone
