"""Utility functions for the Joint Attention mechanism.

Provides:
- JSD (Jensen-Shannon Divergence) between attention distributions
- Inferred attention map from an agent's (position, direction)
- Spatial basis generation for positional encoding
"""
import jax
import jax.numpy as jnp


def jsd_divergence(p: jnp.ndarray, q: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Jensen-Shannon Divergence between two distributions.

    Args:
        p: distribution of shape (..., H, W), must sum to 1 over last two dims
        q: distribution of shape (..., H, W), must sum to 1 over last two dims
        eps: small constant for numerical stability

    Returns:
        JSD value (scalar or batch), non-negative. 0 when p == q.
    """
    m = 0.5 * (p + q)
    kl_pm = jnp.sum(p * jnp.log(p / (m + eps) + eps), axis=(-2, -1))
    kl_qm = jnp.sum(q * jnp.log(q / (m + eps) + eps), axis=(-2, -1))
    return 0.5 * kl_pm + 0.5 * kl_qm


def _cone_forward_lateral(h: int, w: int, pos_xy: jnp.ndarray, dir_idx: jnp.ndarray):
    """Compute forward and lateral distances for every cell in an (H, W) grid.

    Uses the same convention as overcooked_po_wrapper._cone_forward_lateral.

    Args:
        h: grid height (number of rows)
        w: grid width (number of columns)
        pos_xy: agent position as (x, y)
        dir_idx: facing direction index (0=N, 1=S, 2=E, 3=W)

    Returns:
        (forward, lateral) arrays of shape (H, W)
    """
    x0, y0 = pos_xy[0], pos_xy[1]
    xs = jnp.arange(w)[None, :]   # columns = x
    ys = jnp.arange(h)[:, None]   # rows = y
    dx = xs - x0
    dy = ys - y0

    # DIR_TO_VEC: 0=NORTH(0,-1), 1=SOUTH(0,1), 2=EAST(1,0), 3=WEST(-1,0)
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


def inferred_attention(
    pos_xy: jnp.ndarray,
    dir_idx: jnp.ndarray,
    h: int,
    w: int,
    dist_sigma: float = 3.0,
    ang_sigma: float = 1.5,
) -> jnp.ndarray:
    """Construct an attention distribution from an agent's position and direction.

    Models the idea that an agent attends most to what is directly in front,
    with Gaussian decay in distance and off-axis angle. The result is normalized
    to form a proper probability distribution over (H, W).

    Args:
        pos_xy: agent position (x, y)
        dir_idx: facing direction index (0=N, 1=S, 2=E, 3=W)
        h: grid height
        w: grid width
        dist_sigma: softness of distance decay
        ang_sigma: softness of angular decay

    Returns:
        Normalized attention map of shape (H, W), sums to 1.
    """
    forward, lateral = _cone_forward_lateral(h, w, pos_xy, dir_idx)

    forward_pos = jnp.maximum(forward, 0.0)
    # Distance weight: stronger close, decays with forward distance
    w_dist = jnp.exp(-forward_pos / jnp.maximum(dist_sigma, 1e-6))
    # Angular weight: stronger on-axis, penalizes off-center
    lat_norm = lateral / (forward_pos + 1.0)
    w_ang = jnp.exp(-(lat_norm ** 2) / jnp.maximum(ang_sigma ** 2, 1e-6))

    # Only attend to cells in front (forward >= 0)
    in_front = (forward >= 0).astype(jnp.float32)
    weights = w_dist * w_ang * in_front

    # Normalize to a probability distribution
    return weights / (jnp.sum(weights) + 1e-8)


def make_spatial_basis(h: int, w: int) -> jnp.ndarray:
    """Create a fixed spatial basis tensor encoding (x, y) position.

    Following Mott et al. (2019), this provides positional information
    that is retained after the attention-weighted sum compresses spatial dims.

    Args:
        h: grid height
        w: grid width

    Returns:
        Spatial basis of shape (H, W, 2) with normalized x and y coordinates.
    """
    xs = jnp.arange(w, dtype=jnp.float32) / max(w - 1, 1)
    ys = jnp.arange(h, dtype=jnp.float32) / max(h - 1, 1)
    grid_x, grid_y = jnp.meshgrid(xs, ys)  # both (H, W)
    return jnp.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)
