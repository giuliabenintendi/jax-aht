"""Utility functions for the Joint Attention mechanism.

Implements the approach from Lee et al. (2021):
  "Joint Attention for Multi-Agent Coordination and Social Learning"

Provides:
- JSD (Jensen-Shannon Divergence) between attention distributions
- Sinusoidal 2D spatial basis (positional encoding)
- Inferred attention map from agent (position, direction) — used by PO ego training
"""
import jax.numpy as jnp

from envs.overcooked.po_utils import cone_forward_lateral


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
    kl_pm = jnp.sum(p * (jnp.log(p + eps) - jnp.log(m + eps)), axis=(-2, -1))
    kl_qm = jnp.sum(q * (jnp.log(q + eps) - jnp.log(m + eps)), axis=(-2, -1))
    return 0.5 * kl_pm + 0.5 * kl_qm


def make_sinusoidal_spatial_basis(h: int, w: int, depth: int = 8) -> jnp.ndarray:
    """Sinusoidal 2D positional encoding, ported from the reference code.

    Direct JAX port of `get_spatial_basis(h, w, d)` from
    `social_rl/multiagent_tfagents/joint_attention/attention_networks.py`.

    First half of channels encodes the height (row) coordinate,
    second half encodes the width (column) coordinate, each with
    interleaved sin/cos pairs at geometrically spaced frequencies.

    NOTE: Two discrepancies between the paper's equations (Appendix A.1)
    and the reference code (we follow the code, which is what actually ran):

      1. Frequency indexing: The paper uses i=1..c_s/4 (1-indexed), giving
         frequencies [1/10, 1/100] for c_s=8. The code uses arange(0, half_d, 2)
         (0-indexed), giving frequencies [1/1, 1/10]. The code has an extra
         high-frequency component and misses the lowest frequency.

      2. Coordinate order: The paper says first half encodes x (horizontal/width).
         The code puts h_grid (height/vertical) first. This is just an ordering
         difference — the network learns to use whichever channel is which.

    Reference code:
        div = exp(arange(0, half_d, 2) * -log(100) / half_d)
        basis[:,:,0:half_d:2]    = sin(h_grid * div)
        basis[:,:,1:half_d:2]    = cos(h_grid * div)
        basis[:,:,half_d::2]     = sin(w_grid * div)
        basis[:,:,half_d+1::2]   = cos(w_grid * div)

    Args:
        h: grid height
        w: grid width
        depth: total number of encoding channels (default 8, matching
               "spatial basis of depth 8" from the paper's Appendix A)

    Returns:
        Spatial basis of shape (H, W, depth).
    """
    half_d = depth // 2

    # Frequency divisors: same formula as the reference code
    # div = exp(arange(0, half_d, 2) * -log(100) / half_d)
    # This gives half_d/2 frequencies geometrically spaced from 1.0 to 1/100
    div = jnp.exp(
        jnp.arange(0, half_d, 2, dtype=jnp.float32) * (-jnp.log(100.0) / half_d)
    )  # (half_d // 2,)

    h_grid = jnp.arange(h, dtype=jnp.float32)[:, None]  # (H, 1)
    w_grid = jnp.arange(w, dtype=jnp.float32)[:, None]  # (W, 1)

    # Height encoding: sin/cos interleaved, broadcast across width
    h_sin = jnp.sin(h_grid * div)  # (H, half_d//2)
    h_cos = jnp.cos(h_grid * div)  # (H, half_d//2)
    # Interleave: [sin_0, cos_0, sin_1, cos_1, ...]
    h_enc = jnp.stack([h_sin, h_cos], axis=-1).reshape(h, half_d)  # (H, half_d)
    h_enc = jnp.broadcast_to(h_enc[:, None, :], (h, w, half_d))  # (H, W, half_d)

    # Width encoding: sin/cos interleaved, broadcast across height
    w_sin = jnp.sin(w_grid * div)  # (W, half_d//2)
    w_cos = jnp.cos(w_grid * div)  # (W, half_d//2)
    w_enc = jnp.stack([w_sin, w_cos], axis=-1).reshape(w, half_d)  # (W, half_d)
    w_enc = jnp.broadcast_to(w_enc[None, :, :], (h, w, half_d))  # (H, W, half_d)

    return jnp.concatenate([h_enc, w_enc], axis=-1)  # (H, W, depth)



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

    Used by PO ego training (ppo_ego.py) for inferred partner attention.
    Not used by the FO JA-IPPO path, which compares learned attention maps.

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
    forward, lateral = cone_forward_lateral(h, w, pos_xy, dir_idx)

    forward_pos = jnp.maximum(forward, 0.0)
    # Distance weight: stronger close, decays with forward distance
    w_dist = jnp.exp(-forward_pos / jnp.maximum(dist_sigma, 1e-6))
    # Angular weight: stronger on-axis, penalizes off-center
    lat_norm = lateral / (forward_pos + 1.0)
    w_ang = jnp.exp(-(lat_norm ** 2) / jnp.maximum(ang_sigma ** 2, 1e-6))

    # Only attend to cells in front (forward >= 0)
    in_front = (forward >= 0).astype(jnp.float32)
    weights = w_dist * w_ang * in_front

    return weights / (jnp.sum(weights) + 1e-8)
