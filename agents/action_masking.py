"""Utilities for applying action masks to policy logits."""

import jax.numpy as jnp


def mask_action_logits(action_logits: jnp.ndarray, avail_actions: jnp.ndarray,
                       clip_value: float = 20.0) -> jnp.ndarray:
    """Clip logits for stability, then hard-mask unavailable actions.

    Masking must happen after clipping. If we clip after subtracting a large
    penalty, masked logits can be raised back up to the clip floor and tie with
    valid logits under greedy argmax.
    """
    clipped_logits = jnp.clip(action_logits, -clip_value, clip_value)
    invalid_floor = jnp.finfo(clipped_logits.dtype).min
    return jnp.where(avail_actions > 0, clipped_logits, invalid_floor)
