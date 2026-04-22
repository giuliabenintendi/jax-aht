"""Shared action helpers for the static five-card coordination task.

Communication action space (size 2 * NUM_CARDS = 10):
  0..NUM_CARDS-1          — pick card (decision step only)
  NUM_CARDS..2*NUM_CARDS-1 — send message "card k" (deliberation step only)

No-communication action space (size NUM_CARDS + 1 = 6):
  0..NUM_CARDS-1 — pick card (decision step only)
  NUM_CARDS      — noop (deliberation step only)
"""

import jax.numpy as jnp

from envs.card_game.rendering import NUM_CARDS


NO_COMM_NOOP_ACTION = NUM_CARDS
NO_COMM_ACTION_DIM = NUM_CARDS + 1

COMM_MESSAGE_BASE = NUM_CARDS
COMM_ACTION_DIM = 2 * NUM_CARDS


def decode_pick_or_noop(action):
    """Decode the non-communication action space into a pick or noop."""
    action = jnp.asarray(action, dtype=jnp.int32)
    return jnp.where(action < NUM_CARDS, action, jnp.int32(-1))


def decode_comm_action(action):
    """Decode the communication action space into (pick, message)."""
    action = jnp.asarray(action, dtype=jnp.int32)
    is_pick = action < NUM_CARDS
    is_message = (action >= COMM_MESSAGE_BASE) & (action < COMM_ACTION_DIM)
    pick = jnp.where(is_pick, action, jnp.int32(-1))
    message = jnp.where(is_message, action - COMM_MESSAGE_BASE, jnp.int32(-1))
    return pick, message


def remap_recoloured_action(action, inv_recolouring, communication):
    """Map a recoloured-space action back to ground-truth color identity.

    Strict: actions that decode to neither pick nor message pass through raw,
    so upstream mask bugs surface instead of being silently remapped.
    """
    action = jnp.asarray(action, dtype=jnp.int32)

    if communication:
        pick, message = decode_comm_action(action)
        remapped_pick = jnp.where(pick >= 0, inv_recolouring[pick], jnp.int32(-1))
        remapped_message = jnp.where(
            message >= 0, inv_recolouring[message], jnp.int32(-1)
        )
        return jnp.where(
            pick >= 0,
            remapped_pick,
            jnp.where(
                message >= 0,
                remapped_message + COMM_MESSAGE_BASE,
                action,
            ),
        )

    pick = decode_pick_or_noop(action)
    return jnp.where(pick >= 0, inv_recolouring[pick], action)


def get_action_mask(is_decision, communication):
    """Return the valid action mask for the current phase."""
    is_decision = jnp.asarray(is_decision)
    pick_avail = jnp.where(is_decision, jnp.ones(NUM_CARDS), jnp.zeros(NUM_CARDS))

    if communication:
        msg_avail = jnp.where(is_decision, jnp.zeros(NUM_CARDS), jnp.ones(NUM_CARDS))
        return jnp.concatenate([pick_avail, msg_avail])

    noop_avail = jnp.where(is_decision, jnp.zeros(1), jnp.ones(1))
    return jnp.concatenate([pick_avail, noop_avail])
