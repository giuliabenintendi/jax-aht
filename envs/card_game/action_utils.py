"""Action helpers for the card-coordination game.

The action space is unified: a single Discrete(NUM_CARDS) at every step.
The same action is interpreted as a message during deliberation and as a
pick on the decision step — the env's `is_decision` flag does the routing,
not the action layout.
"""

import jax.numpy as jnp


def remap_recoloured_action(action, inv_recolouring):
    """Map a recoloured-space action back to ground-truth color identity."""
    action = jnp.asarray(action, dtype=jnp.int32)
    return inv_recolouring[action]
