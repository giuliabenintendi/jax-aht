import distrax
import jax
import jax.numpy as jnp

from agents.action_masking import mask_action_logits
from envs import make_env


def test_mask_action_logits_prevents_greedy_pick_of_unavailable_action():
    raw_logits = jnp.full((1, 1, 11), -100.0, dtype=jnp.float32)
    avail_actions = jnp.array([0.0, 1.0, 1.0] + [0.0] * 8, dtype=jnp.float32)

    masked_logits = mask_action_logits(raw_logits, avail_actions)
    pi = distrax.Categorical(logits=masked_logits)
    action = pi.mode()

    assert int(action.squeeze()) != 0
    assert masked_logits[..., 0].max() < masked_logits[..., 1:].max()


def test_card_game_dynamic_decision_mask_matches_present_colors():
    env = make_env("card-game-dynamic", {"max_steps": 1})
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)

    avail = env.get_avail_actions(state)
    expected = state.env_state.card_present.astype(jnp.float32)

    for agent in env.agents:
        agent_mask = avail[agent]
        assert agent_mask.shape == (env.num_colors + 1,)
        assert jnp.array_equal(agent_mask[:env.num_colors], expected)
        assert float(agent_mask[env.num_colors]) == 0.0
