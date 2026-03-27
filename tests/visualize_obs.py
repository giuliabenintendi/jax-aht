"""Quick test to visualize card game observations at different steps."""
import os
os.makedirs("tests/obs_imgs", exist_ok=True)
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from envs.card_game.card_game import CardGameEnv

env = CardGameEnv(max_steps=8, communication=True)
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)

def save_obs(obs_dict, prefix):
    for agent in ["agent_0", "agent_1"]:
        flat = np.array(obs_dict[agent])
        img = (flat[:21*35*3].reshape(21, 35, 3) * 255).astype(np.uint8)
        Image.fromarray(img).resize((350, 210), Image.NEAREST).save(f"tests/obs_imgs/{prefix}_{agent}.png")
        print(f"Saved tests/obs_imgs/{prefix}_{agent}.png")

# Reset: no message, no decision square
save_obs(obs, "obs_reset")
print(f"Messages after reset: {np.array(state.env_state.messages)}")

# Step 1: send messages (actions 25-29 = msg only during deliberation)
# agent_0 sends msg=2, agent_1 sends msg=1
obs2, state2, _, _, _ = env.step(key, state, {"agent_0": jnp.int32(27), "agent_1": jnp.int32(26)})
save_obs(obs2, "obs_step1")
print(f"Messages after step1: {np.array(state2.env_state.messages)}")

# Run to step 7 (last deliberation, next is decision)
s = state2
for i in range(5):
    obs_i, s, _, _, _ = env.step(jax.random.PRNGKey(10+i), s, {"agent_0": jnp.int32(25), "agent_1": jnp.int32(25)})

# Step 7 obs (decision is next)
save_obs(obs_i, "obs_step6")
print(f"Step count before decision: {np.array(s.env_state.step_count)}")

# Decision step obs (should show white square)
obs_dec, s_dec, _, _, _ = env.step(jax.random.PRNGKey(99), s, {"agent_0": jnp.int32(25), "agent_1": jnp.int32(25)})
save_obs(obs_dec, "obs_decision")
print(f"Step count at decision: {np.array(s_dec.env_state.step_count)}")
print(f"\nCards: {np.array(state.env_state.card_permutation)}")
