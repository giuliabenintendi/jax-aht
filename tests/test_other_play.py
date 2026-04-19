"""Tests for Other-Play wrappers (position shuffle + recolouring)."""
import jax
import jax.numpy as jnp

from envs.card_game.card_game import CardGameEnv
from envs.card_game.rendering import (
    TILE_PIXELS,
    NUM_CARDS,
    CARD_COLORS,
    AGENT_0_COLOR,
)
from envs.card_game.other_play import (
    CardGamePositionShuffleWrapper,
    CardGameRecolouringWrapper,
)


class LeverStyleCardGameEnv(CardGameEnv):
    """Test-only reward variant mirroring the OP lever-game asymmetry.

    Coordinating on the focal ground-truth color pays `focal_reward`, while
    coordinating on any other ground-truth color pays `symmetric_reward`.
    """

    def __init__(
        self,
        focal_color: int = 0,
        focal_reward: float = 0.9,
        symmetric_reward: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.focal_color = jnp.int32(focal_color)
        self.focal_reward = float(focal_reward)
        self.symmetric_reward = float(symmetric_reward)

    def _base_reward(self, pick_0, pick_1, is_decision, target_color):
        valid_match = (pick_0 >= 0) & (pick_1 >= 0) & jnp.equal(pick_0, pick_1)
        coord_reward = jnp.where(
            jnp.equal(pick_0, self.focal_color),
            self.focal_reward,
            self.symmetric_reward,
        )
        return jnp.where(is_decision & valid_match, coord_reward, 0.0)


def _get_card_color_at(flat_obs, pos, img_h, img_w):
    """Extract the RGB color of the card at a given position from a flat obs.

    Samples the center pixel of the card tile's colored rectangle.
    """
    img = (flat_obs * 255.0).astype(jnp.uint8).reshape(img_h, img_w, 3)
    TP = TILE_PIXELS
    # Card mask spans rows 1-5, cols 1-5 within each tile; center is (3,3)
    cy = TP + 3
    cx = pos * TP + 3
    return img[cy, cx, :]


# ---------------------------------------------------------------------------
#  1. Inverse permutation roundtrip
# ---------------------------------------------------------------------------

def test_inverse_perm_roundtrip():
    for seed in range(20):
        key = jax.random.PRNGKey(seed)
        perm = jax.random.permutation(key, NUM_CARDS)
        inv = CardGameRecolouringWrapper._invert_perm(perm)
        identity = jnp.arange(NUM_CARDS)
        assert jnp.all(perm[inv] == identity)
        assert jnp.all(inv[perm] == identity)


# ---------------------------------------------------------------------------
#  2–3. Recolouring preserves reward (match / mismatch)
# ---------------------------------------------------------------------------

def test_recolouring_reward_match():
    """Both agents target the same ground-truth color → reward 1.0."""
    env = CardGameEnv(max_steps=2, shuffle=False)
    wrapped = CardGameRecolouringWrapper(env)

    key = jax.random.PRNGKey(42)
    obs, state = wrapped.reset(key)

    π0 = state.per_agent_recolouring["agent_0"]
    π1 = state.per_agent_recolouring["agent_1"]

    # Both agents want GT color 2.
    # In their recoloured view, GT color 2 appears as π_i[2].
    a0 = jnp.int32(π0[2])
    a1 = jnp.int32(π1[2])

    # Deliberation
    key, subkey = jax.random.split(key)
    noop = {"agent_0": jnp.int32(5), "agent_1": jnp.int32(5)}
    obs, state, reward, dones, info = wrapped.step(subkey, state, noop)

    # Decision
    key, subkey = jax.random.split(key)
    obs, state, reward, dones, info = wrapped.step(
        subkey, state, {"agent_0": a0, "agent_1": a1})
    assert float(reward["agent_0"]) == 1.0


def test_recolouring_reward_mismatch():
    """Agents target different ground-truth colors → reward 0.0."""
    env = CardGameEnv(max_steps=2, shuffle=False)
    wrapped = CardGameRecolouringWrapper(env)

    key = jax.random.PRNGKey(43)
    obs, state = wrapped.reset(key)

    π0 = state.per_agent_recolouring["agent_0"]
    π1 = state.per_agent_recolouring["agent_1"]

    # Agent 0 wants GT 1, Agent 1 wants GT 3
    a0 = jnp.int32(π0[1])
    a1 = jnp.int32(π1[3])

    key, subkey = jax.random.split(key)
    noop = {"agent_0": jnp.int32(5), "agent_1": jnp.int32(5)}
    obs, state, reward, dones, info = wrapped.step(subkey, state, noop)

    key, subkey = jax.random.split(key)
    obs, state, reward, dones, info = wrapped.step(
        subkey, state, {"agent_0": a0, "agent_1": a1})
    assert float(reward["agent_0"]) == 0.0


def test_recolouring_reward_all_gt_colors():
    """Sweep all 5 GT colors: matching → 1.0, each against a different → 0.0."""
    env = CardGameEnv(max_steps=2, shuffle=False)
    wrapped = CardGameRecolouringWrapper(env)

    for gt_color in range(NUM_CARDS):
        key = jax.random.PRNGKey(100 + gt_color)
        obs, state = wrapped.reset(key)

        π0 = state.per_agent_recolouring["agent_0"]
        π1 = state.per_agent_recolouring["agent_1"]

        # Both pick same GT color
        a0 = jnp.int32(π0[gt_color])
        a1 = jnp.int32(π1[gt_color])

        key, subkey = jax.random.split(key)
        noop = {"agent_0": jnp.int32(5), "agent_1": jnp.int32(5)}
        obs, state, _, _, _ = wrapped.step(subkey, state, noop)

        key, subkey = jax.random.split(key)
        _, _, reward, _, _ = wrapped.step(
            subkey, state, {"agent_0": a0, "agent_1": a1})
        assert float(reward["agent_0"]) == 1.0, f"GT color {gt_color} failed"


# ---------------------------------------------------------------------------
#  4. Observation correctness — recolouring
# ---------------------------------------------------------------------------

def test_recolouring_obs_correctness():
    """Card pixels in the observation match CARD_COLORS[recolouring[gt_color]]."""
    env = CardGameEnv(max_steps=8, shuffle=False)
    wrapped = CardGameRecolouringWrapper(env)

    key = jax.random.PRNGKey(7)
    obs, state = wrapped.reset(key)

    for agent in ["agent_0", "agent_1"]:
        recolouring = state.per_agent_recolouring[agent]
        for pos in range(NUM_CARDS):
            # Canonical order: GT color at position pos = pos
            expected = CARD_COLORS[recolouring[pos]]
            actual = _get_card_color_at(obs[agent], pos, env._img_h, env._img_w)
            assert jnp.all(actual == expected), (
                f"{agent} pos {pos}: expected {expected}, got {actual}")


# ---------------------------------------------------------------------------
#  5. Observation correctness — position shuffle
# ---------------------------------------------------------------------------

def test_position_shuffle_obs_correctness():
    """After shuffle with perm, position j shows tile from original position perm[j]."""
    env = CardGameEnv(max_steps=8, shuffle=False)
    wrapped = CardGamePositionShuffleWrapper(env)

    key = jax.random.PRNGKey(7)
    obs, state = wrapped.reset(key)

    for agent in ["agent_0", "agent_1"]:
        perm = state.per_agent_perm[agent]
        for pos in range(NUM_CARDS):
            # Canonical: color i at position i.
            # Shuffled: position pos shows tile from perm[pos] → color perm[pos].
            expected = CARD_COLORS[perm[pos]]
            actual = _get_card_color_at(obs[agent], pos, env._img_h, env._img_w)
            assert jnp.all(actual == expected), (
                f"{agent} pos {pos}: expected {expected}, got {actual}")


# ---------------------------------------------------------------------------
#  6. Agents see different observations
# ---------------------------------------------------------------------------

def test_agents_see_different_card_rows():
    """With OP, card rows differ between agents (when permutations differ)."""
    TP = TILE_PIXELS

    def _card_row(flat_obs, h, w):
        img = (flat_obs * 255).astype(jnp.uint8).reshape(h, w, 3)
        return img[TP:2 * TP, :, :]

    # Position shuffle
    env = CardGameEnv(max_steps=8, shuffle=False)
    wrapped = CardGamePositionShuffleWrapper(env)
    # Use multiple seeds to ensure we find one where perms differ
    found_diff = False
    for seed in range(10):
        key = jax.random.PRNGKey(seed)
        obs, state = wrapped.reset(key)
        p0 = state.per_agent_perm["agent_0"]
        p1 = state.per_agent_perm["agent_1"]
        if not jnp.all(p0 == p1):
            cr0 = _card_row(obs["agent_0"], env._img_h, env._img_w)
            cr1 = _card_row(obs["agent_1"], env._img_h, env._img_w)
            assert not jnp.array_equal(cr0, cr1)
            found_diff = True
            break
    assert found_diff, "Failed to find a seed where position perms differ"

    # Recolouring
    env2 = CardGameEnv(max_steps=8, shuffle=False)
    wrapped2 = CardGameRecolouringWrapper(env2)
    found_diff = False
    for seed in range(10):
        key = jax.random.PRNGKey(seed)
        obs2, state2 = wrapped2.reset(key)
        r0 = state2.per_agent_recolouring["agent_0"]
        r1 = state2.per_agent_recolouring["agent_1"]
        if not jnp.all(r0 == r1):
            cr0 = _card_row(obs2["agent_0"], env2._img_h, env2._img_w)
            cr1 = _card_row(obs2["agent_1"], env2._img_h, env2._img_w)
            assert not jnp.array_equal(cr0, cr1)
            found_diff = True
            break
    assert found_diff, "Failed to find a seed where recolouring perms differ"


# ---------------------------------------------------------------------------
#  7. Auto-reset re-samples permutations
# ---------------------------------------------------------------------------

def test_auto_reset_obs_matches_new_perms():
    """After auto-reset, observation matches the newly sampled permutations."""
    env = CardGameEnv(max_steps=2, shuffle=False)
    wrapped = CardGamePositionShuffleWrapper(env)

    key = jax.random.PRNGKey(0)
    obs, state = wrapped.reset(key)

    # Deliberation
    key, subkey = jax.random.split(key)
    noop = {"agent_0": jnp.int32(5), "agent_1": jnp.int32(5)}
    obs, state, _, _, _ = wrapped.step(subkey, state, noop)

    # Decision → done=True, auto-reset
    key, subkey = jax.random.split(key)
    actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
    obs, state, _, dones, _ = wrapped.step(subkey, state, actions)
    assert dones["__all__"]

    # The returned obs is from the NEW episode; verify it matches new perms
    for agent in ["agent_0", "agent_1"]:
        perm = state.per_agent_perm[agent]
        for pos in range(NUM_CARDS):
            expected = CARD_COLORS[perm[pos]]
            actual = _get_card_color_at(obs[agent], pos, env._img_h, env._img_w)
            assert jnp.all(actual == expected), (
                f"After reset, {agent} pos {pos}: expected {expected}, got {actual}")


# ---------------------------------------------------------------------------
#  8. Communication message remapping
# ---------------------------------------------------------------------------

def test_recolouring_message_remapping():
    """Messages are inverse-mapped through the recolouring."""
    env = CardGameEnv(max_steps=3, shuffle=False, communication=True)
    wrapped = CardGameRecolouringWrapper(env)

    key = jax.random.PRNGKey(99)
    obs, state = wrapped.reset(key)

    inv0 = state.per_agent_inv_recolouring["agent_0"]
    inv1 = state.per_agent_inv_recolouring["agent_1"]

    # Agent 0 sends message 2 (recoloured), Agent 1 sends message 4 (recoloured)
    # Deliberation with communication: action = NUM_CARDS + msg_idx
    a0 = jnp.int32(NUM_CARDS + 2)
    a1 = jnp.int32(NUM_CARDS + 4)

    key, subkey = jax.random.split(key)
    obs, state, _, _, _ = wrapped.step(
        subkey, state, {"agent_0": a0, "agent_1": a1})

    # Env should store inv[msg] as the true message
    inner_state = state.env_state.env_state  # WrappedEnvState → CardGameState
    assert int(inner_state.messages[0]) == int(inv0[2])
    assert int(inner_state.messages[1]) == int(inv1[4])


# ---------------------------------------------------------------------------
#  9. Combined wrappers
# ---------------------------------------------------------------------------

def test_combined_wrappers_reward():
    """Position shuffle + recolouring composed: reward on ground truth."""
    env = CardGameEnv(max_steps=2, shuffle=False)
    env = CardGamePositionShuffleWrapper(env)
    env = CardGameRecolouringWrapper(env)

    key = jax.random.PRNGKey(55)
    obs, state = env.reset(key)

    π0 = state.per_agent_recolouring["agent_0"]
    π1 = state.per_agent_recolouring["agent_1"]

    # Both pick GT color 4
    a0 = jnp.int32(π0[4])
    a1 = jnp.int32(π1[4])

    key, subkey = jax.random.split(key)
    noop = {"agent_0": jnp.int32(5), "agent_1": jnp.int32(5)}
    obs, state, _, _, _ = env.step(subkey, state, noop)

    key, subkey = jax.random.split(key)
    _, _, reward, _, _ = env.step(
        subkey, state, {"agent_0": a0, "agent_1": a1})
    assert float(reward["agent_0"]) == 1.0


def test_recolouring_preserves_focal_low_reward_card():
    """Recolouring must preserve the paper-style distinctive 0.9 payoff card."""
    env = LeverStyleCardGameEnv(max_steps=2, shuffle=True, focal_color=0)
    wrapped = CardGameRecolouringWrapper(env)

    for seed in range(20):
        key = jax.random.PRNGKey(400 + seed)
        obs, state = wrapped.reset(key)

        recolour_0 = state.per_agent_recolouring["agent_0"]
        recolour_1 = state.per_agent_recolouring["agent_1"]

        for gt_color, expected_reward in ((0, 0.9), (1, 1.0)):
            a0 = jnp.int32(recolour_0[gt_color])
            a1 = jnp.int32(recolour_1[gt_color])

            key, subkey = jax.random.split(key)
            noop = {"agent_0": jnp.int32(5), "agent_1": jnp.int32(5)}
            obs, state, _, _, _ = wrapped.step(subkey, state, noop)

            key, subkey = jax.random.split(key)
            _, _, reward, _, _ = wrapped.step(
                subkey, state, {"agent_0": a0, "agent_1": a1}
            )
            assert float(reward["agent_0"]) == expected_reward, (
                f"seed={seed} gt_color={gt_color} should yield {expected_reward} under recolouring"
            )

            key, reset_key = jax.random.split(key)
            obs, state = wrapped.reset(reset_key)
            recolour_0 = state.per_agent_recolouring["agent_0"]
            recolour_1 = state.per_agent_recolouring["agent_1"]


def test_combined_wrappers_preserve_focal_low_reward_card():
    """Position shuffle + recolouring must preserve the distinctive 0.9 card."""
    env = LeverStyleCardGameEnv(max_steps=2, shuffle=False, focal_color=0)
    env = CardGamePositionShuffleWrapper(env)
    env = CardGameRecolouringWrapper(env)

    for seed in range(20):
        key = jax.random.PRNGKey(500 + seed)
        obs, state = env.reset(key)

        recolour_0 = state.per_agent_recolouring["agent_0"]
        recolour_1 = state.per_agent_recolouring["agent_1"]

        for gt_color, expected_reward in ((0, 0.9), (1, 1.0)):
            a0 = jnp.int32(recolour_0[gt_color])
            a1 = jnp.int32(recolour_1[gt_color])

            key, subkey = jax.random.split(key)
            noop = {"agent_0": jnp.int32(5), "agent_1": jnp.int32(5)}
            obs, state, _, _, _ = env.step(subkey, state, noop)

            key, subkey = jax.random.split(key)
            _, _, reward, _, _ = env.step(
                subkey, state, {"agent_0": a0, "agent_1": a1}
            )
            assert float(reward["agent_0"]) == expected_reward, (
                f"seed={seed} gt_color={gt_color} should yield {expected_reward} with combined OP wrappers"
            )

            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key)
            recolour_0 = state.per_agent_recolouring["agent_0"]
            recolour_1 = state.per_agent_recolouring["agent_1"]


def test_save_op_observations():
    """Save PNGs showing ground truth vs each agent's OP-transformed view."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from pathlib import Path
    from envs.card_game.rendering import GRID_ROWS, GRID_COLS, render_card_game

    out_dir = Path("tests/card_game_op_obs")
    out_dir.mkdir(exist_ok=True)

    h = GRID_ROWS * TILE_PIXELS
    w = GRID_COLS * TILE_PIXELS
    scale = 20
    card_names = ["RED", "BLUE", "GREEN", "YELLOW", "PURPLE"]

    base = CardGameEnv(max_steps=8, shuffle=False)
    env = CardGamePositionShuffleWrapper(base)
    env = CardGameRecolouringWrapper(env)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)

    # Ground truth: canonical order (shuffle=False)
    gt_img = np.array(render_card_game(jnp.arange(NUM_CARDS)))
    gt_pil = Image.fromarray(gt_img).resize((w * scale, h * scale), Image.NEAREST)
    gt_pil.save(out_dir / "ground_truth.png")

    pos_perm_0 = np.array(state.env_state.per_agent_perm["agent_0"])
    pos_perm_1 = np.array(state.env_state.per_agent_perm["agent_1"])
    recolour_0 = np.array(state.per_agent_recolouring["agent_0"])
    recolour_1 = np.array(state.per_agent_recolouring["agent_1"])
    inv_0 = np.array(state.per_agent_inv_recolouring["agent_0"])
    inv_1 = np.array(state.per_agent_inv_recolouring["agent_1"])

    print(f"\nGround truth: {card_names}")
    print(f"Agent 0 pos perm:  {pos_perm_0} → [{', '.join(card_names[c] for c in pos_perm_0)}]")
    print(f"Agent 0 recolour:  {recolour_0} (GT color c appears as {recolour_0})")
    print(f"Agent 0 inv:       {inv_0} (visual color v → GT {inv_0})")
    print(f"Agent 1 pos perm:  {pos_perm_1} → [{', '.join(card_names[c] for c in pos_perm_1)}]")
    print(f"Agent 1 recolour:  {recolour_1} (GT color c appears as {recolour_1})")
    print(f"Agent 1 inv:       {inv_1} (visual color v → GT {inv_1})")

    # What each agent sees (combined effect)
    print("\nAgent 0 sees at each position:")
    for pos in range(NUM_CARDS):
        gt_color = pos_perm_0[pos]
        visual_color = recolour_0[gt_color]
        print(f"  pos {pos}: GT {card_names[gt_color]} appears as {card_names[visual_color]}")

    print("Agent 1 sees at each position:")
    for pos in range(NUM_CARDS):
        gt_color = pos_perm_1[pos]
        visual_color = recolour_1[gt_color]
        print(f"  pos {pos}: GT {card_names[gt_color]} appears as {card_names[visual_color]}")

    for agent in ["agent_0", "agent_1"]:
        img = (np.array(obs[agent]) * 255).astype(np.uint8).reshape(h, w, 3)
        pil_img = Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST)
        pil_img.save(out_dir / f"{agent}_op_view.png")
        print(f"Saved {out_dir / agent}_op_view.png")

    # Simulate a decision: both agents output action 0 in their recoloured space
    # Show what GT color that maps to
    for a_idx, (agent, inv) in enumerate(
            [("agent_0", inv_0), ("agent_1", inv_1)]):
        for visual_action in range(NUM_CARDS):
            gt_action = inv[visual_action]
            print(f"  {agent} picks visual {card_names[visual_action]} "
                  f"→ GT {card_names[gt_action]}")


def test_combined_wrappers_obs_correctness():
    """With both wrappers, obs reflects position shuffle AND recolouring."""
    base = CardGameEnv(max_steps=8, shuffle=False)
    env = CardGamePositionShuffleWrapper(base)
    env = CardGameRecolouringWrapper(env)

    key = jax.random.PRNGKey(77)
    obs, state = env.reset(key)

    for agent in ["agent_0", "agent_1"]:
        recolouring = state.per_agent_recolouring[agent]
        pos_perm = state.env_state.per_agent_perm[agent]  # inner wrapper state

        for pos in range(NUM_CARDS):
            # Position shuffle: position pos shows tile from pos_perm[pos]
            # That tile's GT color = pos_perm[pos] (canonical order)
            # Recolouring: GT color c appears as CARD_COLORS[recolouring[c]]
            gt_color = pos_perm[pos]
            expected = CARD_COLORS[recolouring[gt_color]]
            actual = _get_card_color_at(obs[agent], pos, base._img_h, base._img_w)
            assert jnp.all(actual == expected), (
                f"{agent} pos {pos}: gt_color={gt_color}, "
                f"expected {expected}, got {actual}")


# ---------------------------------------------------------------------------
#  10. End-to-end OP + comm coordination
# ---------------------------------------------------------------------------

def _find_dot_card_pos(flat_obs, img_h, img_w, dot_color):
    """Locate the card column containing the partner's message dot.

    Returns (found, card_pos). The dot is drawn in a partner agent color that
    does not match any CARD_COLORS, so the recolouring wrapper leaves it alone.
    """
    TP = TILE_PIXELS
    img = (flat_obs * 255).astype(jnp.uint8).reshape(img_h, img_w, 3)
    card_row = img[TP:2 * TP, :, :]  # (TP, NUM_CARDS*TP, 3)
    match = jnp.all(card_row == dot_color[None, None, :], axis=-1)
    any_col = jnp.any(match, axis=0)  # (NUM_CARDS*TP,)
    found = bool(jnp.any(any_col))
    first_col = int(jnp.argmax(any_col))
    return found, first_col // TP


def _read_card_visual_color(flat_obs, card_pos, img_h, img_w):
    """Read the card's visual color at a pixel below the dot region.

    The 4x4 dot starts at tile-local (y=1, x=1) and spans (y=1..4, x=1..4).
    Sampling at tile-local (y=5, x=1) is the card's bottom-left corner, which
    is inside the 5x5 card mask but outside the dot footprint.
    """
    TP = TILE_PIXELS
    img = (flat_obs * 255).astype(jnp.uint8).reshape(img_h, img_w, 3)
    rgb = img[TP + 5, card_pos * TP + 1, :]
    # Nearest CARD_COLORS by L1 distance, defensive against uint8 round-trip drift
    diff = jnp.sum(jnp.abs(CARD_COLORS.astype(jnp.int32) - rgb.astype(jnp.int32)), axis=-1)
    return int(jnp.argmin(diff))


def test_op_comm_end_to_end_coordination():
    """Full OP+comm loop: agent 0 messages a GT color, agent 1 decodes the
    dot from its own recoloured+shuffled view and picks the card under it.

    Exercises message inversion, dot placement through position shuffle, dot
    survival through the recolouring pass, pick inversion, and reward on GT
    match. Failure of any link in the chain shows up as either dot-not-found
    or reward != 1.0.
    """
    for gt_color in range(NUM_CARDS):
        for seed in range(5):
            base = CardGameEnv(max_steps=2, shuffle=False, communication=True)
            env = CardGamePositionShuffleWrapper(base)
            env = CardGameRecolouringWrapper(env)

            key = jax.random.PRNGKey(2000 + gt_color * 100 + seed)
            obs, state = env.reset(key)

            recolour_0 = state.per_agent_recolouring["agent_0"]
            recolour_1 = state.per_agent_recolouring["agent_1"]

            # Step 1 (deliberation): agent 0 messages GT gt_color; agent 1
            # sends an arbitrary message (needed: masks disallow idle).
            msg_a0_visual = int(recolour_0[gt_color])
            msg_a1_visual = int(recolour_1[0])
            action_step1 = {
                "agent_0": jnp.int32(NUM_CARDS + msg_a0_visual),
                "agent_1": jnp.int32(NUM_CARDS + msg_a1_visual),
            }
            key, subkey = jax.random.split(key)
            obs, state, _, dones, _ = env.step(subkey, state, action_step1)
            assert not dones["__all__"]

            # Agent 1 locates agent 0's orange dot in its own view and reads
            # the visual color of the card beneath it.
            found, dot_pos = _find_dot_card_pos(
                obs["agent_1"], base._img_h, base._img_w, AGENT_0_COLOR
            )
            assert found, (
                f"gt={gt_color} seed={seed}: no orange dot in agent 1's obs"
            )
            pick_a1_visual = _read_card_visual_color(
                obs["agent_1"], dot_pos, base._img_h, base._img_w
            )

            # Step 2 (decision): agent 0 picks GT gt_color; agent 1 picks the
            # dotted card.
            action_step2 = {
                "agent_0": jnp.int32(msg_a0_visual),
                "agent_1": jnp.int32(pick_a1_visual),
            }
            key, subkey = jax.random.split(key)
            _, _, reward, dones, _ = env.step(subkey, state, action_step2)

            assert float(reward["agent_0"]) == 1.0, (
                f"gt={gt_color} seed={seed}: dot_pos={dot_pos}, "
                f"pick_a1_visual={pick_a1_visual}, reward={float(reward['agent_0'])}"
            )
