"""Smoke test for the copied Overcooked v2 env.

Verifies the copied package imports against the pinned jaxmarl 0.0.7 (i.e. that
0.0.7's MultiAgentEnv/spaces are compatible) and that demo_cook_simple resets and
steps with the partial-observation view radius the paper uses (r=2).
"""
import jax
import jax.numpy as jnp
import numpy as np

from envs import make_env
from envs.overcooked_v2.overcooked import OvercookedV2
from envs.overcooked_v2.common import Actions, Position, StaticObject
from envs.overcooked_v2.observation_rendering import render_obs_state
from envs.overcooked_v2.overcooked_v2_image_wrapper import OvercookedV2ImageWrapper
from envs.overcooked_v2.rendering import INGREDIENT_COLORS, TILE_PIXELS, render_state


def test_overcooked_v2_demo_cook_simple_smoke():
    env = OvercookedV2(layout="demo_cook_simple", agent_view_size=2)
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)
    assert set(obs.keys()) == set(env.agents)

    obs_shape = env.observation_space().shape
    # agent_view_size=2 -> a (2*2+1)=5 cell square egocentric window
    assert obs_shape[0] == 5 and obs_shape[1] == 5
    for a in env.agents:
        assert obs[a].shape == obs_shape

    actions = {a: jnp.array(Actions.stay) for a in env.agents}
    next_obs, next_state, reward, done, info = env.step(key, state, actions)
    assert set(reward.keys()) == set(env.agents)
    assert "__all__" in done
    for a in env.agents:
        assert next_obs[a].shape == obs_shape


def test_overcooked_v2_render_state_shape():
    env = OvercookedV2(layout="demo_cook_simple", agent_view_size=2)
    _, state = env.reset(jax.random.PRNGKey(0))
    img = render_state(state, TILE_PIXELS)
    assert img.shape == (env.height * TILE_PIXELS, env.width * TILE_PIXELS, 3)
    assert img.dtype == jnp.uint8


def test_overcooked_v2_obs_render_state_shape():
    env = OvercookedV2(layout="demo_cook_simple", agent_view_size=2)
    _, state = env.reset(jax.random.PRNGKey(0))
    img = render_obs_state(state, TILE_PIXELS)
    assert img.shape == (env.height * TILE_PIXELS, env.width * TILE_PIXELS, 3)
    assert img.dtype == jnp.uint8


def test_overcooked_v2_obs_renderer_is_separate_from_eval_renderer():
    env = OvercookedV2(layout="demo_cook_simple", agent_view_size=2)
    _, state = env.reset(jax.random.PRNGKey(0))
    obs_img = render_obs_state(state, TILE_PIXELS)
    eval_img = render_state(state, TILE_PIXELS)
    assert obs_img.shape == eval_img.shape
    assert not bool(jnp.array_equal(obs_img, eval_img))


def test_overcooked_v2_image_wrapper():
    env = make_env(
        "overcooked-v2",
        {
            "layout": "demo_cook_simple",
            "obs_type": "image",
            "agent_view_size": 2,
            "random_agent_positions": False,
        },
    )
    flat = env.grid_height * env.grid_width * env.tile_size * env.tile_size * 3
    assert env.observation_space("agent_0").shape == (flat,)

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    o0, o1 = obs["agent_0"], obs["agent_1"]
    assert o0.shape == (flat,) and o0.dtype == jnp.float32
    assert float(o0.max()) <= 1.0 and float(o0.min()) >= 0.0
    # Partial observability: r=2 in an 11-wide grid masks some cells to zero.
    assert float((o0 == 0.0).mean()) > 0.0
    # Distinct per-agent views (different mask + ego marker).
    assert not bool(jnp.allclose(o0, o1))

    actions = {a: jnp.array(Actions.stay) for a in env.agents}
    step = jax.jit(env.step)
    next_obs, next_state, reward, done, info = step(key, state, actions)
    assert set(reward.keys()) == set(env.agents)
    assert "base_return" in info and "shaped_reward" in info
    assert next_obs["agent_0"].shape == (flat,)


def test_overcooked_v2_make_env_op_is_image_wrapper():
    env = make_env(
        "overcooked-v2",
        {
            "layout": "demo_cook_simple",
            "obs_type": "image",
            "agent_view_size": 2,
            "op_ingredient_permutations": [0, 1],
        },
    )
    assert isinstance(env, OvercookedV2ImageWrapper)
    assert env.env.op_ingredient_permutations == [0, 1]

    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    actions = {a: jnp.array(Actions.stay) for a in env.agents}
    next_obs, *_ = jax.jit(env.step)(key, state, actions)
    assert next_obs["agent_0"].shape == obs["agent_0"].shape


def test_overcooked_v2_op_identity_palette_is_noop():
    env = OvercookedV2(layout="demo_cook_simple", agent_view_size=2)
    _, state = env.reset(jax.random.PRNGKey(0))
    img_default = render_state(state, TILE_PIXELS)
    img_identity = render_state(state, TILE_PIXELS, ingredient_colors=INGREDIENT_COLORS)
    assert bool(jnp.array_equal(img_default, img_identity))


def test_overcooked_v2_op_swap_palette_recolours_indicator_and_piles():
    """Regression test for the anti-aliasing OP bug: the palette swap must reach
    the recipe indicator and ingredient piles (drawn before downsampling), and
    must not touch ingredient-free tiles."""
    env = OvercookedV2(layout="demo_cook_simple", agent_view_size=2)
    _, state = env.reset(jax.random.PRNGKey(0))

    swapped = (
        INGREDIENT_COLORS.at[0].set(INGREDIENT_COLORS[1]).at[1].set(INGREDIENT_COLORS[0])
    )
    img_a = np.asarray(render_state(state, TILE_PIXELS))
    img_b = np.asarray(render_state(state, TILE_PIXELS, ingredient_colors=swapped))

    ts = TILE_PIXELS
    static = np.asarray(state.grid[:, :, 0])
    h, w = static.shape
    cell_diff = (
        (img_a != img_b).any(axis=-1).reshape(h, ts, w, ts).any(axis=(1, 3))
    )

    indicator = static == StaticObject.RECIPE_INDICATOR
    piles = static >= StaticObject.INGREDIENT_PILE_BASE
    # Only ing-0/1 piles are affected by the swap; distractor piles (ing 2+)
    # must render identically.
    swapped_piles = piles & (static - StaticObject.INGREDIENT_PILE_BASE <= 1)
    assert indicator.any() and swapped_piles.any()
    # demo_cook recipes are all-ing0 or all-ing1, so the swap must change the
    # indicator tile and every ing-0/1 pile tile.
    assert cell_diff[indicator].all()
    assert cell_diff[swapped_piles].all()
    # At reset no counters/pots/agents hold ingredients: nothing else changes.
    assert not cell_diff[~(indicator | swapped_piles)].any()


def test_overcooked_v2_op_agents_see_permuted_views():
    """End-to-end: an agent with a swapped permutation must see the true-ing0
    pile in ing1's colour, while an identity agent sees it unpermuted."""
    env = make_env(
        "overcooked-v2",
        {
            "layout": "demo_cook_simple",
            "obs_type": "image",
            "agent_view_size": None,  # full view so both agents see the piles
            "random_agent_positions": False,
            "op_ingredient_permutations": [0, 1],
        },
    )

    identity = jnp.array([0, 1])
    obs = state = perms = None
    for seed in range(32):
        obs, state = env.reset(jax.random.PRNGKey(seed))
        perms = state.env_state.ingredient_permutations[:, :2]
        if bool((perms[0] == identity).all()) != bool((perms[1] == identity).all()):
            break
    else:
        raise AssertionError("no reset key produced differing agent permutations")

    id_agent = 0 if bool((perms[0] == identity).all()) else 1
    sw_agent = 1 - id_agent

    static = np.asarray(state.env_state.grid[:, :, 0])
    ys, xs = np.where(static == StaticObject.INGREDIENT_PILE_BASE)  # true ing 0
    y, x = int(ys[0]), int(xs[0])

    ts = env.tile_size
    h, w = static.shape

    def pile_tile(agent_idx):
        img = np.asarray(obs[env.agents[agent_idx]].reshape(h * ts, w * ts, 3)) * 255.0
        return img[y * ts:(y + 1) * ts, x * ts:(x + 1) * ts].reshape(-1, 3)

    c0 = np.asarray(INGREDIENT_COLORS[0], dtype=np.float64)
    c1 = np.asarray(INGREDIENT_COLORS[1], dtype=np.float64)

    id_tile, sw_tile = pile_tile(id_agent), pile_tile(sw_agent)
    # Blob interiors survive downsampling as pure palette colours.
    assert (np.abs(id_tile - c0).max(axis=-1) < 1.0).any()
    assert not (np.abs(id_tile - c1).max(axis=-1) < 1.0).any()
    assert (np.abs(sw_tile - c1).max(axis=-1) < 1.0).any()
    assert not (np.abs(sw_tile - c0).max(axis=-1) < 1.0).any()


def test_ocv2_op_vertical_flip_mirrors_swapped_agents():
    # flip-swap OP: agents whose ingredients are swapped ALSO see a vertically
    # mirrored world and have up/down actions swapped; identity agents untouched.
    # The mirror is input-side (grid rows + agent poses fed to the renderer), so
    # tile POSITIONS swap rows while within-tile artwork stays upright -- a
    # pixel-level flip of the frame would mirror pot rims/timers and recipe dots,
    # betraying the flipped lens.
    kw = dict(layout="demo_cook_simple", obs_type="image", agent_view_size=2,
              random_agent_positions=False, op_ingredient_permutations=[0, 1])
    flip_env = make_env("overcooked-v2", {**kw, "op_vertical_flip": True})
    ref_env = make_env("overcooked-v2", {**kw, "op_vertical_flip": False})

    _, state = flip_env.reset(jax.random.PRNGKey(0))
    # Agent 0 swapped (perm [1,0,2]) -> flipped; agent 1 identity -> untouched.
    perm = jnp.array([[1, 0, 2], [0, 1, 2]])
    state = state.replace(env_state=state.env_state.replace(ingredient_permutations=perm))

    ts = flip_env.tile_size
    h = flip_env.grid_height * ts
    w = flip_env.grid_width * ts
    obs_flip = flip_env._make_obs(state.env_state)
    obs_ref = ref_env._make_obs(state.env_state)
    a0, a1 = flip_env.agents
    f0 = obs_flip[a0].reshape(h, w, 3)
    r0 = obs_ref[a0].reshape(h, w, 3)

    def tile(img, row, col):
        return img[row * ts : (row + 1) * ts, col * ts : (col + 1) * ts]

    # Positions mirror: the visible ingredient piles at (0,8)/(4,8) hold
    # different ingredients, so their pixel-exact swap checks a real row swap.
    assert bool(jnp.array_equal(tile(f0, 0, 8), tile(r0, 4, 8)))
    assert bool(jnp.array_equal(tile(f0, 4, 8), tile(r0, 0, 8)))
    assert not bool(jnp.array_equal(tile(r0, 0, 8), tile(r0, 4, 8)))
    # Artwork stays upright: the pot sits ON the mirror axis (2,7), so its tile
    # must be untouched, not a pixel mirror of itself.
    assert bool(jnp.array_equal(tile(f0, 2, 7), tile(r0, 2, 7)))
    assert not bool(jnp.array_equal(tile(f0, 2, 7), jnp.flip(tile(r0, 2, 7), axis=0)))
    # Identity agent 1 is unchanged.
    assert bool(jnp.allclose(obs_flip[a1], obs_ref[a1]))

    # The lens state mirrors poses and facing: UP<->DOWN, RIGHT/LEFT unchanged.
    m = flip_env._mirrored_state(state.env_state)
    dir_flip = jnp.array([1, 0, 2, 3])  # Direction UP, DOWN, RIGHT, LEFT
    assert bool(jnp.array_equal(m.agents.dir, dir_flip[state.env_state.agents.dir]))
    assert bool(jnp.array_equal(
        m.agents.pos.y, flip_env.grid_height - 1 - state.env_state.agents.pos.y
    ))

    # Action remap: flipped agent 0 issuing 'up' reaches the env as 'down'.
    up = {a0: jnp.array(int(Actions.up)), a1: jnp.array(int(Actions.stay))}
    down = {a0: jnp.array(int(Actions.down)), a1: jnp.array(int(Actions.stay))}
    _, flip_after, *_ = flip_env.step(jax.random.PRNGKey(1), state, up)
    _, ref_after, *_ = ref_env.step(jax.random.PRNGKey(1), state, down)
    assert int(flip_after.env_state.agents.dir[0]) == int(ref_after.env_state.agents.dir[0])
    # Sanity: without the remap, 'up' and 'down' would differ.
    _, ref_up, *_ = ref_env.step(jax.random.PRNGKey(1), state, up)
    assert int(flip_after.env_state.agents.dir[0]) != int(ref_up.env_state.agents.dir[0])


def test_ocv2_op_vertical_flip_off_is_noop():
    kw = dict(layout="demo_cook_simple", obs_type="image", agent_view_size=2,
              random_agent_positions=False, op_ingredient_permutations=[0, 1])
    flip_off = make_env("overcooked-v2", {**kw, "op_vertical_flip": False})
    _, state = flip_off.reset(jax.random.PRNGKey(0))
    perm = jnp.array([[1, 0, 2], [0, 1, 2]])
    state = state.replace(env_state=state.env_state.replace(ingredient_permutations=perm))
    # With the flag off, flip bits are all False regardless of the swap.
    assert not bool(flip_off._flip_bits(state.env_state).any())


def _make_mate_mechanism(gating):
    from agents.overcooked_v2.ja_overcooked_v2_dense_occupancy import (
        OvercookedV2DenseObjectOccupancyMechanism,
    )
    env = make_env(
        "overcooked-v2",
        {
            "layout": "demo_cook_simple",
            "obs_type": "image",
            "agent_view_size": 2,
            "random_agent_positions": False,
        },
    )
    config = {
        "ROLLOUT_LENGTH": 4,
        "NUM_ENVS": 1,
        "FEED_OTHER_ATTN": True,
        "JA_VISIBILITY_GATING": gating,
    }
    return OvercookedV2DenseObjectOccupancyMechanism(config, env)


def _witness_traj_extras(mech, T, A):
    """Partner of actor 0 (= actor 1) attends far, near, far cells."""
    from types import SimpleNamespace

    M = 2
    mask_all = jnp.zeros((T, A, M, mech.feat_h, mech.feat_w), dtype=jnp.float32)
    # Object 0 is actor 0's self-selected dummy object.
    mask_all = mask_all.at[:, :, 0, 1, 1].set(1.0)
    # Object 1 is the partner-selected object. For receiver 0, put it outside
    # the receiver's view at t=0/t=2 and inside at t=1.
    mask_all = mask_all.at[0, :, 1, 5, 12].set(1.0)
    mask_all = mask_all.at[1, :, 1, 5, 4].set(1.0)
    mask_all = mask_all.at[2, :, 1, 5, 12].set(1.0)
    sel = jnp.array([[0, 1], [0, 1], [0, 1]], dtype=jnp.int32)
    extras = {
        "ja_future_object_mask_all": mask_all,
        "ja_future_object_visible_all": jnp.ones((T, A, M), dtype=jnp.float32),
        "ja_future_object_idx": sel,
        "ja_partner_visible": jnp.ones((T, A), dtype=jnp.float32),
    }
    return SimpleNamespace(done=jnp.zeros((T, A), dtype=bool), extras=extras)


def test_ocv2_mate_partner_aux_supervises_hidden_footprint():
    # Seeing the partner IS the witness: the target is the partner-attended
    # object's full footprint even when that object lies outside the receiver's
    # own view box (the feed channel lights up those cells). No receiver-view
    # clipping of aux targets.
    mech = _make_mate_mechanism(gating=True)
    assert (mech.feat_h, mech.feat_w) == (10, 22)
    T, A = 3, 2
    traj = _witness_traj_extras(mech, T, A)

    _, partner_occ = mech._occupancy_targets(traj)
    tgt = partner_occ[:, 0].reshape(T, mech.feat_h * mech.feat_w)
    in_idx = 5 * mech.feat_w + 4    # partner target inside receiver view at t=1
    out_idx = 5 * mech.feat_w + 12  # outside receiver RGB view, lit by the feed

    assert float(tgt[0, out_idx]) > 0.99 and float(tgt[0, in_idx]) == 0.0
    assert float(tgt[1, in_idx]) > 0.99
    assert float(tgt[2, out_idx]) > 0.99

    # Partner-out-of-view still gates everything off.
    traj.extras["ja_partner_visible"] = jnp.zeros((T, A), dtype=jnp.float32)
    _, occ_gated = mech._occupancy_targets(traj)
    assert float(occ_gated[:, 0].sum()) < 1e-3

    # Attender grounding: if the ATTENDER cannot see its own attended object,
    # the event dies even though the partner is visible to the receiver.
    traj2 = _witness_traj_extras(mech, T, A)
    traj2.extras["ja_future_object_visible_all"] = (
        traj2.extras["ja_future_object_visible_all"].at[:, 1, 1].set(0.0)
    )
    _, occ_blind = mech._occupancy_targets(traj2)
    assert float(occ_blind[:, 0].sum()) < 1e-3


def test_ocv2_mate_aux_visible_only_masks_hidden_partner_steps():
    # With JA_AUX_PARTNER_VISIBLE_ONLY, steps where the partner is hidden carry
    # no aux gradient: supervision exists only while the feed lights the target.
    from types import SimpleNamespace
    from agents.overcooked_v2.ja_overcooked_v2_dense_occupancy import (
        OvercookedV2DenseObjectOccupancyMechanism,
    )

    env = make_env(
        "overcooked-v2",
        {"layout": "demo_cook_simple", "obs_type": "image", "agent_view_size": 2,
         "random_agent_positions": False},
    )
    def mk(flag):
        return OvercookedV2DenseObjectOccupancyMechanism(
            {"ROLLOUT_LENGTH": 4, "NUM_ENVS": 1, "FEED_OTHER_ATTN": True,
             "JA_VISIBILITY_GATING": True, "JA_OBJECT_AUX_COEF": 1e-4,
             "JA_AUX_PARTNER_VISIBLE_ONLY": flag}, env)

    mech_gated, mech_plain = mk(True), mk(False)
    fh, fw = mech_gated.feat_h, mech_gated.feat_w
    T, A = 2, 1
    target = jnp.zeros((T, A, fh, fw)).at[:, :, 5, 12].set(1.0)
    # Step 0: attention on the target (low NLL). Step 1: attention elsewhere
    # (high NLL), partner hidden.
    attn = jnp.full((T, A, fh, fw), 1e-8).at[0, 0, 5, 12].set(1.0).at[1, 0, 0, 0].set(1.0)
    traj = SimpleNamespace(
        done=jnp.zeros((T, A), dtype=bool),
        extras={
            "ja_future_partner_occ": target,
            "ja_partner_visible": jnp.array([[1.0], [0.0]]),
        },
    )
    _, loss_gated = mech_gated.aux_loss(attn, traj, None)
    _, loss_plain = mech_plain.aux_loss(attn, traj, None)
    # Gated: only the visible step counts -> near-zero NLL. Plain: the hidden
    # high-NLL step is averaged in.
    assert float(loss_gated) < 1e-4
    assert float(loss_plain) > 5.0


def test_ocv2_mate_allocentric_event_visibility_is_attender_view():
    # Allocentric object frame: footprint masks exist for every valid object
    # (supervisable beyond the receiver's view), while the `visible` flags used
    # for gaze-event validity require the ACTOR to see the object. At the fixed
    # demo_cook_simple starts (agents (2,6)/(2,8), radius 2) the goal (2,10) is
    # outside agent_0's view but inside agent_1's; the pot (2,7) is in both.
    from agents.overcooked_v2.ja_overcooked_v2_attention import TaskObject

    mech = _make_mate_mechanism(gating=True)
    env = make_env(
        "overcooked-v2",
        {
            "layout": "demo_cook_simple",
            "obs_type": "image",
            "agent_view_size": 2,
            "random_agent_positions": False,
        },
    )
    _, state = env.reset(jax.random.PRNGKey(0))
    masks, vis = mech._object_frame(_batch1(state), 2)

    goal_i = int(jnp.argmax(mech.static_object_cat == TaskObject.GOAL))
    pot_i = int(jnp.argmax(mech.static_object_cat == TaskObject.POT))
    assert float(vis[0, goal_i]) == 0.0 and float(vis[1, goal_i]) == 1.0
    assert float(vis[0, pot_i]) == 1.0 and float(vis[1, pot_i]) == 1.0
    # Masks are world-frame footprints independent of who can see the object.
    assert float(masks[0, goal_i].sum()) > 0.0
    assert bool(jnp.allclose(masks[0, goal_i], masks[1, goal_i], atol=1e-6))


def test_ocv2_eval_feed_mask_matches_training_whole_map():
    # Eval-parity guard: make_visibility_mask_fn must reproduce the training
    # feed_mask (ones * partner_visible), NOT a view-box-clipped mask. At the
    # fixed demo_cook_simple starts agents (6,2)/(8,2) are within radius 2, so the
    # mask must be all-ones EVERYWHERE (including cells outside each receiver's own
    # view box); a view-box mask would leave far cells zero.
    from agents.overcooked_v2.ja_overcooked_v2_attention import (
        make_visibility_mask_fn,
        overcooked_v2_object_ctx,
    )
    env = make_env(
        "overcooked-v2",
        {"layout": "demo_cook_simple", "obs_type": "image", "agent_view_size": 2,
         "random_agent_positions": False},
    )
    ctx = overcooked_v2_object_ctx({"ROLLOUT_LENGTH": 4, "NUM_ENVS": 1}, env)
    assert ctx["egocentric"] is False
    _, state = env.reset(jax.random.PRNGKey(0))
    m0, m1 = make_visibility_mask_fn(ctx)(state)
    # Partner visible at reset -> whole map available, no view-box clipping.
    assert float(m0.min()) == 1.0 and float(m1.min()) == 1.0


def test_ocv2_mate_gating_off_matches_swap():
    mech = _make_mate_mechanism(gating=False)
    T, A = 3, 2
    traj = _witness_traj_extras(mech, T, A)
    self_occ, partner_occ = mech._occupancy_targets(traj)
    half = A // 2
    swapped = jnp.concatenate([self_occ[:, half:], self_occ[:, :half]], axis=1)
    assert bool(jnp.allclose(partner_occ, swapped, atol=1e-6))


def test_ocv2_mate_mask_event_shadows_future_object():
    mech = _make_mate_mechanism(gating=False)
    T, A = 2, 1
    current = jnp.zeros((T, A, mech.feat_h, mech.feat_w), dtype=jnp.float32)
    current = current.at[0, 0, 1, 1].set(1.0)
    current = current.at[0, 0, 1, 2].set(1.0)
    current = current.at[1, 0, 7, 7].set(1.0)
    target = mech._attended_mask_occupancy(
        current, jnp.zeros((T, A), dtype=bool),
    ).reshape(T, A, mech.feat_h, mech.feat_w)

    # Replace semantics (default): a valid event at t=0 produces exactly the
    # t=0 object mask, with no discounted t=1 mass at unrelated cells.
    assert float(target[0, 0, 7, 7]) == 0.0
    assert float(target[0, 0, 1, 1]) > 0.49
    assert float(target[0, 0, 1, 2]) > 0.49


def test_ocv2_mate_occ_future_blend_mixes_later_foci():
    # JA_OCC_FUTURE_BLEND restores the LBF per-cell rule: the target is the
    # LIST of upcoming foci with gamma-per-step relative weights. Events at
    # t=0 (two cells) and t=2, nothing at t=1: the t=0 target keeps the t=2
    # focus at gamma^2 = 0.64 against 1 per current-focus cell.
    from agents.overcooked_v2.ja_overcooked_v2_dense_occupancy import (
        OvercookedV2DenseObjectOccupancyMechanism,
    )

    env = make_env(
        "overcooked-v2",
        {"layout": "demo_cook_simple", "obs_type": "image", "agent_view_size": 2,
         "random_agent_positions": False},
    )
    mech = OvercookedV2DenseObjectOccupancyMechanism(
        {"ROLLOUT_LENGTH": 4, "NUM_ENVS": 1, "FEED_OTHER_ATTN": True,
         "JA_VISIBILITY_GATING": False, "JA_OCC_FUTURE_BLEND": True,
         "JA_FUTURE_GAMMA_OCC": 0.8}, env)
    T, A = 3, 1
    current = jnp.zeros((T, A, mech.feat_h, mech.feat_w), dtype=jnp.float32)
    current = current.at[0, 0, 1, 1].set(1.0)
    current = current.at[0, 0, 1, 2].set(1.0)
    current = current.at[2, 0, 7, 7].set(1.0)
    target = mech._attended_mask_occupancy(
        current, jnp.zeros((T, A), dtype=bool),
    ).reshape(T, A, mech.feat_h, mech.feat_w)

    s = 1.0 + 1.0 + 0.64
    assert abs(float(target[0, 0, 1, 1]) - 1.0 / s) < 1e-5
    assert abs(float(target[0, 0, 1, 2]) - 1.0 / s) < 1e-5
    assert abs(float(target[0, 0, 7, 7]) - 0.64 / s) < 1e-5
    assert abs(float(target.reshape(T, -1)[0].sum()) - 1.0) < 1e-5
    # A lone upcoming focus renormalizes to full strength (as in LBF): the
    # discount weights objects against each other, not overall confidence.
    assert abs(float(target[1, 0, 7, 7]) - 1.0) < 1e-5


def test_ocv2_mate_feed_mask_keeps_unmasked_peak_scale():
    mech = _make_mate_mechanism(gating=True)
    A, fh, fw = 2, mech.feat_h, mech.feat_w
    attn = jnp.zeros((A, fh, fw)).at[:, 5, 12].set(0.9).at[:, 5, 4].set(0.1)
    # Receiver 0's view covers feature cols 0-9 only; receiver 1 sees everything.
    mask = jnp.stack([
        jnp.zeros((fh, fw)).at[:, :10].set(1.0),
        jnp.ones((fh, fw)),
    ])
    obs = jnp.zeros((A, mech.img_h * mech.img_w * 3))
    aug = mech.augment_obs(obs, {"partner_attn": attn, "feed_mask": mask})
    ch = aug.reshape(A, mech.img_h, mech.img_w, 4)[..., 3]
    # Masked receiver: peak removed, residual normalized by the UNMASKED peak
    # (0.1/0.9), not re-inflated to 1. Unmasked receiver keeps its true peak.
    assert 0.05 < float(ch[0].max()) < 0.2
    assert float(ch[1].max()) > 0.99


def test_ocv2_full_map_feed_mask_is_partner_visibility_gate():
    env = make_env(
        "overcooked-v2",
        {
            "layout": "demo_cook_simple",
            "obs_type": "image",
            "agent_view_size": 2,
            "random_agent_positions": False,
        },
    )
    from agents.overcooked_v2.ja_overcooked_v2_dense_occupancy import (
        OvercookedV2DenseObjectOccupancyMechanism,
    )
    mech = OvercookedV2DenseObjectOccupancyMechanism(
        {
            "ROLLOUT_LENGTH": 4,
            "NUM_ENVS": 1,
            "FEED_OTHER_ATTN": True,
            "JA_VISIBILITY_GATING": True,
        },
        env,
    )
    assert not mech.egocentric

    _, state = env.reset(jax.random.PRNGKey(0))
    attn = jnp.zeros((1, 2, mech.feat_h, mech.feat_w))
    attn = attn.at[0, 0, 1, mech.feat_w - 1].set(1.0)
    attn = attn.at[0, 1, 2, 0].set(1.0)
    reward = jnp.zeros((2,), dtype=jnp.float32)
    done = jnp.zeros((2,), dtype=bool)
    carry = mech.init_carry(2)

    _, next_carry, _ = mech.step(
        attn_map=attn,
        env_state=_batch1(state),
        new_env_state=_batch1(state),
        action=None,
        env_reward=reward,
        info={},
        done_actors=done,
        carry=carry,
        num_actors=2,
        update_steps=jnp.array(0),
    )
    assert bool(jnp.all(next_carry["feed_mask"] == 1.0))
    # Agent 1 receives agent 0's whole allocentric map, including cells outside
    # agent 1's own view window.
    assert float(next_carry["partner_attn"][1, 1, mech.feat_w - 1]) == 1.0

    raw = state.env_state
    far_agents = raw.agents.replace(
        pos=Position(x=jnp.array([0, mech.grid_w - 1]), y=jnp.array([0, mech.grid_h - 1]))
    )
    far_state = state.replace(env_state=raw.replace(agents=far_agents))
    _, far_carry, _ = mech.step(
        attn_map=attn,
        env_state=_batch1(far_state),
        new_env_state=_batch1(far_state),
        action=None,
        env_reward=reward,
        info={},
        done_actors=done,
        carry=carry,
        num_actors=2,
        update_steps=jnp.array(0),
    )
    assert bool(jnp.all(far_carry["feed_mask"] == 0.0))


def test_ocv2_full_map_feed_augmentation_preserves_off_view_attention():
    mech = _make_mate_mechanism(gating=True)
    assert not mech.egocentric
    attn = jnp.zeros((2, mech.feat_h, mech.feat_w))
    attn = attn.at[0, 1, mech.feat_w - 1].set(1.0)
    obs = jnp.zeros((2, mech.img_h * mech.img_w * 3))
    aug = mech.augment_obs(
        obs,
        {
            "partner_attn": attn,
            "feed_mask": jnp.ones((2, mech.feat_h, mech.feat_w), dtype=jnp.float32),
        },
    )
    ch = aug.reshape(2, mech.img_h, mech.img_w, 4)[..., 3]
    c0 = (mech.feat_w - 1) * mech.img_w // mech.feat_w
    c1 = mech.img_w
    r0 = 1 * mech.img_h // mech.feat_h
    r1 = 2 * mech.img_h // mech.feat_h
    assert float(ch[0, r0:r1, c0:c1].max()) > 0.99


def _make_ego_mate_mechanism(border_project):
    from agents.overcooked_v2.ja_overcooked_v2_dense_occupancy import (
        OvercookedV2DenseObjectOccupancyMechanism,
    )
    env = make_env(
        "overcooked-v2",
        {
            "layout": "demo_cook_simple",
            "obs_type": "image",
            "agent_view_size": 2,
            "egocentric": True,
            "agent_fov_size": 5,
            "agent_fov_centered": True,
            "rotate_obs": False,
            "random_agent_positions": False,
        },
    )
    config = {
        "ROLLOUT_LENGTH": 4,
        "NUM_ENVS": 1,
        "FEED_OTHER_ATTN": True,
        "JA_VISIBILITY_GATING": True,
        "JA_FEED_BORDER_PROJECT": border_project,
    }
    return env, OvercookedV2DenseObjectOccupancyMechanism(config, env)


def _delta_attn_on_world_tile(mech, actor, world_r, world_c, agent_r, agent_c):
    """All-mass attention map for `actor` on the crop tile showing world (r, c)."""
    local_r = world_r - (agent_r - mech.view_size)
    local_c = world_c - (agent_c - mech.view_size)
    fr = (local_r * mech.tile_size + mech.tile_size // 2) * mech.feat_h // mech.img_h
    fc = (local_c * mech.tile_size + mech.tile_size // 2) * mech.feat_w // mech.img_w
    attn = jnp.zeros((2, mech.feat_h, mech.feat_w))
    return attn.at[actor, fr, fc].set(1.0)


def _batch1(state):
    """Add the NUM_ENVS=1 leading axis the training mechanism expects."""
    return jax.tree.map(lambda x: x[None] if hasattr(x, "ndim") else x, state)


def test_ocv2_feed_border_project_direction_and_mass():
    # demo_cook_simple fixed starts: agent_0 (x=6, y=2), agent_1 (x=8, y=2).
    # Agent_0 attends the westmost column of ITS crop (x=4) at the onion-route
    # row (1) then the broccoli-route row (3) — inside agent_0's view, outside
    # agent_1's window (x in 6..10). Legacy drops the mass entirely; border
    # projection lands it on agent_1's west border at the SAME ROW, so direction
    # (= flavor) survives the reframe.
    env, mech = _make_ego_mate_mechanism(border_project=True)
    _, state = env.reset(jax.random.PRNGKey(0))
    for route_y in (1, 3):
        attn = _delta_attn_on_world_tile(mech, 0, route_y, 4, 2, 6)
        out = mech._reframe_partner_attention(attn, _batch1(state), _batch1(state), 2)
        rec = out[1]  # agent_1 receives agent_0's map
        assert abs(float(rec.sum()) - 1.0) < 1e-5  # in-grid mass conserved
        fr = (route_y * mech.tile_size + mech.tile_size // 2) * mech.feat_h // mech.img_h
        fc = (0 * mech.tile_size + mech.tile_size // 2) * mech.feat_w // mech.img_w
        assert float(rec[fr, fc]) > 0.99  # west border, row = route row

    env2, mech_off = _make_ego_mate_mechanism(border_project=False)
    _, state2 = env2.reset(jax.random.PRNGKey(0))
    attn = _delta_attn_on_world_tile(mech_off, 0, 1, 4, 2, 6)
    out_off = mech_off._reframe_partner_attention(attn, _batch1(state2), _batch1(state2), 2)
    assert float(out_off[1].sum()) < 1e-6  # legacy: out-of-window mass dropped


def test_ocv2_feed_border_project_in_window_unchanged():
    # Attention on the pot column (x=7, inside both windows) must reframe
    # identically with the flag on and off.
    env, mech_on = _make_ego_mate_mechanism(border_project=True)
    _, mech_off = _make_ego_mate_mechanism(border_project=False)
    _, state = env.reset(jax.random.PRNGKey(0))
    attn = _delta_attn_on_world_tile(mech_on, 0, 2, 7, 2, 6)
    out_on = mech_on._reframe_partner_attention(attn, _batch1(state), _batch1(state), 2)
    out_off = mech_off._reframe_partner_attention(attn, _batch1(state), _batch1(state), 2)
    assert bool(jnp.allclose(out_on, out_off, atol=1e-6))
    assert float(out_on[1].sum()) > 0.99


def test_ocv2_border_project_object_masks():
    # With the flag on, out-of-crop objects get border-clamped masks in the
    # receiver frame while `visible` still reports TRUE in-crop visibility
    # (selection/event-validity semantics unchanged). Object 2 = left onion
    # pile (1,0); blue (actor 1) starts at (8,2) and can never see it.
    env, mech_on = _make_ego_mate_mechanism(border_project=True)
    _, mech_off = _make_ego_mate_mechanism(border_project=False)
    _, state = env.reset(jax.random.PRNGKey(0))
    masks_on, vis_on = mech_on._object_frame(_batch1(state), 2)
    masks_off, vis_off = mech_off._object_frame(_batch1(state), 2)

    assert float(vis_on[1, 2]) == 0.0 and float(vis_off[1, 2]) == 0.0
    assert float(masks_off[1, 2].sum()) == 0.0          # legacy: no mask
    assert float(masks_on[1, 2].sum()) > 0.0            # border blob exists
    # Blob sits on blue's west border at the pile's row (world (1,0) -> blue
    # local tile (1,0)).
    fr = (1 * mech_on.tile_size + mech_on.tile_size // 2) * mech_on.feat_h // mech_on.img_h
    assert float(masks_on[1, 2, fr, 0]) > 0.0
    # In-crop objects unchanged: pot (obj 4) identical masks on/off.
    assert bool(jnp.allclose(masks_on[1, 4], masks_off[1, 4], atol=1e-6))


def test_ocv2_border_project_partner_target_without_receiver_visibility():
    # Partner-attended events whose CELL the receiver cannot see must supervise
    # the aux when the flag is on (witness = partner visible), and must NOT when
    # off (legacy receiver-visibility kill).
    from types import SimpleNamespace

    def traj(mech):
        T, A, M = 2, 2, 2
        mask_all = jnp.zeros((T, A, M, mech.feat_h, mech.feat_w), dtype=jnp.float32)
        mask_all = mask_all.at[:, :, 0, 1, 1].set(1.0)   # obj 0: both see it
        mask_all = mask_all.at[:, :, 1, 3, 0].set(1.0)   # obj 1: border blob in receiver frame
        vis = jnp.ones((T, A, M), dtype=jnp.float32)
        vis = vis.at[:, 0, 1].set(0.0)  # receiver (actor 0) can NOT see obj 1
        sel = jnp.zeros((T, A), dtype=jnp.int32).at[:, 1].set(1)  # partner attends obj 1
        extras = {
            "ja_future_object_mask_all": mask_all,
            "ja_future_object_visible_all": vis,
            "ja_future_object_idx": sel,
            "ja_partner_visible": jnp.ones((T, A), dtype=jnp.float32),
        }
        return SimpleNamespace(done=jnp.zeros((T, A), dtype=bool), extras=extras)

    _, mech_on = _make_ego_mate_mechanism(border_project=True)
    _, mech_off = _make_ego_mate_mechanism(border_project=False)
    _, occ_on = mech_on._occupancy_targets(traj(mech_on))
    _, occ_off = mech_off._occupancy_targets(traj(mech_off))
    # Receiver = actor 0: border-projected event supervises with flag on only.
    assert float(occ_on[:, 0, 3, 0].sum()) > 1.9
    assert float(occ_off[:, 0].sum()) < 1e-3
    # Attender-side visibility still required: making the ATTENDER blind to its
    # own object kills the event even with the flag on.
    t2 = traj(mech_on)
    t2.extras["ja_future_object_visible_all"] = (
        t2.extras["ja_future_object_visible_all"].at[:, 1, 1].set(0.0)
    )
    _, occ_dead = mech_on._occupancy_targets(t2)
    assert float(occ_dead[:, 0].sum()) < 1e-3


def test_ocv2_feed_border_project_eval_parity():
    # The eval-path duplicate must match the training reframe with the flag on.
    from agents.overcooked_v2.ja_overcooked_v2_attention import (
        overcooked_v2_object_ctx,
        reframe_partner_attention_for_eval,
    )
    env, mech = _make_ego_mate_mechanism(border_project=True)
    ctx = overcooked_v2_object_ctx(
        {"ROLLOUT_LENGTH": 4, "NUM_ENVS": 1, "JA_FEED_BORDER_PROJECT": True}, env
    )
    assert ctx["feed_border_project"] is True
    _, state = env.reset(jax.random.PRNGKey(0))
    attn = jnp.zeros((2, mech.feat_h, mech.feat_w))
    attn = attn.at[0, 2, 3].set(0.7).at[0, 8, 1].set(0.3)
    attn = attn.at[1, 4, 4].set(1.0)
    mech_out = mech._reframe_partner_attention(attn, _batch1(state), _batch1(state), 2)
    eval_0, eval_1 = reframe_partner_attention_for_eval(
        attn[0], attn[1], state, state, ctx
    )
    assert bool(jnp.allclose(mech_out[0], eval_0, atol=1e-6))
    assert bool(jnp.allclose(mech_out[1], eval_1, atol=1e-6))
