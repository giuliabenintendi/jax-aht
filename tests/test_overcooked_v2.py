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
from envs.overcooked_v2.common import Actions, StaticObject
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
