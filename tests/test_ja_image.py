"""Tests for image-based JA pipeline: rendering, wrapper, network, training."""
import math
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image

from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.overcooked.overcooked_v1 import OvercookedV1
from envs.overcooked.rendering import render_state
from envs.overcooked.rendering.overcooked_rendering import TILE_PIXELS


def test_render_state_shape():
    """render_state produces (H*7, W*7, 3) uint8 from an OvercookedV1 state."""
    env = OvercookedV1()
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    img = render_state(state)
    h, w = env.height, env.width
    assert img.shape == (h * TILE_PIXELS, w * TILE_PIXELS, 3)
    assert img.dtype == jnp.uint8


def test_save_image_obs():
    """Save a sample image observation as PNG for visual inspection."""
    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "image"})
    rng = jax.random.PRNGKey(0)
    obs, _ = env.reset(rng)

    h = env.grid_height * TILE_PIXELS
    w = env.grid_width * TILE_PIXELS

    obs_0 = np.array(obs["agent_0"])
    img = (obs_0 * 255).astype(np.uint8).reshape(h, w, 3)

    print(f"\nImage shape: ({h}, {w}, 3)")

    scale = 10
    img_large = np.kron(img, np.ones((scale, scale, 1))).astype(np.uint8)
    Image.fromarray(img_large).save("image_obs.png")
    print(f"Saved image_obs.png ({img_large.shape[0]}x{img_large.shape[1]})")


def test_image_wrapper():
    """OvercookedImageWrapper produces correct obs shape and values in [0,1]."""
    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "image"})
    rng = jax.random.PRNGKey(1)

    obs, state = env.reset(rng)
    for agent in env.agents:
        assert obs[agent].shape == (env.observation_space(agent).shape[0],)
        assert float(obs[agent].min()) >= 0.0
        assert float(obs[agent].max()) <= 1.0

    # Step and check auto-reset works
    actions = {"agent_0": jnp.int32(0), "agent_1": jnp.int32(0)}
    rng, step_rng = jax.random.split(rng)
    obs2, state2, rewards, dones, info = env.step(step_rng, state, actions)
    for agent in env.agents:
        assert obs2[agent].shape == obs[agent].shape


def test_make_env_image():
    """make_env with obs_type='image' returns a working env; symbolic path unchanged."""
    # Image path
    img_env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "image"})
    rng = jax.random.PRNGKey(2)
    obs, _ = img_env.reset(rng)
    img_obs_dim = obs["agent_0"].shape[0]

    # Symbolic path (default)
    sym_env = make_env("overcooked-v1", {"layout": "cramped_room"})
    obs_sym, _ = sym_env.reset(rng)
    sym_obs_dim = obs_sym["agent_0"].shape[0]

    # Image obs should be much larger than symbolic
    assert img_obs_dim > sym_obs_dim


def test_ja_image_forward_pass():
    """Init JA image policy with tiny sizes, run forward pass, check shapes."""
    from agents.ja_image_actor_critic_agent import JAImageActorCriticPolicy

    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "image"})
    image_wrapper = env  # no LogWrapper here

    img_h = image_wrapper.grid_height * image_wrapper.tile_size
    img_w = image_wrapper.grid_width * image_wrapper.tile_size

    policy = JAImageActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        img_height=img_h,
        img_width=img_w,
        num_scalars=0,
        conv_filters=4,
        num_heads=2,
        head_features=4,
        fc_hidden_dim=4,
        lstm_hidden_dim=4,
        spatial_basis_depth=4,
        scalar_embed_dim=2,
    )

    rng = jax.random.PRNGKey(3)
    rng, init_rng, act_rng = jax.random.split(rng, 3)
    params = policy.init_params(init_rng)

    batch_size = 1
    seq_len = 1
    hstate = policy.init_hstate(batch_size)

    obs_dim = env.observation_space(env.agents[0]).shape[0]
    dummy_obs = jnp.zeros((seq_len, batch_size, obs_dim))
    dummy_done = jnp.zeros((seq_len, batch_size))
    dummy_avail = jnp.ones((seq_len, batch_size, env.action_space(env.agents[0]).n))

    action, val, pi, new_hstate, attn_map = policy.get_action_value_policy(
        params, dummy_obs, dummy_done, dummy_avail, hstate, act_rng,
    )
    assert action.shape == (seq_len, batch_size)
    assert val.shape == (seq_len, batch_size)
    assert new_hstate.shape == hstate.shape
    # Attention map should be at downsampled resolution (÷4)
    import math
    expected_ah = math.ceil(img_h / 4)
    expected_aw = math.ceil(img_w / 4)
    assert attn_map.shape == (seq_len, batch_size, expected_ah, expected_aw)


def test_ja_image_train_loop():
    """Run make_train for 2 updates with image obs — catches shape mismatches."""
    from marl.ja_ippo import make_train

    env = make_env("overcooked-v1", {
        "layout": "cramped_room", "max_steps": 10, "obs_type": "image",
    })
    env = LogWrapper(env)

    config = {
        "ENV_NAME": "overcooked-v1",
        "ENV_KWARGS": {"layout": "cramped_room", "max_steps": 10, "obs_type": "image"},
        "OBS_TYPE": "image",
        "NUM_ENVS": 2,
        "TOTAL_TIMESTEPS": 40,
        "ROLLOUT_LENGTH": 10,
        "NUM_MINIBATCHES": 2,
        "UPDATE_EPOCHS": 1,
        "NUM_CHECKPOINTS": 2,
        "LR": 1e-3,
        "ANNEAL_LR": False,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 1.0,
        "ACTIVATION": "relu",
        "JA_CONV_FILTERS": 4,
        "JA_NUM_HEADS": 2,
        "JA_HEAD_FEATURES": 4,
        "JA_SPATIAL_BASIS_DEPTH": 4,
        "JA_SCALAR_EMBED_DIM": 2,
        "FC_HIDDEN_DIM": 4,
        "LSTM_HIDDEN_DIM": 4,
        "JA_BETA_MAX": 0.01,
        "JA_WARMUP_ENV_STEPS": 100,
        "TRAIN_SEED": 0,
    }

    train_fn = make_train(config, env)
    rng = jax.random.PRNGKey(0)
    out = jax.jit(train_fn)(rng)

    assert "final_params" in out
    assert "metrics" in out
    assert "params" in out["final_params"]
    assert jnp.all(out["metrics"]["jsd_mean"] >= 0)


def test_network_architecture():
    """Print the full network architecture with nn.tabulate and compare to reference.

    Run with: uv run pytest -s tests/test_ja_image.py::test_network_architecture
    """
    import flax.linen as nn
    from agents.ja_image_actor_critic import JAImageActorCritic, JAImageScannedLSTM

    # -- Setup: cramped_room with production-size hyperparams --
    env = make_env("overcooked-v1", {"layout": "cramped_room", "obs_type": "image"})
    img_h = env.grid_height * TILE_PIXELS  # 5*7 = 35
    img_w = env.grid_width * TILE_PIXELS   # 5*7 = 35
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    action_dim = env.action_space(env.agents[0]).n
    feat_h = math.ceil(img_h / 4)  # after 2x MaxPool(stride=2)
    feat_w = math.ceil(img_w / 4)

    conv_filters = 64
    num_heads = 4
    head_features = 16
    lstm_hidden_dim = 64
    spatial_basis_depth = 8
    scalar_embed_dim = 5
    fc_hidden_dim = 64

    # -- 1. nn.tabulate on the core module (ResNet + attention + LSTM) --
    # Tabulate JAImageScannedLSTM directly — the full JAImageActorCritic
    # can't be tabulated because distrax.Categorical isn't YAML-serializable.
    lstm_module = JAImageScannedLSTM(
        img_height=img_h,
        img_width=img_w,
        num_scalars=0,
        conv_filters=conv_filters,
        num_heads=num_heads,
        head_features=head_features,
        lstm_hidden_dim=lstm_hidden_dim,
        spatial_basis_depth=spatial_basis_depth,
        scalar_embed_dim=scalar_embed_dim,
    )

    batch = 1
    seq = 1
    carry = JAImageScannedLSTM.initialize_carry(batch, lstm_hidden_dim)

    dummy_obs = jnp.zeros((seq, batch, obs_dim))
    dummy_done = jnp.zeros((seq, batch))
    dummy_partner_h = jnp.zeros((seq, batch, lstm_hidden_dim))
    scan_input = (dummy_obs, dummy_done, dummy_partner_h)

    print("\n" + "=" * 80)
    print("JAImageScannedLSTM — nn.tabulate()")
    print("(this is the core module; actor/critic each have an identical copy)")
    print("=" * 80)
    table_fn = nn.tabulate(lstm_module, jax.random.PRNGKey(0))
    print(table_fn(carry, scan_input))

    print("\nFC heads on top (per actor/critic, not shown in tabulate):")
    print(f"  LSTM output ({lstm_hidden_dim},)")
    print(f"  -> Dense({fc_hidden_dim}) -> ReLU")
    print(f"  -> Dense({fc_hidden_dim}) -> ReLU")
    print(f"  -> actor: Dense({action_dim}) -> Categorical logits")
    print(f"  -> critic: Dense(1) -> scalar value")

    # -- 2. Concrete shapes at each stage --
    print("\n" + "=" * 80)
    print("DATA FLOW — per agent, per timestep (cramped_room)")
    print("=" * 80)
    print(f"""
Observation:
  flat obs              : ({obs_dim},) = image pixels ({img_h}*{img_w}*3 = {img_h*img_w*3}), no scalars

Image encoder (2 ResNet stacks, each: Conv3x3 -> MaxPool(stride=2) -> 2 ResBlocks):
  input image           : ({img_h}, {img_w}, 3)         = ({env.grid_height}x{TILE_PIXELS}, {env.grid_width}x{TILE_PIXELS}, RGB)
  after Stack 0         : ({math.ceil(img_h/2)}, {math.ceil(img_w/2)}, {conv_filters//2})      filters={conv_filters//2}
  after Stack 1         : ({feat_h}, {feat_w}, {conv_filters})       filters={conv_filters}
  after ReLU            : ({feat_h}, {feat_w}, {conv_filters})       = features F

Spatial attention:
  spatial basis         : ({feat_h}, {feat_w}, {spatial_basis_depth})        sinusoidal positional encoding
  F + basis concat      : ({feat_h}, {feat_w}, {conv_filters + spatial_basis_depth})
  Keys (1x1 conv)       : ({feat_h}*{feat_w}, {num_heads}, {head_features})  = ({feat_h*feat_w}, {num_heads}, {head_features})
  Values (1x1 conv)     : ({feat_h}*{feat_w}, {num_heads}, {head_features})  = ({feat_h*feat_w}, {num_heads}, {head_features})
  Query (Dense)         : ({num_heads}, {head_features})            from concat(own_lstm_h, partner_lstm_h)
  attn logits           : ({feat_h*feat_w}, {num_heads})           Q . K, no sqrt scaling
  attn weights          : ({feat_h*feat_w}, {num_heads})           softmax over spatial dim
  attended output       : ({num_heads}, {head_features}) -> flat ({num_heads * head_features},)

LSTM input:
  attended features     : ({num_heads * head_features},)
  LSTM hidden dim       : {lstm_hidden_dim}

FC heads (after LSTM):
  Dense -> ReLU         : ({fc_hidden_dim},)
  Dense -> ReLU         : ({fc_hidden_dim},)
  actor projection      : ({action_dim},) -> Categorical logits
  critic projection     : (1,) -> scalar value

Attention map output:
  mean over {num_heads} heads    : ({feat_h}, {feat_w})    ~{feat_h/env.grid_height:.1f} feature px per grid cell
""")

    # -- 3. Comparison with reference code --
    print("=" * 80)
    print("COMPARISON: reference code (attention_networks.py) vs ours")
    print("=" * 80)
    print(f"""
                          Reference (use_stacks=True)     Ours (JAImageActorCritic)
                          ─────────────────────────────   ──────────────────────────────
Image input               (H, W, C) structured grid       ({img_h}, {img_w}, 3) RGB pixels
Encoder                   Stack(f//2) -> Stack(f) -> ReLU  Stack(f//2) -> Stack(f) -> ReLU   [SAME]
Feature resolution        (H//4, W//4, f)                  ({feat_h}, {feat_w}, {conv_filters})
conv_filters (default)    8                                64
Spatial basis depth       16                               8
Spatial basis             get_spatial_basis (0-indexed)     make_sinusoidal_spatial_basis      [SAME formula]

Q input                   concat(all agents' h AND c)      concat(own_h, partner_h)           [h only, not c]
K projection              Conv2D(conv_filters, 1x1)        Conv(m*c_m, 1x1)
V projection              Conv2D(conv_filters, 1x1)        Conv(m*c_m, 1x1)
Attention heads           conv_filters // depth_per_head    {num_heads} heads x {head_features} features
Attention computation     sum(Q*K, axis=-1) -> softmax      einsum(K, Q) -> softmax            [equivalent]

Direction embed           one_hot(4) -> Dense(5)            [none — no scalars]                [DIFFERENT]
Position embed            cast_and_scale -> Dense(5)        [none — no scalars]                [DIFFERENT]

FC BEFORE LSTM            Dense(200) -> Dense(100)          [none]                             [DIFFERENT]
LSTM                      LSTM(128)                         LSTM({lstm_hidden_dim})
FC AFTER LSTM             [none]                            Dense({fc_hidden_dim}) x2          [DIFFERENT]
Actor output              CategoricalProjection             Dense({action_dim})

Attn map for JSD          (H//4, W//4)                      ({feat_h}, {feat_w})
Actor/Critic              separate, no shared weights       separate, no shared weights        [SAME]
""")

    # -- 4. Parameter count --
    full_network = JAImageActorCritic(
        action_dim=action_dim,
        img_height=img_h,
        img_width=img_w,
        num_scalars=0,
        conv_filters=conv_filters,
        num_heads=num_heads,
        head_features=head_features,
        fc_hidden_dim=fc_hidden_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        spatial_basis_depth=spatial_basis_depth,
        scalar_embed_dim=scalar_embed_dim,
    )
    actor_carry = JAImageScannedLSTM.initialize_carry(batch, lstm_hidden_dim)
    critic_carry = JAImageScannedLSTM.initialize_carry(batch, lstm_hidden_dim)
    full_hidden = (actor_carry, critic_carry)
    dummy_avail = jnp.ones((seq, batch, action_dim))
    full_x = (dummy_obs, dummy_done, dummy_avail, dummy_partner_h)
    params = full_network.init(jax.random.PRNGKey(0), full_hidden, full_x)
    param_count = sum(p.size for p in jax.tree.leaves(params))
    print(f"Total parameters (full actor-critic): {param_count:,}")
    print()
