import jax

from agents.mlp_actor_critic_agent import MLPActorCriticPolicy, ActorWithDoubleCriticPolicy, \
    ActorWithConditionalCriticPolicy, PseudoActorWithDoubleCriticPolicy, \
    PseudoActorWithConditionalCriticPolicy
from agents.rnn_actor_critic_agent import RNNActorCriticPolicy
from agents.s5_actor_critic_agent import S5ActorCriticPolicy
from agents.ja_actor_critic_agent import JAActorCriticPolicy
from agents.ja_image_actor_critic_agent import JAImageActorCriticPolicy
from agents.ja_dual_image_actor_critic_agent import JADualImageActorCriticPolicy
from agents.image_actor_critic_agent import ImageActorCriticPolicy
from envs.base_env import get_inner_env
from agents.liam_agent import LIAMPolicy, initialize_liam_encoder_decoder
from agents.meliba_agent import MeLIBAPolicy, initialize_meliba_encoder_decoder

def initialize_s5_agent(config, env, rng):
    """Initialize an S5 agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        policy: S5ActorCriticPolicy, the policy object
        params: dict, initial parameters for the agent
    """
    # Create the S5 policy with direct parameters
    policy = S5ActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        d_model=config.get("S5_D_MODEL", 128),
        ssm_size=config.get("S5_SSM_SIZE", 128),
        # d_model=config.get("S5_D_MODEL", 16),
        # ssm_size=config.get("S5_SSM_SIZE", 16),
        ssm_n_layers=config.get("S5_N_LAYERS", 2),
        blocks=config.get("S5_BLOCKS", 1),
        fc_hidden_dim=config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 1024),
        fc_n_layers=config.get("FC_N_LAYERS", 3),
        # fc_hidden_dim=config.get("S5_ACTOR_CRITIC_HIDDEN_DIM", 64),
        # fc_n_layers=config.get("FC_N_LAYERS", 2),
        s5_activation=config.get("S5_ACTIVATION", "full_glu"),
        s5_do_norm=config.get("S5_DO_NORM", True),
        s5_prenorm=config.get("S5_PRENORM", True),
        s5_do_gtrxl_norm=config.get("S5_DO_GTRXL_NORM", True),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_rnn_agent(config, env, rng):
    """Initialize an RNN agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        policy: RNNActorCriticPolicy, the policy object
        params: dict, initial parameters for the agent
    """
    # Create the RNN policy
    policy = RNNActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        gru_hidden_dim=config.get("GRU_HIDDEN_DIM", 64),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_ja_agent(config, env, rng):
    """Initialize a Joint Attention agent with the given config.

    Mirrors Lee et al. (2021) architecture exactly (Appendix A):
      - Conv(3x3, 64) + spatial basis (depth 8) + 4 heads (depth 16)
      - LSTM cell size 64
      - Two FC layers hidden size 64 (per head)
      - Actor and critic: identical architecture, no shared weights
      - Scalar features: direction Dense(5), position Dense(5)
    """
    # obs_shape is (width, height, channels) but the actual array is (height, width, channels).
    inner_env = get_inner_env(env)
    obs_width = inner_env.obs_shape[0]
    obs_height = inner_env.obs_shape[1]
    obs_channels = inner_env.obs_shape[2]

    policy = JAActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        obs_height=obs_height,
        obs_width=obs_width,
        obs_channels=obs_channels,
        activation=config.get("ACTIVATION", "relu"),
        conv_filters=config.get("JA_CONV_FILTERS", 64),
        num_heads=config.get("JA_NUM_HEADS", 4),
        head_features=config.get("JA_HEAD_FEATURES", 16),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        lstm_hidden_dim=config.get("LSTM_HIDDEN_DIM", 64),
        spatial_basis_depth=config.get("JA_SPATIAL_BASIS_DEPTH", 8),
        scalar_embed_dim=config.get("JA_SCALAR_EMBED_DIM", 5),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def _get_image_dims(env):
    """Extract image dimensions from an image/FOV wrapper.

    Both wrappers now produce image-only obs (no appended scalars).
    Works for OvercookedImageWrapper, OvercookedFOVWrapper, and LBFImageWrapper.
    """
    wrapper = env._env if hasattr(env, '_env') else env
    num_scalars = getattr(wrapper, 'num_scalar_obs', 0)
    if hasattr(wrapper, 'fov_px'):
        return wrapper.fov_px, wrapper.fov_px, num_scalars
    else:
        img_h = wrapper.grid_height * wrapper.tile_size
        img_w = wrapper.grid_width * wrapper.tile_size
        return img_h, img_w, num_scalars


def initialize_ja_image_agent(config, env, rng):
    """Initialize a Joint Attention agent with image observations."""
    img_h, img_w, num_scalars = _get_image_dims(env)
    num_channels = 4 if config.get("FEED_OTHER_ATTN", False) else 3
    # Communication message is rendered visually (partner border), not as one-hot suffix
    message_dim = 0
    obs_dim = img_h * img_w * num_channels + num_scalars

    policy = JAImageActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=obs_dim,
        img_height=img_h,
        img_width=img_w,
        conv_filters=config.get("CONV_FILTERS", 32),
        conv_num_blocks=config.get("CONV_NUM_BLOCKS", 4),
        conv_kernel_size=config.get("CONV_KERNEL_SIZE", 3),
        conv_stride=config.get("CONV_STRIDE", 2),
        conv_padding=config.get("CONV_PADDING", "SAME"),
        num_heads=config.get("JA_NUM_HEADS", 4),
        head_features=config.get("JA_HEAD_FEATURES", 16),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        lstm_hidden_dim=config.get("LSTM_HIDDEN_DIM", 64),
        spatial_basis_depth=config.get("JA_SPATIAL_BASIS_DEPTH", 8),
        num_channels=num_channels,
        message_dim=message_dim,
        scalar_dim=num_scalars,
        scalar_embed_dim=config.get("JA_SCALAR_EMBED_DIM", 5),
        cross_agent_attn=config.get("CROSS_AGENT_ATTN", False),
        query_partner_lstm=config.get("QUERY_PARTNER_LSTM", False),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_ja_dual_image_agent(config, env, rng):
    """Initialize a Joint Attention dual-critic agent with image observations."""
    img_h, img_w, num_scalars = _get_image_dims(env)
    num_channels = 4 if config.get("FEED_OTHER_ATTN", False) else 3
    obs_dim = img_h * img_w * num_channels + num_scalars

    policy = JADualImageActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=obs_dim,
        img_height=img_h,
        img_width=img_w,
        conv_filters=config.get("CONV_FILTERS", 32),
        conv_num_blocks=config.get("CONV_NUM_BLOCKS", 4),
        conv_kernel_size=config.get("CONV_KERNEL_SIZE", 3),
        conv_stride=config.get("CONV_STRIDE", 2),
        conv_padding=config.get("CONV_PADDING", "SAME"),
        num_heads=config.get("JA_NUM_HEADS", 4),
        head_features=config.get("JA_HEAD_FEATURES", 16),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        lstm_hidden_dim=config.get("LSTM_HIDDEN_DIM", 64),
        spatial_basis_depth=config.get("JA_SPATIAL_BASIS_DEPTH", 8),
        num_channels=num_channels,
        scalar_dim=num_scalars,
        scalar_embed_dim=config.get("JA_SCALAR_EMBED_DIM", 5),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_image_agent(config, env, rng):
    """Initialize an Image agent (ResNet+LSTM, no attention) with image observations."""
    img_h, img_w, num_scalars = _get_image_dims(env)

    policy = ImageActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=env.observation_space(env.agents[0]).shape[0],
        img_height=img_h,
        img_width=img_w,
        conv_filters=config.get("CONV_FILTERS", 32),
        conv_num_blocks=config.get("CONV_NUM_BLOCKS", 4),
        conv_kernel_size=config.get("CONV_KERNEL_SIZE", 3),
        conv_stride=config.get("CONV_STRIDE", 2),
        conv_padding=config.get("CONV_PADDING", "SAME"),
        fc_hidden_dim=config.get("FC_HIDDEN_DIM", 64),
        lstm_hidden_dim=config.get("LSTM_HIDDEN_DIM", 64),
    )

    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_mlp_agent(config, env, rng):
    """
    Initialize an MLP agent with the given config.
    """
    policy = MLPActorCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_actor_with_double_critic(config, env, rng):
    """Initialize an actor with double critic with the given config."""
    policy = ActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_pseudo_actor_with_double_critic(config, env, rng):
    """Initialize a pseudo actor with double critic with the given config."""
    policy = PseudoActorWithDoubleCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_actor_with_conditional_critic(config, env, rng):
    """Initialize an actor with conditional critic with the given config."""
    policy = ActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_pseudo_actor_with_conditional_critic(config, env, rng):
    """Initialize a pseudo actor with conditional critic with the given config."""
    policy = PseudoActorWithConditionalCriticPolicy(
        action_dim=env.action_space(env.agents[0]).n,
        obs_dim=config.get("POLICY_INPUT_DIM", env.observation_space(env.agents[0]).shape[0]),
        pop_size=config["POP_SIZE"],
        activation=config.get("ACTIVATION", "tanh"),
    )
    rng, init_rng = jax.random.split(rng)
    init_params = policy.init_params(init_rng)

    return policy, init_params

def initialize_liam_agent(config, env, rng):
    """Initialize the LIAM ego agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        liam: LIAMPolicy, the policy object
        params: tuple, initial parameters for the {encoder, decoder} and policy
    """
    rng, init_encoder_decoder_rng, init_policy_rng = jax.random.split(rng, 3)

    # Initialize the policy based on the specified type
    if config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_ego_params = initialize_s5_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_ego_params = initialize_mlp_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "rnn":
        ego_policy, init_ego_params = initialize_rnn_agent(config, env, init_policy_rng)

    # Initialize the encoder and decoder for LIAM
    encoder, decoder, init_encoder_decoder_params = initialize_liam_encoder_decoder(config, env, init_encoder_decoder_rng)

    liam = LIAMPolicy(
        policy=ego_policy,
        encoder=encoder,
        decoder=decoder
    )
    params = {'encoder': init_encoder_decoder_params['encoder'],
              'decoder': init_encoder_decoder_params['decoder'],
              'policy': init_ego_params}
    return liam, params

def initialize_meliba_agent(config, env, rng):
    """Initialize the MeLIBA ego agent with the given config.

    Args:
        config: dict, config for the agent
        env: gymnasium environment
        rng: jax.random.PRNGKey, random key for initialization

    Returns:
        meliba: MeLIBAPolicy, the policy object
        params: tuple, initial parameters for the {encoder, decoder} and policy
    """
    rng, init_encoder_decoder_rng, init_policy_rng = jax.random.split(rng, 3)

    # Initialize the policy based on the specified type
    if config["EGO_ACTOR_TYPE"] == "s5":
        ego_policy, init_ego_params = initialize_s5_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "mlp":
        ego_policy, init_ego_params = initialize_mlp_agent(config, env, init_policy_rng)
    elif config["EGO_ACTOR_TYPE"] == "rnn":
        ego_policy, init_ego_params = initialize_rnn_agent(config, env, init_policy_rng)

    # Initialize the encoder and decoder for LIAM
    encoder, decoder, init_encoder_decoder_params = initialize_meliba_encoder_decoder(config, env, init_encoder_decoder_rng)

    meliba = MeLIBAPolicy(
        policy=ego_policy,
        encoder=encoder,
        decoder=decoder
    )
    params = {'encoder': init_encoder_decoder_params['encoder'],
              'decoder': init_encoder_decoder_params['decoder'],
              'policy': init_ego_params}
    return meliba, params
