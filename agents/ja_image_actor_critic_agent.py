"""Policy wrapper for the Joint Attention Actor-Critic with image observations.

Inherits all hstate packing/unpacking and action methods from JAActorCriticPolicy.
Only overrides __init__ to construct JAImageActorCritic instead of JAActorCritic.
"""
from agents.ja_actor_critic_agent import JAActorCriticPolicy
from agents.ja_image_actor_critic import JAImageActorCritic


class JAImageActorCriticPolicy(JAActorCriticPolicy):

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        img_height: int,
        img_width: int,
        conv_filters: int = 32,
        conv_num_blocks: int = 4,
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        conv_padding: str = "SAME",
        num_heads: int = 4,
        head_features: int = 16,
        fc_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        spatial_basis_depth: int = 8,
        num_channels: int = 3,
        message_dim: int = 0,
    ):
        # Skip JAActorCriticPolicy.__init__ — we set self.network directly
        # but still call AgentPolicy.__init__ for action_dim/obs_dim
        from agents.agent_interface import AgentPolicy
        AgentPolicy.__init__(self, action_dim, obs_dim)

        self.img_height = img_height
        self.img_width = img_width
        self.message_dim = message_dim
        self.network = JAImageActorCritic(
            action_dim=action_dim,
            img_height=img_height,
            img_width=img_width,
            conv_filters=conv_filters,
            conv_num_blocks=conv_num_blocks,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            conv_padding=conv_padding,
            num_heads=num_heads,
            head_features=head_features,
            fc_hidden_dim=fc_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            spatial_basis_depth=spatial_basis_depth,
            num_channels=num_channels,
            message_dim=message_dim,
        )
        self.lstm_hidden_dim = lstm_hidden_dim
