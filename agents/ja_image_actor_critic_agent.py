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
        num_scalars: int = 6,
        activation: str = "relu",
        conv_filters: int = 64,
        num_heads: int = 4,
        head_features: int = 16,
        fc_hidden_dim: int = 64,
        lstm_hidden_dim: int = 64,
        spatial_basis_depth: int = 8,
        scalar_embed_dim: int = 5,
    ):
        # Skip JAActorCriticPolicy.__init__ — we set self.network directly
        # but still call AgentPolicy.__init__ for action_dim/obs_dim
        from agents.agent_interface import AgentPolicy
        AgentPolicy.__init__(self, action_dim, obs_dim)

        self.img_height = img_height
        self.img_width = img_width
        self.network = JAImageActorCritic(
            action_dim=action_dim,
            img_height=img_height,
            img_width=img_width,
            num_scalars=num_scalars,
            conv_filters=conv_filters,
            num_heads=num_heads,
            head_features=head_features,
            fc_hidden_dim=fc_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            spatial_basis_depth=spatial_basis_depth,
            scalar_embed_dim=scalar_embed_dim,
            activation=activation,
        )
        self.lstm_hidden_dim = lstm_hidden_dim
