"""Multi-Destination Spread joint-attention mechanism for the unified ja_ippo trainer.

Plugs the coverage-env JA behaviour into the env-agnostic trainer skeleton in
`marl/ja_ippo.py`. Unlike the card game (2 agents must agree on the SAME card)
this is an N-agent *coverage* task: agents must spread to DISTINCT goals. The JA
signal is therefore not attention agreement but partner-target awareness:

  - aux loss (the method): each agent's per-goal attention is trained (soft
    cross-entropy) to predict where its partners are attending — their
    aggregated per-goal attention. This shapes the representation to encode
    partner intent; the policy, optimising the coverage reward, is then free to
    route to an UNcovered goal.
  - partner feed (opt-in, off by default): append the partners' aggregated
    per-goal attention to the obs as `num_goals` scalars. It works at train time
    (the trainer sizes the net from `entity_feed_dim()`), but feed-on is NOT yet
    usable end to end: (a) the N-agent cross-play rollout
    (`run_xp_nagent_from_params`) does not inject the feed, so XP mismatches the
    obs shape; (b) on eval/XP reload `JA_ENTITY_FEED_DIM` is not in the saved
    config, so it must be re-resolved from this mechanism (deliberately kept out
    of the shared `initialize_ja_image_agent` to avoid per-env branches there).
    Both are self-contained follow-ups; keep the feed off until they land.

Goals are static within a layout, so each goal's feature cell is precomputed
once (no per-step positions / `eaten` mask, unlike LBF fruits). N agents (4 in
the cardinal layout), parameter-shared; the partner aggregate is a mean over the
other agents, which is permutation-invariant — correct because the partners
share one obs colour.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


class MDSMechanism:
    """Env-specific JA hooks for Multi-Destination Spread (N agents, param-shared)."""

    name = "multi-destination-spread"
    scalar_keys = [
        ("aux_partner_target_loss", "Losses"),
        ("partner_on_goal_mass_mean", "JA"),
    ]

    def __init__(self, config, env):
        from agents.initialize_agents import _get_image_dims
        from agents.ja_actor_critic import _compute_resnet_output_dims
        from envs.base_env import get_inner_env

        self.num_agents = env.num_agents
        inner = get_inner_env(env)
        goal_pos = inner.goal_pos          # (K, 2) int [x, y], static within a layout
        self.num_goals = int(goal_pos.shape[0])
        tile_size = int(inner.tile_size)

        img_h, img_w, _ = _get_image_dims(env)
        feat_h, feat_w = _compute_resnet_output_dims(
            img_h, img_w,
            stride=config.get("CONV_STRIDE", 2),
            kernel_size=config.get("CONV_KERNEL_SIZE", 3),
            padding=config.get("CONV_PADDING", "SAME"),
            num_blocks=config.get("CONV_NUM_BLOCKS", 4),
        )
        self.feat_h, self.feat_w = feat_h, feat_w

        # Each goal's flat feature-cell index (tile-centre pixel -> feature cell),
        # mirroring per_fruit_attn's mapping. goal_pos[:,0]=x->col, [:,1]=y->row.
        centre_r = goal_pos[:, 1] * tile_size + tile_size // 2
        centre_c = goal_pos[:, 0] * tile_size + tile_size // 2
        fr = jnp.clip(centre_r * feat_h // img_h, 0, feat_h - 1).astype(jnp.int32)
        fc = jnp.clip(centre_c * feat_w // img_w, 0, feat_w - 1).astype(jnp.int32)
        self.goal_feat_idx = (fr * feat_w + fc).astype(jnp.int32)   # (K,)

        self.aux_coef = float(config.get("JA_AUX_PARTNER_ARGMAX_COEF", 0.0))
        self.aux_active = self.aux_coef > 0.0
        self.partner_feed_active = bool(config.get("JA_MDS_PARTNER_FEED", False))

    def entity_feed_dim(self) -> int:
        return self.num_goals if self.partner_feed_active else 0

    def init_carry(self, num_actors):
        """Partner's previous aggregated per-goal attention (uniform) + validity."""
        partner_dest_attn = (
            jnp.ones((num_actors, self.num_goals), dtype=jnp.float32) / float(self.num_goals)
        )
        partner_valid = jnp.zeros((num_actors,), dtype=bool)
        return partner_dest_attn, partner_valid

    def augment_obs(self, obs_batch_2d, carry):
        if not self.partner_feed_active:
            return obs_batch_2d
        partner_dest_attn, _ = carry
        return jnp.concatenate([obs_batch_2d, partner_dest_attn], axis=-1)

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del env_state, new_env_state, action, info, update_steps  # coverage shaping uses none
        prev_partner_dest_attn, prev_partner_valid = carry

        attn_2d = attn_map.squeeze(0)                          # (num_actors, fh, fw)
        per_goal_norm, on_mass = self._per_goal(attn_2d)
        partner_mean = self._partner_mean(per_goal_norm, num_actors)

        # Aux target: with the partner feed ON use the PREVIOUS step's aggregate
        # (causal, matches what the obs was fed); with it OFF use the current
        # step's aggregate (there is no obs leak to be causal about).
        if self.partner_feed_active:
            aux_target = prev_partner_dest_attn
            aux_valid = prev_partner_valid
        else:
            aux_target = partner_mean
            aux_valid = jnp.ones((num_actors,), dtype=bool)

        # No intrinsic reward shaping: the JA signal is the aux loss (+ optional
        # feed); spreading is driven by the env coverage reward.
        reward = env_reward

        uniform = jnp.ones((self.num_goals,), dtype=jnp.float32) / float(self.num_goals)
        new_partner_dest_attn = jnp.where(done_actors[:, None], uniform[None], partner_mean)
        new_carry = (new_partner_dest_attn, ~done_actors)

        extras = {
            "partner_dest_attn": jax.lax.stop_gradient(aux_target.astype(jnp.float32)),
            "partner_valid": aux_valid,
            "partner_on_goal_mass": on_mass,
        }
        return reward, new_carry, extras

    def aux_loss(self, attn_map_apply, traj_batch, config):
        """Soft cross-entropy pushing each agent's per-goal attention to predict
        the partners' aggregated per-goal attention (partner-target prediction)."""
        del config
        if not self.aux_active:
            return self.aux_coef, jnp.float32(0.0)
        ex = traj_batch.extras
        agent_per_goal, _ = self._per_goal(attn_map_apply)     # (T, A, K)
        target_soft = jax.lax.stop_gradient(ex["partner_dest_attn"])
        log_probs = jnp.log(agent_per_goal + 1e-8)
        nll_flat = -(target_soft * log_probs).sum(axis=-1).reshape(-1)
        # Weight by partner confidence (peak mass) and validity, matching LBF:
        # a near-uniform partner aggregate (~1/K peak) contributes little.
        partner_peak = target_soft.max(axis=-1)
        aux_weight = jax.lax.stop_gradient(
            ex["partner_valid"].reshape(-1).astype(jnp.float32)
            * partner_peak.reshape(-1)
        )
        aux_loss = (nll_flat * aux_weight).sum() / jnp.maximum(aux_weight.sum(), 1e-8)
        return self.aux_coef, aux_loss

    def rollout_metrics(self, traj_batch, loss_info):
        return {
            "aux_partner_target_loss": loss_info.aux_loss.mean(),
            "partner_on_goal_mass_mean": traj_batch.extras["partner_on_goal_mass"].mean(),
        }

    def report(self, config, out, logger):
        from common.train_logging import report_ja_training_outputs
        report_ja_training_outputs(config, out, logger)

    def eval_outputs(self, algorithm_config, env, out, logger):
        """N-agent eval video on the best-ckpt params (guarded; XP runs separately)."""
        import hydra

        from agents.initialize_agents import initialize_ja_image_agent
        from envs.base_env import get_inner_env

        policy, _ = initialize_ja_image_agent(algorithm_config, env, jax.random.PRNGKey(0))
        savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 100))
        try:
            from common.eval_media import rollout_and_log_video
            rollout_and_log_video(
                jax.random.PRNGKey(42), get_inner_env(env), algorithm_config["ENV_NAME"],
                jax.tree.map(lambda x: x[0], out["final_params"]), policy, max_steps,
                tag="Eval/episode_video", savedir=f"{savedir}/videos", logger=logger,
            )
        except Exception as e:
            print(f"[ja_ippo:{self.name}] WARN: N-agent eval video failed ({e}); continuing.", flush=True)

    # -------------------------------------------------------------- internals
    def _per_goal(self, attn_2d):
        """Point-gather attention at each static goal's feature cell, normalised
        over the mass that landed on goals. attn_2d: (..., feat_h, feat_w)."""
        attn_flat = attn_2d.reshape(*attn_2d.shape[:-2], self.feat_h * self.feat_w)
        per_goal = attn_flat[..., self.goal_feat_idx]          # (..., K)
        on_mass = per_goal.sum(axis=-1)
        per_goal_norm = per_goal / (on_mass[..., None] + 1e-8)
        return per_goal_norm, on_mass

    def _partner_mean(self, per_actor, num_actors):
        """Mean per-goal attention over the OTHER agents, per actor.

        Actor order is [agent_0 envs, agent_1 envs, ...]; the mean excludes self
        and is permutation-invariant over partners (they share one obs colour).
        """
        num_envs = num_actors // self.num_agents
        per_3d = per_actor.reshape(self.num_agents, num_envs, self.num_goals)
        total = per_3d.sum(axis=0)                              # (num_envs, K)
        partner_sum = total[None] - per_3d                     # (num_agents, num_envs, K)
        partner_mean = partner_sum / float(self.num_agents - 1)
        return partner_mean.reshape(num_actors, self.num_goals)


## Tests

def test_partner_mean_excludes_self():
    """Each actor's partner aggregate is the mean of the OTHER agents only."""
    from envs.multi_destination_spread.multi_destination_spread import (
        MultiDestinationSpreadEnv,
    )

    env = MultiDestinationSpreadEnv(layout="cardinal_5x5", max_steps=20)
    mech = MDSMechanism({"JA_AUX_PARTNER_ARGMAX_COEF": 0.1}, env)
    k = mech.num_goals
    # One env, 4 agents each one-hot on a distinct goal slot.
    per_actor = jnp.eye(mech.num_agents, k, dtype=jnp.float32)
    pm = mech._partner_mean(per_actor, mech.num_agents)
    # agent_0's partners are agents 1..3 -> zero on slot 0, 1/3 on slots 1..3.
    expected0 = jnp.array([0.0, 1 / 3, 1 / 3, 1 / 3])[:k]
    assert jnp.allclose(pm[0], expected0, atol=1e-6)
    assert mech.entity_feed_dim() == 0  # partner feed off by default


def test_per_goal_gather_shapes():
    from envs.multi_destination_spread.multi_destination_spread import (
        MultiDestinationSpreadEnv,
    )

    env = MultiDestinationSpreadEnv(layout="cardinal_5x5", max_steps=20)
    mech = MDSMechanism({"JA_AUX_PARTNER_ARGMAX_COEF": 0.1}, env)
    attn = jnp.ones((8, mech.feat_h, mech.feat_w), dtype=jnp.float32)
    per_goal, on_mass = mech._per_goal(attn)
    assert per_goal.shape == (8, mech.num_goals)
    assert on_mass.shape == (8,)
    assert jnp.allclose(per_goal.sum(axis=-1), 1.0, atol=1e-5)
