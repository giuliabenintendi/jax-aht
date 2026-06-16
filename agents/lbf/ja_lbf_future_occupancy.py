"""Future-occupancy Joint Attention mechanism for LBF.

This variant keeps the normal JA image policy but replaces entity-specific LBF
shaping with a post-rollout future-occupancy target:

  R_partner: overlap(attention_i[t], future_occupancy_partner[t])
  R_self:    overlap(attention_i[t], future_occupancy_self[t])

It deliberately avoids fruit-slot pooling. When FEED_OTHER_ATTN is enabled,
the partner's previous spatial attention is appended as a fourth image channel.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.future_occupancy import discounted_first_occupancy, overlap
from agents.lbf.ja_lbf_attention import (
    agent_positions_from_log_state,
    as_spatial_attention,
    lbf_attention_ctx,
)


class FutureOccupancyLBFMechanism:
    """Future-occupancy JA hooks for 2-agent fully observable LBF."""

    name = "lbf-ja-future-occupancy"
    scalar_keys = [
        ("ja_future_partner_overlap", "JA"),
        ("ja_future_self_overlap", "JA"),
        ("ja_future_reward_mean", "JA"),
        ("ja_future_warmup_scale", "JA"),
        ("aux_partner_occ_loss", "Losses"),
    ]

    def __init__(self, config, env):
        self.num_agents = env.num_agents
        if self.num_agents != 2:
            raise NotImplementedError("Future-occupancy LBF JA currently assumes 2 agents.")
        ctx = lbf_attention_ctx(config, env)
        self.img_h, self.img_w = ctx["img_h"], ctx["img_w"]
        self.feat_h, self.feat_w = ctx["feat_h"], ctx["feat_w"]
        self.tile_size = ctx["tile_size"]

        self.partner_coef = float(config.get("JA_FUTURE_PARTNER_COEF", 0.0))
        self.self_coef = float(config.get("JA_FUTURE_SELF_COEF", 0.0))
        self.gamma_occ = float(config.get("JA_FUTURE_GAMMA_OCC", 0.95))
        self.warmup_env_steps = float(config.get("JA_FUTURE_WARMUP_ENV_STEPS", 0.0))
        self.ramp_env_steps = float(config.get("JA_FUTURE_RAMP_ENV_STEPS", 1.0))
        self.aux_occ_coef = float(config.get("JA_FUTURE_AUX_PARTNER_OCC_COEF", 0.0))
        self.feed_other_attn = bool(config.get("FEED_OTHER_ATTN", False))

        # This JA variant is intentionally entity-free.
        self.aux_coef = self.aux_occ_coef
        self.aux_active = self.aux_occ_coef > 0.0

    def entity_feed_dim(self) -> int:
        return 0

    def init_carry(self, num_actors):
        uniform = jnp.ones(
            (num_actors, self.feat_h, self.feat_w),
            dtype=jnp.float32,
        ) / float(self.feat_h * self.feat_w)
        return {"partner_attn": uniform}

    def augment_obs(self, obs_batch_2d, carry):
        if not self.feed_other_attn:
            return obs_batch_2d
        rgb = obs_batch_2d.reshape(obs_batch_2d.shape[0], self.img_h, self.img_w, 3)
        partner_attn = carry["partner_attn"]
        partner_attn_img = jax.image.resize(
            partner_attn,
            (partner_attn.shape[0], self.img_h, self.img_w),
            method="nearest",
        )
        partner_attn_img = partner_attn_img / jnp.maximum(
            partner_attn_img.max(axis=(1, 2), keepdims=True),
            1e-8,
        )
        return jnp.concatenate([rgb, partner_attn_img[..., None]], axis=-1).reshape(
            obs_batch_2d.shape[0], -1,
        )

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del action, info, update_steps
        attn_2d = as_spatial_attention(attn_map).squeeze(0)

        pos_pre = agent_positions_from_log_state(env_state)
        pos_post = agent_positions_from_log_state(new_env_state)
        pos_post = jnp.where(done_actors.reshape(self.num_agents, -1).T[:, :, None],
                             pos_pre, pos_post)
        pos_post_actors = jnp.swapaxes(pos_post, 0, 1).reshape(num_actors, 2)

        extras = {
            "ja_future_attn_2d": jax.lax.stop_gradient(attn_2d.astype(jnp.float32)),
            "ja_future_pos_post": pos_post_actors.astype(jnp.int32),
        }
        swapped_attn = self._swap_partner(attn_2d, num_actors)
        uniform = jnp.ones((self.feat_h, self.feat_w), dtype=jnp.float32) / float(
            self.feat_h * self.feat_w
        )
        new_partner_attn = jnp.where(done_actors[:, None, None], uniform[None], swapped_attn)
        return env_reward, {"partner_attn": jax.lax.stop_gradient(new_partner_attn)}, extras

    def _occupancy_targets(self, traj_batch):
        # Jumanji LBF positions are [row, col]; the shared helper expects [x, y].
        pos_rc = traj_batch.extras["ja_future_pos_post"]
        pos_xy = jnp.stack([pos_rc[..., 1], pos_rc[..., 0]], axis=-1)
        self_occ = discounted_first_occupancy(
            pos_xy, self.gamma_occ, self.tile_size,
            self.feat_h, self.feat_w, self.img_h, self.img_w,
            done=traj_batch.done,
        ).reshape(pos_xy.shape[0], pos_xy.shape[1], self.feat_h, self.feat_w)
        half = self_occ.shape[1] // 2
        partner_occ = jnp.concatenate([self_occ[:, half:], self_occ[:, :half]], axis=1)
        return self_occ, partner_occ

    def _swap_partner(self, x, num_actors):
        half = num_actors // self.num_agents
        return jnp.concatenate([x[half:], x[:half]], axis=0)

    def _warmup_scale(self, config, update_steps):
        env_steps = update_steps.astype(jnp.float32) * float(config["ROLLOUT_LENGTH"] * config["NUM_ENVS"])
        return jnp.clip(
            (env_steps - self.warmup_env_steps) / max(self.ramp_env_steps, 1.0),
            0.0,
            1.0,
        )

    def postprocess_trajectory(self, traj_batch, config, update_steps):
        self_occ, partner_occ = self._occupancy_targets(traj_batch)
        attn = traj_batch.extras["ja_future_attn_2d"]
        valid = (~traj_batch.done).astype(jnp.float32)

        partner_overlap = overlap(attn.reshape(*attn.shape[:2], -1),
                                  partner_occ.reshape(*partner_occ.shape[:2], -1)) * valid
        self_overlap = overlap(attn.reshape(*attn.shape[:2], -1),
                               self_occ.reshape(*self_occ.shape[:2], -1)) * valid
        warm = self._warmup_scale(config, update_steps)
        ja_reward = warm * (
            self.partner_coef * partner_overlap
            + self.self_coef * self_overlap
        )

        extras = dict(traj_batch.extras)
        extras.update({
            "ja_future_self_occ": self_occ,
            "ja_future_partner_occ": partner_occ,
            "ja_future_partner_overlap": partner_overlap,
            "ja_future_self_overlap": self_overlap,
            "ja_future_reward": ja_reward,
            "ja_future_warmup_scale": jnp.broadcast_to(warm, traj_batch.reward.shape),
        })
        return traj_batch._replace(
            reward=traj_batch.reward + ja_reward,
            extras=extras,
        )

    def aux_loss(self, attn_map_apply, traj_batch, config):
        del config
        if not self.aux_active:
            return self.aux_occ_coef, jnp.float32(0.0)
        target = jax.lax.stop_gradient(traj_batch.extras["ja_future_partner_occ"])
        attn = as_spatial_attention(attn_map_apply)
        nll = -(target * jnp.log(attn + 1e-8)).sum(axis=(-2, -1))
        valid = (~traj_batch.done).astype(jnp.float32)
        loss = (nll * valid).sum() / jnp.maximum(valid.sum(), 1e-8)
        return self.aux_occ_coef, loss

    def rollout_metrics(self, traj_batch, loss_info):
        return {
            "ja_future_partner_overlap": traj_batch.extras["ja_future_partner_overlap"].mean(),
            "ja_future_self_overlap": traj_batch.extras["ja_future_self_overlap"].mean(),
            "ja_future_reward_mean": traj_batch.extras["ja_future_reward"].mean(),
            "ja_future_warmup_scale": traj_batch.extras["ja_future_warmup_scale"].mean(),
            "aux_partner_occ_loss": loss_info.aux_loss.mean(),
        }

    def report(self, config, out, logger):
        from common.train_logging import (
            IMAGE_IPPO_SCALAR_KEYS,
            report_basic_training_outputs,
        )
        report_basic_training_outputs(
            config, out, logger,
            scalar_keys=list(IMAGE_IPPO_SCALAR_KEYS) + list(self.scalar_keys),
            print_prefix="ja_ippo:lbf_future_occupancy",
        )

    def eval_outputs(self, algorithm_config, env, out, logger):
        del algorithm_config, env, out, logger
