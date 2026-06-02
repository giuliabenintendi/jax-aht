"""LBF joint-attention mechanism for the unified ja_ippo trainer.

Plugs the LBF-specific behaviour into the env-agnostic trainer skeleton in
`marl/ja_ippo.py`: per-fruit attention pooling, the partner per-fruit feed,
the own-argmax distance-progress shaping reward (r_self), and the per-fruit
soft cross-entropy aux loss. No Other-Play; fruits are lex-sorted each step so
slot k is a stable fruit identity within an episode.

The trainer owns the rollout/PPO/eval/XP skeleton and calls these hooks:
  entity_feed_dim()                  scalar-suffix width for the partner feed
  init_carry(num_actors)             per-step carry (partner attn + validity)
  augment_obs(obs, carry)            append the partner feed
  step(...)                          pool attention, shape reward, advance carry
  aux_loss(attn_map_apply, traj)     per-fruit soft-CE term
  rollout_metrics(traj, loss_info)   scalars for logging
  eval_outputs(cfg, env, out, log)   eval video + cross-play
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.lbf.ja_lbf_attention import (
    agent_positions_from_log_state,
    as_spatial_attention,
    food_state_from_log_state,
    lbf_attention_ctx,
    lex_sort_food,
    per_fruit_attn,
    swap_partner,
)


class LBFMechanism:
    """Env-specific JA hooks for Level-Based Foraging (2 agents, param-shared)."""

    name = "lbf"
    scalar_keys = [
        ("aux_partner_argmax_loss", "Losses"),
        ("r_self_mean", "JA"),
    ]

    def __init__(self, config, env):
        self.num_agents = env.num_agents
        ctx = lbf_attention_ctx(config, env)
        self.img_h, self.img_w = ctx["img_h"], ctx["img_w"]
        self.feat_h, self.feat_w = ctx["feat_h"], ctx["feat_w"]
        self.tile_size = ctx["tile_size"]
        self.num_fruits = ctx["num_fruits"]
        self.aux_coef = float(config.get("JA_AUX_PARTNER_ARGMAX_COEF", 0.0))
        self.r_self_coef = float(config.get("JA_FRUIT_R_SELF_COEF", 0.0))
        self.aux_active = self.aux_coef > 0.0
        self.partner_feed_active = bool(config.get("JA_FRUIT_PARTNER_FEED", True))

    def entity_feed_dim(self) -> int:
        return self.num_fruits if self.partner_feed_active else 0

    def init_carry(self, num_actors):
        """Partner's previous per-fruit attention (uniform) + validity flag."""
        partner_fruit_attn = (
            jnp.ones((num_actors, self.num_fruits), dtype=jnp.float32) / float(self.num_fruits)
        )
        partner_valid = jnp.zeros((num_actors,), dtype=bool)
        return partner_fruit_attn, partner_valid

    def augment_obs(self, obs_batch_2d, carry):
        if not self.partner_feed_active:
            return obs_batch_2d
        partner_fruit_attn, _ = carry
        return jnp.concatenate([obs_batch_2d, partner_fruit_attn], axis=-1)

    def _tile_to_actors(self, x_env):
        reps = (self.num_agents,) + (1,) * (x_env.ndim - 1)
        return jnp.tile(x_env, reps)

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del action, info, update_steps  # LBF shaping uses none of these
        prev_partner_fruit_attn, prev_partner_valid = carry

        attn_2d = as_spatial_attention(attn_map).squeeze(0)
        # Pre-step food state, lex-sorted so slot k is a stable fruit identity.
        food_pos_env_raw, food_eaten_env_raw = food_state_from_log_state(env_state)
        food_pos_env, food_eaten_env = lex_sort_food(food_pos_env_raw, food_eaten_env_raw)
        food_pos_actors = self._tile_to_actors(food_pos_env)
        food_eaten_actors = self._tile_to_actors(food_eaten_env)

        per_fruit_norm, _on_mass = per_fruit_attn(
            attn_2d, food_pos_actors, food_eaten_actors,
            self.tile_size, self.feat_h, self.feat_w, self.img_h, self.img_w,
        )
        own_argmax = jnp.argmax(per_fruit_norm, axis=-1).astype(jnp.int32)

        pos_pre = agent_positions_from_log_state(env_state)
        pos_pre_actors = jnp.swapaxes(pos_pre, 0, 1).reshape(num_actors, 2)
        pos_post = agent_positions_from_log_state(new_env_state)
        pos_post_actors = jnp.swapaxes(pos_post, 0, 1).reshape(num_actors, 2)

        # Distance progress toward the agent's OWN current-argmax fruit
        # (card-game r_attn_self analog), weighted by its own peak mass. Zeroed
        # on the terminal transition where the env has auto-reset.
        actor_idx = jnp.arange(num_actors)
        own_target_pos = food_pos_actors[actor_idx, own_argmax]
        d_pre = jnp.sum(jnp.abs(pos_pre_actors - own_target_pos), axis=-1).astype(jnp.float32)
        d_post = jnp.sum(jnp.abs(pos_post_actors - own_target_pos), axis=-1).astype(jnp.float32)
        own_peak_mass = per_fruit_norm[actor_idx, own_argmax]
        r_self = jnp.where(
            ~done_actors,
            own_peak_mass * jnp.maximum(d_pre - d_post, 0.0),
            0.0,
        )
        reward = env_reward + self.r_self_coef * r_self

        # Swap own per-fruit vector across agent halves; reset to uniform on done.
        swapped_fruit_attn = swap_partner(per_fruit_norm, self.num_agents)
        uniform_fruit = jnp.ones((self.num_fruits,), dtype=jnp.float32) / float(self.num_fruits)
        new_partner_fruit_attn = jnp.where(
            done_actors[:, None], uniform_fruit[None], swapped_fruit_attn,
        )
        new_carry = (new_partner_fruit_attn, ~done_actors)

        extras = {
            "food_pos": food_pos_actors.astype(jnp.int32),
            "food_eaten": food_eaten_actors,
            "partner_prev_fruit_attn": prev_partner_fruit_attn,
            "partner_prev_valid": prev_partner_valid,
            "r_self": r_self,
        }
        return reward, new_carry, extras

    def aux_loss(self, attn_map_apply, traj_batch, config):
        """Per-fruit soft cross-entropy against the partner's per-fruit attention.

        Returns (coef, loss). Gradient on the agent's attention flows only
        through the N fruit cells, which keeps cuDNN happy at large NUM_ENVS.
        """
        del config
        if not self.aux_active:
            return self.aux_coef, jnp.float32(0.0)
        ex = traj_batch.extras
        attn_2d = as_spatial_attention(attn_map_apply)
        agent_per_fruit, _ = per_fruit_attn(
            attn_2d, ex["food_pos"], ex["food_eaten"],
            self.tile_size, self.feat_h, self.feat_w, self.img_h, self.img_w,
        )
        target_soft = jax.lax.stop_gradient(ex["partner_prev_fruit_attn"])
        log_probs = jnp.log(agent_per_fruit + 1e-8)
        nll_flat = -(target_soft * log_probs).sum(axis=-1).reshape(-1)
        # Partner confidence weight: flat partner ~1/N, confident ~peak mass.
        partner_peak = target_soft.max(axis=-1)
        aux_weight = jax.lax.stop_gradient(
            ex["partner_prev_valid"].reshape(-1).astype(jnp.float32)
            * partner_peak.reshape(-1)
        )
        aux_denom = jnp.maximum(aux_weight.sum(), 1e-8)
        aux_loss = (nll_flat * aux_weight).sum() / aux_denom
        return self.aux_coef, aux_loss

    def rollout_metrics(self, traj_batch, loss_info):
        return {
            "aux_partner_argmax_loss": loss_info.aux_loss.mean(),
            "r_self_mean": traj_batch.extras["r_self"].mean(),
        }

    def report(self, config, out, logger):
        from common.train_logging import (
            IMAGE_IPPO_SCALAR_KEYS,
            report_basic_training_outputs,
        )
        report_basic_training_outputs(
            config, out, logger,
            scalar_keys=list(IMAGE_IPPO_SCALAR_KEYS) + list(self.scalar_keys),
            print_prefix="ja_ippo:lbf",
        )

    def eval_outputs(self, algorithm_config, env, out, logger):
        """Eval-overlay video (seed 0) + in-memory cross-play (>=2 seeds)."""
        import os

        import hydra

        from agents.initialize_agents import initialize_ja_image_agent
        from marl.eval_lbf import _render_lbf_eval_frames

        try:
            from evaluation.vis_episodes import make_attention_video, run_episode_with_states

            rng = jax.random.PRNGKey(0)
            policy, _ = initialize_ja_image_agent(algorithm_config, env, rng)
            final_params = jax.tree.map(lambda x: x[0], out["final_params"])
            inner_env = env._env
            lbf_ctx = None
            if self.partner_feed_active:
                lbf_ctx = lbf_attention_ctx(algorithm_config, env)
            max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
            ep_states, attn_data, _a, _m = run_episode_with_states(
                jax.random.PRNGKey(42), inner_env, final_params, policy,
                final_params, policy, max_steps,
                collect_attention=True, lbf_ctx=lbf_ctx,
            )
            savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            video_dir = f"{savedir}/videos"
            os.makedirs(video_dir, exist_ok=True)
            frames = _render_lbf_eval_frames(inner_env, ep_states)
            from moviepy import ImageSequenceClip
            base_path = f"{video_dir}/eval_final.mp4"
            ImageSequenceClip(frames, fps=10).write_videofile(
                base_path, fps=10, codec="libx264", audio=False,
                bitrate="8000k", preset="slow",
            )
            logger.log_video("Eval/episode_video", base_path, commit=False)
            attn_base = f"{video_dir}/eval_attention.mp4"
            make_attention_video(frames, attn_data, filename=attn_base, fps=10)
            for suffix in ("agent0", "agent1", "combined"):
                p = f"{video_dir}/eval_attention_{suffix}.mp4"
                if os.path.exists(p):
                    logger.log_video(f"Eval/attention_{suffix}", p, commit=False)
        except Exception as e:
            print(f"[ja_ippo:lbf] WARN: eval video failed ({e}); continuing.", flush=True)
