"""No-MATE mechanism: the IPPO / OP / HE IPPO baselines on OvercookedV2.

The trainer always talks to a mechanism object, so the baselines need one too.
This is that object: it leaves the observation and the attention map alone and
adds no auxiliary loss, so the network trains on plain PPO.

The one thing it does do is add the shaped reward back in. The OvercookedV2
task sets `do_reward_shaping: false`, so the wrapper returns only the sparse
delivery reward and exposes the shaped term in `info` -- this mechanism is what
folds it into the learning reward, and without it OvercookedV2 self-play
collapses.

`REW_SHAPING_HORIZON` scales that term linearly from 1 to 0 across the horizon.
The released configs set it far beyond the training length (1e12 vs 3e7 steps),
so the weight is constant at ~1.0 in every reported run: annealing the shaping
away costs both SP and XP here.
"""
from __future__ import annotations

import jax.numpy as jnp


class BaselineMechanism:
    """Plain-PPO hooks plus the OvercookedV2 shaped-reward term."""

    scalar_keys = [
        ("rew_shaping_frac", "JA/rew_shaping_frac"),
        ("delivery_per_step", "Reward/delivery_per_step"),
        ("shaped_applied_per_step", "Reward/shaped_applied_per_step"),
    ]

    def __init__(self, config, env):
        self.name = config.get("ENV_NAME", "baseline")
        self.num_agents = env.num_agents
        self.env_steps_per_update = int(config["ROLLOUT_LENGTH"]) * int(config["NUM_ENVS"])
        # 0 disables the term entirely, in which case shaping would have to come
        # from the wrapper's own do_reward_shaping fold instead.
        self.rew_shaping_horizon = float(config.get("REW_SHAPING_HORIZON", 0.0))

    def entity_feed_dim(self) -> int:
        return 0

    def init_carry(self, num_actors):
        return {}

    def augment_obs(self, obs_batch_2d, carry):
        return obs_batch_2d

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del attn_map, env_state, new_env_state, action, done_actors
        # shaped_reward is (num_envs, num_agents); transpose+flatten to
        # agent-major (num_actors,) so it lines up with env_reward.
        # Constant at ~1.0 for the released horizon; see the module docstring.
        shaping_frac = jnp.array(1.0)
        shaped_total = jnp.zeros((num_actors,))
        if self.rew_shaping_horizon > 0.0 and "shaped_reward" in info:
            shaped_actors = info["shaped_reward"].swapaxes(0, 1).reshape(-1)
            env_steps = update_steps * self.env_steps_per_update
            shaping_frac = jnp.clip(1.0 - env_steps / self.rew_shaping_horizon, 0.0, 1.0)
            shaped_total = shaping_frac * shaped_actors

        extras = {
            "rew_shaping_frac": jnp.broadcast_to(shaping_frac, (num_actors,)),
            "delivery": env_reward,          # sparse env (delivery) reward, per step
            "shaped_applied": shaped_total,  # weight * shaped, per step
        }
        return env_reward + shaped_total, carry, extras

    def aux_loss(self, attn_map_apply, traj_batch, config):
        del attn_map_apply, traj_batch, config
        return 0.0, jnp.float32(0.0)

    def rollout_metrics(self, traj_batch, loss_info):
        del loss_info
        ex = traj_batch.extras
        return {
            "rew_shaping_frac": ex["rew_shaping_frac"].mean(),
            "delivery_per_step": ex["delivery"].mean(),
            "shaped_applied_per_step": ex["shaped_applied"].mean(),
        }

    def report(self, config, out, logger):
        from common.train_logging import report_ja_training_outputs
        report_ja_training_outputs(config, out, logger)

    def eval_outputs(self, algorithm_config, env, out, logger):
        from marl.eval_logging import log_eval_video, log_greedy_eval
        log_greedy_eval(algorithm_config, env, out, logger)
        log_eval_video(algorithm_config, env, out, logger)

    def log_ckpt_video(self, algorithm_config, env, params, policy, tag, savedir, logger):
        """Per-checkpoint gameplay video, with attention overlays for ocv2."""
        import os

        import jax

        env_name = algorithm_config["ENV_NAME"]
        # One-level unwrap (LogWrapper -> image wrapper). NOT get_inner_env, which
        # follows `.env` down to the raw symbolic OvercookedV2 (no get_avail_actions).
        inner_env = getattr(env, "_env", env)
        max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
        ckpt_attention = bool(algorithm_config.get("CKPT_VIDEO_ATTENTION", True))

        if env_name != "overcooked-v2" or not ckpt_attention:
            try:
                from common.eval_media import rollout_and_log_video
                rollout_and_log_video(
                    jax.random.PRNGKey(42), inner_env, env_name, params, policy,
                    max_steps, tag=tag, savedir=savedir, logger=logger,
                )
            except Exception as e:
                print(f"[ja_ippo:{self.name}] WARN: ckpt video failed ({e}); continuing.", flush=True)
            return

        try:
            from evaluation.vis_episodes import make_attention_video, run_episode_with_states
            from envs.render_registry import get_eval_frames
            from moviepy import ImageSequenceClip

            ep_states, attn_data, _a, _m = run_episode_with_states(
                jax.random.PRNGKey(42), inner_env, params, policy, params, policy,
                max_steps, collect_attention=True,
            )
            os.makedirs(savedir, exist_ok=True)
            frames = get_eval_frames(env_name, inner_env, ep_states)
            # ep_states has the initial state plus the post-done auto-reset state;
            # trim to the attention length so overlay and gameplay stay aligned.
            n_attn = len(attn_data.get("agent_0", []))
            frames = list(frames[:n_attn] if n_attn else frames)
            stem = f"{savedir}/{tag.replace('/', '_')}"
            ImageSequenceClip(frames, fps=10).write_videofile(
                f"{stem}.mp4", fps=10, codec="libx264", audio=False, preset="ultrafast",
            )
            logger.log_video(tag, f"{stem}.mp4", commit=False)
            make_attention_video(frames, attn_data, filename=f"{stem}_attn.mp4", fps=10)
            for suffix in ("agent0", "agent1", "combined"):
                p = f"{stem}_attn_{suffix}.mp4"
                if os.path.exists(p):
                    logger.log_video(f"{tag}_attention_{suffix}", p, commit=False)
        except Exception as e:
            print(f"[ja_ippo:{self.name}] WARN: ckpt attention video failed ({e}); continuing.", flush=True)
