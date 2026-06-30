"""Lee et al. (2021) JSD-intrinsic JA mechanism for the unified ja_ippo trainer.

Used by image-obs envs with no entity-level attention, partner feed, comm, or
aux loss — Overcooked and Hanabi. The only JA-specific signal is the intrinsic
reward r_JA = -JSD(attn_0, attn_1), scaled by beta ramping from 0 to
JA_BETA_MAX over JA_WARMUP_ENV_STEPS. With JA_BETA_MAX=0 (the default) this is
the JA network trained on plain PPO. Report + eval reuse the shared image-JA
helpers (report_ja_training_outputs, log_greedy_eval, log_eval_video,
run_xp_from_params).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.ja_utils import jsd_divergence


class JSDMechanism:
    """JSD-intrinsic JA hooks for image envs without entity attention/comm/aux."""

    scalar_keys = [
        ("jsd_mean", "JA/jsd"),
        ("ja_beta", "JA/beta"),
        ("rew_shaping_frac", "JA/rew_shaping_frac"),
        ("delivery_per_step", "Reward/delivery_per_step"),
        ("shaped_applied_per_step", "Reward/shaped_applied_per_step"),
    ]

    def __init__(self, config, env):
        self.name = config.get("ENV_NAME", "ja")
        self.num_agents = env.num_agents
        self.ja_beta_max = float(config.get("JA_BETA_MAX", 0.0))
        ja_warmup_env_steps = float(config.get("JA_WARMUP_ENV_STEPS", 0))
        env_steps_per_update = int(config["ROLLOUT_LENGTH"]) * int(config["NUM_ENVS"])
        self.env_steps_per_update = env_steps_per_update
        self.ja_warmup_updates = ja_warmup_env_steps / max(env_steps_per_update, 1)
        # Linear reward-shaping anneal (1 -> 0 over REW_SHAPING_HORIZON env steps),
        # matching JaxMARL's Overcooked PPO. 0 disables annealing (shaping then comes
        # from the wrapper's do_reward_shaping fold, as for overcooked-v1/hanabi).
        self.rew_shaping_horizon = float(config.get("REW_SHAPING_HORIZON", 0.0))
        # The pairwise attention JSD is only defined for 2 agents. Allow >2-agent
        # envs only as a no-method baseline (JA_BETA_MAX=0); fail fast otherwise so
        # a requested JSD reward is never silently dropped.
        if self.num_agents != 2 and self.ja_beta_max > 0.0:
            raise NotImplementedError(
                f"JSDMechanism's pairwise attention JSD is 2-agent only, but env "
                f"'{self.name}' has {self.num_agents} agents. Use JA_BETA_MAX=0 for the "
                "no-method baseline, or implement an N-agent JA mechanism."
            )

    def entity_feed_dim(self) -> int:
        return 0

    def init_carry(self, num_actors):
        return {}

    def augment_obs(self, obs_batch_2d, carry):
        return obs_batch_2d

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del env_state, new_env_state, action, done_actors
        ja_beta = jnp.minimum(
            self.ja_beta_max,
            self.ja_beta_max * update_steps / jnp.maximum(self.ja_warmup_updates, 1.0),
        )
        if self.num_agents == 2:
            num_envs = num_actors // self.num_agents
            a0 = attn_map[:, :num_envs, ...].squeeze(0)
            a1 = attn_map[:, num_envs:, ...].squeeze(0)
            jsd_env = jsd_divergence(a0, a1)                  # (num_envs,)
            jsd_actors = jnp.concatenate([jsd_env, jsd_env])  # (num_actors,)
        else:
            # >2 agents: no pairwise JSD signal (beta is forced 0 by __init__), so
            # emit zero — the JA network then trains as a plain-PPO baseline.
            jsd_actors = jnp.zeros((num_actors,))
        # Linear reward-shaping anneal (JaxMARL-style): the wrapper exposes the
        # per-agent shaped reward in info and returns the sparse reward, so we add
        # the decaying shaped term here. shaped_reward is (num_envs, num_agents);
        # transpose+flatten to agent-major (num_actors,) to match env_reward.
        shaping_frac = jnp.array(1.0)
        shaped_total = jnp.zeros((num_actors,))
        if self.rew_shaping_horizon > 0.0 and "shaped_reward" in info:
            shaped_actors = info["shaped_reward"].swapaxes(0, 1).reshape(-1)
            env_steps = update_steps * self.env_steps_per_update
            shaping_frac = jnp.clip(1.0 - env_steps / self.rew_shaping_horizon, 0.0, 1.0)
            shaped_total = shaping_frac * shaped_actors
        # Intrinsic reward = beta * (-JSD): reward attention agreement.
        reward = env_reward + shaped_total - ja_beta * jsd_actors
        extras = {
            "jsd": jsd_actors,
            "ja_beta": jnp.broadcast_to(ja_beta, (num_actors,)),
            "rew_shaping_frac": jnp.broadcast_to(shaping_frac, (num_actors,)),
            "delivery": env_reward,          # sparse env (delivery) reward, per step
            "shaped_applied": shaped_total,  # anneal_factor * shaped, per step
        }
        return reward, carry, extras

    def aux_loss(self, attn_map_apply, traj_batch, config):
        del attn_map_apply, traj_batch, config
        return 0.0, jnp.float32(0.0)

    def rollout_metrics(self, traj_batch, loss_info):
        del loss_info
        ex = traj_batch.extras
        return {
            "jsd_mean": ex["jsd"].mean(),
            "ja_beta": ex["ja_beta"].mean(),
            "rew_shaping_frac": ex["rew_shaping_frac"].mean(),
            "delivery_per_step": ex["delivery"].mean(),
            "shaped_applied_per_step": ex["shaped_applied"].mean(),
        }

    def report(self, config, out, logger):
        from common.train_logging import report_ja_training_outputs
        report_ja_training_outputs(config, out, logger)

    def eval_outputs(self, algorithm_config, env, out, logger):
        if env.num_agents != 2:
            # 2-agent log_greedy_eval/log_eval_video/pairwise-XP are agent_0/agent_1
            # hardcoded. For >2-agent envs: a video via the shared N-agent
            # path, plus inline ego/partner cross-play when multi-seed — both on the
            # (best-ckpt) params already in `out`. Each guarded so neither crashes
            # the run.
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
            return
        from marl.eval_logging import log_eval_video, log_greedy_eval
        log_greedy_eval(algorithm_config, env, out, logger)
        log_eval_video(algorithm_config, env, out, logger)

    def log_ckpt_video(self, algorithm_config, env, params, policy, tag, savedir, logger):
        """Per-checkpoint video. For overcooked-v2 render the god's-eye gameplay
        plus per-agent (Blues/Reds) and combined (jet) attention overlays; other
        envs fall back to the generic gameplay-only video.
        """
        import os

        env_name = algorithm_config["ENV_NAME"]
        # One-level unwrap (LogWrapper -> image wrapper). NOT get_inner_env, which
        # follows `.env` down to the raw symbolic OvercookedV2 (no get_avail_actions).
        inner_env = getattr(env, "_env", env)
        max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))

        if env_name != "overcooked-v2":
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
