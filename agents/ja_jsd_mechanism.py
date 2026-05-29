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
    ]

    def __init__(self, config, env):
        self.name = config.get("ENV_NAME", "ja")
        self.num_agents = env.num_agents
        self.ja_beta_max = float(config.get("JA_BETA_MAX", 0.0))
        ja_warmup_env_steps = float(config.get("JA_WARMUP_ENV_STEPS", 0))
        env_steps_per_update = int(config["ROLLOUT_LENGTH"]) * int(config["NUM_ENVS"])
        self.ja_warmup_updates = ja_warmup_env_steps / max(env_steps_per_update, 1)

    def entity_feed_dim(self) -> int:
        return 0

    def init_carry(self, num_actors):
        return {}

    def augment_obs(self, obs_batch_2d, carry):
        return obs_batch_2d

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del env_state, new_env_state, action, info, done_actors
        num_envs = num_actors // self.num_agents
        a0 = attn_map[:, :num_envs, ...].squeeze(0)
        a1 = attn_map[:, num_envs:, ...].squeeze(0)
        jsd_env = jsd_divergence(a0, a1)                  # (num_envs,)
        jsd_actors = jnp.concatenate([jsd_env, jsd_env])  # (num_actors,)
        ja_beta = jnp.minimum(
            self.ja_beta_max,
            self.ja_beta_max * update_steps / jnp.maximum(self.ja_warmup_updates, 1.0),
        )
        # Intrinsic reward = beta * (-JSD): reward attention agreement.
        reward = env_reward - ja_beta * jsd_actors
        extras = {
            "jsd": jsd_actors,
            "ja_beta": jnp.broadcast_to(ja_beta, (num_actors,)),
        }
        return reward, carry, extras

    def aux_loss(self, attn_map_apply, traj_batch, config):
        del attn_map_apply, traj_batch, config
        return 0.0, jnp.float32(0.0)

    def rollout_metrics(self, traj_batch, loss_info):
        del loss_info
        ex = traj_batch.extras
        return {"jsd_mean": ex["jsd"].mean(), "ja_beta": ex["ja_beta"].mean()}

    def report(self, config, out, logger):
        from common.train_logging import report_ja_training_outputs
        report_ja_training_outputs(config, out, logger)

    def eval_outputs(self, algorithm_config, env, out, logger):
        from marl.eval_logging import log_eval_video, log_greedy_eval
        log_greedy_eval(algorithm_config, env, out, logger)
        log_eval_video(algorithm_config, env, out, logger)
        num_seeds = jax.tree.leaves(out["final_params"])[0].shape[0]
        if num_seeds > 1:
            import hydra
            import wandb

            from agents.initialize_agents import initialize_ja_image_agent
            from evaluation.run_xp_seeds import run_xp_from_params
            policy, _ = initialize_ja_image_agent(algorithm_config, env, jax.random.PRNGKey(0))
            savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            run_xp_from_params(
                env, policy, out["final_params"], algorithm_config,
                savedir=savedir, task_name=algorithm_config.get("ENV_NAME"),
                wb_run=getattr(wandb, "run", None), greedy_eval=True, wb_prefix="XP",
            )
