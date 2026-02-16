'''Main entry point for running teammate generation algorithms.'''
import os
import logging

import jax
import hydra
from omegaconf import OmegaConf

from evaluation.heldout_eval import run_heldout_evaluation, log_heldout_metrics
from common.plot_utils import get_metric_names
from common.wandb_visualizations import Logger
from teammate_generation.BRDiv import run_brdiv
from teammate_generation.LBRDiv import run_lbrdiv
from teammate_generation.CoMeDi import run_comedi
from teammate_generation.fcp import run_fcp
from teammate_generation.train_ego import train_ego_agent

log = logging.getLogger(__name__)


def log_videos_to_wandb(cfg, wandb_logger, env, ego_policy, ego_params,
                        partner_policy, single_partner_params, num_episodes=1):
    """Render a few episodes and upload MP4s to WandB.

    Args:
        single_partner_params: params for a single partner agent (no batch dims).
    """
    from evaluation.vis_episodes import save_video
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    video_dir = os.path.join(savedir, "videos")

    env_name = cfg["task"]["ENV_NAME"]
    max_steps = cfg["task"]["ROLLOUT_LENGTH"]

    # Use the first ego seed's final params
    single_ego_params = jax.tree.map(lambda x: x[0], ego_params)

    try:
        video_path = save_video(
            env, env_name,
            agent_0_param=single_ego_params, agent_0_policy=ego_policy,
            agent_1_param=single_partner_params, agent_1_policy=partner_policy,
            max_episode_steps=max_steps, num_eps=num_episodes,
            savevideo=True, save_dir=video_dir, save_name="ego_vs_partner")
        wandb_logger.log_video("Videos/ego_vs_partner", video_path)
        log.info(f"Video logged to WandB: {video_path}")
    except Exception as e:
        log.warning(f"Video rendering failed (non-fatal): {e}")


@hydra.main(version_base=None, config_path="configs", config_name="base_config_teammate")
def run_training(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    wandb_logger = Logger(cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # train partner population
    if cfg["algorithm"]["ALG"] == "brdiv":
        partner_params, partner_population = run_brdiv(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "fcp":
        partner_params, partner_population = run_fcp(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "lbrdiv":
        partner_params, partner_population = run_lbrdiv(cfg, wandb_logger)
    elif cfg["algorithm"]["ALG"] == "comedi":
        partner_params, partner_population = run_comedi(cfg, wandb_logger)
    else:
        raise NotImplementedError("Selected method not implemented.")

    metric_names = get_metric_names(cfg["task"]["ENV_NAME"])
    if cfg["train_ego"]:
        ego_params, ego_policy, init_ego_params = train_ego_agent(cfg, wandb_logger, partner_params, partner_population)

    if cfg["run_heldout_eval"]:
        eval_metrics, ego_names, heldout_names = run_heldout_evaluation(cfg, ego_policy, ego_params, init_ego_params, ego_as_2d=False)
        log_heldout_metrics(cfg, wandb_logger, eval_metrics, ego_names, heldout_names, metric_names, ego_as_2d=False)

    # Render and log videos to WandB
    if cfg["train_ego"] and cfg["task"]["ENV_NAME"] == "overcooked-v1":
        from envs import make_env
        raw_env = make_env(cfg["algorithm"]["ENV_NAME"], cfg["algorithm"]["ENV_KWARGS"])
        # partner_params shape: (num_seeds, pop_size, ...) — pick first seed, first partner
        single_partner = jax.tree.map(lambda x: x[0, 0], partner_params)
        log_videos_to_wandb(cfg, wandb_logger, raw_env, ego_policy, ego_params,
                            partner_population.policy_cls, single_partner,
                            num_episodes=2)

    wandb_logger.close()

if __name__ == '__main__':
    run_training()
