'''Main entry point for running MARL algorithms.'''
import hydra
from omegaconf import OmegaConf

from common.wandb_visualizations import Logger
from marl.ippo import run_ippo
from marl.ja_ippo import run_ja_ippo
from marl.ja_ippo_no_share import run_ja_ippo_no_share
from marl.image_ippo import run_image_ippo
from marl.ja_dual_ippo import run_ja_dual_ippo
from marl.gaze_ippo import run_gaze_ippo


@hydra.main(version_base=None, config_path="configs", config_name="base_config_marl")
def main(config):
    print(OmegaConf.to_yaml(config, resolve=True))
    wandb_logger = Logger(config)

    if config.algorithm["ALG"] == "ippo":
        run_ippo(config, wandb_logger)
    elif config.algorithm["ALG"] == "ja_ippo":
        if config.algorithm.get("USE_DUAL_CRITIC", False):
            run_ja_dual_ippo(config, wandb_logger)
        else:
            run_ja_ippo(config, wandb_logger)
    elif config.algorithm["ALG"] == "ja_ippo_no_share":
        run_ja_ippo_no_share(config, wandb_logger)
    elif config.algorithm["ALG"] == "image_ippo":
        run_image_ippo(config, wandb_logger)
    elif config.algorithm["ALG"] == "gaze_ippo":
        run_gaze_ippo(config, wandb_logger)
    else:
        raise NotImplementedError(f"Algorithm {config['ALG']} not implemented.")
        
    wandb_logger.close()

if __name__ == "__main__":
    main()
