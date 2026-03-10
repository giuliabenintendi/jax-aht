import os
import wandb
from omegaconf import OmegaConf


def _format_timesteps(n: float) -> str:
    """Format timestep count as human-readable string, e.g. 2e6 → '2M'."""
    n = float(n)
    if n >= 1e6 and n % 1e6 == 0:
        return f"{int(n // 1e6)}M"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3 and n % 1e3 == 0:
        return f"{int(n // 1e3)}K"
    return str(int(n))


def _build_run_string(alg_config: dict) -> str:
    """Build the descriptive part of the run name from algorithm config."""
    alg = alg_config["ALG"]
    parts = [alg]
    if "TOTAL_TIMESTEPS" in alg_config:
        parts.append(_format_timesteps(alg_config["TOTAL_TIMESTEPS"]))
    if "JA_BETA_MAX" in alg_config:
        parts.append(f"BETA{alg_config['JA_BETA_MAX']}")
    return "_".join(parts)


def _build_tags(config) -> list[str]:
    """Build tags list from config for wandb filtering."""
    alg_config = config["algorithm"]
    tags = [
        str(alg_config["ALG"]),
        str(config["TASK_NAME"]),
        f"seed={alg_config['TRAIN_SEED']}",
        f"num_envs={alg_config['NUM_ENVS']}",
    ]
    label = config.get("label", "default_label")
    if label != "default_label":
        tags.append(str(label))
    return tags


def _build_group(config) -> str:
    """Build group string: TASK_NAME/ALG."""
    return f"{config['TASK_NAME']}/{config['algorithm']['ALG']}"


class Logger:
    """
    Class to initialize logger object for writing experiment results to wandb.
    """
    def __init__(self, config):
        self.verbose = config["logger"].get("verbose", False)
        tags = _build_tags(config)
        group_string = _build_group(config)
        run_string = _build_run_string(config["algorithm"])

        if len(run_string) > 250:
            raise ValueError("Run name exceeds file name length limit.")

        self.run = wandb.init(
            project=config["logger"]["project"],
            entity=config["logger"]["entity"],
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            tags=tags,
            notes=config["logger"].get("notes", None),
            group=group_string,
            mode=config["logger"].get("mode", None),
            save_code=True,
            reinit=True,
        )

        if self.run.sweep_id is not None:
            self.run.name = self.run.sweep_id + "___" + run_string
        else:
            self.run.name = str(self.run.name) + "___" + run_string

        self.define_metrics()

    def log(self, data, step=None, commit=False):
        wandb.log(data, step=step, commit=commit)

    def log_item(self, tag, val, step=None, commit=True, **kwargs):
        self.log({tag: val, **kwargs}, step=step, commit=commit)
        if self.verbose:
            print(f"{tag}: {val}")

    def commit(self):
        self.log({}, commit=True)

    def log_xp_matrix(self, tag, mat, step=None, columns=None, rows=None, commit=True, **kwargs):
        if rows is None:
            rows = [str(i) for i in range(mat.shape[0])]
        if columns is None:
            columns = [str(i) for i in range(mat.shape[1])]
        tab = wandb.Table(
                columns=columns,
                data=mat,
                rows=rows
                )
        wandb.log({tag: tab, **kwargs}, step=step, commit=commit)

    def define_metrics(self):
        wandb.define_metric("train_step")
        wandb.define_metric("checkpoint")
        wandb.define_metric("Train/*", step_metric="train_step")
        wandb.define_metric("Losses/*", step_metric="train_step")
        wandb.define_metric("Eval/*", step_metric="train_step")
        wandb.define_metric("Returns/*", step_metric="train_step")
        wandb.define_metric("HeldoutEval/*", step_metric="iter")
    
    def log_artifact(self, name, path, type_name):
        artifact = wandb.Artifact(name, type=type_name)
        # check if path is a directory or a file
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self.run.log_artifact(artifact)
    
    def log_video(self, tag, path, commit=True):
        wandb.log({tag: wandb.Video(path)}, commit=commit)
    
    def close(self):
        wandb.finish()
