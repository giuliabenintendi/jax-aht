import os
from datetime import datetime

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


def _get_layout_short(config) -> str:
    """Extract short layout name from task config."""
    task = str(config.get("TASK_NAME", ""))
    # e.g. "overcooked-v1/cramped_room" → "cramped_room"
    return task.split("/")[-1] if "/" in task else task


def _build_run_string(config: dict) -> str:
    """Build a concise, searchable run name.

    Format: layout_alg_timesteps_bX.X_sN_DDMMYYYY
    e.g. cramped_room_ja_ippo_5M_b0.5_s42_15032026
    """
    alg_config = config["algorithm"]
    alg = alg_config["ALG"]
    layout = _get_layout_short(config)
    date = datetime.now().strftime("%d%m%Y")

    parts = [layout, alg]
    if "TOTAL_TIMESTEPS" in alg_config:
        parts.append(_format_timesteps(alg_config["TOTAL_TIMESTEPS"]))
    if "JA_BETA_MAX" in alg_config:
        parts.append(f"b{alg_config['JA_BETA_MAX']}")
    if alg_config.get("USE_DUAL_CRITIC", False):
        jsd_gae = "jsdgae" if alg_config.get("DUAL_CRITIC_ACTOR_JA", False) else "nojsdgae"
        parts.append(f"dual_{jsd_gae}")
    ent_coef = alg_config.get("ENT_COEF", 0.01)
    if ent_coef != 0.01:
        parts.append(f"ent{ent_coef}")
    env_kwargs = alg_config.get("ENV_KWARGS", {})
    max_cards = env_kwargs.get("max_cards", None)
    if max_cards is not None:
        # Dynamic env with configurable card count — include num_colors
        from envs.card_game.rendering_dynamic import NUM_COLORS
        parts.append(f"{NUM_COLORS}c")
    if alg_config.get("COMMUNICATION", False):
        parts.append("comm")
    if alg_config.get("FEED_OTHER_ATTN", False):
        parts.append("feed_attn")
    if alg_config.get("FILTER_ATTN_TOP1", False):
        parts.append("top1")
    if env_kwargs.get("shuffle") is False:
        parts.append("no_shuffle")
    fp = env_kwargs.get("fixed_partner_pos", -1)
    if fp >= 0:
        parts.append(f"fixed_partner{fp}")
    num_seeds = alg_config.get("NUM_SEEDS", 1)
    if num_seeds > 1:
        parts.append(f"s{num_seeds}")
    parts.append(date)
    return "_".join(parts)


def _build_tags(config) -> list[str]:
    """Build tags list from config for wandb filtering."""
    alg_config = config["algorithm"]
    layout = _get_layout_short(config)
    date = datetime.now().strftime("%d%m%Y")
    tags = [
        str(alg_config["ALG"]),
        layout,
        f"seed={alg_config.get('TRAIN_SEED', 0)}",
        f"envs={alg_config['NUM_ENVS']}",
        date,
    ]
    if "JA_BETA_MAX" in alg_config:
        tags.append(f"beta={alg_config['JA_BETA_MAX']}")
    if "TOTAL_TIMESTEPS" in alg_config:
        tags.append(_format_timesteps(alg_config["TOTAL_TIMESTEPS"]))
    ent_coef = alg_config.get("ENT_COEF", 0.01)
    tags.append(f"ent={ent_coef}")
    if alg_config.get("USE_DUAL_CRITIC", False):
        tags.append("dual_critic")
        if alg_config.get("DUAL_CRITIC_ACTOR_JA", False):
            tags.append("jsdgae_on")
        else:
            tags.append("jsdgae_off")
    if alg_config.get("FEED_OTHER_ATTN", False):
        tags.append("feed_attn")
    else:
        tags.append("no_feed_attn")
    if alg_config.get("FILTER_ATTN_TOP1", False):
        tags.append("top1")
    env_kwargs = alg_config.get("ENV_KWARGS", {})
    if env_kwargs.get("max_cards") is not None:
        from envs.card_game.rendering_dynamic import NUM_COLORS
        tags.append(f"{NUM_COLORS}cards")
    if env_kwargs.get("shuffle") is False:
        tags.append("no_shuffle")
    label = config.get("label", "default_label")
    if label != "default_label":
        tags.append(str(label))
        # Add sweep tag if label starts with "sweep"
        if str(label).lower().startswith("sweep"):
            tags.append("SWEEP")
    return tags


def _build_group(config) -> str:
    """Build group string for wandb seed aggregation.

    Runs in the same group get mean±std plots automatically.
    """
    alg_config = config["algorithm"]
    parts = [str(config["TASK_NAME"]), str(alg_config["ALG"])]
    if "JA_BETA_MAX" in alg_config:
        parts.append(f"b{alg_config['JA_BETA_MAX']}")
    label = config.get("label", "default_label")
    if label != "default_label":
        parts.append(str(label))
    return "/".join(parts)


class Logger:
    """
    Class to initialize logger object for writing experiment results to wandb.
    """
    def __init__(self, config):
        self.verbose = config["logger"].get("verbose", False)
        tags = _build_tags(config)
        group_string = _build_group(config)
        run_string = _build_run_string(config)

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

        self.run.name = run_string

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
