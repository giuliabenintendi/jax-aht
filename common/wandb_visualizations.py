import os
import re
from datetime import datetime

import wandb
from omegaconf import OmegaConf

try:
    from hydra.core.hydra_config import HydraConfig
except ModuleNotFoundError:  # pragma: no cover - only for lightweight import checks
    HydraConfig = None


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
    # e.g. "overcooked-v2/demo_cook_simple" -> "demo_cook_simple"
    return task.split("/")[-1] if "/" in task else task


def _sanitize_name_part(value) -> str:
    """Make override fragments safe for W&B run names and checkpoint paths."""
    text = str(value).strip().strip("'\"")
    text = text.replace("/", "-")
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "-", text)
    return text.strip("-")


def _compact_name_part(value) -> str:
    """Normalise a name fragment for fuzzy duplicate checks."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _override_name_part(key: str, value: str) -> str:
    """Short display form for an algorithm override in the run name."""
    clean_value = _sanitize_name_part(value)
    aliases = {
        "ANNEAL_LR": "anneal",
        "CLIP_EPS": "clip",
        "ENT_COEF": "ent",
        "GAE_LAMBDA": "gae",
        "NORMALIZE_REWARDS": "normrew",
        "UPDATE_EPOCHS": "epochs",
        "NUM_MINIBATCHES": "mb",
        "MAX_GRAD_NORM": "gradclip",
    }
    display_key = aliases.get(key, key)
    return f"{display_key}{clean_value}"


def _algorithm_override_suffixes(label: str = "") -> list[str]:
    """Return CLI algorithm hyperparameter overrides to append to the run name.

    Hydra exposes task overrides exactly as typed on the command line. That lets
    names reflect user-requested deviations from the selected config defaults
    without hard-coding every default value here.
    """
    if HydraConfig is None:
        return []
    try:
        overrides = list(HydraConfig.get().overrides.task)
    except Exception:
        return []

    already_named = {"TOTAL_TIMESTEPS", "NUM_SEEDS"}
    ignored = {"ALG", "ENV_NAME", "ENV_KWARGS", "ROLLOUT_LENGTH"}
    suffixes: list[str] = []
    seen: set[str] = set()
    compact_label = _compact_name_part(label)
    for raw in overrides:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.lstrip("+")
        if not key.startswith("algorithm."):
            continue
        short_key = key.split(".")[-1]
        if short_key in already_named or short_key in ignored:
            continue
        part = _override_name_part(short_key, value)
        compact_part = _compact_name_part(part)
        compact_key = _compact_name_part(_override_name_part(short_key, ""))
        compact_value = _compact_name_part(value)
        # If the user put the same idea in `label` (e.g. `annealLR3e4`), do not
        # repeat it in the automatic override suffix.
        if compact_part in compact_label:
            continue
        if compact_key and compact_key in compact_label:
            continue
        if compact_value and short_key == "LR" and f"lr{compact_value}" in compact_label:
            continue
        if part and part not in seen:
            suffixes.append(part)
            seen.add(part)
    return suffixes


def _build_run_string(config: dict) -> str:
    """Build a concise, label-driven run name.

    Format: {layout}_{alg}_[{label}_]{timesteps}_[s{N}_]{DDMMYYYY}_[overrides...]
    Algorithm CLI overrides are appended so runs with changed hyperparameters
    stay identifiable from the W&B run list.
    """
    alg_config = config["algorithm"]
    layout = _get_layout_short(config)
    parts = [layout, str(alg_config["ALG"])]

    label = str(config.get("label", "default_label"))
    if label and label != "default_label":
        parts.append(label)
    compact_label = _compact_name_part(label)
    if "TOTAL_TIMESTEPS" in alg_config:
        timesteps = _format_timesteps(alg_config["TOTAL_TIMESTEPS"])
        if _compact_name_part(timesteps) not in compact_label:
            parts.append(timesteps)
    num_seeds = alg_config.get("NUM_SEEDS", 1)
    if num_seeds > 1:
        seed_token = f"s{num_seeds}"
        compact_seed = _compact_name_part(seed_token)
        compact_seed_alt = _compact_name_part(f"{num_seeds}s")
        if compact_seed not in compact_label and compact_seed_alt not in compact_label:
            parts.append(seed_token)
    parts.append(datetime.now().strftime("%d%m%Y"))
    parts.extend(_algorithm_override_suffixes(label))
    return "_".join(parts)


def _build_tags(config) -> list[str]:
    """Build wandb tags using key/value pattern for faceted filtering.

    Categories that always appear (alg/, task/, seed/, date/, ent/, beta/, comm/, op/)
    let you slice the run table predictably; on-only flags (ja_card/on, feed_attn/on,
    qplstm/on, follow/{x}, match/{x}, ja_card_jsd/{x}) are added only when active.
    """
    alg_config = config["algorithm"]
    env_kwargs = alg_config.get("ENV_KWARGS", {})
    env_name = str(alg_config.get("ENV_NAME", ""))
    layout = _get_layout_short(config)
    date = datetime.now().strftime("%d%m%Y")

    tags = [
        f"alg/{alg_config['ALG']}",
        f"task/{env_name}",
        f"seed/{alg_config.get('TRAIN_SEED', 0)}",
        f"date/{date}",
    ]
    # Layout tag only when it adds info beyond the env (e.g. overcooked-v2/demo_cook_simple).
    if layout and layout != env_name:
        tags.append(f"layout/{layout}")
    if "ENT_COEF" in alg_config:
        tags.append(f"ent/{alg_config['ENT_COEF']}")

    op_on = bool(
        env_kwargs.get("other_play_position_shuffle")
        or env_kwargs.get("other_play_recolouring")
    )
    tags.append("op/on" if op_on else "op/off")

    if alg_config.get("JA_CARD_ATTN", False):
        tags.append("ja_card/on")
    if alg_config.get("FEED_OTHER_ATTN", False):
        tags.append("feed_attn/on")
    if alg_config.get("QUERY_PARTNER_LSTM", False):
        tags.append("qplstm/on")


    label = str(config.get("label", "default_label"))
    if label.lower().startswith("sweep"):
        tags.append("sweep")
    return tags


def _build_group(config) -> str:
    """Build group string for wandb seed aggregation.

    Runs in the same group get mean±std plots automatically.
    """
    alg_config = config["algorithm"]
    parts = [str(config["TASK_NAME"]), str(alg_config["ALG"])]
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

        # Keep wandb's auto-generated name (e.g. "dainty-cherry-42") as a unique prefix,
        # then append the descriptive run_string. This preserves wandb's collision-free
        # identifier while keeping the searchable config tag in the run name.
        auto_name = self.run.name or ""
        composed = f"{auto_name}_{run_string}" if auto_name else run_string
        # wandb run names are file-path components; keep under a sane length.
        if len(composed) > 250:
            composed = composed[:250]
        self.run.name = composed

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
        wandb.define_metric("env_step")
        wandb.define_metric("Train/*", step_metric="train_step")
        wandb.define_metric("Losses/*", step_metric="train_step")
        wandb.define_metric("Eval/*", step_metric="train_step")
        wandb.define_metric("Returns/*", step_metric="train_step")
        wandb.define_metric("HeldoutEval/*", step_metric="iter")
        # Live per-chunk metrics pushed during training; x-axis is env_step so per-seed
        # curves align across the same env-step trajectory rather than wandb's auto _step.
        wandb.define_metric("LiveTrain/*", step_metric="env_step")
    
    def log_artifact(self, name, path, type_name):
        artifact = wandb.Artifact(name, type=type_name)
        # check if path is a directory or a file
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        self.run.log_artifact(artifact)
    
    def log_video(self, tag, path, commit=True, caption=None):
        # caption lets callers stamp the checkpoint env_step (or seed pairing for XP
        # videos) onto the W&B media tile so it's obvious which weights produced it.
        kwargs = {"caption": caption} if caption else {}
        wandb.log({tag: wandb.Video(path, **kwargs)}, commit=commit)
    
    def close(self):
        wandb.finish()
