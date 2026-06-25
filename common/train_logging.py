"""Shared training-output logging helpers.

This module keeps metric aggregation, CSV/plot export, and live W&B logging
out of trainer files so the trainers can stay focused on rollout/update logic.
"""

from __future__ import annotations

import csv
import os
import shutil

import hydra
import numpy as np

from common.plot_utils import get_metric_names, get_stats, plot_seed_aggregate
from common.save_load_utils import save_train_run


JA_LIVE_SCALAR_KEYS = [
    ("ja_beta", "JA/beta"),
    ("comm_scale", "Comm/scale"),
    ("jsd_mean", "JA/jsd"),
    ("card_jsd_mean", "JA/card_jsd"),
    ("ja_attn_shaping_mean", "JA/total_shaping"),
    ("ja_attn_self_mean", "JA/action_matches_own_attn"),
    ("ja_gaze_pick_mean", "JA/action_matches_prev_partner_attn"),
    ("aux_partner_argmax_loss", "JA/aux_partner_argmax_nll"),
    ("r_self_mean", "JA/r_self"),
    ("loss_total", "Loss/total"),
    ("loss_value", "Loss/value"),
    ("loss_policy", "Loss/policy"),
    ("entropy", "Loss/entropy"),
    ("grad_norm", "Loss/grad_norm"),
    ("approx_kl", "Loss/approx_kl"),
    ("clip_frac", "Loss/clip_frac"),
    ("explained_var", "Loss/explained_var"),
    ("raw_env_reward_mean", "Reward/env_raw"),
    ("comm_reward_mean", "Reward/comm"),
]


JA_SCALAR_KEYS = [
    ("ja_beta", "JA/beta"),
    ("comm_scale", "Comm/scale"),
    ("jsd_mean", "JA/jsd"),
    ("card_jsd_mean", "JA/card_jsd"),
    ("ja_attn_shaping_mean", "JA/total_shaping"),
    ("ja_attn_self_mean", "JA/action_matches_own_attn"),
    ("ja_gaze_pick_mean", "JA/action_matches_prev_partner_attn"),
    ("aux_partner_argmax_loss", "JA/aux_partner_argmax_nll"),
    ("raw_env_reward_mean", "Reward/env_raw"),
    ("combined_reward_mean", "Reward/combined_raw"),
    ("comm_reward_mean", "Reward/comm"),
    ("comm_match_bonus_mean", "Reward/comm_match_bonus"),
    ("comm_stability_bonus_mean", "Reward/comm_stability_bonus"),
    ("comm_follow_bonus_agent0_mean", "Reward/comm_follow_bonus_agent0"),
    ("comm_follow_bonus_agent1_mean", "Reward/comm_follow_bonus_agent1"),
    ("loss_total", "Loss/total"),
    ("loss_value", "Loss/value"),
    ("loss_policy", "Loss/policy"),
    ("entropy", "Loss/entropy"),
    ("grad_norm", "Loss/grad_norm"),
    ("approx_kl", "Loss/approx_kl"),
    ("approx_kl_all", "Loss/approx_kl_all"),
    ("approx_kl_max", "Loss/approx_kl_max"),
    ("clip_frac", "Loss/clip_frac"),
    ("ratio_mean", "Loss/ratio_mean"),
    ("ratio_std", "Loss/ratio_std"),
    ("explained_var", "Loss/explained_var"),
    ("advantage_std", "Loss/advantage_std"),
    ("value_mean", "Value/mean"),
]


IMAGE_IPPO_SCALAR_KEYS = [
    ("loss_total", "Losses"),
    ("loss_value", "Losses"),
    ("loss_policy", "Losses"),
    ("entropy", "Losses"),
    ("grad_norm", "Losses"),
    ("approx_kl", "Losses"),
    ("clip_frac", "Losses"),
    ("value_mean", "Values"),
]


# Hanabi-specific per-step metrics, pushed via log_live_chunk_metrics when
# the wrapper has populated them in the info dict. Each is averaged across
# the chunk: `score` and `turn` average the running game state across all
# steps; `action_*` average to fractions of steps spent in each action class.
HANABI_LIVE_SCALAR_KEYS = [
    ("score", "Hanabi/score"),
    ("lives_remaining", "Hanabi/lives_remaining"),
    ("info_tokens_remaining", "Hanabi/info_tokens_remaining"),
    ("turn", "Hanabi/turn"),
    ("bombed", "Hanabi/bombed_frac"),
    ("action_discard", "Hanabi/action_discard"),
    ("action_play", "Hanabi/action_play"),
    ("action_hint_colour", "Hanabi/action_hint_colour"),
    ("action_hint_rank", "Hanabi/action_hint_rank"),
    ("action_noop", "Hanabi/action_noop"),
]


def log_live_chunk_metrics(chunk_metrics, env_step, seed_idx, logger, mech_scalar_keys=None):
    """Push chunk-aggregated training metrics to W&B during training."""
    if logger is None or getattr(logger, "run", None) is None:
        return

    returned = np.asarray(chunk_metrics["returned_episode"])
    returns = np.asarray(chunk_metrics["returned_episode_returns"])
    n_ep = float(returned.sum())
    if n_ep > 0:
        mean = float((returns * returned).sum() / n_ep)
        sq = float(((returns - mean) ** 2 * returned).sum() / n_ep)
        std = float(np.sqrt(max(sq, 0.0)))
    else:
        mean = float("nan")
        std = float("nan")

    data = {
        f"LiveTrain/seed_{seed_idx}/return_mean": mean,
        f"LiveTrain/seed_{seed_idx}/return_std": std,
        f"LiveTrain/seed_{seed_idx}/n_episodes": int(n_ep),
        "env_step": int(env_step),
    }

    for key, name in JA_LIVE_SCALAR_KEYS:
        if key in chunk_metrics:
            data[f"LiveTrain/seed_{seed_idx}/{name}"] = float(
                np.asarray(chunk_metrics[key]).mean()
            )

    # Hanabi-specific per-step metrics (only present when the Hanabi wrapper
    # populated them — silently skipped otherwise so this also runs cleanly
    # for card-game / lbf / overcooked).
    for key, name in HANABI_LIVE_SCALAR_KEYS:
        if key in chunk_metrics:
            data[f"LiveTrain/seed_{seed_idx}/{name}"] = float(
                np.asarray(chunk_metrics[key]).mean()
            )

    # Mechanism-declared scalar metrics (e.g. the future-occupancy aux loss and
    # attention/occupancy overlaps), logged live under their group so they appear
    # alongside the standard losses during training rather than only at end-of-run.
    for key, group in (mech_scalar_keys or []):
        if key in chunk_metrics:
            data[f"LiveTrain/seed_{seed_idx}/{group}/{key}"] = float(
                np.asarray(chunk_metrics[key]).mean()
            )

    logger.log(data, commit=True)


def _compute_episode_stats_mean(train_stats):
    """Aggregate per-seed episode metrics into mean/std curves."""
    episode_stats_mean = {}
    for key, values in train_stats.items():
        values_arr = np.asarray(values)
        seed_means = values_arr[:, :, 0]
        episode_stats_mean[key] = np.stack(
            [seed_means.mean(axis=0), seed_means.std(axis=0)],
            axis=-1,
        )
    return episode_stats_mean


def _compute_scalar_mean_std(train_metrics, scalar_keys):
    """Aggregate scalar metrics into mean/std curves across seeds."""
    scalar_mean = {}
    scalar_std = {}
    for key, _ in scalar_keys:
        if key in train_metrics:
            values = np.asarray(train_metrics[key])
            scalar_mean[key] = np.mean(values, axis=0)
            scalar_std[key] = np.std(values, axis=0)
    return scalar_mean, scalar_std


def _log_train_curve_plots(savedir, train_stats, rollout_length, num_envs, logger):
    plot_seed_aggregate(
        train_stats,
        num_rollout_steps=rollout_length,
        num_envs=num_envs,
        savedir=savedir,
        savename="train_curve",
    )

    import wandb

    for name in train_stats:
        png_path = os.path.join(savedir, f"train_curve_{name}.png")
        if os.path.exists(png_path):
            logger.log_item(
                f"Plots/train_curve_{name}",
                wandb.Image(png_path),
                commit=False,
            )


def _export_train_stats_csv(
    csv_path,
    metric_names,
    train_stats,
    scalar_mean,
    scalar_std,
    num_updates,
    rollout_length,
    num_envs,
    env_name,
):
    csv_header = ["update", "timestep"]
    for name in metric_names:
        csv_header.extend([f"{name}_mean", f"{name}_std"])
    if env_name == "overcooked-v1" and "base_return" in metric_names:
        csv_header.append("soups_delivered")
    for key, _ in JA_SCALAR_KEYS:
        if key in scalar_mean:
            csv_header.extend([f"{key}_mean", f"{key}_std"])

    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_header)
        for step in range(num_updates):
            row = [step, (step + 1) * rollout_length * num_envs]
            for name in metric_names:
                stat_data = np.asarray(train_stats[name])
                seed_means = stat_data[:, step, 0]
                row.extend([float(seed_means.mean()), float(seed_means.std())])
            if env_name == "overcooked-v1" and "base_return" in metric_names:
                base_data = np.asarray(train_stats["base_return"])
                row.append(float(base_data[:, step, 0].mean()) / 20.0)
            for key, _ in JA_SCALAR_KEYS:
                if key in scalar_mean:
                    row.extend(
                        [float(scalar_mean[key][step]), float(scalar_std[key][step])]
                    )
            writer.writerow(row)

    return csv_header


def report_ja_training_outputs(config, out, logger):
    """Export plots/CSV and log aggregated JA-IPPO training outputs."""
    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    algorithm_config = dict(config.algorithm)
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    rollout_length = int(algorithm_config["ROLLOUT_LENGTH"])
    num_envs = int(algorithm_config["NUM_ENVS"])

    _log_train_curve_plots(savedir, train_stats, rollout_length, num_envs, logger)

    num_seeds = train_metrics["returned_episode"].shape[0]
    num_updates = train_metrics["returned_episode"].shape[1]
    episode_stats_mean = _compute_episode_stats_mean(train_stats)
    scalar_mean, scalar_std = _compute_scalar_mean_std(train_metrics, JA_SCALAR_KEYS)

    csv_path = os.path.join(savedir, "train_stats.csv")
    csv_header = _export_train_stats_csv(
        csv_path=csv_path,
        metric_names=metric_names,
        train_stats=train_stats,
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
        num_updates=num_updates,
        rollout_length=rollout_length,
        num_envs=num_envs,
        env_name=config.task["ENV_NAME"],
    )

    print(
        f"[report_ja_training_outputs] CSV: {csv_path} "
        f"({num_updates} updates, {num_seeds} seeds, {len(csv_header)} cols)"
    )

    import wandb

    wandb.save(csv_path, base_path=savedir)

    print_interval = max(1, num_updates // 20)
    for step in range(num_updates):
        env_steps = (step + 1) * rollout_length * num_envs
        # env_step shadows wandb's internal _step so post-hoc training curves
        # align with the same x-axis as the live (per-chunk) metrics.
        step_data = {"train_step": step, "env_step": env_steps}

        for stat_name, stat_data in episode_stats_mean.items():
            step_data[f"Train/{stat_name}_mean"] = float(stat_data[step, 0])
            if num_seeds > 1:
                step_data[f"Train/{stat_name}_std"] = float(stat_data[step, 1])
        if "base_return" in episode_stats_mean and config.task["ENV_NAME"] == "overcooked-v1":
            step_data["Train/soups_delivered"] = float(
                episode_stats_mean["base_return"][step, 0] / 20.0
            )

        for key, wandb_name in JA_SCALAR_KEYS:
            if key in scalar_mean:
                step_data[f"{wandb_name}/mean"] = float(scalar_mean[key][step])
                if num_seeds > 1:
                    step_data[f"{wandb_name}/std"] = float(scalar_std[key][step])

        for stat_name in train_stats:
            stat_data = np.asarray(train_stats[stat_name])
            for seed_idx in range(num_seeds):
                step_data[f"Seeds/{stat_name}/seed_{seed_idx}"] = float(
                    stat_data[seed_idx, step, 0]
                )

        logger.log(step_data, commit=True)

        if step % print_interval == 0 or step == num_updates - 1:
            pct = (step + 1) / num_updates * 100
            ret_str = "  ".join(
                f"{stat_name}={stat_data[step, 0]:.2f}"
                for stat_name, stat_data in episode_stats_mean.items()
            )
            jsd = float(scalar_mean.get("jsd_mean", np.zeros(num_updates))[step])
            beta = float(scalar_mean.get("ja_beta", np.zeros(num_updates))[step])
            loss = float(scalar_mean.get("loss_total", np.zeros(num_updates))[step])
            grad = float(scalar_mean.get("grad_norm", np.zeros(num_updates))[step])
            extra = ""
            if "base_return" in episode_stats_mean and config.task["ENV_NAME"] == "overcooked-v1":
                soups = episode_stats_mean["base_return"][step, 0] / 20.0
                extra = f"  soups={soups:.1f}"
            print(
                f"[{pct:5.1f}%] step={step}/{num_updates}  env_steps={env_steps}  "
                f"{ret_str}{extra}  jsd={jsd:.4f}  beta={beta:.4f}  "
                f"loss={loss:.4f}  grad={grad:.3f}"
            )

    logger.commit()

    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(
            name="saved_train_run",
            path=out_savepath,
            type_name="train_run",
        )
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)


def report_basic_training_outputs(
    config,
    out,
    logger,
    scalar_keys,
    print_prefix="train",
):
    """Log mean training curves and save the final train artifact.

    This is the lighter-weight variant used by trainers that do not yet export
    CSVs or seed-aggregate plot files.
    """
    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)
    train_stats = {key: np.mean(np.asarray(values), axis=0) for key, values in train_stats.items()}

    scalar_data = {}
    for key, _ in scalar_keys:
        if key in train_metrics:
            scalar_data[key] = np.mean(np.asarray(train_metrics[key]), axis=0)

    num_updates = train_metrics["returned_episode"].shape[1]
    print_interval = max(1, num_updates // 20)
    rollout_length = int(config.algorithm["ROLLOUT_LENGTH"])
    num_envs = int(config.algorithm["NUM_ENVS"])

    for step in range(num_updates):
        env_steps = (step + 1) * rollout_length * num_envs
        # env_step shadows wandb's internal _step so post-hoc training curves
        # align with the same x-axis as the live (per-chunk) metrics.
        commit_payload = {"train_step": step, "env_step": env_steps}

        for stat_name, stat_data in train_stats.items():
            commit_payload[f"Train/{stat_name}"] = float(stat_data[step, 0])
        if "base_return" in train_stats and config.task["ENV_NAME"] == "overcooked-v1":
            soups = train_stats["base_return"][step, 0] / 20.0
            commit_payload["Train/soups_delivered"] = float(soups)

        for key, prefix in scalar_keys:
            if key in scalar_data:
                commit_payload[f"{prefix}/{key}"] = float(scalar_data[key][step])

        # NOTE: do NOT pass step=step here. Live logging already advanced
        # wandb's internal _step past `num_updates`, so any explicit step<_step
        # is silently dropped — that's how the post-hoc curves collapsed to a
        # single point. Let wandb auto-increment instead.
        logger.log(commit_payload, commit=True)

        if step % print_interval == 0 or step == num_updates - 1:
            pct = (step + 1) / num_updates * 100
            ret_str = "  ".join(
                f"{stat_name}={stat_data[step, 0]:.2f}"
                for stat_name, stat_data in train_stats.items()
            )
            loss = float(scalar_data.get("loss_total", np.zeros(num_updates))[step])
            grad = float(scalar_data.get("grad_norm", np.zeros(num_updates))[step])
            extra = ""
            if "base_return" in train_stats and config.task["ENV_NAME"] == "overcooked-v1":
                soups = train_stats["base_return"][step, 0] / 20.0
                extra = f"  soups={soups:.1f}"
            print(
                f"[{print_prefix} {pct:5.1f}%] step={step}/{num_updates}  "
                f"env_steps={env_steps}  {ret_str}{extra}  "
                f"loss={loss:.4f}  grad={grad:.3f}"
            )

    logger.commit()

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(
            name="saved_train_run",
            path=out_savepath,
            type_name="train_run",
        )
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
