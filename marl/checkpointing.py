"""Shared checkpoint cadence + best-checkpoint selection for the IPPO trainers.

`ja_ippo` saves a checkpoint at each chunk boundary, scores
each one by the mean episodic return over the chunk that produced it, and emit
per-checkpoint folders plus a `chunk_scores.json` so downstream eval can load
the best checkpoint per seed. This module holds that machinery so the two
trainers stay in lockstep.
"""
from __future__ import annotations

import json
import os

import hydra
import jax
import numpy as np

from common.save_load_utils import REPO_PATH, save_train_run


def compute_chunk_boundaries(config, num_updates, env_steps_per_update):
    """Checkpoint cadence expressed as update-index boundaries.

    Returns `(chunk_boundaries, num_ckpts, freq_updates)`. `freq_updates` is the
    per-checkpoint update count when `CHECKPOINT_FREQ_TIMESTEPS > 0` drives a
    fixed-frequency cadence, else `None` (evenly-spaced `NUM_CHECKPOINTS`).
    """
    freq_timesteps = float(config.get("CHECKPOINT_FREQ_TIMESTEPS", 0) or 0)
    if freq_timesteps > 0:
        freq_updates = max(1, int(round(freq_timesteps / env_steps_per_update)))
        chunk_boundaries: list[int] = []
        b = 0
        while b < num_updates:
            b = min(b + freq_updates, num_updates)
            chunk_boundaries.append(b)
        return chunk_boundaries, len(chunk_boundaries), freq_updates

    num_ckpts = config.get("NUM_CHECKPOINTS", 5)
    ckpt_interval = num_updates // max(1, num_ckpts - 1)
    chunk_boundaries = [min((i + 1) * ckpt_interval, num_updates) for i in range(num_ckpts)]
    if chunk_boundaries[-1] < num_updates:
        chunk_boundaries.append(num_updates)
    return chunk_boundaries, num_ckpts, None


def select_best_per_seed_ckpt(out, chunk_boundaries):
    """Score each saved checkpoint by mean episodic return over its producing chunk.

    Returns (best_params, best_idx, per_ckpt_chunk_return). Picking by the chunk
    that produced a checkpoint approximates eval-time return without extra
    rollouts; the argmax is per seed.
    """
    metrics = out["metrics"]
    stacked_ckpts = out["checkpoints"]
    num_seeds, num_ckpts = jax.tree.leaves(stacked_ckpts)[0].shape[:2]

    returned = np.asarray(metrics["returned_episode"])
    returns = np.asarray(metrics["returned_episode_returns"])

    n_chunks = min(num_ckpts, len(chunk_boundaries))
    los = [0] + list(chunk_boundaries[:n_chunks - 1])
    his = list(chunk_boundaries[:n_chunks])

    per_ckpt_returns = np.zeros((num_seeds, num_ckpts), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(los, his)):
        m = returned[:, lo:hi]
        v = returns[:, lo:hi]
        reduce_axes = tuple(range(1, m.ndim))
        denom = np.maximum(m.sum(axis=reduce_axes), 1)
        numer = (v * m).sum(axis=reduce_axes)
        per_ckpt_returns[:, i] = numer / denom

    best_idx = per_ckpt_returns.argmax(axis=1).astype(np.int32)
    seed_arange = np.arange(num_seeds)
    best_params = jax.tree.map(lambda c: c[seed_arange, best_idx], stacked_ckpts)
    return best_params, best_idx, per_ckpt_returns


def finalize_best_checkpoints(config, out, chunk_boundaries, num_ckpts,
                              env_steps_per_update, logger, *, print_prefix):
    """Select the best checkpoint per seed, annotate `out`, and emit
    per-checkpoint folders + `chunk_scores.json`. Returns `best_params`.

    Sets `out["best_params"]`, `out["best_ckpt_idx"]`,
    `out["per_ckpt_chunk_return"]`, and `out["ckpt_env_steps"]`.
    """
    best_params, best_idx, per_ckpt_returns = select_best_per_seed_ckpt(out, chunk_boundaries)
    num_seeds = int(best_idx.shape[0])
    ckpt_env_steps = [int(b) * env_steps_per_update for b in chunk_boundaries[:num_ckpts]]
    out["best_params"] = best_params
    out["best_ckpt_idx"] = best_idx
    out["per_ckpt_chunk_return"] = per_ckpt_returns
    out["ckpt_env_steps"] = np.asarray(ckpt_env_steps, dtype=np.int64)
    best_env_steps = [ckpt_env_steps[int(i)] for i in best_idx]

    savedir_for_scores = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_root_setting = str(config.get("CHECKPOINT_ROOT", "checkpoints"))
    if not os.path.isabs(ckpt_root_setting):
        ckpt_root_setting = os.path.join(REPO_PATH, ckpt_root_setting)
    run_name = None
    if logger is not None and getattr(logger, "run", None) is not None:
        run_name = getattr(logger.run, "name", None)
    if not run_name:
        run_name = os.path.basename(savedir_for_scores.rstrip("/")) or "unnamed_run"
    ckpt_root = os.path.join(ckpt_root_setting, run_name)

    stacked_ckpts = out["checkpoints"]
    ckpt_folder_paths: list[str] = []
    if bool(config.get("SAVE_CHECKPOINT_FOLDER", True)):
        os.makedirs(ckpt_root, exist_ok=True)
        for i in range(num_ckpts):
            params_i = jax.tree.map(lambda c, _i=i: c[:, _i], stacked_ckpts)
            ret_mean = float(per_ckpt_returns[:, i].mean())
            ckpt_name = f"ckpt_{i:02d}_ret_{ret_mean:.2f}"
            save_train_run(params_i, ckpt_root, ckpt_name)
            ckpt_folder_paths.append(os.path.join(ckpt_root, ckpt_name))
        print(f"[{print_prefix}] Checkpoint folder: {ckpt_root} ({num_ckpts} ckpts)", flush=True)

    scores_dir = ckpt_root if bool(config.get("SAVE_CHECKPOINT_FOLDER", True)) else savedir_for_scores
    os.makedirs(scores_dir, exist_ok=True)
    with open(os.path.join(scores_dir, "chunk_scores.json"), "w") as _fh:
        json.dump({
            "run_name": run_name,
            "num_seeds": int(num_seeds),
            "num_ckpts": int(num_ckpts),
            "ckpt_root": ckpt_root,
            "ckpt_env_steps": ckpt_env_steps,
            "ckpt_update_boundaries": [int(b) for b in chunk_boundaries[:num_ckpts]],
            "ckpt_folder_paths": ckpt_folder_paths,
            "per_seed_per_ckpt_return": per_ckpt_returns.tolist(),
            "best_ckpt_idx_per_seed": best_idx.tolist(),
            "best_env_step_per_seed": best_env_steps,
            "best_chunk_return_per_seed": [float(per_ckpt_returns[s, best_idx[s]]) for s in range(num_seeds)],
        }, _fh, indent=2)

    return best_params
