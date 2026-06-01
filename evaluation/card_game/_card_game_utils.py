"""Shared loading + setup utilities for card-game evaluation drivers.

Centralizes:
  - Hydra config loading from `<run_dir>/.hydra/config.yaml`
  - env construction (with OP wrappers as configured at training time;
    `scramble_partner_msg` is always forced False at eval).
  - policy initialization (auto-routed by obs type and dual-critic flag).
  - per-seed best-checkpoint selection by mean episodic return over each
    saved chunk. Final params are never used at eval time.

Drivers should call `load_card_game_eval(checkpoint_path)` and use the
returned dataclass instead of duplicating this boilerplate.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from omegaconf import OmegaConf

from common.save_load_utils import load_train_run_no_convert
from envs import make_env
from envs.log_wrapper import LogWrapper


@dataclass
class CardGameEval:
    cfg: dict
    alg_config: dict
    env_kwargs: dict
    env: Any              # inner env (with OP wrappers, no LogWrapper)
    env_wrapped: Any      # LogWrapper(env), used for policy init only
    policy: Any
    params: Any           # best per-seed params, stacked along leading axis
    num_seeds: int
    max_steps: int
    label: str
    best_idx: np.ndarray         # (num_seeds,) chunk index that was best per seed
    per_ckpt_return: np.ndarray  # (num_seeds, num_ckpts) chunk-mean returns


def _get_obs_type(alg_config: dict) -> str:
    return alg_config.get(
        "OBS_TYPE",
        alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"),
    )


def _compute_chunk_boundaries(alg_config: dict) -> list[int]:
    """Recompute training chunk boundaries from config (the same logic as
    `marl.ja_ippo.run_ja_ippo`). Needed at eval time to match each saved
    checkpoint to its producing chunk's metrics."""
    num_updates = int(
        float(alg_config["TOTAL_TIMESTEPS"])
        // float(alg_config["ROLLOUT_LENGTH"])
        // float(alg_config["NUM_ENVS"])
    )
    env_steps_per_update = int(alg_config["ROLLOUT_LENGTH"]) * int(alg_config["NUM_ENVS"])
    freq_timesteps = float(alg_config.get("CHECKPOINT_FREQ_TIMESTEPS", 0) or 0)
    if freq_timesteps > 0:
        freq_updates = max(1, int(round(freq_timesteps / env_steps_per_update)))
        boundaries: list[int] = []
        b = 0
        while b < num_updates:
            b = min(b + freq_updates, num_updates)
            boundaries.append(b)
        return boundaries
    num_ckpts = alg_config.get("NUM_CHECKPOINTS", 5)
    ckpt_interval = num_updates // max(1, num_ckpts - 1)
    boundaries = [
        min((i + 1) * ckpt_interval, num_updates) for i in range(num_ckpts)
    ]
    if boundaries[-1] < num_updates:
        boundaries.append(num_updates)
    return boundaries


def select_best_per_seed_params(run_data, alg_config: dict):
    """Per-seed best by mean episodic return over each saved chunk.

    Returns (best_params, best_idx, per_ckpt_return).
    Same scoring as `marl.ja_ippo._select_best_per_seed_ckpt`.
    """
    metrics = run_data["metrics"]
    stacked_ckpts = run_data["checkpoints"]
    num_seeds, num_ckpts = jax.tree.leaves(stacked_ckpts)[0].shape[:2]

    returned = np.asarray(metrics["returned_episode"])
    returns = np.asarray(metrics["returned_episode_returns"])

    chunk_boundaries = _compute_chunk_boundaries(alg_config)
    n_chunks = min(num_ckpts, len(chunk_boundaries))
    los = [0] + list(chunk_boundaries[: n_chunks - 1])
    his = list(chunk_boundaries[:n_chunks])

    per_ckpt_return = np.zeros((num_seeds, num_ckpts), dtype=np.float64)
    for i, (lo, hi) in enumerate(zip(los, his)):
        m = returned[:, lo:hi]
        v = returns[:, lo:hi]
        reduce_axes = tuple(range(1, m.ndim))
        denom = np.maximum(m.sum(axis=reduce_axes), 1)
        numer = (v * m).sum(axis=reduce_axes)
        per_ckpt_return[:, i] = numer / denom

    best_idx = per_ckpt_return.argmax(axis=1).astype(np.int32)
    seed_arange = np.arange(num_seeds)
    best_params = jax.tree.map(lambda c: c[seed_arange, best_idx], stacked_ckpts)
    return best_params, best_idx, per_ckpt_return


def load_card_game_eval(
    checkpoint_path: str,
    use_latest: bool = False,
) -> CardGameEval:
    """Load a card-game checkpoint and prepare everything an eval driver needs.

    - Reads `<run_dir>/.hydra/config.yaml`.
    - Builds env (with the same OP wrappers as training) plus a LogWrapper-
      wrapped copy used only for policy init.
    - Initializes the right policy class (JA / JA-image) based on `OBS_TYPE`.
    - Loads the saved train run and selects the best per-seed checkpoint by
      mean episodic return (or the latest chunk slice when `use_latest`).
    - Forces `scramble_partner_msg` to False at eval time so interventions
      are deterministic.
    """
    from agents.initialize_agents import (
        initialize_ja_agent,
        initialize_ja_image_agent,
    )

    run_dir = os.path.dirname(checkpoint_path)
    cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml")),
        resolve=True,
    )
    alg_config = cfg["algorithm"]

    env_kwargs = dict(alg_config.get("ENV_KWARGS", {}))
    if alg_config.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    env_kwargs["scramble_partner_msg"] = False

    env = make_env(alg_config["ENV_NAME"], env_kwargs)
    env_wrapped = LogWrapper(env)
    obs_type = _get_obs_type(alg_config)
    init_fn = initialize_ja_image_agent if obs_type == "image" else initialize_ja_agent
    policy, _ = init_fn(alg_config, env_wrapped, jax.random.PRNGKey(0))

    run_data = load_train_run_no_convert(checkpoint_path)
    if "checkpoints" not in run_data:
        # Reconstructed checkpoint: reconstruct_best.py rebuilds saved_train_run
        # from the per-chunk folders after a post-train failure, saving only
        # best_params + final_params (no chunk stack or metrics to score). Use
        # the already-selected best_params directly (final_params under use_latest).
        best_params = run_data["final_params"] if use_latest else run_data["best_params"]
        best_idx = run_data.get("best_ckpt_idx")
        per_ckpt_return = None
    else:
        stacked_ckpts = run_data["checkpoints"]
        if use_latest:
            # Last saved chunk per seed. Skips per-ckpt scoring — fine for any
            # qualitative analysis where "trained policy" is enough.
            best_params = jax.tree.map(lambda c: np.asarray(c)[:, -1], stacked_ckpts)
            best_idx = None
            per_ckpt_return = None
        else:
            best_params, best_idx, per_ckpt_return = select_best_per_seed_params(
                run_data, alg_config,
            )

    num_seeds = int(jax.tree.leaves(best_params)[0].shape[0])
    max_steps = int(env_kwargs.get("max_steps", 8))
    label = cfg.get("label", "(unlabeled)")

    return CardGameEval(
        cfg=cfg, alg_config=alg_config, env_kwargs=env_kwargs,
        env=env, env_wrapped=env_wrapped, policy=policy,
        params=best_params, num_seeds=num_seeds,
        max_steps=max_steps, label=label,
        best_idx=best_idx, per_ckpt_return=per_ckpt_return,
    )
