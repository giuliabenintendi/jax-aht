"""Dual Destination IPPO trainer (image IPPO + DreamTeam checkpoint/logging structure).

Reuses `image_ippo.make_train` for the network + PPO update (our ResNet+LSTM image
policy, no joint attention), and wraps it with the DreamTeam-style run structure the
Dual Destination work expects:

  - a config snapshot (`config.pckl`) written at run start, after creating
    `<CHECKPOINT_ROOT>/<run_name>/` up front so a long run cannot die at save time;
  - per-seed checkpoints saved during training on the checkpoint cadence as pickled
    `{"params": ...}` payloads — `seed_{s}/params_{i}.pt` plus a rolling `params.pt`;
  - per-update W&B logging (return + losses), and per-checkpoint ego-view gifs
    (`agent0_{i}.gif`, `agent1_{i}.gif`).

Kept separate from `image_ippo` (mirroring the `ja_ippo_lbf` precedent) so the shared
trainers are untouched. Dispatched via `ALG: dd_ippo`.
"""
from __future__ import annotations

import os
import pickle

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from common.save_load_utils import REPO_PATH
from envs import make_env
from envs.dual_destination.rendering import render_dual_destination_ego_frames
from envs.log_wrapper import LogWrapper
from marl.image_ippo import make_train


def _ckpt_root(algorithm_config: dict, logger) -> str:
    """`<CHECKPOINT_ROOT>/<run_name>/`, mirroring ja_ippo's checkpoint-root logic."""
    root = str(algorithm_config.get("CHECKPOINT_ROOT", "checkpoints"))
    if not os.path.isabs(root):
        root = os.path.join(REPO_PATH, root)
    run_name = None
    if logger is not None and getattr(logger, "run", None) is not None:
        run_name = getattr(logger.run, "name", None)
    if not run_name:
        try:
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            run_name = os.path.basename(output_dir.rstrip("/")) or "unnamed_run"
        except Exception:
            run_name = "unnamed_run"
    return os.path.join(root, run_name)


def _define_metrics(logger) -> None:
    if logger is None or getattr(logger, "run", None) is None:
        return
    import wandb

    wandb.define_metric("env_step")
    wandb.define_metric("DualDest/*", step_metric="env_step")


def _save_ckpt(seed_dir: str, ckpt_idx: int, params) -> None:
    """Pickle `{"params": ...}` as DreamTeam does — `params_{i}.pt` and `params.pt`."""
    payload = {"params": jax.device_get(params)}
    with open(os.path.join(seed_dir, f"params_{ckpt_idx}.pt"), "wb") as f:
        pickle.dump(payload, f)
    with open(os.path.join(seed_dir, "params.pt"), "wb") as f:
        pickle.dump(payload, f)


def _log_update(metric: dict, logger, seed_idx: int, env_step: int) -> None:
    """Per-update W&B push (DreamTeam logs every outer step): return + losses."""
    if logger is None or getattr(logger, "run", None) is None:
        return
    returned = np.asarray(metric["returned_episode"])
    returns = np.asarray(metric["returned_episode_returns"])
    n_ep = float(returned.sum())
    mean = float((returns * returned).sum() / n_ep) if n_ep > 0 else float("nan")

    pre = f"DualDest/seed_{seed_idx}"
    data = {
        f"{pre}/return_mean": mean,
        f"{pre}/n_episodes": int(n_ep),
        "env_step": int(env_step),
    }
    for key in ("loss_total", "loss_value", "loss_policy", "entropy", "grad_norm"):
        if key in metric:
            data[f"{pre}/{key}"] = float(np.asarray(metric[key]).mean())
    logger.log(data, commit=True)


def _save_ckpt_gifs(
    seed_dir: str, ckpt_idx: int, inner_env, params, policy, max_steps: int,
    logger, seed_idx: int,
) -> None:
    """Roll one greedy episode and save per-agent ego-view gifs (best-effort)."""
    try:
        import imageio

        from evaluation.vis_episodes import run_episode_with_states

        rollout = run_episode_with_states(
            jax.random.PRNGKey(42), inner_env, params, policy, params, policy, max_steps,
        )
        ep_states = rollout[0]
        frames0, frames1 = render_dual_destination_ego_frames(inner_env, ep_states)
        p0 = os.path.join(seed_dir, f"agent0_{ckpt_idx}.gif")
        p1 = os.path.join(seed_dir, f"agent1_{ckpt_idx}.gif")
        imageio.mimsave(p0, frames0, fps=4)
        imageio.mimsave(p1, frames1, fps=4)
        if logger is not None and getattr(logger, "run", None) is not None:
            logger.log_video(f"DualDest/seed_{seed_idx}/agent0_ckpt{ckpt_idx}", p0, commit=False)
            logger.log_video(f"DualDest/seed_{seed_idx}/agent1_ckpt{ckpt_idx}", p1, commit=False)
    except Exception as e:
        print(f"[dd_ippo] WARN: checkpoint gif failed ({e}); continuing.", flush=True)


def run_dd_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)
    inner_env = env._env

    num_seeds = algorithm_config["NUM_SEEDS"]
    num_updates = int(
        algorithm_config["TOTAL_TIMESTEPS"]
        // algorithm_config["ROLLOUT_LENGTH"]
        // algorithm_config["NUM_ENVS"]
    )
    num_ckpts = algorithm_config.get("NUM_CHECKPOINTS", 5)
    ckpt_interval = num_updates // max(1, num_ckpts - 1)
    env_steps_per_update = algorithm_config["ROLLOUT_LENGTH"] * algorithm_config["NUM_ENVS"]
    save_gifs = bool(algorithm_config.get("SAVE_CKPT_GIFS", True))
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 100))

    # DreamTeam: create the checkpoint dir and snapshot the config up front.
    ckpt_root = _ckpt_root(algorithm_config, logger)
    os.makedirs(ckpt_root, exist_ok=True)
    with open(os.path.join(ckpt_root, "config.pckl"), "wb") as f:
        pickle.dump(OmegaConf.to_container(config, resolve=True), f)
    print(f"[dd_ippo] Checkpoint root: {ckpt_root}", flush=True)

    _define_metrics(logger)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, num_seeds)
    init_fn, make_step_fn = make_train(algorithm_config, env)

    print(f"[dd_ippo] NUM_UPDATES={num_updates}, NUM_SEEDS={num_seeds}, "
          f"NUM_ENVS={algorithm_config['NUM_ENVS']}", flush=True)

    seed_outputs = []
    for s in range(num_seeds):
        seed_dir = os.path.join(ckpt_root, f"seed_{s}")
        os.makedirs(seed_dir, exist_ok=True)

        runner_state, policy = init_fn(rngs[s])
        step_fn = make_step_fn(policy)
        checkpoints = []
        all_metrics = []
        update_steps = jnp.int32(0)

        for step in range(num_updates):
            runner_state, update_steps, metric = step_fn(runner_state, update_steps)
            all_metrics.append(metric)
            _log_update(metric, logger, seed_idx=s,
                        env_step=(step + 1) * env_steps_per_update)

            should_ckpt = (step % ckpt_interval == 0) or (step == num_updates - 1)
            if should_ckpt and len(checkpoints) < num_ckpts:
                ckpt_idx = len(checkpoints)
                params = runner_state[0].params
                checkpoints.append(params)
                _save_ckpt(seed_dir, ckpt_idx, params)
                if save_gifs:
                    _save_ckpt_gifs(
                        seed_dir, ckpt_idx, inner_env, params, policy,
                        max_steps, logger, seed_idx=s,
                    )

            if step % max(1, num_updates // 10) == 0 or step == num_updates - 1:
                print(f"[dd_ippo] Seed {s+1}/{num_seeds}: step {step+1}/{num_updates}", flush=True)

        stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *all_metrics)
        stacked_ckpts = jax.tree.map(lambda *xs: jnp.stack(xs), *checkpoints)
        seed_outputs.append({
            "final_params": runner_state[0].params,
            "metrics": stacked_metrics,
            "checkpoints": stacked_ckpts,
            "final_ckpt_idx": len(checkpoints),
        })

    print("[dd_ippo] Training complete.", flush=True)
    return jax.tree.map(lambda *xs: jnp.stack(xs), *seed_outputs)
