"""Backfill missing post-hoc training history in completed W&B runs.

Re-creates the rows that `report_basic_training_outputs` would have written
if the `step=step` regression (fixed in this branch) hadn't dropped them.
Reads the run's `saved_train_run` orbax artifact, recomputes the same
per-update scalars, and appends them to the existing run via `resume="must"`.

Usage:
    python tools/backfill_wandb_history.py <run_id> [<run_id> ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import orbax.checkpoint
import wandb

ENTITY = "g-benintendi-university-of-brescia"
PROJECT = "aht-benchmark"

IMAGE_IPPO_SCALAR_KEYS = [
    ("loss_total", "Losses"),
    ("loss_value", "Losses"),
    ("loss_policy", "Losses"),
    ("entropy", "Losses"),
    ("grad_norm", "Losses"),
    ("value_mean", "Values"),
]

LBF_EXTRA_SCALAR_KEYS = [
    ("aux_partner_argmax_loss", "Losses"),
    ("r_self_mean", "JA"),
]


def get_metric_names(env_name: str) -> tuple[str, ...]:
    if env_name == "lbf":
        return ("percent_eaten", "returned_episode_returns")
    if env_name == "overcooked-v1":
        return ("base_return", "returned_episode_returns")
    return ("returned_episode_returns",)


def get_stats_numpy(metrics: dict, stats: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Numpy port of common.plot_utils.get_stats — final-episode-step masked mean/std."""
    mask = np.asarray(metrics["returned_episode"])
    denom = mask.sum(axis=(-2, -1))
    denom_safe = np.where(denom > 0, denom, 1)
    out = {}
    for stat_name in stats:
        data = np.asarray(metrics[stat_name])
        masked = np.where(mask, data, 0)
        means = masked.sum(axis=(-2, -1)) / denom_safe
        sq = (masked - means[..., None, None]) ** 2
        var = np.where(mask, sq, 0).sum(axis=(-2, -1)) / denom_safe
        stds = np.sqrt(var)
        out[stat_name] = np.stack([means, stds], axis=-1)
    return out


def download_train_run_artifact(api: wandb.Api, run_id: str, dest_root: Path) -> Path:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    arts = [a for a in run.logged_artifacts() if a.type == "train_run"]
    if not arts:
        raise RuntimeError(f"No train_run artifact on {run_id}")
    art = arts[0]
    target = dest_root / run_id
    target.mkdir(parents=True, exist_ok=True)
    print(f"[{run_id}] downloading artifact {art.name} ({art.size/1e6:.0f} MB)...", flush=True)
    art.download(root=str(target))
    return target


def find_orbax_root(downloaded: Path) -> Path:
    """Locate the orbax checkpoint root (the dir containing _CHECKPOINT_METADATA)."""
    if (downloaded / "_CHECKPOINT_METADATA").exists():
        return downloaded.resolve()
    for cand in downloaded.rglob("_CHECKPOINT_METADATA"):
        return cand.parent.resolve()
    raise RuntimeError(f"Couldn't locate orbax _CHECKPOINT_METADATA under {downloaded}")


def load_metrics(orbax_root: Path) -> dict:
    """Restore the full checkpoint and pluck `metrics`.

    The orbax-API for partial restore needs a target pytree template that's
    awkward to build without the trainer config; restoring the whole thing
    materializes the params too but the metrics dict is what we need.
    """
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = checkpointer.restore(str(orbax_root))
    if "metrics" not in restored:
        raise RuntimeError(f"metrics not in checkpoint at {orbax_root}; keys={list(restored)}")
    return restored["metrics"]


def build_step_payloads(metrics: dict, env_name: str, rollout_length: int,
                        num_envs: int, scalar_keys: list[tuple[str, str]]) -> list[dict]:
    metric_names = get_metric_names(env_name)
    train_stats = get_stats_numpy(metrics, metric_names)

    num_seeds, num_updates = np.asarray(metrics["returned_episode"]).shape[:2]

    train_stats_mean = {
        name: np.asarray(arr)[:, :, 0].mean(axis=0)  # mean across seeds of the per-update means
        for name, arr in train_stats.items()
    }

    scalar_data = {}
    for key, _ in scalar_keys:
        if key in metrics:
            scalar_data[key] = np.asarray(metrics[key]).mean(axis=0)  # mean across seeds

    payloads = []
    for step in range(num_updates):
        env_steps = (step + 1) * rollout_length * num_envs
        payload = {"train_step": step, "env_step": env_steps}
        for name, series in train_stats_mean.items():
            payload[f"Train/{name}"] = float(series[step])
        for key, prefix in scalar_keys:
            if key in scalar_data:
                payload[f"{prefix}/{key}"] = float(scalar_data[key][step])
        payloads.append(payload)
    return payloads


def backfill_run(run_id: str, artifact_root: Path, dry_run: bool) -> None:
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    cfg = run.config
    alg = cfg["algorithm"]
    env_name = alg["ENV_NAME"]
    rollout_length = int(alg["ROLLOUT_LENGTH"])
    num_envs = int(alg["NUM_ENVS"])

    # LBF runs use IMAGE_IPPO_SCALAR_KEYS + LBF extras (see agents/lbf/ja_lbf_future_occupancy.py).
    if env_name == "lbf":
        scalar_keys = IMAGE_IPPO_SCALAR_KEYS + LBF_EXTRA_SCALAR_KEYS
    else:
        scalar_keys = IMAGE_IPPO_SCALAR_KEYS

    download_dir = download_train_run_artifact(api, run_id, artifact_root)
    orbax_root = find_orbax_root(download_dir)
    print(f"[{run_id}] loading metrics from {orbax_root}...", flush=True)
    metrics = load_metrics(orbax_root)

    print(f"[{run_id}] metrics shapes:")
    for k, v in metrics.items():
        arr = np.asarray(v)
        print(f"    {k:40s} shape={arr.shape} dtype={arr.dtype}")

    payloads = build_step_payloads(metrics, env_name, rollout_length, num_envs, scalar_keys)
    print(f"[{run_id}] built {len(payloads)} payloads; first/last:")
    print(f"    first: {payloads[0]}")
    print(f"    last : {payloads[-1]}")

    if dry_run:
        print(f"[{run_id}] DRY RUN — skipping wandb resume/log")
        return

    print(f"[{run_id}] resuming run and re-logging {len(payloads)} rows...", flush=True)
    wandb.init(entity=ENTITY, project=PROJECT, id=run_id, resume="must")
    try:
        for p in payloads:
            wandb.log(p, commit=True)
        print(f"[{run_id}] done.", flush=True)
    finally:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--artifact-root", default="artifacts/backfill_tmp")
    args = parser.parse_args()

    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    for rid in args.run_ids:
        try:
            backfill_run(rid, artifact_root, args.dry_run)
        except Exception as exc:
            print(f"[{rid}] FAILED: {exc}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
