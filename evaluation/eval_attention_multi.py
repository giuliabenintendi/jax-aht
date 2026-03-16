"""Multi-episode attention metrics evaluation for JA-IPPO checkpoints.

Loads a trained JA-IPPO checkpoint, runs N eval episodes per seed with
attention collection, and computes averaged attention metrics (stasis,
pct_objects, coverage).

Usage:
    uv run python -m evaluation.eval_attention_multi \
        results/overcooked-v1/cramped_room/ja_ippo/norm_off/2026-03-15_21-50-03 \
        --num-episodes 64 --wandb
"""
from __future__ import annotations

import argparse
import json
import os
import time

import jax
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_image_agent, _get_image_dims
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from common.save_load_utils import load_train_run
from envs import make_env
from evaluation.vis_episodes import (
    run_episode_with_states,
    compute_attention_stasis,
    compute_object_coverage,
)


def load_config_from_run_dir(run_dir: str) -> dict:
    """Load the resolved Hydra config from a run directory."""
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No .hydra/config.yaml found in {run_dir}. "
            "Is this a valid Hydra output directory?"
        )
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def eval_attention_metrics(
    run_dir: str,
    num_episodes: int = 64,
    log_wandb: bool = False,
) -> dict:
    """Run multi-episode attention evaluation for all seeds in a checkpoint.

    Args:
        run_dir: path to a Hydra output directory containing saved_train_run/
                 and .hydra/config.yaml.
        num_episodes: number of eval episodes per seed.
        log_wandb: if True, log results to wandb.

    Returns:
        dict mapping seed index to per-seed metrics.
    """
    cfg = load_config_from_run_dir(run_dir)
    algo_cfg = cfg["algorithm"]

    env_name = algo_cfg["ENV_NAME"]
    env_kwargs = algo_cfg.get("ENV_KWARGS", {})
    max_steps = int(env_kwargs.get("max_steps", 400))

    env = make_env(env_name, env_kwargs)
    # No LogWrapper here, so env is the raw wrapper (e.g. OvercookedImageWrapper).
    # run_episode_with_states expects the unwrapped env directly.
    inner_env = env

    # Load checkpoint
    ckpt_path = os.path.join(run_dir, "saved_train_run")
    run_data = load_train_run(ckpt_path)
    all_final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(all_final_params)[0].shape[0]

    # Initialize policy
    rng = jax.random.PRNGKey(0)
    policy, _ = initialize_ja_image_agent(algo_cfg, env, rng)

    # Feature map dims (needed for object coverage on Overcooked)
    is_overcooked = env_name in ("overcooked-v1",)
    feat_h = feat_w = None
    if is_overcooked:
        img_h, img_w, _ = _get_image_dims(env)
        feat_h, feat_w = _compute_resnet_output_dims(
            img_h, img_w,
            stride=algo_cfg.get("CONV_STRIDE", 2),
            kernel_size=algo_cfg.get("CONV_KERNEL_SIZE", 3),
            padding=algo_cfg.get("CONV_PADDING", "SAME"),
            num_blocks=algo_cfg.get("CONV_NUM_BLOCKS", 4),
        )

    all_results = {}
    start = time.time()

    for seed_idx in range(num_seeds):
        print(f"\n--- Seed {seed_idx}/{num_seeds - 1} ---")
        params = jax.tree.map(lambda x: x[seed_idx], all_final_params)

        # Accumulators across episodes
        stasis_agent0_vals = []
        stasis_agent1_vals = []
        pct_obj_agent0_vals = []
        pct_obj_agent1_vals = []
        category_accum_agent0: dict[str, float] = {}
        category_accum_agent1: dict[str, float] = {}
        episode_lengths = []

        for ep in range(num_episodes):
            ep_rng = jax.random.PRNGKey(42 + seed_idx * 10000 + ep)

            ep_states, attn_data = run_episode_with_states(
                ep_rng, inner_env, params, policy,
                params, policy, max_steps,
                collect_attention=True,
            )
            episode_lengths.append(len(ep_states) - 1)

            # Stasis
            stasis = compute_attention_stasis(attn_data)
            stasis_agent0_vals.append(stasis["agent_0_stasis"])
            stasis_agent1_vals.append(stasis["agent_1_stasis"])

            # Object coverage (Overcooked only)
            if is_overcooked and feat_h is not None and feat_w is not None:
                obj = compute_object_coverage(attn_data, ep_states, feat_h, feat_w)
                pct_obj_agent0_vals.append(obj["agent_0_pct_objects"])
                pct_obj_agent1_vals.append(obj["agent_1_pct_objects"])

                for agent_key, accum in [
                    ("agent_0", category_accum_agent0),
                    ("agent_1", category_accum_agent1),
                ]:
                    cat_mass = obj[f"{agent_key}_category_mass"]
                    for cat, val in cat_mass.items():
                        accum[cat] = accum.get(cat, 0.0) + val

            if (ep + 1) % max(1, num_episodes // 4) == 0:
                print(f"  episodes {ep + 1}/{num_episodes} done")

        # Aggregate
        seed_metrics: dict = {
            "num_episodes": num_episodes,
            "mean_episode_length": float(np.mean(episode_lengths)),
            "stasis_agent0_mean": float(np.nanmean(stasis_agent0_vals)),
            "stasis_agent0_std": float(np.nanstd(stasis_agent0_vals)),
            "stasis_agent1_mean": float(np.nanmean(stasis_agent1_vals)),
            "stasis_agent1_std": float(np.nanstd(stasis_agent1_vals)),
        }

        if is_overcooked and pct_obj_agent0_vals:
            seed_metrics["pct_objects_agent0_mean"] = float(np.nanmean(pct_obj_agent0_vals))
            seed_metrics["pct_objects_agent0_std"] = float(np.nanstd(pct_obj_agent0_vals))
            seed_metrics["pct_objects_agent1_mean"] = float(np.nanmean(pct_obj_agent1_vals))
            seed_metrics["pct_objects_agent1_std"] = float(np.nanstd(pct_obj_agent1_vals))
            seed_metrics["category_mass_agent0"] = {
                k: v / num_episodes for k, v in category_accum_agent0.items()
            }
            seed_metrics["category_mass_agent1"] = {
                k: v / num_episodes for k, v in category_accum_agent1.items()
            }

        all_results[seed_idx] = seed_metrics
        _print_seed_metrics(seed_idx, seed_metrics, is_overcooked)

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")

    # Save JSON
    out_path = os.path.join(run_dir, "attention_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    # Print cross-seed summary
    _print_summary(all_results, is_overcooked)

    if log_wandb:
        _log_to_wandb(all_results, algo_cfg, cfg, run_dir, is_overcooked)

    return all_results


def _print_seed_metrics(seed_idx: int, m: dict, is_overcooked: bool) -> None:
    print(f"  stasis: agent0={m['stasis_agent0_mean']:.4f} +/- {m['stasis_agent0_std']:.4f}, "
          f"agent1={m['stasis_agent1_mean']:.4f} +/- {m['stasis_agent1_std']:.4f}")
    if is_overcooked and "pct_objects_agent0_mean" in m:
        print(f"  pct_objects: agent0={m['pct_objects_agent0_mean']:.4f} +/- {m['pct_objects_agent0_std']:.4f}, "
              f"agent1={m['pct_objects_agent1_mean']:.4f} +/- {m['pct_objects_agent1_std']:.4f}")
        # Top categories by mass
        for agent_label in ("agent0", "agent1"):
            cat = m.get(f"category_mass_{agent_label}", {})
            if cat:
                sorted_cats = sorted(cat.items(), key=lambda x: -x[1])[:5]
                parts = [f"{k}={v:.3f}" for k, v in sorted_cats]
                print(f"    {agent_label} top categories: {', '.join(parts)}")


def _print_summary(all_results: dict, is_overcooked: bool) -> None:
    print("\nCross-seed summary:")
    stasis_0 = [r["stasis_agent0_mean"] for r in all_results.values()]
    stasis_1 = [r["stasis_agent1_mean"] for r in all_results.values()]
    print(f"  stasis agent0: {np.mean(stasis_0):.4f} +/- {np.std(stasis_0):.4f}")
    print(f"  stasis agent1: {np.mean(stasis_1):.4f} +/- {np.std(stasis_1):.4f}")

    if is_overcooked:
        pct_0 = [r["pct_objects_agent0_mean"] for r in all_results.values()
                 if "pct_objects_agent0_mean" in r]
        pct_1 = [r["pct_objects_agent1_mean"] for r in all_results.values()
                 if "pct_objects_agent1_mean" in r]
        if pct_0:
            print(f"  pct_objects agent0: {np.mean(pct_0):.4f} +/- {np.std(pct_0):.4f}")
            print(f"  pct_objects agent1: {np.mean(pct_1):.4f} +/- {np.std(pct_1):.4f}")


def _log_to_wandb(
    all_results: dict,
    algo_cfg: dict,
    full_cfg: dict,
    run_dir: str,
    is_overcooked: bool,
) -> None:
    import wandb

    layout = algo_cfg.get("ENV_KWARGS", {}).get("layout", algo_cfg["ENV_NAME"])
    beta = algo_cfg.get("JA_BETA_MAX", "?")

    wb_run = wandb.init(
        project="aht-benchmark",
        entity="g-benintendi-university-of-brescia",
        config=algo_cfg,
        tags=[str(algo_cfg.get("ALG", "")), layout, f"BETA{beta}", "attn_eval"],
        group=f"{algo_cfg['ENV_NAME']}/{algo_cfg.get('ALG', '')}",
        name=f"attn_eval_BETA{beta}_{layout}",
        dir=run_dir,
    )

    for seed_idx, m in all_results.items():
        prefix = f"attn_eval/seed_{seed_idx}"
        wb_run.summary[f"{prefix}/stasis_agent0"] = m["stasis_agent0_mean"]
        wb_run.summary[f"{prefix}/stasis_agent1"] = m["stasis_agent1_mean"]
        if is_overcooked and "pct_objects_agent0_mean" in m:
            wb_run.summary[f"{prefix}/pct_objects_agent0"] = m["pct_objects_agent0_mean"]
            wb_run.summary[f"{prefix}/pct_objects_agent1"] = m["pct_objects_agent1_mean"]

    # Cross-seed averages
    stasis_0 = np.mean([r["stasis_agent0_mean"] for r in all_results.values()])
    stasis_1 = np.mean([r["stasis_agent1_mean"] for r in all_results.values()])
    wb_run.summary["attn_eval/stasis_agent0_avg"] = float(stasis_0)
    wb_run.summary["attn_eval/stasis_agent1_avg"] = float(stasis_1)

    if is_overcooked:
        pct_0 = [r["pct_objects_agent0_mean"] for r in all_results.values()
                 if "pct_objects_agent0_mean" in r]
        pct_1 = [r["pct_objects_agent1_mean"] for r in all_results.values()
                 if "pct_objects_agent1_mean" in r]
        if pct_0:
            wb_run.summary["attn_eval/pct_objects_agent0_avg"] = float(np.mean(pct_0))
            wb_run.summary["attn_eval/pct_objects_agent1_avg"] = float(np.mean(pct_1))

    # Upload the JSON artifact
    json_path = os.path.join(run_dir, "attention_metrics.json")
    if os.path.exists(json_path):
        wandb.save(json_path, base_path=run_dir)

    wb_run.finish()
    print(f"wandb run: {wb_run.url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-episode attention metrics evaluation for JA-IPPO"
    )
    parser.add_argument(
        "run_dir",
        help="Path to Hydra output directory (e.g. results/overcooked-v1/cramped_room/ja_ippo/...)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=64,
        help="Number of eval episodes per seed (default: 64)",
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Log results to wandb",
    )
    args = parser.parse_args()

    eval_attention_metrics(
        run_dir=args.run_dir,
        num_episodes=args.num_episodes,
        log_wandb=args.wandb,
    )
