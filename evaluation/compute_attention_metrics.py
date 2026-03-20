"""Compute attention stasis and object coverage from saved JA-IPPO checkpoints.

Usage:
    uv run python -m evaluation.compute_attention_metrics \
        --checkpoint results/.../saved_train_run \
        --num-episodes 64 \
        --output results/analysis/attention_metrics.json
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_image_agent, initialize_ja_dual_image_agent, _get_image_dims,
)
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import (
    run_episode_with_states, compute_attention_stasis, compute_object_coverage,
)


def _load_hydra_config(checkpoint_path: str) -> dict:
    """Load the Hydra config.yaml with resolved interpolations."""
    run_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def compute_metrics(checkpoint_path: str, num_episodes: int = 64,
                    output_path: str | None = None):
    cfg = _load_hydra_config(checkpoint_path)
    alg_config = cfg["algorithm"]
    env_name = alg_config["ENV_NAME"]
    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    beta = alg_config.get("JA_BETA_MAX", None)
    layout = alg_config.get("ENV_KWARGS", {}).get("layout", None)
    is_overcooked = env_name in ("overcooked-v1",)

    # Build environment and policy
    env = make_env(env_name, alg_config["ENV_KWARGS"])
    env = LogWrapper(env)

    use_dual = alg_config.get("USE_DUAL_CRITIC", False)
    init_fn = initialize_ja_dual_image_agent if use_dual else initialize_ja_image_agent
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env, rng)

    # Feature map dims (for object coverage)
    feat_h, feat_w = None, None
    if is_overcooked:
        img_h, img_w, _ = _get_image_dims(env)
        feat_h, feat_w = _compute_resnet_output_dims(
            img_h, img_w,
            stride=alg_config.get("CONV_STRIDE", 2),
            kernel_size=alg_config.get("CONV_KERNEL_SIZE", 3),
            padding=alg_config.get("CONV_PADDING", "SAME"),
            num_blocks=alg_config.get("CONV_NUM_BLOCKS", 4),
        )

    # Load checkpoint
    run_data = load_train_run(checkpoint_path)
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    inner_env = env._env

    print(f"[attn metrics] env={env_name} layout={layout} beta={beta} "
          f"seeds={num_seeds} episodes={num_episodes} max_steps={max_steps}")
    if is_overcooked:
        print(f"[attn metrics] feature map: {feat_h}x{feat_w}")

    per_seed_results = []

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)

        # Accumulate per-episode metrics for this seed
        stasis_per_ep = {"agent_0": [], "agent_1": []}
        coverage_per_ep = {"agent_0": [], "agent_1": []}
        category_mass_accum = {"agent_0": {}, "agent_1": {}}

        for ep in range(num_episodes):
            ep_rng = jax.random.PRNGKey(42 + seed_idx * 10000 + ep)
            ep_states, attn_data, _ = run_episode_with_states(
                ep_rng, inner_env, params, policy,
                params, policy, max_steps, collect_attention=True,
            )

            # Attention stasis
            stasis = compute_attention_stasis(attn_data)
            for agent in ("agent_0", "agent_1"):
                stasis_per_ep[agent].append(stasis[f"{agent}_stasis"])

            # Object coverage (Overcooked only)
            if is_overcooked:
                cov = compute_object_coverage(
                    attn_data, ep_states, feat_h, feat_w,
                )
                for agent in ("agent_0", "agent_1"):
                    coverage_per_ep[agent].append(cov[f"{agent}_pct_objects"])
                    ep_cat = cov[f"{agent}_category_mass"]
                    for cat, val in ep_cat.items():
                        category_mass_accum[agent][cat] = (
                            category_mass_accum[agent].get(cat, 0.0) + val
                        )

        # Per-seed summary
        seed_result = {"seed_idx": seed_idx}
        for agent in ("agent_0", "agent_1"):
            s_arr = np.array(stasis_per_ep[agent])
            entry = {
                "stasis_mean": float(np.nanmean(s_arr)),
            }
            if is_overcooked:
                c_arr = np.array(coverage_per_ep[agent])
                entry["pct_objects_mean"] = float(np.nanmean(c_arr))
            seed_result[agent] = entry

        per_seed_results.append(seed_result)
        print(f"[attn metrics] seed {seed_idx}: "
              f"stasis a0={seed_result['agent_0']['stasis_mean']:.4f} "
              f"a1={seed_result['agent_1']['stasis_mean']:.4f}"
              + (f" | coverage a0={seed_result['agent_0'].get('pct_objects_mean', 0):.3f} "
                 f"a1={seed_result['agent_1'].get('pct_objects_mean', 0):.3f}"
                 if is_overcooked else ""))

    # Aggregate across seeds
    def _aggregate(key: str, agent: str) -> tuple[float, float, float]:
        vals = [s[agent][key] for s in per_seed_results if key in s[agent]]
        arr = np.array(vals)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        sem = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        return mean, std, sem

    output = {
        "layout": layout,
        "env_name": env_name,
        "beta": beta,
        "num_seeds": num_seeds,
        "num_episodes": num_episodes,
    }

    for agent in ("agent_0", "agent_1"):
        s_mean, s_std, s_sem = _aggregate("stasis_mean", agent)
        agent_entry = {
            "stasis_mean": s_mean,
            "stasis_std": s_std,
            "stasis_sem": s_sem,
        }
        if is_overcooked:
            c_mean, c_std, c_sem = _aggregate("pct_objects_mean", agent)
            agent_entry["pct_objects_mean"] = c_mean
            agent_entry["pct_objects_std"] = c_std
            agent_entry["pct_objects_sem"] = c_sem
            # Average category mass across all seeds and episodes
            cat_mass = {}
            total_eps = num_seeds * num_episodes
            for agent_key in ("agent_0", "agent_1"):
                if agent_key == agent:
                    for cat in category_mass_accum[agent]:
                        cat_mass[cat] = round(
                            category_mass_accum[agent][cat] / total_eps, 4
                        )
            agent_entry["category_mass"] = cat_mass

        output[agent] = agent_entry

    output["per_seed"] = per_seed_results

    # Save JSON
    if output_path is None:
        run_dir = os.path.dirname(checkpoint_path)
        output_path = os.path.join(run_dir, "attention_metrics.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[attn metrics] Results saved to {output_path}")
    print(f"  stasis  a0={output['agent_0']['stasis_mean']:.4f} +/- {output['agent_0']['stasis_std']:.4f}")
    print(f"  stasis  a1={output['agent_1']['stasis_mean']:.4f} +/- {output['agent_1']['stasis_std']:.4f}")
    if is_overcooked:
        print(f"  coverage a0={output['agent_0']['pct_objects_mean']:.3f} +/- {output['agent_0']['pct_objects_std']:.3f}")
        print(f"  coverage a1={output['agent_1']['pct_objects_mean']:.3f} +/- {output['agent_1']['pct_objects_std']:.3f}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute attention stasis and object coverage from JA-IPPO checkpoints"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved_train_run directory")
    parser.add_argument("--num-episodes", type=int, default=64,
                        help="Number of eval episodes per seed (default: 64)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: <checkpoint_parent>/attention_metrics.json)")
    args = parser.parse_args()

    compute_metrics(args.checkpoint, args.num_episodes, args.output)
