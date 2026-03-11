"""Cross-play evaluation for separately trained seeds.

Loads final_params from N checkpoint directories (one per seed),
builds the NxN cross-play matrix, and prints/saves results.

Usage:
    uv run python -m evaluation.run_xp_seeds \
        --task overcooked-v1-image/cramped_room \
        --checkpoints path/to/seed0/saved_train_run path/to/seed1/saved_train_run ...
"""
import argparse
import os
import time

import jax
import numpy as np
import yaml

from agents.initialize_agents import initialize_ja_image_agent
from common.plot_utils import get_metric_names
from common.run_episodes import run_episodes
from common.save_load_utils import load_train_run
from common.tree_utils import tree_stack
from envs import make_env
from envs.log_wrapper import LogWrapper


EVAL_SEED = 34957
NUM_EVAL_EPISODES = 64
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs", "task")
ALGO_BASE_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "marl", "configs", "algorithm", "ja_ippo", "_base_.yaml"
)


def load_task_config(task_name: str) -> dict:
    config_path = os.path.join(CONFIGS_DIR, f"{task_name}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_algo_config() -> dict:
    with open(ALGO_BASE_CONFIG) as f:
        return yaml.safe_load(f)


def run_xp_evaluation(task_name: str, checkpoint_paths: list[str]):
    task_cfg = load_task_config(task_name)
    algo_cfg = load_algo_config()
    env = make_env(task_cfg["ENV_NAME"], task_cfg["ENV_KWARGS"])
    env = LogWrapper(env)

    num_seeds = len(checkpoint_paths)
    print(f"[xp_seeds] task={task_name}, seeds={num_seeds}, episodes={NUM_EVAL_EPISODES}")

    # Initialize policy (same architecture for all seeds)
    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, init_rng = jax.random.split(rng)
    policy, init_params = initialize_ja_image_agent(algo_cfg, env, init_rng)

    # Load final_params from each seed
    seed_params = []
    for i, ckpt_path in enumerate(checkpoint_paths):
        run_data = load_train_run(ckpt_path)
        params = run_data["final_params"]
        # final_params has shape (1, ...param_dims) since NUM_SEEDS=1
        params = jax.tree.map(lambda x: x[0], params)
        # Verify param structure matches
        assert jax.tree.structure(params) == jax.tree.structure(init_params), \
            f"Param structure mismatch for seed {i}"
        seed_params.append(params)
        print(f"  seed {i}: loaded from {ckpt_path}")

    # Build NxN cross-play matrix
    max_steps = task_cfg["ROLLOUT_LENGTH"]
    rng, eval_rng = jax.random.split(rng)
    outer_rngs = jax.random.split(eval_rng, num_seeds)

    all_metrics = []
    start_time = time.time()
    for i in range(num_seeds):
        rng_i = outer_rngs[i]
        partner_rngs = jax.random.split(rng_i, num_seeds)
        row_metrics = []
        for j in range(num_seeds):
            label = "SP" if i == j else "XP"
            print(f"  [{label}] seed {i} x seed {j} ...", end=" ", flush=True)
            metrics = run_episodes(
                partner_rngs[j], env,
                agent_0_param=seed_params[i], agent_0_policy=policy,
                agent_1_param=seed_params[j], agent_1_policy=policy,
                max_episode_steps=max_steps,
                num_eps=NUM_EVAL_EPISODES,
                agent_0_test_mode=True,
                agent_1_test_mode=True,
            )
            row_metrics.append(metrics)
            ret = np.array(metrics["returned_episode_returns"]).mean()
            print(f"return={ret:.2f}")
        all_metrics.append(tree_stack(row_metrics))

    xp_metrics = tree_stack(all_metrics)
    elapsed = time.time() - start_time
    print(f"[xp_seeds] evaluation done in {elapsed:.1f}s")

    # Print results
    metric_names = get_metric_names(task_cfg["ENV_NAME"])
    seed_names = [f"seed_{i}" for i in range(num_seeds)]
    for metric_name in metric_names:
        print_xp_table(xp_metrics, metric_name, seed_names)

    print_sp_vs_xp_summary(xp_metrics, metric_names, num_seeds)


def print_xp_table(xp_metrics, metric_name, seed_names):
    from prettytable import PrettyTable

    # (N, N, num_episodes, num_agents) -> avg over agents
    data = np.array(xp_metrics[metric_name]).mean(axis=-1)
    n = len(seed_names)
    table = PrettyTable()
    table.field_names = ["agent_0 \\ agent_1"] + seed_names

    for i in range(n):
        row = [seed_names[i]]
        for j in range(n):
            ep_returns = data[i, j]
            mean = ep_returns.mean()
            std = ep_returns.std()
            row.append(f"{mean:.2f} +/- {std:.2f}")
        table.add_row(row)

    print(f"\n{metric_name} (mean +/- std over {data.shape[2]} episodes):")
    print(table)


def print_sp_vs_xp_summary(xp_metrics, metric_names, num_seeds):
    print("\n=== Self-Play vs Cross-Play Summary ===")
    for metric_name in metric_names:
        data = np.array(xp_metrics[metric_name]).mean(axis=-1)  # (N, N, episodes)
        sp_vals = []
        xp_vals = []
        for i in range(num_seeds):
            for j in range(num_seeds):
                ep_mean = data[i, j].mean()
                if i == j:
                    sp_vals.append(ep_mean)
                else:
                    xp_vals.append(ep_mean)
        sp_mean, sp_std = np.mean(sp_vals), np.std(sp_vals)
        xp_mean, xp_std = np.mean(xp_vals), np.std(xp_vals)
        print(f"  {metric_name}:  SP = {sp_mean:.2f} +/- {sp_std:.2f}  |  XP = {xp_mean:.2f} +/- {xp_std:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-play evaluation across seeds")
    parser.add_argument("--task", required=True,
                        help="Task config name (e.g. overcooked-v1-image/cramped_room)")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to saved_train_run directories (one per seed)")
    args = parser.parse_args()

    run_xp_evaluation(args.task, args.checkpoints)
