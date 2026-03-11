"""Cross-play evaluation for multi-seed training runs.

Loads final_params (shape: num_seeds, ...) from a single checkpoint,
builds the NxN cross-play matrix, and reports SP/XP with proper SEM
following the pairing scheme from the ZSC literature.

Usage:
    uv run python -m evaluation.run_xp_seeds \
        --task overcooked-v1-image/cramped_room \
        --checkpoint results/.../saved_train_run
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


def xp_mean_and_sem(xp_matrix):
    """Compute XP mean and SEM using the pairing scheme from the ZSC literature.

    With n seeds and 2 players, we pair seeds into m = n//2 groups:
    (0,1), (2,3), (4,5), ... Each group gives one independent XP sample
    by averaging the two permutations: [M[2i, 2i+1] + M[2i+1, 2i]] / 2.

    Args:
        xp_matrix: (n, n) array where entry (i,j) is the mean return
                   when seed i is agent 0 and seed j is agent 1.
    Returns:
        (mean, sem) over the m independent XP samples.
    """
    n = xp_matrix.shape[0]
    m = n // 2
    samples = np.zeros(m)
    for k in range(m):
        i, j = 2 * k, 2 * k + 1
        samples[k] = (xp_matrix[i, j] + xp_matrix[j, i]) / 2
    return np.mean(samples), np.std(samples) / np.sqrt(m)


def run_xp_evaluation(task_name: str, checkpoint_path: str):
    task_cfg = load_task_config(task_name)
    algo_cfg = load_algo_config()
    env = make_env(task_cfg["ENV_NAME"], task_cfg["ENV_KWARGS"])
    env = LogWrapper(env)

    # Load all seeds from single checkpoint
    run_data = load_train_run(checkpoint_path)
    all_final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(all_final_params)[0].shape[0]
    print(f"[xp_seeds] task={task_name}, seeds={num_seeds}, episodes={NUM_EVAL_EPISODES}")

    # Initialize policy
    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, init_rng = jax.random.split(rng)
    policy, init_params = initialize_ja_image_agent(algo_cfg, env, init_rng)

    # Extract per-seed params
    seed_params = []
    for i in range(num_seeds):
        params_i = jax.tree.map(lambda x: x[i], all_final_params)
        assert jax.tree.structure(params_i) == jax.tree.structure(init_params), \
            f"Param structure mismatch for seed {i}"
        seed_params.append(params_i)
        print(f"  seed {i}: loaded")

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

    # Print full matrix and SP vs XP summary
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
            mean = data[i, j].mean()
            std = data[i, j].std()
            row.append(f"{mean:.2f} +/- {std:.2f}")
        table.add_row(row)

    print(f"\n{metric_name} (mean +/- std over {data.shape[2]} episodes):")
    print(table)


def print_sp_vs_xp_summary(xp_metrics, metric_names, num_seeds):
    """Report SP and XP with proper SEM using the seed-pairing scheme."""
    print("\n=== Self-Play vs Cross-Play Summary ===")
    m = num_seeds // 2
    print(f"  ({num_seeds} seeds -> {m} independent XP samples)")
    if num_seeds % 2 != 0:
        print(f"  WARNING: odd number of seeds, last seed excluded from SEM computation")

    for metric_name in metric_names:
        # (N, N, episodes, agents) -> avg over agents and episodes -> (N, N)
        data = np.array(xp_metrics[metric_name]).mean(axis=(-1, -2))

        # SP: diagonal entries
        sp_scores = np.diag(data)
        sp_mean = np.mean(sp_scores)
        sp_sem = np.std(sp_scores) / np.sqrt(len(sp_scores))

        # XP: proper SEM via seed pairing
        xp_mean, xp_sem = xp_mean_and_sem(data)

        print(f"  {metric_name}:  SP = {sp_mean:.2f} +/- {sp_sem:.2f}  |  XP = {xp_mean:.2f} +/- {xp_sem:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-play evaluation across seeds")
    parser.add_argument("--task", required=True,
                        help="Task config name (e.g. overcooked-v1-image/cramped_room)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved_train_run directory")
    args = parser.parse_args()

    run_xp_evaluation(args.task, args.checkpoint)
