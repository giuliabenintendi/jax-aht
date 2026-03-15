"""Benchmark: full scan path (purejaxrl-style) for JA-IPPO.

Runs the scan path on cramped_room with a small number of updates and seeds.
If it OOMs, we know we need jax.checkpoint.

Usage: ./run_gpu.sh 0 benchmark_scan_vs_loop
"""
import time
import jax
import jax.numpy as jnp

from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ja_ippo import make_train_scan

ENV_KWARGS = {
    "layout": "cramped_room",
    "obs_type": "image",
    "random_obj_state": True,
    "do_reward_shaping": True,
    "reward_shaping_params": {
        "PLACEMENT_IN_POT_REW": 0,
        "PLATE_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "ONION_PICKUP_REWARD": 0,
        "COUNTER_PICKUP_REWARD": 0,
        "COUNTER_DROP_REWARD": 0,
    },
}

NUM_SEEDS = 3
NUM_ENVS = 64
ROLLOUT_LENGTH = 400
NUM_UPDATES_TARGET = 50
TOTAL_TIMESTEPS = NUM_UPDATES_TARGET * ROLLOUT_LENGTH * NUM_ENVS

CONFIG = {
    "ENV_NAME": "overcooked-v1",
    "ENV_KWARGS": ENV_KWARGS,
    "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
    "ROLLOUT_LENGTH": ROLLOUT_LENGTH,
    "NUM_ENVS": NUM_ENVS,
    "NUM_SEEDS": NUM_SEEDS,
    "TRAIN_SEED": 42,
    "NUM_CHECKPOINTS": 2,
    "LR": 4e-4,
    "ANNEAL_LR": True,
    "MAX_GRAD_NORM": 0.5,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.8,
    "VF_COEF": 0.5,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 16,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "ACTIVATION": "relu",
    "JA_CONV_FILTERS": 64,
    "JA_NUM_HEADS": 4,
    "JA_HEAD_FEATURES": 16,
    "JA_SPATIAL_BASIS_DEPTH": 8,
    "JA_SCALAR_EMBED_DIM": 5,
    "FC_HIDDEN_DIM": 256,
    "LSTM_HIDDEN_DIM": 128,
    "CONV_FILTERS": 32,
    "CONV_NUM_BLOCKS": 4,
    "CONV_KERNEL_SIZE": 3,
    "CONV_STRIDE": 2,
    "CONV_PADDING": "SAME",
    "JA_BETA_MAX": 0.05,
    "JA_WARMUP_ENV_STEPS": 700_000,
    "NORMALIZE_REWARDS": True,
    "OBS_TYPE": "image",
}


def main():
    env = make_env("overcooked-v1", ENV_KWARGS)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, NUM_SEEDS)

    config = dict(CONFIG)

    print(f"=== Scan benchmark: {NUM_SEEDS} seeds, {NUM_UPDATES_TARGET} updates ===")
    print(f"JAX devices: {jax.devices()}\n")

    train_fn = make_train_scan(config, env)
    train_jit = jax.jit(jax.vmap(train_fn))

    print("[SCAN] Compiling...")
    t0 = time.perf_counter()
    out = train_jit(rngs)
    jax.block_until_ready(out["final_params"])
    t1 = time.perf_counter()
    print(f"[SCAN] Compile + run: {t1 - t0:.1f}s")

    t0 = time.perf_counter()
    out = train_jit(rngs)
    jax.block_until_ready(out["final_params"])
    t1 = time.perf_counter()
    print(f"[SCAN] Execution only: {t1 - t0:.1f}s")

    # Sanity check
    final = out["final_params"]
    num_params = sum(x.size for x in jax.tree.leaves(final))
    num_nan = sum(int(jnp.isnan(x).sum()) for x in jax.tree.leaves(final))
    print(f"\nParams per seed: {num_params // NUM_SEEDS}, NaN: {num_nan}")


if __name__ == "__main__":
    main()
