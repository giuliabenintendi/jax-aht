"""Benchmark scan vs loop training paths for JA-IPPO.

Runs both paths on cramped_room with a small number of updates and seeds,
comparing wall-clock time (excluding compilation).
"""
import time
import jax
import jax.numpy as jnp

from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ja_ippo import make_train_scan, make_train_loop, reward_norm_init

# Minimal config matching cramped_room but with fewer timesteps
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
# 50 updates = ~1.28M env steps — enough to benchmark, fast enough to iterate
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
    # PPO
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
    # JA network
    "ACTIVATION": "relu",
    "JA_CONV_FILTERS": 64,
    "JA_NUM_HEADS": 4,
    "JA_HEAD_FEATURES": 16,
    "JA_SPATIAL_BASIS_DEPTH": 8,
    "JA_SCALAR_EMBED_DIM": 5,
    "FC_HIDDEN_DIM": 256,
    "LSTM_HIDDEN_DIM": 128,
    # ResNet
    "CONV_FILTERS": 32,
    "CONV_NUM_BLOCKS": 4,
    "CONV_KERNEL_SIZE": 3,
    "CONV_STRIDE": 2,
    "CONV_PADDING": "SAME",
    # JA reward
    "JA_BETA_MAX": 0.05,
    "JA_WARMUP_ENV_STEPS": 700_000,
    "NORMALIZE_REWARDS": True,
    "OBS_TYPE": "image",
}

SCAN_CHUNK = 50  # for loop path


def benchmark_scan(env, rngs):
    """Full scan path: jit(vmap(train_fn))."""
    config = dict(CONFIG)
    train_fn = make_train_scan(config, env)

    print(f"[SCAN] Compiling jit(vmap(train_fn)) for {NUM_SEEDS} seeds, {config['NUM_UPDATES']} updates...")
    train_jit = jax.jit(jax.vmap(train_fn))

    # Warmup / compile
    t0 = time.perf_counter()
    out = train_jit(rngs)
    jax.block_until_ready(out["final_params"])
    compile_time = time.perf_counter() - t0
    print(f"[SCAN] Compile + first run: {compile_time:.1f}s")

    # Second run (pure execution, no compilation)
    t0 = time.perf_counter()
    out = train_jit(rngs)
    jax.block_until_ready(out["final_params"])
    run_time = time.perf_counter() - t0
    print(f"[SCAN] Second run (execution only): {run_time:.1f}s")

    return compile_time, run_time


def benchmark_scan_checkpoint(env, rngs):
    """Scan path with jax.checkpoint on the scan body."""
    config = dict(CONFIG)

    # Monkey-patch: rebuild make_train_scan with checkpoint
    # We'll do it inline by copying the essentials
    train_fn = make_train_scan(config, env)

    # We can't easily inject checkpoint into the existing function,
    # so let's test if vanilla scan fits first, then try checkpoint if OOM.
    print(f"[SCAN+CKPT] Skipping — test vanilla scan first. If OOM, re-run with checkpoint.")
    return None, None


def benchmark_loop(env, rngs):
    """Loop path: Python loop with chunked jit calls."""
    import functools

    config = dict(CONFIG)
    num_updates = int(config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = num_updates
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )

    init_fn, make_step_fn, init_policy_fn, init_state_fn = make_train_loop(config, env)

    num_ckpts = config.get("NUM_CHECKPOINTS", 2)
    ckpt_interval = num_updates // max(1, num_ckpts - 1)

    chunk_sizes = []
    remaining = num_updates
    while remaining > 0:
        cs = min(SCAN_CHUNK, remaining)
        chunk_sizes.append(cs)
        remaining -= cs

    policy = init_policy_fn(rngs[0])
    runner_state = jax.vmap(lambda rng: init_state_fn(rng, policy))(rngs)
    _, _, raw_single_step = make_step_fn(policy)

    @functools.partial(jax.jit, static_argnums=(3,))
    def vmapped_chunked_step(runner_states, update_steps_all, rew_norm_states, chunk_size):
        def per_seed_chunk(rs, us, rns):
            def _scan_body(carry, _):
                rs, us, rns = carry
                rs, us, rns, metric = raw_single_step(rs, us, rns)
                return (rs, us, rns), metric
            (rs, us, rns), metrics = jax.lax.scan(
                _scan_body, (rs, us, rns), None, length=chunk_size)
            return rs, us, rns, metrics
        return jax.vmap(per_seed_chunk)(runner_states, update_steps_all, rew_norm_states)

    update_steps = jnp.zeros(NUM_SEEDS, dtype=jnp.int32)
    rew_norm_state = jax.vmap(lambda _: reward_norm_init())(jnp.arange(NUM_SEEDS))

    print(f"[LOOP] Compiling + first run ({len(chunk_sizes)} chunks, {num_updates} updates)...")
    t0 = time.perf_counter()

    checkpoints = []
    steps_done = 0
    next_ckpt = 0

    for ci, cs in enumerate(chunk_sizes):
        runner_state, update_steps, rew_norm_state, metric = vmapped_chunked_step(
            runner_state, update_steps, rew_norm_state, cs)
        steps_done += cs
        while next_ckpt < steps_done and len(checkpoints) < num_ckpts:
            checkpoints.append(jax.tree.map(jnp.copy, runner_state[0].params))
            next_ckpt += ckpt_interval
        if steps_done == num_updates and len(checkpoints) < num_ckpts:
            checkpoints.append(jax.tree.map(jnp.copy, runner_state[0].params))

    jax.block_until_ready(runner_state[0].params)
    first_run_time = time.perf_counter() - t0
    print(f"[LOOP] Compile + first run: {first_run_time:.1f}s")

    # Second run — re-init and re-run
    runner_state = jax.vmap(lambda rng: init_state_fn(rng, policy))(rngs)
    update_steps = jnp.zeros(NUM_SEEDS, dtype=jnp.int32)
    rew_norm_state = jax.vmap(lambda _: reward_norm_init())(jnp.arange(NUM_SEEDS))

    t0 = time.perf_counter()
    checkpoints = []
    steps_done = 0
    next_ckpt = 0
    for ci, cs in enumerate(chunk_sizes):
        runner_state, update_steps, rew_norm_state, metric = vmapped_chunked_step(
            runner_state, update_steps, rew_norm_state, cs)
        steps_done += cs
        while next_ckpt < steps_done and len(checkpoints) < num_ckpts:
            checkpoints.append(jax.tree.map(jnp.copy, runner_state[0].params))
            next_ckpt += ckpt_interval
        if steps_done == num_updates and len(checkpoints) < num_ckpts:
            checkpoints.append(jax.tree.map(jnp.copy, runner_state[0].params))

    jax.block_until_ready(runner_state[0].params)
    run_time = time.perf_counter() - t0
    print(f"[LOOP] Second run (execution only): {run_time:.1f}s")

    return first_run_time, run_time


def main():
    env = make_env("overcooked-v1", ENV_KWARGS)
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(42)
    rngs = jax.random.split(rng, NUM_SEEDS)

    print(f"=== Benchmark: {NUM_SEEDS} seeds, {NUM_UPDATES_TARGET} updates, "
          f"{NUM_ENVS} envs, rollout={ROLLOUT_LENGTH} ===")
    print(f"Total env steps per seed: {TOTAL_TIMESTEPS:,.0f}")
    print(f"JAX devices: {jax.devices()}\n")

    # Run loop path first (known to work)
    loop_compile, loop_run = benchmark_loop(env, rngs)
    print()

    # Run scan path
    try:
        scan_compile, scan_run = benchmark_scan(env, rngs)
    except Exception as e:
        print(f"[SCAN] FAILED: {e}")
        scan_compile, scan_run = None, None

    print("\n=== Results ===")
    print(f"{'Path':<20} {'Compile+Run':>12} {'Execution':>12}")
    print(f"{'Loop (chunked)':<20} {loop_compile:>11.1f}s {loop_run:>11.1f}s")
    if scan_run is not None:
        print(f"{'Scan (full XLA)':<20} {scan_compile:>11.1f}s {scan_run:>11.1f}s")
        speedup = loop_run / scan_run if scan_run > 0 else float('inf')
        print(f"\nScan speedup (execution only): {speedup:.2f}x")
    else:
        print(f"{'Scan (full XLA)':<20} {'OOM/FAIL':>12} {'—':>12}")
        print("\nTry adding jax.checkpoint or reducing NUM_SEEDS.")


if __name__ == "__main__":
    main()
