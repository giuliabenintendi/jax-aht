"""Channel-ablation eval for a JA-IPPO LBF ego.

Pairs the trained ego with itself (self-play, same seed, parameter-shared)
and measures how the team return changes when we corrupt the
JA_FRUIT_PARTNER_FEED channel input for BOTH agents symmetrically.

Both agents are the SAME trained policy, so partner identity is held
constant — any return change isolates the policy's reliance on the channel
as an action-relevant input.

Conditions:
  - normal       : agent_0 sees the real per_fruit_attn(agent_1's attention).
                    Reference for training-time team return.
  - zeros        : agent_0's channel = all zeros (hard kill of the signal).
  - uniform-alive: agent_0's channel = 1/num_alive on alive slots, 0 on eaten
                    (in-distribution shape, no peakedness).

Usage:
    ./run_gpu.sh 0 evaluation.channel_ablation \\
        --from-wandb --run-id be9uqslh --num-episodes 64
"""
from __future__ import annotations

import argparse
import csv
import os

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_image_agent
from agents.lbf.ja_lbf_attention import (
    lbf_attention_ctx,
    per_fruit_attn,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.add_eval_videos import _materialize_from_wandb


CONDITIONS = ("normal", "zeros", "uniform-alive", "constant-uniform")


def _unwrap_lbf(state):
    s = state
    for _ in range(3):
        if hasattr(s, "food_items"):
            return s
        s = getattr(s, "env_state", None)
        if s is None:
            break
    raise AttributeError("Could not find inner LBF state with food_items")


def _compute_lex_per_fruit(spatial_attn: jnp.ndarray, env_state, lbf_ctx) -> jnp.ndarray:
    """Reproduce the trainer's mechanism: gather attention at each fruit's
    centre feature cell, alive-mask + renormalize, then permute into the
    canonical lex (row, col) slot ordering."""
    lbf_state = _unwrap_lbf(env_state)
    food_pos = lbf_state.food_items.position
    food_eaten = lbf_state.food_items.eaten
    pf, _ = per_fruit_attn(
        spatial_attn[None, :, :],
        food_pos[None, :, :],
        food_eaten[None, :],
        lbf_ctx["tile_size"], lbf_ctx["feat_h"], lbf_ctx["feat_w"],
        lbf_ctx["img_h"], lbf_ctx["img_w"],
    )
    order = jnp.lexsort((food_pos[:, 1], food_pos[:, 0]))
    return pf.squeeze(0)[order]


def _uniform_alive_lex(env_state, num_fruits: int) -> jnp.ndarray:
    """Lex-ordered alive mask, renormalized: 1/num_alive on alive, 0 on eaten."""
    lbf_state = _unwrap_lbf(env_state)
    food_pos = lbf_state.food_items.position
    food_eaten = lbf_state.food_items.eaten
    order = jnp.lexsort((food_pos[:, 1], food_pos[:, 0]))
    alive_lex = 1.0 - food_eaten[order].astype(jnp.float32)
    n_alive = jnp.maximum(alive_lex.sum(), 1.0)
    return alive_lex / n_alive


def _apply_ablation(pca_real: jnp.ndarray, env_state, condition: str,
                    num_fruits: int) -> jnp.ndarray:
    """Replace agent_0's partner-feed input with the ablated vector.

    `pca_real` is what agent_0 *would* see naturally — used as-is for the
    'normal' baseline so the same code path executes for every condition.
    """
    if condition == "normal":
        return pca_real
    if condition == "zeros":
        return jnp.zeros(num_fruits, dtype=jnp.float32)
    if condition == "uniform-alive":
        return _uniform_alive_lex(env_state, num_fruits)
    if condition == "constant-uniform":
        # State-independent: [1/N, ..., 1/N] every step regardless of eaten mask.
        # Strongest form of "no signal" — also OOD vs training, since the
        # trainer's per_fruit_attn always zeros out eaten slots.
        return jnp.ones(num_fruits, dtype=jnp.float32) / float(num_fruits)
    raise ValueError(f"Unknown condition: {condition!r}")


def _run_episode(rng, env, policy, params, condition: str, lbf_ctx,
                 max_steps: int) -> float:
    inner_env = env._env
    num_fruits = int(lbf_ctx["num_fruits"])

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = inner_env.reset(reset_rng)
    done = {k: jnp.zeros((1,), dtype=bool) for k in inner_env.agents + ["__all__"]}

    hstate_0 = policy.init_hstate(1)
    hstate_1 = policy.init_hstate(1)

    # Both channels start uniform (matches LBFMechanism.init_carry).
    prev_pca_0 = jnp.ones(num_fruits, dtype=jnp.float32) / float(num_fruits)
    prev_pca_1 = jnp.ones(num_fruits, dtype=jnp.float32) / float(num_fruits)

    total_reward = 0.0
    step = 0
    while not bool(done["__all__"]) and step < max_steps:
        avail = inner_env.get_avail_actions(env_state)

        # Apply the ablation symmetrically to BOTH agents so the comparison
        # measures the policy's reliance on the channel as an input — not just
        # one side's degradation while the other gets normal signal.
        pca_0_input = _apply_ablation(prev_pca_0, env_state, condition, num_fruits)
        pca_1_input = _apply_ablation(prev_pca_1, env_state, condition, num_fruits)

        obs_0 = jnp.concatenate([obs["agent_0"], pca_0_input])
        obs_1 = jnp.concatenate([obs["agent_1"], pca_1_input])

        rng, rng0, rng1, step_rng = jax.random.split(rng, 4)
        act_0, hstate_0, attn_0 = policy.get_action_and_attention(
            params=params, obs=obs_0.reshape(1, 1, -1),
            done=done["agent_0"].reshape(1, 1),
            avail_actions=avail["agent_0"].astype(jnp.float32),
            hstate=hstate_0, rng=rng0, greedy=True,
        )
        act_1, hstate_1, attn_1 = policy.get_action_and_attention(
            params=params, obs=obs_1.reshape(1, 1, -1),
            done=done["agent_1"].reshape(1, 1),
            avail_actions=avail["agent_1"].astype(jnp.float32),
            hstate=hstate_1, rng=rng1, greedy=True,
        )

        # Next-step partner-feed values are computed from THIS step's real
        # attention regardless of the ablation — the ablation only affects
        # what agent_0 *sees* in its obs, not what attention is produced.
        prev_pca_0 = _compute_lex_per_fruit(attn_1.squeeze(), env_state, lbf_ctx)
        prev_pca_1 = _compute_lex_per_fruit(attn_0.squeeze(), env_state, lbf_ctx)

        env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1.squeeze()}
        obs, env_state, reward, done, _info = inner_env.step(step_rng, env_state, env_act)
        total_reward += float(reward["agent_0"])
        step += 1

    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        help="Local saved_train_run dir. Required unless --from-wandb.")
    parser.add_argument("--from-wandb", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    parser.add_argument("--scratch-dir", default="artifacts/channel_ablation_tmp")
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--ego-seeds", nargs="+", type=int, default=None,
                        help="Subset of ego training seeds to eval. Default: all.")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                        choices=list(CONDITIONS),
                        help="Which channel-ablation conditions to run. "
                             "Default: all 3 (normal, zeros, uniform-alive).")
    parser.add_argument("--output-csv", default=None,
                        help="Per-episode CSV path. Default: "
                             "artifacts/channel_ablation_<run-id>.csv")
    args = parser.parse_args()

    if args.from_wandb:
        scratch_root = os.path.abspath(args.scratch_dir)
        args.checkpoint = _materialize_from_wandb(
            args.run_id, args.project, args.entity, scratch_root,
        )
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --from-wandb")

    run_dir = os.path.dirname(args.checkpoint)
    cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml")),
        resolve=True,
    )
    alg_config = cfg["algorithm"]
    if alg_config["ENV_NAME"] not in ("lbf", "lbf-reward-shaping"):
        raise SystemExit(f"LBF-only; got env_name={alg_config['ENV_NAME']!r}")
    if not bool(alg_config.get("JA_FRUIT_PARTNER_FEED", True)):
        raise SystemExit("JA_FRUIT_PARTNER_FEED is OFF — ego obs has no "
                         "partner-feed channel to ablate.")

    env = make_env(alg_config["ENV_NAME"], alg_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(0)
    policy, _ = initialize_ja_image_agent(alg_config, env, rng)

    run_data = load_train_run(args.checkpoint)
    if "final_params" not in run_data:
        raise KeyError(f"final_params missing; keys={list(run_data.keys())}")
    all_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(all_params)[0].shape[0]
    seed_indices = args.ego_seeds if args.ego_seeds is not None else list(range(num_seeds))

    lbf_ctx = lbf_attention_ctx(alg_config, env)
    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 50))

    print(f"[channel_ablation] run={args.run_id}  seeds={seed_indices}  "
          f"episodes/seed={args.num_episodes}  conditions={args.conditions}",
          flush=True)
    print(f"[channel_ablation] partner=trained ego (self-play, same seed)  "
          f"ablation applied SYMMETRICALLY to both agents' channels", flush=True)

    rows = []                                        # (seed, condition, ep, return)
    seed_means = {c: [] for c in args.conditions}
    for cond in args.conditions:
        print(f"\n[channel_ablation] === condition: {cond} ===", flush=True)
        # Re-seed per condition so each condition uses the same env-RNG sequence
        # across seeds — that holds the env trajectories fixed across conditions.
        for seed_idx in seed_indices:
            params_i = jax.tree.map(lambda x: x[seed_idx], all_params)
            eval_rng = jax.random.PRNGKey(args.eval_seed + seed_idx)
            returns = np.empty(args.num_episodes, dtype=np.float64)
            for ep in range(args.num_episodes):
                eval_rng, ep_rng = jax.random.split(eval_rng)
                returns[ep] = _run_episode(
                    ep_rng, env, policy, params_i, cond, lbf_ctx, max_steps,
                )
                rows.append((seed_idx, cond, ep, float(returns[ep])))
            mean = float(returns.mean()); std = float(returns.std())
            seed_means[cond].append(mean)
            print(f"[channel_ablation]   ego_seed={seed_idx:2d}  "
                  f"return={mean:.4f} ± {std:.4f} (n={args.num_episodes})",
                  flush=True)

    print()
    print("=" * 78)
    print(f"SUMMARY for run {args.run_id} (final_params, self-play, both-agent channel ablated)")
    print(f"  ego seeds: {len(seed_indices)}   episodes/seed: {args.num_episodes}")
    print("-" * 78)
    print(f"  {'condition':<16s}  {'mean of seed-means':>22s}  "
          f"{'std across seeds':>20s}  {'Δ vs normal':>14s}")
    normal_mean = float(np.mean(seed_means["normal"])) if "normal" in seed_means else None
    for cond in args.conditions:
        vals = np.asarray(seed_means[cond])
        delta = (vals.mean() - normal_mean) if normal_mean is not None else 0.0
        delta_str = f"{delta:+.4f}" if normal_mean is not None and cond != "normal" else ""
        print(f"  {cond:<16s}  {vals.mean():>22.4f}  {vals.std():>20.4f}  "
              f"{delta_str:>14s}")
    print("=" * 78)

    output_csv = args.output_csv or os.path.join(
        "artifacts", f"channel_ablation_{args.run_id}.csv",
    )
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ego_seed", "condition", "episode", "return"])
        w.writerows(rows)
    print(f"[channel_ablation] wrote {len(rows)} rows -> {output_csv}", flush=True)


if __name__ == "__main__":
    main()
