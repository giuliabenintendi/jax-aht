"""Inspect what's in the JA_FRUIT_PARTNER_FEED channel during self-play and
whether picked fruits were signaled by the channel in the steps before.

For each step of an ego × ego self-play episode (parameter-shared, same seed)
we print agent_0's incoming partner-feed vector (= per_fruit_attn of agent_1's
spatial attention, lex-sorted), its action, and any fruit eaten this step.
For each eat event we then print the per-fruit value AT THE EATEN SLOT in the
K previous steps, plus the slot's rank at each of those steps — directly
answering "was the eaten apple signaled by the channel beforehand?".

Usage:
    ./run_gpu.sh 0 tools.inspect_channel_picks \\
        --from-wandb --run-id be9uqslh --num-episodes 3 --seeds 0 \\
        [--lookback 5]
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_image_agent
from agents.lbf.ja_lbf_attention import lbf_attention_ctx, per_fruit_attn
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.add_eval_videos import _materialize_from_wandb


def _unwrap_lbf(state):
    s = state
    for _ in range(3):
        if hasattr(s, "food_items"):
            return s
        s = getattr(s, "env_state", None)
        if s is None:
            break
    raise AttributeError("Could not find inner LBF state with food_items")


def _compute_lex_per_fruit(spatial_attn, env_state, lbf_ctx) -> jnp.ndarray:
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
    return np.asarray(pf.squeeze(0)[order])


def _lex_order(env_state) -> np.ndarray:
    """Permutation that lex-sorts the env's food positions."""
    lbf_state = _unwrap_lbf(env_state)
    food_pos = np.asarray(lbf_state.food_items.position)
    return np.lexsort((food_pos[:, 1], food_pos[:, 0]))


def _eaten_lex(env_state) -> np.ndarray:
    lbf_state = _unwrap_lbf(env_state)
    eaten = np.asarray(lbf_state.food_items.eaten)
    return eaten[_lex_order(env_state)]


def _format_vec(v: np.ndarray, width: int = 6, prec: int = 3) -> str:
    return "[" + " ".join(f"{x:>{width}.{prec}f}" for x in v) + "]"


def _slot_rank(v: np.ndarray, slot: int) -> int:
    """1-indexed rank of `slot` in `v` (1 = highest mass; ties → highest rank)."""
    order = np.argsort(-v, kind="stable")  # descending
    return int(np.where(order == slot)[0][0]) + 1


def _run_one_episode(rng, env, policy, params, lbf_ctx, max_steps: int,
                     ep_idx: int, lookback: int):
    """Run a self-play episode (both agents = same trained policy, same seed)
    and trace agent_0's partner-feed + its action + per-step eat events.

    Returns aggregate stats: a list of (lookback_idx, mass_at_eaten_slot,
    rank_of_eaten_slot) tuples for each eat event."""
    inner_env = env._env
    num_fruits = int(lbf_ctx["num_fruits"])

    rng, reset_rng = jax.random.split(rng)
    obs, env_state = inner_env.reset(reset_rng)
    done = {k: jnp.zeros((1,), dtype=bool) for k in inner_env.agents + ["__all__"]}

    hstate_0 = policy.init_hstate(1)
    hstate_1 = policy.init_hstate(1)
    prev_pca_0 = jnp.ones(num_fruits, dtype=jnp.float32) / float(num_fruits)
    prev_pca_1 = jnp.ones(num_fruits, dtype=jnp.float32) / float(num_fruits)

    # Per-step log: (step, pf_a0_vec_np, action_a0, eaten_this_step_lex_slot or -1)
    history = []
    eat_lookbacks = []  # list of dicts {step, slot, lookback_table}
    total_reward = 0.0
    step = 0
    prev_eaten_lex = _eaten_lex(env_state)  # all-False at episode start

    print(f"\n=== Episode {ep_idx} ===  (lookback={lookback} steps)")
    print(f"{'step':>4s} | {'partner-feed (agent_0 input, lex order)':<70s} "
          f"| argmax | act_a0 | eat (lex slot)")
    print("-" * 120)

    while not bool(done["__all__"]) and step < max_steps:
        avail = inner_env.get_avail_actions(env_state)
        obs_0 = jnp.concatenate([obs["agent_0"], prev_pca_0])
        obs_1 = jnp.concatenate([obs["agent_1"], prev_pca_1])

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

        pf_a0 = np.asarray(prev_pca_0)  # what agent_0 actually saw THIS step
        act_a0 = int(act_0.squeeze())

        # Step the env and detect which (if any) fruit went from un-eaten to eaten.
        env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1.squeeze()}
        obs, env_state, reward, done, _info = inner_env.step(step_rng, env_state, env_act)
        total_reward += float(reward["agent_0"])

        new_eaten_lex = _eaten_lex(env_state)
        newly_eaten = np.where(np.logical_and(~prev_eaten_lex, new_eaten_lex))[0]
        eaten_slot = int(newly_eaten[0]) if len(newly_eaten) > 0 else -1
        prev_eaten_lex = new_eaten_lex

        history.append((step, pf_a0, act_a0, eaten_slot))
        argmax_slot = int(pf_a0.argmax())
        print(f"{step:>4d} | {_format_vec(pf_a0):<70s} | {argmax_slot:>6d} | "
              f"{act_a0:>6d} | {eaten_slot if eaten_slot >= 0 else '-':>13}")

        # Capture lookback when an eat happens.
        if eaten_slot >= 0 and step > 0:
            table = []
            for k in range(1, lookback + 1):
                ti = step - k
                if ti < 0:
                    break
                pf_back = history[ti][1]
                mass = float(pf_back[eaten_slot])
                rank = _slot_rank(pf_back, eaten_slot)
                table.append((k, ti, mass, rank, int(pf_back.argmax())))
            eat_lookbacks.append({"step": step, "slot": eaten_slot, "table": table})

        # Update prev_pca from REAL attention (no ablation here).
        prev_pca_0 = _compute_lex_per_fruit(attn_1.squeeze(), env_state, lbf_ctx)
        prev_pca_1 = _compute_lex_per_fruit(attn_0.squeeze(), env_state, lbf_ctx)
        step += 1

    print(f"\n  total_reward (agent_0): {total_reward:.4f}  steps: {step}  "
          f"eats: {len(eat_lookbacks)}")

    if eat_lookbacks:
        print(f"  --- Eat-event lookbacks (was the eaten slot signaled before?) ---")
        for ev in eat_lookbacks:
            print(f"  step {ev['step']:>2d}: slot {ev['slot']} eaten")
            print(f"    {'k':>3s} {'t-k':>4s} {'mass@slot':>10s} {'rank@slot':>10s} {'argmax@t-k':>12s}")
            for (k, ti, mass, rank, am) in ev["table"]:
                print(f"    {k:>3d} {ti:>4d} {mass:>10.4f} {rank:>10d} {am:>12d}")

    return eat_lookbacks, total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Local saved_train_run dir.")
    parser.add_argument("--from-wandb", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    parser.add_argument("--scratch-dir", default="artifacts/channel_picks_tmp")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="Ego seeds to inspect. Default: seed 0 only.")
    parser.add_argument("--eval-seed", type=int, default=4242)
    parser.add_argument("--lookback", type=int, default=5,
                        help="How many steps before an eat to inspect.")
    args = parser.parse_args()

    if args.from_wandb:
        scratch_root = os.path.abspath(args.scratch_dir)
        args.checkpoint = _materialize_from_wandb(
            args.run_id, args.project, args.entity, scratch_root,
        )
    if not args.checkpoint:
        parser.error("--checkpoint required unless --from-wandb is set")

    run_dir = os.path.dirname(args.checkpoint)
    cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml")), resolve=True,
    )
    alg_config = cfg["algorithm"]
    if not bool(alg_config.get("JA_FRUIT_PARTNER_FEED", True)):
        raise SystemExit("JA_FRUIT_PARTNER_FEED is OFF — there's no channel to inspect.")

    env = make_env(alg_config["ENV_NAME"], alg_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(0)
    policy, _ = initialize_ja_image_agent(alg_config, env, rng)

    run_data = load_train_run(args.checkpoint)
    if "final_params" not in run_data:
        raise KeyError(f"final_params missing; keys={list(run_data.keys())}")
    all_params = run_data["final_params"]
    lbf_ctx = lbf_attention_ctx(alg_config, env)
    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 50))

    eval_rng = jax.random.PRNGKey(args.eval_seed)
    all_lookbacks = []
    for seed_idx in args.seeds:
        params_i = jax.tree.map(lambda x: x[seed_idx], all_params)
        print(f"\n#### Ego seed {seed_idx} ####")
        for ep in range(args.num_episodes):
            eval_rng, ep_rng = jax.random.split(eval_rng)
            ep_lookbacks, _ret = _run_one_episode(
                ep_rng, env, policy, params_i, lbf_ctx, max_steps,
                ep_idx=ep, lookback=args.lookback,
            )
            all_lookbacks.extend(ep_lookbacks)

    # Aggregate: rank distribution and mass distribution at each k.
    if not all_lookbacks:
        print("\nNo eat events observed.")
        return

    print("\n" + "=" * 72)
    print(f"AGGREGATE — was the eaten slot signaled by agent_0's partner-feed?")
    print(f"  total eat events: {len(all_lookbacks)}")
    print(f"  K-step-before  mean_mass  median_mass   %top1   %top3   %top5   uniform_baseline_mass(=1/8)")
    for k in range(1, args.lookback + 1):
        masses, ranks = [], []
        for ev in all_lookbacks:
            for (kk, _ti, mass, rank, _am) in ev["table"]:
                if kk == k:
                    masses.append(mass); ranks.append(rank)
                    break
        if not masses:
            continue
        masses = np.asarray(masses); ranks = np.asarray(ranks)
        pct_top1 = float((ranks == 1).mean()) * 100
        pct_top3 = float((ranks <= 3).mean()) * 100
        pct_top5 = float((ranks <= 5).mean()) * 100
        print(f"  k={k:>2d}            {masses.mean():>9.4f}  {np.median(masses):>11.4f}  "
              f"{pct_top1:>5.1f}%  {pct_top3:>5.1f}%  {pct_top5:>5.1f}%   0.1250")
    print("=" * 72)
    print("  Interpretation:")
    print("    - mean_mass >> 0.125 means the channel was concentrated on the eaten slot before the eat.")
    print("    - %top1 high means the eaten slot was the partner's argmax → channel was a clean leading indicator.")
    print("    - If %top1 ≈ 100/N_alive at every k, the channel had no predictive content.")


if __name__ == "__main__":
    main()
