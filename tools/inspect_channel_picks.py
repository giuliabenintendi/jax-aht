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
from agents.lbf.ja_lbf_attention import _unwrap_lbf_state, lbf_attention_ctx, per_fruit_attn
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.add_eval_videos import _materialize_from_wandb


def _compute_lex_per_fruit(spatial_attn, env_state, lbf_ctx):
    """Returns (lex-ordered per-fruit normalized vector, on_mass).

    on_mass is the SUM of raw attention values gathered at fruit centre
    cells before renormalization — it's the fraction of the agent's
    spatial-softmax mass that actually landed on apple cells. Compare to
    `uniform_baseline = num_alive_fruits / (feat_h * feat_w)`: if on_mass
    is at that baseline, attention is uniform over the feature map and
    the renormalized per-fruit vector is purely a normalization artifact.
    """
    lbf_state = _unwrap_lbf_state(env_state)
    food_pos = lbf_state.food_items.position
    food_eaten = lbf_state.food_items.eaten
    pf, on_mass = per_fruit_attn(
        spatial_attn[None, :, :],
        food_pos[None, :, :],
        food_eaten[None, :],
        lbf_ctx["tile_size"], lbf_ctx["feat_h"], lbf_ctx["feat_w"],
        lbf_ctx["img_h"], lbf_ctx["img_w"],
    )
    order = jnp.lexsort((food_pos[:, 1], food_pos[:, 0]))
    return np.asarray(pf.squeeze(0)[order]), float(on_mass.squeeze(0))


def _lex_order(env_state) -> np.ndarray:
    """Permutation that lex-sorts the env's food positions."""
    lbf_state = _unwrap_lbf_state(env_state)
    food_pos = np.asarray(lbf_state.food_items.position)
    return np.lexsort((food_pos[:, 1], food_pos[:, 0]))


def _eaten_lex(env_state) -> np.ndarray:
    lbf_state = _unwrap_lbf_state(env_state)
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
    # The very first step's channel is the uniform LBFMechanism.init_carry — no
    # spatial attention has been computed yet, so on_mass is undefined. Carry an
    # 'NA' sentinel for step 0 and fill from agent_1's actual attention onward.
    prev_on_mass = float("nan")

    # Per-step log: (step, pf_a0_vec_np, on_mass, action_a0, eaten_this_step or -1)
    history = []
    eat_lookbacks = []  # list of dicts {step, slot, lookback_table}
    total_reward = 0.0
    step = 0
    prev_eaten_lex = _eaten_lex(env_state)  # all-False at episode start

    feat_cells = int(lbf_ctx["feat_h"]) * int(lbf_ctx["feat_w"])

    print(f"\n=== Episode {ep_idx} ===  (lookback={lookback} steps,  "
          f"feat_cells={feat_cells})")
    print(f"{'step':>4s} | {'partner-feed (agent_0 input, lex order)':<70s} "
          f"| argmax | {'on_mass':>8s} | {'unif_bl':>8s} | act_a0 | eat")
    print("-" * 140)

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
        on_mass_now = prev_on_mass       # the on_mass that produced pf_a0
        act_a0 = int(act_0.squeeze())

        # Uniform baseline at this step: if attention were uniform over the
        # feature map, the gathered on_mass would be num_alive / feat_cells.
        n_alive_now = int((1 - _eaten_lex(env_state).astype(np.int32)).sum())
        uniform_baseline = n_alive_now / float(feat_cells) if feat_cells > 0 else float("nan")

        # Step the env and detect which (if any) fruit went from un-eaten to eaten.
        env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1.squeeze()}
        obs, env_state, reward, done, _info = inner_env.step(step_rng, env_state, env_act)
        total_reward += float(reward["agent_0"])

        new_eaten_lex = _eaten_lex(env_state)
        newly_eaten = np.where(np.logical_and(~prev_eaten_lex, new_eaten_lex))[0]
        eaten_slot = int(newly_eaten[0]) if len(newly_eaten) > 0 else -1
        prev_eaten_lex = new_eaten_lex

        history.append((step, pf_a0, on_mass_now, uniform_baseline, act_a0, eaten_slot))
        argmax_slot = int(pf_a0.argmax())
        on_mass_str = f"{on_mass_now:.4f}" if not np.isnan(on_mass_now) else "  NA  "
        ub_str = f"{uniform_baseline:.4f}"
        print(f"{step:>4d} | {_format_vec(pf_a0):<70s} | {argmax_slot:>6d} | "
              f"{on_mass_str:>8s} | {ub_str:>8s} | {act_a0:>6d} | "
              f"{eaten_slot if eaten_slot >= 0 else '-':>3}")

        # Capture lookback when an eat happens.
        if eaten_slot >= 0 and step > 0:
            table = []
            for k in range(1, lookback + 1):
                ti = step - k
                if ti < 0:
                    break
                _, pf_back, on_mass_back, ub_back, _act, _eat = history[ti]
                mass = float(pf_back[eaten_slot])
                rank = _slot_rank(pf_back, eaten_slot)
                table.append((k, ti, mass, rank, int(pf_back.argmax()),
                              on_mass_back, ub_back))
            eat_lookbacks.append({"step": step, "slot": eaten_slot, "table": table})

        # Update prev_pca from REAL attention (no ablation here). We capture the
        # on_mass that will accompany the NEXT step's channel input.
        prev_pca_0, prev_on_mass = _compute_lex_per_fruit(
            attn_1.squeeze(), env_state, lbf_ctx,
        )
        prev_pca_1, _ = _compute_lex_per_fruit(
            attn_0.squeeze(), env_state, lbf_ctx,
        )
        step += 1

    print(f"\n  total_reward (agent_0): {total_reward:.4f}  steps: {step}  "
          f"eats: {len(eat_lookbacks)}")

    if eat_lookbacks:
        print(f"  --- Eat-event lookbacks (was the eaten slot signaled before?) ---")
        for ev in eat_lookbacks:
            print(f"  step {ev['step']:>2d}: slot {ev['slot']} eaten")
            print(f"    {'k':>3s} {'t-k':>4s} {'mass@slot':>10s} {'rank@slot':>10s} "
                  f"{'argmax@t-k':>12s} {'on_mass':>9s} {'unif_bl':>9s}")
            for (k, ti, mass, rank, am, om, ub) in ev["table"]:
                om_str = f"{om:.4f}" if not np.isnan(om) else "  NA  "
                print(f"    {k:>3d} {ti:>4d} {mass:>10.4f} {rank:>10d} {am:>12d} "
                      f"{om_str:>9s} {ub:>9.4f}")

    return eat_lookbacks, total_reward, history


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
    all_step_on_masses = []      # on_mass at EVERY step (excludes the NA first step)
    all_step_uniform_bls = []    # the matching uniform baselines
    per_seed = {
        s: {"lookbacks": [], "on_masses": [], "uniform_bls": [], "returns": []}
        for s in args.seeds
    }
    for seed_idx in args.seeds:
        params_i = jax.tree.map(lambda x: x[seed_idx], all_params)
        print(f"\n#### Ego seed {seed_idx} ####")
        for ep in range(args.num_episodes):
            eval_rng, ep_rng = jax.random.split(eval_rng)
            ep_lookbacks, ep_ret, history = _run_one_episode(
                ep_rng, env, policy, params_i, lbf_ctx, max_steps,
                ep_idx=ep, lookback=args.lookback,
            )
            all_lookbacks.extend(ep_lookbacks)
            per_seed[seed_idx]["lookbacks"].extend(ep_lookbacks)
            per_seed[seed_idx]["returns"].append(ep_ret)
            for (_st, _pf, om, ub, _act, _eat) in history:
                if not np.isnan(om):
                    all_step_on_masses.append(om)
                    all_step_uniform_bls.append(ub)
                    per_seed[seed_idx]["on_masses"].append(om)
                    per_seed[seed_idx]["uniform_bls"].append(ub)

    # Per-seed comparison table — most useful when --seeds picks more than one
    # ego seed and you want a direct head-to-head on channel meaningfulness.
    if len(args.seeds) >= 2:
        print("\n" + "=" * 88)
        print("PER-SEED COMPARISON (channel meaningfulness side-by-side)")
        print("-" * 88)
        header = (f"  {'seed':>4s}  {'eats':>5s}  {'ret_mean':>9s}  "
                  f"{'mass@k=1':>9s}  {'%top1@k=1':>10s}  {'%top3@k=1':>10s}  "
                  f"{'on_mass_mean':>13s}  {'ratio_mean':>11s}")
        print(header)
        for s in args.seeds:
            seed_data = per_seed[s]
            n_eats = len(seed_data["lookbacks"])
            ret_mean = float(np.mean(seed_data["returns"])) if seed_data["returns"] else float("nan")
            # mass / rank at k=1
            mass_k1, rank_k1 = [], []
            for ev in seed_data["lookbacks"]:
                for row in ev["table"]:
                    if row[0] == 1:
                        mass_k1.append(row[2]); rank_k1.append(row[3])
                        break
            mass_k1_str = f"{np.mean(mass_k1):.4f}" if mass_k1 else "  NA  "
            top1_str = f"{(np.asarray(rank_k1) == 1).mean()*100:5.1f}%" if rank_k1 else " NA "
            top3_str = f"{(np.asarray(rank_k1) <= 3).mean()*100:5.1f}%" if rank_k1 else " NA "
            # on_mass / ratio
            if seed_data["on_masses"]:
                om_arr = np.asarray(seed_data["on_masses"])
                ub_arr = np.asarray(seed_data["uniform_bls"])
                ratio_arr = om_arr / np.maximum(ub_arr, 1e-8)
                om_str = f"{om_arr.mean():.4f}"
                ratio_str = f"{ratio_arr.mean():.3f}×"
            else:
                om_str = "  NA  "; ratio_str = "  NA  "
            print(f"  {s:>4d}  {n_eats:>5d}  {ret_mean:>9.4f}  "
                  f"{mass_k1_str:>9s}  {top1_str:>10s}  {top3_str:>10s}  "
                  f"{om_str:>13s}  {ratio_str:>11s}")
        print("=" * 88)

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
            for row in ev["table"]:
                kk, _ti, mass, rank = row[0], row[1], row[2], row[3]
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

    # on_mass aggregate — answers "is attention actually on apples, or just
    # uniform-over-the-feature-map and the per-fruit signal is a normalization artifact?"
    if all_step_on_masses:
        om = np.asarray(all_step_on_masses)
        ub = np.asarray(all_step_uniform_bls)
        ratio = om / np.maximum(ub, 1e-8)
        print()
        print("=" * 72)
        print(f"AGGREGATE — is attention actually ON apples or just uniform-over-feat-map?")
        print(f"  steps inspected: {len(om)}")
        print(f"  on_mass (fraction of spatial-attn mass landing on fruit cells):")
        print(f"    mean={om.mean():.4f}  median={float(np.median(om)):.4f}  "
              f"q05={float(np.quantile(om, 0.05)):.4f}  q95={float(np.quantile(om, 0.95)):.4f}")
        print(f"  uniform_baseline (= num_alive / feat_cells):")
        print(f"    mean={ub.mean():.4f}  median={float(np.median(ub)):.4f}")
        print(f"  ratio on_mass / uniform_baseline:")
        print(f"    mean={ratio.mean():.3f}×  median={float(np.median(ratio)):.3f}×  "
              f"q05={float(np.quantile(ratio, 0.05)):.3f}×  q95={float(np.quantile(ratio, 0.95)):.3f}×")
        print("=" * 72)
        print("  Interpretation:")
        print("    - ratio ≈ 1.0 → attention is uniform over the feat-map; the per-fruit")
        print("      'signal' is a normalization artifact (no real apple focus).")
        print("    - ratio ≫ 1.0 → attention is genuinely concentrated on apple cells.")
        print("    - For 12x12-8food: feat_cells = feat_h*feat_w (typically 9 with stride=2,")
        print("      4 blocks, SAME padding). Uniform baseline ≈ num_alive/9.")


if __name__ == "__main__":
    main()
