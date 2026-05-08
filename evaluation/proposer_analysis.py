"""Per-seed proposer-follower scan on a card-game-op checkpoint (text-only).

For each seed, runs N self-play episodes and partitions them into 5 mutually
exclusive categories based on who follows whom:

  only_a0_followed   — A0 yielded at least once; A1 never did.
                       => clean proposer/follower with A1 = proposer, A0 = follower.
  only_a1_followed   — symmetric: A0 = proposer, A1 = follower.
  both_followed      — both agents yielded at some step.
                       => role-mixing within the episode (each agent plays
                          BOTH proposer AND follower at different steps).
  no_follow_success  — no follow events but decision picks matched
                       (lucky alignment from step 1).
  failed             — decision picks didn't match.

Per seed it reports each category as a percentage of the 200 episodes, plus
per-agent role-mixing metrics within the both_followed bucket. Then suggests
XP pair candidates based on the dominant pattern.

Usage:
    ./run_gpu.sh <gpu> evaluation.proposer_analysis \\
        --checkpoint /path/to/saved_train_run \\
        --num-episodes 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from agents.initialize_agents import (
    initialize_ja_agent,
    initialize_ja_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import run_episode_with_states
from omegaconf import OmegaConf


def _episode_role_classification(ep_messages, ep_actions, n_delib):
    """Walk one episode's deliberation messages and return:
        (success, a0_follow_count, a1_follow_count)
    `a0_follow_count` = how many steps A0 yielded to A1's previous message.
    `a1_follow_count` = symmetric.
    Note: if A1 followed at some step, that step had A0 as the *proposer*.
    So `a1_follow_count` is also the number of steps A0 was being-followed.
    """
    if not ep_actions:
        return False, 0, 0
    pick_0 = int(ep_actions[-1][0])
    pick_1 = int(ep_actions[-1][1])
    success = pick_0 >= 0 and pick_1 >= 0 and pick_0 == pick_1

    deliberation = ep_messages[:n_delib] if ep_messages else []
    n_avail = len(deliberation)
    a0 = 0
    a1 = 0
    for t in range(1, n_avail):
        prev_0 = int(deliberation[t - 1][0])
        prev_1 = int(deliberation[t - 1][1])
        curr_0 = int(deliberation[t][0])
        curr_1 = int(deliberation[t][1])
        if prev_0 < 0 or prev_1 < 0 or prev_0 == prev_1:
            continue
        if curr_0 == prev_1 and curr_0 != prev_0:
            a0 += 1
        if curr_1 == prev_0 and curr_1 != prev_1:
            a1 += 1
    return success, a0, a1


def _seed_stats(inner_env, policy, params, max_steps, num_episodes, rng_seed_base):
    n_delib = max_steps - 1
    counts = {
        "only_a0_followed":  0,
        "only_a1_followed":  0,
        "both_followed":     0,
        "no_follow_success": 0,
        "failed":            0,
    }
    a0_played_both_eps = 0   # eps where A0 was follower AND proposer at different steps
    a1_played_both_eps = 0
    a0_total_follow_events = 0   # within both_followed eps, how often A0 yielded
    a1_total_follow_events = 0   # symmetric

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(rng_seed_base + ep)
        _, ep_actions, ep_messages = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=False,
        )
        success, a0, a1 = _episode_role_classification(ep_messages, ep_actions, n_delib)
        if not success:
            counts["failed"] += 1
            continue
        if a0 == 0 and a1 == 0:
            counts["no_follow_success"] += 1
        elif a0 > 0 and a1 == 0:
            counts["only_a0_followed"] += 1
        elif a0 == 0 and a1 > 0:
            counts["only_a1_followed"] += 1
        else:  # both > 0
            counts["both_followed"] += 1
            # In this category each agent played BOTH roles within the episode:
            # A0 yielded `a0` times AND was the proposer-being-followed `a1` times.
            a0_played_both_eps += 1   # by definition
            a1_played_both_eps += 1
            a0_total_follow_events += a0
            a1_total_follow_events += a1

    n = num_episodes
    pcts = {k: 100.0 * v / n for k, v in counts.items()}
    successes = n - counts["failed"]

    both = counts["both_followed"]
    mean_a0_follows_in_mixed = a0_total_follow_events / both if both else 0.0
    mean_a1_follows_in_mixed = a1_total_follow_events / both if both else 0.0

    follower_asymmetry = pcts["only_a0_followed"] - pcts["only_a1_followed"]

    return {
        "counts": counts,
        "pcts": pcts,
        "follower_asymmetry": follower_asymmetry,
        "pct_a0_played_both": (
            100.0 * a0_played_both_eps / successes if successes else 0.0
        ),
        "pct_a1_played_both": (
            100.0 * a1_played_both_eps / successes if successes else 0.0
        ),
        "mean_a0_follows_in_mixed": mean_a0_follows_in_mixed,
        "mean_a1_follows_in_mixed": mean_a1_follows_in_mixed,
        "success_rate": pcts["failed"],  # placeholder — fixed below
        "successes": successes,
    }


def _categorize(stats, asym_thresh=10.0):
    """Tag based on the fixed-role percentages."""
    asym = stats["follower_asymmetry"]
    if abs(asym) < asym_thresh:
        return "symmetric"
    return "a0_follower_prone" if asym > 0 else "a1_follower_prone"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--use-best", action="store_true")
    parser.add_argument(
        "--asym-threshold", type=float, default=10.0,
        help="abs(follower_asymmetry %) above this is treated as 'biased'.",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent
    cfg_path = run_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    algo_cfg = cfg["algorithm"]

    run_data = load_train_run(str(ckpt_path))
    params_key = "best_params" if (args.use_best and "best_params" in run_data) else "final_params"
    if params_key not in run_data:
        params_key = "final_params"
    stacked_params = run_data[params_key]
    num_seeds = jax.tree.leaves(stacked_params)[0].shape[0]

    env_kwargs = dict(algo_cfg["ENV_KWARGS"])
    if algo_cfg.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    if algo_cfg.get("ENV_NAME") == "card-game":
        env_kwargs.setdefault("scramble_partner_msg", False)
    env = make_env(algo_cfg["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)
    obs_type = algo_cfg.get("ENV_KWARGS", {}).get("obs_type", "image")
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent
    policy, _ = init_fn(algo_cfg, env, jax.random.PRNGKey(0))

    max_steps = int(algo_cfg.get(
        "ROLLOUT_LENGTH", algo_cfg.get("ENV_KWARGS", {}).get("max_steps", 8),
    ))

    print(
        f"\n=== Per-seed role-pattern scan ({args.num_episodes} SP eps/seed) ===\n"
        f"  categories (% of all episodes):\n"
        f"    only_a0_followed    → A1 = proposer, A0 = follower (fixed roles)\n"
        f"    only_a1_followed    → A0 = proposer, A1 = follower (fixed roles)\n"
        f"    both_followed       → role-mixing: each agent played both roles\n"
        f"    no_follow_success   → lucky aligned at step 1, never had to follow\n"
        f"    failed              → decision picks didn't match\n"
    )

    header = (
        f"  {'seed':>4s}  {'fail':>5s}  "
        f"{'only_a0':>7s}  {'only_a1':>7s}  {'both':>5s}  {'no_foll':>7s}  "
        f"{'asym':>6s}  {'tag':>20s}"
    )
    print(header)

    rows = []
    for s in range(num_seeds):
        seed_params = jax.tree.map(lambda x: x[s], stacked_params)
        stats = _seed_stats(
            env._env, policy, seed_params, max_steps,
            args.num_episodes, rng_seed_base=600 + s * 1000,
        )
        tag = _categorize(stats, asym_thresh=args.asym_threshold)
        rows.append((s, stats, tag))
        p = stats["pcts"]
        print(
            f"  {s:>4d}  {p['failed']:>4.1f}%  "
            f"{p['only_a0_followed']:>6.1f}%  {p['only_a1_followed']:>6.1f}%  "
            f"{p['both_followed']:>4.1f}%  {p['no_follow_success']:>6.1f}%  "
            f"{stats['follower_asymmetry']:>+5.1f}  {tag:>20s}"
        )

    print(
        f"\n  role-mixing detail (% of successful eps where each agent "
        f"played BOTH roles in the same episode):"
    )
    print(f"  {'seed':>4s}  {'a0_both':>7s}  {'a1_both':>7s}  "
          f"{'mean a0 yields/mixed-ep':>22s}  {'mean a1 yields/mixed-ep':>22s}")
    for s, stats, _tag in rows:
        print(
            f"  {s:>4d}  {stats['pct_a0_played_both']:>6.1f}%  "
            f"{stats['pct_a1_played_both']:>6.1f}%  "
            f"{stats['mean_a0_follows_in_mixed']:>22.2f}  "
            f"{stats['mean_a1_follows_in_mixed']:>22.2f}"
        )

    print("\n  ranked by |follower_asymmetry|:")
    for s, stats, tag in sorted(rows, key=lambda r: -abs(r[1]["follower_asymmetry"])):
        print(
            f"    seed {s:>2d}: asym = {stats['follower_asymmetry']:>+5.1f}%  "
            f"tag = {tag}  (success {100 - stats['pcts']['failed']:.0f}%)"
        )

    a0_prone = [r for r in rows if r[2] == "a0_follower_prone"]
    a1_prone = [r for r in rows if r[2] == "a1_follower_prone"]
    sym = [r for r in rows if r[2] == "symmetric"]
    print(
        f"\n  cohort sizes: a0_follower_prone={len(a0_prone)}, "
        f"a1_follower_prone={len(a1_prone)}, symmetric={len(sym)}  "
        f"(threshold |asym| >= {args.asym_threshold:.0f}%)"
    )

    def _pick(group, n=2, key=lambda r: -abs(r[1]["follower_asymmetry"])):
        return [r[0] for r in sorted(group, key=key)[:n]]

    print("\n  suggested XP pairs to try:")
    if len(a0_prone) >= 2:
        print(f"    two A0-follower-prone   : {_pick(a0_prone)}  → both yield in slot 0; expect deadlock")
    elif len(a0_prone) == 1:
        print(f"    only one A0-follower-prone seed ({a0_prone[0][0]}) — can't pair two")
    else:
        print("    no A0-follower-prone seeds at this threshold")
    if len(a1_prone) >= 2:
        print(f"    two A1-follower-prone   : {_pick(a1_prone)}  → symmetric mirror")
    elif len(a1_prone) == 1:
        print(f"    only one A1-follower-prone seed ({a1_prone[0][0]}) — can't pair two")
    else:
        print("    no A1-follower-prone seeds at this threshold")
    if a0_prone and a1_prone:
        print(f"    mixed (A0-prone × A1-prone): ({_pick(a0_prone, n=1)[0]}, {_pick(a1_prone, n=1)[0]})  → complementary; expect smooth coordination")
    if len(sym) >= 2:
        sym_seeds = _pick(sym, n=2, key=lambda r: abs(r[1]["follower_asymmetry"]))
        print(f"    two symmetric (baseline): {sym_seeds}")

    print("\nDone.")


if __name__ == "__main__":
    main()
