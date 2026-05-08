"""Per-seed proposer-follower scan on a card-game-op checkpoint.

For each seed, runs N SP episodes and computes:
  - success rate
  - mean follow-rate of A0 (= P(a0 adopts a1's previous message at any step))
  - mean follow-rate of A1 (symmetric)
  - PFI (proposer-follower index) = mean_follow_a0 - mean_follow_a1
  - first-follower bias = P(A0 is the first to follow) - P(A1 is)
  - stable-role rate    = fraction of successful eps where exactly one agent
                          ever followed (= "clean fast-follow" episodes)

Then prints a sorted table and suggests interesting XP pair candidates:
  - two strongly-A0-follower-prone seeds  → both yield in slot 0 → likely deadlock
  - two strongly-A1-follower-prone seeds  → symmetric case
  - mixed (A0-follower-prone × A1-follower-prone) → complementary
  - two near-symmetric seeds              → baseline

No figures. Pure text. Pure SP — XP-pair experiments come next, gated on
whether real per-seed asymmetry actually emerges.

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


def _get_obs_type(cfg: dict) -> str:
    return cfg.get("ENV_KWARGS", {}).get("obs_type", "image")


def _episode_follow_stats(ep_messages, ep_actions, n_delib):
    """For one episode, return (success, a0_follow_count, a1_follow_count,
    first_follower_idx_or_None, both_followed_at_some_step)."""
    if not ep_actions:
        return False, 0, 0, None, False
    pick_0 = int(ep_actions[-1][0])
    pick_1 = int(ep_actions[-1][1])
    success = pick_0 >= 0 and pick_1 >= 0 and pick_0 == pick_1

    deliberation = ep_messages[:n_delib] if ep_messages else []
    n_avail = len(deliberation)
    a0 = 0
    a1 = 0
    first_follower = None
    for t in range(1, n_avail):
        prev_0 = int(deliberation[t - 1][0])
        prev_1 = int(deliberation[t - 1][1])
        curr_0 = int(deliberation[t][0])
        curr_1 = int(deliberation[t][1])
        if prev_0 < 0 or prev_1 < 0 or prev_0 == prev_1:
            continue
        a0_follow = curr_0 == prev_1 and curr_0 != prev_0
        a1_follow = curr_1 == prev_0 and curr_1 != prev_1
        if a0_follow:
            a0 += 1
            if first_follower is None:
                first_follower = 0
        if a1_follow:
            a1 += 1
            if first_follower is None:
                first_follower = 1
    both_followed = a0 > 0 and a1 > 0
    return success, a0, a1, first_follower, both_followed


def _seed_stats(inner_env, policy, params, max_steps, num_episodes, rng_seed_base):
    n_delib = max_steps - 1
    successes = 0
    a0_follows_per_ep = []
    a1_follows_per_ep = []
    first_a0 = 0
    first_a1 = 0
    no_first = 0
    clean_fast_follow = 0
    both_followed_eps = 0

    for ep in range(num_episodes):
        ep_rng = jax.random.PRNGKey(rng_seed_base + ep)
        _, ep_actions, ep_messages = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=False,
        )
        success, a0, a1, first_follower, both = _episode_follow_stats(
            ep_messages, ep_actions, n_delib,
        )
        if success:
            successes += 1
            a0_follows_per_ep.append(a0)
            a1_follows_per_ep.append(a1)
            if first_follower == 0:
                first_a0 += 1
            elif first_follower == 1:
                first_a1 += 1
            else:
                no_first += 1
            if both:
                both_followed_eps += 1
            elif (a0 > 0) != (a1 > 0):  # exactly one agent ever followed
                clean_fast_follow += 1

    s = max(1, successes)
    mean_a0 = float(np.mean(a0_follows_per_ep)) if a0_follows_per_ep else 0.0
    mean_a1 = float(np.mean(a1_follows_per_ep)) if a1_follows_per_ep else 0.0
    pfi = mean_a0 - mean_a1

    first_total = first_a0 + first_a1
    first_pfi = (first_a0 - first_a1) / first_total if first_total else 0.0
    stable_role_rate = clean_fast_follow / s

    return {
        "success_rate": successes / num_episodes,
        "mean_follow_a0": mean_a0,
        "mean_follow_a1": mean_a1,
        "pfi": pfi,
        "first_follower_bias": first_pfi,
        "first_a0_count": first_a0,
        "first_a1_count": first_a1,
        "no_first_count": no_first,
        "stable_role_rate": stable_role_rate,
        "both_followed_eps": both_followed_eps,
        "successes": successes,
    }


def _categorize(stats, pfi_thresh=0.05):
    """Loose tag based on PFI sign + magnitude."""
    pfi = stats["pfi"]
    if abs(pfi) < pfi_thresh:
        return "symmetric"
    return "a0_follower_prone" if pfi > 0 else "a1_follower_prone"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--use-best", action="store_true")
    parser.add_argument(
        "--pfi-threshold", type=float, default=0.05,
        help="abs(PFI) above this is treated as 'biased'.",
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
    obs_type = _get_obs_type(algo_cfg)
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent
    policy, _ = init_fn(algo_cfg, env, jax.random.PRNGKey(0))

    max_steps = int(algo_cfg.get(
        "ROLLOUT_LENGTH", algo_cfg.get("ENV_KWARGS", {}).get("max_steps", 8),
    ))

    print(f"\n=== Per-seed proposer-follower scan ({args.num_episodes} SP eps/seed) ===")
    header = (
        f"  {'seed':>4s}  {'success':>8s}  {'foll_a0':>7s}  {'foll_a1':>7s}  "
        f"{'PFI':>7s}  {'1st_a0':>6s}  {'1st_a1':>6s}  {'1st_bias':>8s}  "
        f"{'stable':>6s}  {'tag':>20s}"
    )
    print(header)

    rows = []
    for s in range(num_seeds):
        seed_params = jax.tree.map(lambda x: x[s], stacked_params)
        stats = _seed_stats(
            env._env, policy, seed_params, max_steps,
            args.num_episodes, rng_seed_base=600 + s * 1000,
        )
        tag = _categorize(stats, pfi_thresh=args.pfi_threshold)
        rows.append((s, stats, tag))
        print(
            f"  {s:>4d}  {stats['success_rate']:>8.2%}  "
            f"{stats['mean_follow_a0']:>7.3f}  {stats['mean_follow_a1']:>7.3f}  "
            f"{stats['pfi']:>+7.3f}  "
            f"{stats['first_a0_count']:>6d}  {stats['first_a1_count']:>6d}  "
            f"{stats['first_follower_bias']:>+8.3f}  "
            f"{stats['stable_role_rate']:>6.2f}  {tag:>20s}"
        )

    # Sorted by |PFI| (largest asymmetry first).
    print("\n  ranked by |PFI|:")
    for s, stats, tag in sorted(rows, key=lambda r: -abs(r[1]["pfi"])):
        print(
            f"    seed {s:>2d}: PFI = {stats['pfi']:>+.3f}  tag = {tag}  "
            f"(success {stats['success_rate']:.0%})"
        )

    # Suggested XP pair candidates.
    a0_prone = [r for r in rows if r[2] == "a0_follower_prone"]
    a1_prone = [r for r in rows if r[2] == "a1_follower_prone"]
    sym = [r for r in rows if r[2] == "symmetric"]

    print(
        f"\n  cohort sizes: a0_follower_prone={len(a0_prone)}, "
        f"a1_follower_prone={len(a1_prone)}, symmetric={len(sym)}  "
        f"(threshold |PFI| >= {args.pfi_threshold:.2f})"
    )

    def _pick(group, n=2, key=lambda r: -abs(r[1]["pfi"])):
        return [r[0] for r in sorted(group, key=key)[:n]]

    print("\n  suggested XP pairs to try:")
    if len(a0_prone) >= 2:
        seeds = _pick(a0_prone)
        print(f"    two A0-follower-prone   : {seeds}  → expect both A0 yielding in slot 0; mutual yielding")
    elif len(a0_prone) == 1:
        print(f"    only one A0-follower-prone seed ({a0_prone[0][0]}) — can't pair two")
    else:
        print("    no A0-follower-prone seeds at this threshold")

    if len(a1_prone) >= 2:
        seeds = _pick(a1_prone)
        print(f"    two A1-follower-prone   : {seeds}  → symmetric case")
    elif len(a1_prone) == 1:
        print(f"    only one A1-follower-prone seed ({a1_prone[0][0]}) — can't pair two")
    else:
        print("    no A1-follower-prone seeds at this threshold")

    if a0_prone and a1_prone:
        a0_pick = _pick(a0_prone, n=1)[0]
        a1_pick = _pick(a1_prone, n=1)[0]
        print(f"    mixed (A0-prone × A1-prone): ({a0_pick}, {a1_pick})  → complementary; expect smooth coordination")

    if len(sym) >= 2:
        sym_seeds = _pick(sym, n=2, key=lambda r: abs(r[1]["pfi"]))
        print(f"    two symmetric (baseline): {sym_seeds}")

    print("\nDone.")


if __name__ == "__main__":
    main()
