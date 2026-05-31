"""Eval a JA-IPPO LBF ego against a fork-original scripted partner.

Pairs the trained ego (agent_0) with `RandomAgent` or `SequentialFruitAgent`
(agent_1) and runs N episodes per ego seed. Each step the ego's
JA_FRUIT_PARTNER_FEED channel is populated with a **uniform-over-alive**
vector (1/num_alive on alive lex-slots, 0 on eaten) — matches the shape
of training-time `per_fruit_attn` for an uninformative partner (alive mask
+ renormalization), but with no peakedness anywhere. Run on aux and
baseline egos and compare the gap.

Usage:
    ./run_gpu.sh 0 evaluation.eval_scripted_partners \\
        --from-wandb --run-id be9uqslh \\
        --partner seq_nearest --num-episodes 64 [--upload-wandb]
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
from agents.lbf.agent_policy_wrappers import (
    LBFRandomPolicyWrapper,
    LBFSequentialFruitPolicyWrapper,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.add_eval_videos import _materialize_from_wandb


# Maps a CLI shorthand -> (display label, factory(env_kwargs) -> wrapper).
# All SequentialFruitAgent strategies that ship with the repo are exposed
# so we can sweep over partner-predictability if needed.
PARTNERS = {
    "random": (
        "RandomAgent",
        lambda ek: LBFRandomPolicyWrapper(),
    ),
    "seq_lex": (
        "SequentialFruitAgent(lexicographic)",
        lambda ek: LBFSequentialFruitPolicyWrapper(
            grid_size=int(ek.get("grid_size", 7)),
            num_fruits=int(ek.get("num_food", 3)),
            ordering_strategy="lexicographic",
            using_log_wrapper=False,
        ),
    ),
    "seq_revlex": (
        "SequentialFruitAgent(reverse_lexicographic)",
        lambda ek: LBFSequentialFruitPolicyWrapper(
            grid_size=int(ek.get("grid_size", 7)),
            num_fruits=int(ek.get("num_food", 3)),
            ordering_strategy="reverse_lexicographic",
            using_log_wrapper=False,
        ),
    ),
    "seq_col": (
        "SequentialFruitAgent(column_major)",
        lambda ek: LBFSequentialFruitPolicyWrapper(
            grid_size=int(ek.get("grid_size", 7)),
            num_fruits=int(ek.get("num_food", 3)),
            ordering_strategy="column_major",
            using_log_wrapper=False,
        ),
    ),
    "seq_revcol": (
        "SequentialFruitAgent(reverse_column_major)",
        lambda ek: LBFSequentialFruitPolicyWrapper(
            grid_size=int(ek.get("grid_size", 7)),
            num_fruits=int(ek.get("num_food", 3)),
            ordering_strategy="reverse_column_major",
            using_log_wrapper=False,
        ),
    ),
    "seq_nearest": (
        "SequentialFruitAgent(nearest_agent)",
        lambda ek: LBFSequentialFruitPolicyWrapper(
            grid_size=int(ek.get("grid_size", 7)),
            num_fruits=int(ek.get("num_food", 3)),
            ordering_strategy="nearest_agent",
            using_log_wrapper=False,
        ),
    ),
    "seq_farthest": (
        "SequentialFruitAgent(farthest_agent)",
        lambda ek: LBFSequentialFruitPolicyWrapper(
            grid_size=int(ek.get("grid_size", 7)),
            num_fruits=int(ek.get("num_food", 3)),
            ordering_strategy="farthest_agent",
            using_log_wrapper=False,
        ),
    ),
}


def _unwrap_lbf(state):
    s = state
    for _ in range(3):
        if hasattr(s, "food_items"):
            return s
        s = getattr(s, "env_state", None)
        if s is None:
            break
    raise AttributeError("Could not find inner LBF state with food_items")


def _uniform_over_alive_lex(env_state) -> jnp.ndarray:
    """Build the partner-feed vector the aux ego saw at training shape:
    uniform-over-alive in LEX (row, col) slot order, zeros on eaten slots.

    Matches `per_fruit_attn`'s alive mask + renormalization — but with a
    flat (no-information) shape, since this script's partners don't actually
    emit a per-fruit signal.
    """
    lbf_state = _unwrap_lbf(env_state)
    food_pos = lbf_state.food_items.position           # (N, 2) int
    food_eaten = lbf_state.food_items.eaten            # (N,) bool
    order = jnp.lexsort((food_pos[:, 1], food_pos[:, 0]))
    eaten_lex = food_eaten[order]
    alive_lex = 1.0 - eaten_lex.astype(jnp.float32)
    num_alive = jnp.maximum(alive_lex.sum(), 1.0)
    return alive_lex / num_alive


def _run_episode(rng, env, ego_policy, ego_params, partner_policy,
                 max_steps, use_partner_feed: bool, num_fruits: int) -> float:
    inner_env = env._env
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = inner_env.reset(reset_rng)
    done = {k: jnp.zeros((1,), dtype=bool) for k in inner_env.agents + ["__all__"]}

    hstate_ego = ego_policy.init_hstate(1)
    hstate_partner = partner_policy.init_hstate(1, aux_info={"agent_id": 1})

    total_reward = 0.0
    step = 0
    while not bool(done["__all__"]) and step < max_steps:
        avail = inner_env.get_avail_actions(env_state)

        obs_ego = obs["agent_0"]
        if use_partner_feed:
            partner_feed = _uniform_over_alive_lex(env_state)
            obs_ego = jnp.concatenate([obs_ego, partner_feed])

        rng, ego_rng, partner_rng, step_rng = jax.random.split(rng, 4)

        if hasattr(ego_policy, "get_action_and_attention"):
            act_ego, hstate_ego, _ = ego_policy.get_action_and_attention(
                params=ego_params,
                obs=obs_ego.reshape(1, 1, -1),
                done=done["agent_0"].reshape(1, 1),
                avail_actions=avail["agent_0"].astype(jnp.float32),
                hstate=hstate_ego, rng=ego_rng, greedy=True,
            )
        else:
            act_ego, hstate_ego = ego_policy.get_action(
                params=ego_params,
                obs=obs_ego.reshape(1, 1, -1),
                done=done["agent_0"].reshape(1, 1),
                avail_actions=avail["agent_0"].astype(jnp.float32),
                hstate=hstate_ego, rng=ego_rng, greedy=True,
            )
        act_ego = act_ego.squeeze()

        act_partner, hstate_partner = partner_policy.get_action(
            params=None,
            obs=obs["agent_1"],
            done=done["agent_1"],
            avail_actions=avail["agent_1"],
            hstate=hstate_partner,
            rng=partner_rng,
            env_state=env_state,
            greedy=True,
        )
        act_partner = jnp.asarray(act_partner).squeeze()

        env_act = {"agent_0": act_ego, "agent_1": act_partner}
        obs, env_state, reward, done, _info = inner_env.step(step_rng, env_state, env_act)
        total_reward += float(reward["agent_0"])
        step += 1

    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        help="Local saved_train_run dir (with .hydra/config.yaml sibling). "
                             "Required unless --from-wandb is set.")
    parser.add_argument("--from-wandb", action="store_true",
                        help="Download the saved_train_run artifact + reconstruct config "
                             "from the wandb run.")
    parser.add_argument("--run-id", required=True,
                        help="Existing wandb run ID. Used to fetch the artifact when "
                             "--from-wandb, and to label outputs / optionally upload to.")
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    parser.add_argument("--scratch-dir", default="artifacts/eval_scripted_tmp")
    parser.add_argument("--partner", required=True, choices=list(PARTNERS),
                        help=f"Which scripted partner to pair with. "
                             f"Options: {', '.join(PARTNERS)}.")
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--ego-seeds", nargs="+", type=int, default=None,
                        help="Subset of ego training seeds to eval. Default: all.")
    parser.add_argument("--output-csv", default=None,
                        help="Per-episode CSV path. Default: "
                             "artifacts/eval_scripted_<run-id>_<partner>.csv")
    parser.add_argument("--upload-wandb", action="store_true",
                        help="Resume the ego's wandb run and log the partner's return "
                             "summary under ScriptedPartnerEval/<partner>/...")
    args = parser.parse_args()

    if args.from_wandb:
        scratch_root = os.path.abspath(args.scratch_dir)
        args.checkpoint = _materialize_from_wandb(
            args.run_id, args.project, args.entity, scratch_root,
        )
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --from-wandb is set")

    run_dir = os.path.dirname(args.checkpoint)
    cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml")),
        resolve=True,
    )
    alg_config = cfg["algorithm"]
    env_name = alg_config["ENV_NAME"]
    if env_name not in ("lbf", "lbf-reward-shaping"):
        raise SystemExit(f"This eval is LBF-only; got env_name={env_name!r}")

    env = make_env(env_name, alg_config["ENV_KWARGS"])
    env = LogWrapper(env)

    use_partner_feed = bool(alg_config.get("JA_FRUIT_PARTNER_FEED", True))
    inner = env._env if hasattr(env, "_env") else env
    num_fruits = int(getattr(inner, "_num_food",
                             alg_config["ENV_KWARGS"].get("num_food", 3)))

    rng = jax.random.PRNGKey(0)
    ego_policy, _ = initialize_ja_image_agent(alg_config, env, rng)

    run_data = load_train_run(args.checkpoint)
    if "final_params" not in run_data:
        raise KeyError(f"final_params missing; keys={list(run_data.keys())}")
    all_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(all_params)[0].shape[0]
    seed_indices = args.ego_seeds if args.ego_seeds is not None else list(range(num_seeds))

    partner_label, partner_factory = PARTNERS[args.partner]
    partner_policy = partner_factory(alg_config["ENV_KWARGS"])
    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 50))

    print(f"[eval_scripted_partners] ego=final_params  seeds={seed_indices}  "
          f"episodes/seed={args.num_episodes}", flush=True)
    print(f"[eval_scripted_partners] partner={partner_label}", flush=True)
    print(f"[eval_scripted_partners] partner-feed injection: "
          f"{'uniform-over-alive (lex)' if use_partner_feed else 'OFF (ego obs has no partner-feed channel)'}",
          flush=True)

    eval_rng = jax.random.PRNGKey(args.eval_seed)
    rows = []
    seed_means = []
    for seed_idx in seed_indices:
        params_i = jax.tree.map(lambda x: x[seed_idx], all_params)
        returns = np.empty(args.num_episodes, dtype=np.float64)
        for ep in range(args.num_episodes):
            eval_rng, ep_rng = jax.random.split(eval_rng)
            returns[ep] = _run_episode(
                ep_rng, env, ego_policy, params_i, partner_policy,
                max_steps, use_partner_feed, num_fruits,
            )
            rows.append((seed_idx, args.partner, ep, float(returns[ep])))
        seed_mean = float(returns.mean())
        seed_std = float(returns.std())
        seed_means.append(seed_mean)
        print(f"[eval_scripted_partners]   ego_seed={seed_idx:2d}  "
              f"return={seed_mean:.4f} ± {seed_std:.4f} (n={args.num_episodes})",
              flush=True)

    vals = np.asarray(seed_means)
    print()
    print("=" * 72)
    print(f"SUMMARY for run {args.run_id} (final_params) vs {partner_label}")
    print(f"  ego seeds: {len(seed_indices)}   episodes/seed: {args.num_episodes}")
    print(f"  mean of seed-means: {vals.mean():.4f}")
    print(f"  std across seeds  : {vals.std():.4f}")
    print(f"  min seed-mean     : {vals.min():.4f}")
    print(f"  max seed-mean     : {vals.max():.4f}")
    print("=" * 72)

    output_csv = args.output_csv or os.path.join(
        "artifacts", f"eval_scripted_{args.run_id}_{args.partner}.csv",
    )
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ego_seed", "partner", "episode", "return"])
        w.writerows(rows)
    print(f"[eval_scripted_partners] wrote {len(rows)} rows -> {output_csv}", flush=True)

    if args.upload_wandb:
        import wandb
        run = wandb.init(
            project=args.project, entity=args.entity,
            id=args.run_id, resume="must",
        )
        run.log({
            f"ScriptedPartnerEval/{args.partner}/return_mean": float(vals.mean()),
            f"ScriptedPartnerEval/{args.partner}/return_std": float(vals.std()),
            f"ScriptedPartnerEval/{args.partner}/n_seeds": int(len(seed_indices)),
            f"ScriptedPartnerEval/{args.partner}/n_eps_per_seed": int(args.num_episodes),
        }, commit=True)
        run.finish()
        print(f"[eval_scripted_partners] uploaded summary to wandb run {args.run_id}",
              flush=True)


if __name__ == "__main__":
    main()
