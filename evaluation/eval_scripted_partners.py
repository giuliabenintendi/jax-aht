"""Eval a JA-IPPO LBF ego against a fork-original scripted partner.

Pairs the trained ego (agent_0) with `RandomAgent` or `SequentialFruitAgent`
(agent_1) and runs N episodes per ego seed. Each step the ego's
JA_FRUIT_PARTNER_FEED channel is populated by `--feed-mode`.

For sequential partners, `target-onehot` is the useful diagnostic: the feed is
the partner's previous-step scripted target fruit as a one-hot vector in the
same lexicographic fruit-slot order used by training-time `per_fruit_attn`.
This tests whether an ego trained with aux + per-fruit partner feed benefits
from a clean scripted apple/fruit-attention signal.

Usage:
    ./run_gpu.sh 0 evaluation.eval_scripted_partners \\
        --from-wandb --run-id be9uqslh \\
        --partner seq_nearest --feed-mode target-onehot \\
        --num-episodes 64 [--upload-wandb]
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

FEED_MODES = ("uniform-alive", "target-onehot", "zeros", "constant-uniform")


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


def _constant_uniform(num_fruits: int) -> jnp.ndarray:
    return jnp.ones(num_fruits, dtype=jnp.float32) / float(num_fruits)


def _scripted_target_onehot_lex(env_state, partner_hstate,
                                num_fruits: int) -> jnp.ndarray:
    """One-hot scripted target in the ego channel's lex fruit-slot order.

    SequentialFruitAgent stores its current target in `hstate.sequence[hstate.idx]`.
    That sequence may be nearest/farthest/etc.; the JA scalar suffix, however,
    is always lex-sorted by fruit position. This converts from the scripted
    agent's target position back into the lex slot index.
    """
    if not hasattr(partner_hstate, "sequence") or not hasattr(partner_hstate, "idx"):
        return _uniform_over_alive_lex(env_state)

    lbf_state = _unwrap_lbf(env_state)
    food_pos = lbf_state.food_items.position
    food_eaten = lbf_state.food_items.eaten
    order = jnp.lexsort((food_pos[:, 1], food_pos[:, 0]))
    food_pos_lex = food_pos[order]
    eaten_lex = food_eaten[order]

    idx = jnp.clip(partner_hstate.idx, 0, num_fruits - 1)
    target = partner_hstate.sequence[idx]
    matches = jnp.all(food_pos_lex == target, axis=-1) & ~eaten_lex
    valid = jnp.any(matches)
    onehot = matches.astype(jnp.float32)
    return jax.lax.cond(
        valid,
        lambda: onehot / jnp.maximum(onehot.sum(), 1.0),
        lambda: _uniform_over_alive_lex(env_state),
    )


def _feed_from_mode(feed_mode: str, env_state, partner_hstate,
                    num_fruits: int) -> jnp.ndarray:
    if feed_mode == "uniform-alive":
        return _uniform_over_alive_lex(env_state)
    if feed_mode == "target-onehot":
        return _scripted_target_onehot_lex(env_state, partner_hstate, num_fruits)
    if feed_mode == "zeros":
        return jnp.zeros(num_fruits, dtype=jnp.float32)
    if feed_mode == "constant-uniform":
        return _constant_uniform(num_fruits)
    raise ValueError(f"Unknown feed mode: {feed_mode!r}")


def _make_episode_runner(env, ego_policy, ego_uses_attention: bool,
                         partner_policy, max_steps: int,
                         use_partner_feed: bool, num_fruits: int,
                         feed_mode: str):
    """Build a jitted single-episode runner. The previous Python-while-loop
    version syncs host↔device every step (via `bool(done)` / `float(reward)`),
    which dominates wall-clock — 64 episodes × 12 seeds takes ~20 min. This
    version uses `lax.scan` over `max_steps` and returns a single accumulated
    reward, so the whole episode runs as one GPU kernel. ~30x speedup, lets
    us push to 256+ episodes without pain.
    """
    inner_env = env._env

    # The scripted partner's hstate factory uses a Python dict (`aux_info`)
    # which isn't traceable, so we capture an init hstate once outside jit.
    partner_hstate_init = partner_policy.init_hstate(1, aux_info={"agent_id": 1})
    ego_hstate_init = ego_policy.init_hstate(1)

    def episode_fn(rng, ego_params):
        rng, reset_rng = jax.random.split(rng)
        obs0, env_state0 = inner_env.reset(reset_rng)
        # `inner_env.step` returns `done` values as scalar bool[] — match that
        # shape in the init so `lax.scan` sees identical carry types in/out.
        done0 = {k: jnp.bool_(False) for k in inner_env.agents + ["__all__"]}
        prev_partner_feed0 = _constant_uniform(num_fruits)

        def step_body(carry, _):
            (rng_c, obs_c, env_state_c, done_c,
             hstate_ego_c, hstate_partner_c,
             prev_partner_feed_c, total_reward_c, terminated_c) = carry

            avail = inner_env.get_avail_actions(env_state_c)

            obs_ego_in = obs_c["agent_0"]
            if use_partner_feed:
                obs_ego_in = jnp.concatenate([obs_ego_in, prev_partner_feed_c])

            rng_c, ego_rng, partner_rng, step_rng = jax.random.split(rng_c, 4)

            if ego_uses_attention:
                act_ego, hstate_ego_new, _ = ego_policy.get_action_and_attention(
                    params=ego_params,
                    obs=obs_ego_in.reshape(1, 1, -1),
                    done=done_c["agent_0"].reshape(1, 1),
                    avail_actions=avail["agent_0"].astype(jnp.float32),
                    hstate=hstate_ego_c, rng=ego_rng, greedy=True,
                )
            else:
                act_ego, hstate_ego_new = ego_policy.get_action(
                    params=ego_params,
                    obs=obs_ego_in.reshape(1, 1, -1),
                    done=done_c["agent_0"].reshape(1, 1),
                    avail_actions=avail["agent_0"].astype(jnp.float32),
                    hstate=hstate_ego_c, rng=ego_rng, greedy=True,
                )
            act_ego = act_ego.squeeze()

            act_partner, hstate_partner_new = partner_policy.get_action(
                params=None,
                obs=obs_c["agent_1"],
                done=done_c["agent_1"],
                avail_actions=avail["agent_1"],
                hstate=hstate_partner_c,
                rng=partner_rng,
                env_state=env_state_c,
                greedy=True,
            )
            act_partner = jnp.asarray(act_partner).squeeze()

            next_partner_feed = _feed_from_mode(
                feed_mode, env_state_c, hstate_partner_new, num_fruits,
            )

            env_act = {"agent_0": act_ego, "agent_1": act_partner}
            new_obs, new_env_state, reward, new_done, _info = inner_env.step(
                step_rng, env_state_c, env_act,
            )

            # Gate the reward by terminated_c so we don't accumulate past episode end.
            gated_r = jnp.where(terminated_c, 0.0, reward["agent_0"].astype(jnp.float32))
            new_total = total_reward_c + gated_r
            new_terminated = terminated_c | new_done["__all__"].squeeze()

            new_feed = jnp.where(
                new_done["__all__"].squeeze(),
                _constant_uniform(num_fruits),
                next_partner_feed,
            ) if use_partner_feed else prev_partner_feed_c

            new_carry = (
                rng_c, new_obs, new_env_state, new_done,
                hstate_ego_new, hstate_partner_new,
                new_feed, new_total, new_terminated,
            )
            return new_carry, None

        init_carry = (
            rng, obs0, env_state0, done0,
            ego_hstate_init, partner_hstate_init,
            prev_partner_feed0,
            jnp.float32(0.0),
            jnp.bool_(False),
        )
        final_carry, _ = jax.lax.scan(step_body, init_carry, None, length=max_steps)
        total_reward = final_carry[7]
        return total_reward

    return jax.jit(episode_fn)


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
    parser.add_argument("--feed-mode", default="uniform-alive",
                        choices=list(FEED_MODES),
                        help="How to populate the ego's JA_FRUIT_PARTNER_FEED "
                             "suffix when it exists. target-onehot uses the "
                             "sequential partner's lagged scripted target fruit.")
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
          f"{args.feed_mode if use_partner_feed else 'OFF (ego obs has no partner-feed channel)'}",
          flush=True)

    eval_rng = jax.random.PRNGKey(args.eval_seed)

    # Build the jitted single-episode runner ONCE (compiles on first call).
    # Subsequent episodes reuse the same compiled program — that's where
    # the ~30x speedup over the previous Python while-loop comes from.
    ego_uses_attention = hasattr(ego_policy, "get_action_and_attention")
    episode_runner = _make_episode_runner(
        env, ego_policy, ego_uses_attention,
        partner_policy, max_steps,
        use_partner_feed, num_fruits, args.feed_mode,
    )

    rows = []
    seed_means = []
    for seed_idx in seed_indices:
        params_i = jax.tree.map(lambda x: x[seed_idx], all_params)
        returns = np.empty(args.num_episodes, dtype=np.float64)
        for ep in range(args.num_episodes):
            eval_rng, ep_rng = jax.random.split(eval_rng)
            returns[ep] = float(episode_runner(ep_rng, params_i))
            rows.append((seed_idx, args.partner, args.feed_mode, ep, float(returns[ep])))
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
    print(f"  feed mode: {args.feed_mode if use_partner_feed else 'off'}")
    print(f"  ego seeds: {len(seed_indices)}   episodes/seed: {args.num_episodes}")
    print(f"  mean of seed-means: {vals.mean():.4f}")
    print(f"  std across seeds  : {vals.std():.4f}")
    print(f"  min seed-mean     : {vals.min():.4f}")
    print(f"  max seed-mean     : {vals.max():.4f}")
    print("=" * 72)

    output_csv = args.output_csv or os.path.join(
        "artifacts", f"eval_scripted_{args.run_id}_{args.partner}_{args.feed_mode}.csv",
    )
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ego_seed", "partner", "feed_mode", "episode", "return"])
        w.writerows(rows)
    print(f"[eval_scripted_partners] wrote {len(rows)} rows -> {output_csv}", flush=True)

    if args.upload_wandb:
        import wandb
        run = wandb.init(
            project=args.project, entity=args.entity,
            id=args.run_id, resume="must",
        )
        run.log({
            f"ScriptedPartnerEval/{args.partner}/{args.feed_mode}/return_mean": float(vals.mean()),
            f"ScriptedPartnerEval/{args.partner}/{args.feed_mode}/return_std": float(vals.std()),
            f"ScriptedPartnerEval/{args.partner}/{args.feed_mode}/n_seeds": int(len(seed_indices)),
            f"ScriptedPartnerEval/{args.partner}/{args.feed_mode}/n_eps_per_seed": int(args.num_episodes),
        }, commit=True)
        run.finish()
        print(f"[eval_scripted_partners] uploaded summary to wandb run {args.run_id}",
              flush=True)


if __name__ == "__main__":
    main()
