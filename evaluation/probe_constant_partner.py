"""Probe: pair each focal seed with a scripted constant-token partner.

For each (focal seed S, constant token T) cell, run N episodes with seed S as
agent 0 and a scripted agent that emits action T at every step (deliberation
messages and the decision pick) as agent 1. Cell value = mean per-episode
decision-success reward (= fraction of episodes where both picks matched in
GT frame).

Runs without OP wrappers (other_play_position_shuffle=false,
other_play_recolouring=false) so the scripted "always T" partner directly
shows seed S a constant message of GT colour T. The env's internal
position-shuffle is preserved, so card *positions* still vary per episode.

Usage:
    uv run python -m evaluation.probe_constant_partner \\
        --checkpoint results/.../saved_train_run \\
        --num-episodes 1024 \\
        --output-dir results/.../probe_constant_partner
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agents.initialize_agents import (
    initialize_ja_agent,
    initialize_ja_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import NUM_CARDS
from envs.log_wrapper import LogWrapper


def _run_probe_episode(rng, env, focal_params, focal_policy,
                        scripted_token, max_episode_steps, action_sizes):
    """Run one probe episode (focal seed as A0, ConstantAgent(T) as A1).

    Returns success indicator: 1.0 if focal's decision pick == T, else 0.0.
    Uses GT-frame comparison via env reward (no OP wrappers, so GT == raw).
    """
    rng, reset_rng = jax.random.split(rng)
    init_obs, init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1,), dtype=bool) for k in env.agents + ["__all__"]}

    hstate_0 = focal_policy.init_hstate(1, aux_info={"agent_id": 0})

    avail_actions = env.get_avail_actions(init_env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    avail_0 = avail_actions["agent_0"].astype(jnp.float32)

    rng, act0_rng, step_rng = jax.random.split(rng, 3)
    act_0, hstate_0 = focal_policy.get_action(
        params=focal_params,
        obs=init_obs["agent_0"].reshape(1, 1, -1),
        done=init_done["agent_0"].reshape(1, 1),
        avail_actions=avail_0,
        hstate=hstate_0,
        rng=act0_rng,
        greedy=True,
    )
    act_0 = act_0.squeeze()
    act_1 = jnp.int32(scripted_token)

    env_act = {"agent_0": act_0, "agent_1": act_1}
    obs, env_state, reward, done, info = env.step(step_rng, init_env_state, env_act)

    def scan_step(carry, _):
        ep_ts, env_state, obs, hstate_0, done, last_info, rng = carry

        def take_step(c):
            ep_ts, env_state, obs, hstate_0, done, last_info, rng = c
            avail = env.get_avail_actions(env_state)
            avail = jax.lax.stop_gradient(avail)
            avail_0 = avail["agent_0"].astype(jnp.float32)

            rng, act0_rng, step_rng = jax.random.split(rng, 3)
            act_0, hstate_0_next = focal_policy.get_action(
                params=focal_params,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=done["agent_0"].reshape(1, 1),
                avail_actions=avail_0,
                hstate=hstate_0,
                rng=act0_rng,
                greedy=True,
            )
            act_0 = act_0.squeeze()
            act_1 = jnp.int32(scripted_token)
            env_act = {"agent_0": act_0, "agent_1": act_1}
            obs_next, env_state_next, _r, done_next, info_next = env.step(
                step_rng, env_state, env_act,
            )
            return (ep_ts + 1, env_state_next, obs_next, hstate_0_next,
                    done_next, info_next, rng)

        new_carry = jax.lax.cond(
            carry[4]["__all__"],
            lambda c: c,
            take_step,
            operand=carry,
        )
        return new_carry, None

    init_carry = (1, env_state, obs, hstate_0, done, info, rng)
    final_carry, _ = jax.lax.scan(
        scan_step, init_carry, None, length=max_episode_steps,
    )
    final_info = final_carry[5]
    # base_return is the binary decision-success indicator (1.0 if picks
    # matched at decision step, 0.0 otherwise) in GT colour space.
    return final_info["base_return"][0]


def _run_probe_cell(rng, env, focal_params, focal_policy, scripted_token,
                    max_episode_steps, num_eps, action_sizes):
    """Vmap _run_probe_episode over num_eps episodes; return per-episode rewards."""
    rngs = jax.random.split(rng, num_eps)
    vmap_fn = jax.vmap(
        lambda r: _run_probe_episode(
            r, env, focal_params, focal_policy, scripted_token,
            max_episode_steps, action_sizes,
        )
    )
    return vmap_fn(rngs)


def _sample_from_mask(rng, allowed_mask):
    """Sample an action uniformly from indices where allowed_mask is True."""
    probs = allowed_mask.astype(jnp.float32)
    probs = probs / probs.sum()
    return jax.random.choice(rng, NUM_CARDS, p=probs).astype(jnp.int32)


def _run_probe_masked_episode(rng, env, focal_params, focal_policy,
                               allowed_mask, max_episode_steps, action_sizes):
    """One probe episode where the scripted partner samples uniformly from
    `allowed_mask` (a (NUM_CARDS,) bool array) at every step. Returns the
    binary decision-success indicator from the env."""
    rng, reset_rng = jax.random.split(rng)
    init_obs, init_env_state = env.reset(reset_rng)
    init_done = {k: jnp.zeros((1,), dtype=bool) for k in env.agents + ["__all__"]}

    hstate_0 = focal_policy.init_hstate(1, aux_info={"agent_id": 0})

    avail_actions = env.get_avail_actions(init_env_state)
    avail_actions = jax.lax.stop_gradient(avail_actions)
    avail_0 = avail_actions["agent_0"].astype(jnp.float32)

    rng, act0_rng, sample_rng, step_rng = jax.random.split(rng, 4)
    act_0, hstate_0 = focal_policy.get_action(
        params=focal_params,
        obs=init_obs["agent_0"].reshape(1, 1, -1),
        done=init_done["agent_0"].reshape(1, 1),
        avail_actions=avail_0,
        hstate=hstate_0,
        rng=act0_rng,
        greedy=True,
    )
    act_0 = act_0.squeeze()
    act_1 = _sample_from_mask(sample_rng, allowed_mask)
    env_act = {"agent_0": act_0, "agent_1": act_1}
    obs, env_state, reward, done, info = env.step(step_rng, init_env_state, env_act)

    def scan_step(carry, _):
        ep_ts, env_state, obs, hstate_0, done, last_info, rng = carry

        def take_step(c):
            ep_ts, env_state, obs, hstate_0, done, last_info, rng = c
            avail = env.get_avail_actions(env_state)
            avail = jax.lax.stop_gradient(avail)
            avail_0 = avail["agent_0"].astype(jnp.float32)

            rng, act0_rng, sample_rng, step_rng = jax.random.split(rng, 4)
            act_0, hstate_0_next = focal_policy.get_action(
                params=focal_params,
                obs=obs["agent_0"].reshape(1, 1, -1),
                done=done["agent_0"].reshape(1, 1),
                avail_actions=avail_0,
                hstate=hstate_0,
                rng=act0_rng,
                greedy=True,
            )
            act_0 = act_0.squeeze()
            act_1 = _sample_from_mask(sample_rng, allowed_mask)
            env_act = {"agent_0": act_0, "agent_1": act_1}
            obs_next, env_state_next, _r, done_next, info_next = env.step(
                step_rng, env_state, env_act,
            )
            return (ep_ts + 1, env_state_next, obs_next, hstate_0_next,
                    done_next, info_next, rng)

        new_carry = jax.lax.cond(
            carry[4]["__all__"],
            lambda c: c,
            take_step,
            operand=carry,
        )
        return new_carry, None

    init_carry = (1, env_state, obs, hstate_0, done, info, rng)
    final_carry, _ = jax.lax.scan(
        scan_step, init_carry, None, length=max_episode_steps,
    )
    final_info = final_carry[5]
    return final_info["base_return"][0]


def _run_probe_masked_cell(rng, env, focal_params, focal_policy, allowed_mask,
                            max_episode_steps, num_eps, action_sizes):
    rngs = jax.random.split(rng, num_eps)
    vmap_fn = jax.vmap(
        lambda r: _run_probe_masked_episode(
            r, env, focal_params, focal_policy, allowed_mask,
            max_episode_steps, action_sizes,
        )
    )
    return vmap_fn(rngs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=1024)
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to <run_dir>/probe_constant_partner/")
    parser.add_argument("--use-best", action="store_true")
    parser.add_argument("--vocab-threshold", type=float, default=0.5,
                        help="Reward threshold for inclusion in a seed's vocabulary "
                             "(used to derive the non-vocab mask for the masked-partner pass).")
    parser.add_argument("--skip-masked", action="store_true",
                        help="Skip the masked-partner pass; only run the constant-partner matrix.")
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

    # Force OP off; preserve the env's internal shuffle (cards in random
    # positions per episode but both agents see the same view).
    env_kwargs = dict(algo_cfg["ENV_KWARGS"])
    env_kwargs["other_play_position_shuffle"] = False
    env_kwargs["other_play_recolouring"] = False
    env_kwargs["shuffle"] = True
    if algo_cfg.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    env_kwargs.setdefault("scramble_partner_msg", False)

    env = make_env(algo_cfg["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    obs_type = algo_cfg.get("ENV_KWARGS", {}).get("obs_type", "image")
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent
    policy, _ = init_fn(algo_cfg, env, jax.random.PRNGKey(0))
    inner_env = env._env
    max_steps = int(algo_cfg.get(
        "ROLLOUT_LENGTH", algo_cfg.get("ENV_KWARGS", {}).get("max_steps", 8),
    ))

    action_sizes = {
        agent: int(env.action_space(agent).n) for agent in env.agents
    }

    out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "probe_constant_partner")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[probe] checkpoint: {ckpt_path}\n"
        f"[probe] {num_seeds} seeds × {NUM_CARDS} fixed tokens × {args.num_episodes} eps\n"
        f"[probe] env: shuffle=True, OP off, communication={algo_cfg.get('COMMUNICATION', False)}\n"
        f"[probe] output: {out_dir}"
    )

    cell_fn = jax.jit(
        lambda rng_c, p, t: _run_probe_cell(
            rng_c, inner_env, p, policy, t,
            max_steps, args.num_episodes, action_sizes,
        ),
        static_argnums=(),
    )

    # Matrix layout: rows = card identity the scripted partner emits,
    # cols = seed. Cell = mean decision-success over eps.
    matrix_mean = np.zeros((NUM_CARDS, num_seeds), dtype=np.float32)
    matrix_std = np.zeros((NUM_CARDS, num_seeds), dtype=np.float32)
    base_rng = jax.random.PRNGKey(0)
    for s in range(num_seeds):
        seed_params = jax.tree.map(lambda x: x[s], stacked_params)
        for c in range(NUM_CARDS):
            cell_rng = jax.random.fold_in(base_rng, s * NUM_CARDS + c)
            rewards = cell_fn(cell_rng, seed_params, jnp.int32(c))
            rewards_np = np.asarray(rewards)
            matrix_mean[c, s] = float(rewards_np.mean())
            matrix_std[c, s] = float(rewards_np.std())
            print(f"  seed {s:>2d} | card={c}: mean reward = {matrix_mean[c, s]:.3f} "
                  f"(± {matrix_std[c, s]:.3f})")

    png_path = out_dir / "probe_constant_partner_matrix.png"
    csv_path = out_dir / "probe_constant_partner_matrix.csv"
    _save_matrix_png(
        matrix_mean, matrix_std, NUM_CARDS, num_seeds,
        filepath=str(png_path),
    )
    _save_matrix_csv(matrix_mean, matrix_std, num_seeds, str(csv_path))
    print(f"\nMatrix saved to {png_path}")

    if args.skip_masked:
        return

    # Masked-partner pass: per seed, partner samples each step uniformly from
    # cards NOT in the seed's vocabulary (vocab = cards where the constant-T
    # cell exceeded --vocab-threshold). Tests how the seed responds when
    # forced to coordinate on cards it can't pick at decision step.
    print(f"\n[probe] masked-partner pass (vocab threshold={args.vocab_threshold})")
    in_vocab = matrix_mean > args.vocab_threshold  # (NUM_CARDS, num_seeds)
    masked_cell_fn = jax.jit(
        lambda rng_c, p, mask: _run_probe_masked_cell(
            rng_c, inner_env, p, policy, mask,
            max_steps, args.num_episodes, action_sizes,
        ),
        static_argnums=(),
    )

    masked_mean = np.full(num_seeds, np.nan, dtype=np.float32)
    masked_std = np.full(num_seeds, np.nan, dtype=np.float32)
    masked_vocab_size = np.zeros(num_seeds, dtype=np.int32)
    for s in range(num_seeds):
        seed_vocab = in_vocab[:, s]
        non_vocab_count = int((~seed_vocab).sum())
        masked_vocab_size[s] = int(seed_vocab.sum())
        if non_vocab_count == 0:
            print(f"  seed {s:>2d}: full vocab (no non-vocab cards) → skipping")
            continue
        allowed = jnp.asarray(~seed_vocab, dtype=jnp.bool_)
        seed_params = jax.tree.map(lambda x: x[s], stacked_params)
        cell_rng = jax.random.fold_in(base_rng, num_seeds * NUM_CARDS + s)
        rewards = masked_cell_fn(cell_rng, seed_params, allowed)
        rewards_np = np.asarray(rewards)
        masked_mean[s] = float(rewards_np.mean())
        masked_std[s] = float(rewards_np.std())
        non_vocab_cards = [int(c) for c in range(NUM_CARDS) if not seed_vocab[c]]
        print(f"  seed {s:>2d} | vocab={[int(c) for c in range(NUM_CARDS) if seed_vocab[c]]} "
              f"| partner samples from {non_vocab_cards}: "
              f"reward = {masked_mean[s]:.3f} (± {masked_std[s]:.3f})")

    masked_png = out_dir / "probe_masked_partner.png"
    masked_csv = out_dir / "probe_masked_partner.csv"
    _save_masked_png(masked_mean, masked_std, masked_vocab_size, str(masked_png))
    _save_masked_csv(masked_mean, masked_std, masked_vocab_size, str(masked_csv))
    print(f"\nMasked-partner result saved to {masked_png}")


def _save_matrix_png(matrix_mean, matrix_std, n_rows: int, n_cols: int,
                      filepath: str):
    fig, ax = plt.subplots(figsize=(1.5 + n_cols * 0.9, 1.0 + n_rows * 0.9))
    im = ax.imshow(matrix_mean, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(n_rows):
        for j in range(n_cols):
            m, s = matrix_mean[i, j], matrix_std[i, j]
            text = f"{m:.2f}\n±{s:.2f}"
            color = "white" if m > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color=color)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels([f"seed_{j}" for j in range(n_cols)], fontsize=9, rotation=0)
    ax.set_yticklabels([f"card_{i}" for i in range(n_rows)], fontsize=9)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Card emitted by scripted partner")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[probe] heatmap saved: {filepath}")


def _save_matrix_csv(matrix_mean, matrix_std, n_cols: int, filepath: str):
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        header = ["card \\ seed"] + [f"seed_{j}" for j in range(n_cols)]
        w.writerow([header[0] + "_mean"] + header[1:])
        for i in range(matrix_mean.shape[0]):
            w.writerow([f"card_{i}"] + [f"{matrix_mean[i, j]:.4f}" for j in range(n_cols)])
        w.writerow([])
        w.writerow([header[0] + "_std"] + header[1:])
        for i in range(matrix_std.shape[0]):
            w.writerow([f"card_{i}"] + [f"{matrix_std[i, j]:.4f}" for j in range(n_cols)])
    print(f"[probe] CSV saved: {filepath}")


def _save_masked_png(reward_mean, reward_std, vocab_size, filepath: str):
    n = reward_mean.shape[0]
    fig, ax = plt.subplots(figsize=(1.5 + n * 0.6, 3.5))
    xs = np.arange(n)
    valid = ~np.isnan(reward_mean)
    bars = ax.bar(xs, np.where(valid, reward_mean, 0.0),
                   yerr=np.where(valid, reward_std, 0.0),
                   color="steelblue", edgecolor="black", linewidth=0.5,
                   capsize=2)
    for i, (m, v) in enumerate(zip(reward_mean, vocab_size)):
        if np.isnan(m):
            ax.text(i, 0.02, "full\nvocab", ha="center", va="bottom",
                    fontsize=7, color="gray")
        else:
            ax.text(i, max(m, 0) + 0.02, f"{m:.2f}", ha="center", va="bottom",
                    fontsize=8)
            ax.text(i, -0.06, f"|V|={v}", ha="center", va="top",
                    fontsize=7, color="gray")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"s{i}" for i in range(n)], fontsize=9)
    ax.set_ylabel("Mean decision-success reward")
    ax.set_xlabel("Seed")
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[probe] masked bar chart saved: {filepath}")


def _save_masked_csv(reward_mean, reward_std, vocab_size, filepath: str):
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "vocab_size", "reward_mean", "reward_std"])
        for i in range(reward_mean.shape[0]):
            w.writerow([
                f"seed_{i}",
                int(vocab_size[i]),
                f"{reward_mean[i]:.4f}" if not np.isnan(reward_mean[i]) else "NA",
                f"{reward_std[i]:.4f}" if not np.isnan(reward_std[i]) else "NA",
            ])
    print(f"[probe] masked CSV saved: {filepath}")


if __name__ == "__main__":
    main()
