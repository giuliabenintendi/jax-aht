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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=1024)
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to <run_dir>/probe_constant_partner/")
    parser.add_argument("--use-best", action="store_true")
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

    matrix_mean = np.zeros((NUM_CARDS, num_seeds), dtype=np.float32)
    matrix_std = np.zeros((NUM_CARDS, num_seeds), dtype=np.float32)
    base_rng = jax.random.PRNGKey(0)
    for s in range(num_seeds):
        seed_params = jax.tree.map(lambda x: x[s], stacked_params)
        for t in range(NUM_CARDS):
            cell_rng = jax.random.fold_in(base_rng, s * NUM_CARDS + t)
            rewards = cell_fn(cell_rng, seed_params, jnp.int32(t))
            rewards_np = np.asarray(rewards)
            matrix_mean[t, s] = float(rewards_np.mean())
            matrix_std[t, s] = float(rewards_np.std())
            print(f"  seed {s:>2d} | T={t}: mean reward = {matrix_mean[t, s]:.3f} "
                  f"(± {matrix_std[t, s]:.3f})")

    png_path = out_dir / "probe_constant_partner_matrix.png"
    csv_path = out_dir / "probe_constant_partner_matrix.csv"
    _save_matrix_png(
        matrix_mean, matrix_std, NUM_CARDS, num_seeds,
        title="Decision-success vs constant-T partner (no OP, shuffle on)",
        filepath=str(png_path),
    )
    _save_matrix_csv(matrix_mean, matrix_std, num_seeds, str(csv_path))
    print(f"\nDone. Matrix saved to {png_path}")


def _save_matrix_png(matrix_mean, matrix_std, n_rows: int, n_cols: int,
                      title: str, filepath: str):
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
    ax.set_yticklabels([f"T={i}" for i in range(n_rows)], fontsize=9)
    ax.set_xlabel("Focal seed (A0)")
    ax.set_ylabel("Scripted partner emits T")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"[probe] heatmap saved: {filepath}")


def _save_matrix_csv(matrix_mean, matrix_std, n_cols: int, filepath: str):
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        header = ["T \\ seed"] + [f"seed_{j}" for j in range(n_cols)]
        w.writerow([h + "_mean" for h in header[:1]] + header[1:])
        for i in range(matrix_mean.shape[0]):
            w.writerow([f"T={i}"] + [f"{matrix_mean[i, j]:.4f}" for j in range(n_cols)])
        w.writerow([])
        w.writerow([h + "_std" for h in header[:1]] + header[1:])
        for i in range(matrix_std.shape[0]):
            w.writerow([f"T={i}"] + [f"{matrix_std[i, j]:.4f}" for j in range(n_cols)])
    print(f"[probe] CSV saved: {filepath}")


if __name__ == "__main__":
    main()
