"""Compute Speaker Consistency (SC) for a trained card-game checkpoint.

Per-seed and aggregate SC_i^(k) for each agent `i` and each deliberation
slot `k`, using paired `(M_{i,k}, A_i)` samples from N rollouts. A_i is the
decision-step pick; messages are decoded from the Discrete(10) comm action
space. See `evaluation.comm_metrics` for the underlying MI formula.

Usage:
    ./run_gpu.sh 0 evaluation.eval_comm_sc \
        --checkpoint <path_to_saved_train_run> \
        --num-episodes 512
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_dual_image_agent,
    initialize_ja_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import NUM_CARDS
from envs.log_wrapper import LogWrapper
from evaluation.comm_metrics import compute_sc_slotwise
from evaluation.vis_episodes import run_episode_with_states


def _get_obs_type(alg_config):
    return alg_config.get(
        "OBS_TYPE",
        alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"),
    )


def _collect_rollouts(
    inner_env, policy, params, num_episodes: int, max_steps: int,
    seed_offset: int, greedy: bool,
) -> tuple[list, list]:
    """Run `num_episodes` SP episodes and return `(ep_messages, ep_actions)`."""
    all_msgs: list[list[tuple[int, int]]] = []
    all_acts: list[list[tuple[int, int]]] = []
    for ep in range(num_episodes):
        rng = jax.random.PRNGKey(7_000 + seed_offset * 10_000 + ep)
        ep_states, ep_actions, ep_messages = run_episode_with_states(
            rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=False, greedy=greedy,
        )
        all_msgs.append(ep_messages)
        all_acts.append(ep_actions)
    return all_msgs, all_acts


def _format_sc_grid(sc: np.ndarray) -> str:
    """Pretty-print a (2, K) SC grid in nats with 3 decimals."""
    K = sc.shape[1]
    header = "slot:  " + "  ".join(f"  k={k}" for k in range(K))
    a0 = "agent0 " + "  ".join(f"{v:6.3f}" for v in sc[0])
    a1 = "agent1 " + "  ".join(f"{v:6.3f}" for v in sc[1])
    return "\n".join((header, a0, a1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=512)
    parser.add_argument(
        "--greedy", action="store_true",
        help="Use argmax actions. Default: sample from policy (matches Lowe et al.).",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = os.path.dirname(args.checkpoint)
    cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml")), resolve=True,
    )
    alg_config = cfg["algorithm"]

    env_name = alg_config["ENV_NAME"]
    env_kwargs = dict(alg_config.get("ENV_KWARGS", {}))
    if alg_config.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    if not env_kwargs.get("communication", False):
        raise SystemExit(
            "This checkpoint was trained without communication; SC is undefined."
        )

    env = make_env(env_name, env_kwargs)
    inner_env = env
    env_wrapped = LogWrapper(env)

    obs_type = _get_obs_type(alg_config)
    use_dual = alg_config.get("USE_DUAL_CRITIC", False)
    if obs_type in ("image", "fov"):
        init_fn = initialize_ja_dual_image_agent if use_dual else initialize_ja_image_agent
    else:
        from agents.initialize_agents import initialize_ja_agent
        init_fn = initialize_ja_agent

    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env_wrapped, rng)

    run_data = load_train_run(args.checkpoint)
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    max_steps = int(env_kwargs.get("max_steps", 8))

    scramble = env_kwargs.get("scramble_partner_msg", False)
    label = cfg.get("label", "(unlabeled)")
    print(
        f"\nCheckpoint: {args.checkpoint}"
        f"\n  label={label}  comm=on  scramble={scramble}"
        f"  seeds={num_seeds}  episodes/seed={args.num_episodes}"
        f"  max_steps={max_steps}  greedy={args.greedy}"
    )

    n_messages = NUM_CARDS
    n_picks = NUM_CARDS

    all_sc = []
    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)
        ep_messages, ep_actions = _collect_rollouts(
            inner_env, policy, params, args.num_episodes, max_steps,
            seed_offset=seed_idx, greedy=args.greedy,
        )
        sc = compute_sc_slotwise(ep_messages, ep_actions, n_messages, n_picks)
        all_sc.append(sc)
        print(f"\nSeed {seed_idx}  SC (nats, max={math.log(NUM_CARDS):.3f}):")
        print(_format_sc_grid(sc))

    all_sc_arr = np.stack(all_sc, axis=0)  # (num_seeds, 2, K)
    mean_sc = all_sc_arr.mean(axis=0)
    sem_sc = all_sc_arr.std(axis=0, ddof=1) / math.sqrt(num_seeds) if num_seeds > 1 else np.zeros_like(mean_sc)

    print(f"\nAggregate across {num_seeds} seeds (mean ± SEM):")
    K = mean_sc.shape[1]
    for agent_idx in range(2):
        parts = [
            f"{mean_sc[agent_idx, k]:6.3f}±{sem_sc[agent_idx, k]:.3f}"
            for k in range(K)
        ]
        print(f"  agent{agent_idx}: " + "  ".join(parts))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = label.replace("/", "_").replace(" ", "_")
        np.save(out / f"sc_{slug}.npy", all_sc_arr)
        print(f"\nSaved {out / f'sc_{slug}.npy'}  shape={all_sc_arr.shape}")


if __name__ == "__main__":
    main()
