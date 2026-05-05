"""Scripted-speaker behavioral probe for card-game checkpoints.

Replaces one agent (the "speaker") with a deterministic action script while
the other agent (the "listener") plays from the trained policy. Tests how
the listener's eventual pick depends on what the speaker emits, under
in-distribution-like message trajectories.

Two experiments:
  - constant: speaker emits X at every deliberation slot (X in {0..NUM_CARDS-1}).
    Headline: P(listener pick = X | speaker says X) -- "follow rate". Chance
    is 1 / NUM_CARDS (= 0.20 for 5 cards).
  - switch:   speaker emits X for slots 0..k-1 then Y for slots k..K-1
              (X != Y, k in {1..K-1}; "stick after switch" semantics).
    Headline: P(listener pick = Y) per k -- "follow-late rate". A listener
    that perfectly follows the speaker's most recent committed message has
    rate ~ 1.0 across all k; one that locks in early has rate decaying
    as k grows (because Y is observed at fewer steps before the decision).

Timing reminder: a message emitted by the speaker at slot k is rendered
into the listener's observation at step k+1. So a switch at slot k means
the listener first perceives Y at step k+1; for k = K-1 the listener sees
Y only at the decision step itself (one obs of exposure).

Eval-time `scramble_partner_msg` is forced False so the listener sees
exactly the scripted message dot.

Usage:
    ./run_gpu.sh 0 evaluation.eval_scripted_speaker \\
        --checkpoint <path_to_saved_train_run> \\
        --experiment both \\
        --num-episodes 128
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
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


def _get_obs_type(alg_config):
    return alg_config.get(
        "OBS_TYPE",
        alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"),
    )


def _greedy_action(policy, params, obs_dict, agent_id, hstate, avail, rng):
    """Forward `policy.get_action` greedily for a single agent."""
    obs = obs_dict[f"agent_{agent_id}"].reshape(1, 1, -1)
    done = jnp.zeros((1, 1), dtype=bool)
    avail_a = avail[f"agent_{agent_id}"].astype(jnp.float32)
    action, new_hstate = policy.get_action(
        params=params,
        obs=obs,
        done=done,
        avail_actions=avail_a,
        hstate=hstate,
        rng=rng,
        greedy=True,
    )
    return int(action.squeeze()), new_hstate


def _run_scripted_episode(
    env, policy, params, reset_rng, speaker_idx, script, max_steps,
):
    """Run one episode with the speaker scripted; return coordination signal.

    `script` is an int array of length max_steps - 1 giving the speaker's
    deliberation action at each slot (k = 0..max_steps-2). The speaker's
    decision-step action is sampled greedily (we don't use the value, just
    let the speaker pick from policy). Listener acts greedily throughout.

    Returns (coord, listener_pick) where:
      coord: 1 if both agents coordinated on the same canonical color at
        the decision step (i.e. env's `base_reward` is positive), 0 otherwise.
      listener_pick: listener's decision-step action in the listener's
        recoloured frame (informational; not directly comparable to script
        when OP recolouring is on).
    """
    listener_idx = 1 - speaker_idx
    rng = jax.random.fold_in(reset_rng, 1)
    obs, state = env.reset(reset_rng)
    h_speaker = policy.init_hstate(1)
    h_listener = policy.init_hstate(1)

    listener_pick = -1
    coord = 0
    for c in range(max_steps):
        avail = env.get_avail_actions(state)
        rng, k_s, k_l, k_step = jax.random.split(rng, 4)

        a_s_sampled, h_speaker = _greedy_action(
            policy, params, obs, speaker_idx, h_speaker, avail, k_s,
        )
        a_l, h_listener = _greedy_action(
            policy, params, obs, listener_idx, h_listener, avail, k_l,
        )

        if c < max_steps - 1:
            a_s = int(script[c])
        else:
            a_s = a_s_sampled
            listener_pick = a_l  # listener's decision-step action

        env_act = {
            f"agent_{speaker_idx}": jnp.int32(a_s),
            f"agent_{listener_idx}": jnp.int32(a_l),
        }
        obs, state, reward, _d, info = env.step(k_step, state, env_act)
        if c == max_steps - 1:
            # Pull the env's base reward (canonical-frame coordination signal)
            # rather than the shaped/comm reward; same value for both agents.
            coord = int(float(info["base_reward"][0]) > 0)

    return coord, listener_pick


def _eval_constant(env, policy, params, n_episodes, max_steps, base_seed):
    """For each (direction, X), run N episodes with script = [X]*K.

    Returns:
        coord_count: (2, NUM_CARDS) int -- # episodes where canonical
            coordination was achieved when the speaker said X at every slot.
        n_total: (2, NUM_CARDS) int -- # episodes per cell (= n_episodes).
    """
    K = max_steps - 1
    coord_count = np.zeros((2, NUM_CARDS), dtype=np.int64)
    n_total = np.zeros((2, NUM_CARDS), dtype=np.int64)
    for direction in range(2):
        for X in range(NUM_CARDS):
            script = np.full(K, X, dtype=np.int32)
            for ep in range(n_episodes):
                reset_rng = jax.random.PRNGKey(base_seed + ep + X * 1009)
                coord, _ = _run_scripted_episode(
                    env, policy, params, reset_rng,
                    speaker_idx=direction, script=script, max_steps=max_steps,
                )
                coord_count[direction, X] += coord
                n_total[direction, X] += 1
    return coord_count, n_total


def _eval_switch(env, policy, params, n_episodes, max_steps, base_seed):
    """For each (direction, X, Y!=X, k in 1..K-1), run N episodes with
    script = [X]*k + [Y]*(K-k) ("stick after switch").

    Returns:
        coord_count: (2, K-1) int -- # episodes where canonical coordination
            was achieved per (direction, k_idx). Aggregated across all
            (X, Y!=X) pairs.
        n_total: (2, K-1) int -- total episodes per cell.

    Note: k_idx 0 corresponds to switch slot k=1 (skipping k=0 which is
        equivalent to constant Y).
    """
    K = max_steps - 1
    n_k = K - 1  # k = 1..K-1 -> n_k slots
    coord_count = np.zeros((2, n_k), dtype=np.int64)
    n_total = np.zeros((2, n_k), dtype=np.int64)

    for direction in range(2):
        for X in range(NUM_CARDS):
            for Y in range(NUM_CARDS):
                if Y == X:
                    continue
                for k_idx, k in enumerate(range(1, K)):
                    script = np.concatenate([
                        np.full(k, X, dtype=np.int32),
                        np.full(K - k, Y, dtype=np.int32),
                    ])
                    for ep in range(n_episodes):
                        reset_rng = jax.random.PRNGKey(
                            base_seed + ep + X * 31 + Y * 113 + k * 1097,
                        )
                        coord, _ = _run_scripted_episode(
                            env, policy, params, reset_rng,
                            speaker_idx=direction, script=script,
                            max_steps=max_steps,
                        )
                        coord_count[direction, k_idx] += coord
                        n_total[direction, k_idx] += 1
    return coord_count, n_total


def _format_constant(coord: np.ndarray, total: np.ndarray) -> str:
    """Format the per-seed (2, NUM_CARDS) coordination-rate table."""
    rate = coord / np.maximum(total, 1)
    header = "    direction  " + "  ".join(f"X={x}" for x in range(NUM_CARDS)) + "    mean"
    rows = []
    for d, name in enumerate(("agent_0->1", "agent_1->0")):
        per_X = "  ".join(f"{rate[d, x]:.3f}" for x in range(NUM_CARDS))
        rows.append(f"    {name}  {per_X}    {rate[d].mean():.3f}")
    return "\n".join((header, *rows))


def _format_switch(coord: np.ndarray, total: np.ndarray) -> str:
    """Format coordination rate per (direction, k)."""
    rate = coord / np.maximum(total, 1)
    n_k = rate.shape[1]
    header = "    k:    " + "  ".join(f"k={k+1}" for k in range(n_k))
    rows = []
    for d, name in enumerate(("0->1 ", "1->0 ")):
        per_k = "  ".join(f"{rate[d, ki]:.3f}" for ki in range(n_k))
        rows.append(f"    {name}  {per_k}")
    return "\n".join((header, *rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=128)
    parser.add_argument(
        "--experiment", choices=("constant", "switch", "both"), default="both",
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
            "Checkpoint was trained without communication; scripted-speaker"
            " probe is undefined."
        )
    env_kwargs["scramble_partner_msg"] = False

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

    label = cfg.get("label", "(unlabeled)")
    print(
        f"\nCheckpoint: {args.checkpoint}"
        f"\n  label={label}  comm=on  scramble_eval=False"
        f"  seeds={num_seeds}  episodes/condition={args.num_episodes}"
        f"  max_steps={max_steps}"
        f"  experiment={args.experiment}"
        f"\n  Listener: greedy. Metric: P(canonical coordination) per condition."
        f"\n  Chance coordination = {1.0/NUM_CARDS:.3f} (1 / NUM_CARDS)."
    )

    do_constant = args.experiment in ("constant", "both")
    do_switch = args.experiment in ("switch", "both")

    constant_per_seed = []  # list of (coord_count[2, 5], n_total[2, 5]) tuples
    switch_per_seed = []    # list of (coord_count[2, K-1], n_total[2, K-1]) tuples

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)
        base_seed = 8_000 + seed_idx * 100_000

        if do_constant:
            print(f"\n[Seed {seed_idx}] Running constant experiment "
                  f"(2 dirs x {NUM_CARDS} X-values x {args.num_episodes} eps)...")
            coord_count, n_total = _eval_constant(
                inner_env, policy, params, args.num_episodes, max_steps, base_seed,
            )
            constant_per_seed.append((coord_count, n_total))
            print(f"\nSeed {seed_idx}  Constant coordination rate "
                  f"(P[both agents picked same canonical color | speaker says X], "
                  f"chance={1.0/NUM_CARDS:.2f}):")
            print(_format_constant(coord_count, n_total))

        if do_switch:
            n_pairs = NUM_CARDS * (NUM_CARDS - 1)
            n_k_slots = max_steps - 2
            n_eps_total = 2 * n_pairs * n_k_slots * args.num_episodes
            print(f"\n[Seed {seed_idx}] Running switch experiment "
                  f"({n_eps_total} episodes total)...")
            coord_count, n_total = _eval_switch(
                inner_env, policy, params, args.num_episodes, max_steps, base_seed + 5_000,
            )
            switch_per_seed.append((coord_count, n_total))
            print(f"\nSeed {seed_idx}  Switch coordination rate "
                  f"(P[canonical coordination | speaker switched X->Y at slot k]):")
            print(_format_switch(coord_count, n_total))

    # --- Aggregate across seeds ---
    if do_constant:
        per_seed_rate = np.zeros((num_seeds, 2, NUM_CARDS), dtype=np.float64)
        for s, (coord, total) in enumerate(constant_per_seed):
            per_seed_rate[s] = coord / np.maximum(total, 1)
        mean_rate = per_seed_rate.mean(axis=0)
        sem_rate = (
            per_seed_rate.std(axis=0, ddof=1) / math.sqrt(num_seeds)
            if num_seeds > 1 else np.zeros_like(mean_rate)
        )
        print(f"\nAggregate Constant coordination rate across {num_seeds} seeds (mean ± SEM):")
        for d, name in enumerate(("agent_0->1", "agent_1->0")):
            parts = [
                f"{mean_rate[d, x]:.3f}±{sem_rate[d, x]:.3f}"
                for x in range(NUM_CARDS)
            ]
            print(f"  {name}: " + "  ".join(parts)
                  + f"    mean={mean_rate[d].mean():.3f}")

    if do_switch:
        n_k = max_steps - 2
        per_seed_rate = np.zeros((num_seeds, 2, n_k), dtype=np.float64)
        for s, (coord, total) in enumerate(switch_per_seed):
            per_seed_rate[s] = coord / np.maximum(total, 1)
        mean_rate = per_seed_rate.mean(axis=0)
        sem_rate = (
            per_seed_rate.std(axis=0, ddof=1) / math.sqrt(num_seeds)
            if num_seeds > 1 else np.zeros_like(mean_rate)
        )
        print(f"\nAggregate Switch coordination rate across {num_seeds} seeds (mean ± SEM):")
        for d, name in enumerate(("0->1", "1->0")):
            parts = [
                f"{mean_rate[d, k]:.3f}±{sem_rate[d, k]:.3f}"
                for k in range(n_k)
            ]
            print(f"  {name}: " + "  ".join(parts))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = label.replace("/", "_").replace(" ", "_")
        if do_constant:
            coord_arr = np.stack([c for c, _ in constant_per_seed])
            total_arr = np.stack([t for _, t in constant_per_seed])
            np.savez(
                out / f"scripted_constant_{slug}.npz",
                coord=coord_arr, total=total_arr,
            )
            print(f"\nSaved {out / f'scripted_constant_{slug}.npz'}")
        if do_switch:
            coord_arr = np.stack([c for c, _ in switch_per_seed])
            total_arr = np.stack([t for _, t in switch_per_seed])
            np.savez(
                out / f"scripted_switch_{slug}.npz",
                coord=coord_arr, total=total_arr,
            )
            print(f"Saved {out / f'scripted_switch_{slug}.npz'}")


if __name__ == "__main__":
    main()
