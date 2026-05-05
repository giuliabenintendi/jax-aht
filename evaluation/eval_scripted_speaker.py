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
    """Run one episode with the speaker scripted; return listener's decision pick.

    `script` is an int array of length max_steps - 1 giving the speaker's
    deliberation action at each slot (k = 0..max_steps-2). The speaker's
    decision-step action is sampled greedily (we don't use it).
    Listener acts greedily from policy at every step.
    """
    listener_idx = 1 - speaker_idx
    rng = jax.random.fold_in(reset_rng, 1)
    obs, state = env.reset(reset_rng)
    h_speaker = policy.init_hstate(1)
    h_listener = policy.init_hstate(1)

    listener_pick = -1
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
            listener_pick = a_l  # listener's decision-step action = pick

        env_act = {
            f"agent_{speaker_idx}": jnp.int32(a_s),
            f"agent_{listener_idx}": jnp.int32(a_l),
        }
        obs, state, _r, _d, _info = env.step(k_step, state, env_act)

    return listener_pick


def _eval_constant(env, policy, params, n_episodes, max_steps, base_seed):
    """For each (direction, X), run N episodes with script = [X]*K.

    Returns:
        counts: (2, NUM_CARDS, NUM_CARDS) int -- counts of listener picks.
            counts[direction, X, pick] = how often the listener picked `pick`
            when the speaker (= agent direction) said `X` at every slot.
    """
    K = max_steps - 1
    counts = np.zeros((2, NUM_CARDS, NUM_CARDS), dtype=np.int64)
    for direction in range(2):
        for X in range(NUM_CARDS):
            script = np.full(K, X, dtype=np.int32)
            for ep in range(n_episodes):
                reset_rng = jax.random.PRNGKey(base_seed + ep + X * 1009)
                pick = _run_scripted_episode(
                    env, policy, params, reset_rng,
                    speaker_idx=direction, script=script, max_steps=max_steps,
                )
                counts[direction, X, pick] += 1
    return counts


def _eval_switch(env, policy, params, n_episodes, max_steps, base_seed):
    """For each (direction, X, Y!=X, k in 1..K-1), run N episodes with
    script = [X]*k + [Y]*(K-k) ("stick after switch").

    Returns:
        late: (2, K-1) int -- counts of "listener picked Y" per (direction, k_idx).
        early: (2, K-1) int -- counts of "listener picked X" per (direction, k_idx).
        other: (2, K-1) int -- counts of "listener picked something else".
        total: (2, K-1) int -- total episodes per cell.
        Note: k_idx 0 corresponds to switch slot k=1 (skipping k=0 which is
        equivalent to constant Y).
    """
    K = max_steps - 1
    n_k = K - 1  # k = 1..K-1 -> n_k slots
    late = np.zeros((2, n_k), dtype=np.int64)
    early = np.zeros((2, n_k), dtype=np.int64)
    other = np.zeros((2, n_k), dtype=np.int64)
    total = np.zeros((2, n_k), dtype=np.int64)

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
                        pick = _run_scripted_episode(
                            env, policy, params, reset_rng,
                            speaker_idx=direction, script=script,
                            max_steps=max_steps,
                        )
                        total[direction, k_idx] += 1
                        if pick == Y:
                            late[direction, k_idx] += 1
                        elif pick == X:
                            early[direction, k_idx] += 1
                        else:
                            other[direction, k_idx] += 1
    return late, early, other, total


def _format_constant(counts: np.ndarray) -> str:
    """Format the per-seed (2, 5, 5) constant counts as follow-rate table."""
    totals = counts.sum(axis=2, keepdims=True)
    probs = counts / np.maximum(totals, 1)
    follow_diag = np.array([probs[d, x, x] for d in range(2) for x in range(NUM_CARDS)]).reshape(2, NUM_CARDS)
    header = "    direction  " + "  ".join(f"X={x}" for x in range(NUM_CARDS)) + "    mean"
    rows = []
    for d, name in enumerate(("agent_0->1", "agent_1->0")):
        per_X = "  ".join(f"{follow_diag[d, x]:.3f}" for x in range(NUM_CARDS))
        rows.append(f"    {name}  {per_X}    {follow_diag[d].mean():.3f}")
    return "\n".join((header, *rows))


def _format_switch(late: np.ndarray, total: np.ndarray) -> str:
    """Format follow-late rate per (direction, k)."""
    rate = late / np.maximum(total, 1)
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
        f"\n  Listener: greedy. Chance follow rate (constant) = {1.0/NUM_CARDS:.3f}."
    )

    do_constant = args.experiment in ("constant", "both")
    do_switch = args.experiment in ("switch", "both")

    constant_per_seed = []  # list of (2, 5, 5) count arrays
    switch_per_seed = []    # list of dicts of (2, K-1) count arrays

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)
        base_seed = 8_000 + seed_idx * 100_000

        if do_constant:
            print(f"\n[Seed {seed_idx}] Running constant experiment "
                  f"(2 dirs x {NUM_CARDS} X-values x {args.num_episodes} eps)...")
            const_counts = _eval_constant(
                inner_env, policy, params, args.num_episodes, max_steps, base_seed,
            )
            constant_per_seed.append(const_counts)
            print(f"\nSeed {seed_idx}  Constant follow rate "
                  f"(P[listener pick = X | speaker says X], chance={1.0/NUM_CARDS:.2f}):")
            print(_format_constant(const_counts))

        if do_switch:
            n_pairs = NUM_CARDS * (NUM_CARDS - 1)
            n_k_slots = max_steps - 2
            n_eps_total = 2 * n_pairs * n_k_slots * args.num_episodes
            print(f"\n[Seed {seed_idx}] Running switch experiment "
                  f"({n_eps_total} episodes total)...")
            late, early, other, total = _eval_switch(
                inner_env, policy, params, args.num_episodes, max_steps, base_seed + 5_000,
            )
            switch_per_seed.append(
                {"late": late, "early": early, "other": other, "total": total}
            )
            print(f"\nSeed {seed_idx}  Switch follow-late rate "
                  f"(P[listener pick = Y | speaker switched X->Y at slot k]):")
            print(_format_switch(late, total))

    # --- Aggregate across seeds ---
    if do_constant:
        all_const = np.stack(constant_per_seed, axis=0)  # (S, 2, 5, 5)
        per_seed_follow = np.zeros((num_seeds, 2, NUM_CARDS), dtype=np.float64)
        for s in range(num_seeds):
            totals = all_const[s].sum(axis=2, keepdims=True)
            probs = all_const[s] / np.maximum(totals, 1)
            per_seed_follow[s] = np.array(
                [[probs[d, x, x] for x in range(NUM_CARDS)] for d in range(2)],
            )
        mean_follow = per_seed_follow.mean(axis=0)
        sem_follow = (
            per_seed_follow.std(axis=0, ddof=1) / math.sqrt(num_seeds)
            if num_seeds > 1 else np.zeros_like(mean_follow)
        )
        print(f"\nAggregate Constant follow rate across {num_seeds} seeds (mean ± SEM):")
        for d, name in enumerate(("agent_0->1", "agent_1->0")):
            parts = [
                f"{mean_follow[d, x]:.3f}±{sem_follow[d, x]:.3f}"
                for x in range(NUM_CARDS)
            ]
            print(f"  {name}: " + "  ".join(parts)
                  + f"    mean={mean_follow[d].mean():.3f}")

    if do_switch:
        n_k = max_steps - 2
        per_seed_late = np.zeros((num_seeds, 2, n_k), dtype=np.float64)
        for s, dat in enumerate(switch_per_seed):
            per_seed_late[s] = dat["late"] / np.maximum(dat["total"], 1)
        mean_late = per_seed_late.mean(axis=0)
        sem_late = (
            per_seed_late.std(axis=0, ddof=1) / math.sqrt(num_seeds)
            if num_seeds > 1 else np.zeros_like(mean_late)
        )
        print(f"\nAggregate Switch follow-late rate across {num_seeds} seeds (mean ± SEM):")
        for d, name in enumerate(("0->1", "1->0")):
            parts = [
                f"{mean_late[d, k]:.3f}±{sem_late[d, k]:.3f}"
                for k in range(n_k)
            ]
            print(f"  {name}: " + "  ".join(parts))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = label.replace("/", "_").replace(" ", "_")
        if do_constant:
            np.save(out / f"scripted_constant_{slug}.npy", np.stack(constant_per_seed))
            print(f"\nSaved {out / f'scripted_constant_{slug}.npy'}")
        if do_switch:
            late_arr = np.stack([d["late"] for d in switch_per_seed])
            early_arr = np.stack([d["early"] for d in switch_per_seed])
            total_arr = np.stack([d["total"] for d in switch_per_seed])
            np.savez(
                out / f"scripted_switch_{slug}.npz",
                late=late_arr, early=early_arr, total=total_arr,
            )
            print(f"Saved {out / f'scripted_switch_{slug}.npz'}")


if __name__ == "__main__":
    main()
