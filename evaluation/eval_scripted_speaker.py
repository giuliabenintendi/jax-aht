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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """Run one episode with the speaker scripted; return canonical signals.

    `script` is an int array of length max_steps - 1 giving the speaker's
    deliberation action at each slot (k = 0..max_steps-2). The speaker's
    decision-step action is sampled greedily (we don't use the value, just
    let the speaker pick from policy). Listener acts greedily throughout.

    Returns (coord, follow, listener_pick) where:
      coord: 1 if both agents picked the same canonical color at decision
        (env's `base_reward` is positive), else 0.
      follow: 1 if the listener's canonical pick matches the canonical
        version of the LAST scripted message (`script[-1]` mapped through
        speaker's `inv_recolouring`), else 0. For constant scripts the last
        message is just X; for switch scripts it is Y.
      listener_pick: listener's decision-step action in their recoloured
        frame (informational).
    """
    listener_idx = 1 - speaker_idx
    rng = jax.random.fold_in(reset_rng, 1)
    obs, state = env.reset(reset_rng)
    speaker_inv = np.asarray(state.per_agent_inv_recolouring[f"agent_{speaker_idx}"])
    listener_inv = np.asarray(state.per_agent_inv_recolouring[f"agent_{listener_idx}"])
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
            coord = int(float(info["base_reward"][0]) > 0)

    target_canonical = int(speaker_inv[int(script[-1])])
    listener_canonical = int(listener_inv[listener_pick])
    follow = int(listener_canonical == target_canonical)
    return coord, follow, listener_pick


def _eval_constant(env, policy, params, n_episodes, max_steps, base_seed):
    """For each (direction, X), run N episodes with script = [X]*K.

    Returns:
        coord_count: (2, NUM_CARDS) int -- canonical coordination counts.
        follow_count: (2, NUM_CARDS) int -- canonical FOLLOW counts.
        n_total: (2, NUM_CARDS) int -- total episodes per cell.
    """
    K = max_steps - 1
    coord_count = np.zeros((2, NUM_CARDS), dtype=np.int64)
    follow_count = np.zeros((2, NUM_CARDS), dtype=np.int64)
    n_total = np.zeros((2, NUM_CARDS), dtype=np.int64)
    for direction in range(2):
        for X in range(NUM_CARDS):
            script = np.full(K, X, dtype=np.int32)
            for ep in range(n_episodes):
                reset_rng = jax.random.PRNGKey(base_seed + ep + X * 1009)
                coord, follow, _ = _run_scripted_episode(
                    env, policy, params, reset_rng,
                    speaker_idx=direction, script=script, max_steps=max_steps,
                )
                coord_count[direction, X] += coord
                follow_count[direction, X] += follow
                n_total[direction, X] += 1
    return coord_count, follow_count, n_total


def _eval_switch(env, policy, params, n_episodes, max_steps, base_seed):
    """For each (direction, X, Y!=X, k in 1..K-1), run N episodes with
    script = [X]*k + [Y]*(K-k) ("stick after switch").

    Returns:
        coord_count: (2, K-1) int -- canonical coordination counts.
        follow_count: (2, K-1) int -- canonical FOLLOW(Y) counts: listener's
            canonical pick == canonical Y (the LATE message after the switch).
        n_total: (2, K-1) int -- total episodes per cell.

    Note: k_idx 0 corresponds to switch slot k=1 (skipping k=0 which is
        equivalent to constant Y).
    """
    K = max_steps - 1
    n_k = K - 1  # k = 1..K-1 -> n_k slots
    coord_count = np.zeros((2, n_k), dtype=np.int64)
    follow_count = np.zeros((2, n_k), dtype=np.int64)
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
                        coord, follow, _ = _run_scripted_episode(
                            env, policy, params, reset_rng,
                            speaker_idx=direction, script=script,
                            max_steps=max_steps,
                        )
                        coord_count[direction, k_idx] += coord
                        follow_count[direction, k_idx] += follow
                        n_total[direction, k_idx] += 1
    return coord_count, follow_count, n_total


def _plot_constant(follow_rate, coord_rate, output_path, label: str):
    """Bar chart of FOLLOW (and COORD) rate per X, both directions, with chance line.

    Inputs are (num_seeds, 2, NUM_CARDS) arrays. Per-seed scatter is overlaid.
    """
    n_seeds = follow_rate.shape[0]
    chance = 1.0 / NUM_CARDS
    x_positions = np.arange(NUM_CARDS)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    titles = ["agent 0 → 1 (agent 0 scripted)", "agent 1 → 0 (agent 1 scripted)"]
    for d in range(2):
        ax = axes[d]
        mean_f = follow_rate[:, d, :].mean(axis=0)
        sem_f = (follow_rate[:, d, :].std(axis=0, ddof=1) / math.sqrt(n_seeds)
                 if n_seeds > 1 else np.zeros_like(mean_f))
        mean_c = coord_rate[:, d, :].mean(axis=0)
        sem_c = (coord_rate[:, d, :].std(axis=0, ddof=1) / math.sqrt(n_seeds)
                 if n_seeds > 1 else np.zeros_like(mean_c))

        ax.bar(x_positions - width / 2, mean_f, width, yerr=sem_f,
               label="FOLLOW", color="#3a86ff", capsize=3)
        ax.bar(x_positions + width / 2, mean_c, width, yerr=sem_c,
               label="COORD", color="#ffbe0b", capsize=3, alpha=0.85)

        # per-seed scatter for FOLLOW
        for s in range(n_seeds):
            ax.scatter(x_positions - width / 2, follow_rate[s, d, :],
                       color="black", s=8, alpha=0.4, zorder=3)

        ax.axhline(chance, color="gray", linestyle="--",
                   label=f"chance = {chance:.2f}")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"X={x}" for x in range(NUM_CARDS)])
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(titles[d])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Constant scripted speaker — {label}  ({n_seeds} seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_switch(follow_rate, coord_rate, output_path, label: str,
                 threshold: float):
    """Line chart of FOLLOW(Y) rate vs k, both directions, with chance line and
    'last accepted switch' annotation.

    Inputs are (num_seeds, 2, K-1) arrays.
    """
    n_seeds, _, n_k = follow_rate.shape
    chance = 1.0 / NUM_CARDS
    ks = np.arange(1, n_k + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    titles = ["agent 0 → 1 (agent 0 scripted)", "agent 1 → 0 (agent 1 scripted)"]
    for d in range(2):
        ax = axes[d]
        mean_f = follow_rate[:, d, :].mean(axis=0)
        sem_f = (follow_rate[:, d, :].std(axis=0, ddof=1) / math.sqrt(n_seeds)
                 if n_seeds > 1 else np.zeros_like(mean_f))
        mean_c = coord_rate[:, d, :].mean(axis=0)

        # per-seed thin lines
        for s in range(n_seeds):
            ax.plot(ks, follow_rate[s, d, :], color="#3a86ff",
                    alpha=0.18, linewidth=0.8)
        # mean ± SEM
        ax.plot(ks, mean_f, color="#3a86ff", linewidth=2.0, label="FOLLOW(Y) mean")
        ax.fill_between(ks, mean_f - sem_f, mean_f + sem_f,
                        color="#3a86ff", alpha=0.18, label="±SEM")
        ax.plot(ks, mean_c, color="#ffbe0b", linewidth=1.5,
                linestyle="--", label="COORD mean")

        ax.axhline(chance, color="gray", linestyle=":",
                   label=f"chance = {chance:.2f}")
        ax.axhline(threshold, color="red", linestyle=":",
                   label=f"threshold = {threshold}")

        # Last-accepted-switch annotation
        accepted = ks[mean_f >= threshold]
        if accepted.size > 0:
            last_k = int(accepted.max())
            y = float(mean_f[last_k - 1])
            ax.annotate(
                f"last-accepted k={last_k}",
                xy=(last_k, y),
                xytext=(last_k, y + 0.12),
                fontsize=10, color="red",
                ha="center",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2),
            )
        else:
            ax.text(
                ks[-1], threshold + 0.02,
                f"never accepted (max FOLLOW < {threshold})",
                fontsize=9, color="red", ha="right",
            )

        ax.set_xticks(ks)
        ax.set_xlabel("switch slot k (speaker emits Y from slot k onward)")
        ax.set_ylabel("rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(titles[d])
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Switch scripted speaker — {label}  ({n_seeds} seeds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument(
        "--accept-threshold", type=float, default=0.5,
        help="FOLLOW rate threshold for the 'last accepted switch' annotation.",
    )
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

    constant_per_seed = []  # list of (coord, follow, total) triples, each (2, 5)
    switch_per_seed = []    # list of (coord, follow, total) triples, each (2, K-1)

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)
        base_seed = 8_000 + seed_idx * 100_000

        if do_constant:
            print(f"\n[Seed {seed_idx}] Running constant experiment "
                  f"(2 dirs x {NUM_CARDS} X-values x {args.num_episodes} eps)...")
            coord_count, follow_count, n_total = _eval_constant(
                inner_env, policy, params, args.num_episodes, max_steps, base_seed,
            )
            constant_per_seed.append((coord_count, follow_count, n_total))
            print(f"\nSeed {seed_idx}  Constant FOLLOW rate "
                  f"(P[listener canonical pick == speaker canonical msg | X], "
                  f"chance={1.0/NUM_CARDS:.2f}):")
            print(_format_constant(follow_count, n_total))
            print(f"  COORD rate (env reward at decision):")
            print(_format_constant(coord_count, n_total))

        if do_switch:
            n_pairs = NUM_CARDS * (NUM_CARDS - 1)
            n_k_slots = max_steps - 2
            n_eps_total = 2 * n_pairs * n_k_slots * args.num_episodes
            print(f"\n[Seed {seed_idx}] Running switch experiment "
                  f"({n_eps_total} episodes total)...")
            coord_count, follow_count, n_total = _eval_switch(
                inner_env, policy, params, args.num_episodes, max_steps, base_seed + 5_000,
            )
            switch_per_seed.append((coord_count, follow_count, n_total))
            print(f"\nSeed {seed_idx}  Switch FOLLOW(Y) rate "
                  f"(P[listener canonical pick == canonical Y | switch at slot k]):")
            print(_format_switch(follow_count, n_total))
            print(f"  COORD rate (env reward at decision):")
            print(_format_switch(coord_count, n_total))

    # --- Aggregate across seeds ---
    def _aggregate(per_seed, n_axis: int):
        """Returns (per_seed_rate, mean, sem) for a given metric per seed."""
        rates = np.stack(per_seed) / np.maximum(np.stack(
            [t for *_, t in [s for s in per_seed]]
        ), 1)
        return rates

    constant_follow_rate = constant_coord_rate = None
    switch_follow_rate = switch_coord_rate = None

    if do_constant:
        constant_coord_rate = np.stack(
            [c / np.maximum(t, 1) for c, _, t in constant_per_seed]
        )
        constant_follow_rate = np.stack(
            [f / np.maximum(t, 1) for _, f, t in constant_per_seed]
        )
        print(f"\nAggregate Constant FOLLOW rate across {num_seeds} seeds (mean ± SEM):")
        mean_f = constant_follow_rate.mean(axis=0)
        sem_f = (constant_follow_rate.std(axis=0, ddof=1) / math.sqrt(num_seeds)
                 if num_seeds > 1 else np.zeros_like(mean_f))
        for d, name in enumerate(("agent_0->1", "agent_1->0")):
            parts = [f"{mean_f[d, x]:.3f}±{sem_f[d, x]:.3f}" for x in range(NUM_CARDS)]
            print(f"  {name}: " + "  ".join(parts)
                  + f"    mean={mean_f[d].mean():.3f}")

    if do_switch:
        n_k = max_steps - 2
        switch_coord_rate = np.stack(
            [c / np.maximum(t, 1) for c, _, t in switch_per_seed]
        )
        switch_follow_rate = np.stack(
            [f / np.maximum(t, 1) for _, f, t in switch_per_seed]
        )
        print(f"\nAggregate Switch FOLLOW(Y) rate across {num_seeds} seeds (mean ± SEM):")
        mean_f = switch_follow_rate.mean(axis=0)
        sem_f = (switch_follow_rate.std(axis=0, ddof=1) / math.sqrt(num_seeds)
                 if num_seeds > 1 else np.zeros_like(mean_f))
        for d, name in enumerate(("0->1", "1->0")):
            parts = [f"{mean_f[d, k]:.3f}±{sem_f[d, k]:.3f}" for k in range(n_k)]
            print(f"  {name}: " + "  ".join(parts))

        # "Last accepted switch" = max k where mean FOLLOW rate >= threshold.
        threshold = args.accept_threshold
        ks = np.arange(1, n_k + 1)
        for d, name in enumerate(("0->1", "1->0")):
            accepted = ks[mean_f[d] >= threshold]
            last = int(accepted.max()) if accepted.size > 0 else None
            label_str = f"k={last}" if last is not None else f"none (threshold {threshold})"
            print(f"  {name} last-accepted switch: {label_str}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = label.replace("/", "_").replace(" ", "_")
        if do_constant:
            np.savez(
                out / f"scripted_constant_{slug}.npz",
                coord=np.stack([c for c, _, _ in constant_per_seed]),
                follow=np.stack([f for _, f, _ in constant_per_seed]),
                total=np.stack([t for _, _, t in constant_per_seed]),
            )
            print(f"\nSaved {out / f'scripted_constant_{slug}.npz'}")
            _plot_constant(constant_follow_rate, constant_coord_rate,
                           out / f"constant_follow_{slug}.png", label)
            print(f"Saved {out / f'constant_follow_{slug}.png'}")
        if do_switch:
            np.savez(
                out / f"scripted_switch_{slug}.npz",
                coord=np.stack([c for c, _, _ in switch_per_seed]),
                follow=np.stack([f for _, f, _ in switch_per_seed]),
                total=np.stack([t for _, _, t in switch_per_seed]),
            )
            print(f"Saved {out / f'scripted_switch_{slug}.npz'}")
            _plot_switch(switch_follow_rate, switch_coord_rate,
                         out / f"switch_follow_{slug}.png", label,
                         threshold=args.accept_threshold)
            print(f"Saved {out / f'switch_follow_{slug}.png'}")


if __name__ == "__main__":
    main()
