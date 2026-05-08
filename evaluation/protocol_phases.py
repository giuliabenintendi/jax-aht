"""Per-step protocol-phase classification + within-seed convergence diagnostics.

Classifies every step of every greedy eval episode into one of five regimes
based on the joint trajectory of the two agents' GT-space messages:

  - hold_and_match    A_t == M_t  &  A_t == A_{t-1}  &  M_t == M_{t-1}
  - fresh_match       A_t == M_t  &  at least one moved
  - lag_echo          A_t != M_t  &  one agent matches the partner's previous
                      (xor of the two echo conditions)
  - mutual_swap       A_t != M_t  &  both echoed each other's previous
  - drift             none of the above

Step 0 has no t-1 reference, so it can only be `fresh_match` (if A_0==M_0)
or `drift`. The decision step is recorded separately as a binary success
(picks_match_in_GT).

Outputs per seed:
  - first_match_step.png     histogram of step index of first match
                             (`fresh_match` or `hold_and_match`); a "never"
                             bucket holds episodes that never matched.
  - first_lock_step.png      histogram of step index of first hold_and_match.
  - regime_fractions.png     stacked-bar of per-step regime distribution.
  - episode_outcomes.csv     one row per episode: first_match_step,
                             first_lock_step, decision_success, regime counts.

Plus an aggregate summary across seeds.

The within-seed distribution shape is the point: a robust seed has its
first-match histogram tight and to the left, with high decision success;
a fragile seed has a long tail or a 'never' spike even when its average
success is high.

Usage:
    ./run_gpu.sh <gpu> evaluation.protocol_phases \\
        --checkpoint /path/to/saved_train_run \\
        --num-episodes 256 \\
        --output-dir plots/card_game/<run_name>
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import run_episode_with_states


REGIMES = ("hold_and_match", "fresh_match", "lag_echo", "mutual_swap", "drift")
REGIME_COLORS = {
    "hold_and_match": "#2ca02c",
    "fresh_match":    "#9ecf91",
    "lag_echo":       "#ffbb78",
    "mutual_swap":    "#d62728",
    "drift":          "#bcbcbc",
}


def _get_obs_type(alg_config: dict) -> str:
    return alg_config.get("OBS_TYPE",
                          alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))


def _classify_step(a_t, m_t, a_prev, m_prev) -> str:
    """Classify one step of a single episode given GT messages.

    `*_prev` may be None at t=0 (no previous reference).
    """
    matched = a_t == m_t
    if a_prev is None or m_prev is None:
        return "fresh_match" if matched else "drift"
    held_self = a_t == a_prev
    held_partner = m_t == m_prev
    if matched and held_self and held_partner:
        return "hold_and_match"
    if matched:
        return "fresh_match"
    a_echoes_m = a_t == m_prev
    m_echoes_a = m_t == a_prev
    if a_echoes_m and m_echoes_a:
        return "mutual_swap"
    if a_echoes_m or m_echoes_a:
        return "lag_echo"
    return "drift"


def _classify_episode(ep_messages, ep_actions, max_steps: int):
    """Return (per_step_regimes, first_match, first_lock, decision_success)."""
    regimes = []
    a_prev, m_prev = None, None
    first_match = None
    first_lock = None
    n_delib = max_steps - 1
    for t in range(min(n_delib, len(ep_messages))):
        a_t, m_t = int(ep_messages[t][0]), int(ep_messages[t][1])
        if a_t < 0 or m_t < 0:
            regimes.append("drift")
            continue
        regime = _classify_step(a_t, m_t, a_prev, m_prev)
        regimes.append(regime)
        if regime in ("fresh_match", "hold_and_match") and first_match is None:
            first_match = t
        if regime == "hold_and_match" and first_lock is None:
            first_lock = t
        a_prev, m_prev = a_t, m_t

    decision_success = False
    if max_steps - 1 < len(ep_actions):
        p0, p1 = ep_actions[max_steps - 1]
        decision_success = (p0 >= 0 and p1 >= 0 and int(p0) == int(p1))

    return regimes, first_match, first_lock, decision_success


def _plot_first_step_hist(values, n_episodes, n_delib_steps, title, out_path):
    """Histogram of first-occurrence step. None values bucket as 'never'."""
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = list(range(n_delib_steps + 1))
    counts = [0] * n_delib_steps
    never = 0
    for v in values:
        if v is None:
            never += 1
        else:
            counts[v] += 1
    ax.bar(bins[:-1], counts, color="#2ca02c", edgecolor="black", linewidth=0.4)
    ax.bar([n_delib_steps], [never], color="#888", edgecolor="black", linewidth=0.4)
    ax.set_xticks(list(range(n_delib_steps)) + [n_delib_steps])
    ax.set_xticklabels([str(i) for i in range(n_delib_steps)] + ["never"])
    ax.set_xlabel("step index")
    ax.set_ylabel(f"# episodes (of {n_episodes})")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_regime_fractions(per_step_counts, n_episodes, out_path, title):
    """Stacked-bar of regime fractions per step."""
    n_steps = per_step_counts.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * n_steps), 3.5))
    bottoms = np.zeros(n_steps)
    for r_idx, regime in enumerate(REGIMES):
        fracs = per_step_counts[:, r_idx] / max(n_episodes, 1)
        ax.bar(np.arange(n_steps), fracs, bottom=bottoms,
               color=REGIME_COLORS[regime], label=regime, edgecolor="white", linewidth=0.4)
        bottoms += fracs
    ax.set_xticks(np.arange(n_steps))
    ax.set_xlabel("deliberation step")
    ax.set_ylabel("episode fraction")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved train run checkpoint directory")
    parser.add_argument("--num-episodes", type=int, default=256,
                        help="Greedy eval episodes per seed (>=128 for stable hist)")
    parser.add_argument("--output-dir", default="plots/card_game/protocol_phases")
    parser.add_argument("--episode-rng-base", type=int, default=4000)
    parser.add_argument("--use-best", action="store_true",
                        help="Use best_params instead of final_params")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    config_path = None
    cur = run_dir
    for _ in range(4):
        candidate = cur / ".hydra" / "config.yaml"
        if candidate.exists():
            config_path = candidate
            break
        cur = cur.parent
    if config_path is None:
        raise FileNotFoundError(f"No .hydra/config.yaml found near {run_dir}")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]
    if not alg_config.get("COMMUNICATION", False):
        raise ValueError(
            "protocol_phases requires a communication-enabled checkpoint."
        )

    env_kwargs = dict(alg_config["ENV_KWARGS"])
    env_kwargs["communication"] = True
    env_kwargs["scramble_partner_msg"] = False
    alg_config["ENV_KWARGS"] = env_kwargs

    env = make_env(alg_config["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)
    inner_env = env._env
    max_steps = int(env_kwargs.get("max_steps", 8))
    n_delib = max_steps - 1

    obs_type = _get_obs_type(alg_config)
    init_fn = (initialize_ja_image_agent if obs_type in ("image", "fov")
               else initialize_ja_agent)
    policy, _ = init_fn(alg_config, env, jax.random.PRNGKey(0))

    run_data = load_train_run(str(ckpt_path))
    params_key = "best_params" if args.use_best else "final_params"
    if params_key not in run_data:
        raise KeyError(f"{params_key!r} not found; keys: {list(run_data.keys())}")
    stacked_params = run_data[params_key]
    num_seeds = int(jax.tree.leaves(stacked_params)[0].shape[0])
    print(f"[protocol_phases] {params_key}, {num_seeds} seeds, "
          f"{args.num_episodes} eps/seed, max_steps={max_steps}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[protocol_phases] saving to {output_dir.resolve()}")

    aggregate_counts = np.zeros((n_delib, len(REGIMES)), dtype=np.int64)
    per_seed_summary = []
    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], stacked_params)
        seed_dir = output_dir / f"seed_{seed_idx}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        first_match_steps = []
        first_lock_steps = []
        decision_successes = []
        per_step_counts = np.zeros((n_delib, len(REGIMES)), dtype=np.int64)
        regime_idx = {r: i for i, r in enumerate(REGIMES)}

        rows = []
        for ep in range(args.num_episodes):
            ep_rng = jax.random.PRNGKey(
                args.episode_rng_base + seed_idx * 10000 + ep
            )
            _, ep_actions, ep_messages = run_episode_with_states(
                ep_rng, inner_env, params, policy, params, policy, max_steps,
                collect_attention=False, greedy=True,
            )
            regimes, first_match, first_lock, success = _classify_episode(
                ep_messages, ep_actions, max_steps,
            )
            for t, r in enumerate(regimes):
                if t < n_delib:
                    per_step_counts[t, regime_idx[r]] += 1
                    aggregate_counts[t, regime_idx[r]] += 1
            first_match_steps.append(first_match)
            first_lock_steps.append(first_lock)
            decision_successes.append(success)
            rows.append({
                "episode": ep,
                "first_match_step": "" if first_match is None else first_match,
                "first_lock_step":  "" if first_lock  is None else first_lock,
                "decision_success": int(success),
                **{r: int((np.array(regimes) == r).sum()) for r in REGIMES},
            })

        # Per-seed plots + CSV
        n_eps = args.num_episodes
        success_rate = float(np.mean(decision_successes))
        match_ever = float(np.mean([v is not None for v in first_match_steps]))
        lock_ever = float(np.mean([v is not None for v in first_lock_steps]))
        mean_first_match = float(
            np.mean([v for v in first_match_steps if v is not None])
            if any(v is not None for v in first_match_steps) else float("nan")
        )
        mean_first_lock = float(
            np.mean([v for v in first_lock_steps if v is not None])
            if any(v is not None for v in first_lock_steps) else float("nan")
        )
        per_seed_summary.append({
            "seed": seed_idx,
            "decision_success_rate": success_rate,
            "match_ever_rate": match_ever,
            "lock_ever_rate": lock_ever,
            "mean_first_match_step": mean_first_match,
            "mean_first_lock_step": mean_first_lock,
        })

        title_suffix = (f"seed {seed_idx}  |  decision={success_rate:.2f}  "
                        f"match-ever={match_ever:.2f}  lock-ever={lock_ever:.2f}")
        _plot_first_step_hist(
            first_match_steps, n_eps, n_delib,
            title=f"First match step — {title_suffix}",
            out_path=seed_dir / "first_match_step.png",
        )
        _plot_first_step_hist(
            first_lock_steps, n_eps, n_delib,
            title=f"First hold-and-match step — {title_suffix}",
            out_path=seed_dir / "first_lock_step.png",
        )
        _plot_regime_fractions(
            per_step_counts, n_eps,
            out_path=seed_dir / "regime_fractions.png",
            title=f"Regime fractions per step — {title_suffix}",
        )
        with open(seed_dir / "episode_outcomes.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["episode", "first_match_step", "first_lock_step",
                               "decision_success", *REGIMES],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"  seed {seed_idx}: success={success_rate:.3f}  "
              f"first_match≈{mean_first_match:.2f}  first_lock≈{mean_first_lock:.2f}  "
              f"never_matched={1.0 - match_ever:.2f}  never_locked={1.0 - lock_ever:.2f}")

    # Aggregate plot + summary
    _plot_regime_fractions(
        aggregate_counts, args.num_episodes * num_seeds,
        out_path=output_dir / "regime_fractions_all_seeds.png",
        title=f"Regime fractions per step — all {num_seeds} seeds",
    )
    with open(output_dir / "per_seed_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_summary[0].keys()))
        writer.writeheader()
        for row in per_seed_summary:
            writer.writerow(row)
    print(f"[protocol_phases] done. summary: {output_dir / 'per_seed_summary.csv'}")


if __name__ == "__main__":
    main()
