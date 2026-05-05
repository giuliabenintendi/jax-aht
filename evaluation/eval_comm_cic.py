"""Compute slotwise Causal Influence of Communication (CIC) for a card-game checkpoint.

For each deliberation slot `k`, each speaker `i`, and each listener `j = 1-i`:
  1. Run a baseline episode under the trained policy. Save:
     - the sampled action sequence (used to "replay" the episode under intervention)
     - the speaker's natural message distribution at each slot, p_c[k, i, :].
       This comes from the policy's softmax output (pi.probs[5:10]) at slot k —
       NOT empirical counts. Per-context, per the paper.
  2. For each m ∈ {0..NUM_CARDS-1}: replay the same episode using baseline actions,
     EXCEPT at slot k force the speaker's action to a message of value m. Read off
     the listener's policy distribution at the next step (k+1). For k < K-1 that's
     a 5-dim distribution over messages; for k = K-1 it's a 5-dim distribution
     over picks (the env-affecting decision action). Both are masked-Discrete(5).
     Stack into p_a_given_do_c[5, 5].
  3. CIC^(k)_{i->j}(ep) = calc_cic(p_a_given_do_c, p_c[k, i, :], 5, 5).

Average across episodes per (seed, slot, speaker), then aggregate across seeds.
Output: a (2, K) grid per seed + mean±SEM aggregate, same format as the SC driver.

Design notes:
- We never propagate cascades. Each slot is an independent one-step CIC measurement,
  matching the paper's Algorithm 1 applied per-slot. No greedy, no sampling for
  intervention. Listener's distribution is read directly from the policy's softmax.
- Eval-time `scramble_partner_msg` is forced to False so we have full control over
  what the listener sees, regardless of how the checkpoint was trained.
- Replay uses the same env wrappers (LogWrapper / OP) as the inner env; the
  intervention is just a forced action at one step.

Usage:
    ./run_gpu.sh 0 evaluation.eval_comm_cic \
        --checkpoint <path_to_saved_train_run> \
        --num-episodes 256
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.card_game.rendering import NUM_CARDS
from evaluation._card_game_utils import load_card_game_eval
from evaluation.comm_metrics import calc_cic


def _policy_step(policy, params, obs_dict, agent_id, hstate, avail, rng):
    """One forward pass of `policy.get_action_value_policy` for a single agent.

    Returns (sampled_action_int, pi_probs[NUM_CARDS], new_hstate). The action
    space is a unified Discrete(NUM_CARDS): the same emitted value is
    interpreted as a message during deliberation and as a pick on the
    decision step, so `pi.probs` is always a NUM_CARDS-dim distribution.
    """
    obs = obs_dict[f"agent_{agent_id}"].reshape(1, 1, -1)
    done = jnp.zeros((1, 1), dtype=bool)
    avail_a = avail[f"agent_{agent_id}"].astype(jnp.float32)
    action, _value, pi, new_hstate, _attn = policy.get_action_value_policy(
        params=params,
        obs=obs,
        done=done,
        avail_actions=avail_a,
        hstate=hstate,
        rng=rng,
    )
    probs = np.asarray(pi.probs).reshape(-1)
    return int(action.squeeze()), probs, new_hstate


def _baseline_pass(env, policy, params, reset_rng, max_steps):
    """Run one full episode with sampled actions; record what CIC will need.

    Returns:
        baseline_actions: (max_steps, 2) int32. Sampled action at each step
            for each agent. Used to replay the trajectory under intervention.
        p_c: (K, 2, NUM_CARDS) float64. Speaker's natural message distribution
            at each deliberation slot, sliced from pi.probs[5:10]. K = max_steps - 1.
    """
    K = max_steps - 1
    rng = jax.random.fold_in(reset_rng, 1)
    obs, state = env.reset(reset_rng)
    h_0 = policy.init_hstate(1)
    h_1 = policy.init_hstate(1)

    baseline_actions = np.zeros((max_steps, 2), dtype=np.int32)
    p_c = np.zeros((K, 2, NUM_CARDS), dtype=np.float64)

    for c in range(max_steps):
        avail = env.get_avail_actions(state)
        rng, k0, k1, kstep = jax.random.split(rng, 4)
        a0, probs0, h_0 = _policy_step(policy, params, obs, 0, h_0, avail, k0)
        a1, probs1, h_1 = _policy_step(policy, params, obs, 1, h_1, avail, k1)

        baseline_actions[c, 0] = a0
        baseline_actions[c, 1] = a1
        if c < K:
            # Deliberation slot: pi.probs is the speaker's msg distribution.
            p_c[c, 0] = probs0
            p_c[c, 1] = probs1

        env_act = {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)}
        obs, state, _r, _d, _info = env.step(kstep, state, env_act)

    return baseline_actions, p_c


def _replay_with_intervention(
    env, policy, params, reset_rng, baseline_actions,
    slot_k, speaker_idx, intervened_m, max_steps,
):
    """Replay the same episode with one forced action at slot k; return
    listener's distribution at step k+1.

    Mechanics:
      - For step c in [0, k-1]: feed both agents' policies on the actual
        baseline obs (advances LSTMs), then step env with baseline actions.
        Listener LSTM at the start of step k+1 is identical to baseline,
        because nothing observed by either agent up to step k differs from
        baseline (the intervention only affects obs at step k+1).
      - At step c == k: listener uses baseline action; speaker's action is
        forced to `intervened_m`. Step env. Now env_state.messages[i] = m.
      - At step c == k+1: listener observes the modified dot. Forward
        listener's policy ONCE on this obs. Return its NUM_CARDS-dim
        distribution (over picks if k+1 is the decision step, over msgs
        otherwise — both routed by the env's `is_decision`, same
        Discrete(NUM_CARDS) layout).

    The function returns just the listener's 5-dim distribution; we don't
    care about the speaker's distribution past slot k for CIC.
    """
    listener_idx = 1 - speaker_idx
    # Derive a deterministic-but-distinct rng from reset_rng so different
    # interventions in the same episode use different keys (only matters for
    # env-internal randomness; actions are scripted, not sampled).
    rng = jax.random.fold_in(
        reset_rng, 2 + slot_k * 1_000 + speaker_idx * 10 + int(intervened_m),
    )
    obs, state = env.reset(reset_rng)
    h_0 = policy.init_hstate(1)
    h_1 = policy.init_hstate(1)

    for c in range(slot_k + 1):
        avail = env.get_avail_actions(state)
        rng, k0, k1, kstep = jax.random.split(rng, 4)
        # Forward both policies — only LSTM advance matters here, action is
        # overridden by baseline (or intervention at slot k).
        _, _, h_0 = _policy_step(policy, params, obs, 0, h_0, avail, k0)
        _, _, h_1 = _policy_step(policy, params, obs, 1, h_1, avail, k1)

        a0 = int(baseline_actions[c, 0])
        a1 = int(baseline_actions[c, 1])
        if c == slot_k:
            forced = int(intervened_m)
            if speaker_idx == 0:
                a0 = forced
            else:
                a1 = forced
        env_act = {"agent_0": jnp.int32(a0), "agent_1": jnp.int32(a1)}
        obs, state, _r, _d, _info = env.step(kstep, state, env_act)

    # Now `obs` is observed BEFORE listener acts at step k+1. Forward the
    # listener's policy once and read its NUM_CARDS-dim distribution.
    avail = env.get_avail_actions(state)
    rng, kl = jax.random.split(rng)
    _, probs_l, _ = _policy_step(
        policy, params, obs, listener_idx, (h_0, h_1)[listener_idx], avail, kl,
    )
    return probs_l


def _compute_cic_one_episode(env, policy, params, reset_rng, max_steps):
    """Per-episode (2, K) CIC array.

    cic[i, k] = CIC^(k)_{i->j=1-i} for this episode.
    """
    K = max_steps - 1
    baseline_actions, p_c = _baseline_pass(env, policy, params, reset_rng, max_steps)

    cic = np.zeros((2, K), dtype=np.float64)
    for speaker_idx in range(2):
        for k in range(K):
            p_a_given_do_c = np.zeros((NUM_CARDS, NUM_CARDS), dtype=np.float64)
            for m in range(NUM_CARDS):
                p_a_given_do_c[m] = _replay_with_intervention(
                    env, policy, params, reset_rng, baseline_actions,
                    slot_k=k, speaker_idx=speaker_idx,
                    intervened_m=m, max_steps=max_steps,
                )
            cic[speaker_idx, k] = calc_cic(
                p_a_given_do_c, p_c[k, speaker_idx], n_comm=NUM_CARDS, n_acts=NUM_CARDS,
            )
    return cic


def _format_cic_grid(cic: np.ndarray) -> str:
    K = cic.shape[1]
    header = "slot:  " + "  ".join(f"  k={k}" for k in range(K))
    a0 = "0->1   " + "  ".join(f"{v:6.3f}" for v in cic[0])
    a1 = "1->0   " + "  ".join(f"{v:6.3f}" for v in cic[1])
    return "\n".join((header, a0, a1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    if not ev.env_kwargs.get("communication", False):
        raise SystemExit("Checkpoint was trained without communication; CIC is undefined.")

    print(
        f"\nCheckpoint: {args.checkpoint}"
        f"\n  label={ev.label}  comm=on  scramble_eval=False  best_idx={ev.best_idx.tolist()}"
        f"\n  seeds={ev.num_seeds}  episodes/seed={args.num_episodes}"
        f"  max_steps={ev.max_steps}"
        f"\n  CIC floor (no influence) = log({NUM_CARDS}) = {math.log(NUM_CARDS):.3f}"
    )

    K = ev.max_steps - 1
    all_cic = []
    for seed_idx in range(ev.num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], ev.params)
        ep_cics = np.zeros((args.num_episodes, 2, K), dtype=np.float64)
        for ep in range(args.num_episodes):
            reset_rng = jax.random.PRNGKey(7_000 + seed_idx * 10_000 + ep)
            ep_cics[ep] = _compute_cic_one_episode(
                ev.env, ev.policy, params, reset_rng, ev.max_steps,
            )
            if (ep + 1) % 32 == 0:
                running = ep_cics[: ep + 1].mean(axis=0)
                print(
                    f"  seed {seed_idx} ep {ep + 1}/{args.num_episodes}: "
                    f"running mean CIC[*, K-1] = "
                    f"{running[0, -1]:.3f}, {running[1, -1]:.3f}"
                )
        seed_cic = ep_cics.mean(axis=0)
        all_cic.append(seed_cic)
        print(f"\nSeed {seed_idx}  CIC (nats, floor={math.log(NUM_CARDS):.3f}):")
        print(_format_cic_grid(seed_cic))

    all_cic_arr = np.stack(all_cic, axis=0)
    mean_cic = all_cic_arr.mean(axis=0)
    sem_cic = (
        all_cic_arr.std(axis=0, ddof=1) / math.sqrt(ev.num_seeds)
        if ev.num_seeds > 1 else np.zeros_like(mean_cic)
    )

    print(f"\nAggregate across {ev.num_seeds} seeds (mean ± SEM):")
    for direction_idx, label_dir in enumerate(("0->1", "1->0")):
        parts = [
            f"{mean_cic[direction_idx, k]:6.3f}±{sem_cic[direction_idx, k]:.3f}"
            for k in range(K)
        ]
        print(f"  {label_dir}: " + "  ".join(parts))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = ev.label.replace("/", "_").replace(" ", "_")
        np.save(out / f"cic_{slug}.npy", all_cic_arr)
        print(f"\nSaved {out / f'cic_{slug}.npy'}  shape={all_cic_arr.shape}")


if __name__ == "__main__":
    main()
