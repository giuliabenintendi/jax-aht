"""Emergent-communication metrics from Lowe et al. 2019
"On the Pitfalls of Measuring Emergent Communication" (AAMAS).

Provides:
- Speaker Consistency (SC): empirical MI between an agent's own message and
  its own subsequent environment action. Observational, no interventions.
- Causal Influence of Communication (CIC): MI under do-intervention on the
  speaker's message, measured via the listener's policy distribution. Needs
  model access at eval time.

For a multi-slot dialogue (our card game emits one message per deliberation
step, K messages before a single decision-step pick per agent), we report
both metrics slot-by-slot. See `compute_sc_slotwise` and the CIC-side driver
in `evaluation.eval_comm_cic` for the per-slot extensions.
"""

from __future__ import annotations

import math

import numpy as np


def calc_mutinfo(
    acts: np.ndarray,
    comms: np.ndarray,
    n_acts: int,
    n_comm: int,
) -> float:
    """Empirical mutual information I(A; C) from paired action/message samples.

    Joint p(a, c) is the empirical co-occurrence, NOT an intervention — this
    exactly matches Lowe et al.'s `calc_mutinfo` (which is used for both SC
    and IC in the paper). Natural-log units (nats).
    """
    acts = np.asarray(acts, dtype=np.int64).reshape(-1)
    comms = np.asarray(comms, dtype=np.int64).reshape(-1)
    if acts.shape != comms.shape:
        raise ValueError(
            f"acts and comms must have the same length, got {acts.shape} vs {comms.shape}"
        )
    if acts.size == 0:
        return 0.0

    counts_a = np.bincount(acts, minlength=n_acts).astype(np.float64)
    counts_c = np.bincount(comms, minlength=n_comm).astype(np.float64)
    counts_ac = np.zeros((n_comm, n_acts), dtype=np.float64)
    np.add.at(counts_ac, (comms, acts), 1.0)

    p_a = counts_a / counts_a.sum()
    p_c = counts_c / counts_c.sum()
    p_ac = counts_ac / counts_ac.sum()

    mi = 0.0
    for c in range(n_comm):
        if p_c[c] <= 0:
            continue
        for a in range(n_acts):
            if p_ac[c, a] > 0 and p_a[a] > 0:
                mi += p_ac[c, a] * math.log(p_ac[c, a] / (p_c[c] * p_a[a]))
    return mi


def _extract_pairs(
    ep_messages: list[list[tuple[int, int]]],
    ep_actions: list[list[tuple[int, int]]],
    agent_idx: int,
    slot: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (msgs, picks) for agent `agent_idx` at deliberation slot `slot`.

    Pairs are dropped if either the slot message or the decision-step pick is
    invalid (-1); those only happen on episodes that were cut short, which
    shouldn't arise from a healthy rollout of a fixed-length episode.
    """
    msgs = []
    picks = []
    for ep_msgs, ep_acts in zip(ep_messages, ep_actions):
        m = ep_msgs[slot][agent_idx]
        p = ep_acts[-1][agent_idx]
        if m >= 0 and p >= 0:
            msgs.append(m)
            picks.append(p)
    return np.asarray(msgs, dtype=np.int64), np.asarray(picks, dtype=np.int64)


def compute_sc_slotwise(
    ep_messages: list[list[tuple[int, int]]],
    ep_actions: list[list[tuple[int, int]]],
    n_messages: int,
    n_picks: int,
) -> np.ndarray:
    """Per-agent, per-slot Speaker Consistency.

    Args:
        ep_messages: list of episodes; each episode is a list of per-step
            `(msg_0, msg_1)` tuples. A step where the agent did not speak has
            msg = -1 for that agent.
        ep_actions: list of episodes; each episode is a list of per-step
            `(pick_0, pick_1)` tuples. The decision step carries the valid
            picks; deliberation steps have -1.
        n_messages: communication-action cardinality (e.g., NUM_CARDS=5).
        n_picks: environment-action cardinality (e.g., NUM_CARDS=5).

    Returns:
        `sc` of shape `(2, K)` where `K` is the number of deliberation slots
        inferred from the first episode's length minus one. `sc[i, k]` is
        I(M_{i,k}; A_i) in nats.
    """
    if not ep_messages or not ep_actions:
        raise ValueError("ep_messages and ep_actions must be non-empty")
    if len(ep_messages) != len(ep_actions):
        raise ValueError("ep_messages and ep_actions must have the same length")

    n_steps = len(ep_messages[0])
    K = n_steps - 1  # last step is decision, the rest are deliberation slots

    sc = np.zeros((2, K), dtype=np.float64)
    for agent_idx in range(2):
        for k in range(K):
            msgs, picks = _extract_pairs(ep_messages, ep_actions, agent_idx, k)
            if msgs.size == 0:
                sc[agent_idx, k] = 0.0
                continue
            sc[agent_idx, k] = calc_mutinfo(
                picks, msgs, n_acts=n_picks, n_comm=n_messages,
            )
    return sc


def calc_cic(
    p_a_given_do_c: np.ndarray,
    p_c: np.ndarray,
    n_comm: int,
    n_acts: int,
) -> float:
    """Causal Influence of Communication for a single context.

    Direct port of Lowe et al. 2019's `calc_cic`. Computes MI between
    `do(c)` and `a` for one (state, speaker, listener) triple.

    Args:
        p_a_given_do_c: shape (n_comm, n_acts). Row `c` is the listener's
            action distribution when the speaker's message is forced to `c`.
        p_c: shape (n_comm,). The speaker's *natural* message distribution
            in the same context (softmax of policy logits, not empirical).

    Returns:
        CIC in nats. Floor (no causal influence) is `log(n_comm)` because
        the paper uses `np.mean` (not `np.sum`) when marginalizing over `c`,
        which adds a `log(n_comm)` offset to the standard MI.
    """
    p_ac = p_a_given_do_c * np.expand_dims(p_c, axis=1)
    p_ac /= np.sum(p_ac)
    p_a = np.mean(p_ac, axis=0)

    cic = 0.0
    for c in range(n_comm):
        if p_c[c] <= 0:
            continue
        for a in range(n_acts):
            if p_ac[c, a] > 0 and p_a[a] > 0:
                cic += p_ac[c, a] * math.log(p_ac[c, a] / (p_c[c] * p_a[a]))
    return cic


## Tests

def _test_calc_mutinfo_independence() -> None:
    """Uniform independent acts and comms should yield MI near 0."""
    rng = np.random.default_rng(0)
    n = 10_000
    acts = rng.integers(0, 5, size=n)
    comms = rng.integers(0, 5, size=n)
    mi = calc_mutinfo(acts, comms, n_acts=5, n_comm=5)
    assert mi < 0.01


def _test_calc_mutinfo_perfect_dependence() -> None:
    """When act == comm deterministically, MI should equal H(act) = log(5)."""
    acts = np.tile(np.arange(5), 1000)
    comms = acts.copy()
    mi = calc_mutinfo(acts, comms, n_acts=5, n_comm=5)
    assert abs(mi - math.log(5)) < 1e-9


def _test_calc_mutinfo_partial_dependence() -> None:
    """Constant msg → zero MI regardless of acts (messages carry no info)."""
    rng = np.random.default_rng(1)
    acts = rng.integers(0, 5, size=5000)
    comms = np.zeros_like(acts)
    mi = calc_mutinfo(acts, comms, n_acts=5, n_comm=5)
    assert mi < 1e-9


def _test_compute_sc_slotwise_shapes() -> None:
    ep_messages = [
        [(0, 1), (2, 3), (-1, -1)],
        [(4, 2), (1, 0), (-1, -1)],
    ]
    ep_actions = [
        [(-1, -1), (-1, -1), (3, 2)],
        [(-1, -1), (-1, -1), (1, 4)],
    ]
    sc = compute_sc_slotwise(ep_messages, ep_actions, n_messages=5, n_picks=5)
    assert sc.shape == (2, 2)


def _test_calc_cic_no_influence() -> None:
    """If p(a|do(c)) is identical for every c, CIC should equal log(n_comm)."""
    n_comm, n_acts = 5, 5
    p_a = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    p_a_given_do_c = np.tile(p_a, (n_comm, 1))
    p_c = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    cic = calc_cic(p_a_given_do_c, p_c, n_comm, n_acts)
    assert abs(cic - math.log(n_comm)) < 1e-9


def _test_calc_cic_perfect_influence() -> None:
    """If do(c=k) deterministically forces a=k, CIC = 2*log(n_comm)."""
    n_comm = n_acts = 5
    p_a_given_do_c = np.eye(n_comm, n_acts)
    p_c = np.full(n_comm, 1.0 / n_comm)
    cic = calc_cic(p_a_given_do_c, p_c, n_comm, n_acts)
    assert abs(cic - 2 * math.log(n_comm)) < 1e-9
