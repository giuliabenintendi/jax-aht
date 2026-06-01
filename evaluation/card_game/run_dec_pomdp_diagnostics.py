"""Compute the five Dec-POMDP diagnostics (Tessera et al., AAMAS 2026) on
card-game trajectories produced by `extract_diag_trajectories.py`.

This runs in an ISOLATED environment: it imports only numpy and
`dec_pomdp_diagnostics`, never JAX or this repo, so it runs locally on macOS
where the training env cannot. The diagnostics package is pure scipy/sklearn.

    uv run --no-project --with dec-pomdp-diagnostics --with numpy \\
        python evaluation/card_game/run_dec_pomdp_diagnostics.py \\
        --data-dir evaluation/card_game/diag_data \\
        --out evaluation/card_game/diag_data/diagnostics_table.csv

Observation model
-----------------
Under Other-Play the ground-truth board is constant (symmetrized into per-agent
recolour/shuffle), so the only varying, cross-agent-commensurable observation is
the communication channel. Per agent per step:

    O_i(t) = onehot(step)                                   (no-comm conditions)
    O_i(t) = onehot(step) || onehot(partner's previous msg) (comm conditions)

The action A_i(t) is the ground-truth emitted card (a message on deliberation
steps, the pick on the decision step). The partner's previous message is the
ground-truth card the partner emitted at t-1 (what the in-obs dot encodes); it is
included only for `has_comm` runs because no-comm agents cannot observe it. One
UserData = one run: each SP seed and each XP pair is scored independently, then
aggregated per condition. SP values on shaped conditions are near-circular (the
shaping reward directly pays for message coordination); the XP value and the
SP->XP drop are the load-bearing numbers.

The package reports each metric as a raw kNN-MI `*_max` (the stronger of the two
agents/directions) against a permutation-null `*_max_null`; we report the
`excess = raw - null` (signal above chance) plus the four Decision-Rule flags,
which are the headline per-condition result.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import dec_pomdp_diagnostics as dpd

NUM_CARDS = 5
AGENTS = ("agent_0", "agent_1")

# metrics= argument to compute_diagnostics (canonical short names).
METRIC_ARG = ("oar", "har", "pif", "aa", "dai")

# (label, raw key, null key, normalized key) in the returned metrics dict.
# The `_ohist` variants are the observation-history estimators used when no RNN
# hidden states are supplied.
METRIC_SPECS = (
    ("oar", "oar_max", "oar_max_null", "oarR_max"),
    ("har", "har_ohist_max", "har_ohist_max_null", "harRcond_ohist_max"),
    ("pif", "pif_ohist_max", "pif_ohist_max_null", "pifRcond_ohist_max"),
    ("aa", "aa_max", "aa_max_null", "aaRcond_max"),
    ("dai", "daiOA_ohist_max", "daiOA_ohist_max_null", "daiOARcond_ohist_max"),
)
FLAG_KEYS = (
    "history_dependence",
    "uses_hidden_teammate_info",
    "synchronous_coordination",
    "temporal_coordination",
)


def _onehot(idx: np.ndarray, n: int, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """One-hot `idx` into `n` columns, plus small Gaussian jitter so the kNN MI
    estimators do not choke on exact ties in discrete-valued observations."""
    oh = np.zeros((idx.shape[0], n), dtype=np.float64)
    oh[np.arange(idx.shape[0]), idx] = 1.0
    if jitter > 0:
        oh += jitter * rng.standard_normal(oh.shape)
    return oh


def _build_userdata(intents: np.ndarray, has_comm: bool, scenario: str, seed: int,
                    jitter: float) -> dpd.UserData:
    """`intents`: `(E, T, 2)` ground-truth emitted card per agent per step."""
    rng = np.random.default_rng(seed)
    e, t, _ = intents.shape
    step = np.tile(np.arange(t), e).astype(np.int64)
    episode_ids = np.repeat(np.arange(e), t).astype(np.int64)

    observations, actions = {}, {}
    for i, agent in enumerate(AGENTS):
        feats = [_onehot(step, t, jitter, rng)]
        if has_comm:
            partner_prev = np.full((e, t), -1, dtype=np.int64)
            partner_prev[:, 1:] = intents[:, :-1, 1 - i]
            # shift -1 (no message yet) into category 0
            feats.append(_onehot(partner_prev.reshape(-1) + 1, NUM_CARDS + 1, jitter, rng))
        observations[agent] = np.concatenate(feats, axis=1)
        actions[agent] = intents[:, :, i].reshape(-1).astype(np.int64)

    timesteps = {a: step for a in AGENTS}
    eids = {a: episode_ids for a in AGENTS}
    # alg_name deliberately omits "RNN": we feed observation history, not hidden
    # states, so the obs-history estimators are the intended path.
    return dpd.UserData(
        observations=observations, actions=actions, timesteps=timesteps,
        episode_ids=eids, env_name="card_game", alg_name="JA-IPPO",
        seed=seed, scenario_name=scenario,
    )


def _score_run(intents: np.ndarray, has_comm: bool, scenario: str, seed: int,
               history_k: int, null_reps: int, jitter: float) -> dict:
    data = _build_userdata(intents, has_comm, scenario, seed, jitter)
    result = dpd.compute_diagnostics(
        data, history_k=history_k, null_reps=null_reps, metrics=METRIC_ARG,
    )
    m = result.metrics
    row: dict = {"scenario": scenario}
    for label, raw_k, null_k, norm_k in METRIC_SPECS:
        raw = float(m.get(raw_k, float("nan")))
        null = float(m.get(null_k, float("nan")))
        row[f"raw.{label}"] = raw
        row[f"null.{label}"] = null
        row[f"excess.{label}"] = raw - null
        row[f"norm.{label}"] = float(m.get(norm_k, float("nan")))
    for fk in FLAG_KEYS:
        row[f"flag.{fk}"] = bool(result.flags.get(fk, False))
    return row


def _nanmean(vals: list[float]) -> float:
    arr = np.array(vals, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _summarize(rows: list[dict]) -> dict:
    """Mean excess per metric + flag-True fraction over a set of run rows."""
    out: dict = {}
    for label, *_ in METRIC_SPECS:
        out[f"excess.{label}"] = _nanmean([r[f"excess.{label}"] for r in rows])
    for fk in FLAG_KEYS:
        out[f"flag.{fk}"] = (
            float(np.mean([r[f"flag.{fk}"] for r in rows])) if rows else float("nan")
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="evaluation/card_game/diag_data",
                        help="Directory of <condition>.npz files from the extractor.")
    parser.add_argument("--conditions", nargs="*", default=None,
                        help="Condition basenames to include (default: all .npz in data-dir).")
    parser.add_argument("--out", default=None, help="CSV path for the per-run table.")
    parser.add_argument("--history-k", type=int, default=3)
    parser.add_argument("--null-reps", type=int, default=5)
    parser.add_argument("--jitter", type=float, default=1e-6)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = (
        [data_dir / f"{c}.npz" for c in args.conditions] if args.conditions
        else sorted(data_dir.glob("*.npz"))
    )
    if not files:
        raise SystemExit(f"No .npz files found in {data_dir}")

    per_run: list[dict] = []
    summary: list[dict] = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        cond = str(d["condition"])
        has_comm = bool(d["has_comm"])
        sp = d["sp_intents"]
        xp = d["xp_intents"]
        print(f"\n=== {cond}  (comm={has_comm})  sp={sp.shape}  xp={xp.shape} ===")

        sp_rows, xp_rows = [], []
        for kind, block, rows in (("sp", sp, sp_rows), ("xp", xp, xp_rows)):
            for k in range(block.shape[0]):
                r = _score_run(block[k], has_comm, f"{cond}/{kind}/{k}",
                               seed=(0 if kind == "sp" else 100) + k,
                               history_k=args.history_k, null_reps=args.null_reps,
                               jitter=args.jitter)
                r.update(condition=cond, kind=kind, idx=k)
                per_run.append(r)
                rows.append(r)
                excess = "  ".join(f"{lab}={r['excess.' + lab]:+.3f}" for lab, *_ in METRIC_SPECS)
                flags = "".join("1" if r["flag." + fk] else "0" for fk in FLAG_KEYS)
                print(f"  {kind} {k:2d}: {excess}  flags[{''.join(fk[0] for fk in FLAG_KEYS)}]={flags}")

        summary.append({"condition": cond, "comm": has_comm,
                        "sp": _summarize(sp_rows), "xp": _summarize(xp_rows)})

    _print_summary(summary)

    if args.out:
        out_path = Path(args.out)
        keys = sorted({k for r in per_run for k in r})
        with out_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(per_run)
        print(f"\nSaved per-run table -> {out_path}")


def _print_summary(summary: list[dict]) -> None:
    labels = [lab for lab, *_ in METRIC_SPECS]
    print("\n" + "=" * 88)
    print("PER-CONDITION SUMMARY")
    print(f"\nMean excess-over-null per metric  (flag legend: {', '.join(FLAG_KEYS)})")
    head = "condition".ljust(22) + "kind  " + "".join(f"{lab:>9}" for lab in labels) + "   flags"
    print(head)
    for s in summary:
        for kind in ("sp", "xp"):
            agg = s[kind]
            metr = "".join(f"{agg['excess.' + lab]:>+9.3f}" for lab in labels)
            flg = " ".join(f"{agg['flag.' + fk]:.2f}" for fk in FLAG_KEYS)
            print(f"{s['condition'].ljust(22)}{kind:<6}{metr}   {flg}")


if __name__ == "__main__":
    main()
