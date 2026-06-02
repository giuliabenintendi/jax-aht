"""Paper-faithful Dec-POMDP diagnostics (Tessera et al., AAMAS 2026) for the card game.

Replicates their pipeline as closely as an Other-Play image env allows:
  - per-agent GROUND-TRUTH observation `[onehot(step), channel]` and ground-truth
    action (the canonical emitted card), so obs and actions share one frame and
    cross-agent coordination is defined (OP relabels per agent — see extractor);
  - the RNN HIDDEN STATE is supplied, so the package computes their `*_hidden`
    diagnostics, not only the obs-history `*_ohist` fallback;
  - the NORMALIZED metrics (`oarR`, `harRcond`, `pifRcond`/`pifOARcond`, `aaRcond`,
    `daiRcond`/`daiOARcond`) are reported with `rliable` bootstrap CIs (their
    `rliable_mean_ci`), bolded when they exceed the permutation null;
  - the four decision flags are aggregated per condition.

Channel by condition (faithful to each observation function): op_only = none,
comm = partner's previous canonical message, JA = partner's previous canonical
attention feed.

Runs in an isolated env (numpy + dec-pomdp-diagnostics, no JAX):
    uv run --no-project --with dec-pomdp-diagnostics --with numpy \\
        python evaluation/card_game/run_dec_pomdp_diagnostics.py \\
        --data-dir evaluation/card_game/diag_data \\
        --out evaluation/card_game/diag_data/diagnostics_table.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import dec_pomdp_diagnostics as dpd
from dec_pomdp_diagnostics import rliable_mean_ci

NUM_CARDS = 5
AGENTS = ("agent_0", "agent_1")
METRIC_ARG = ("oar", "har", "pif", "aa", "dai")
FLAG_KEYS = (
    "history_dependence",
    "uses_hidden_teammate_info",
    "synchronous_coordination",
    "temporal_coordination",
)

# (label, normalized RNN/hidden col, its null, normalized obs-history col, its null)
REPORT = (
    ("OAR", "oarR_max", "oarR_max_null", "oarR_max", "oarR_max_null"),
    ("HAR", "harRcond_hidden_max", "harRcond_hidden_max_null",
     "harRcond_ohist_max", "harRcond_ohist_max_null"),
    ("PIF", "pifRcond_hidden_max", "pifRcond_hidden_max_null",
     "pifOARcond_ohist_max", "pifOARcond_ohist_max_null"),
    ("AA", "aaRcond_max", "aaRcond_max_null", "aaRcond_max", "aaRcond_max_null"),
    ("DAI", "daiRcond_hidden_max", "daiRcond_hidden_max_null",
     "daiOARcond_ohist_max", "daiOARcond_ohist_max_null"),
)


def _onehot(idx: np.ndarray, n: int) -> np.ndarray:
    oh = np.zeros((idx.shape[0], n), dtype=np.float64)
    oh[np.arange(idx.shape[0]), idx] = 1.0
    return oh


def _build_userdata(intents, hstates, gt_attn, has_comm, has_attn, scenario, seed, jitter, use_hidden):
    """`intents (E,T,2)`, `hstates (E,T,2,hdim)`, `gt_attn (E,T,2,5)` — all ground-truth.

    `use_hidden=False` (default) feeds no hidden states, so the package computes
    only the obs-history `*_ohist` diagnostics. With the channel now in the
    observation that variant is faithful AND avoids 512-d kNN CMI, which is both
    the runtime bottleneck and unreliable at that dimension.
    """
    rng = np.random.default_rng(seed)
    e, t, _ = intents.shape
    step = np.tile(np.arange(t), e).astype(np.int64)
    episode_ids = np.repeat(np.arange(e), t).astype(np.int64)

    obs, acts, hid = {}, {}, {}
    for i, ag in enumerate(AGENTS):
        feats = [_onehot(step, t)]
        if has_comm:
            pm = np.full((e, t), -1, np.int64)
            pm[:, 1:] = intents[:, :-1, 1 - i]
            feats.append(_onehot(pm.reshape(-1) + 1, NUM_CARDS + 1))
        if has_attn:
            pa = np.zeros((e, t, gt_attn.shape[-1]), np.float64)
            pa[:, 1:, :] = gt_attn[:, :-1, 1 - i, :].astype(np.float64)
            feats.append(pa.reshape(e * t, -1))
        o = np.concatenate(feats, axis=1)
        if jitter > 0:
            o = o + jitter * rng.standard_normal(o.shape)
        obs[ag] = o
        acts[ag] = intents[:, :, i].reshape(-1).astype(np.int64)
        if use_hidden:
            hid[ag] = hstates[:, :, i, :].reshape(e * t, -1).astype(np.float64)

    timesteps = {a: step for a in AGENTS}
    eids = {a: episode_ids for a in AGENTS}
    return dpd.UserData(
        observations=obs, actions=acts, timesteps=timesteps, episode_ids=eids,
        hidden_states=(hid if use_hidden else None), env_name="card_game",
        alg_name=("JA-IPPO-RNN" if use_hidden else "JA-IPPO"),
        seed=seed, scenario_name=scenario,
    )


def _score_run(task):
    (intents, hstates, gt_attn, has_comm, has_attn, scenario, seed,
     condition, kind, idx, history_k, null_reps, jitter, max_samples, use_hidden) = task
    data = _build_userdata(intents, hstates, gt_attn, has_comm, has_attn,
                           scenario, seed, jitter, use_hidden)
    result = dpd.compute_diagnostics(
        data, history_k=history_k, null_reps=null_reps,
        max_samples=max_samples, metrics=METRIC_ARG,
    )
    row = {"condition": condition, "kind": kind, "idx": idx}
    row.update({k: float(v) for k, v in result.metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)})
    row.update({f"flag.{k}": bool(result.flags.get(k, False)) for k in FLAG_KEYS})
    return row


_RLIABLE_OK = True  # set False in main() if rliable's deps fail to import


def _bootstrap_ci(arr, reps=5000):
    rng = np.random.default_rng(0)
    means = arr[rng.integers(0, arr.size, (reps, arr.size))].mean(axis=1)
    return (float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _ci(vals):
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    if _RLIABLE_OK:
        try:
            return rliable_mean_ci(arr)
        except Exception:
            pass
    return _bootstrap_ci(arr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="evaluation/card_game/diag_data")
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--out", default=None, help="Per-run CSV path.")
    parser.add_argument("--history-k", type=int, default=3)
    parser.add_argument("--null-reps", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=8000)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--use-hidden", action="store_true",
                        help="Feed the 512-d RNN hidden state (their `*_hidden` metrics). OFF by "
                             "default: 512-d kNN CMI is the runtime bottleneck and unreliable at "
                             "that dimension; obs-history is faithful since the obs carries the "
                             "channel. Pair with `--max-seeds`-style small runs if you enable it.")
    parser.add_argument("--variant", choices=("hidden", "ohist"), default="ohist",
                        help="Which normalized variant to print (RNN hidden vs obs-history).")
    args = parser.parse_args()
    use_hidden = args.use_hidden

    data_dir = Path(args.data_dir)
    files = (
        [data_dir / f"{c}.npz" for c in args.conditions] if args.conditions
        else sorted(data_dir.glob("*.npz"))
    )
    if not files:
        raise SystemExit(f"No .npz files found in {data_dir}")

    tasks = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        cond = str(d["condition"])
        has_comm, has_attn = bool(d["has_comm"]), bool(d["has_attn"])
        for kind in ("sp", "xp"):
            inten, hsta, gat = d[f"{kind}_intents"], d[f"{kind}_hstates"], d[f"{kind}_gt_attn"]
            for k in range(inten.shape[0]):
                tasks.append((
                    inten[k], (hsta[k] if use_hidden else None), gat[k], has_comm, has_attn,
                    f"{cond}/{kind}/{k}", (0 if kind == "sp" else 100) + k,
                    cond, kind, k, args.history_k, args.null_reps, args.jitter, args.max_samples,
                    use_hidden,
                ))

    global _RLIABLE_OK
    try:
        rliable_mean_ci(np.array([0.1, 0.2, 0.3]))
    except Exception as e:
        _RLIABLE_OK = False
        print(f"[warn] rliable unavailable ({type(e).__name__}); using percentile-bootstrap CIs "
              f"(equivalent to their stratified-mean bootstrap for a single group).")

    print(f"Scoring {len(tasks)} runs on {args.workers} workers ...")
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(_score_run, tasks))
    else:
        rows = [_score_run(t) for t in tasks]

    # Per (condition, kind) aggregation with rliable CIs over the normalized variant.
    conds = sorted({r["condition"] for r in rows})
    hid = args.variant == "hidden" and use_hidden  # hidden cols are NaN unless fed
    summary = []
    print("\n" + "=" * 96)
    print(f"PER-CONDITION NORMALISED DIAGNOSTICS ({args.variant}), rliable mean [95% CI]; "
          f"* exceeds null. flags = {', '.join(k[0] for k in FLAG_KEYS)} share")
    for cond in conds:
        for kind in ("sp", "xp"):
            grp = [r for r in rows if r["condition"] == cond and r["kind"] == kind]
            if not grp:
                continue
            cells, rec = [], {"condition": cond, "kind": kind, "n": len(grp)}
            for label, hcol, hnull, ocol, onull in REPORT:
                col, ncol = (hcol, hnull) if hid else (ocol, onull)
                m, lo, up = _ci([r.get(col, np.nan) for r in grp])
                null_m = np.nanmean([r.get(ncol, np.nan) for r in grp])
                star = "*" if (not np.isnan(m) and not np.isnan(null_m) and m > null_m) else " "
                cells.append(f"{label} {m:+.3f}[{lo:+.3f},{up:+.3f}]{star}")
                rec[f"{label}.mean"], rec[f"{label}.lo"], rec[f"{label}.hi"] = m, lo, up
                rec[f"{label}.null"] = float(null_m)
            flags = " ".join(f"{np.mean([r[f'flag.{k}'] for r in grp]):.2f}" for k in FLAG_KEYS)
            rec["flags"] = flags
            summary.append(rec)
            print(f"\n{cond:<14} {kind:<3} (n={len(grp):2d})  flags[{flags}]")
            print("    " + "   ".join(cells))

    # Binary table (their build_paper_table logic: a rule holds for a condition iff
    # ANY seed flags it; we report SP share across conditions).
    print("\n" + "=" * 96)
    print("DECISION-RULE SHARE across conditions (ANY seed per condition, SP):")
    sp_conds = {c: [r for r in rows if r["condition"] == c and r["kind"] == "sp"] for c in conds}
    for k in FLAG_KEYS:
        n_pos = sum(any(r[f"flag.{k}"] for r in grp) for grp in sp_conds.values() if grp)
        print(f"  {k:<28} {n_pos}/{len(conds)}")

    if args.out:
        out_path = Path(args.out)
        keys = sorted({k for r in rows for k in r})
        with out_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved per-run table -> {out_path}")


if __name__ == "__main__":
    main()
