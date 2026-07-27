"""Post-delivery re-synchronization analysis on eval_first_insert traces.

Every delivery resamples the recipe (sample_recipe_on_delivery), so an episode
is a sequence of re-coordination events. Deliveries where the recipe stays the
same need no re-sync and act as the control; deliveries where it switches
force the pair to re-communicate. Per config this reports how the pair pays
that recurring cost: wrong inserts after a switch, latency to the first
correct insert, and who moves first (the recurring version of the
first-insert metric, measured at steady state instead of the episode opening).

Usage:
    uv run python -m evaluation.analyze_resync --traces ~/first_insert_traces/ippo --mode greedy
"""
import argparse
import glob
import os

import numpy as np

from evaluation.eval_first_insert import ingredient_counts


def episode_inserts(pot_row, inv_row, acts_row, ax_row, T, pot_x):
    """All inserts in one episode: list of (t, ingredient, agent, agent_is_left)."""
    cnt = ingredient_counts(pot_row[:T])
    diff = np.diff(cnt, axis=0)
    out = []
    for t in np.nonzero(diff.sum(-1) > 0)[0]:
        ing = int(diff[t].argmax())
        who = -1
        for a in range(2):
            held = ingredient_counts(inv_row[t, a])[..., ing] > 0
            if acts_row[t, a] == 5 and held:  # Actions.interact
                who = a if who == -1 else 2
        left = bool(ax_row[t, who if who in (0, 1) else 0] < pot_x)
        out.append((int(t), ing, who, left))
    return out


def analyze_seed(d):
    valid = d["valid"]
    recipe = d["recipe"]
    delivery = d["delivery"]
    pot_x = int(d["pot_yx"][1])
    obs_left = bool(d["observer_side_left"])
    E = valid.shape[0]

    ev = {"switch": [], "same": []}
    for e in range(E):
        T = int(valid[e].sum())
        if T < 2:
            continue
        inserts = episode_inserts(d["pot"][e], d["inv"][e], d["acts"][e],
                                  d["ax"][e], T, pot_x)
        for t in np.nonzero(delivery[e, :T])[0]:
            if t + 1 >= T:
                continue
            switched = recipe[e, t + 1] != recipe[e, t]
            post = [i for i in inserts if i[0] > t]
            if not post:
                continue
            t1, ing1, who1, left1 = post[0]
            rec = recipe[e, min(t1, T - 1)]
            correct1 = bool(ingredient_counts(rec)[..., ing1] > 0)
            # wrong inserts and latency until the first correct one
            wrong = 0
            lat = None
            for (ti, ingi, _, _) in post:
                ri = recipe[e, min(ti, T - 1)]
                if ingredient_counts(ri)[..., ingi] > 0:
                    lat = ti - t
                    break
                wrong += 1
            by_nonobs = (left1 != obs_left) if who1 in (0, 1) else None
            ev["switch" if switched else "same"].append(
                dict(correct1=correct1, wrong=wrong, lat=lat, by_nonobs=by_nonobs))
    res = {}
    for k, rows in ev.items():
        n = len(rows)
        lats = [r["lat"] for r in rows if r["lat"] is not None]
        nob = [r["by_nonobs"] for r in rows if r["by_nonobs"] is not None]
        res[k] = dict(
            n=n,
            p_first_correct=float(np.mean([r["correct1"] for r in rows])) if n else np.nan,
            mean_wrong=float(np.mean([r["wrong"] for r in rows])) if n else np.nan,
            mean_latency=float(np.mean(lats)) if lats else np.nan,
            p_first_by_nonobs=float(np.mean(nob)) if nob else np.nan,
        )
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traces", required=True, help="dir with <label>_<mode>.npz files")
    p.add_argument("--mode", default="greedy", choices=["greedy", "sampled"])
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.traces),
                                          f"*_{args.mode}.npz")))
    if not files:
        raise SystemExit(f"no *_{args.mode}.npz under {args.traces}")

    agg = {"switch": {k: [] for k in ("p_first_correct", "mean_wrong",
                                      "mean_latency", "p_first_by_nonobs")},
           "same": {k: [] for k in ("p_first_correct", "mean_wrong",
                                    "mean_latency", "p_first_by_nonobs")}}
    for f in files:
        d = np.load(f)
        res = analyze_seed(d)
        label = os.path.basename(f).replace(f"_{args.mode}.npz", "")
        print(f"{label}:")
        for k in ("switch", "same"):
            r = res[k]
            print(f"  {k:6s} n={r['n']:4d}  p_first_correct={r['p_first_correct']:.3f}  "
                  f"wrong={r['mean_wrong']:.2f}  latency={r['mean_latency']:.1f}  "
                  f"p_first_by_nonobs={r['p_first_by_nonobs']:.3f}")
            for m in agg[k]:
                if not np.isnan(r[m]):
                    agg[k][m].append(r[m])

    print("\n== config means +/- sem over seeds ==")
    for k in ("switch", "same"):
        parts = []
        for m, vals in agg[k].items():
            v = np.asarray(vals)
            parts.append(f"{m}={v.mean():.3f}+/-{v.std() / np.sqrt(len(v)):.3f}")
        print(f"  {k:6s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
