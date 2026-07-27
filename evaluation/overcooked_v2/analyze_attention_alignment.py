"""Test whether MATE-vs-OP attention agreement survives a peakedness-free metric.

The histogram intersection sum_e min(p1, p2) is maximised by DIFFUSE distributions, so
a method whose attention is spread over task objects scores higher than a peaked one
even with no agreement at all. MATE's aux spreads attention across task objects by
construction and OP's is near one-hot, so that metric conflates "agrees" with "diffuse".

This script reports, per method, over several episodes:
  - argmax agreement: do both agents' peak objects coincide? (invariant to peakedness)
  - the same split by whether the FOVs overlap, since egocentric attention maps have
    disjoint support when the views are disjoint, forcing intersection to 0 there
  - attention entropy, to quantify the peakedness confound directly
  - a null: agreement of agent 0 at time t with agent 1 at a shifted time, which keeps
    both marginals and destroys only the synchrony

GPU only. Usage:
    ./run_gpu.sh 2 evaluation.overcooked_v2.analyze_attention_alignment \
        --mate <ckpt> --op <ckpt> --episodes 8
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np

from evaluation.eval_first_insert import _base_state
from evaluation.overcooked_v2.extract_mate_figure_data import (
    _figure_objects,
    _fov_mask,
    _object_distribution,
    _run,
)


def _episode_stats(checkpoint, episode, no_op, obj_pos=None):
    conf, env, env_l, ctx, subdiv, ep_states, attn = _run(checkpoint, episode, no_op)
    if obj_pos is None:
        obj_pos, _ = _figure_objects(conf, env_l)
    n = min(len(attn["agent_0"]), len(attn["agent_1"]), len(ep_states))
    P = np.zeros((2, n, len(obj_pos)))
    for t in range(n):
        for i in (0, 1):
            P[i, t], _ = _object_distribution(attn[f"agent_{i}"][t], obj_pos, subdiv)
    shared = np.array([(_fov_mask(ep_states[t], ctx)[0]
                        & _fov_mask(ep_states[t], ctx)[1]).sum() for t in range(n)])
    return P, shared, obj_pos


def _report(tag, P, shared):
    ov = shared > 0
    agree = (P[0].argmax(1) == P[1].argmax(1))
    inter = np.minimum(P[0], P[1]).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.array([[-(p[p > 0] * np.log(p[p > 0])).sum() for p in P[i]]
                        for i in (0, 1)])
    # Null: pair agent 0 at t with agent 1 at t+17, preserving both marginals and
    # destroying only the timing. Real coordination must beat this.
    sh = 17
    null = (P[0].argmax(1)[:-sh] == P[1].argmax(1)[sh:])
    null_ov = null[ov[:-sh]]
    print(f"\n=== {tag} ===")
    print(f"  frames {len(shared)}   overlap {ov.sum()} ({100*ov.mean():.1f}%)")
    print(f"  argmax agreement   all {agree.mean():.3f} | overlap {agree[ov].mean():.3f} "
          f"| disjoint {agree[~ov].mean():.3f}")
    print(f"  time-shifted null  overlap {null_ov.mean():.3f}  "
          f"(agreement must beat this to mean anything)")
    print(f"  intersection       all {inter.mean():.3f} | overlap {inter[ov].mean():.3f} "
          f"| disjoint {inter[~ov].mean():.3f}")
    print(f"  attention entropy  a0 {ent[0].mean():.3f}  a1 {ent[1].mean():.3f} "
          f"(max {np.log(P.shape[2]):.3f}) -- the peakedness confound")
    return dict(agree_ov=agree[ov].mean(), null_ov=null_ov.mean(),
                inter_ov=inter[ov].mean(), ent=ent.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mate", required=True)
    p.add_argument("--op", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--keep-op", action="store_true")
    args = p.parse_args()
    no_op = not args.keep_op

    obj_pos = None
    acc = {}
    for tag, ckpt in (("MATE", args.mate), ("OP", args.op)):
        Ps, shs = [], []
        for ep in range(args.episodes):
            P, sh, obj_pos = _episode_stats(ckpt, ep, no_op, obj_pos)
            Ps.append(P)
            shs.append(sh)
        n = min(x.shape[1] for x in Ps)
        acc[tag] = _report(f"{tag}  ({args.episodes} episodes)",
                           np.concatenate([x[:, :n] for x in Ps], axis=1),
                           np.concatenate([s[:n] for s in shs]))

    m, o = acc["MATE"], acc["OP"]
    print("\n=== verdict (overlap frames only) ===")
    print(f"  argmax agreement   MATE {m['agree_ov']:.3f}  vs  OP {o['agree_ov']:.3f}")
    print(f"  vs its own null    MATE {m['null_ov']:.3f}        OP {o['null_ov']:.3f}")
    print(f"  entropy            MATE {m['ent']:.3f}        OP {o['ent']:.3f}")
    lift_m = m["agree_ov"] - m["null_ov"]
    lift_o = o["agree_ov"] - o["null_ov"]
    print(f"  agreement ABOVE null: MATE {lift_m:+.3f}   OP {lift_o:+.3f}")
    print("  -> a real synchrony effect needs MATE's lift over its own null to exceed "
          "OP's;\n     a gap in raw intersection alone is explained by entropy.")


if __name__ == "__main__":
    main()
