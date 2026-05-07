"""Quick diagnostic: are per-agent OP recolourings actually independent at eval?

Resets the env 200 times with different keys and checks how often the two
agents' recolouring permutations land on the same index, and how often their
inverse-recolouring agree on view-color 1 specifically. Under uniform-over-S5
sampling (the wrapper's stated behaviour), expected:

  - identical recolourings:   200 / 120 ≈ 1–2 / 200
  - inv_rec[1] agreement:     200 * 1/5 = 40 / 200

If either number is much higher, OP recolouring is not independent across
agents, which would explain SP > 1/5 for a constant view-color policy.

Usage:
    ./run_gpu.sh <gpu> evaluation.diagnose_op_recolouring \\
        [--checkpoint /path/to/saved_train_run] \\
        [--num-resets 200]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from envs import make_env
from envs.log_wrapper import LogWrapper


DEFAULT_CKPT = (
    "/scratch/benintendi/jax-aht/results/card-game/ja_ippo/"
    "default_label/2026-05-04_23-42-20/saved_train_run"
)


def _find_attr(state, attr):
    s = state
    while s is not None:
        if hasattr(s, attr):
            return s
        s = getattr(s, "env_state", None)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--num-resets", type=int, default=200)
    parser.add_argument("--base-key", type=int, default=34957)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    cfg_dir = run_dir
    config_path = None
    for _ in range(4):
        cand = cfg_dir / ".hydra" / "config.yaml"
        if cand.exists():
            config_path = cand
            break
        cfg_dir = cfg_dir.parent
    if config_path is None:
        raise FileNotFoundError(f"No .hydra/config.yaml found near {run_dir}")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg = cfg["algorithm"]

    env_kwargs = dict(alg["ENV_KWARGS"])
    if alg.get("COMMUNICATION", False):
        env_kwargs["communication"] = True

    print(f"checkpoint:    {ckpt_path}")
    print(f"config:        {config_path}")
    print(f"env_kwargs:    {env_kwargs}")

    env = make_env(alg["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    n = args.num_resets
    identical = 0
    agree_on_1 = 0
    pair_counts = np.zeros((5, 5), dtype=int)
    saw_recol_state = False

    for i in range(n):
        _obs, st = env.reset(jax.random.PRNGKey(args.base_key + i))
        recol_state = _find_attr(st, "per_agent_recolouring")
        if recol_state is None:
            print("WARN: no per_agent_recolouring on env state — recolouring wrapper not active.")
            break
        saw_recol_state = True
        r0 = np.asarray(recol_state.per_agent_recolouring["agent_0"])
        r1 = np.asarray(recol_state.per_agent_recolouring["agent_1"])
        if np.array_equal(r0, r1):
            identical += 1
        inv_r0 = np.argsort(r0)
        inv_r1 = np.argsort(r1)
        if inv_r0[1] == inv_r1[1]:
            agree_on_1 += 1
        pair_counts[int(inv_r0[1]), int(inv_r1[1])] += 1
        if i < 3:
            print(f"  reset {i}: recol_0={r0.tolist()}  recol_1={r1.tolist()}  "
                  f"inv_r0[1]={int(inv_r0[1])}  inv_r1[1]={int(inv_r1[1])}")

    if not saw_recol_state:
        return

    print()
    print(f"=== over {n} fresh resets ===")
    print(f"identical recolourings: {identical}/{n}  (expected ~{n/120:.1f} if "
          f"uniform-over-S5)")
    print(f"inv_rec[1] agreement:   {agree_on_1}/{n}  (expected ~{n/5:.0f} under "
          f"independence; ~{int(0.625*n + 0.075*n)}-ish if SP=0.70 explained "
          f"by correlation)")

    print()
    print(f"joint distribution of (inv_r0[1], inv_r1[1]) across {n} resets:")
    print("           inv_r1[1]:  0    1    2    3    4")
    for r in range(5):
        row = "  ".join(f"{pair_counts[r,c]:3d}" for c in range(5))
        print(f"   inv_r0[1]={r}:    {row}")
    diag = np.diag(pair_counts).sum()
    off = pair_counts.sum() - diag
    print(f"\nsum on diagonal (matches): {diag}, off-diagonal: {off}")
    print(f"empirical P(match)       : {diag / max(pair_counts.sum(), 1):.3f}")
    print(f"if independent uniform   : 0.200")


if __name__ == "__main__":
    main()
