"""Trace num_scalars resolution for the misunderstood-feather-1321 checkpoint.

Why: load fails with ScopeParamShapeError expecting (20, 5) but generator made (0, 5).
This isolates whether num_scalars resolves to 20 (expected with per-head feed) or 0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_image_agent, _get_image_dims
from envs import make_env
from envs.log_wrapper import LogWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Hydra run dir (parent of saved_train_run/)")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    alg = cfg["algorithm"]

    print("=== flags from saved config ===")
    for k in ("JA_CARD_ATTN", "JA_CARD_PARTNER_FEED",
              "JA_PARTNER_FEED_PER_HEAD", "JA_NUM_HEADS",
              "COMMUNICATION"):
        v = alg.get(k)
        print(f"  {k:30s} = {v!r}  (type={type(v).__name__})")

    env_kwargs = dict(alg.get("ENV_KWARGS", {}))
    if alg.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    env_kwargs["scramble_partner_msg"] = False
    env = make_env(alg["ENV_NAME"], env_kwargs)
    env_wrapped = LogWrapper(env)

    print("\n=== env wrapper chain ===")
    cur = env_wrapped
    for i in range(6):
        print(f"  L{i}: {type(cur).__name__}  num_scalar_obs="
              f"{getattr(cur, 'num_scalar_obs', '<missing>')}")
        if not hasattr(cur, "_env"):
            break
        cur = cur._env

    h, w, num_scalars_env = _get_image_dims(env_wrapped)
    print(f"\n_get_image_dims -> H={h} W={w} num_scalars={num_scalars_env}")

    # Replicate initialize_ja_image_agent's scalar bookkeeping
    num_scalars = num_scalars_env
    if alg.get("JA_CARD_ATTN", False) and alg.get("JA_CARD_PARTNER_FEED", True):
        if alg.get("JA_PARTNER_FEED_PER_HEAD", False):
            num_scalars += 5 * alg.get("JA_NUM_HEADS", 4)
        else:
            num_scalars += 5
    print(f"final num_scalars (manual replay) = {num_scalars}  "
          f"(expected: 20 for per-head feed)")

    # Now actually invoke the real init and check the param tree
    print("\n=== invoking initialize_ja_image_agent ===")
    policy, params = initialize_ja_image_agent(
        alg, env_wrapped, jax.random.PRNGKey(0)
    )

    def find_scalar_embed(tree, path=""):
        if hasattr(tree, "keys"):
            for k, v in tree.items():
                find_scalar_embed(v, f"{path}/{k}")
        else:
            if "scalar_embed" in path:
                print(f"  {path}: shape={getattr(tree, 'shape', tree)}")

    find_scalar_embed(params)


if __name__ == "__main__":
    main()
