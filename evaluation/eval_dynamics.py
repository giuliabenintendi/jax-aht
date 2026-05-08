"""Run coordination + role-pattern diagnostics on a saved card-game checkpoint.

Loads a `saved_train_run` checkpoint, replays N self-play eval episodes per
seed, and writes coordination_lock_in.png + role_dynamics.png plus prints
the textual breakdown — without re-running training.

Usage:
    ./run_gpu.sh <gpu> evaluation.eval_dynamics \\
        --checkpoint /path/to/saved_train_run \\
        --num-episodes 50 \\
        --output-dir plots/card_game/<tag>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax

from agents.initialize_agents import (
    initialize_ja_agent,
    initialize_ja_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.eval_card_game import _log_card_game_dynamics
from omegaconf import OmegaConf


class _NoopLogger:
    def log(self, *args, **kwargs):
        pass


def _get_obs_type(cfg: dict) -> str:
    return cfg.get("ENV_KWARGS", {}).get("obs_type", "image")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved_train_run directory.")
    parser.add_argument("--seed-idx", type=int, default=None,
                        help="If set, only run for this seed; otherwise run all.")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--output-dir", default="plots/card_game/dynamics")
    parser.add_argument("--use-best", action="store_true",
                        help="Use best_params instead of final_params.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find hydra config at {cfg_path}")
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    algo_cfg = cfg["algorithm"]

    run_data = load_train_run(str(ckpt_path))
    params_key = "best_params" if (args.use_best and "best_params" in run_data) else "final_params"
    if params_key not in run_data:
        params_key = "final_params"
    stacked_params = run_data[params_key]
    num_seeds = jax.tree.leaves(stacked_params)[0].shape[0]

    # Apply training-time runtime overrides to env_kwargs (mirroring
    # `run_ja_ippo`): COMMUNICATION enables the env's comm channel, and
    # for card-game we disable scramble_partner_msg at eval time.
    env_kwargs = dict(algo_cfg["ENV_KWARGS"])
    if algo_cfg.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    if algo_cfg.get("ENV_NAME") == "card-game":
        env_kwargs.setdefault("scramble_partner_msg", False)
    env = make_env(algo_cfg["ENV_NAME"], env_kwargs)
    env = LogWrapper(env)

    obs_type = _get_obs_type(algo_cfg)
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algo_cfg, env, rng)

    seeds = [args.seed_idx] if args.seed_idx is not None else list(range(num_seeds))
    max_steps = int(algo_cfg.get("ROLLOUT_LENGTH", algo_cfg.get("ENV_KWARGS", {}).get("max_steps", 8)))

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    logger = _NoopLogger()

    for s in seeds:
        seed_params = jax.tree.map(lambda x: x[s], stacked_params)
        seed_dir = out_root / f"seed_{s}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        tag = f"Eval/seed_{s}"
        _log_card_game_dynamics(
            env._env, policy, seed_params, max_steps,
            tag, str(seed_dir), logger,
            num_episodes=args.num_episodes,
        )
        print(f"[eval_dynamics] seed {s}: outputs in {seed_dir}")

    print(f"\nDone. All outputs under {out_root}/")


if __name__ == "__main__":
    main()
