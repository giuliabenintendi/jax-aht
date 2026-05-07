"""Instrumented self-play diagnostic for one seed.

Runs N self-play episodes for a chosen seed (default seed 3), and at each
decision step prints the per-agent recolouring, raw policy action, and resulting
GT pick. Then aggregates: empirical P(SP coord), empirical P(both raw actions
equal), and the joint table of (raw_act_0, raw_act_1).

Useful to settle whether a measured "100% view-color X" policy actually picks
that view-color in the eval pipeline, vs whether the action distribution
collapses something subtler than that.

Usage:
    ./run_gpu.sh <gpu> evaluation.diagnose_sp_episode \\
        [--checkpoint /path/to/saved_train_run] \\
        [--seed-idx 3] \\
        [--num-episodes 200] \\
        [--use-best]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_dual_image_agent, initialize_ja_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import run_episode_with_states


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
    parser.add_argument("--seed-idx", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--use-best", action="store_true",
                        help="use best_params instead of final_params")
    parser.add_argument("--base-key", type=int, default=34957)
    parser.add_argument("--print-first", type=int, default=10,
                        help="print per-episode details for the first N episodes")
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
    env = LogWrapper(make_env(alg["ENV_NAME"], env_kwargs))
    inner_env = env._env
    max_steps = env_kwargs.get("max_steps", 8)
    print(f"checkpoint:  {ckpt_path}")
    print(f"env_kwargs:  {env_kwargs}")
    print(f"seed_idx:    {args.seed_idx}")
    print(f"num eps:     {args.num_episodes}")

    obs_type = alg.get("OBS_TYPE", env_kwargs.get("obs_type", "symbolic"))
    use_dual = alg.get("USE_DUAL_CRITIC", False)
    init_fn = (
        initialize_ja_dual_image_agent if use_dual
        else initialize_ja_image_agent
    )
    policy, _ = init_fn(alg, env, jax.random.PRNGKey(0))

    feed_attn_dims = None
    ja_card_masks = None
    feed_attn = alg.get("FEED_OTHER_ATTN", False)
    ja_card_attn = alg.get("JA_CARD_ATTN", False)
    ja_card_partner_feed = ja_card_attn and alg.get("JA_CARD_PARTNER_FEED", True)
    if feed_attn or ja_card_partner_feed:
        from agents.ja_image_actor_critic import _compute_resnet_output_dims
        from agents.ja_utils import build_card_masks
        img_h = inner_env.grid_height * inner_env.tile_size
        img_w = inner_env.grid_width * inner_env.tile_size
        feat_h, feat_w = _compute_resnet_output_dims(
            img_h, img_w,
            stride=alg.get("CONV_STRIDE", 2),
            kernel_size=alg.get("CONV_KERNEL_SIZE", 3),
            padding=alg.get("CONV_PADDING", "SAME"),
            num_blocks=alg.get("CONV_NUM_BLOCKS", 4),
        )
        if feed_attn:
            feed_attn_dims = (img_h, img_w, feat_h, feat_w)
        if ja_card_partner_feed:
            ja_card_masks = build_card_masks(img_h, img_w, feat_h, feat_w)

    data = load_train_run(str(ckpt_path))
    key = "best_params" if args.use_best else "final_params"
    if key not in data:
        raise KeyError(f"{key} not in checkpoint; have: {list(data.keys())}")
    params = jax.tree.map(lambda x: x[args.seed_idx], data[key])
    print(f"using {key}")

    # Run N episodes, collecting per-episode (raw_act_0, raw_act_1, pick_0_gt,
    # pick_1_gt, recol_0, recol_1, coord).
    coord_count = 0
    raw_match = 0
    raw_pair_counts = np.zeros((5, 5), dtype=int)
    gt_pair_counts = np.zeros((5, 5), dtype=int)
    raw_act_per_agent = {0: np.zeros(5, dtype=int), 1: np.zeros(5, dtype=int)}
    gt_pick_per_agent = {0: np.zeros(5, dtype=int), 1: np.zeros(5, dtype=int)}
    n = args.num_episodes
    for ep in range(n):
        ep_rng = jax.random.PRNGKey(args.base_key + ep)
        ep_states, _, ep_actions, ep_messages = run_episode_with_states(
            ep_rng, inner_env, params, policy, params, policy, max_steps,
            collect_attention=True, greedy=True,
            feed_other_attn_dims=feed_attn_dims,
            ja_card_masks=ja_card_masks,
        )
        # Decision step is the last entry of ep_actions/ep_messages.
        T = min(len(ep_actions), len(ep_messages))
        if T == 0:
            continue
        # Decision step picks (GT). ep_actions[T-1] is (pick_0_gt, pick_1_gt) at decision.
        gt_pick_0 = int(ep_actions[T - 1][0])
        gt_pick_1 = int(ep_actions[T - 1][1])
        # Recover raw policy actions from picks via the recolouring.
        last_state = ep_states[T - 1]
        recol_state = _find_attr(last_state, "per_agent_recolouring")
        r0 = np.asarray(recol_state.per_agent_recolouring["agent_0"])
        r1 = np.asarray(recol_state.per_agent_recolouring["agent_1"])
        # raw_act = recolouring[gt] (since recol[gt] = original action)
        raw_act_0 = int(r0[gt_pick_0]) if gt_pick_0 >= 0 else -1
        raw_act_1 = int(r1[gt_pick_1]) if gt_pick_1 >= 0 else -1

        if gt_pick_0 == gt_pick_1 and gt_pick_0 >= 0:
            coord_count += 1
        if raw_act_0 == raw_act_1 and raw_act_0 >= 0:
            raw_match += 1
        if 0 <= raw_act_0 < 5 and 0 <= raw_act_1 < 5:
            raw_pair_counts[raw_act_0, raw_act_1] += 1
        if 0 <= gt_pick_0 < 5 and 0 <= gt_pick_1 < 5:
            gt_pair_counts[gt_pick_0, gt_pick_1] += 1
        if 0 <= raw_act_0 < 5:
            raw_act_per_agent[0][raw_act_0] += 1
        if 0 <= raw_act_1 < 5:
            raw_act_per_agent[1][raw_act_1] += 1
        if 0 <= gt_pick_0 < 5:
            gt_pick_per_agent[0][gt_pick_0] += 1
        if 0 <= gt_pick_1 < 5:
            gt_pick_per_agent[1][gt_pick_1] += 1
        if ep < args.print_first:
            print(f"  ep {ep:3d}: r0={r0.tolist()} r1={r1.tolist()}  "
                  f"raw=({raw_act_0},{raw_act_1})  gt_pick=({gt_pick_0},{gt_pick_1})  "
                  f"coord={'YES' if gt_pick_0==gt_pick_1 else 'no'}")

    print()
    print(f"=== over {n} self-play decision steps ===")
    print(f"P(coord on GT pick)        : {coord_count}/{n} = {coord_count/n:.3f}")
    print(f"P(both raw actions equal)  : {raw_match}/{n}  = {raw_match/n:.3f}")
    print()
    print("agent_0 raw action histogram:", raw_act_per_agent[0].tolist())
    print("agent_1 raw action histogram:", raw_act_per_agent[1].tolist())
    print("agent_0 gt pick histogram   :", gt_pick_per_agent[0].tolist())
    print("agent_1 gt pick histogram   :", gt_pick_per_agent[1].tolist())
    print()
    print("joint (raw_act_0, raw_act_1) counts:")
    print("              raw_act_1: 0    1    2    3    4")
    for r in range(5):
        row = "  ".join(f"{raw_pair_counts[r,c]:3d}" for c in range(5))
        print(f"   raw_act_0={r}:    {row}")
    print("\njoint (gt_pick_0, gt_pick_1) counts:")
    print("              gt_pick_1: 0    1    2    3    4")
    for r in range(5):
        row = "  ".join(f"{gt_pair_counts[r,c]:3d}" for c in range(5))
        print(f"   gt_pick_0={r}:    {row}")


if __name__ == "__main__":
    main()
