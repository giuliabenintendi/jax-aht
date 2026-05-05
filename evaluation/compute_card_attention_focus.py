"""Quantify per-step attention focus on coordination targets (card game).

For each step in each episode, computes how much of each agent's attention
mass falls on:
  - the card the agent will pick at the decision step (`pick_mass`)
  - the card the partner is currently signalling via the message dot
    (`partner_msg_mass`); empty at step 0 since no message has been sent yet
  - the remaining card tiles (`other_cards_mass`)
  - non-card regions (`non_card_mass` = 1 - total mass on all cards)

Card masks come from `build_card_masks`, indexed by image column. This matches
the agent's view column under Other-Play because attention is computed over the
agent's own-view obs. Pick and message indices live in GT-card-id space and are
mapped to view columns through the per-agent OP permutation.

Note on overlap: when the agent's pick equals the partner's current message,
`pick_mass == partner_msg_mass` and the four masses no longer partition (their
sum exceeds 1). The CSV records `pick_overlaps_partner_msg` so those rows can
be filtered or re-grouped post-hoc.

Outputs:
  - Printed per-step summary per seed (averaged across episodes).
  - CSV with raw per-step-per-agent-per-episode breakdown.

Usage:
    ./run_gpu.sh <gpu> evaluation.compute_card_attention_focus \\
        --checkpoint /path/to/saved/train_run \\
        --num-episodes 100 \\
        --output-dir results/.../attn_focus
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_agent,
    initialize_ja_image_agent,
)
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import build_card_masks
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import NUM_CARDS
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import run_episode_with_states


def _resolve_run_config(ckpt_path: Path) -> dict:
    """Walk parents to find .hydra/config.yaml (matches analyze_attention.py)."""
    run_dir = ckpt_path.parent if ckpt_path.is_file() else ckpt_path
    cur = run_dir
    for _ in range(4):
        candidate = cur / ".hydra" / "config.yaml"
        if candidate.exists():
            return OmegaConf.to_container(OmegaConf.load(candidate), resolve=True)
        cur = cur.parent
    raise FileNotFoundError(f"Could not locate .hydra/config.yaml near {run_dir}")


def _get_obs_type(alg_config) -> str:
    return alg_config.get(
        "OBS_TYPE",
        alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"),
    )


def _walk(state, attr: str):
    s = state
    while s is not None and not hasattr(s, attr):
        s = getattr(s, "env_state", None)
    return s


def _gt_to_view_col(state, agent_idx: int, gt_card_id: int) -> int | None:
    """Map a GT card identity to the agent's current view column.

    Returns None when no GT id is available (e.g. step 0 with no partner
    message yet) or when the wrappers don't expose the needed info.
    """
    if gt_card_id is None or gt_card_id < 0:
        return None
    cg = _walk(state, "card_permutation")
    if cg is None:
        return None
    card_perm = np.asarray(cg.card_permutation)
    phys_cols = np.where(card_perm == int(gt_card_id))[0]
    if len(phys_cols) == 0:
        return None
    phys_col = int(phys_cols[0])
    op = _walk(state, "per_agent_perm")
    if op is None:
        # Without OP position-shuffle, view column = physical column.
        return phys_col
    pos_perm = np.asarray(op.per_agent_perm[f"agent_{agent_idx}"])
    view_cols = np.where(pos_perm == phys_col)[0]
    return int(view_cols[0]) if len(view_cols) else None


def _resolve_eval_inputs(alg_config, inner_env):
    """Compute card masks (always) and the optional policy-side inputs.

    Returns (card_masks, feed_attn_dims, masks_for_policy):
      - card_masks: (NUM_CARDS, feat_h, feat_w) numpy array, used for
        attention-mass measurement regardless of how the policy was trained.
      - feed_attn_dims: 4-tuple if FEED_OTHER_ATTN, else None.
      - masks_for_policy: card_masks (jax) if JA_CARD_ATTN, else None — needed
        so the policy receives the same translated-partner-attention input it
        had at training time.
    """
    feed_attn = alg_config.get("FEED_OTHER_ATTN", False)
    ja_card_attn = alg_config.get("JA_CARD_ATTN", False)
    img_h = inner_env.grid_height * inner_env.tile_size
    img_w = inner_env.grid_width * inner_env.tile_size
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=alg_config.get("CONV_STRIDE", 2),
        kernel_size=alg_config.get("CONV_KERNEL_SIZE", 3),
        padding=alg_config.get("CONV_PADDING", "SAME"),
        num_blocks=alg_config.get("CONV_NUM_BLOCKS", 4),
    )
    masks_jax = build_card_masks(img_h, img_w, feat_h, feat_w)
    card_masks = np.asarray(masks_jax)
    feed_dims = (img_h, img_w, feat_h, feat_w) if feed_attn else None
    masks_for_policy = masks_jax if ja_card_attn else None
    return card_masks, feed_dims, masks_for_policy


def _per_step_focus(
    attn_map,
    card_masks: np.ndarray,
    pick_view_col: int | None,
    partner_msg_view_col: int | None,
) -> dict:
    """Decompose one (agent, step) attention map by target."""
    attn = np.asarray(attn_map).squeeze().astype(np.float64)
    per_card = np.einsum("hw,chw->c", attn, card_masks)  # (NUM_CARDS,)
    total_card = float(per_card.sum())
    non_card = max(0.0, 1.0 - total_card)
    pick_mass = (
        float(per_card[pick_view_col]) if pick_view_col is not None else 0.0
    )
    partner_mass = (
        float(per_card[partner_msg_view_col])
        if partner_msg_view_col is not None else 0.0
    )
    excluded: set[int] = set()
    if pick_view_col is not None:
        excluded.add(pick_view_col)
    if partner_msg_view_col is not None:
        excluded.add(partner_msg_view_col)
    other_indices = [c for c in range(NUM_CARDS) if c not in excluded]
    other_mass = float(per_card[other_indices].sum())
    overlap = (
        pick_view_col is not None and partner_msg_view_col is not None
        and pick_view_col == partner_msg_view_col
    )
    argmax_col = int(np.argmax(per_card))
    return {
        "per_card": per_card,
        "non_card": non_card,
        "pick": pick_mass,
        "partner_msg": partner_mass,
        "other_cards": other_mass,
        "pick_overlaps_partner": overlap,
        "argmax_view_col": argmax_col,
        "argmax_mass": float(per_card[argmax_col]),
    }


def _print_seed_table(
    seed_idx: int,
    successes: list[bool],
    per_step: dict,
    max_steps: int,
):
    n_eps = len(successes)
    sr = float(np.mean(successes)) if successes else float("nan")
    print(f"\n=== seed {seed_idx}  (eps={n_eps}, success_rate={sr:.3f}) ===")
    for agent_idx, key in enumerate(("agent_0", "agent_1")):
        print(f"  {key}:")
        print("    step | pick    pmsg    other   non_card | argmax_col argmax_mass")
        for t in range(max_steps):
            stash = per_step[agent_idx]
            if not stash["pick"][t]:
                continue
            p = float(np.mean(stash["pick"][t]))
            m = float(np.mean(stash["pmsg"][t]))
            o = float(np.mean(stash["other"][t]))
            nc = float(np.mean(stash["ncard"][t]))
            am = float(np.mean(stash["argmax_mass"][t]))
            ac_arr = np.asarray(stash["argmax_col"][t])
            # Modal argmax column for the printed table; CSV preserves per-ep values.
            ac = int(np.bincount(ac_arr, minlength=NUM_CARDS).argmax())
            tag = "D" if t == max_steps - 1 else " "
            print(
                f"    {t:>2}{tag}  | {p:.3f}   {m:.3f}   {o:.3f}   {nc:.3f}    "
                f"|     {ac}       {am:.3f}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to saved train run checkpoint directory")
    parser.add_argument("--seed-idx", type=int, default=0,
                        help="Single seed to analyse; ignored when --all-seeds.")
    parser.add_argument("--all-seeds", action="store_true",
                        help="Analyse every seed in the checkpoint.")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--episode-rng-base", type=int, default=100,
                        help="Base seed for per-episode RNGs; key = base + ep + seed*10000")
    parser.add_argument("--output-dir", default="results/card_game_attn_focus")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    cfg = _resolve_run_config(ckpt_path)
    alg_config = cfg["algorithm"]

    if alg_config.get("COMMUNICATION", False):
        env_kwargs = dict(alg_config["ENV_KWARGS"])
        env_kwargs["communication"] = True
        alg_config["ENV_KWARGS"] = env_kwargs

    env = make_env(alg_config["ENV_NAME"], alg_config["ENV_KWARGS"])
    env = LogWrapper(env)
    inner_env = env._env
    max_steps = int(alg_config["ENV_KWARGS"].get("max_steps", 8))

    obs_type = _get_obs_type(alg_config)
    init_fn = (
        initialize_ja_image_agent if obs_type in ("image", "fov")
        else initialize_ja_agent
    )
    policy, _ = init_fn(alg_config, env, jax.random.PRNGKey(0))

    card_masks, feed_attn_dims, masks_for_policy = _resolve_eval_inputs(
        alg_config, inner_env,
    )

    run_data = load_train_run(str(ckpt_path))
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    if args.all_seeds:
        seed_indices = list(range(num_seeds))
    else:
        if args.seed_idx >= num_seeds:
            raise ValueError(
                f"seed_idx={args.seed_idx} out of range for {num_seeds} seeds"
            )
        seed_indices = [args.seed_idx]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_step_attention_focus.csv"

    print(f"Loaded checkpoint: {ckpt_path}")
    print(
        f"  seeds in ckpt: {num_seeds}, analysing: {seed_indices}, "
        f"episodes/seed: {args.num_episodes}, max_steps: {max_steps}"
    )
    print(
        f"  FEED_OTHER_ATTN={alg_config.get('FEED_OTHER_ATTN', False)}, "
        f"JA_CARD_ATTN={alg_config.get('JA_CARD_ATTN', False)}"
    )

    fieldnames = [
        "seed", "episode", "step", "is_decision", "agent",
        "pick_view_col", "partner_msg_view_col",
        "card_mass_0", "card_mass_1", "card_mass_2", "card_mass_3", "card_mass_4",
        "non_card_mass", "pick_mass", "partner_msg_mass", "other_cards_mass",
        "pick_overlaps_partner_msg",
        "argmax_view_col", "argmax_mass",
        "success",
    ]
    rows: list[dict] = []

    for seed_idx in seed_indices:
        params = jax.tree.map(lambda x, _i=seed_idx: x[_i], final_params)
        per_step = {
            i: {
                "pick": [[] for _ in range(max_steps)],
                "pmsg": [[] for _ in range(max_steps)],
                "other": [[] for _ in range(max_steps)],
                "ncard": [[] for _ in range(max_steps)],
                "argmax_col": [[] for _ in range(max_steps)],
                "argmax_mass": [[] for _ in range(max_steps)],
            }
            for i in (0, 1)
        }
        successes: list[bool] = []

        for ep in range(args.num_episodes):
            ep_rng = jax.random.PRNGKey(
                args.episode_rng_base + ep + seed_idx * 10_000,
            )
            ep_states, attn_maps, ep_actions, ep_messages = run_episode_with_states(
                ep_rng, inner_env, params, policy, params, policy, max_steps,
                collect_attention=True,
                feed_other_attn_dims=feed_attn_dims,
                ja_card_masks=masks_for_policy,
            )

            n_steps = min(
                len(attn_maps["agent_0"]),
                len(attn_maps["agent_1"]),
                max_steps,
            )
            if n_steps == 0:
                continue

            final_pick = ep_actions[-1] if ep_actions else (-1, -1)
            success = bool(
                int(final_pick[0]) >= 0
                and int(final_pick[0]) == int(final_pick[1])
            )
            successes.append(success)

            for t in range(n_steps):
                is_decision = (t == n_steps - 1)
                # Partner message visible at step t was emitted at step t-1.
                # At step 0 nothing has been said yet.
                if t >= 1 and (t - 1) < len(ep_messages):
                    partner_msgs_gt = ep_messages[t - 1]
                else:
                    partner_msgs_gt = (-1, -1)

                for agent_idx, key in enumerate(("agent_0", "agent_1")):
                    pick_view = _gt_to_view_col(
                        ep_states[t], agent_idx, int(final_pick[agent_idx]),
                    )
                    partner_gt = int(partner_msgs_gt[1 - agent_idx])
                    pmsg_view = _gt_to_view_col(
                        ep_states[t], agent_idx, partner_gt,
                    )

                    res = _per_step_focus(
                        attn_maps[key][t], card_masks, pick_view, pmsg_view,
                    )
                    stash = per_step[agent_idx]
                    stash["pick"][t].append(res["pick"])
                    stash["pmsg"][t].append(res["partner_msg"])
                    stash["other"][t].append(res["other_cards"])
                    stash["ncard"][t].append(res["non_card"])
                    stash["argmax_col"][t].append(res["argmax_view_col"])
                    stash["argmax_mass"][t].append(res["argmax_mass"])

                    rows.append({
                        "seed": seed_idx, "episode": ep, "step": t,
                        "is_decision": int(is_decision), "agent": agent_idx,
                        "pick_view_col": pick_view if pick_view is not None else -1,
                        "partner_msg_view_col": (
                            pmsg_view if pmsg_view is not None else -1
                        ),
                        **{
                            f"card_mass_{c}": float(res["per_card"][c])
                            for c in range(NUM_CARDS)
                        },
                        "non_card_mass": res["non_card"],
                        "pick_mass": res["pick"],
                        "partner_msg_mass": res["partner_msg"],
                        "other_cards_mass": res["other_cards"],
                        "pick_overlaps_partner_msg": int(
                            res["pick_overlaps_partner"]
                        ),
                        "argmax_view_col": res["argmax_view_col"],
                        "argmax_mass": res["argmax_mass"],
                        "success": int(success),
                    })

        _print_seed_table(seed_idx, successes, per_step, max_steps)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {csv_path.resolve()}")


if __name__ == "__main__":
    main()
