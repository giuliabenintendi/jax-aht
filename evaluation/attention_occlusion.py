"""Head-input attribution via occlusion.

For each attention head h, test how much its behaviour changes when different
regions of the input observation are occluded (set to 0). Tells us which
input regions drive each head.

Occlusion regions:
    - whole obs (sanity check)
    - timestep counter (top-left 6x6 pixels)
    - whole card row (y=7..14)
    - each card tile individually (5 separate occlusions)
    - non-card region (everything except card row)

For each (head, occlusion), reports:
    - argmax_change_rate: fraction of steps where head's view-slot argmax differs vs baseline
    - mean_jsd: mean JSD between baseline and occluded attention distributions

Usage:
    ./run_gpu.sh 5 evaluation.attention_occlusion \
        --checkpoint <path>/saved_train_run \
        --num-episodes 64
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.ja_utils import build_card_masks
from envs.card_game.rendering import (
    CARD_RECT_W, CARD_RECT_H, CARD_RECT_Y, NUM_CARDS, TILE_PIXELS,
)
from evaluation._card_game_utils import load_card_game_eval


IMG_H, IMG_W = 21, 35
FEAT_H, FEAT_W = 6, 9


def _occlusion_masks() -> dict:
    """name -> (H, W, 1) keep-mask. 1 = keep, 0 = occlude."""
    out = {}
    out["all_obs"] = np.zeros((IMG_H, IMG_W, 1), dtype=np.float32)

    keep = np.ones((IMG_H, IMG_W, 1), dtype=np.float32)
    keep[:6, :6, :] = 0.0
    out["timestep_counter"] = keep

    keep = np.ones((IMG_H, IMG_W, 1), dtype=np.float32)
    keep[CARD_RECT_Y:CARD_RECT_Y + CARD_RECT_H, :, :] = 0.0
    out["whole_card_row"] = keep

    for c in range(NUM_CARDS):
        keep = np.ones((IMG_H, IMG_W, 1), dtype=np.float32)
        x_lo = 1 + c * TILE_PIXELS
        x_hi = x_lo + CARD_RECT_W
        keep[CARD_RECT_Y:CARD_RECT_Y + CARD_RECT_H, x_lo:x_hi, :] = 0.0
        out[f"card_{c}"] = keep

    keep = np.zeros((IMG_H, IMG_W, 1), dtype=np.float32)
    keep[CARD_RECT_Y:CARD_RECT_Y + CARD_RECT_H, :, :] = 1.0
    out["non_card_region"] = keep

    return out


def _head_argmax_view_slot(head_attn_2d: np.ndarray, card_masks: np.ndarray) -> int:
    per_slot = np.einsum("hw,chw->c", head_attn_2d, card_masks)
    if per_slot.sum() < 1e-4:
        return -1
    return int(per_slot.argmax())


def _attention_jsd(p: np.ndarray, q: np.ndarray) -> float:
    """JSD between two 2D attention maps. Both flattened and normalized."""
    p = p.flatten() + 1e-8
    q = q.flatten() + 1e-8
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def _forward_attention(policy, params, obs_flat, done, avail, hstate, rng) -> np.ndarray:
    """Single forward pass; returns (fh, fw, num_heads) attention."""
    _, _, _, _, attn_map = policy.get_action_value_policy(
        params=params, obs=obs_flat, done=done,
        avail_actions=avail, hstate=hstate, rng=rng,
    )
    raw = np.asarray(attn_map).squeeze()
    if raw.ndim == 2:
        raw = raw[..., None]
    return raw  # (fh, fw, num_heads)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument("--seed-idx", type=int, default=-1)
    parser.add_argument("--episode-rng-base", type=int, default=200)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint)
    run_dir = Path(args.checkpoint).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else (run_dir / "attention_occlusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  label={ev.label}  seeds={ev.num_seeds}  eps={args.num_episodes}")
    print(f"  output: {out_dir}\n")

    card_masks = np.asarray(build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W))
    occ_masks = _occlusion_masks()
    occ_names = list(occ_masks.keys())
    img_flat_dim = IMG_H * IMG_W * 3

    seeds_to_run = ([args.seed_idx] if args.seed_idx >= 0
                    else list(range(ev.num_seeds)))

    csv_path = out_dir / "attention_occlusion.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "agent", "head", "occlusion",
                    "argmax_change_rate", "mean_jsd"])

        for seed_idx in seeds_to_run:
            params = jax.tree.map(lambda x: x[seed_idx], ev.params)

            for agent_idx, agent_key in enumerate(("agent_0", "agent_1")):
                # Per-(head, occlusion) counters; init after first forward
                num_heads = None
                arg_change_count = None  # [head][occlusion] -> int
                total_count = None
                jsd_sum = None

                for ep in range(args.num_episodes):
                    rng = jax.random.PRNGKey(
                        args.episode_rng_base + seed_idx * 1000 + ep
                    )
                    obs_dict, state = ev.env.reset(rng)
                    obs = obs_dict[agent_key]
                    hstate = ev.policy.init_hstate(1)
                    done = jnp.zeros((1, 1), dtype=bool)
                    avail = ev.env.get_avail_actions(state)[agent_key]
                    avail = jnp.asarray(avail).reshape(1, 1, -1).astype(jnp.float32)
                    obs_flat = jnp.asarray(obs).reshape(1, 1, -1)

                    # Baseline
                    raw_b = _forward_attention(
                        ev.policy, params, obs_flat, done, avail, hstate, rng,
                    )
                    if num_heads is None:
                        num_heads = raw_b.shape[-1]
                        arg_change_count = np.zeros((num_heads, len(occ_names)), dtype=np.int64)
                        total_count = np.zeros((num_heads, len(occ_names)), dtype=np.int64)
                        jsd_sum = np.zeros((num_heads, len(occ_names)), dtype=np.float64)
                    baseline_argmaxes = np.array(
                        [_head_argmax_view_slot(raw_b[..., h], card_masks)
                         for h in range(num_heads)]
                    )

                    for oi, name in enumerate(occ_names):
                        img_part = obs_flat[0, 0, :img_flat_dim].reshape(IMG_H, IMG_W, 3)
                        mask_b = jnp.asarray(occ_masks[name])
                        img_occluded = img_part * mask_b
                        suffix = obs_flat[0, 0, img_flat_dim:]
                        obs_occluded = jnp.concatenate(
                            [img_occluded.flatten(), suffix]
                        ).reshape(1, 1, -1)
                        raw_o = _forward_attention(
                            ev.policy, params, obs_occluded, done, avail, hstate, rng,
                        )

                        for h in range(num_heads):
                            argmax_b = baseline_argmaxes[h]
                            argmax_o = _head_argmax_view_slot(raw_o[..., h], card_masks)
                            arg_change_count[h, oi] += int(argmax_b != argmax_o)
                            total_count[h, oi] += 1
                            jsd_sum[h, oi] += _attention_jsd(
                                raw_b[..., h], raw_o[..., h]
                            )

                # Aggregate and report
                arg_change_rate = arg_change_count / np.maximum(total_count, 1)
                mean_jsd = jsd_sum / np.maximum(total_count, 1)

                print(f"=== seed {seed_idx}  {agent_key} ({num_heads} heads, {args.num_episodes} eps) ===")
                # Print: occlusion x head matrix of argmax_change_rate
                header = "  " + f"{'occlusion':<20s}" + "  ".join(
                    f"h{h}_chg  h{h}_jsd" for h in range(num_heads)
                )
                print(header)
                for oi, name in enumerate(occ_names):
                    cells = "  ".join(
                        f"{arg_change_rate[h, oi]:>6.3f}  {mean_jsd[h, oi]:>6.4f}"
                        for h in range(num_heads)
                    )
                    print(f"  {name:<20s}  {cells}")
                    for h in range(num_heads):
                        w.writerow([
                            seed_idx, agent_key, h, name,
                            f"{arg_change_rate[h, oi]:.4f}",
                            f"{mean_jsd[h, oi]:.4f}",
                        ])
                print()

    print(f"\nDone. CSV: {csv_path}")
    print("\nReading: argmax_change_rate > 0.5 means the head's argmax view-slot "
          "is sensitive to that occlusion. JSD > 0.1 means the head's full attention "
          "distribution shifts meaningfully. Heads with all-zero changes are "
          "input-invariant (likely doing scratch-space computation).")


if __name__ == "__main__":
    main()
