"""Attention / message filmstrips for card-game checkpoints.

For chosen self-play and cross-play episodes, saves a 2xT filmstrip:
row 0 = agent 0, row 1 = agent 1, one column per step. Each cell shows the
agent's own (OP-transformed) observation with its attention heat overlaid;
the picked card is boxed (yellow on coordination success). For the
explicit-communication checkpoint the partner-message dot is also drawn into
each agent's view.

The proposer/follower asymmetry is read directly off a strip: the proposer's
row locks onto one card from t=0, the follower's row drifts then matches.
The per-episode settle step (see eval_time_to_agree) is printed so episodes
can be picked by behaviour.

Usage:
    ./run_gpu.sh 0 evaluation.card_game.eval_filmstrips \\
        --checkpoint <path_to_saved_train_run> \\
        --sp-seeds 0 --xp-pairs "0,1" --num-episodes 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax

from agents.ja_utils import build_card_masks
from evaluation.card_game._card_game_utils import load_card_game_eval
from evaluation.card_game.eval_time_to_agree import (
    FEAT_H,
    FEAT_W,
    IMG_H,
    IMG_W,
    episode_agreement,
    extract_intentions,
    setup_card_feed,
)
from evaluation.vis_episodes import run_episode_with_states
from marl.eval_card_game import (
    _log_card_game_attention_grid,
    _log_card_game_gt_attn_filmstrip,
)


class _NullLogger:
    """No-op stand-in for the wandb run object expected by the grid logger."""

    def log(self, *args, **kwargs) -> None:
        pass


def _parse_ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_pairs(spec: str) -> list[tuple[int, int]]:
    """Parse 'i,j;i,j' into a list of `(int, int)` seed pairs."""
    pairs: list[tuple[int, int]] = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        i, j = (int(x) for x in part.split(","))
        pairs.append((i, j))
    return pairs


def _rollout(ev, params_a, params_b, rng, partner_feed_dim, card_masks, greedy):
    return run_episode_with_states(
        rng, ev.env, params_a, ev.policy, params_b, ev.policy, ev.max_steps,
        collect_attention=True, collect_obs=True, greedy=greedy,
        ja_card_masks=card_masks, partner_feed_dim=partner_feed_dim,
    )


def _save_strip(attn_maps, ep_actions, ep_messages, ep_obs, ep_states,
                out_dir: Path, out_name: str) -> None:
    """Render and save one 2xT attention/message filmstrip PNG."""
    _log_card_game_attention_grid(
        frames=None, attn_data=attn_maps, ep_actions=ep_actions,
        tag="filmstrip", video_dir=str(out_dir), logger=_NullLogger(),
        ep_messages=ep_messages or None, card_permutation=None,
        ep_obs=ep_obs, ep_states=ep_states, out_name=out_name,
    )


def _save_gt_strip(attn_maps, ep_actions, ep_messages, ep_states,
                   viz_card_masks, out_dir: Path, out_name: str) -> None:
    """Render and save one 2xT GT-frame coolwarm within-card spatial
    attention filmstrip PNG."""
    _log_card_game_gt_attn_filmstrip(
        attn_data=attn_maps, ep_actions=ep_actions, ep_states=ep_states,
        ja_card_masks=viz_card_masks, video_dir=str(out_dir),
        out_name=out_name, tag="filmstrip", logger=None,
        ep_messages=ep_messages or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sp-seeds", default="0",
                        help="Comma-separated seed indices for self-play strips.")
    parser.add_argument("--xp-pairs", default="0,1",
                        help="Cross-play pairs 'i,j;i,j' — agent 0 uses seed i, "
                             "agent 1 uses seed j.")
    parser.add_argument("--num-episodes", type=int, default=8,
                        help="Episodes (strips) per seed / per pair.")
    parser.add_argument("--rng-base", type=int, default=20260520)
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to <run_dir>/filmstrips/")
    parser.add_argument("--sampled", action="store_true",
                        help="Sampled actions (default greedy).")
    parser.add_argument("--latest", action="store_true",
                        help="Use the last saved chunk per seed instead of "
                             "the best-by-return checkpoint.")
    args = parser.parse_args()

    ev = load_card_game_eval(args.checkpoint, use_latest=args.latest)
    greedy = not args.sampled
    T = ev.max_steps
    partner_feed_dim, card_masks = setup_card_feed(ev)
    # The GT-frame filmstrip needs the (5, fh, fw) view-slot card masks
    # regardless of whether the policy receives the partner-card-attention
    # augmentation, so build them locally from the obs/feat geometry.
    viz_card_masks = build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W)

    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(args.checkpoint).resolve().parent / "filmstrips")
    out_dir.mkdir(parents=True, exist_ok=True)

    sp_seeds = _parse_ints(args.sp_seeds)
    xp_pairs = _parse_pairs(args.xp_pairs)
    print(
        f"\nCheckpoint: {args.checkpoint}"
        f"\n  label={ev.label}  comm={ev.env_kwargs.get('communication', False)}"
        f"  seeds={ev.num_seeds}  max_steps={T}  greedy={greedy}"
        f"\n  SP seeds={sp_seeds}  XP pairs={xp_pairs}  output={out_dir}"
    )

    summary: list[str] = []

    def _run_block(tag: str, params_a, params_b, rng_offset: int) -> None:
        for ep in range(args.num_episodes):
            rng = jax.random.PRNGKey(args.rng_base + rng_offset * 1000 + ep)
            ep_states, attn_maps, ep_actions, ep_messages, ep_obs = _rollout(
                ev, params_a, params_b, rng, partner_feed_dim, card_masks, greedy,
            )
            intents = extract_intentions(ep_actions, ep_messages, T)
            _, settle = episode_agreement(intents)
            success = settle < T
            out_name = f"filmstrip_{tag}_ep{ep}.png"
            _save_strip(attn_maps, ep_actions, ep_messages, ep_obs, ep_states,
                        out_dir, out_name)
            gt_out_name = f"gt_filmstrip_{tag}_ep{ep}.png"
            _save_gt_strip(
                attn_maps, ep_actions, ep_messages, ep_states,
                viz_card_masks, out_dir, gt_out_name,
            )
            settle_str = str(settle) if success else "never"
            summary.append(
                f"  {out_name}: settle={settle_str}  "
                f"success={'yes' if success else 'no'}"
            )
            print(f"[saved] {out_dir / out_name}  (settle={settle_str})")
            print(f"[saved] {out_dir / gt_out_name}")

    for s in sp_seeds:
        params = jax.tree.map(lambda x: x[s], ev.params)
        _run_block(f"sp_seed{s}", params, params, rng_offset=s)

    for i, j in xp_pairs:
        params_i = jax.tree.map(lambda x: x[i], ev.params)
        params_j = jax.tree.map(lambda x: x[j], ev.params)
        _run_block(f"xp_{i}v{j}", params_i, params_j, rng_offset=100 + i * 12 + j)

    print("\nSummary (pick strips by settle step):")
    for line in summary:
        print(line)


if __name__ == "__main__":
    main()
