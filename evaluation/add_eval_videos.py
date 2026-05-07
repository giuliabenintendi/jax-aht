"""Add eval videos to an existing wandb run from a saved checkpoint.

Usage:
    uv run python -m evaluation.add_eval_videos \
        --checkpoint <path_to_saved_train_run> \
        --run-id <wandb_run_id>
"""
import argparse
import os

import jax
import wandb
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_image_agent, initialize_ja_dual_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from evaluation.vis_episodes import (
    run_episode_with_states, make_attention_video, render_episode_frames,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", required=True, help="Existing wandb run ID to resume")
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    parser.add_argument("--xp-pairs", nargs="+", default=[],
                        help='Optional cross-play seed pairs as "i,j" tokens, e.g. '
                             '`--xp-pairs 0,2 1,3`. Card-game env only. Adds an XP '
                             'video for each pair on top of the per-seed SP videos.')
    parser.add_argument("--all-xp-pairs", action="store_true",
                        help="Render every upper-triangle pair (i<j). Overrides --xp-pairs. "
                             "Card-game env only.")
    parser.add_argument("--xp-per-agent", action="store_true",
                        help="Render XP videos in each agent's OP-recoloured/shuffled "
                             "view (the actual policy input) instead of the canonical scene.")
    parser.add_argument("--no-sp-videos", action="store_true",
                        help="Skip per-seed self-play videos (card-game env only). "
                             "Useful when --xp-pairs is the only thing you want.")
    parser.add_argument("--use-best", action="store_true",
                        help="Use best_params (per-seed best ckpt) instead of final_params.")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Episodes per seed (SP) or per pair (XP). Default 5; bump to "
                             "~30+ to make distributional patterns visible in the rollout.")
    parser.add_argument("--sp-seeds", nargs="+", type=int, default=None,
                        help="Only render SP videos for these seed indices "
                             "(e.g. `--sp-seeds 5`). Default: all seeds.")
    args = parser.parse_args()
    xp_pairs = []
    for tok in args.xp_pairs:
        i_str, j_str = tok.split(",")
        xp_pairs.append((int(i_str), int(j_str)))

    # Load config
    run_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]

    # Card-game obs only includes the partner-message channel when the env is
    # built with communication=True; mirror algorithm.COMMUNICATION into
    # ENV_KWARGS so make_env reproduces the trained obs shape.
    if alg_config.get("COMMUNICATION", False):
        env_kwargs = dict(alg_config["ENV_KWARGS"])
        env_kwargs["communication"] = True
        alg_config["ENV_KWARGS"] = env_kwargs

    env_name = alg_config["ENV_NAME"]
    env = make_env(env_name, alg_config["ENV_KWARGS"])
    env = LogWrapper(env)

    # Detect dual critic and init policy
    obs_type = alg_config.get("OBS_TYPE", alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))
    use_dual = alg_config.get("USE_DUAL_CRITIC", False)
    if obs_type in ("image", "fov"):
        init_fn = initialize_ja_dual_image_agent if use_dual else initialize_ja_image_agent
    else:
        from agents.initialize_agents import initialize_ja_agent
        init_fn = initialize_ja_agent

    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env, rng)

    # Load params
    run_data = load_train_run(args.checkpoint)
    params_key = "best_params" if args.use_best else "final_params"
    if params_key not in run_data:
        raise KeyError(f"{params_key!r} not found in checkpoint; keys: {list(run_data.keys())}")
    final_params = run_data[params_key]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    print(f"[add_eval_videos] using {params_key} ({num_seeds} seeds)")

    if args.all_xp_pairs:
        xp_pairs = [(i, j) for i in range(num_seeds) for j in range(i + 1, num_seeds)]
        print(f"[add_eval_videos] --all-xp-pairs -> {len(xp_pairs)} pair(s)")

    inner_env = env._env
    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 400))

    # Resume wandb run
    wb_run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=args.run_id,
        resume="must",
    )

    if env_name == "card-game":
        from marl.eval_card_game import (
            _log_card_game_eval_video,
            _log_card_game_per_agent_obs_video,
            _log_card_game_xp_videos,
            _log_card_game_per_agent_xp_videos,
        )
        # JA policies append a 4th obs channel (FEED_OTHER_ATTN) and/or a 5-dim
        # translated-partner-attention scalar suffix (JA_CARD_PARTNER_FEED).
        # Compute the matching feed dims/masks so the policy gets the input
        # shape it trained against.
        feed_attn = alg_config.get("FEED_OTHER_ATTN", False)
        ja_card_attn = alg_config.get("JA_CARD_ATTN", False)
        ja_card_partner_feed = ja_card_attn and alg_config.get("JA_CARD_PARTNER_FEED", True)
        feed_attn_dims = None
        ja_card_masks = None
        if feed_attn or ja_card_partner_feed:
            from agents.ja_image_actor_critic import _compute_resnet_output_dims
            from agents.ja_utils import build_card_masks
            img_h = inner_env.grid_height * inner_env.tile_size
            img_w = inner_env.grid_width * inner_env.tile_size
            feat_h, feat_w = _compute_resnet_output_dims(
                img_h, img_w,
                stride=alg_config.get("CONV_STRIDE", 2),
                kernel_size=alg_config.get("CONV_KERNEL_SIZE", 3),
                padding=alg_config.get("CONV_PADDING", "SAME"),
                num_blocks=alg_config.get("CONV_NUM_BLOCKS", 4),
            )
            if feed_attn:
                feed_attn_dims = (img_h, img_w, feat_h, feat_w)
            if ja_card_partner_feed:
                ja_card_masks = build_card_masks(img_h, img_w, feat_h, feat_w)

        class _WandbVideoLogger:
            def __init__(self, run): self.run = run
            def log_video(self, tag, path, commit=True):
                self.run.log({tag: wandb.Video(path, format="mp4")}, commit=commit)

        wandb_logger = _WandbVideoLogger(wb_run)
        if not args.no_sp_videos:
            sp_seed_indices = args.sp_seeds if args.sp_seeds is not None else list(range(num_seeds))
            for seed_idx in sp_seed_indices:
                params = jax.tree.map(lambda x: x[seed_idx], final_params)
                video_dir = os.path.join(run_dir, "videos", f"seed_{seed_idx}")
                os.makedirs(video_dir, exist_ok=True)
                # Canonical-frame view (same physical layout for both agents) — good
                # for "what happened in the world".
                _log_card_game_eval_video(
                    inner_env, policy, params, max_steps,
                    tag=f"Eval/seed_{seed_idx}",
                    video_dir=video_dir,
                    logger=wandb_logger,
                    feed_attn_dims=feed_attn_dims,
                    ja_card_masks=ja_card_masks,
                    num_episodes=args.num_episodes, fps=3,
                )
                # Per-agent OP-recoloured/shuffled view — the actual policy input,
                # with attention overlaid. Partner-message dot is drawn into the
                # observation by the env itself at delivered timing.
                _log_card_game_per_agent_obs_video(
                    inner_env, policy, params, max_steps,
                    tag=f"Eval/seed_{seed_idx}",
                    video_dir=video_dir,
                    logger=wandb_logger,
                    feed_attn_dims=feed_attn_dims,
                    ja_card_masks=ja_card_masks,
                    num_episodes=args.num_episodes, fps=3,
                )
                print(f"Seed {seed_idx}: SP videos in {video_dir}")

        if xp_pairs:
            xp_dir = os.path.join(run_dir, "videos", "xp_per_agent" if args.xp_per_agent else "xp")
            os.makedirs(xp_dir, exist_ok=True)
            xp_render_fn = (
                _log_card_game_per_agent_xp_videos if args.xp_per_agent
                else _log_card_game_xp_videos
            )
            print(f"XP pairs requested: {len(xp_pairs)} pair(s) "
                  f"({'per-agent view' if args.xp_per_agent else 'canonical view'}) -> {xp_dir}")
            xp_render_fn(
                inner_env, policy, final_params, max_steps,
                tag="Eval/xp",
                video_dir=xp_dir,
                logger=wandb_logger,
                feed_attn_dims=feed_attn_dims,
                ja_card_masks=ja_card_masks,
                seed_pairs=xp_pairs,
                num_episodes=args.num_episodes, fps=3,
            )
        wb_run.log({}, commit=True)
        wb_run.finish()
        print(f"Card-game eval videos added to {wb_run.url}")
        return

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)

        ep_states, attn_data, _, _ = run_episode_with_states(
            jax.random.PRNGKey(42 + seed_idx), inner_env, params, policy,
            params, policy, max_steps,
            collect_attention=True,
        )
        print(f"Seed {seed_idx}: {len(ep_states)} frames collected")

        video_dir = os.path.join(run_dir, "videos", f"seed_{seed_idx}")
        os.makedirs(video_dir, exist_ok=True)

        # Render frames
        if env_name in ("lbf", "lbf-image", "lbf-reward-shaping"):
            from marl.eval_lbf import _render_lbf_eval_frames
            frames = _render_lbf_eval_frames(inner_env, ep_states)
        else:
            frames = render_episode_frames(ep_states, inner_env.agent_view_size, pixels_per_tile=32)

        # Game video
        from moviepy import ImageSequenceClip
        video_path = os.path.join(video_dir, "eval_final.mp4")
        clip = ImageSequenceClip(frames, fps=10)
        clip.write_videofile(video_path, fps=10, codec='libx264', audio=False,
                             bitrate='8000k', preset='slow')
        tag = f"Eval/seed_{seed_idx}"
        wb_run.log({f"{tag}/episode_video": wandb.Video(video_path, format="mp4")}, commit=False)

        # Attention overlay videos
        attn_video_base = os.path.join(video_dir, "eval_attention.mp4")
        make_attention_video(frames, attn_data, filename=attn_video_base, fps=10)
        for suffix in ("agent0", "agent1", "combined"):
            vpath = os.path.join(video_dir, f"eval_attention_{suffix}.mp4")
            wb_run.log({f"{tag}/attention_{suffix}": wandb.Video(vpath, format="mp4")}, commit=False)

    wb_run.log({}, commit=True)
    wb_run.finish()
    print(f"Eval videos added to {wb_run.url}")


if __name__ == "__main__":
    main()
