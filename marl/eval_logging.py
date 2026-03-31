"""Main eval orchestration: greedy/stochastic eval and eval video logging."""
import os

import hydra
import jax
import jax.numpy as jnp
import numpy as np

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent, _get_image_dims
from agents.ja_image_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import jsd_divergence, augment_obs_for_eval
from marl.eval_card_game import _log_card_game_attention_grid, _log_card_game_eval_video
from marl.eval_lbf import _render_lbf_eval_frames


def _get_obs_type(config):
    return config.get("OBS_TYPE", config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))


def log_greedy_eval(algorithm_config, env, out, logger, num_episodes=64, init_fn=None):
    """Run greedy and stochastic eval episodes, print per-episode and summary stats."""
    import wandb
    if init_fn is None:
        obs_type = _get_obs_type(algorithm_config)
        init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algorithm_config, env, rng)

    inner_env = env._env
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    num_seeds = jax.tree.leaves(out["final_params"])[0].shape[0]

    feed_attn = algorithm_config.get("FEED_OTHER_ATTN", False)
    eval_filter_top1 = algorithm_config.get("FILTER_ATTN_TOP1", False)
    if feed_attn:
        _img_h, _img_w, _ = _get_image_dims(env)
        _feat_h, _feat_w = _compute_resnet_output_dims(
            _img_h, _img_w,
            stride=algorithm_config.get("CONV_STRIDE", 2),
            kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
            padding=algorithm_config.get("CONV_PADDING", "SAME"),
            num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
        )

    def _apply_top1(attn):
        """Filter attention to global argmax (single spike)."""
        flat = attn.reshape(-1)
        idx = jnp.argmax(flat)
        return jnp.zeros_like(flat).at[idx].set(1.0).reshape(attn.shape)

    for mode_name, greedy in [("greedy", True), ("stochastic", False)]:
        all_returns = []
        all_jsds = []
        for seed_idx in range(num_seeds):
            params = jax.tree.map(lambda x: x[seed_idx], out["final_params"])
            seed_returns = []
            seed_jsds = []
            for ep in range(num_episodes):
                rng = jax.random.PRNGKey(2000 + seed_idx * 10000 + ep)
                rng, reset_rng = jax.random.split(rng)

                obs, env_state = inner_env.reset(reset_rng)
                done = {k: jnp.zeros((1,), dtype=bool) for k in inner_env.agents + ["__all__"]}
                hstate_0 = policy.init_hstate(1)
                hstate_1 = policy.init_hstate(1)

                if feed_attn:
                    prev_attn_0 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)
                    prev_attn_1 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)

                total_reward = 0.0
                ep_jsds = []
                step = 0
                while not done["__all__"] and step < max_steps:
                    avail_actions = inner_env.get_avail_actions(env_state)
                    avail_actions = jax.lax.stop_gradient(avail_actions)

                    obs_0 = obs["agent_0"]
                    obs_1 = obs["agent_1"]
                    if feed_attn:
                        obs_0 = augment_obs_for_eval(obs_0, prev_attn_1, _img_h, _img_w)
                        obs_1 = augment_obs_for_eval(obs_1, prev_attn_0, _img_h, _img_w)

                    rng, rng0, rng1, step_rng = jax.random.split(rng, 4)
                    act_0, hstate_0, attn_0 = policy.get_action_and_attention(
                        params=params,
                        obs=obs_0.reshape(1, 1, -1),
                        done=done["agent_0"].reshape(1, 1),
                        avail_actions=avail_actions["agent_0"].astype(jnp.float32),
                        hstate=hstate_0, rng=rng0, greedy=greedy,
                    )
                    act_1, hstate_1, attn_1 = policy.get_action_and_attention(
                        params=params,
                        obs=obs_1.reshape(1, 1, -1),
                        done=done["agent_1"].reshape(1, 1),
                        avail_actions=avail_actions["agent_1"].astype(jnp.float32),
                        hstate=hstate_1, rng=rng1, greedy=greedy,
                    )

                    if eval_filter_top1:
                        attn_0 = _apply_top1(attn_0.squeeze())[None, None]
                        attn_1 = _apply_top1(attn_1.squeeze())[None, None]

                    if feed_attn:
                        prev_attn_0 = attn_0.squeeze()
                        prev_attn_1 = attn_1.squeeze()

                    jsd_val = float(jsd_divergence(
                        attn_0.squeeze(0), attn_1.squeeze(0)).mean())
                    ep_jsds.append(jsd_val)

                    env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1.squeeze()}
                    obs, env_state, reward, done, info = inner_env.step(step_rng, env_state, env_act)
                    total_reward += float(reward["agent_0"])
                    step += 1

                ep_jsd_mean = float(np.mean(ep_jsds)) if ep_jsds else 0.0
                seed_returns.append(total_reward)
                seed_jsds.append(ep_jsd_mean)
                print(f"[eval] {mode_name} seed={seed_idx} ep={ep}: "
                      f"return={total_reward:.1f}  jsd={ep_jsd_mean:.4f}  steps={step}")

            all_returns.extend(seed_returns)
            all_jsds.extend(seed_jsds)
            print(f"[eval] seed {seed_idx} {mode_name} summary: "
                  f"return={np.mean(seed_returns):.1f} ± {np.std(seed_returns):.1f}  "
                  f"jsd={np.mean(seed_jsds):.4f} ± {np.std(seed_jsds):.4f}")

        ret_mean, ret_std = np.mean(all_returns), np.std(all_returns)
        jsd_mean, jsd_std = np.mean(all_jsds), np.std(all_jsds)
        print(f"[eval] {mode_name} overall ({len(all_returns)} eps): "
              f"return={ret_mean:.1f} ± {ret_std:.1f}  "
              f"jsd={jsd_mean:.4f} ± {jsd_std:.4f}")
        wandb.run.summary[f"Eval/{mode_name}_return_mean"] = float(ret_mean)
        wandb.run.summary[f"Eval/{mode_name}_return_std"] = float(ret_std)
        wandb.run.summary[f"Eval/{mode_name}_jsd_mean"] = float(jsd_mean)
        wandb.run.summary[f"Eval/{mode_name}_jsd_std"] = float(jsd_std)

        # Per-episode results table
        table = wandb.Table(
            columns=["episode", "return", "jsd"],
            data=[[i, all_returns[i], all_jsds[i]] for i in range(len(all_returns))],
        )
        logger.log({
            f"Eval/{mode_name}_returns_chart": wandb.plot.bar(
                table, "episode", "return", title=f"{mode_name} per-episode return"),
            f"Eval/{mode_name}_jsd_chart": wandb.plot.bar(
                table, "episode", "jsd", title=f"{mode_name} per-episode JSD"),
        }, commit=False)

    logger.log({}, commit=True)


def log_eval_video(algorithm_config, env, out, logger, init_fn=None):
    """Run eval episodes for all seeds, log videos + attention to wandb."""
    from evaluation.vis_episodes import (
        run_episode_with_states, log_attention_to_wandb, make_attention_video,
        compute_attention_stasis, compute_object_coverage,
    )

    env_name = algorithm_config["ENV_NAME"]

    if init_fn is None:
        obs_type = _get_obs_type(algorithm_config)
        init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent

    # Reconstruct policy (same for both agents -- shared params)
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algorithm_config, env, rng)

    num_seeds = jax.tree.leaves(out["final_params"])[0].shape[0]
    inner_env = env._env
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))

    feed_attn = algorithm_config.get("FEED_OTHER_ATTN", False)
    feed_attn_dims = None
    if feed_attn:
        ev_img_h, ev_img_w, _ = _get_image_dims(env)
        ev_feat_h, ev_feat_w = _compute_resnet_output_dims(
            ev_img_h, ev_img_w,
            stride=algorithm_config.get("CONV_STRIDE", 2),
            kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
            padding=algorithm_config.get("CONV_PADDING", "SAME"),
            num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
        )
        feed_attn_dims = (ev_img_h, ev_img_w, ev_feat_h, ev_feat_w)

    # Build fixed partner attention for eval visualization
    fixed_partner_pos_eval = algorithm_config.get("ENV_KWARGS", {}).get("fixed_partner_pos", -1)
    fixed_partner_attn_eval = None
    if fixed_partner_pos_eval >= 0:
        ev_img_h_fp, ev_img_w_fp, _ = _get_image_dims(env)
        ev_fh_fp, ev_fw_fp = _compute_resnet_output_dims(
            ev_img_h_fp, ev_img_w_fp,
            stride=algorithm_config.get("CONV_STRIDE", 2),
            kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
            padding=algorithm_config.get("CONV_PADDING", "SAME"),
            num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
        )
        from envs.card_game.rendering import TILE_PIXELS as _TP_eval
        pixel_col = fixed_partner_pos_eval * _TP_eval + _TP_eval // 2
        pixel_row = 1 * _TP_eval + _TP_eval // 2
        fc = min(int(pixel_col / (ev_img_h_fp / ev_fh_fp)), ev_fw_fp - 1)
        fr = min(int(pixel_row / (ev_img_h_fp / ev_fh_fp)), ev_fh_fp - 1)
        fixed_partner_attn_eval = jnp.zeros((ev_fh_fp, ev_fw_fp))
        fixed_partner_attn_eval = fixed_partner_attn_eval.at[fr, fc].set(1.0)

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    for seed_idx in range(num_seeds):
        final_params = jax.tree.map(lambda x: x[seed_idx], out["final_params"])

        # Card games: 1 episode. Other envs: 5 episodes for longer videos.
        is_card_game = env_name in ("card-game", "card-game-dynamic", "card-game-flip")
        num_eval_video_eps = 1 if is_card_game else 5

        all_ep_states = []
        all_attn_data = {"agent_0": [], "agent_1": []}
        all_ep_actions = []
        all_ep_messages = []
        for ep_i in range(num_eval_video_eps):
            ep_states_i, attn_data_i, ep_actions_i, ep_messages_i = run_episode_with_states(
                jax.random.PRNGKey(42 + seed_idx * 100 + ep_i), inner_env, final_params, policy,
                final_params, policy, max_steps,
                collect_attention=True,
                feed_other_attn_dims=feed_attn_dims,
                fixed_partner_attn=fixed_partner_attn_eval,
            )
            all_ep_states.extend(ep_states_i)
            for ak in ("agent_0", "agent_1"):
                all_attn_data[ak].extend(attn_data_i.get(ak, []))
            all_ep_actions.extend(ep_actions_i)
            all_ep_messages.extend(ep_messages_i)
        ep_states = all_ep_states
        attn_data = all_attn_data
        ep_actions = all_ep_actions
        ep_messages = all_ep_messages
        # Apply top1 filter to collected attention maps (match training behavior)
        if algorithm_config.get("FILTER_ATTN_TOP1", False):
            import numpy as _np
            def _top1_np(attn):
                a = _np.array(attn).squeeze()
                out = _np.zeros_like(a)
                out.flat[_np.argmax(a)] = 1.0
                return out
            for agent_key in ("agent_0", "agent_1"):
                attn_data[agent_key] = [_top1_np(m) for m in attn_data.get(agent_key, [])]

        print(f"[ja_ippo] Seed {seed_idx}: eval episode {len(ep_states)} frames collected")

        video_dir = f"{savedir}/videos/seed_{seed_idx}"
        os.makedirs(video_dir, exist_ok=True)

        # Render frames from episode states
        if env_name in ("lbf", "lbf-image", "lbf-reward-shaping"):
            frames = _render_lbf_eval_frames(inner_env, ep_states)
        elif env_name in ("card-game", "card-game-flip"):
            from envs.card_game.rendering import render_card_game_eval_frames
            frames = render_card_game_eval_frames(ep_states, scale=32)
        elif env_name == "card-game-dynamic":
            from envs.card_game.rendering_dynamic import render_card_game_eval_frames as render_dynamic_frames
            frames = render_dynamic_frames(ep_states, scale=32)
        else:
            from evaluation.vis_episodes import render_episode_frames
            frames = render_episode_frames(ep_states, inner_env.agent_view_size, pixels_per_tile=32)

        tag = f"Eval/seed_{seed_idx}"

        if env_name in ("card-game", "card-game-dynamic", "card-game-flip"):
            # Card game: 2xT grid image + multi-episode video
            # Pass card layout for border drawing
            import numpy as _np
            from envs.card_game.rendering import _unwrap_card_game_state
            _card_pos = None
            _card_perm = None
            es0 = _unwrap_card_game_state(ep_states[0])
            if hasattr(es0, 'card_positions'):
                _card_pos = _np.array(es0.card_positions)
            if hasattr(es0, 'card_permutation'):
                _card_perm = _np.array(es0.card_permutation)
            _log_card_game_attention_grid(
                frames, attn_data, ep_actions, tag, video_dir, logger,
                ep_messages=ep_messages, card_positions=_card_pos,
                card_permutation=_card_perm,
            )
            _log_card_game_eval_video(
                inner_env, policy, final_params, max_steps, tag, video_dir, logger,
                feed_attn_dims=feed_attn_dims,
                fixed_partner_attn=fixed_partner_attn_eval,
                filter_top1=algorithm_config.get("FILTER_ATTN_TOP1", False),
                num_episodes=30, fps=3,
            )
        else:
            # Other envs: videos + attention overlays
            from moviepy import ImageSequenceClip
            video_path = f"{video_dir}/eval_final.mp4"
            clip = ImageSequenceClip(frames, fps=10)
            clip.write_videofile(video_path, fps=10, codec='libx264', audio=False,
                                 bitrate='8000k', preset='slow')
            logger.log_video(f"{tag}/episode_video", video_path, commit=False)

            log_attention_to_wandb(
                attn_data, logger, step=None, tag_prefix=tag, commit=False,
                frames=frames,
            )

            attn_video_base = f"{video_dir}/eval_attention.mp4"
            make_attention_video(frames, attn_data, filename=attn_video_base, fps=10)
            logger.log_video(f"{tag}/attention_agent0", f"{video_dir}/eval_attention_agent0.mp4", commit=False)
            logger.log_video(f"{tag}/attention_agent1", f"{video_dir}/eval_attention_agent1.mp4", commit=False)
            logger.log_video(f"{tag}/attention_combined", f"{video_dir}/eval_attention_combined.mp4", commit=False)

        # Multi-episode attention metrics
        import numpy as np

        num_eval_episodes = int(algorithm_config.get("NUM_EVAL_EPISODES", 64))
        is_overcooked = env_name in ("overcooked-v1",)

        attn_feat_h = attn_feat_w = None
        if is_overcooked:
            attn_img_h, attn_img_w, _ = _get_image_dims(env)
            attn_feat_h, attn_feat_w = _compute_resnet_output_dims(
                attn_img_h, attn_img_w,
                stride=algorithm_config.get("CONV_STRIDE", 2),
                kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
                padding=algorithm_config.get("CONV_PADDING", "SAME"),
                num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
            )

        stasis_agent0_vals = []
        stasis_agent1_vals = []
        pct_obj_agent0_vals = []
        pct_obj_agent1_vals = []
        category_accum_agent0: dict[str, float] = {}
        category_accum_agent1: dict[str, float] = {}

        def _accumulate_episode(ep_attn_data, ep_ep_states):
            stasis = compute_attention_stasis(ep_attn_data)
            stasis_agent0_vals.append(stasis["agent_0_stasis"])
            stasis_agent1_vals.append(stasis["agent_1_stasis"])

            if is_overcooked and attn_feat_h is not None and attn_feat_w is not None:
                obj = compute_object_coverage(ep_attn_data, ep_ep_states, attn_feat_h, attn_feat_w)
                pct_obj_agent0_vals.append(obj["agent_0_pct_objects"])
                pct_obj_agent1_vals.append(obj["agent_1_pct_objects"])
                for agent_key, accum in [
                    ("agent_0", category_accum_agent0),
                    ("agent_1", category_accum_agent1),
                ]:
                    cat_mass = obj[f"{agent_key}_category_mass"]
                    for cat, val in cat_mass.items():
                        accum[cat] = accum.get(cat, 0.0) + val

        _accumulate_episode(attn_data, ep_states)

        for ep in range(1, num_eval_episodes):
            ep_rng = jax.random.PRNGKey(42 + seed_idx * 10000 + ep)
            ep_states_extra, attn_data_extra, _, _ = run_episode_with_states(
                ep_rng, inner_env, final_params, policy,
                final_params, policy, max_steps,
                collect_attention=True,
                feed_other_attn_dims=feed_attn_dims,
            )
            _accumulate_episode(attn_data_extra, ep_states_extra)

            if (ep + 1) % max(1, num_eval_episodes // 4) == 0:
                print(f"[ja_ippo] Seed {seed_idx}: eval attention episode {ep + 1}/{num_eval_episodes}")

        n_eps = len(stasis_agent0_vals)
        stasis_a0_mean = float(np.nanmean(stasis_agent0_vals))
        stasis_a0_std = float(np.nanstd(stasis_agent0_vals))
        stasis_a1_mean = float(np.nanmean(stasis_agent1_vals))
        stasis_a1_std = float(np.nanstd(stasis_agent1_vals))

        print(f"[ja_ippo] Seed {seed_idx} stasis ({n_eps} eps): "
              f"agent0={stasis_a0_mean:.4f} +/- {stasis_a0_std:.4f}, "
              f"agent1={stasis_a1_mean:.4f} +/- {stasis_a1_std:.4f}")


        if is_overcooked and pct_obj_agent0_vals:
            pct_a0_mean = float(np.nanmean(pct_obj_agent0_vals))
            pct_a0_std = float(np.nanstd(pct_obj_agent0_vals))
            pct_a1_mean = float(np.nanmean(pct_obj_agent1_vals))
            pct_a1_std = float(np.nanstd(pct_obj_agent1_vals))

            print(f"[ja_ippo] Seed {seed_idx} pct_objects ({n_eps} eps): "
                  f"agent0={pct_a0_mean:.4f} +/- {pct_a0_std:.4f}, "
                  f"agent1={pct_a1_mean:.4f} +/- {pct_a1_std:.4f}")

            logger.log({
                f"{tag}/pct_objects_agent0_mean": pct_a0_mean,
                f"{tag}/pct_objects_agent0_std": pct_a0_std,
                f"{tag}/pct_objects_agent1_mean": pct_a1_mean,
                f"{tag}/pct_objects_agent1_std": pct_a1_std,
            }, commit=False)

            for agent_label, accum in [("agent_0", category_accum_agent0),
                                        ("agent_1", category_accum_agent1)]:
                sorted_cats = sorted(accum.items(), key=lambda x: -x[1])[:5]
                parts = [f"{k}={v / n_eps:.3f}" for k, v in sorted_cats]
                print(f"[ja_ippo] Seed {seed_idx} {agent_label} top categories: {', '.join(parts)}")
                for cat, val in sorted_cats:
                    logger.log({f"{tag}/{agent_label}_attn_{cat}": val / n_eps}, commit=False)
