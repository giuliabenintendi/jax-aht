"""Main eval orchestration: greedy eval and eval video logging."""
import os

import hydra
import jax
import jax.numpy as jnp
import numpy as np

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent, _get_image_dims
from agents.ja_actor_critic import _compute_resnet_output_dims
from agents.ja_utils import jsd_divergence, augment_obs_for_eval
from marl.eval_card_game import (
    _log_card_game_action_distributions,
    _log_card_game_attention_grid,
    _log_card_game_eval_video,
    _log_card_game_per_agent_obs_video,
    _log_card_game_xp_videos,
)
from marl.eval_lbf import _render_lbf_eval_frames


def _get_obs_type(config):
    return config.get("OBS_TYPE", config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))


def log_greedy_eval(algorithm_config, env, out, logger, num_episodes=64, init_fn=None):
    """Run greedy eval episodes, print per-episode and summary stats."""
    import wandb
    if init_fn is None:
        obs_type = _get_obs_type(algorithm_config)
        init_fn = initialize_ja_image_agent if obs_type == "image" else initialize_ja_agent
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algorithm_config, env, rng)

    inner_env = env._env
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
    num_seeds = jax.tree.leaves(out["final_params"])[0].shape[0]

    feed_attn = algorithm_config.get("FEED_OTHER_ATTN", False)
    ja_card_attn = algorithm_config.get("JA_CARD_ATTN", False)
    # JA_CARD_PARTNER_FEED gates whether the partner attention vector is
    # appended to the obs at eval time (must match training).
    ja_card_partner_feed = ja_card_attn and algorithm_config.get("JA_CARD_PARTNER_FEED", True)
    partner_feed_dim = 5
    query_partner_lstm = algorithm_config.get("QUERY_PARTNER_LSTM", False)
    _lstm_dim = algorithm_config.get("LSTM_HIDDEN_DIM", 128)
    if feed_attn or ja_card_attn:
        _img_h, _img_w, _ = _get_image_dims(env)
        _feat_h, _feat_w = _compute_resnet_output_dims(
            _img_h, _img_w,
            stride=algorithm_config.get("CONV_STRIDE", 2),
            kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
            padding=algorithm_config.get("CONV_PADDING", "SAME"),
            num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
        )
    if ja_card_attn:
        from agents.ja_utils import build_card_masks
        _card_masks_eval = build_card_masks(_img_h, _img_w, _feat_h, _feat_w)

    mode_name = "greedy"
    greedy = True
    all_returns = []
    all_jsds = []
    # Confusion matrices for card game communication analysis (per seed)
    _num_cards = getattr(inner_env, 'num_cards', 5)
    _is_comm_env = hasattr(inner_env, 'communication') and inner_env.communication
    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], out["final_params"])
        seed_returns = []
        seed_jsds = []
        # Per-seed communication matrices
        own_msg_vs_pick = np.zeros((_num_cards, _num_cards), dtype=np.int32)
        partner_msg_vs_pick = np.zeros((_num_cards, _num_cards), dtype=np.int32)
        first_msg_pair = np.zeros((_num_cards, _num_cards), dtype=np.int32)
        last_msg_pair = np.zeros((_num_cards, _num_cards), dtype=np.int32)
        for ep in range(num_episodes):
            rng = jax.random.PRNGKey(2000 + seed_idx * 10000 + ep)
            rng, reset_rng = jax.random.split(rng)

            obs, env_state = inner_env.reset(reset_rng)
            done = {k: jnp.zeros((1,), dtype=bool) for k in inner_env.agents + ["__all__"]}
            hstate_0 = policy.init_hstate(1)
            hstate_1 = policy.init_hstate(1)
            use_prev_io = getattr(policy, "uses_prev_reward_action", False)
            if use_prev_io:
                prev_reward_0 = jnp.zeros((1, 1), dtype=jnp.float32)
                prev_reward_1 = jnp.zeros((1, 1), dtype=jnp.float32)
                prev_action_0 = jnp.zeros((1, 1), dtype=jnp.float32)
                prev_action_1 = jnp.zeros((1, 1), dtype=jnp.float32)

            if feed_attn:
                prev_attn_0 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)
                prev_attn_1 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)

            if ja_card_partner_feed:
                prev_partner_card_attn_0 = jnp.zeros(partner_feed_dim)
                prev_partner_card_attn_1 = jnp.zeros(partner_feed_dim)

            if query_partner_lstm:
                plh_a0 = jnp.zeros((1, 1, _lstm_dim))
                plh_a1 = jnp.zeros((1, 1, _lstm_dim))
                plh_c0 = jnp.zeros((1, 1, _lstm_dim))
                plh_c1 = jnp.zeros((1, 1, _lstm_dim))

            total_reward = 0.0
            ep_jsds = []
            # Track messages and picks for confusion matrices
            last_msg_0, last_msg_1 = -1, -1
            first_msg_0, first_msg_1 = -1, -1
            first_msg_set_0, first_msg_set_1 = False, False
            ep_own_msg_0, ep_pick_0 = -1, -1
            ep_own_msg_1, ep_pick_1 = -1, -1
            ep_partner_msg_for_0, ep_partner_msg_for_1 = -1, -1
            step = 0
            while not done["__all__"] and step < max_steps:
                avail_actions = inner_env.get_avail_actions(env_state)
                avail_actions = jax.lax.stop_gradient(avail_actions)

                obs_0 = obs["agent_0"]
                obs_1 = obs["agent_1"]
                if feed_attn:
                    obs_0 = augment_obs_for_eval(obs_0, prev_attn_1, _img_h, _img_w)
                    obs_1 = augment_obs_for_eval(obs_1, prev_attn_0, _img_h, _img_w)
                if ja_card_partner_feed:
                    obs_0 = jnp.concatenate([obs_0, prev_partner_card_attn_0])
                    obs_1 = jnp.concatenate([obs_1, prev_partner_card_attn_1])

                rng, rng0, rng1, step_rng = jax.random.split(rng, 4)
                plh_kw0 = dict(plh_actor=plh_a0, plh_critic=plh_c0) if query_partner_lstm else {}
                plh_kw1 = dict(plh_actor=plh_a1, plh_critic=plh_c1) if query_partner_lstm else {}
                act_0, hstate_0, attn_0 = policy.get_action_and_attention(
                    params=params,
                    obs=obs_0.reshape(1, 1, -1),
                    done=done["agent_0"].reshape(1, 1),
                    avail_actions=avail_actions["agent_0"].astype(jnp.float32),
                    hstate=hstate_0, rng=rng0, greedy=greedy,
                    prev_reward=prev_reward_0 if use_prev_io else None,
                    prev_action=prev_action_0 if use_prev_io else None,
                    **plh_kw0,
                )
                act_1, hstate_1, attn_1 = policy.get_action_and_attention(
                    params=params,
                    obs=obs_1.reshape(1, 1, -1),
                    done=done["agent_1"].reshape(1, 1),
                    avail_actions=avail_actions["agent_1"].astype(jnp.float32),
                    hstate=hstate_1, rng=rng1, greedy=greedy,
                    prev_reward=prev_reward_1 if use_prev_io else None,
                    prev_action=prev_action_1 if use_prev_io else None,
                    **plh_kw1,
                )
                if query_partner_lstm:
                    # Swap: each agent gets the other's actor/critic h
                    plh_a0 = hstate_1[:, :, :_lstm_dim]
                    plh_a1 = hstate_0[:, :, :_lstm_dim]
                    plh_c0 = hstate_1[:, :, 2*_lstm_dim:3*_lstm_dim]
                    plh_c1 = hstate_0[:, :, 2*_lstm_dim:3*_lstm_dim]

                if feed_attn:
                    prev_attn_0 = attn_0.squeeze()
                    prev_attn_1 = attn_1.squeeze()

                if ja_card_partner_feed:
                    # Pool attention to per-card slots, scatter to canonical frame
                    # via each agent's position perm, then translate back into the
                    # OTHER agent's view-frame. Mirrors the rollout in marl/ja_ippo.py.
                    a0_sq = attn_0.squeeze()
                    a1_sq = attn_1.squeeze()
                    perm_0 = env_state.env_state.per_agent_perm["agent_0"]
                    perm_1 = env_state.env_state.per_agent_perm["agent_1"]
                    card_attn_0 = jnp.einsum("hw,chw->c", a0_sq, _card_masks_eval)
                    card_attn_1 = jnp.einsum("hw,chw->c", a1_sq, _card_masks_eval)
                    phys_0 = jnp.zeros(5).at[perm_0].set(card_attn_0)
                    phys_1 = jnp.zeros(5).at[perm_1].set(card_attn_1)
                    prev_partner_card_attn_0 = phys_1[perm_0]
                    prev_partner_card_attn_1 = phys_0[perm_1]
                    if done["__all__"]:
                        prev_partner_card_attn_0 = jnp.zeros(partner_feed_dim)
                        prev_partner_card_attn_1 = jnp.zeros(partner_feed_dim)

                jsd_val = float(jsd_divergence(
                    attn_0.squeeze(0), attn_1.squeeze(0)).mean())
                ep_jsds.append(jsd_val)

                env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1.squeeze()}
                a0_int = int(act_0.squeeze())
                a1_int = int(act_1.squeeze())
                # Track messages during deliberation, picks on decision step.
                # Unified action: a0_int / a1_int are card identities; the env
                # phase (deliberation vs decision) determines the role.
                if _is_comm_env:
                    is_last = (step + 1) >= max_steps
                    if not is_last:
                        last_msg_0 = a0_int
                        if not first_msg_set_0:
                            first_msg_0 = last_msg_0
                            first_msg_set_0 = True
                        last_msg_1 = a1_int
                        if not first_msg_set_1:
                            first_msg_1 = last_msg_1
                            first_msg_set_1 = True
                    else:
                        pick_0_local = a0_int
                        pick_1_local = a1_int
                        # Actions/messages are color-based. Under OP recolouring we
                        # invert back to ground-truth color identity; otherwise the
                        # raw IDs are already in the correct label space.
                        has_recolour = hasattr(env_state, "per_agent_inv_recolouring")
                        if has_recolour:
                            inv_0 = np.array(env_state.per_agent_inv_recolouring["agent_0"])
                            inv_1 = np.array(env_state.per_agent_inv_recolouring["agent_1"])
                            ep_pick_0 = int(inv_0[pick_0_local]) if pick_0_local >= 0 else -1
                            ep_pick_1 = int(inv_1[pick_1_local]) if pick_1_local >= 0 else -1
                            ep_own_msg_0 = int(inv_0[last_msg_0]) if last_msg_0 >= 0 else -1
                            ep_own_msg_1 = int(inv_1[last_msg_1]) if last_msg_1 >= 0 else -1
                            ep_first_msg_0 = int(inv_0[first_msg_0]) if first_msg_0 >= 0 else -1
                            ep_first_msg_1 = int(inv_1[first_msg_1]) if first_msg_1 >= 0 else -1
                        else:
                            ep_pick_0 = pick_0_local
                            ep_pick_1 = pick_1_local
                            ep_own_msg_0 = last_msg_0
                            ep_own_msg_1 = last_msg_1
                            ep_first_msg_0 = first_msg_0
                            ep_first_msg_1 = first_msg_1
                        ep_partner_msg_for_0 = ep_own_msg_1
                        ep_partner_msg_for_1 = ep_own_msg_0

                obs, env_state, reward, done, info = inner_env.step(step_rng, env_state, env_act)
                if use_prev_io:
                    prev_reward_0 = reward["agent_0"].reshape(1, 1).astype(jnp.float32)
                    prev_reward_1 = reward["agent_1"].reshape(1, 1).astype(jnp.float32)
                    prev_action_0 = act_0.reshape(1, 1).astype(jnp.float32)
                    prev_action_1 = act_1.reshape(1, 1).astype(jnp.float32)
                total_reward += float(reward["agent_0"])
                step += 1

            ep_jsd_mean = float(np.mean(ep_jsds)) if ep_jsds else 0.0
            seed_returns.append(total_reward)
            seed_jsds.append(ep_jsd_mean)
            # Accumulate confusion matrices (both agents contribute)
            if _is_comm_env:
                for own_msg, pick in [(ep_own_msg_0, ep_pick_0), (ep_own_msg_1, ep_pick_1)]:
                    if 0 <= own_msg < _num_cards and 0 <= pick < _num_cards:
                        own_msg_vs_pick[own_msg, pick] += 1
                for pmsg, pick in [(ep_partner_msg_for_0, ep_pick_0), (ep_partner_msg_for_1, ep_pick_1)]:
                    if 0 <= pmsg < _num_cards and 0 <= pick < _num_cards:
                        partner_msg_vs_pick[pmsg, pick] += 1
                if 0 <= ep_first_msg_0 < _num_cards and 0 <= ep_first_msg_1 < _num_cards:
                    first_msg_pair[ep_first_msg_0, ep_first_msg_1] += 1
                if 0 <= ep_own_msg_0 < _num_cards and 0 <= ep_own_msg_1 < _num_cards:
                    last_msg_pair[ep_own_msg_0, ep_own_msg_1] += 1
            print(f"[eval] {mode_name} seed={seed_idx} ep={ep}: "
                  f"return={total_reward:.1f}  jsd={ep_jsd_mean:.4f}  steps={step}")

        all_returns.extend(seed_returns)
        all_jsds.extend(seed_jsds)
        # Log confusion matrices for this seed
        if _is_comm_env and np.any(own_msg_vs_pick > 0):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            for name, mat, xlabel, ylabel in [
                ("own_msg_vs_pick", own_msg_vs_pick, "Card Picked (physical)", "Own Message (physical)"),
                ("partner_msg_vs_pick", partner_msg_vs_pick, "Card Picked (physical)", "Partner Message (physical)"),
                ("first_msg_pair", first_msg_pair, "Agent 1 First Msg (physical)", "Agent 0 First Msg (physical)"),
                ("last_msg_pair", last_msg_pair, "Agent 1 Last Msg (physical)", "Agent 0 Last Msg (physical)"),
            ]:
                fig, ax = plt.subplots(figsize=(4, 4))
                row_sums = mat.sum(axis=1, keepdims=True)
                norm_mat = np.zeros_like(mat, dtype=np.float32)
                nonzero_rows = row_sums[:, 0] > 0
                if np.any(nonzero_rows):
                    norm_mat[nonzero_rows] = (
                        mat[nonzero_rows].astype(np.float32)
                        / row_sums[nonzero_rows].astype(np.float32)
                    )
                im = ax.imshow(norm_mat, cmap="viridis", vmin=0, vmax=1)
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_xticks(range(_num_cards))
                ax.set_yticks(range(_num_cards))
                ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
                ax.set_title(f"seed {seed_idx}")
                plt.tight_layout()
                logger.log({f"Eval/{name}/seed_{seed_idx}": wandb.Image(fig)}, commit=False)
                plt.close(fig)
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
        init_fn = initialize_ja_image_agent if obs_type == "image" else initialize_ja_agent

    # Reconstruct policy (same for both agents -- shared params)
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(algorithm_config, env, rng)

    num_seeds = jax.tree.leaves(out["final_params"])[0].shape[0]
    inner_env = env._env
    max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))

    feed_attn = algorithm_config.get("FEED_OTHER_ATTN", False)
    ja_card_attn = algorithm_config.get("JA_CARD_ATTN", False)
    ja_card_partner_feed = ja_card_attn and algorithm_config.get("JA_CARD_PARTNER_FEED", True)
    eval_partner_feed_dim = 5
    feed_attn_dims = None
    if feed_attn or ja_card_attn:
        ev_img_h, ev_img_w, _ = _get_image_dims(env)
        ev_feat_h, ev_feat_w = _compute_resnet_output_dims(
            ev_img_h, ev_img_w,
            stride=algorithm_config.get("CONV_STRIDE", 2),
            kernel_size=algorithm_config.get("CONV_KERNEL_SIZE", 3),
            padding=algorithm_config.get("CONV_PADDING", "SAME"),
            num_blocks=algorithm_config.get("CONV_NUM_BLOCKS", 4),
        )
    if feed_attn:
        feed_attn_dims = (ev_img_h, ev_img_w, ev_feat_h, ev_feat_w)
    if ja_card_attn:
        from agents.ja_utils import build_card_masks
        _card_masks_eval = build_card_masks(ev_img_h, ev_img_w, ev_feat_h, ev_feat_w)

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Video volume controls: rendering N seeds * many episodes + XP pairs is the long-tail cost.
    # Keep the per-seed metrics loop running for all seeds, but gate the heavy mp4 encodes.
    n_video_seeds = int(algorithm_config.get("EVAL_VIDEO_NUM_SEEDS", num_seeds))
    n_video_seeds = max(0, min(num_seeds, n_video_seeds))
    log_xp_videos = bool(algorithm_config.get("EVAL_VIDEO_LOG_XP", False))
    sp_video_episodes = int(algorithm_config.get("EVAL_VIDEO_NUM_EPISODES", 30))
    xp_video_episodes = int(algorithm_config.get("EVAL_VIDEO_XP_NUM_EPISODES", 5))
    print(f"[eval_video] per-seed videos: {n_video_seeds}/{num_seeds} seeds "
          f"({sp_video_episodes} eps each); XP videos: "
          f"{'on' if log_xp_videos else 'off'} ({xp_video_episodes} eps/pair)")

    for seed_idx in range(num_seeds):
        final_params = jax.tree.map(lambda x: x[seed_idx], out["final_params"])
        do_videos = seed_idx < n_video_seeds
        tag = f"Eval/seed_{seed_idx}"

        # Keep eval videos to a single episode so logging stays quick and watchable.
        num_eval_video_eps = 1

        all_ep_states = []
        all_attn_data = {"agent_0": [], "agent_1": []}
        all_ep_actions = []
        all_ep_messages = []
        all_ep_obs = []
        for ep_i in range(num_eval_video_eps):
            ep_states_i, attn_data_i, ep_actions_i, ep_messages_i, ep_obs_i = run_episode_with_states(
                jax.random.PRNGKey(42 + seed_idx * 100 + ep_i), inner_env, final_params, policy,
                final_params, policy, max_steps,
                collect_attention=True,
                collect_obs=True,
                feed_other_attn_dims=feed_attn_dims,
                ja_card_masks=_card_masks_eval if ja_card_partner_feed else None,
                partner_feed_dim=eval_partner_feed_dim,
            )
            all_ep_states.extend(ep_states_i)
            for ak in ("agent_0", "agent_1"):
                all_attn_data[ak].extend(attn_data_i.get(ak, []))
            all_ep_actions.extend(ep_actions_i)
            all_ep_messages.extend(ep_messages_i)
            all_ep_obs.extend(ep_obs_i)
        ep_states = all_ep_states
        attn_data = all_attn_data
        ep_actions = all_ep_actions
        ep_messages = all_ep_messages
        ep_obs = all_ep_obs

        print(f"[ja_ippo] Seed {seed_idx}: eval episode {len(ep_states)} frames collected")

        if do_videos:
            video_dir = f"{savedir}/videos/seed_{seed_idx}"
            os.makedirs(video_dir, exist_ok=True)

            # Render frames from episode states
            if env_name in ("lbf", "lbf-reward-shaping"):
                frames = _render_lbf_eval_frames(inner_env, ep_states)
            elif env_name == "hanabi":
                from envs.hanabi.rendering import render_hanabi_eval_frames
                frames = render_hanabi_eval_frames(ep_states, scale=8)
                attn_backdrop_frames = frames
            elif env_name == "card-game":
                from envs.card_game.rendering import (
                    render_card_game_eval_frames,
                    render_card_game_eval_frames_per_agent,
                )
                # Stacked A0 / A1 composite per frame using each agent's
                # actual obs so the OP-shuffled+recoloured view is shown.
                frames = render_card_game_eval_frames(
                    ep_states, scale=32, ep_obs=ep_obs, ep_actions=ep_actions,
                )
                # Single-game-width backdrop for the attention grid (its
                # heatmaps are sized to the original obs, not 2x-wide).
                attn_backdrop_frames = render_card_game_eval_frames_per_agent(
                    ep_states, agent_idx=0, scale=32,
                    ep_obs=ep_obs, ep_actions=ep_actions,
                )
            else:
                from evaluation.vis_episodes import render_episode_frames
                frames = render_episode_frames(ep_states, inner_env.agent_view_size, pixels_per_tile=32)

            if env_name == "card-game":
                # Card game: 2xT grid image + multi-episode video.
                # XP videos are rendered ONCE outside this loop (was previously inside, causing N-fold dup).
                import numpy as _np
                from envs.card_game.rendering import _unwrap_card_game_state
                es0 = _unwrap_card_game_state(ep_states[0])
                _card_perm = _np.array(es0.card_permutation)
                _log_card_game_attention_grid(
                    attn_backdrop_frames, attn_data, ep_actions, tag, video_dir, logger,
                    ep_messages=ep_messages, card_permutation=_card_perm,
                    ep_obs=ep_obs, ep_states=ep_states,
                )
                _log_card_game_eval_video(
                    inner_env, policy, final_params, max_steps, tag, video_dir, logger,
                    feed_attn_dims=feed_attn_dims,
                    ja_card_masks=_card_masks_eval if ja_card_partner_feed else None,
                    num_episodes=sp_video_episodes, fps=3,
                    partner_feed_dim=eval_partner_feed_dim,
                )
                _log_card_game_action_distributions(
                    inner_env, policy, final_params, max_steps, tag, logger,
                    feed_attn_dims=feed_attn_dims,
                    ja_card_masks=_card_masks_eval if ja_card_partner_feed else None,
                    num_episodes=30,
                    partner_feed_dim=eval_partner_feed_dim,
                )
                # Under OP the canonical-scene video misrepresents what each
                # agent actually sees; for JA_CARD_ATTN runs also log a video
                # built from the per-agent obs so attention maps land on the
                # correct (recoloured/shuffled) cards.
                if ja_card_attn:
                    _log_card_game_per_agent_obs_video(
                        inner_env, policy, final_params, max_steps, tag, video_dir, logger,
                        feed_attn_dims=feed_attn_dims,
                        ja_card_masks=_card_masks_eval if ja_card_partner_feed else None,
                        num_episodes=sp_video_episodes, fps=3,
                        partner_feed_dim=eval_partner_feed_dim,
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
                ja_card_masks=_card_masks_eval if ja_card_partner_feed else None,
                partner_feed_dim=eval_partner_feed_dim,
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

    # Cross-play videos: render once across all seed pairs (was previously inside the per-seed
    # loop, generating each pair N_SEEDS times). Gated by EVAL_VIDEO_LOG_XP because it's heavy.
    if log_xp_videos and num_seeds > 1 and env_name == "card-game":
        xp_video_dir = f"{savedir}/videos/xp"
        os.makedirs(xp_video_dir, exist_ok=True)
        _log_card_game_xp_videos(
            inner_env, policy, out["final_params"], max_steps,
            "Eval/XP", xp_video_dir, logger,
            feed_attn_dims=feed_attn_dims,
            ja_card_masks=_card_masks_eval if ja_card_partner_feed else None,
            num_episodes=xp_video_episodes, fps=3,
            partner_feed_dim=eval_partner_feed_dim,
        )
