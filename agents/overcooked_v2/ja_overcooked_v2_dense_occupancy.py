"""Dense attended-object occupancy JA mechanism for Overcooked V2 (MATE).

Ports `agents/lbf/ja_lbf_dense_object_occupancy.LBFDenseObjectOccupancyMechanism`
to Overcooked V2. Each step, each agent's own spatial attention selects its
most-attended task object (ROI-pooled around the object's feature cell); that
object's cell becomes a gamma-discounted first-occupancy heatmap over the
feature grid, and the agent's full attention map is trained toward the PARTNER's
attended-object occupancy with a dense cross-entropy. When `FEED_OTHER_ATTN` is
on, the partner's previous attention is appended as a fourth image channel.

Two differences from the LBF mechanism:

  * Task objects are STATIC — the pot, ingredient piles, delivery station and
    recipe indicator never move (see `ja_overcooked_v2_attention`). Their feature
    cells are resolved once at init, so there is no per-step state read, no
    lex-sort, and no "eaten" mask.
  * Other-Play in V2 permutes ingredient COLOURS, not positions, so both agents
    share the exact allocentric frame. The LBF OP un-mirroring
    (`read_op_elems` / `transform_feat_maps`) drops out entirely — attention,
    occupancy and the partner feed all live directly in the shared feature grid.

Overcooked V2 also needs the JaxMARL-style reward-shaping anneal that
`JSDMechanism` applies (folding the decaying `info["shaped_reward"]` into the
learning reward); without it V2 self-play collapses. It is reproduced here so
the mechanism is self-contained.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.overcooked_v2.ja_overcooked_v2_attention import overcooked_v2_object_ctx


def _as_spatial_attention(attn_map):
    """Return spatial attention with shape (..., feat_h, feat_w)."""
    if attn_map.ndim == 4:
        return attn_map
    raise ValueError(f"Unexpected attention map rank: {attn_map.ndim}")


def _overlap(p, q):
    """Distribution overlap sum_cells(p * q) over the last axis. (..., C) -> (...)."""
    return (p * q).sum(axis=-1)


class OvercookedV2DenseObjectOccupancyMechanism:
    """Dense future-occupancy of the partner-attended task object, attention-only (MATE)."""

    name = "overcooked-v2-ja-dense-object-occupancy"
    scalar_keys = [
        ("ja_future_partner_overlap", "JA"),
        ("ja_future_self_overlap", "JA"),
        ("aux_partner_occ_loss", "Losses"),
        ("ja_obj_on_mass", "JA"),
        ("rew_shaping_frac", "JA/rew_shaping_frac"),
        ("delivery_per_step", "Reward/delivery_per_step"),
        ("shaped_applied_per_step", "Reward/shaped_applied_per_step"),
    ]

    def __init__(self, config, env):
        self.num_agents = env.num_agents
        if self.num_agents != 2:
            raise NotImplementedError("Dense-object V2 JA currently assumes 2 agents.")
        ctx = overcooked_v2_object_ctx(config, env)
        self.img_h, self.img_w = ctx["img_h"], ctx["img_w"]
        self.feat_h, self.feat_w = ctx["feat_h"], ctx["feat_w"]
        self.tile_size = ctx["tile_size"]
        # Fixed feature-grid cell (fr, fc) of each detected task object. Constant
        # across steps and episodes because task objects are static.
        self.object_feat_rc = ctx["object_feat_rc"]  # (M, 2)
        self.num_objects = ctx["num_objects"]

        self.partner_coef = float(config.get("JA_FUTURE_PARTNER_COEF", 0.0))
        self.self_coef = float(config.get("JA_FUTURE_SELF_COEF", 0.0))
        self.gamma_occ = float(config.get("JA_FUTURE_GAMMA_OCC", 0.95))
        self.warmup_env_steps = float(config.get("JA_FUTURE_WARMUP_ENV_STEPS", 0.0))
        self.ramp_env_steps = float(config.get("JA_FUTURE_RAMP_ENV_STEPS", 1.0))
        self.feed_other_attn = bool(config.get("FEED_OTHER_ATTN", False))

        self.obj_roi_radius = int(config.get("JA_OBJECT_ROI_RADIUS", 1))
        # Target blob half-width on the feature grid. 0 -> single attended-object
        # cell; >0 -> a (2r+1)^2 box, so the dense CE rewards attention NEAR the
        # object, not only on its centre cell.
        self.obj_target_radius = int(config.get("JA_OBJECT_TARGET_RADIUS", 0))
        self.aux_occ_coef = float(config.get("JA_OBJECT_AUX_COEF", 0.005))
        self.aux_coef = self.aux_occ_coef
        self.aux_active = self.aux_occ_coef > 0.0

        # JaxMARL-style linear shaped-reward anneal (1 -> 0 over REW_SHAPING_HORIZON
        # env steps). 0 disables it (shaping then comes from the wrapper's
        # do_reward_shaping fold). Required to keep V2 self-play from collapsing.
        self.rew_shaping_horizon = float(config.get("REW_SHAPING_HORIZON", 0.0))
        self.env_steps_per_update = int(config["ROLLOUT_LENGTH"]) * int(config["NUM_ENVS"])

    def entity_feed_dim(self) -> int:
        return 0

    def init_carry(self, num_actors):
        uniform = jnp.ones(
            (num_actors, self.feat_h, self.feat_w),
            dtype=jnp.float32,
        ) / float(self.feat_h * self.feat_w)
        return {"partner_attn": uniform}

    def augment_obs(self, obs_batch_2d, carry):
        if not self.feed_other_attn:
            return obs_batch_2d
        rgb = obs_batch_2d.reshape(obs_batch_2d.shape[0], self.img_h, self.img_w, 3)
        partner_attn = carry["partner_attn"]  # shared frame; no OP transform in V2
        partner_attn_img = jax.image.resize(
            partner_attn,
            (partner_attn.shape[0], self.img_h, self.img_w),
            method="nearest",
        )
        partner_attn_img = partner_attn_img / jnp.maximum(
            partner_attn_img.max(axis=(1, 2), keepdims=True),
            1e-8,
        )
        return jnp.concatenate([rgb, partner_attn_img[..., None]], axis=-1).reshape(
            obs_batch_2d.shape[0], -1,
        )

    def _object_roi_mass(self, attn_2d):
        """Raw attention mass inside a (2r+1)^2 ROI around each task object cell.

        attn_2d: (A, feat_h, feat_w). Returns (object_mass (A, M), on_mass (A,)).
        """
        fr = self.object_feat_rc[:, 0]  # (M,)
        fc = self.object_feat_rc[:, 1]
        attn_flat = attn_2d.reshape(attn_2d.shape[0], self.feat_h * self.feat_w)
        object_mass = jnp.zeros((attn_2d.shape[0], self.num_objects), dtype=attn_2d.dtype)
        radius = self.obj_roi_radius
        for dr in range(-radius, radius + 1):
            rr = fr + dr
            valid_r = (rr >= 0) & (rr < self.feat_h)
            rr = jnp.clip(rr, 0, self.feat_h - 1)
            for dc in range(-radius, radius + 1):
                cc = fc + dc
                valid = valid_r & (cc >= 0) & (cc < self.feat_w)
                cc = jnp.clip(cc, 0, self.feat_w - 1)
                idx = rr * self.feat_w + cc  # (M,)
                mass = attn_flat[:, idx]  # (A, M)
                object_mass = object_mass + jnp.where(valid[None, :], mass, 0.0)
        return object_mass, object_mass.sum(axis=-1)

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del action, new_env_state, env_state
        attn_2d = _as_spatial_attention(attn_map).squeeze(0)  # (A, feat_h, feat_w)
        object_mass, on_mass = self._object_roi_mass(attn_2d)  # (A, M), (A,)

        sel = jnp.argmax(object_mass, axis=-1)  # (A,) most-attended object
        attended_feat = self.object_feat_rc[sel]  # (A, 2) feature cell (fr, fc)

        # JaxMARL-style annealed reward shaping (the V2 collapse fix). shaped_reward
        # is (num_envs, num_agents); transpose+flatten to agent-major (num_actors,).
        shaping_frac = jnp.array(1.0)
        shaped_total = jnp.zeros((num_actors,))
        if self.rew_shaping_horizon > 0.0 and "shaped_reward" in info:
            shaped_actors = info["shaped_reward"].swapaxes(0, 1).reshape(-1)
            env_steps = update_steps * self.env_steps_per_update
            shaping_frac = jnp.clip(1.0 - env_steps / self.rew_shaping_horizon, 0.0, 1.0)
            shaped_total = shaping_frac * shaped_actors

        extras = {
            "ja_future_attn_2d": jax.lax.stop_gradient(attn_2d.astype(jnp.float32)),
            "ja_future_feat_post": attended_feat.astype(jnp.int32),
            "ja_obj_on_mass": jax.lax.stop_gradient(on_mass),
            "rew_shaping_frac": jnp.broadcast_to(shaping_frac, (num_actors,)),
            "delivery": env_reward,
            "shaped_applied": shaped_total,
        }
        swapped_attn = self._swap_partner(attn_2d, num_actors)
        uniform = jnp.ones((self.feat_h, self.feat_w), dtype=jnp.float32) / float(
            self.feat_h * self.feat_w
        )
        new_partner_attn = jnp.where(done_actors[:, None, None], uniform[None], swapped_attn)
        reward = env_reward + shaped_total
        return reward, {"partner_attn": jax.lax.stop_gradient(new_partner_attn)}, extras

    def _attended_box_occupancy(self, fr, fc, done, radius):
        """Discounted first-occupancy of a (2*radius+1)^2 box around the attended
        object cell. radius=0 reduces to the single-cell target."""
        num_cells = self.feat_h * self.feat_w
        box = jnp.zeros(fr.shape + (num_cells,), dtype=jnp.float32)  # (T, A, C)
        for dr in range(-radius, radius + 1):
            rr = fr + dr
            valid_r = (rr >= 0) & (rr < self.feat_h)
            rr = jnp.clip(rr, 0, self.feat_h - 1)
            for dc in range(-radius, radius + 1):
                cc = fc + dc
                valid = valid_r & (cc >= 0) & (cc < self.feat_w)
                cc = jnp.clip(cc, 0, self.feat_w - 1)
                idx = rr * self.feat_w + cc
                box = box + jnp.where(
                    valid[..., None],
                    jax.nn.one_hot(idx, num_cells, dtype=jnp.float32),
                    0.0,
                )
        box = (box > 0.0).astype(jnp.float32)

        def _scan(m_next, x_t):
            box_t, done_t = x_t
            m_next = jnp.where(done_t[..., None], 0.0, m_next)
            m_t = jnp.where(box_t > 0.0, 1.0, self.gamma_occ * m_next)
            return m_t, m_t

        init = jnp.zeros(box.shape[1:], dtype=jnp.float32)  # (A, C)
        _, m = jax.lax.scan(_scan, init, (box, done), reverse=True)  # (T, A, C)
        return m / (m.sum(axis=-1, keepdims=True) + 1e-8)

    def _occupancy_targets(self, traj_batch):
        feat_rc = traj_batch.extras["ja_future_feat_post"]  # (T, A, 2)
        fr, fc = feat_rc[..., 0], feat_rc[..., 1]  # (T, A)
        self_occ = self._attended_box_occupancy(
            fr, fc, traj_batch.done, self.obj_target_radius,
        ).reshape(feat_rc.shape[0], feat_rc.shape[1], self.feat_h, self.feat_w)
        half = self_occ.shape[1] // 2
        partner_occ = jnp.concatenate([self_occ[:, half:], self_occ[:, :half]], axis=1)
        return self_occ, partner_occ

    def _swap_partner(self, x, num_actors):
        half = num_actors // self.num_agents
        return jnp.concatenate([x[half:], x[:half]], axis=0)

    def _warmup_scale(self, config, update_steps):
        env_steps = update_steps.astype(jnp.float32) * float(config["ROLLOUT_LENGTH"] * config["NUM_ENVS"])
        return jnp.clip(
            (env_steps - self.warmup_env_steps) / max(self.ramp_env_steps, 1.0),
            0.0,
            1.0,
        )

    def postprocess_trajectory(self, traj_batch, config, update_steps):
        self_occ, partner_occ = self._occupancy_targets(traj_batch)
        attn = traj_batch.extras["ja_future_attn_2d"]
        valid = (~traj_batch.done).astype(jnp.float32)

        partner_overlap = _overlap(attn.reshape(*attn.shape[:2], -1),
                                   partner_occ.reshape(*partner_occ.shape[:2], -1)) * valid
        self_overlap = _overlap(attn.reshape(*attn.shape[:2], -1),
                                self_occ.reshape(*self_occ.shape[:2], -1)) * valid
        warm = self._warmup_scale(config, update_steps)
        ja_reward = warm * (
            self.partner_coef * partner_overlap
            + self.self_coef * self_overlap
        )

        extras = dict(traj_batch.extras)
        extras.update({
            "ja_future_self_occ": self_occ,
            "ja_future_partner_occ": partner_occ,
            "ja_future_partner_overlap": partner_overlap,
            "ja_future_self_overlap": self_overlap,
            "ja_future_reward": ja_reward,
            "ja_future_warmup_scale": jnp.broadcast_to(warm, traj_batch.reward.shape),
        })
        return traj_batch._replace(
            reward=traj_batch.reward + ja_reward,
            extras=extras,
        )

    def aux_loss(self, attn_map_apply, traj_batch, config):
        del config
        if not self.aux_active:
            return self.aux_occ_coef, jnp.float32(0.0)
        target = jax.lax.stop_gradient(traj_batch.extras["ja_future_partner_occ"])
        attn = _as_spatial_attention(attn_map_apply)  # (T, A, fh, fw); shared frame
        nll = -(target * jnp.log(attn + 1e-8)).sum(axis=(-2, -1))
        valid = (~traj_batch.done).astype(jnp.float32)
        loss = (nll * valid).sum() / jnp.maximum(valid.sum(), 1e-8)
        return self.aux_occ_coef, loss

    def rollout_metrics(self, traj_batch, loss_info):
        ex = traj_batch.extras
        return {
            "ja_future_partner_overlap": ex["ja_future_partner_overlap"].mean(),
            "ja_future_self_overlap": ex["ja_future_self_overlap"].mean(),
            "aux_partner_occ_loss": loss_info.aux_loss.mean(),
            "ja_obj_on_mass": ex["ja_obj_on_mass"].mean(),
            "rew_shaping_frac": ex["rew_shaping_frac"].mean(),
            "delivery_per_step": ex["delivery"].mean(),
            "shaped_applied_per_step": ex["shaped_applied"].mean(),
        }

    def report(self, config, out, logger):
        from common.train_logging import (
            IMAGE_IPPO_SCALAR_KEYS,
            report_basic_training_outputs,
        )
        report_basic_training_outputs(
            config, out, logger,
            scalar_keys=list(IMAGE_IPPO_SCALAR_KEYS) + list(self.scalar_keys),
            print_prefix="ja_ippo:overcooked_v2_dense_object_occupancy",
        )

    def eval_outputs(self, algorithm_config, env, out, logger):
        del algorithm_config, env, out, logger

    def log_ckpt_video(self, algorithm_config, env, params, policy, tag, savedir, logger):
        """Per-checkpoint god's-eye gameplay + attention-overlay video.

        Reconstructs the partner-attention 4th channel via `feed_other_attn_dims`
        (as `augment_obs` does at train time) so the JA image policy gets the obs
        shape it was trained on, and renders V2 frames via the render registry.
        """
        import os

        try:
            from envs.render_registry import get_eval_frames
            from evaluation.vis_episodes import make_attention_video, run_episode_with_states
            from moviepy import ImageSequenceClip

            # One-level unwrap (LogWrapper -> image wrapper); NOT get_inner_env, which
            # descends to the raw symbolic OvercookedV2 (no get_avail_actions).
            inner_env = getattr(env, "_env", env)
            env_name = algorithm_config["ENV_NAME"]
            feed_dims = (
                (self.img_h, self.img_w, self.feat_h, self.feat_w)
                if self.feed_other_attn else None
            )
            max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
            ep_states, attn_data, _a, _m = run_episode_with_states(
                jax.random.PRNGKey(42), inner_env, params, policy, params, policy,
                max_steps, collect_attention=True, feed_other_attn_dims=feed_dims,
            )
            os.makedirs(savedir, exist_ok=True)
            frames = get_eval_frames(env_name, inner_env, ep_states)
            n_attn = len(attn_data.get("agent_0", []))
            frames = list(frames[:n_attn] if n_attn else frames)
            stem = f"{savedir}/{tag.replace('/', '_')}"
            ImageSequenceClip(frames, fps=10).write_videofile(
                f"{stem}.mp4", fps=10, codec="libx264", audio=False, preset="ultrafast",
            )
            logger.log_video(tag, f"{stem}.mp4", commit=False)
            make_attention_video(frames, attn_data, filename=f"{stem}_attn.mp4", fps=10)
            for suffix in ("agent0", "agent1", "combined"):
                p = f"{stem}_attn_{suffix}.mp4"
                if os.path.exists(p):
                    logger.log_video(f"{tag}_attention_{suffix}", p, commit=False)
        except Exception as e:
            print(f"[ja_ippo:{self.name}] WARN: ckpt attention video failed ({e}); continuing.", flush=True)
