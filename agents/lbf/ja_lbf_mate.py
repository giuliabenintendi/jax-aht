"""Dense attended-object occupancy JA mechanism for LBF (MATE).

Each step, each agent's own spatial attention selects its most-attended apple
(ROI-pooled, in the true frame under Other-Play). That apple's grid cell is turned
into a gamma-discounted first-occupancy heatmap over the feature grid, and the
agent's full spatial attention map is trained toward the PARTNER's attended-apple
occupancy with a dense cross-entropy.

The occupancy SOURCE is the partner's attended apple, not the agent's position: it
carries no partner action/position knowledge, only what the partner attends to.
Under Other-Play each agent's attention lives in its own mirror frame; it is
un-mirrored to the true frame (identity when OP is off) before ROI-pooling, the
aux, and the partner-attention feed. When FEED_OTHER_ATTN is enabled the partner's
previous spatial attention is appended as a fourth image channel.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.lbf.ja_lbf_attention import (
    as_spatial_attention,
    food_state_from_log_state,
    lbf_attention_ctx,
    lex_sort_food,
)
from agents.lbf.op_equivariance import (
    feat_transform_table,
    read_op_elems,
    transform_feat_maps,
)


class LBFDenseObjectOccupancyMechanism:
    """Dense future-occupancy of the partner-attended apple, attention-only (MATE)."""

    name = "lbf-ja-dense-object-occupancy"
    scalar_keys = [
        ("aux_partner_occ_loss", "Losses"),
        ("ja_obj_on_mass", "JA"),
    ]

    def __init__(self, config, env):
        self.num_agents = env.num_agents
        if self.num_agents != 2:
            raise NotImplementedError("Dense-object LBF JA currently assumes 2 agents.")
        ctx = lbf_attention_ctx(config, env)
        self.img_h, self.img_w = ctx["img_h"], ctx["img_w"]
        self.feat_h, self.feat_w = ctx["feat_h"], ctx["feat_w"]
        self.tile_size = ctx["tile_size"]

        # eta in Eq. 10: discount on the partner's future focus.
        self.gamma_occ = float(config.get("JA_FUTURE_GAMMA_OCC", 0.95))
        self.feed_other_attn = bool(config.get("FEED_OTHER_ATTN", False))

        # Other-Play: align attention with its true-frame target. Reflection table
        # is applied as identity when OP is off (read_op_elems -> None).
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self._op_table = feat_transform_table(self.feat_h, self.feat_w)

        self.obj_roi_radius = int(config.get("JA_OBJECT_ROI_RADIUS", 1))
        # Target blob half-width on the feature grid. 0 -> single attended-apple
        # cell (original dense variant); >0 -> a (2r+1)^2 box around the apple, so
        # the dense CE rewards attention NEAR the apple, not only on its centre.
        self.obj_target_radius = int(config.get("JA_OBJECT_TARGET_RADIUS", 0))
        # lambda_MATE in Eq. 11.
        self.aux_occ_coef = float(config.get("JA_OBJECT_AUX_COEF", 1e-4))
        self.aux_coef = self.aux_occ_coef
        self.aux_active = self.aux_occ_coef > 0.0

    def entity_feed_dim(self) -> int:
        return 0

    def _op_elem_actors(self, env_state, num_actors):
        """Per-actor OP element index (agent-major); zeros (identity) when OP is off."""
        op = read_op_elems(env_state, self.agents)
        if op is None:
            return jnp.zeros((num_actors,), dtype=jnp.int32)
        return jnp.concatenate([op[a] for a in self.agents]).astype(jnp.int32)

    def init_carry(self, num_actors):
        uniform = jnp.ones(
            (num_actors, self.feat_h, self.feat_w),
            dtype=jnp.float32,
        ) / float(self.feat_h * self.feat_w)
        return {"partner_attn": uniform,
                "op_elem": jnp.zeros((num_actors,), dtype=jnp.int32)}

    def augment_obs(self, obs_batch_2d, carry):
        if not self.feed_other_attn:
            return obs_batch_2d
        rgb = obs_batch_2d.reshape(obs_batch_2d.shape[0], self.img_h, self.img_w, 3)
        partner_attn = carry["partner_attn"]
        # Partner attention is stored true-frame; bring it into the consumer's OP
        # frame so the feed channel matches the OP-transformed base channels.
        partner_attn = transform_feat_maps(partner_attn, carry["op_elem"], self._op_table)
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

    def _food_per_actor(self, env_state):
        """Lex-sorted food positions/eaten tiled to actor order (agent-major)."""
        food_pos, food_eaten = food_state_from_log_state(env_state)
        food_pos, food_eaten = lex_sort_food(food_pos, food_eaten)
        food_pos = jnp.tile(food_pos, (self.num_agents, 1, 1))
        food_eaten = jnp.tile(food_eaten, (self.num_agents, 1))
        return food_pos.astype(jnp.int32), food_eaten

    def _food_feature_coords(self, food_pos):
        centre_r = food_pos[..., 0] * self.tile_size + self.tile_size // 2
        centre_c = food_pos[..., 1] * self.tile_size + self.tile_size // 2
        fr = jnp.clip(centre_r * self.feat_h // self.img_h, 0, self.feat_h - 1)
        fc = jnp.clip(centre_c * self.feat_w // self.img_w, 0, self.feat_w - 1)
        return fr.astype(jnp.int32), fc.astype(jnp.int32)

    def _food_roi_mass(self, attn_2d, food_pos, food_eaten):
        """Raw attention mass inside a (2r+1)^2 ROI around each alive food."""
        fr, fc = self._food_feature_coords(food_pos)
        attn_flat = attn_2d.reshape(*attn_2d.shape[:-2], self.feat_h * self.feat_w)
        food_mass = jnp.zeros(fr.shape, dtype=attn_2d.dtype)
        radius = self.obj_roi_radius
        for dr in range(-radius, radius + 1):
            rr = fr + dr
            valid_r = (rr >= 0) & (rr < self.feat_h)
            rr = jnp.clip(rr, 0, self.feat_h - 1)
            for dc in range(-radius, radius + 1):
                cc = fc + dc
                valid = valid_r & (cc >= 0) & (cc < self.feat_w)
                cc = jnp.clip(cc, 0, self.feat_w - 1)
                idx = rr * self.feat_w + cc
                mass = jnp.take_along_axis(attn_flat, idx, axis=-1)
                food_mass = food_mass + jnp.where(valid, mass, 0.0)
        alive = 1.0 - food_eaten.astype(jnp.float32)
        food_mass = food_mass * alive
        return food_mass, food_mass.sum(axis=-1)

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del action, info, new_env_state, update_steps
        attn_2d = as_spatial_attention(attn_map).squeeze(0)  # (A, feat_h, feat_w)
        # Un-mirror to the true frame so apple ROI-pooling uses true-frame food.
        op_elem = self._op_elem_actors(env_state, num_actors)
        attn_2d = transform_feat_maps(attn_2d, op_elem, self._op_table)
        food_pos, food_eaten = self._food_per_actor(env_state)  # (A, N, 2), (A, N)
        food_mass, on_mass = self._food_roi_mass(attn_2d, food_pos, food_eaten)  # (A, N), (A,)

        sel = jnp.argmax(food_mass, axis=-1)  # (A,) most-attended alive apple
        attended_pos = jnp.take_along_axis(
            food_pos, sel[:, None, None], axis=1).squeeze(1)  # (A, 2) tile [row, col]

        extras = {
            "ja_future_attn_2d": jax.lax.stop_gradient(attn_2d.astype(jnp.float32)),
            "ja_future_pos_post": attended_pos.astype(jnp.int32),
            "ja_obj_on_mass": jax.lax.stop_gradient(on_mass),
            "op_elem": op_elem,
        }
        swapped_attn = self._swap_partner(attn_2d, num_actors)
        uniform = jnp.ones((self.feat_h, self.feat_w), dtype=jnp.float32) / float(
            self.feat_h * self.feat_w
        )
        new_partner_attn = jnp.where(done_actors[:, None, None], uniform[None], swapped_attn)
        return env_reward, {"partner_attn": jax.lax.stop_gradient(new_partner_attn),
                            "op_elem": op_elem}, extras

    def _attended_box_occupancy(self, fr, fc, done, radius):
        """Discounted first-occupancy of a (2*radius+1)^2 box around the attended
        apple cell. radius=0 reduces to the single-cell target (original variant);
        a bigger box rewards attention near the apple, not only on its centre."""
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
        # ja_future_pos_post is the attended-apple tile; build the dense occupancy
        # target from a box around its feature cell.
        pos_rc = traj_batch.extras["ja_future_pos_post"]  # (T, A, 2)
        fr, fc = self._food_feature_coords(pos_rc)  # (T, A)
        self_occ = self._attended_box_occupancy(
            fr, fc, traj_batch.done, self.obj_target_radius,
        ).reshape(pos_rc.shape[0], pos_rc.shape[1], self.feat_h, self.feat_w)
        half = self_occ.shape[1] // 2
        partner_occ = jnp.concatenate([self_occ[:, half:], self_occ[:, :half]], axis=1)
        return self_occ, partner_occ

    def _swap_partner(self, x, num_actors):
        half = num_actors // self.num_agents
        return jnp.concatenate([x[half:], x[:half]], axis=0)

    def postprocess_trajectory(self, traj_batch, config, update_steps):
        """Attach the eta-discounted partner-occupancy target q consumed by aux_loss.

        MATE is loss-only (Eq. 11): the reward stream is left untouched.
        """
        del config, update_steps
        _self_occ, partner_occ = self._occupancy_targets(traj_batch)
        extras = dict(traj_batch.extras)
        extras["ja_future_partner_occ"] = partner_occ
        return traj_batch._replace(extras=extras)

    def aux_loss(self, attn_map_apply, traj_batch, config):
        del config
        if not self.aux_active:
            return self.aux_occ_coef, jnp.float32(0.0)
        target = jax.lax.stop_gradient(traj_batch.extras["ja_future_partner_occ"])
        attn = as_spatial_attention(attn_map_apply)
        # Un-mirror live attention to the true frame to match the target (identity
        # when OP is off); gradient flows through the gather.
        _t, _a = attn.shape[0], attn.shape[1]
        _elem = traj_batch.extras["op_elem"].reshape(_t * _a)
        attn = transform_feat_maps(
            attn.reshape(_t * _a, self.feat_h, self.feat_w), _elem, self._op_table
        ).reshape(_t, _a, self.feat_h, self.feat_w)
        nll = -(target * jnp.log(attn + 1e-8)).sum(axis=(-2, -1))
        valid = (~traj_batch.done).astype(jnp.float32)
        loss = (nll * valid).sum() / jnp.maximum(valid.sum(), 1e-8)
        return self.aux_occ_coef, loss

    def rollout_metrics(self, traj_batch, loss_info):
        return {
            "aux_partner_occ_loss": loss_info.aux_loss.mean(),
            "ja_obj_on_mass": traj_batch.extras["ja_obj_on_mass"].mean(),
        }

    def report(self, config, out, logger):
        from common.train_logging import (
            BASE_SCALAR_KEYS,
            report_basic_training_outputs,
        )
        report_basic_training_outputs(
            config, out, logger,
            scalar_keys=list(BASE_SCALAR_KEYS) + list(self.scalar_keys),
            print_prefix="ja_ippo:lbf_dense_object_occupancy",
        )

    def eval_outputs(self, algorithm_config, env, out, logger):
        del algorithm_config, env, out, logger

    def log_ckpt_video(self, algorithm_config, env, params, policy, tag, savedir, logger):
        """Per-checkpoint attention-overlay video.

        Reconstructs the partner-attention 4th channel via `feed_other_attn_dims`,
        exactly as `augment_obs` does at train time, so the JA image policy gets the
        obs shape it was trained on (the generic trainer video path omits this and
        feeds a 3-channel obs into the 4-channel network).
        """
        import os
        try:
            from evaluation.vis_episodes import make_attention_video, run_episode_with_states
            from marl.eval_lbf import _render_lbf_eval_frames

            inner_env = getattr(env, "_env", env)
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
            # Under OP each agent's attention is in its own mirror frame; pass the
            # per-agent element so the combined video reframes both into GT and the
            # per-agent videos use each agent's own (flipped) view. None when OP off.
            op_elems = read_op_elems(ep_states[0], list(inner_env.agents))
            frames = _render_lbf_eval_frames(inner_env, ep_states)
            # ep_states carries the initial state plus the post-done auto-reset
            # state of the next episode; trim to the attention length so the plain
            # video drops the trailing reset frame and stays aligned with the overlay.
            n_attn = len(attn_data.get("agent_0", []))
            if n_attn:
                frames = frames[:n_attn]
            from moviepy import ImageSequenceClip
            stem = f"{savedir}/{tag.replace('/', '_')}"
            ImageSequenceClip(frames, fps=10).write_videofile(
                f"{stem}.mp4", fps=10, codec="libx264", audio=False, bitrate="8000k", preset="ultrafast",
            )
            logger.log_video(tag, f"{stem}.mp4", commit=False)
            make_attention_video(frames, attn_data, filename=f"{stem}_attn.mp4", fps=10, op_elems=op_elems)
            for suffix in ("agent0", "agent1", "combined"):
                p = f"{stem}_attn_{suffix}.mp4"
                if os.path.exists(p):
                    logger.log_video(f"{tag}_attention_{suffix}", p, commit=False)
        except Exception as e:
            print(f"[ja_ippo:{self.name}] WARN: ckpt attention video failed ({e}); continuing.", flush=True)
