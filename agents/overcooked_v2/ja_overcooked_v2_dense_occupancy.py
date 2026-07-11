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

Visibility gating (`JA_VISIBILITY_GATING`, strict gaze-following): V2 obs are
view-masked, so without gating the feed would leak attention the receiver could
not perceive and the aux would pull attention toward events it could never have
known about. When on, the feed channel is cell-masked to the receiver's view box
and zeroed entirely unless the partner is inside it, and the aux occupancy
target only counts partner-attention events that were witnessable (cell AND
partner in the receiver's view) AT THE MOMENT they happened — masked inside the
backward scan, so an unwitnessed event neither creates a target nor shadows a
later witnessed one. The receiver's attention may still anticipate currently
off-view cells; it is just never supervised by events it could not have seen.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.overcooked_v2.ja_overcooked_v2_attention import (
    agent_tiles_from_state,
    make_visibility_mask_fn,
    object_feature_masks_from_tiles,
    overcooked_v2_object_ctx,
    partner_in_view,
    recipe_contains_ingredient,
    useful_dynamic_item_mask,
    view_feature_masks,
)


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
        ("aux_partner_occ_weighted", "Losses"),
        ("aux_to_policy_loss_abs", "Losses"),
        ("aux_to_value_term_abs", "Losses"),
        ("aux_to_total_loss_abs", "Losses"),
        ("ja_obj_on_mass", "JA"),
        ("ja_partner_visible_frac", "JA"),
        ("ja_aux_active_frac", "JA"),
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
        self.grid_h, self.grid_w = ctx["grid_h"], ctx["grid_w"]
        self.view_size = ctx["agent_view_size"]
        self.egocentric = ctx["egocentric"]
        self.agent_fov_size = ctx["agent_fov_size"]
        self.agent_fov_centered = ctx["agent_fov_centered"]
        self.rotate_obs = ctx["rotate_obs"]
        if self.egocentric and not self.agent_fov_centered:
            raise ValueError(
                "OvercookedV2 egocentric MATE currently supports centered crops only."
            )
        if self.egocentric and self.view_size is None:
            raise ValueError(
                "OvercookedV2 egocentric MATE needs agent_view_size for crop-local "
                "object transforms."
            )
        if self.egocentric and self.agent_fov_size != 2 * self.view_size + 1:
            raise ValueError(
                "OvercookedV2 egocentric MATE expects agent_fov_size to match "
                "2 * agent_view_size + 1."
            )

        self.visibility_gating = bool(config.get("JA_VISIBILITY_GATING", False))
        if self.visibility_gating and self.view_size is None:
            raise ValueError(
                "JA_VISIBILITY_GATING needs a partially observable env "
                "(agent_view_size set); with full observability there is nothing to gate."
            )
        # Fixed feature-grid cell (fr, fc) of each detected task object. Constant
        # across steps and episodes for static objects. Dynamic slots are fixed
        # candidate cells/inventory slots whose validity is recomputed per state.
        self.object_pos = ctx["object_pos"]  # (M, 2) tile row/col in layout frame
        self.static_object_pos = ctx["static_object_pos"]
        self.static_object_cat = ctx["static_object_cat"]
        self.static_object_ing = ctx["static_object_ing"]
        self.dynamic_grid_pos = ctx["dynamic_grid_pos"]
        self.static_num_objects = int(ctx["static_num_objects"])
        self.dynamic_grid_num_objects = int(ctx["dynamic_grid_num_objects"])
        self.inventory_num_objects = int(ctx["inventory_num_objects"])
        self.num_objects = ctx["num_objects"]

        self.partner_coef = float(config.get("JA_FUTURE_PARTNER_COEF", 0.0))
        self.self_coef = float(config.get("JA_FUTURE_SELF_COEF", 0.0))
        self.gamma_occ = float(config.get("JA_FUTURE_GAMMA_OCC", 0.95))
        self.warmup_env_steps = float(config.get("JA_FUTURE_WARMUP_ENV_STEPS", 0.0))
        self.ramp_env_steps = float(config.get("JA_FUTURE_RAMP_ENV_STEPS", 1.0))
        self.feed_other_attn = bool(config.get("FEED_OTHER_ATTN", False))

        self.aux_occ_coef = float(config.get("JA_OBJECT_AUX_COEF", 0.005))
        self.aux_coef = self.aux_occ_coef
        self.aux_active = self.aux_occ_coef > 0.0
        self.vf_coef = float(config.get("VF_COEF", 0.5))

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
        # Positions are unknown before the first step; under gating start fully
        # masked (conservative), otherwise pass-through.
        mask_init = jnp.zeros if self.visibility_gating else jnp.ones
        return {
            "partner_attn": uniform,
            "feed_mask": mask_init((num_actors, self.feat_h, self.feat_w), dtype=jnp.float32),
        }

    def augment_obs(self, obs_batch_2d, carry):
        if not self.feed_other_attn:
            return obs_batch_2d
        rgb = obs_batch_2d.reshape(obs_batch_2d.shape[0], self.img_h, self.img_w, 3)
        partner_attn = carry["partner_attn"]  # shared frame; no OP transform in V2
        # Normalize by the UNMASKED peak, then mask: masked-out attention reads
        # as zero instead of re-normalizing residual leakage into a fake peak.
        peak = jnp.maximum(partner_attn.max(axis=(1, 2), keepdims=True), 1e-8)
        partner_attn = partner_attn * carry["feed_mask"]
        partner_attn_img = jax.image.resize(
            partner_attn,
            (partner_attn.shape[0], self.img_h, self.img_w),
            method="nearest",
        )
        partner_attn_img = partner_attn_img / peak
        return jnp.concatenate([rgb, partner_attn_img[..., None]], axis=-1).reshape(
            obs_batch_2d.shape[0], -1,
        )

    def _actor_tiles(self, env_state, num_actors):
        """Agent-major (rows, cols, partner_rows, partner_cols), each (num_actors,)."""
        rows, cols = agent_tiles_from_state(env_state)  # (num_envs, num_agents)
        rows = rows.swapaxes(0, 1).reshape(-1)
        cols = cols.swapaxes(0, 1).reshape(-1)
        return rows, cols, self._swap_partner(rows, num_actors), self._swap_partner(cols, num_actors)

    def _actor_dirs(self, env_state):
        raw = env_state
        while hasattr(raw, "env_state"):
            raw = raw.env_state
        return raw.agents.dir.swapaxes(0, 1).reshape(-1)

    def _object_roi_mass(self, attn_2d, object_masks, object_visible):
        """Raw attention mass inside each task object's feature-mask footprint.

        attn_2d: (A, feat_h, feat_w). Returns (object_mass (A, M), on_mass (A,)).
        """
        object_mass = (attn_2d[:, None, :, :] * object_masks).sum(axis=(-2, -1))
        object_mass = object_mass * object_visible
        return object_mass, object_mass.sum(axis=-1)

    def _feature_masks_for_actor_positions(self, object_pos_actor, actor_rows, actor_cols,
                                           actor_dirs, valid):
        """Feature masks for per-actor object positions.

        `object_pos_actor` is `(A, M, 2)` in layout tile coordinates. `valid`
        marks whether the slot currently contains a useful target before
        visibility/crop constraints are applied.
        """
        if not self.egocentric:
            local_rows = object_pos_actor[..., 0]
            local_cols = object_pos_actor[..., 1]
            visible = valid.astype(jnp.float32)
            masks = object_feature_masks_from_tiles(
                local_rows,
                local_cols,
                visible,
                tile_size=self.tile_size,
                feat_h=self.feat_h,
                feat_w=self.feat_w,
                img_h=self.img_h,
                img_w=self.img_w,
            )
            return masks, visible

        local_r = object_pos_actor[..., 0] - actor_rows[:, None] + self.view_size
        local_c = object_pos_actor[..., 1] - actor_cols[:, None] + self.view_size
        visible = (
            (local_r >= 0)
            & (local_r < self.agent_fov_size)
            & (local_c >= 0)
            & (local_c < self.agent_fov_size)
            & valid.astype(bool)
        )

        if self.rotate_obs:
            k = jnp.array([0, 2, 1, 3])[actor_dirs]

            def _rot_one(r, c, kk):
                return jax.lax.switch(
                    kk,
                    (
                        lambda x: x,
                        lambda x: (self.agent_fov_size - 1 - x[1], x[0]),
                        lambda x: (self.agent_fov_size - 1 - x[0], self.agent_fov_size - 1 - x[1]),
                        lambda x: (x[1], self.agent_fov_size - 1 - x[0]),
                    ),
                    (r, c),
                )

            local_r, local_c = jax.vmap(_rot_one)(local_r, local_c, k)

        visible = visible.astype(jnp.float32)
        masks = object_feature_masks_from_tiles(
            local_r,
            local_c,
            visible,
            tile_size=self.tile_size,
            feat_h=self.feat_h,
            feat_w=self.feat_w,
            img_h=self.img_h,
            img_w=self.img_w,
        )
        return masks, visible

    def _object_frame(self, env_state, num_actors):
        raw = env_state
        while hasattr(raw, "env_state"):
            raw = raw.env_state
        rows_env, cols_env = agent_tiles_from_state(raw)  # (num_envs, num_agents)
        actor_rows = rows_env.swapaxes(0, 1).reshape(-1)
        actor_cols = cols_env.swapaxes(0, 1).reshape(-1)
        actor_dirs = self._actor_dirs(raw)
        num_envs = rows_env.shape[0]

        recipe = raw.recipe  # (num_envs,)
        actor_recipe = jnp.tile(recipe, self.num_agents)  # agent-major actor order

        # Static actionable objects. Ingredient piles are valid only if their
        # ingredient appears in the current recipe; this removes distractor piles.
        static_pos = jnp.broadcast_to(
            self.static_object_pos[None, :, :],
            (num_actors, self.static_num_objects, 2),
        )
        static_ing = self.static_object_ing[None, :]
        static_is_ing = static_ing >= 0
        safe_static_ing = jnp.maximum(static_ing, 0)
        static_valid = (~static_is_ing) | recipe_contains_ingredient(
            actor_recipe[:, None], safe_static_ing
        )

        # Dynamic objects on the grid: plates, recipe ingredients, and correct
        # cooked dishes. Empty/wrong/distractor cells are invalid targets.
        grid = raw.grid
        dyn = grid[:, self.dynamic_grid_pos[:, 0], self.dynamic_grid_pos[:, 1], 1]
        dyn_valid_env = useful_dynamic_item_mask(dyn, recipe[:, None])
        dyn_valid = jnp.tile(dyn_valid_env, (self.num_agents, 1))
        dyn_pos = jnp.broadcast_to(
            self.dynamic_grid_pos[None, :, :],
            (num_actors, self.dynamic_grid_num_objects, 2),
        )

        # Dynamic objects carried by either agent. Each receiver gets slots for
        # both inventories in its environment, located at the carrier's tile.
        inv = raw.agents.inventory  # (num_envs, num_agents)
        inv_valid_env = useful_dynamic_item_mask(inv, recipe[:, None])
        inv_valid = jnp.tile(inv_valid_env, (self.num_agents, 1))
        inv_pos_env = jnp.stack([rows_env, cols_env], axis=-1)  # (num_envs, num_agents, 2)
        inv_pos = jnp.tile(inv_pos_env, (self.num_agents, 1, 1))

        object_pos_actor = jnp.concatenate([static_pos, dyn_pos, inv_pos], axis=1)
        valid = jnp.concatenate([static_valid, dyn_valid, inv_valid], axis=1)
        return self._feature_masks_for_actor_positions(
            object_pos_actor, actor_rows, actor_cols, actor_dirs, valid
        )

    def _reframe_partner_attention(self, attn_2d, env_state, new_env_state, num_actors):
        """Partner attention from its old crop frame into receiver's new crop frame."""
        partner_attn = self._swap_partner(attn_2d, num_actors)
        if not self.egocentric:
            return partner_attn

        rows_t, cols_t = agent_tiles_from_state(env_state)
        src_rows = self._swap_partner(rows_t.swapaxes(0, 1).reshape(-1), num_actors)
        src_cols = self._swap_partner(cols_t.swapaxes(0, 1).reshape(-1), num_actors)
        src_dirs = self._swap_partner(self._actor_dirs(env_state), num_actors)

        dst_rows, dst_cols = agent_tiles_from_state(new_env_state)
        dst_rows = dst_rows.swapaxes(0, 1).reshape(-1)
        dst_cols = dst_cols.swapaxes(0, 1).reshape(-1)
        dst_dirs = self._actor_dirs(new_env_state)

        feat_r, feat_c = jnp.meshgrid(
            jnp.arange(self.feat_h), jnp.arange(self.feat_w), indexing="ij"
        )
        feat_r = feat_r.reshape(-1)
        feat_c = feat_c.reshape(-1)
        # Nearest tile represented by each source feature cell. The default
        # OvercookedV2 crop has an exact 2x2 feature-cell-per-tile ratio.
        local_r = feat_r * self.agent_fov_size // self.feat_h
        local_c = feat_c * self.agent_fov_size // self.feat_w
        num_cells = self.feat_h * self.feat_w

        def _inv_rot(r, c, direction):
            k = jnp.array([0, 2, 1, 3])[direction]
            return jax.lax.switch(
                k,
                (
                    lambda x: x,
                    lambda x: (x[1], self.agent_fov_size - 1 - x[0]),
                    lambda x: (self.agent_fov_size - 1 - x[0], self.agent_fov_size - 1 - x[1]),
                    lambda x: (self.agent_fov_size - 1 - x[1], x[0]),
                ),
                (r, c),
            )

        def _fwd_rot(r, c, direction):
            k = jnp.array([0, 2, 1, 3])[direction]
            return jax.lax.switch(
                k,
                (
                    lambda x: x,
                    lambda x: (self.agent_fov_size - 1 - x[1], x[0]),
                    lambda x: (self.agent_fov_size - 1 - x[0], self.agent_fov_size - 1 - x[1]),
                    lambda x: (x[1], self.agent_fov_size - 1 - x[0]),
                ),
                (r, c),
            )

        def _one(src_map, sr, sc, sd, dr, dc, dd):
            rr, cc = local_r, local_c
            if self.rotate_obs:
                rr, cc = _inv_rot(rr, cc, sd)
            raw_r = rr + sr - self.view_size
            raw_c = cc + sc - self.view_size
            dst_r = raw_r - dr + self.view_size
            dst_c = raw_c - dc + self.view_size
            valid = (
                (dst_r >= 0)
                & (dst_r < self.agent_fov_size)
                & (dst_c >= 0)
                & (dst_c < self.agent_fov_size)
            )
            if self.rotate_obs:
                dst_r, dst_c = _fwd_rot(dst_r, dst_c, dd)
            centre_r = dst_r * self.tile_size + self.tile_size // 2
            centre_c = dst_c * self.tile_size + self.tile_size // 2
            fr = jnp.clip(centre_r * self.feat_h // self.img_h, 0, self.feat_h - 1)
            fc = jnp.clip(centre_c * self.feat_w // self.img_w, 0, self.feat_w - 1)
            dst_idx = (fr * self.feat_w + fc).astype(jnp.int32)
            mass = jnp.where(valid, src_map.reshape(-1), 0.0)
            return jnp.zeros((num_cells,), dtype=src_map.dtype).at[dst_idx].add(mass).reshape(
                self.feat_h, self.feat_w
            )

        return jax.vmap(_one)(
            partner_attn, src_rows, src_cols, src_dirs, dst_rows, dst_cols, dst_dirs
        )

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del action
        attn_2d = _as_spatial_attention(attn_map).squeeze(0)  # (A, feat_h, feat_w)
        object_masks, object_visible = self._object_frame(env_state, num_actors)
        object_mass, on_mass = self._object_roi_mass(
            attn_2d, object_masks, object_visible
        )  # (A, M), (A,)

        sel = jnp.argmax(object_mass, axis=-1)  # (A,) most-attended object
        attended_visible = object_visible[jnp.arange(num_actors), sel]

        # Receiver tile + partner-visible flag at t (pre-step state = the state
        # that generated obs_t / attn_t); consumed by the aux witness mask and
        # logged as a diagnostic even when gating is off.
        if self.view_size is not None:
            rows_t, cols_t, prow_t, pcol_t = self._actor_tiles(env_state, num_actors)
            pvis_t = partner_in_view(rows_t, cols_t, prow_t, pcol_t, self.view_size)
            self_rc_t = jnp.stack([rows_t, cols_t], axis=-1).astype(jnp.int32)
        else:
            pvis_t = jnp.ones((num_actors,), dtype=jnp.float32)
            self_rc_t = jnp.zeros((num_actors, 2), dtype=jnp.int32)

        # Feed mask consumed at t+1, from the post-step state (whose obs the
        # receiver sees next): receiver's view box, zeroed unless the partner is
        # inside it (strict gaze-following).
        if self.visibility_gating:
            rows_n, cols_n, prow_n, pcol_n = self._actor_tiles(new_env_state, num_actors)
            pvis_n = partner_in_view(rows_n, cols_n, prow_n, pcol_n, self.view_size)
            if self.egocentric:
                feed_mask = jnp.ones(
                    (num_actors, self.feat_h, self.feat_w), dtype=jnp.float32
                ) * pvis_n[:, None, None]
            else:
                feed_mask = view_feature_masks(
                    rows_n, cols_n, self.view_size,
                    self.grid_h, self.grid_w, self.feat_h, self.feat_w,
                ) * pvis_n[:, None, None]
        else:
            feed_mask = jnp.ones((num_actors, self.feat_h, self.feat_w), dtype=jnp.float32)

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
            "ja_future_event_visible": attended_visible,
            "ja_future_object_idx": sel.astype(jnp.int32),
            "ja_future_object_mask_all": object_masks.astype(jnp.float32),
            "ja_future_object_visible_all": object_visible,
            "ja_obj_on_mass": jax.lax.stop_gradient(on_mass),
            "ja_self_rc": self_rc_t,
            "ja_partner_visible": pvis_t,
            "rew_shaping_frac": jnp.broadcast_to(shaping_frac, (num_actors,)),
            "delivery": env_reward,
            "shaped_applied": shaped_total,
        }
        partner_attn_next = self._reframe_partner_attention(
            attn_2d, env_state, new_env_state, num_actors
        )
        uniform = jnp.ones((self.feat_h, self.feat_w), dtype=jnp.float32) / float(
            self.feat_h * self.feat_w
        )
        new_partner_attn = jnp.where(done_actors[:, None, None], uniform[None], partner_attn_next)
        reward = env_reward + shaped_total
        return reward, {"partner_attn": jax.lax.stop_gradient(new_partner_attn),
                        "feed_mask": feed_mask}, extras

    def _attended_mask_occupancy(self, event_mask, done, witness=None, event_valid=None):
        """Discounted first-occupancy over full object footprint masks.

        `event_mask` is `(T, A, feat_h, feat_w)` in the receiver's feature frame.
        """
        box = event_mask.reshape(*event_mask.shape[:2], self.feat_h * self.feat_w)
        box = (box > 0.0).astype(jnp.float32)
        if event_valid is not None:
            box = box * event_valid[..., None]
        if witness is not None:
            box = box * witness

        def _scan(m_next, x_t):
            box_t, done_t = x_t
            m_next = jnp.where(done_t[..., None], 0.0, m_next)
            event_t = box_t.sum(axis=-1, keepdims=True) > 0.0
            m_t = jnp.where(event_t, box_t, self.gamma_occ * m_next)
            return m_t, m_t

        init = jnp.zeros(box.shape[1:], dtype=jnp.float32)
        _, m = jax.lax.scan(_scan, init, (box, done), reverse=True)
        return m / (m.sum(axis=-1, keepdims=True) + 1e-8)

    def _occupancy_targets(self, traj_batch):
        mask_all = traj_batch.extras["ja_future_object_mask_all"]  # (T, A, M, fh, fw)
        vis_all = traj_batch.extras["ja_future_object_visible_all"]  # (T, A, M)
        sel = traj_batch.extras["ja_future_object_idx"]  # (T, A)
        t_dim, a_dim = sel.shape[0], sel.shape[1]
        t_idx = jnp.arange(t_dim)[:, None]
        a_idx = jnp.arange(a_dim)[None, :]
        self_valid = vis_all[t_idx, a_idx, sel]
        self_mask = mask_all[t_idx, a_idx, sel]
        self_occ = self._attended_mask_occupancy(
            self_mask, traj_batch.done,
            event_valid=self_valid,
        ).reshape(t_dim, a_dim, self.feat_h, self.feat_w)

        # Partner target built from the PARTNER's attended cells, witness-masked
        # in the RECEIVER's frame. For egocentric crops, each receiver has its
        # own crop-local coordinates for the partner-selected object, so gather
        # the partner's selected object ID from the receiver's per-object table.
        half = a_dim // 2
        p_sel = jnp.concatenate([sel[:, half:], sel[:, :half]], axis=1)
        p_event_valid_src = jnp.concatenate(
            [self_valid[:, half:], self_valid[:, :half]], axis=1
        )
        p_mask = mask_all[t_idx, a_idx, p_sel]
        p_visible_receiver = vis_all[t_idx, a_idx, p_sel]
        p_event_valid = p_event_valid_src * p_visible_receiver
        witness = None
        if self.visibility_gating:
            if self.egocentric:
                wit = jnp.ones(
                    (t_dim, a_dim, self.feat_h, self.feat_w), dtype=jnp.float32
                ) * traj_batch.extras["ja_partner_visible"][..., None, None]
            else:
                rc = traj_batch.extras["ja_self_rc"]  # (T, A, 2) receiver tile
                wit = view_feature_masks(
                    rc[..., 0], rc[..., 1], self.view_size,
                    self.grid_h, self.grid_w, self.feat_h, self.feat_w,
                ) * traj_batch.extras["ja_partner_visible"][..., None, None]
            witness = wit.reshape(t_dim, a_dim, self.feat_h * self.feat_w)
        partner_occ = self._attended_mask_occupancy(
            p_mask, traj_batch.done,
            witness=witness,
            event_valid=p_event_valid,
        ).reshape(t_dim, a_dim, self.feat_h, self.feat_w)
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
        aux_loss = loss_info.aux_loss.mean()
        weighted_aux = self.aux_occ_coef * aux_loss
        eps = jnp.float32(1e-8)
        return {
            "ja_future_partner_overlap": ex["ja_future_partner_overlap"].mean(),
            "ja_future_self_overlap": ex["ja_future_self_overlap"].mean(),
            "aux_partner_occ_loss": aux_loss,
            "aux_partner_occ_weighted": weighted_aux,
            "aux_to_policy_loss_abs": weighted_aux / (jnp.abs(loss_info.policy_loss.mean()) + eps),
            "aux_to_value_term_abs": weighted_aux / (
                self.vf_coef * jnp.abs(loss_info.value_loss.mean()) + eps
            ),
            "aux_to_total_loss_abs": weighted_aux / (jnp.abs(loss_info.total_loss.mean()) + eps),
            "ja_obj_on_mass": ex["ja_obj_on_mass"].mean(),
            "ja_partner_visible_frac": ex["ja_partner_visible"].mean(),
            # Normalized target sums to ~1 when any witnessed event lies ahead,
            # ~0 otherwise -> mean = aux signal density under gating.
            "ja_aux_active_frac": ex["ja_future_partner_occ"].sum(axis=(-2, -1)).mean(),
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
            from agents.overcooked_v2.ja_overcooked_v2_attention import (
                egocentric_attention_to_world_tiles,
                reframe_partner_attention_for_eval,
            )

            # One-level unwrap (LogWrapper -> image wrapper); NOT get_inner_env, which
            # descends to the raw symbolic OvercookedV2 (no get_avail_actions).
            inner_env = getattr(env, "_env", env)
            env_name = algorithm_config["ENV_NAME"]
            feed_dims = (
                (self.img_h, self.img_w, self.feat_h, self.feat_w)
                if self.feed_other_attn else None
            )
            feed_mask_fn = None
            if self.feed_other_attn and self.visibility_gating:
                feed_mask_fn = make_visibility_mask_fn({
                    "feat_h": self.feat_h, "feat_w": self.feat_w,
                    "grid_h": self.grid_h, "grid_w": self.grid_w,
                    "agent_view_size": self.view_size,
                    "egocentric": self.egocentric,
                })
            feed_reframe_fn = None
            if self.feed_other_attn and self.egocentric:
                feed_reframe_ctx = {
                    "feat_h": self.feat_h,
                    "feat_w": self.feat_w,
                    "img_h": self.img_h,
                    "img_w": self.img_w,
                    "tile_size": self.tile_size,
                    "grid_h": self.grid_h,
                    "grid_w": self.grid_w,
                    "agent_view_size": self.view_size,
                    "agent_fov_size": self.agent_fov_size,
                    "rotate_obs": self.rotate_obs,
                    "egocentric": self.egocentric,
                }

                def feed_reframe_fn(a0, a1, s0, s1):
                    return reframe_partner_attention_for_eval(a0, a1, s0, s1, feed_reframe_ctx)

            max_steps = int(algorithm_config.get("ENV_KWARGS", {}).get("max_steps", 400))
            ep_states, attn_data, _a, _m = run_episode_with_states(
                jax.random.PRNGKey(42), inner_env, params, policy, params, policy,
                max_steps, collect_attention=True, feed_other_attn_dims=feed_dims,
                feed_mask_fn=feed_mask_fn, feed_reframe_fn=feed_reframe_fn,
            )
            os.makedirs(savedir, exist_ok=True)
            frames = get_eval_frames(env_name, inner_env, ep_states)
            n_attn = len(attn_data.get("agent_0", []))
            frames = list(frames[:n_attn] if n_attn else frames)
            if self.egocentric:
                attn_video = {"agent_0": [], "agent_1": []}
                for t in range(min(n_attn, len(ep_states))):
                    a0, a1 = egocentric_attention_to_world_tiles(
                        jnp.asarray(attn_data["agent_0"][t]).squeeze(),
                        jnp.asarray(attn_data["agent_1"][t]).squeeze(),
                        ep_states[t],
                        feed_reframe_ctx,
                    )
                    attn_video["agent_0"].append(a0)
                    attn_video["agent_1"].append(a1)
            else:
                attn_video = attn_data
            stem = f"{savedir}/{tag.replace('/', '_')}"
            ImageSequenceClip(frames, fps=10).write_videofile(
                f"{stem}.mp4", fps=10, codec="libx264", audio=False, preset="ultrafast",
            )
            logger.log_video(tag, f"{stem}.mp4", commit=False)
            make_attention_video(frames, attn_video, filename=f"{stem}_attn.mp4", fps=10)
            for suffix in ("agent0", "agent1", "combined"):
                p = f"{stem}_attn_{suffix}.mp4"
                if os.path.exists(p):
                    logger.log_video(f"{tag}_attention_{suffix}", p, commit=False)
        except Exception as e:
            print(f"[ja_ippo:{self.name}] WARN: ckpt attention video failed ({e}); continuing.", flush=True)
