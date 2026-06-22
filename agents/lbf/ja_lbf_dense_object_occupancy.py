"""Dense attended-object occupancy JA mechanism for LBF.

Closes the gap between the (low-bandwidth) per-apple object aux and the (dense,
high-bandwidth) future-occupancy aux, using ONLY partner attention to task
objects -- no partner actions or positions:

    future occupancy : partner future POSITION   -> dense spatial occupancy target
    this mechanism   : partner-attended APPLE     -> dense spatial occupancy target

Each step, each agent's own spatial attention selects its most-attended apple
(ROI-pooled). That apple's grid cell is fed to the SAME discounted first-occupancy
builder future-occupancy uses for an agent position, and the agent's full spatial
attention map is trained toward the PARTNER's attended-apple occupancy with the
SAME dense cross-entropy. The only change from `FutureOccupancyLBFMechanism` is
the source of the occupancy: the attended apple instead of the agent's position.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing_extensions import override

from agents.lbf.ja_lbf_attention import (
    as_spatial_attention,
    food_state_from_log_state,
    lex_sort_food,
)
from agents.lbf.ja_lbf_future_occupancy import FutureOccupancyLBFMechanism


class LBFDenseObjectOccupancyMechanism(FutureOccupancyLBFMechanism):
    """Dense future-occupancy of the partner-attended apple, attention-only."""

    name = "lbf-ja-dense-object-occupancy"
    scalar_keys = [
        ("ja_future_partner_overlap", "JA"),
        ("ja_future_self_overlap", "JA"),
        ("aux_partner_occ_loss", "Losses"),
        ("ja_obj_on_mass", "JA"),
    ]

    def __init__(self, config, env):
        super().__init__(config, env)
        self.obj_roi_radius = int(config.get("JA_OBJECT_ROI_RADIUS", 1))
        # Target blob half-width on the feature grid. 0 -> single attended-apple
        # cell (original dense variant); >0 -> a (2r+1)^2 box around the apple, so
        # the dense CE rewards attention NEAR the apple, not only on its centre.
        self.obj_target_radius = int(config.get("JA_OBJECT_TARGET_RADIUS", 0))
        # Reuse the object aux coef name; fall back to the future-occupancy one.
        self.aux_occ_coef = float(config.get(
            "JA_OBJECT_AUX_COEF",
            config.get("JA_FUTURE_AUX_PARTNER_OCC_COEF", 0.005),
        ))
        self.aux_coef = self.aux_occ_coef
        self.aux_active = self.aux_occ_coef > 0.0

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

    @override
    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del action, info, new_env_state, update_steps
        attn_2d = as_spatial_attention(attn_map).squeeze(0)  # (A, feat_h, feat_w)
        food_pos, food_eaten = self._food_per_actor(env_state)  # (A, N, 2), (A, N)
        food_mass, on_mass = self._food_roi_mass(attn_2d, food_pos, food_eaten)  # (A, N), (A,)

        sel = jnp.argmax(food_mass, axis=-1)  # (A,) most-attended alive apple
        attended_pos = jnp.take_along_axis(
            food_pos, sel[:, None, None], axis=1).squeeze(1)  # (A, 2) tile [row, col]

        extras = {
            "ja_future_attn_2d": jax.lax.stop_gradient(attn_2d.astype(jnp.float32)),
            "ja_future_pos_post": attended_pos.astype(jnp.int32),
            "ja_obj_on_mass": jax.lax.stop_gradient(on_mass),
        }
        swapped_attn = self._swap_partner(attn_2d, num_actors)
        uniform = jnp.ones((self.feat_h, self.feat_w), dtype=jnp.float32) / float(
            self.feat_h * self.feat_w
        )
        new_partner_attn = jnp.where(done_actors[:, None, None], uniform[None], swapped_attn)
        return env_reward, {"partner_attn": jax.lax.stop_gradient(new_partner_attn)}, extras

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

    @override
    def _occupancy_targets(self, traj_batch):
        # ja_future_pos_post is the attended-apple tile; build the dense occupancy
        # target from a box around its feature cell (postprocess_trajectory and the
        # dense-CE aux_loss are inherited unchanged from FutureOccupancyLBFMechanism).
        pos_rc = traj_batch.extras["ja_future_pos_post"]  # (T, A, 2)
        fr, fc = self._food_feature_coords(pos_rc)  # (T, A)
        self_occ = self._attended_box_occupancy(
            fr, fc, traj_batch.done, self.obj_target_radius,
        ).reshape(pos_rc.shape[0], pos_rc.shape[1], self.feat_h, self.feat_w)
        half = self_occ.shape[1] // 2
        partner_occ = jnp.concatenate([self_occ[:, half:], self_occ[:, :half]], axis=1)
        return self_occ, partner_occ

    @override
    def rollout_metrics(self, traj_batch, loss_info):
        return {
            "ja_future_partner_overlap": traj_batch.extras["ja_future_partner_overlap"].mean(),
            "ja_future_self_overlap": traj_batch.extras["ja_future_self_overlap"].mean(),
            "aux_partner_occ_loss": loss_info.aux_loss.mean(),
            "ja_obj_on_mass": traj_batch.extras["ja_obj_on_mass"].mean(),
        }

    @override
    def report(self, config, out, logger):
        from common.train_logging import (
            IMAGE_IPPO_SCALAR_KEYS,
            report_basic_training_outputs,
        )
        report_basic_training_outputs(
            config, out, logger,
            scalar_keys=list(IMAGE_IPPO_SCALAR_KEYS) + list(self.scalar_keys),
            print_prefix="ja_ippo:lbf_dense_object_occupancy",
        )
