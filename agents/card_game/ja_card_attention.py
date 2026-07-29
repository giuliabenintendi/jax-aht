"""MATE for the Card Alignment Game (paper Sec. 4, Algorithm 1).

Plugs into the unified ja_ippo trainer and implements both MATE components for
this environment:

  Sec. 4.1 - partner-attention feed. Each agent receives the partner's previous
      per-card attention appended to its observation (`augment_obs`). The Eq. 9
      reframing (undo the partner's OP symmetry, apply the receiver's) is the
      permutation translation in `_project_card_attention` / `_partner_card_feed`:
      attention is pooled into the shared unpermuted frame, then re-expressed in
      the receiver's frame.

  Sec. 4.2 - object-grounded auxiliary loss. The five cards are the object set
      E_t, one region R(e) per card (`build_card_masks`). `postprocess_trajectory`
      builds the partner's eta-discounted future focus q_t^j by a reverse scan over
      its per-step attention argmax; `aux_loss` is the cross-entropy of the agent's
      own card distribution p_t^i against that target (Eq. 10), weighted by
      lambda_MATE = JA_AUX_PARTNER_ARGMAX_COEF (Eq. 11). eta is JA_FUTURE_GAMMA_OCC,
      default 0.9 for this env.

The module also hosts the Lee et al. (2021) baseline (JA_CARD_JSD_COEF), which
applies their beta * r^JA intrinsic reward (Preliminaries Eq. 7-8) at the card
level. MATE does not use it: MATE is loss-only.

Other-Play (position shuffle + recolouring, Sec. 5.1) is required for the card
metric; without it the shared frame is undefined.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.ja_utils import build_card_masks, jsd_divergence


class CardMechanism:
    """Env-specific JA hooks for the OP card game (2 agents, parameter-shared)."""

    name = "card-game"
    scalar_keys = [
        ("jsd_mean", "JA/jsd"),
        ("card_jsd_mean", "JA/card_jsd"),
        ("ja_attn_shaping_mean", "JA/total_shaping"),
        ("ja_attn_self_mean", "JA/action_matches_own_attn"),
        ("ja_gaze_pick_mean", "JA/action_matches_prev_partner_attn"),
        ("aux_partner_argmax_loss", "JA/aux_partner_argmax_nll"),
    ]

    def __init__(self, config, env):
        from agents.initialize_agents import _get_image_dims
        from agents.ja_actor_critic import _compute_resnet_output_dims

        self.num_agents = env.num_agents
        if self.num_agents != 2:
            raise ValueError("CardMechanism assumes 2 agents (parameter-shared).")

        self.ja_card_attn = bool(config.get("JA_CARD_ATTN", False))
        self.ja_card_partner_feed = self.ja_card_attn and bool(config.get("JA_CARD_PARTNER_FEED", True))
        self.attn_self_coef = float(config.get("JA_ATTN_SELF_COEF", 0.0))
        self.gaze_pick_coef = float(config.get("JA_GAZE_PICK_COEF", 0.0))
        # lambda_MATE in Eq. 11.
        self.aux_coef = float(config.get("JA_AUX_PARTNER_ARGMAX_COEF", 0.0))
        # Lee et al. (2021) intrinsic reward: per-step penalty -coef * card_jsd, pulling
        # the two agents' per-card attention distributions together in the shared frame.
        self.card_jsd_coef = float(config.get("JA_CARD_JSD_COEF", 0.0))
        self.gaze_pick_active = self.gaze_pick_coef > 0
        self.aux_active = self.aux_coef > 0
        self.card_jsd_penalty_active = self.card_jsd_coef > 0
        # Aux target: partner's gamma-discounted first-occupancy over its ATTENTION
        # (argmax each step) — the LBF future-occupancy method on the card game's
        # observable, co-adaptive intent signal (NOT the raw emission, which is an
        # unlearnable stochastic action). The plain-argmax aux is the gamma->0 limit.
        # Built in postprocess_trajectory; gamma reuses the LBF occupancy discount.
        self.aux_target_pick = bool(config.get("JA_AUX_PARTNER_PICK", False))
        # eta in Eq. 10: discount on the partner's future focus.
        self.aux_pick_gamma = float(config.get("JA_FUTURE_GAMMA_OCC", 0.9))
        self.attn_shaping_active = self.attn_self_coef > 0
        self.prev_partner_phys_active = self.gaze_pick_active
        self.card_metric = (
            self.ja_card_attn or bool(config.get("JA_CARD_METRIC", False))
            or self.attn_shaping_active or self.aux_active or self.gaze_pick_active
            or self.card_jsd_penalty_active
        )
        # Other-Play provides each agent's independent position/recolour permutations.
        # The two components are corrected independently: position shuffle governs the
        # attention->card projection frame, recolouring governs the emitted-action->card
        # inversion. Any combination is valid -- both (full OP), either alone, or neither
        # (JA-only control, falling back to the base env's shared card_permutation and
        # identity recolouring).
        ek = config.get("ENV_KWARGS", {})
        self.op_pos_active = bool(ek.get("other_play_position_shuffle"))
        self.op_recolour_active = bool(ek.get("other_play_recolouring"))
        self.op_active = self.op_pos_active and self.op_recolour_active

        img_h, img_w, _ = _get_image_dims(env)
        feat_h, feat_w = _compute_resnet_output_dims(
            img_h, img_w,
            stride=config.get("CONV_STRIDE", 2),
            kernel_size=config.get("CONV_KERNEL_SIZE", 3),
            padding=config.get("CONV_PADDING", "SAME"),
            num_blocks=config.get("CONV_NUM_BLOCKS", 4),
        )
        self.card_masks = (
            build_card_masks(img_h, img_w, feat_h, feat_w)
            if (self.card_metric and feat_h > 0) else None
        )
        self.num_cards = int(self.card_masks.shape[0]) if self.card_masks is not None else 5
        self.partner_feed_dim = 5

    # ------------------------------------------------------------------ hooks
    def entity_feed_dim(self) -> int:
        # The +5 partner-card suffix is added by initialize_ja_image_agent via
        # JA_CARD_ATTN + JA_CARD_PARTNER_FEED, not via JA_ENTITY_FEED_DIM.
        return 0

    def init_carry(self, num_actors):
        return {
            "partner_card_attn": jnp.zeros((num_actors, self.partner_feed_dim), dtype=jnp.float32),
            "prev_partner_phys": jnp.zeros((num_actors, self.num_cards), dtype=jnp.float32),
            "prev_partner_valid": jnp.zeros((num_actors,), dtype=bool),
            "prev_partner_argmax": jnp.zeros((num_actors,), dtype=jnp.int32),
            "prev_partner_argmax_valid": jnp.zeros((num_actors,), dtype=bool),
            "prev_partner_argmax_weight": jnp.zeros((num_actors,), dtype=jnp.float32),
        }

    def augment_obs(self, obs_batch_2d, carry):
        if not self.ja_card_partner_feed:
            return obs_batch_2d
        return jnp.concatenate([obs_batch_2d, carry["partner_card_attn"]], axis=-1)

    def step(self, *, attn_map, env_state, new_env_state, action, env_reward,
             info, done_actors, carry, num_actors, update_steps):
        del new_env_state
        num_envs = num_actors // self.num_agents
        num_cards = self.num_cards

        info.pop("step_count", None)

        # Spatial JSD diagnostic. Stored per env then tiled to actors so every
        # extras leaf has a num_actors axis (required by _create_minibatches);
        # rollout_metrics reduces it. Meaningless under OP but kept for parity.
        a0 = attn_map[:, :num_envs, ...].squeeze(0)
        a1 = attn_map[:, num_envs:, ...].squeeze(0)
        jsd_per_env = jsd_divergence(a0, a1)
        jsd_spatial = jnp.concatenate([jsd_per_env, jsd_per_env])

        if self.card_metric:
            (perm_0, perm_1, phys_0, phys_1,
             q_phys_0, q_phys_1, m_0, m_1) = self._project_card_attention(attn_map, env_state, num_envs)
            card_jsd_env = jsd_divergence(q_phys_0[:, None, :], q_phys_1[:, None, :])
            card_jsd = jnp.concatenate([card_jsd_env, card_jsd_env])
            (partner_argmax_per_actor,
             current_partner_phys, current_partner_mass) = self._partner_argmax_targets(
                phys_0, phys_1, q_phys_0, q_phys_1, perm_0, perm_1, m_0, m_1)
        else:
            perm_0 = perm_1 = jnp.zeros((num_envs, num_cards), dtype=jnp.int32)
            phys_0 = phys_1 = q_phys_0 = q_phys_1 = jnp.zeros((num_envs, num_cards), dtype=jnp.float32)
            card_jsd = jnp.zeros((num_actors,), dtype=jnp.float32)
            partner_argmax_per_actor = jnp.zeros((num_actors,), dtype=jnp.int32)
            current_partner_phys = jnp.zeros((num_actors, num_cards), dtype=jnp.float32)
            current_partner_mass = jnp.zeros((num_actors,), dtype=jnp.float32)

        # Aux target: causal-shifted (prev step) when the partner-feed is on.
        if self.aux_active and self.ja_card_partner_feed:
            aux_argmax = carry["prev_partner_argmax"]
            aux_valid = carry["prev_partner_argmax_valid"]
            aux_weight = carry["prev_partner_argmax_weight"]
        else:
            aux_argmax = partner_argmax_per_actor
            aux_valid = jnp.ones((num_actors,), dtype=bool)
            aux_weight = current_partner_mass

        # Map emitted action to its canonical card identity (invert recolouring).
        env_idx = jnp.arange(num_envs)
        pick_0_view = action[:num_envs]
        pick_1_view = action[num_envs:]
        pick_0_is_card = pick_0_view < num_cards
        pick_1_is_card = pick_1_view < num_cards
        if self.card_metric and self.op_recolour_active:
            inv_recol = self._locate_env_attr(env_state, "per_agent_inv_recolouring")
            inv_recol_0 = inv_recol["agent_0"]
            inv_recol_1 = inv_recol["agent_1"]
            action_0_gt = inv_recol_0[env_idx, jnp.minimum(pick_0_view, num_cards - 1)]
            action_1_gt = inv_recol_1[env_idx, jnp.minimum(pick_1_view, num_cards - 1)]
        elif self.card_metric:
            # No recolouring (JA-only control): the emitted action already is the card identity.
            action_0_gt = jnp.minimum(pick_0_view, num_cards - 1).astype(jnp.int32)
            action_1_gt = jnp.minimum(pick_1_view, num_cards - 1).astype(jnp.int32)
        else:
            action_0_gt = jnp.zeros((num_envs,), dtype=jnp.int32)
            action_1_gt = jnp.zeros((num_envs,), dtype=jnp.int32)

        (r_attn_shaping, r_attn_self, r_gaze_pick) = self._shaping_rewards(
            q_phys_0, q_phys_1, carry["prev_partner_phys"], carry["prev_partner_valid"],
            action_0_gt, action_1_gt, pick_0_is_card, pick_1_is_card, num_actors)

        if self.card_jsd_penalty_active:
            r_card_jsd = jax.lax.stop_gradient(-self.card_jsd_coef * card_jsd)
        else:
            r_card_jsd = jnp.zeros((num_actors,))
        reward = env_reward + r_attn_shaping + r_gaze_pick + r_card_jsd

        # Advance carry (reset on episode boundary).
        if self.ja_card_attn:
            new_partner_card_attn = self._partner_card_feed(done_actors, perm_0, perm_1, phys_0, phys_1)
        else:
            new_partner_card_attn = carry["partner_card_attn"]
        if self.prev_partner_phys_active:
            new_prev_phys = jnp.where(done_actors[:, None], 0.0, current_partner_phys).astype(jnp.float32)
            new_prev_valid = ~done_actors
        else:
            new_prev_phys = carry["prev_partner_phys"]
            new_prev_valid = carry["prev_partner_valid"]
        if self.aux_active and self.ja_card_partner_feed:
            new_prev_argmax = jnp.where(done_actors, 0, partner_argmax_per_actor).astype(jnp.int32)
            new_prev_argmax_valid = ~done_actors
            new_prev_argmax_weight = jnp.where(done_actors, 0.0, current_partner_mass).astype(jnp.float32)
        else:
            new_prev_argmax = carry["prev_partner_argmax"]
            new_prev_argmax_valid = carry["prev_partner_argmax_valid"]
            new_prev_argmax_weight = carry["prev_partner_argmax_weight"]

        new_carry = {
            "partner_card_attn": new_partner_card_attn,
            "prev_partner_phys": new_prev_phys,
            "prev_partner_valid": new_prev_valid,
            "prev_partner_argmax": new_prev_argmax,
            "prev_partner_argmax_valid": new_prev_argmax_valid,
            "prev_partner_argmax_weight": new_prev_argmax_weight,
        }
        # Partner's current ATTENTION argmax this step (already translated to the
        # agent's view frame) — the observable, co-adaptive intent signal, analogous
        # to the partner's position in LBF. postprocess builds the gamma-discounted
        # future-occupancy over this sequence.
        partner_attn_argmax_step = partner_argmax_per_actor.astype(jnp.int32)

        extras = {
            "partner_attn_argmax_step": partner_attn_argmax_step,
            "partner_argmax": aux_argmax,
            "partner_argmax_valid": aux_valid,
            "partner_argmax_weight": aux_weight,
            "card_jsd": card_jsd, "jsd": jsd_spatial,
            "r_attn_shaping": r_attn_shaping,
            "r_attn_self": r_attn_self, "r_gaze_pick": r_gaze_pick,
        }
        return reward, new_carry, extras

    def postprocess_trajectory(self, traj_batch, config, update_steps):
        """Build the partner's eta-discounted future focus q_t^j (Eq. 10 target).

        Algorithm 1, lines 11-13: find the partner's focus m_t^j = argmax_e p_t^j(e),
        then q_t^j <- eta * q_{t+1}^j with q_t^j(m_t^j) <- 1, normalised.


        Each step the partner attends to some card (argmax of its per-card
        attention); over the episode this forms a discrete trajectory. A reverse
        scan turns it into a gamma-discounted first-occupancy distribution over
        cards — the SAME construction as LBF's spatial future-occupancy, in the
        5-card space: the card attended now scores 1, the next new one gamma, etc.
        The partner's attention is observable (it is fed) and co-adaptive, so unlike
        the raw emission it gives a learnable target. No-op unless the occupancy aux
        is active, so the plain-argmax path is untouched.
        """
        del config, update_steps
        if not (self.aux_active and self.aux_target_pick):
            return traj_batch
        ex = traj_batch.extras
        done = traj_batch.done                          # (T, A)
        # Hard discounted FIRST-occupancy over the partner's attention argmax.
        onehot = jax.nn.one_hot(ex["partner_attn_argmax_step"], self.num_cards, dtype=jnp.float32)
        gamma = self.aux_pick_gamma

        def _scan(m_next, x):
            oh_t, done_t = x
            m_next = jnp.where(done_t[..., None], 0.0, m_next)
            m_t = jnp.where(oh_t > 0.0, 1.0, gamma * m_next)
            return m_t, m_t

        init = jnp.zeros(onehot.shape[1:], dtype=jnp.float32)                      # (A, C)
        _, occ = jax.lax.scan(_scan, init, (onehot, done), reverse=True)           # (T, A, C)
        occ = occ / (occ.sum(axis=-1, keepdims=True) + 1e-8)
        new_extras = dict(ex)
        new_extras["partner_card_occ"] = jax.lax.stop_gradient(occ)
        return traj_batch._replace(extras=new_extras)

    def aux_loss(self, attn_map_apply, traj_batch, config):
        """L_MATE for the card game (Eq. 10), returned with lambda_MATE (Eq. 11).

        Pools attention onto cards to get p_t^i (Algorithm 1, line 9-10) and takes
        the cross-entropy against the partner target, weighted by the agent's
        on-card attention mass. Target is the eta-discounted future focus q_t^j
        when `JA_AUX_PARTNER_PICK` is set (the paper setting), else the partner's
        plain attention argmax, which is the eta -> 0 limit.
        """
        del config
        if not self.aux_active:
            return self.aux_coef, jnp.float32(0.0)
        ex = traj_batch.extras
        per_card_attn = jnp.einsum("tahw,chw->tac", attn_map_apply, self.card_masks)
        own_card_mass = per_card_attn.sum(axis=-1)
        per_card_norm = per_card_attn / (own_card_mass[..., None] + 1e-8)
        log_probs = jnp.log(per_card_norm + 1e-8)
        if self.aux_target_pick:
            # Soft cross-entropy against the partner's discounted card-occupancy
            # (mirrors LBF's occupancy aux), weighted by the agent's on-card mass.
            # The terminal pick still supervises earlier timesteps via the reverse
            # scan target; done rows themselves are excluded, as in LBF.
            target = ex["partner_card_occ"]
            nll_flat = -(target * log_probs).sum(axis=-1).reshape(-1)
            valid = (~traj_batch.done).astype(jnp.float32)
            aux_weight_flat = jax.lax.stop_gradient(
                valid.reshape(-1) * own_card_mass.reshape(-1)
            )
        else:
            target_flat = ex["partner_argmax"].reshape(-1)
            log_probs_flat = log_probs.reshape(-1, log_probs.shape[-1])
            nll_flat = -jnp.take_along_axis(
                log_probs_flat, target_flat[:, None], axis=-1,
            ).squeeze(-1)
            aux_weight_flat = jax.lax.stop_gradient(
                ex["partner_argmax_valid"].reshape(-1).astype(jnp.float32)
                * ex["partner_argmax_weight"].reshape(-1)
                * own_card_mass.reshape(-1)
            )
        aux_loss = (nll_flat * aux_weight_flat).sum() / jnp.maximum(aux_weight_flat.sum(), 1e-8)
        return self.aux_coef, aux_loss

    def rollout_metrics(self, traj_batch, loss_info):
        ex = traj_batch.extras
        gaze_count = jnp.maximum(jnp.asarray(ex["r_gaze_pick"].size, dtype=jnp.float32), 1.0)
        return {
            "jsd_mean": ex["jsd"].mean(),
            "card_jsd_mean": ex["card_jsd"].mean(),
            "ja_attn_shaping_mean": ex["r_attn_shaping"].mean(),
            "ja_attn_self_mean": ex["r_attn_self"].mean(),
            "ja_gaze_pick_mean": ex["r_gaze_pick"].sum() / gaze_count,
            "aux_partner_argmax_loss": loss_info.aux_loss.mean(),
        }

    def report(self, config, out, logger):
        from common.train_logging import report_ja_training_outputs
        report_ja_training_outputs(config, out, logger)

    def eval_outputs(self, algorithm_config, env, out, logger):
        from marl.eval_logging import log_eval_video, log_greedy_eval
        log_greedy_eval(algorithm_config, env, out, logger)
        log_eval_video(algorithm_config, env, out, logger)

    # -------------------------------------------------------------- internals
    def _shared_card_permutation(self, env_state):
        """Base env's shared per-episode card_permutation (no-OP JA control).

        Walks the wrapper chain to the inner CardGameState. Shape (num_envs, num_cards):
        card_permutation[pos] = the card identity shown at screen position pos.
        """
        s = env_state
        for _ in range(6):
            if hasattr(s, "card_permutation"):
                return s.card_permutation
            s = getattr(s, "env_state", None)
            if s is None:
                break
        raise AttributeError("could not locate card_permutation in env_state")

    def _locate_env_attr(self, env_state, attr):
        """Walk the wrapper chain to the OP state carrying `attr`.

        OP position/recolour wrappers are each applied only when their flag is set, so
        the per-agent maps sit at different depths depending on which are active. Locate
        by attribute rather than a fixed nesting depth so single-component OP works.
        """
        s = env_state
        for _ in range(6):
            if hasattr(s, attr):
                return getattr(s, attr)
            s = getattr(s, "env_state", None)
            if s is None:
                break
        raise AttributeError(f"could not locate {attr} in env_state")

    def _project_card_attention(self, attn_map, env_state, num_envs):
        """Pool spatial attention onto cards and translate to canonical frame."""
        attn = attn_map.squeeze(0)  # (num_actors, fh, fw)
        card_pos_attn = jnp.einsum("ahw,chw->ac", attn, self.card_masks)
        card_pos_attn_0 = card_pos_attn[:num_envs]
        card_pos_attn_1 = card_pos_attn[num_envs:]
        eps = 1e-8
        if self.op_pos_active:
            per_agent_perm = self._locate_env_attr(env_state, "per_agent_perm")
            perm_0 = per_agent_perm["agent_0"]
            perm_1 = per_agent_perm["agent_1"]
        else:
            # No position OP: both agents share the base env's card_permutation frame.
            # In recolour-only OP this shared layout is still shuffled.
            shared_perm = self._shared_card_permutation(env_state)
            perm_0 = perm_1 = shared_perm
        batch_idx = jnp.arange(num_envs)[:, None]
        phys_0 = jnp.zeros((num_envs, self.num_cards)).at[batch_idx, perm_0].set(card_pos_attn_0)
        phys_1 = jnp.zeros((num_envs, self.num_cards)).at[batch_idx, perm_1].set(card_pos_attn_1)
        m_0 = card_pos_attn_0.sum(axis=-1)
        m_1 = card_pos_attn_1.sum(axis=-1)
        q_phys_0 = phys_0 / (m_0[:, None] + eps)
        q_phys_1 = phys_1 / (m_1[:, None] + eps)
        return perm_0, perm_1, phys_0, phys_1, q_phys_0, q_phys_1, m_0, m_1

    def _partner_argmax_targets(self, phys_0, phys_1, q_phys_0, q_phys_1, perm_0, perm_1, m_0, m_1):
        partner_canon_argmax_0 = phys_1.argmax(axis=-1)
        partner_canon_argmax_1 = phys_0.argmax(axis=-1)
        inv_perm_0 = jnp.argsort(perm_0, axis=-1)
        inv_perm_1 = jnp.argsort(perm_1, axis=-1)
        am0 = jnp.take_along_axis(inv_perm_0, partner_canon_argmax_0[:, None], axis=1).squeeze(-1)
        am1 = jnp.take_along_axis(inv_perm_1, partner_canon_argmax_1[:, None], axis=1).squeeze(-1)
        partner_argmax = jnp.concatenate([am0, am1]).astype(jnp.int32)
        cur_phys = jnp.concatenate([q_phys_1, q_phys_0], axis=0)
        cur_mass = jnp.concatenate([m_1, m_0])
        return (
            jax.lax.stop_gradient(partner_argmax),
            jax.lax.stop_gradient(cur_phys.astype(jnp.float32)),
            jax.lax.stop_gradient(cur_mass.astype(jnp.float32)),
        )

    def _partner_card_feed(self, done_actors, perm_0, perm_1, phys_0, phys_1):
        translated_for_0 = jnp.take_along_axis(phys_1, perm_0, axis=1)
        translated_for_1 = jnp.take_along_axis(phys_0, perm_1, axis=1)
        partner_card_attn = jnp.concatenate([translated_for_0, translated_for_1], axis=0)
        return jnp.where(done_actors[:, None], 0.0, partner_card_attn)

    def _shaping_rewards(self, q_phys_0, q_phys_1, prev_partner_phys, prev_partner_valid,
                         action_0_gt, action_1_gt, pick_0_is_card, pick_1_is_card, num_actors):
        if self.card_metric and self.attn_shaping_active:
            oam0 = jnp.take_along_axis(q_phys_0, action_0_gt[:, None], axis=1).squeeze(-1)
            oam1 = jnp.take_along_axis(q_phys_1, action_1_gt[:, None], axis=1).squeeze(-1)
            oam0 = jnp.where(pick_0_is_card, oam0, 0.0)
            oam1 = jnp.where(pick_1_is_card, oam1, 0.0)
            r_self = self.attn_self_coef * jnp.concatenate([oam0, oam1])
        else:
            r_self = jnp.zeros(num_actors)
        r_shaping = r_self

        if self.gaze_pick_active:
            action_canon = jnp.concatenate([action_0_gt, action_1_gt])
            picked_mass = jnp.take_along_axis(
                prev_partner_phys, action_canon[:, None], axis=1,
            ).squeeze(-1)
            action_is_card = jnp.concatenate([pick_0_is_card, pick_1_is_card])
            ok = prev_partner_valid & action_is_card
            r_gaze = jnp.where(ok, self.gaze_pick_coef * picked_mass, 0.0)
        else:
            r_gaze = jnp.zeros(num_actors)

        return (
            jax.lax.stop_gradient(r_shaping),
            jax.lax.stop_gradient(r_self),
            jax.lax.stop_gradient(r_gaze),
        )
