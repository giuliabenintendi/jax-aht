  # Partial Observability (PO)

  ## What changes

  Without PO, both agents see the full grid observation — a (H, W, 26) tensor with 26 channels
  encoding walls, objects, agent positions, pot states, etc. Every cell is visible regardless of
  where the agent is or which way it faces.

  With PO (po_mode="cone"), each agent only sees what's in front of them. The observation tensor is
  the same shape, but cells outside the agent's field of view are zeroed out (or attenuated) before
  being flattened and fed to the policy.

  ## Where it happens

  It starts in envs/__init__.py:82-91. When po_mode != "none" is in the env kwargs, make_env
  instantiates OvercookedPOWrapper instead of OvercookedWrapper. The PO wrapper subclasses the base
  wrapper and overrides _filter_obs, which is called in both reset and step — right after the
  underlying environment produces raw (H, W, 26) observations and before they get flattened.

  ### The filtering pipeline

  _filter_obs (overcooked_po_wrapper.py:109) does this per agent:

  1. Cone mask (_fov_mask): Uses cone_forward_lateral to transform every grid cell into
  agent-relative (forward, lateral) coordinates. A cell is visible if:
    - forward >= 0 (in front, not behind)
    - forward <= fov_range (within max distance)
    - |lateral| <= fov_slope * forward + 1.0 (inside the cone angle — widens with distance)

  Optionally composes with _occlusion_mask, which ray-marches from the agent to each cell and checks
  if any wall blocks the line of sight.
  2. Soft weighting (_soft_weights): When soft_view=True, instead of a hard binary mask, cells inside
   the cone get continuous weights in [0, 1]:
    - Distance decay: exp(-forward / dist_sigma) — closer cells are clearer
    - Angular decay: exp(-(lateral / (forward + 1))² / ang_sigma²) — on-axis cells are clearer

  The final weight per cell is soft_weight * binary_mask, so cells outside the cone are still zero.
  3. Apply: The (H, W, 26) observation is multiplied channel-wise by the (H, W) weight map, then
  flattened as usual.

###  The effect

  The ego agent receives a degraded observation where it literally cannot see what's behind it or far
   to the sides. It must learn to coordinate with a partner it can only partially observe. This is
  the motivation for Joint Attention — if you can't see everything, you should at least learn to look
   at what matters.

  ---
 # Joint Attention (JA)

  ## The idea

  From Lee et al. 2021: instead of just processing observations passively, the ego agent has a
  learnable attention mechanism that decides where on the grid to focus. This attention is encouraged
   to align with where the partner is looking, via an intrinsic reward. The intuition is that
  coordinating agents should attend to the same things — if the partner is looking at a pot, the ego
  should notice that too.

  ## The architecture — ja_actor_critic.py

  The standard RNN policy takes a flat observation → dense layers → GRU → actor/critic heads.

  The JA policy replaces this with a spatial attention pipeline. Inside JAScannedRNN.__call__, at
  each timestep:

  1. Unflatten: The flat observation is reshaped back to (H, W, 26) — recovering the grid structure
  that was lost by flattening.
  2. Feature extraction: A 3×3 Conv layer produces (H, W, 32) feature maps. A spatial basis
  (normalized x,y coordinates, shape (H, W, 2)) is concatenated, giving (H, W, 34). This spatial
  basis ensures positional information survives the attention-weighted sum later.
  3. Keys and Values: Two 1×1 Conv layers project the features to Keys and Values, each (H, W,
  num_heads × head_features). Reshaped to (H*W, num_heads, head_features) — one key-value pair per
  spatial location per head.
  4. Queries from GRU state: The previous GRU hidden state is projected through a Dense layer to
  produce queries (num_heads, head_features). This is the "top-down" component — queries encode what
  the agent is looking for based on its current goals and memory, not just what's in front of it.
  5. Spatial attention: Standard scaled dot-product attention:
  attn_weights = softmax(Keys · Queries / sqrt(d_k))    shape: (H*W, num_heads)
  attended = sum(attn_weights * Values)                   shape: (num_heads, head_features)
  5. Softmax is over the H*W spatial locations — the agent is learning a probability distribution
  over where to look on the grid.
  6. GRU update: The attended features (flattened to a vector) are fed into the GRU, updating the
  hidden state. This hidden state then generates the next timestep's queries — creating a recurrent
  attention loop.
  7. Output: The GRU output goes to actor/critic MLP heads as usual. Additionally, the attention
  weights averaged across heads are returned as an (H, W) attention map.

  ## The intrinsic reward — computed in ppo_ego.py:188-214

  At each environment step (when USE_JA=True):

  1. Ego attention: Comes directly from the policy's attention map output — shape (NUM_ENVS, H, W).
  This is learned — it reflects where the ego's network actually focused.
  2. Partner inferred attention (ja_utils.py:inferred_attention): Constructed from the partner's
  (position, direction) extracted from the environment state. Uses the same Gaussian cone formula as
  the PO soft weights — stronger weight directly ahead, decaying with distance and off-axis angle.
  Normalized to sum to 1. This is a heuristic proxy — we don't have access to the partner's actual
  internal attention, so we approximate it by "they're probably looking where they're facing."
  3. JSD divergence (ja_utils.py:jsd_divergence): Measures how different the two attention
  distributions are. JSD is symmetric, bounded in [0, ln2], and zero when the distributions are
  identical.
  4. Reward: r_ja = -JSD(ego_attn, partner_inferred_attn). Always ≤ 0, maximized at 0 when the ego
  attends exactly where the partner is looking. Added to the environment reward: r_total = r_env +
  beta * r_ja.
  5. Beta curriculum (ppo_ego.py:328-334): beta ramps linearly from 0 to JA_BETA_MAX (0.5) over
  JA_WARMUP_STEPS (5000) update steps. This prevents the JA signal from dominating early training
  when the ego hasn't learned basic task behavior yet. Early on: pure environment reward. Later:
  environment reward + attention alignment pressure.
  6. Stop gradient: r_ja is wrapped in stop_gradient. The JA reward affects the policy only through
  the PPO objective (the reward signal), not by backpropagating directly through the attention
  computation. The attention mechanism learns indirectly — actions that lead to aligned attention
  lead to higher total reward.

  ## How PO and JA interact

  They're independent but complementary:

  - PO alone: The agent has limited vision but no guidance on where to look. It must figure out
  coordination from partial observations.
  - JA alone: The agent sees everything but is rewarded for attending to the same locations as the
  partner. The attention mechanism learns to prioritize task-relevant regions.
  - PO + JA: The agent has limited vision and is learning where to focus within that limited view.
  The JA reward pushes the ego to look at what the partner is looking at — which under partial
  observability is exactly the information it's missing. The attention map becomes a learned "gaze"
  that compensates for the observation mask.

---

# Updates — 17 Feb

## Code cleanup (review-driven)

Commits on `feat/po-joint-attention`:

- `3af6380` scale attention logits by sqrt(d_k) to prevent softmax saturation
- `847ef41` fix JSD numerical stability with separate log terms
- `498a60e` strip PO-specific kwargs when using non-PO wrapper
- `31576b2` tune JA beta warmup defaults to 0.5 over 5000 steps
- `4149dc3` extract `cone_forward_lateral` to shared `envs/overcooked/po_utils.py` (was duplicated in `ja_utils.py` and `overcooked_po_wrapper.py`)
- `2b8e99d` extract `get_inner_env` helper to `envs/base_env.py` (was duplicated in 3 files)
- `f092b6c` deduplicate PO wrapper as subclass of base `OvercookedWrapper` (cut ~60 lines)
- `e70aa31` add error for unknown `EGO_ACTOR_TYPE`, fix `ENV_NAME` key in `run.py`
- `47bc3a9` derive `fov_range` from layout grid size when not specified (`max(h, w) // 2`)

## FOV range is now layout-dependent

Previous default `fov_range=5` covered the entire cramped_room grid (5x4), making PO meaningless. Now defaults to `max(h, w) // 2`:
- cramped_room (5x4): range 2
- coord_ring (5x5): range 2
- forced_coord (5x5): range 2
- asymm_advantages (9x5): range 4

Can still be overridden via `ENV_KWARGS: { fov_range: 3 }` in config.

## Identified issue: JA reward must be FOV-masked

The inferred partner attention is a full Gaussian cone over the entire grid, but the ego's learned attention can only focus on cells it actually sees (non-zero after PO filtering). The JSD penalizes the ego for not attending to cells it cannot observe. Fix needed: before computing JSD in `ppo_ego.py`, mask `a_partner` by the ego's FOV region and re-normalize. This restricts the reward to "within what you *can* see, are you looking where the partner is looking?"

## Design decision: PO everywhere

Partners should also train under PO during population generation (FCP/CoMeDi/BRDiv), not just the ego. FO-trained partners receiving PO-filtered observations during ego training would get inputs they were never trained on. Option 2 (PO everywhere) is the most realistic and internally consistent approach. Requires adding `po_mode: cone` to task configs.