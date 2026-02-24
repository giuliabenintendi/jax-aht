# JA-IPPO: Paper vs Implementation Comparison

Reference: Lee et al. (2021), "Joint Attention for Multi-Agent Coordination
and Social Learning"

This document tracks how our JAX implementation maps to the paper.
Updated whenever the code changes.

---

## 0. What We're Doing

Two agents learn to cook together in Overcooked. Each runs independent PPO
with no shared parameters — separate networks, separate optimizers, separate
gradient updates. Left alone, independent learners have no mechanism to
coordinate: they develop policies in isolation and hope for the best at
test time.

Lee et al. (2021) add two things that change this:

**Spatial attention as the policy backbone.** Instead of a standard
MLP or plain LSTM policy, each agent's network processes its grid observation
through a learned spatial attention mechanism. At each timestep the agent
produces an attention map over the H x W grid — a probability distribution
over locations — and uses the attended features to decide what to do. The
attention map reflects where the agent is "looking" to make its decision.

The architecture per agent is:

    obs grid → Conv(3x3, 64) → spatial features F
    F + sinusoidal positional encoding → 1x1 Conv → Keys K, Values V
    concat(own LSTM h, partner's LSTM h) → Dense → Query Q
    softmax(Q . K) → attention weights A → weighted sum of V → attended output O
    concat(O, direction embed, position embed) → LSTM → FC → FC → action logits

Actor and critic are identical copies of this pipeline with no shared weights.

**Cross-agent state sharing and a joint attention incentive.** The query Q
is conditioned on *both* agents' recurrent states: agent i receives agent j's
LSTM hidden state h and concatenates it with its own h to form the query.
This means where agent i looks depends on what agent j has been doing.

On top of this, a reward bonus r_JA = -JSD(A^0, A^1) penalizes divergence
between the two agents' attention maps. The JSD (Jensen-Shannon divergence)
is zero when both agents attend to exactly the same spatial distribution and
maximal when they look at completely different places. The negative sign makes
agreement rewarding. This bonus is scaled by a coefficient beta that ramps
linearly from 0 to 1e-2 over the first 200k steps, so agents first learn
basic skills before being nudged toward attentional agreement.

The total objective for agent k is: J(pi^k) = E[sum gamma^t (r_env + beta * r_JA)].

The result is that agents develop a shared "language" of attention — they learn
to look at the same task-relevant features (the pot that needs onions, the
delivery counter) even though they never share parameters or communicate
explicitly. The cross-agent h in the query gives each agent a sense of what
the other is tracking, and the JSD bonus reinforces convergence.

**What this implementation does.** We port this method from the paper's
MultiGrid environments to Overcooked-v1 in JAX. The core architecture matches
the paper's Appendix A. The main adaptation is observation routing: Overcooked
uses a 26-channel spatial encoding rather than MultiGrid's structured dict, so
we map channels to the appropriate network inputs (image channels to conv,
direction/position to scalar embeddings). The training loop runs two
independent PPO instances with cross-agent h exchange and JSD reward
augmentation, matching the paper's "two independent PPO architectures."

---

## 1. Observation Processing

### 1.1 Convolutional feature extraction

**Paper (Appendix A):**
> "The environment observation is first processed by a 3x3 convolutional layer
> with 64 filters, stride 1, padding to maintain the size of the input, and
> ReLU activations."

**Our code** (`ja_actor_critic.py:111-119`):
Conv(3x3, 64 filters, SAME padding) followed by ReLU.

| Detail | Paper | Ours | Match |
|--------|-------|------|-------|
| Kernel | 3x3 | 3x3 | Y |
| Filters | 64 | 64 | Y |
| Stride | 1 | 1 (default) | Y |
| Padding | "maintain size" | SAME | Y |
| Activation | ReLU | ReLU | Y |

**Status: MATCH**

### 1.2 Input channels — MultiGrid vs Overcooked

**Paper (Section 4):**
> "Given an input image X in $R^{h x w x c}$ with c channels..."

The paper uses MultiGrid environments where each agent's observation is a
**structured dict** with separate keys, each routed to its own preprocessing
layer (`attention_networks.py:225-248`):

| Key | Shape | Preprocessing | Routed to |
|-----|-------|--------------|-----------|
| `image` | (H, W, C) | Conv(3x3, 64) → ReLU | Attention (K, V) |
| `direction` | scalar (0-3) | one_hot(4) → Dense(5) | Scalar embed |
| `position` | (x, y) | cast_and_scale → Dense(5) | Scalar embed |
| `policy_state` | (h, c) tuple | identity | Query (Q) |

In MultiGrid, agents are rendered directly in the `image` — they appear as
colored sprites in the grid. So the Conv layer can see where other agents are.

**Overcooked** uses a different representation: a single (H, W, 26) tensor of
binary/integer feature planes (`jaxmarl/.../overcooked.py:251-365`).
These are NOT pixel images — each channel is a spatial mask:

| Channels | Content | Source |
|----------|---------|--------|
| 0 | Ego position (1 at agent cell) | `overcooked.py:315` |
| 1 | Partner position (1 at agent cell) | `overcooked.py:316` |
| 2-5 | Ego direction (one-hot, at ego's cell) | `overcooked.py:346-348` |
| 6-9 | Partner direction (one-hot, at partner's cell) | `overcooked.py:346-348` |
| 10 | Pot locations | `overcooked.py:327` |
| 11 | Counter/wall locations | `overcooked.py:328` |
| 12 | Onion pile locations | `overcooked.py:329` |
| 13 | Tomato pile locations (always zeros) | `overcooked.py:330` |
| 14 | Plate pile locations | `overcooked.py:331` |
| 15 | Delivery/goal locations | `overcooked.py:332` |
| 16 | Onions in pot (0-3, not cooking) | `overcooked.py:307,333` |
| 17 | Tomatoes in pot (always zeros) | `overcooked.py:334` |
| 18 | Onions in soup (cooking/ready) | `overcooked.py:308-309,335` |
| 19 | Tomatoes in soup (always zeros) | `overcooked.py:336` |
| 20 | Pot cooking time remaining (19→0) | `overcooked.py:310,337` |
| 21 | Soup ready indicator | `overcooked.py:311,338` |
| 22 | Plate locations on counters | `overcooked.py:339` |
| 23 | Onion locations on counters | `overcooked.py:340` |
| 24 | Tomato locations (always zeros) | `overcooked.py:341` |
| 25 | Urgency (all 1s when ≤40 steps remain) | `overcooked.py:312,342` |

**Our channel routing** (`ja_actor_critic.py:37-40,110-121,166-193`):

| Channels | Route | What the network sees |
|----------|-------|----------------------|
| 0-1 | Conv → attention (K, V) | Agent positions (spatially, like MultiGrid sprites) |
| 10-25 | Conv → attention (K, V) | Terrain, pots, objects, urgency |
| 2-9 | `.sum(axis=(1,2))` → Dense(5) | Direction vectors (spatial info discarded) |
| 0 | `argmax` → (x, y) → Dense(5) | Ego position as coordinates |
| 1 | `argmax` → (x, y) → Dense(5) | Partner position as coordinates |

### 1.3 Partner visibility

**Previously:** Channel 1 (partner position) was never read by the network —
excluded from the image path and not extracted as scalar coordinates.

**Fix (2026-02-23):** Two changes restore partner visibility:
1. Channels 0-1 (agent positions) are now concatenated with channels 10-25
   before the Conv layer, so the conv sees both agents spatially — matching
   MultiGrid where agents are rendered directly in the image.
2. Partner position is extracted as (x, y) coordinates via `argmax` and
   included in `pos_embed` alongside ego position (Dense input: 4 → 5),
   mirroring how `dir_embed` receives both agents' directions.

**Indirect signals (still present but no longer the only path):**
- Channels 6-9 encode partner direction at the partner's grid cell.
- Channels 10-25: agent positions are overwritten with held items.

**Status: FIXED**

---

## 2. Spatial Attention

### 2.1 Spatial basis (positional encoding)

**Paper (Appendix A):**
> "we concatenate a spatial basis of depth 8 to the image features"

**Paper (Appendix A.1) equations:**
Uses 1-indexed frequencies: i = 1..c_s/4, giving div = 10^{-2i/(c_s/2)}.

**Reference code** (`attention_networks.py:97-109`):
Uses 0-indexed: `arange(0, half_d, 2)`, giving different frequency set.

**Our code** (`ja_utils.py:71-96`):
Direct port of the reference code's `get_spatial_basis()`, not the paper's
equations.

| Detail | Paper | Reference code | Ours | Match |
|--------|-------|---------------|------|-------|
| Depth | 8 | default 16 in init, but configurable | 8 | Paper |
| Frequency indexing | 1-indexed | 0-indexed | 0-indexed | Ref code |
| Coordinate order | x (width) first | h (height) first | h first | Ref code |

We follow the reference code for the actual computation (what ran in their
experiments) and the paper for the depth hyperparameter.

**Status: MATCH (reference code)**

### 2.2 Keys and Values

**Paper (Section 4):**
> "It uses two additional conv layers, which produce a matrix of keys,
> K in R^{h x w x m x c_m}, and values V in R^{h x w x m x c_m}, where
> m is the number of attention heads, and c_m is the number of features
> per head."

**Our code** (`ja_actor_critic.py:128-140`):
Two 1x1 Conv layers producing K and V, each with `m * c_m` output filters,
reshaped to (batch, H*W, m, c_m).

| Detail | Paper | Ours | Match |
|--------|-------|------|-------|
| K projection | conv layer | Conv(1x1) | Y |
| V projection | conv layer | Conv(1x1) | Y |
| Input | F concat S | features_with_pos | Y |
| Heads (m) | 4 | 4 | Y |
| Head features (c_m) | 16 | 16 | Y |

**Status: MATCH**

### 2.3 Query

**Paper (Section 4, Eq 3-4):**
> "the query vectors for timestep t are computed using a feed-forward
> network f_Q, parameterized by θ_Q, applied to the LSTM state at
> timestep t − 1:
> Q_t = f_Q(h_{t-1}; θ_Q)"

The paper describes the single-agent case where Q depends on the agent's own
previous LSTM state h_{t-1}. For the multi-agent case, the cross-agent signal
comes from the reference code's `LSTMStateWrapper`, which injects partner LSTM
state(s) into the observation under the `policy_state` key.

**Reference code** (`utils.py:39-55`, `attention_networks.py:148-149`):
`policy_state` = tuple of (h_all_agents, c_all_agents). These are concatenated
via `Concatenate(axis=-1)` and fed to `Dense(conv_filters)` to produce Q. So
the query sees both h and c from all agents.

**Our code** (`ja_actor_critic.py:142-150`):
`query_input = concat(lstm_h, partner_hstate)` where `partner_hstate` is the
partner's h only. Then `Dense(m * c_m)`.

| Detail | Paper | Reference code | Ours |
|--------|-------|---------------|------|
| Ego input | h_{t-1} | concat(h, c) all agents | h only |
| Partner input | not specified | included via policy_state | h only |
| Output dim | m * c_m | conv_filters (=64) | m * c_m (=64) |

The paper says Q = f_Q(h_{t-1}) for the single-agent case. The paper uses
"h_t" to denote "the hidden cell contents of the LSTM" which is ambiguous —
it could mean just h or the full (h, c). The reference code resolves this by
passing both.

We chose to follow the paper's notation (h only) since the paper is the primary
source. The reference code's choice to include c is an implementation decision
not mandated by the paper.

**Status: FOLLOWS PAPER** (differs from reference code)

### 2.4 Attention computation

**Paper (Section 4, Eq 1-2):**
> $ã^i_{x,y} = Σ_j q^i_j K^i_{x,y,j}$   (dot product, no scaling)
> $a^i_{x,y}$ = softmax over spatial locations

**Reference code** (`attention_networks.py:158-159`):
`tf.reduce_sum(q_heads * k_heads, axis=-1)` — no sqrt(d) scaling.

**Our code** (`ja_actor_critic.py:154-155`):
`einsum("bnmc,bmc->bnm", keys, queries)` then softmax over axis=1 (spatial).

| Detail | Paper | Reference code | Ours | Match |
|--------|-------|---------------|------|-------|
| Dot product | Y | Y | Y | Y |
| sqrt(d) scaling | No | No | No | Y |
| Softmax axis | spatial (x,y) | spatial (axis=1) | spatial (axis=1) | Y |

**Status: MATCH**

### 2.5 Attended output

**Paper (Section 4):**
> o^j = Σ_{x,y} a^k_{x,y} v_{x,y,j}

**Our code** (`ja_actor_critic.py:157-158`):
`einsum("bnm,bnmc->bmc", attn_weights, values)` → reshape to (batch, m*cm).

**Status: MATCH**

### 2.6 Attention map for JA incentive

**Paper (Section 5):**
> "we produce a single h x w attention map for each agent k, by taking the
> mean over each of the m attention heads."

**Our code** (`ja_actor_critic.py:161`):
`attn_map = attn_weights.mean(axis=-1).reshape(batch, h, w)`

**Status: MATCH**

---

## 3. Scalar Features

### 3.1 Direction embedding

**Paper (Appendix A):**
> "The scalar inputs are processed with a single fully connected layer of
> size 5."

**Reference code** (`attention_networks.py:243-245`):
`Sequential([one_hot_layer(scalar_dim=4), Dense(scalar_fc=5)])` — no activation.

**Our code** (`ja_actor_critic.py:164-171`):
Direction channels (ego 4-dim + partner 4-dim = 8-dim) → Dense(5), no activation.

| Detail | Paper | Reference code | Ours |
|--------|-------|---------------|------|
| Input | direction (4-dim one-hot) | direction (4-dim one-hot) | ego+partner dir (8-dim) |
| FC size | 5 | 5 | 5 |
| Activation | not mentioned | none | none |

We include partner direction in the scalar input. The reference code only has
ego direction because in MultiGrid each agent has a single "direction" scalar.
In Overcooked, both agents' directions are encoded in the observation, so we
include both.

**Status: MATCH** (activation matches paper and ref code; input adapted for Overcooked)

### 3.2 Position embedding

**Paper (Appendix A):**
> "The scalar inputs are processed with a single fully connected layer of
> size 5."

**Reference code** (`attention_networks.py:246-248`):
`Sequential([cast_and_scale(), Dense(scalar_fc=5)])` — no activation.

**Our code** (`ja_actor_critic.py:175-193`):
Ego + partner positions extracted as (x, y) coordinates → Dense(5), no activation.
Both agents' positions are included, mirroring how `dir_embed` receives both
agents' directions.

| Detail | Paper | Reference code | Ours | Match |
|--------|-------|---------------|------|-------|
| Input | position | position | ego + partner (x, y) | Adapted |
| FC size | 5 | 5 | 5 | Y |
| Activation | not mentioned | none | none | Y |

**Status: MATCH** (input adapted for Overcooked's two-agent position encoding)

---

## 4. Recurrent Policy

### 4.1 LSTM

**Paper (Section 4, Appendix A):**
> "We parameterize the policy using a recurrent Long Short Term Memory
> (LSTM) recurrent neural network"
> "an LSTM with cell size 64"

> "Let h_t = f_L(O_t, p_t, h_{t-1}; θ_L)"

The LSTM takes: attended output O, scalar features p (position, direction),
and previous state h_{t-1}.

**Our code** (`ja_actor_critic.py:187-193`):
`lstm_input = concat(attended_flat, dir_embed, pos_embed)` → OptimizedLSTMCell(64).

| Detail | Paper | Ours | Match |
|--------|-------|------|-------|
| Type | LSTM | OptimizedLSTMCell | Y |
| Cell size | 64 | 64 | Y |
| Input | O, p (pos+dir) | attended + dir_embed + pos_embed | Y |

**Status: MATCH**

### 4.2 FC heads after LSTM

**Paper (Appendix A):**
> "an LSTM with cell size 64, followed by two fully connected layers with
> hidden size 64"

**Reference code** (`attention_networks.py:185-186`):
`input_fc_layer_params=(200, 100)` — FC layers BEFORE the LSTM, not after.

**Our code** (`ja_actor_critic.py:261-276`):
LSTM → Dense(64) → ReLU → Dense(64) → ReLU → Dense(action_dim).

| Detail | Paper | Reference code | Ours |
|--------|-------|---------------|------|
| FC position | after LSTM | before LSTM | after LSTM |
| FC sizes | 64, 64 | 200, 100 | 64, 64 |
| Activation | not specified | relu (default) | relu |

We follow the paper. The reference code places FC layers before the LSTM with
different sizes (200, 100), which contradicts the paper's Appendix A.

**Status: FOLLOWS PAPER** (differs from reference code)

---

## 5. Actor-Critic Structure

**Paper (Appendix A):**
> "The value network is identical to the policy network and shares no weights."

**Our code** (`ja_actor_critic.py:204-301`):
`JAActorCritic` has two `JAScannedLSTM` instances (actor_lstm, critic_lstm)
with identical architecture but separate parameters. Actor outputs logits,
critic outputs scalar value.

| Detail | Paper | Ours | Match |
|--------|-------|------|-------|
| Separate actor/critic | Y | Y | Y |
| Identical architecture | Y | Y | Y |
| No shared weights | Y | Y | Y |

**Status: MATCH**

---

## 6. Joint Attention Intrinsic Reward

### 6.1 JSD computation

**Paper (Section 5, Eq 5-8):**
> r^JA = -Σ_j Σ_k JSD(A^k || A^j)
> where JSD uses KL divergence and M = 0.5 * (A^j + A^k)

For two agents this simplifies to r^JA = -JSD(A^0, A^1).

**Our code** (`ja_utils.py:16-30`, `ja_ippo.py:150-157`):
`jsd_divergence(p, q)` computes 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = 0.5*(p+q).
Intrinsic reward: `r_ja = -jsd_divergence(attn_0, attn_1)`.

| Detail | Paper | Ours | Match |
|--------|-------|------|-------|
| Metric | JSD | JSD | Y |
| Sign | negative (reward for similarity) | negative | Y |
| Symmetric | Y | Y | Y |
| Epsilon | not specified | 1e-8 in log | Defensive, fine |

**Status: MATCH**

### 6.2 Beta curriculum

**Paper (Appendix A):**
> "During training we anneal the weight of the bonus reward β linearly from
> 0 to 10^{-2} over the first 200000 steps."

**Paper (Section 5, Eq 9):**
> J(π^k) = E[Σ γ^t (r^k_{t+1} + β r^JA_t)]

**Our code** (`ja_ippo.py:55-60, 97-100`):
Config: `JA_BETA_MAX: 0.01`, `JA_WARMUP_ENV_STEPS: 200000`.
Linear ramp: `beta = min(beta_max, beta_max * update_steps / warmup_updates)`.
Warmup converted from env steps to update steps internally.

| Detail | Paper | Ours | Match |
|--------|-------|------|-------|
| β_max | 10^{-2} | 0.01 | Y |
| Warmup | 200,000 env steps | 200,000 env steps | Y |
| Schedule | linear 0 → β_max | linear 0 → β_max | Y |
| Reward form | r_env + β * r_JA | env_reward + ja_beta * r_ja | Y |

**Status: MATCH**

---

## 7. Training Algorithm

### 7.1 PPO

**Paper (Section 4):**
> "We train the policy using Proximal Policy Gradients (PPO)"

**Paper (Appendix A):**
> "We optimize the policy and value networks using Adam with a learning rate
> of 10^{-4}."

**Our code** (`ja_ippo.py`): Standard PPO with clipped objective, GAE,
value function clipping.

| Detail | Paper | Ours | Match |
|--------|-------|------|-------|
| Algorithm | PPO | PPO (clipped) | Y |
| Optimizer | Adam | Adam | Y |
| LR | 1e-4 | 1e-4 (cramped_room config) | Y |

PPO hyperparameters (clip_eps, ent_coef, etc.) are not specified in the paper.
We use standard values tuned per Overcooked layout.

**Status: MATCH**

### 7.2 Cross-agent state routing

**Paper:** Not explicitly described. The paper shows Q_t = f_Q(h_{t-1}) for
the single-agent case. The multi-agent cross-agent mechanism is an
implementation detail visible only in the reference code.

**Reference code** (`drivers.py:53-55`, `utils.py:36-55`):
`LSTMStateWrapper` writes partner LSTM state into observation at each step.

**Our code** (`ja_ippo.py`):
Each agent has its own hstate. Cross-agent routing is direct extraction:
`partner_h_for_0 = policy_1._extract_actor_h(hstate_1)`

This achieves the same effect as the reference code's `LSTMStateWrapper` —
each agent receives the partner's actor LSTM hidden state for the attention
query.

**Status: EQUIVALENT MECHANISM** (different implementation, same effect)

### 7.3 Loss recomputation with stored partner state

**Paper:** Not discussed.

**Our code** (`ja_ippo.py:33-42, 234-242`):
`JATransition` stores `partner_hstate` per timestep. During PPO update epochs,
the stored partner states are replayed so the network sees the same cross-agent
signal as during rollout collection.

The reference code avoids this by computing JSD from stored attention maps
(no recomputation needed). Our approach recomputes the forward pass during
PPO updates, so we need the partner state to reproduce the correct attention.

**Status: CORRECT** (different from reference but necessary for our architecture)

---

## 8. Deliberate Deviations

### 8.1 Parameter sharing

**Paper (Section 6.1):**
> "two independent PPO architectures, each with fully decentralized training"

**Reference code:** Independent per-agent networks with separate parameters.

**Our code:** Independent parameters — two TrainStates, each with its own network,
params, and optimizer. Same architecture, different random initializations.

**Status: MATCH**

### 8.2 FC layers after LSTM (not before)

We follow the paper's Appendix A ("LSTM followed by two FC layers") rather than
the reference code's architecture (FC before LSTM). See Section 4.2 above.

### 8.3 Environment

The paper evaluates on Meetup, ColorGather, StagHunt, and TaskList (all
MultiGrid). We evaluate on Overcooked-v1 layouts (cramped_room, counter_circuit,
forced_coord, asymm_advantages, coord_ring). The observation encoding,
action space, and task structure differ.

---

## 9. Reference Code Discrepancies (not matching paper)

These are cases where the reference code itself differs from the paper.
We document them here so we know what we're choosing between.

| # | Detail | Paper says | Reference code does |
|---|--------|-----------|-------------------|
| 1 | FC position | after LSTM (Appendix A) | before LSTM, sizes (200, 100) |
| 2 | FC sizes | 64, 64 | 200, 100 |
| 3 | Spatial basis depth | 8 | default 16 in constructor |
| 4 | Query input | h_{t-1} (single-agent) | concat(h, c) of all agents |
| 5 | Spatial basis frequencies | 1-indexed (paper A.1) | 0-indexed |

---

## 10. File Map

| Component | File | Lines |
|-----------|------|-------|
| Spatial attention + LSTM | `agents/ja_actor_critic.py` | JAScannedLSTM |
| Actor-Critic network | `agents/ja_actor_critic.py` | JAActorCritic |
| Policy wrapper | `agents/ja_actor_critic_agent.py` | JAActorCriticPolicy |
| JSD, spatial basis | `agents/ja_utils.py` | jsd_divergence, make_sinusoidal_spatial_basis |
| Training loop | `marl/ja_ippo.py` | make_train, JATransition |
| Base config | `marl/configs/algorithm/ja_ippo/_base_.yaml` | — |
| Task configs | `marl/configs/algorithm/ja_ippo/overcooked-v1/*.yaml` | — |

---

## Changelog

- **2026-02-23d**: Align PPO hyperparams with paper/reference code. Base config now
  uses LR 1e-4 (paper), UPDATE_EPOCHS 25, NUM_MINIBATCHES 1, CLIP_EPS 0.2,
  ENT_COEF 0.0 (all from reference code defaults). Per-layout configs simplified
  to TOTAL_TIMESTEPS and NUM_ENVS only. See section 7.1.
- **2026-02-23c**: Refactor `ja_ippo.py` to independent parameters per agent. Two
  TrainStates with separate networks, params, and optimizers. No batchify/unbatchify —
  two forward passes per env step. Cross-agent routing via direct `_extract_actor_h`.
  Checkpoints store nested `{"agent_0": ..., "agent_1": ...}`. Added Section 0 overview.
  See sections 7.2, 8.1.
- **2026-02-23b**: Fix partner visibility bug. Channels 0-1 (agent positions) now
  included in conv input (16→18 channels). Partner position extracted as (x, y) and
  added to `pos_embed` (Dense input 2→4). See sections 1.2, 1.3, 3.2.
- **2026-02-23**: Initial version. Removed ReLU from scalar embeddings (dir_embed,
  pos_embed) to match paper ("single FC layer of size 5", no activation mentioned)
  and reference code.
