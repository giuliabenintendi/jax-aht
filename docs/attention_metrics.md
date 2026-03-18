# Attention Metrics

Two post-hoc metrics evaluate the quality of learned spatial attention in JA-IPPO. Both are computed from greedy rollouts of trained checkpoints (64 episodes per seed, 5 seeds).

---

## 1. Attention Stasis (Temporal Consistency)

**What it measures:** How much an agent's attention distribution changes between consecutive timesteps. A well-trained agent should shift attention dynamically as the game state evolves. An agent whose attention is frozen on a fixed region (e.g., always staring at a corner) has low stasis, suggesting the attention mechanism is not tracking task-relevant changes.

**Formula.** At each timestep $t$, the attention map $A_t$ is a probability distribution over feature-map cells (shape $H \times W$, sums to 1 via softmax). Flatten $A_t$ to a 1-D vector $p_t$.

The Jensen-Shannon Divergence between consecutive maps is:

$$\mathrm{JSD}(p_t, p_{t+1}) = \frac{1}{2} \mathrm{KL}(p_t \| m) + \frac{1}{2} \mathrm{KL}(p_{t+1} \| m), \quad m = \frac{p_t + p_{t+1}}{2}$$

where $\mathrm{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i}$ (with $\varepsilon = 10^{-8}$ for numerical stability).

**Per-episode stasis** is the mean over all consecutive pairs in the episode:

$$S_{\text{ep}} = \frac{1}{T-1} \sum_{t=1}^{T-1} \mathrm{JSD}(p_t, p_{t+1})$$

**Aggregation:**

1. Per-episode mean $S_{\text{ep}}$ (as above)
2. Per-seed mean: average $S_{\text{ep}}$ over 64 episodes
3. Cross-seed mean $\pm$ SEM: average the per-seed means over 5 seeds; SEM $= \sigma / \sqrt{N_{\text{seeds}}}$

**Range and interpretation:**

- $S = 0$: perfectly static attention (identical map every timestep)
- Higher $S$: more dynamic attention that responds to state changes
- JSD is bounded in $[0, \log 2]$, so stasis falls in the same range

---

## 2. Object Coverage (% Attention on Objects)

**What it measures:** The fraction of total attention mass that lands on task-relevant objects (as opposed to walls, floors, and empty space). High coverage means the agent attends to things that matter for the task; low coverage means attention is wasted on background.

**Feature-map to game-grid remapping.** The ResNet encoder downsamples the image observation (rendered at 7 pixels per tile) through stride-2 convolutions, producing a feature map smaller than the game grid. Each feature-map cell covers a rectangular region of the pixel image, potentially spanning parts of multiple game tiles. The coverage computation accounts for this with **fractional overlap**: for each feature-map cell, it computes the pixel-area intersection with every overlapping game tile and assigns proportional weight to each tile's category. The result is a tensor of shape $(H, W, C)$ where $C = 12$ categories and each cell sums to 1.

**Categories.** Every game tile is classified into one of 12 categories:

| | Objects | Non-objects |
|---|---|---|
| Categories | onion, onion_disp, plate, plate_disp, serve, pot, dish, agent | floor, wall, counter, unseen |

Counters are distinguished from walls via adjacency: a wall-type tile that borders at least one walkable (floor) tile is classified as a counter.

**Per-timestep coverage.** Given attention map $A_t$ (shape $H \times W$) and coverage tensor $G_t$ (shape $H \times W \times C$):

$$\text{obj\_mass}_t = \sum_{i,j} A_t[i,j] \cdot \sum_{c \in \text{objects}} G_t[i,j,c]$$

This is the total attention weight on feature cells, weighted by the fraction of each cell that covers object tiles. It ranges from 0 (no attention on any object) to 1 (all attention on pure-object cells).

**Aggregation:** Same four-level pipeline as stasis:

1. Per-timestep $\text{obj\_mass}_t$
2. Per-episode mean over all timesteps
3. Per-seed mean over 64 episodes
4. Cross-seed mean $\pm$ SEM over 5 seeds

**Supplementary output:** The per-category attention mass (averaged across episodes and seeds) is also reported, giving a breakdown of how much attention goes to each of the 12 tile types individually.

---

## Summary Table

| Metric | Measures | Range | Low means | High means |
|---|---|---|---|---|
| Attention Stasis | Temporal change in attention | $[0, \log 2]$ | Static / fixated | Dynamic / tracking |
| Object Coverage | Attention on task-relevant tiles | $[0, 1]$ | Attending to background | Attending to objects |
