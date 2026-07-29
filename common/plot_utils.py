import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

def get_metric_names(env_name):
    if env_name == "lbf":
        return ("percent_eaten", "returned_episode_returns")
    else:
        return ("returned_episode_returns",)

@partial(jax.jit, static_argnames=['stats'])
def get_stats(metrics, stats: tuple):
    '''
    Computes mean and std of metrics of interest for each seed and update, 
    using only the final steps of episodes. Note that each rollout contains multiple episodes.

    metrics is a pytree where each leaf has shape 
        (..., rollout_length, num_envs)
    stats is a tuple of strings, each corresponding to a metric of interest in metrics
    '''
    # Get mask for final steps of episodes
    mask = metrics["returned_episode"]
    
    # Initialize output dictionary
    all_stats = {}
    stats = list(stats) # convert to list to correctly iterate if the tuple only has a single element
    for stat_name in stats:
        # Get the metric array
        metric_data = metrics[stat_name]  # Shape: (..., rollout_length, num_envs)

        # Compute means and stds for each seed and update
        # Use masked operations to only consider final episode steps
        means = jnp.where(mask, metric_data, 0).sum(axis=(-2, -1)) / mask.sum(axis=(-2, -1))
        # For std, first compute masked values
        masked_vals = jnp.where(mask, metric_data, 0)
        squared_diff = (masked_vals - means[..., None, None]) ** 2
        variance = jnp.where(mask, squared_diff, 0).sum(axis=(-2, -1)) / mask.sum(axis=(-2, -1))
        stds = jnp.sqrt(variance)
        # Stack means and stds
        all_stats[stat_name] = jnp.stack([means, stds], axis=-1)
    
    return all_stats




def plot_seed_aggregate(all_stats, num_rollout_steps, num_envs,
                       savedir=None, savename=None):
    """Plot mean ± std across seeds for each metric.

    Args:
        all_stats: dict of {metric_name: array of shape (num_seeds, num_updates, 2)}.
            The [..., 0] slice is the per-seed mean (across envs/episodes within a rollout),
            which we aggregate across seeds to get the cross-seed mean ± std.
        num_rollout_steps: rollout length (for x-axis scaling).
        num_envs: number of parallel envs (for x-axis scaling).
        savedir: directory to save figures.
        savename: base filename for saved figures.
    """
    figures = {}
    for stat_name, stats in all_stats.items():
        stats = np.array(stats)
        num_seeds, num_updates, _ = stats.shape
        xs = np.arange(num_updates) * num_envs * num_rollout_steps

        # Aggregate across seeds on the per-seed means
        seed_means = stats[:, :, 0]  # (num_seeds, num_updates)
        mean = seed_means.mean(axis=0)
        std = seed_means.std(axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(xs, mean, linewidth=1.5)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.3)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(stat_name.replace("_", " ").title())
        ax.set_title(f"{stat_name.replace('_', ' ').title()} ({num_seeds} seeds)")
        fig.tight_layout()

        savepath = None
        if savedir is not None and savename is not None:
            safe_name = stat_name.replace(" ", "_")
            savepath = os.path.join(savedir, f"{savename}_{safe_name}.png")
            fig.savefig(savepath)
        figures[stat_name] = fig
        plt.close(fig)

    return figures



        
