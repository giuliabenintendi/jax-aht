"""Run greedy and stochastic eval episodes and plot return curves.

Plots per-episode returns as a line chart (like training curves) with
running mean and overall mean ± std band.

Usage:
    uv run python -m evaluation.eval_greedy \
        --checkpoint <path_to_saved_train_run> \
        --num-episodes 256 \
        --output-dir plots/
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from agents.initialize_agents import (
    initialize_ja_image_agent, initialize_ja_dual_image_agent,
)
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper


def _get_obs_type(alg_config):
    return alg_config.get("OBS_TYPE", alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))


def run_eval_with_rewards(inner_env, policy, params, num_episodes, max_steps, greedy, seed_offset=0):
    """Run N eval episodes, return per-episode total rewards."""
    returns = []
    for ep in range(num_episodes):
        rng = jax.random.PRNGKey(1000 + seed_offset * 10000 + ep)
        rng, reset_rng = jax.random.split(rng)

        obs, env_state = inner_env.reset(reset_rng)
        done = {k: jnp.zeros((1,), dtype=bool) for k in inner_env.agents + ["__all__"]}

        hstate_0 = policy.init_hstate(1)
        hstate_1 = policy.init_hstate(1)

        total_reward = 0.0
        step = 0
        while not done["__all__"] and step < max_steps:
            avail_actions = inner_env.get_avail_actions(env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)

            obs_0 = obs["agent_0"].reshape(1, 1, -1)
            obs_1 = obs["agent_1"].reshape(1, 1, -1)
            done_0 = done["agent_0"].reshape(1, 1)
            done_1 = done["agent_1"].reshape(1, 1)
            avail_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_1 = avail_actions["agent_1"].astype(jnp.float32)

            rng, act_rng0, act_rng1, step_rng = jax.random.split(rng, 4)

            act_0, hstate_0 = policy.get_action(
                params=params, obs=obs_0, done=done_0,
                avail_actions=avail_0, hstate=hstate_0,
                rng=act_rng0, greedy=greedy,
            )
            act_1, hstate_1 = policy.get_action(
                params=params, obs=obs_1, done=done_1,
                avail_actions=avail_1, hstate=hstate_1,
                rng=act_rng1, greedy=greedy,
            )

            env_act = {"agent_0": act_0.squeeze(), "agent_1": act_1.squeeze()}
            obs, env_state, reward, done, info = inner_env.step(step_rng, env_state, env_act)

            total_reward += float(reward["agent_0"])
            step += 1

        returns.append(total_reward)

        if (ep + 1) % 64 == 0:
            print(f"  {'greedy' if greedy else 'stochastic'} ep {ep+1}/{num_episodes}: "
                  f"mean={np.mean(returns):.1f}")

    return np.array(returns)


def running_mean(arr, window=20):
    """Compute running mean with given window size."""
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    rm = np.empty_like(arr, dtype=float)
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        rm[i] = (cumsum[i + 1] - cumsum[lo]) / (i - lo + 1)
    return rm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", default=None, help="Wandb run ID to resume and upload results")
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--project", default="aht-benchmark")
    parser.add_argument("--entity", default="g-benintendi-university-of-brescia")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    run_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    alg_config = cfg["algorithm"]

    env_name = alg_config["ENV_NAME"]
    env = make_env(env_name, alg_config["ENV_KWARGS"])
    inner_env = env

    obs_type = _get_obs_type(alg_config)
    use_dual = alg_config.get("USE_DUAL_CRITIC", False)
    if obs_type in ("image", "fov"):
        init_fn = initialize_ja_dual_image_agent if use_dual else initialize_ja_image_agent
    else:
        from agents.initialize_agents import initialize_ja_agent
        init_fn = initialize_ja_agent

    env_wrapped = LogWrapper(env)
    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env_wrapped, rng)

    run_data = load_train_run(args.checkpoint)
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 400))

    layout = cfg.get("TASK_NAME", env_name)
    beta = alg_config.get("JA_BETA_MAX", 0)
    ent = alg_config.get("ENT_COEF", 0.01)
    jsd_gae = alg_config.get("DUAL_CRITIC_ACTOR_JA", False)

    print(f"Layout: {layout}, Beta: {beta}, ENT_COEF: {ent}, Dual: {use_dual}, "
          f"JSD GAE: {jsd_gae}, Seeds: {num_seeds}")
    print(f"Running {args.num_episodes} episodes per seed, greedy + stochastic")

    all_greedy = []
    all_stochastic = []

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)
        print(f"\nSeed {seed_idx}:")

        greedy_returns = run_eval_with_rewards(
            inner_env, policy, params, args.num_episodes, max_steps,
            greedy=True, seed_offset=seed_idx)
        stochastic_returns = run_eval_with_rewards(
            inner_env, policy, params, args.num_episodes, max_steps,
            greedy=False, seed_offset=seed_idx)

        all_greedy.append(greedy_returns)
        all_stochastic.append(stochastic_returns)

        print(f"  Greedy:     mean={greedy_returns.mean():.1f} ± {greedy_returns.std():.1f}")
        print(f"  Stochastic: mean={stochastic_returns.mean():.1f} ± {stochastic_returns.std():.1f}")

    # Aggregate across seeds
    all_greedy = np.concatenate(all_greedy)
    all_stochastic = np.concatenate(all_stochastic)

    print(f"\nOverall ({num_seeds} seeds x {args.num_episodes} episodes):")
    print(f"  Greedy:     mean={all_greedy.mean():.1f} ± {all_greedy.std():.1f}")
    print(f"  Stochastic: mean={all_stochastic.mean():.1f} ± {all_stochastic.std():.1f}")

    # Plot: line chart with per-episode returns + running mean + mean±std band
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    episodes = np.arange(1, len(all_stochastic) + 1)

    for ax, data, title, color in [
        (ax1, all_stochastic, "Stochastic", "C1"),
        (ax2, all_greedy, "Greedy", "C0"),
    ]:
        mean = data.mean()
        std = data.std()
        rm = running_mean(data, window=20)

        # Per-episode returns as faint dots
        ax.scatter(episodes, data, s=3, alpha=0.2, color=color, zorder=1)
        # Running mean line
        ax.plot(episodes, rm, color=color, linewidth=1.5, zorder=2, label="Running mean")
        # Overall mean ± std band
        ax.axhline(mean, color=color, linestyle="--", linewidth=1.5, zorder=3)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.1, zorder=0)
        ax.set_xlabel("Episode")
        ax.set_title(title)
        ax.legend([f"Running mean (w=20)",
                   f"Mean={mean:.1f} ± {std:.1f}"],
                  fontsize=8, loc="lower right")

    ax1.set_ylabel("Episode Return")

    jsd_label = "jsdgae" if jsd_gae else "nojsdgae"
    fig.suptitle(f"{layout} | β={beta} | ent={ent} | dual_{jsd_label} | "
                 f"{num_seeds}s x {args.num_episodes}ep",
                 fontsize=11)
    fig.tight_layout()

    slug = layout.replace("/", "_").replace(" ", "_")
    path = os.path.join(args.output_dir, f"eval_greedy_{slug}_b{beta}_ent{ent}.png")
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")

    # Upload to existing wandb run if --run-id provided
    if args.run_id:
        import wandb
        wb_run = wandb.init(
            project=args.project, entity=args.entity,
            id=args.run_id, resume="must",
        )
        wb_run.log({"Eval/greedy_vs_stochastic": wandb.Image(path)}, commit=False)
        wb_run.summary["Eval/greedy_return_mean"] = float(all_greedy.mean())
        wb_run.summary["Eval/greedy_return_std"] = float(all_greedy.std())
        wb_run.summary["Eval/stochastic_return_mean"] = float(all_stochastic.mean())
        wb_run.summary["Eval/stochastic_return_std"] = float(all_stochastic.std())
        wb_run.log({}, commit=True)
        wb_run.finish()
        print(f"Uploaded to wandb run {args.run_id}")


if __name__ == "__main__":
    main()
