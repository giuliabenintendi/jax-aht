"""Run greedy and stochastic eval episodes and plot return distributions.

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
from evaluation.vis_episodes import run_episode_with_states


def _get_obs_type(alg_config):
    return alg_config.get("OBS_TYPE", alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"))


def run_eval_episodes(env, policy, params, num_episodes, max_steps, greedy, seed_offset=0):
    """Run N eval episodes, return per-episode total returns."""
    returns = []
    for ep in range(num_episodes):
        rng = jax.random.PRNGKey(1000 + seed_offset * 10000 + ep)
        ep_states, _ = run_episode_with_states(
            rng, env, params, policy, params, policy, max_steps,
            collect_attention=False, greedy=greedy,
        )
        # Sum rewards from state transitions
        total_return = 0.0
        for i in range(1, len(ep_states)):
            # base_return is in env_state
            pass
        # Use episode length as proxy — actually we need reward from env
        # Simpler: run with LogWrapper info
        returns.append(len(ep_states) - 1)  # placeholder

    return np.array(returns)


def run_eval_with_rewards(inner_env, env, policy, params, num_episodes, max_steps, greedy, seed_offset=0):
    """Run N eval episodes collecting total reward per episode."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--dpi", type=int, default=150)
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
    env = LogWrapper(env)

    obs_type = _get_obs_type(alg_config)
    use_dual = alg_config.get("USE_DUAL_CRITIC", False)
    if obs_type in ("image", "fov"):
        init_fn = initialize_ja_dual_image_agent if use_dual else initialize_ja_image_agent
    else:
        from agents.initialize_agents import initialize_ja_agent
        init_fn = initialize_ja_agent

    rng = jax.random.PRNGKey(0)
    policy, _ = init_fn(alg_config, env, rng)

    run_data = load_train_run(args.checkpoint)
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    max_steps = int(alg_config.get("ENV_KWARGS", {}).get("max_steps", 400))

    layout = cfg.get("TASK_NAME", env_name)
    beta = alg_config.get("JA_BETA_MAX", 0)
    ent = alg_config.get("ENT_COEF", 0.01)

    print(f"Layout: {layout}, Beta: {beta}, ENT_COEF: {ent}, Seeds: {num_seeds}")
    print(f"Running {args.num_episodes} episodes per seed, greedy + stochastic")

    all_greedy = []
    all_stochastic = []

    for seed_idx in range(num_seeds):
        params = jax.tree.map(lambda x: x[seed_idx], final_params)
        print(f"\nSeed {seed_idx}:")

        greedy_returns = run_eval_with_rewards(
            inner_env, env, policy, params, args.num_episodes, max_steps,
            greedy=True, seed_offset=seed_idx)
        stochastic_returns = run_eval_with_rewards(
            inner_env, env, policy, params, args.num_episodes, max_steps,
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

    # Plot: two histograms side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    ax1.hist(all_stochastic, bins=30, color="C1", alpha=0.7, edgecolor="white")
    ax1.axvline(all_stochastic.mean(), color="C1", linestyle="--", linewidth=2,
                label=f"mean={all_stochastic.mean():.1f} ± {all_stochastic.std():.1f}")
    ax1.set_xlabel("Episode Return")
    ax1.set_ylabel("Count")
    ax1.set_title("Stochastic")
    ax1.legend(fontsize=9)

    ax2.hist(all_greedy, bins=30, color="C0", alpha=0.7, edgecolor="white")
    ax2.axvline(all_greedy.mean(), color="C0", linestyle="--", linewidth=2,
                label=f"mean={all_greedy.mean():.1f} ± {all_greedy.std():.1f}")
    ax2.set_xlabel("Episode Return")
    ax2.set_title("Greedy")
    ax2.legend(fontsize=9)

    fig.suptitle(f"{layout} | β={beta} | ENT_COEF={ent} | {num_seeds} seeds x {args.num_episodes} eps",
                 fontsize=11)
    fig.tight_layout()

    slug = layout.replace("/", "_").replace(" ", "_")
    path = os.path.join(args.output_dir, f"eval_greedy_{slug}_b{beta}_ent{ent}.png")
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
