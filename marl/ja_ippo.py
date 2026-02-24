'''
JA-IPPO: Joint Attention IPPO with independent parameters (Lee et al. 2021).

Each agent has its own network, parameters, and optimizer — matching the paper's
"two independent PPO architectures" (Section 6.1).  Cross-agent coordination
comes from:

  1. Cross-agent state routing: each agent receives its partner's actor LSTM
     hidden state h.
  2. JA intrinsic reward: r_JA = -JSD(attn_agent_0, attn_agent_1), scaled by
     a beta that ramps linearly from 0 to JA_BETA_MAX over JA_WARMUP_ENV_STEPS.
  3. JATransition stores partner_hstate per timestep so it can be replayed
     during PPO loss recomputation.
'''
import shutil
from typing import NamedTuple

import hydra
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_ja_agent
from agents.ja_utils import jsd_divergence
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_utils import _create_minibatches


class JATransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    partner_hstate: jnp.ndarray  # (NUM_ENVS, lstm_dim) — stored per timestep
    ja_reward: jnp.ndarray       # (NUM_ENVS,) — raw JA intrinsic reward (unscaled)


def make_train(config, env):
    # Each agent trains on its own batch of NUM_ENVS actors (independent params)
    config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["ROLLOUT_LENGTH"] // config["NUM_MINIBATCHES"]
    )

    num_envs = config["NUM_ENVS"]
    ja_beta_max = config.get("JA_BETA_MAX", 0.01)
    # Paper: "anneal β linearly from 0 to 1e-2 over the first 200000 steps"
    # Config is in env steps; convert to update steps internally.
    ja_warmup_env_steps = config.get("JA_WARMUP_ENV_STEPS", 200_000)
    env_steps_per_update = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
    ja_warmup_updates = ja_warmup_env_steps / env_steps_per_update

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        # INIT TWO INDEPENDENT NETWORKS (paper Section 6.1: "do not share parameters")
        rng, init_rng_0, init_rng_1 = jax.random.split(rng, 3)
        policy_0, init_params_0 = initialize_ja_agent(config, env, init_rng_0)
        policy_1, init_params_1 = initialize_ja_agent(config, env, init_rng_1)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state_0 = TrainState.create(
            apply_fn=policy_0.network.apply, params=init_params_0, tx=tx,
        )
        train_state_1 = TrainState.create(
            apply_fn=policy_1.network.apply, params=init_params_1, tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # PPO update for one agent (called twice — once per agent)
        def _ppo_update(policy, train_state, traj_batch, advantages, targets, rng):
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # Rerun network with stored partner_hstate
                        _, value, pi, _, _ = policy.get_action_value_policy(
                            params=params,
                            obs=traj_batch.obs,
                            done=traj_batch.done,
                            avail_actions=traj_batch.avail_actions,
                            hstate=init_hstate,
                            rng=jax.random.PRNGKey(0),
                            partner_hstate=traj_batch.partner_hstate,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # Policy gradient loss (clipped)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)
                # _create_minibatches works on any pytree — JATransition's extra
                # partner_hstate field is shuffled and reshaped alongside everything else
                minibatches = _create_minibatches(traj_batch, advantages, targets, init_hstate,
                                                  num_envs, config["NUM_MINIBATCHES"], perm_rng)

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            init_hstate = policy.init_hstate(num_envs)
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            return update_state[0], loss_info

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            # Beta curriculum: linear ramp from 0 to ja_beta_max
            ja_beta = jnp.minimum(
                ja_beta_max,
                ja_beta_max * update_steps / jnp.maximum(ja_warmup_updates, 1.0),
            )

            def _env_step(runner_state, unused):
                (train_state_0, train_state_1, env_state, last_obs, last_done,
                 hstate_0, hstate_1, rng) = runner_state

                rng, act_rng_0, act_rng_1 = jax.random.split(rng, 3)

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)

                # Cross-agent state routing: each agent gets its partner's actor h
                partner_h_for_0 = policy_1._extract_actor_h(hstate_1)
                partner_h_for_1 = policy_0._extract_actor_h(hstate_0)

                # Agent 0 forward pass
                obs_0 = last_obs["agent_0"].reshape(1, num_envs, -1)
                done_0 = last_done["agent_0"].reshape(1, num_envs)
                avail_0 = jax.lax.stop_gradient(
                    avail_actions["agent_0"].astype(jnp.float32).reshape(1, num_envs, -1))

                action_0, value_0, pi_0, new_hstate_0, attn_0 = policy_0.get_action_value_policy(
                    params=train_state_0.params, obs=obs_0, done=done_0,
                    avail_actions=avail_0, hstate=hstate_0, rng=act_rng_0,
                    partner_hstate=partner_h_for_0)

                # Agent 1 forward pass
                obs_1 = last_obs["agent_1"].reshape(1, num_envs, -1)
                done_1 = last_done["agent_1"].reshape(1, num_envs)
                avail_1 = jax.lax.stop_gradient(
                    avail_actions["agent_1"].astype(jnp.float32).reshape(1, num_envs, -1))

                action_1, value_1, pi_1, new_hstate_1, attn_1 = policy_1.get_action_value_policy(
                    params=train_state_1.params, obs=obs_1, done=done_1,
                    avail_actions=avail_1, hstate=hstate_1, rng=act_rng_1,
                    partner_hstate=partner_h_for_1)

                log_prob_0 = pi_0.log_prob(action_0).squeeze()
                log_prob_1 = pi_1.log_prob(action_1).squeeze()
                action_0 = action_0.squeeze()
                action_1 = action_1.squeeze()
                value_0 = value_0.squeeze()
                value_1 = value_1.squeeze()

                env_act = {"agent_0": action_0.flatten(), "agent_1": action_1.flatten()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )

                # LogWrapper info: each leaf is (NUM_ENVS, num_agents). Take agent 0's
                # column for metrics (identical across agents in cooperative tasks).
                info = jax.tree.map(lambda x: x[:, 0] if x.ndim > 1 else x, info)

                # JA intrinsic reward: -JSD between agent_0 and agent_1 attention maps
                attn_map_0 = attn_0.squeeze(0)  # (NUM_ENVS, H, W)
                attn_map_1 = attn_1.squeeze(0)  # (NUM_ENVS, H, W)
                r_ja = -jsd_divergence(attn_map_0, attn_map_1)  # (NUM_ENVS,)
                r_ja = jax.lax.stop_gradient(r_ja)

                transition_0 = JATransition(
                    done=new_done["agent_0"],
                    action=action_0,
                    value=value_0,
                    reward=reward["agent_0"] + ja_beta * r_ja,
                    log_prob=log_prob_0,
                    obs=obs_0.squeeze(0),
                    info=info,
                    avail_actions=avail_0.squeeze(0),
                    partner_hstate=partner_h_for_0.squeeze(0),
                    ja_reward=r_ja,
                )
                transition_1 = JATransition(
                    done=new_done["agent_1"],
                    action=action_1,
                    value=value_1,
                    reward=reward["agent_1"] + ja_beta * r_ja,
                    log_prob=log_prob_1,
                    obs=obs_1.squeeze(0),
                    info=info,
                    avail_actions=avail_1.squeeze(0),
                    partner_hstate=partner_h_for_1.squeeze(0),
                    ja_reward=r_ja,
                )
                runner_state = (train_state_0, train_state_1, new_env_state,
                                new_obs, new_done, new_hstate_0, new_hstate_1, rng)
                return runner_state, (transition_0, transition_1)

            runner_state, (traj_batch_0, traj_batch_1) = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            # Final value estimates
            (train_state_0, train_state_1, env_state, last_obs, last_done,
             hstate_0, hstate_1, rng) = runner_state

            last_avail = jax.vmap(env.get_avail_actions)(env_state.env_state)

            # Agent 0
            last_obs_0 = last_obs["agent_0"].reshape(1, num_envs, -1)
            last_done_0 = last_done["agent_0"].reshape(1, num_envs)
            last_avail_0 = jax.lax.stop_gradient(
                last_avail["agent_0"].astype(jnp.float32).reshape(1, num_envs, -1))
            last_partner_h_0 = policy_1._extract_actor_h(hstate_1)
            _, last_val_0, _, _, _ = policy_0.get_action_value_policy(
                params=train_state_0.params, obs=last_obs_0, done=last_done_0,
                avail_actions=last_avail_0, hstate=hstate_0,
                rng=jax.random.PRNGKey(0), partner_hstate=last_partner_h_0,
            )
            last_val_0 = last_val_0.squeeze()

            # Agent 1
            last_obs_1 = last_obs["agent_1"].reshape(1, num_envs, -1)
            last_done_1 = last_done["agent_1"].reshape(1, num_envs)
            last_avail_1 = jax.lax.stop_gradient(
                last_avail["agent_1"].astype(jnp.float32).reshape(1, num_envs, -1))
            last_partner_h_1 = policy_0._extract_actor_h(hstate_0)
            _, last_val_1, _, _, _ = policy_1.get_action_value_policy(
                params=train_state_1.params, obs=last_obs_1, done=last_done_1,
                avail_actions=last_avail_1, hstate=hstate_1,
                rng=jax.random.PRNGKey(0), partner_hstate=last_partner_h_1,
            )
            last_val_1 = last_val_1.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages_0, targets_0 = _calculate_gae(traj_batch_0, last_val_0)
            advantages_1, targets_1 = _calculate_gae(traj_batch_1, last_val_1)

            # PPO updates (independent per agent)
            rng, rng_0, rng_1 = jax.random.split(rng, 3)
            train_state_0, loss_0 = _ppo_update(
                policy_0, train_state_0, traj_batch_0, advantages_0, targets_0, rng_0)
            train_state_1, loss_1 = _ppo_update(
                policy_1, train_state_1, traj_batch_1, advantages_1, targets_1, rng_1)

            # Metrics from agent 0's trajectory (LogWrapper metrics are per-env, identical)
            metric = traj_batch_0.info
            metric["update_steps"] = update_steps
            # JA-specific scalar metrics (per update step)
            metric["ja_beta"] = ja_beta
            metric["ja_reward_mean"] = traj_batch_0.ja_reward.mean()
            metric["jsd_mean"] = -traj_batch_0.ja_reward.mean()

            update_steps += 1
            runner_state = (train_state_0, train_state_1, env_state, last_obs, last_done,
                           hstate_0, hstate_1, rng)
            return (runner_state, update_steps), metric

        ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
        num_ckpts = config["NUM_CHECKPOINTS"]

        def init_ckpt_array(params_pytree):
            return jax.tree.map(
                lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                params_pytree
            )

        def _update_step_with_checkpoint(update_with_ckpt_runner_state, unused):
            (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_runner_state
            update_runner_state, metric = _update_step(update_runner_state, None)
            _, update_steps = update_runner_state
            to_store = jnp.logical_or(
                jnp.equal(jnp.mod(update_steps - 1, ckpt_and_eval_interval), 0),
                jnp.equal(update_steps, config["NUM_UPDATES"]),
            )

            def store_ckpt_fn(args):
                _checkpoint_array, _ckpt_idx = args
                params_both = {
                    "agent_0": update_runner_state[0][0].params,
                    "agent_1": update_runner_state[0][1].params,
                }
                new_checkpoint_array = jax.tree.map(
                    lambda c_arr, p: c_arr.at[_ckpt_idx].set(p),
                    _checkpoint_array,
                    params_both,
                )
                return new_checkpoint_array, _ckpt_idx + 1

            def skip_ckpt_fn(args):
                return args

            checkpoint_array, ckpt_idx = jax.lax.cond(
                to_store, store_ckpt_fn, skip_ckpt_fn, (checkpoint_array, ckpt_idx),
            )

            runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        update_steps = 0
        hstate_0 = policy_0.init_hstate(num_envs)
        hstate_1 = policy_1.init_hstate(num_envs)
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
        runner_state = (train_state_0, train_state_1, env_state, obsv, init_done,
                       hstate_0, hstate_1, _rng)
        update_runner_state = (runner_state, update_steps)
        params_both = {"agent_0": train_state_0.params, "agent_1": train_state_1.params}
        checkpoint_array = init_ckpt_array(params_both)
        ckpt_idx = 0
        update_with_ckpt_runner_state = (update_runner_state, checkpoint_array, ckpt_idx)

        runner_state, metrics = jax.lax.scan(
            _update_step_with_checkpoint,
            update_with_ckpt_runner_state,
            xs=None,
            length=config["NUM_UPDATES"],
        )

        update_runner_state, checkpoint_array, final_ckpt_idx = runner_state

        return {
            "final_params": {
                "agent_0": update_runner_state[0][0].params,
                "agent_1": update_runner_state[0][1].params,
            },
            "metrics": metrics,
            "checkpoints": checkpoint_array,
            "final_ckpt_idx": final_ckpt_idx,
        }

    return train


def run_ja_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])

    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(algorithm_config, env)))
        out = train_jit(rngs)

    log_metrics(config, out, logger)
    return out


def log_metrics(config, out, logger):
    '''Save train run output and log to wandb as artifact.'''
    train_metrics = out["metrics"]
    metric_names = get_metric_names(config["ENV_NAME"])
    train_stats = get_stats(train_metrics, metric_names)

    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    # JA-specific scalars: shape (NUM_SEEDS, NUM_UPDATES) — average over seeds
    ja_scalars = {}
    for key in ("ja_beta", "ja_reward_mean", "jsd_mean"):
        if key in train_metrics:
            ja_scalars[key] = np.mean(np.array(train_metrics[key]), axis=0)

    num_updates = train_metrics["returned_episode"].shape[1]
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/{stat_name}", stat_mean, train_step=step, commit=False)
        for key, vals in ja_scalars.items():
            logger.log_item(f"Train/{key}", float(vals[step]), train_step=step, commit=True)

    logger.commit()

    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    out_savepath = save_train_run(out, savedir, savename="saved_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="saved_train_run", path=out_savepath, type_name="train_run")
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
