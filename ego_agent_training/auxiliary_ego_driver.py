"""Shared driver for ego training with auxiliary encoder/decoder losses."""
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from common.run_episodes import run_episodes
from marl.ppo_utils import _create_minibatches


def train_auxiliary_ego_agent(
    config,
    env,
    train_rng,
    ego_policy,
    init_ego_params,
    n_ego_train_seeds,
    partner_population,
    partner_params,
    make_policy_tx: Callable,
    init_rollout_runner_state: Callable,
    env_step_fn: Callable,
    last_value_fn: Callable,
    aux_loss_fn: Callable,
    metric_from_loss_terms: Callable,
):
    """Shared PPO-style ego training loop for methods with auxiliary losses."""
    num_total_partners = partner_population.pop_size

    def make_train(config):
        num_agents = env.num_agents
        assert num_agents == 2, "This trainer assumes exactly 2 agents."

        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
        config["NUM_UNCONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_CONTROLLED_ACTORS"] = config["NUM_ENVS"]
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
        )
        config["NUM_ACTIONS"] = env.action_space(env.agents[0]).n
        assert config["NUM_CONTROLLED_ACTORS"] % config["NUM_MINIBATCHES"] == 0
        assert config["NUM_CONTROLLED_ACTORS"] >= config["NUM_MINIBATCHES"]

        def linear_schedule(count):
            frac = 1.0 - (
                count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])
            ) / config["NUM_UPDATES"]
            return config["LR"] * frac

        def train(rng):
            train_state = TrainState.create(
                apply_fn=ego_policy.policy.network.apply,
                params=init_ego_params["policy"],
                tx=make_policy_tx(config, linear_schedule),
            )

            encoder_decoder_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["ENCODER_DECODER_LR"], eps=1e-8),
            )
            encoder_decoder_train_state = TrainState.create(
                apply_fn=ego_policy.encoder.model.apply,
                params={
                    "encoder": init_ego_params["encoder"],
                    "decoder": init_ego_params["decoder"],
                },
                tx=encoder_decoder_tx,
            )

            init_ego_hstate = ego_policy.init_hstate(config["NUM_CONTROLLED_ACTORS"])
            init_partner_hstate = partner_population.init_hstate(
                config["NUM_UNCONTROLLED_ACTORS"]
            )

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            def _update_minbatch(init_state, batch_info):
                train_state, encoder_decoder_train_state = init_state
                init_ego_hstate, traj_batch, advantages, returns = batch_info

                grad_fn = jax.value_and_grad(aux_loss_fn, argnums=(0, 1), has_aux=True)
                (loss_val, aux_vals), (grads, encoder_decoder_grads) = grad_fn(
                    train_state.params,
                    encoder_decoder_train_state.params,
                    init_ego_hstate,
                    traj_batch,
                    advantages,
                    returns,
                )
                train_state = train_state.apply_gradients(grads=grads)
                encoder_decoder_train_state = encoder_decoder_train_state.apply_gradients(
                    grads=encoder_decoder_grads
                )

                grad_l2_norms = jax.tree.map(
                    lambda g: jnp.linalg.norm(g.astype(jnp.float32)),
                    grads,
                )
                avg_grad_norm = jax.tree.reduce(lambda x, y: x + y, grad_l2_norms) / len(
                    jax.tree.leaves(grad_l2_norms)
                )

                encoder_grad_l2_norms = jax.tree.map(
                    lambda g: jnp.linalg.norm(g.astype(jnp.float32)),
                    encoder_decoder_grads["encoder"],
                )
                encoder_avg_grad_norm = jax.tree.reduce(
                    lambda x, y: x + y,
                    encoder_grad_l2_norms,
                ) / len(jax.tree.leaves(encoder_grad_l2_norms))

                decoder_grad_l2_norms = jax.tree.map(
                    lambda g: jnp.linalg.norm(g.astype(jnp.float32)),
                    encoder_decoder_grads["decoder"],
                )
                decoder_avg_grad_norm = jax.tree.reduce(
                    lambda x, y: x + y,
                    decoder_grad_l2_norms,
                ) / len(jax.tree.leaves(decoder_grad_l2_norms))

                return (train_state, encoder_decoder_train_state), (
                    loss_val,
                    aux_vals,
                    avg_grad_norm,
                    encoder_avg_grad_norm,
                    decoder_avg_grad_norm,
                )

            def _update_epoch(update_state, unused):
                (
                    train_state,
                    encoder_decoder_train_state,
                    init_ego_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, perm_rng = jax.random.split(rng)
                minibatches = _create_minibatches(
                    traj_batch,
                    advantages,
                    targets,
                    init_ego_hstate,
                    config["NUM_CONTROLLED_ACTORS"],
                    config["NUM_MINIBATCHES"],
                    perm_rng,
                )
                (train_state, encoder_decoder_train_state), losses_and_grads = jax.lax.scan(
                    _update_minbatch,
                    (train_state, encoder_decoder_train_state),
                    minibatches,
                )
                update_state = (
                    train_state,
                    encoder_decoder_train_state,
                    init_ego_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses_and_grads

            def _update_step(update_runner_state, unused):
                train_state, encoder_decoder_train_state, rng, update_steps = update_runner_state

                rng, reset_rng, partner_rng = jax.random.split(rng, 3)
                reset_rngs = jax.random.split(reset_rng, config["NUM_ENVS"])
                init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rngs)
                init_done = {
                    k: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                    for k in env.agents + ["__all__"]
                }
                partner_indices = partner_population.sample_agent_indices(
                    config["NUM_UNCONTROLLED_ACTORS"],
                    partner_rng,
                )

                runner_state = init_rollout_runner_state(
                    train_state=train_state,
                    encoder_decoder_train_state=encoder_decoder_train_state,
                    init_env_state=init_env_state,
                    init_obs=init_obs,
                    init_done=init_done,
                    init_ego_hstate=init_ego_hstate,
                    init_partner_hstate=init_partner_hstate,
                    partner_indices=partner_indices,
                    rng=rng,
                )
                runner_state, traj_batch = jax.lax.scan(
                    env_step_fn,
                    runner_state,
                    None,
                    config["ROLLOUT_LENGTH"],
                )
                last_val = last_value_fn(
                    train_state=train_state,
                    encoder_decoder_train_state=encoder_decoder_train_state,
                    runner_state=runner_state,
                )
                advantages, targets = _calculate_gae(traj_batch, last_val)

                update_state = (
                    train_state,
                    encoder_decoder_train_state,
                    init_ego_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                update_state, losses_and_grads = jax.lax.scan(
                    _update_epoch,
                    update_state,
                    None,
                    config["UPDATE_EPOCHS"],
                )
                train_state = update_state[0]
                encoder_decoder_train_state = update_state[1]
                _, loss_terms, avg_grad_norm, encoder_avg_grad_norm, decoder_avg_grad_norm = (
                    losses_and_grads
                )

                metric = traj_batch.info
                metric["update_steps"] = update_steps
                metric.update(metric_from_loss_terms(loss_terms))
                metric["avg_grad_norm"] = avg_grad_norm
                metric["encoder_avg_grad_norm"] = encoder_avg_grad_norm
                metric["decoder_avg_grad_norm"] = decoder_avg_grad_norm
                new_runner_state = (
                    train_state,
                    encoder_decoder_train_state,
                    rng,
                    update_steps + 1,
                )
                return new_runner_state, metric

            ckpt_and_eval_interval = config["NUM_UPDATES"] // max(
                1,
                config["NUM_CHECKPOINTS"] - 1,
            )
            num_ckpts = config["NUM_CHECKPOINTS"]

            def init_ckpt_array(params_pytree):
                return jax.tree.map(
                    lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                    params_pytree,
                )

            max_episode_steps = config["ROLLOUT_LENGTH"]

            def _update_step_with_ckpt(state_with_ckpt, unused):
                update_state, checkpoint_array, ckpt_idx, init_eval_last_info = state_with_ckpt
                new_update_state, metric = _update_step(update_state, None)
                train_state, encoder_decoder_train_state, rng, update_steps = new_update_state

                to_store = jnp.logical_or(
                    jnp.equal(jnp.mod(update_steps - 1, ckpt_and_eval_interval), 0),
                    jnp.equal(update_steps, config["NUM_UPDATES"]),
                )

                def store_and_eval_ckpt(args):
                    ckpt_arr, cidx, rng, prev_eval_ret_info = args
                    new_ckpt_arr = jax.tree.map(
                        lambda c_arr, p: c_arr.at[cidx].set(p),
                        ckpt_arr,
                        {
                            "encoder": encoder_decoder_train_state.params["encoder"],
                            "decoder": encoder_decoder_train_state.params["decoder"],
                            "policy": train_state.params,
                        },
                    )

                    eval_partner_indices = jnp.arange(num_total_partners)
                    gathered_params = partner_population.gather_agent_params(
                        partner_params,
                        eval_partner_indices,
                    )
                    rng, eval_rng = jax.random.split(rng)
                    eval_eps_last_infos = jax.vmap(
                        lambda x: run_episodes(
                            eval_rng,
                            env,
                            agent_0_param={
                                "encoder": encoder_decoder_train_state.params["encoder"],
                                "decoder": encoder_decoder_train_state.params["decoder"],
                                "policy": train_state.params,
                            },
                            agent_0_policy=ego_policy,
                            agent_1_param=x,
                            agent_1_policy=partner_population.policy_cls,
                            max_episode_steps=max_episode_steps,
                            num_eps=config["NUM_EVAL_EPISODES"],
                        )
                    )(gathered_params)
                    return new_ckpt_arr, cidx + 1, rng, eval_eps_last_infos

                checkpoint_array, ckpt_idx, rng, eval_last_infos = jax.lax.cond(
                    to_store,
                    store_and_eval_ckpt,
                    lambda args: args,
                    (checkpoint_array, ckpt_idx, rng, init_eval_last_info),
                )
                metric["eval_ep_last_info"] = eval_last_infos
                return (
                    (train_state, encoder_decoder_train_state, rng, update_steps),
                    checkpoint_array,
                    ckpt_idx,
                    eval_last_infos,
                ), metric

            checkpoint_array = init_ckpt_array(
                {
                    "encoder": encoder_decoder_train_state.params["encoder"],
                    "decoder": encoder_decoder_train_state.params["decoder"],
                    "policy": train_state.params,
                }
            )
            ckpt_idx = 0

            rng, rng_eval, rng_train = jax.random.split(rng, 3)
            eval_partner_indices = jnp.arange(num_total_partners)
            gathered_params = partner_population.gather_agent_params(
                partner_params,
                eval_partner_indices,
            )
            eval_eps_last_infos = jax.vmap(
                lambda x: run_episodes(
                    rng_eval,
                    env,
                    agent_0_param={
                        "encoder": encoder_decoder_train_state.params["encoder"],
                        "decoder": encoder_decoder_train_state.params["decoder"],
                        "policy": train_state.params,
                    },
                    agent_0_policy=ego_policy,
                    agent_1_param=x,
                    agent_1_policy=partner_population.policy_cls,
                    max_episode_steps=max_episode_steps,
                    num_eps=config["NUM_EVAL_EPISODES"],
                )
            )(gathered_params)

            update_runner_state = (train_state, encoder_decoder_train_state, rng_train, 0)
            state_with_ckpt = (
                update_runner_state,
                checkpoint_array,
                ckpt_idx,
                eval_eps_last_infos,
            )
            state_with_ckpt, metrics = jax.lax.scan(
                _update_step_with_ckpt,
                state_with_ckpt,
                xs=None,
                length=config["NUM_UPDATES"],
            )
            final_runner_state, checkpoint_array, final_ckpt_idx, eval_eps_last_infos = (
                state_with_ckpt
            )
            return {
                "final_params": {
                    "encoder": final_runner_state[1].params["encoder"],
                    "decoder": final_runner_state[1].params["decoder"],
                    "policy": final_runner_state[0].params,
                },
                "metrics": metrics,
                "checkpoints": checkpoint_array,
            }

        return train

    rngs = jax.random.split(train_rng, n_ego_train_seeds)
    train_fn = jax.jit(jax.vmap(make_train(config)))
    return train_fn(rngs)
