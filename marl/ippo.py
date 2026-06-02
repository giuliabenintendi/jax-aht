'''
Based on the IPPO implementation from JaxMarl. Trains a parameter-shared, MLP IPPO agent on a
fully cooperative multi-agent environment. Note that this code is only compatible with MLP policies.
'''
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from agents.initialize_agents import initialize_s5_agent, initialize_mlp_agent, \
    initialize_rnn_agent, initialize_pseudo_actor_with_double_critic, initialize_pseudo_actor_with_conditional_critic
from common.train_logging import report_basic_training_outputs
from envs import make_env
from envs.log_wrapper import LogWrapper
from marl.ppo_core import (
    calculate_gae,
    compute_last_value,
    configure_training_dims,
    make_optimizer,
    run_ppo_epochs,
)
from marl.ppo_utils import Transition, batchify, unbatchify


def initialize_agent(actor_type, algorithm_config, env, init_rng):
    if actor_type == "s5":
        policy, init_params = initialize_s5_agent(algorithm_config, env, init_rng)
    elif actor_type == "mlp":
        policy, init_params = initialize_mlp_agent(algorithm_config, env, init_rng)
    elif actor_type == "rnn":
        policy, init_params = initialize_rnn_agent(algorithm_config, env, init_rng)
    elif actor_type == "pseudo_actor_with_double_critic":
        policy, init_params = initialize_pseudo_actor_with_double_critic(algorithm_config, env, init_rng)
    elif actor_type == "pseudo_actor_with_conditional_critic":
        policy, init_params = initialize_pseudo_actor_with_conditional_critic(algorithm_config, env, init_rng)
    return policy, init_params

def make_train(config, env):
    configure_training_dims(config, env)

    def train(rng):
        # INIT NETWORK
        rng, init_rng = jax.random.split(rng)
        policy, init_params = initialize_agent(config["ACTOR_TYPE"], config, env, init_rng)

        train_state = TrainState.create(
            apply_fn=policy.network.apply,
            params=init_params,
            tx=make_optimizer(config),
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state

                rng, act_rng = jax.random.split(rng)

                last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])

                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(batchify(avail_actions, 
                    env.agents, config["NUM_ACTORS"]).astype(jnp.float32))

                action, value, pi, new_hstate = policy.get_action_value_policy(
                    params=train_state.params,
                    obs=last_obs_batch.reshape(1, config["NUM_ACTORS"], -1),
                    done=last_done_batch.reshape(1, config["NUM_ACTORS"]),
                    avail_actions=avail_actions.reshape(1, config["NUM_ACTORS"], -1),
                    hstate=last_hstate,
                    rng=act_rng
                )
                log_prob = pi.log_prob(action)

                action = action.squeeze()
                log_prob = log_prob.squeeze()
                value = value.squeeze()

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k:v.flatten() for k,v in env_act.items()}

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                new_obs, new_env_state, reward, new_done, info = jax.vmap(env.step, in_axes=(0,0,0))(
                    rng_step, env_state, env_act
                )
                
                # note that num_actors = num_envs * num_agents
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)

                transition = Transition(
                    batchify(new_done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    last_obs_batch,
                    info,
                    avail_actions
                )
                runner_state = (train_state, new_env_state, new_obs, new_done, new_hstate, rng)
                return runner_state, transition
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
            )

            # Get final value estimate for completed trajectory
            train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_obs_batch = last_obs_batch.reshape(1, config["NUM_ACTORS"], -1)
            last_done_batch = batchify(last_done, env.agents, config["NUM_ACTORS"])
            last_done_batch = last_done_batch.reshape(1, config["NUM_ACTORS"])
            last_avail_batch = jax.vmap(env.get_avail_actions)(env_state.env_state)
            last_avail_batch = jax.lax.stop_gradient(batchify(last_avail_batch, 
                env.agents, config["NUM_ACTORS"]).astype(jnp.float32))
            
            last_val = compute_last_value(
                policy,
                train_state.params,
                last_obs_batch,
                last_done_batch,
                last_avail_batch,
                last_hstate,
                config["NUM_ACTORS"],
            )
            advantages, targets = calculate_gae(config, traj_batch, last_val)
            train_state, loss_info, rng = run_ppo_epochs(
                config, policy, train_state, traj_batch, advantages, targets, rng, config["NUM_ACTORS"]
            )
            metric = traj_batch.info
            metric["update_steps"] = update_steps
            
            update_steps += 1
            runner_state = (train_state, env_state, last_obs, last_done, last_hstate, rng)
            return (runner_state, update_steps), metric

        ckpt_and_eval_interval = config["NUM_UPDATES"] // max(1, config["NUM_CHECKPOINTS"] - 1)
        num_ckpts = config["NUM_CHECKPOINTS"]

        # build a pytree that can hold the parameters for all checkpoints.
        def init_ckpt_array(params_pytree):
            return jax.tree.map(
                lambda x: jnp.zeros((num_ckpts,) + x.shape, x.dtype),
                params_pytree
            )

        def _update_step_with_checkpoint(update_with_ckpt_runner_state, unused):
            (update_runner_state, checkpoint_array, ckpt_idx) = update_with_ckpt_runner_state
            # update_runner_state is ((train_state, env_state, obs, done, hstate, rng), update_steps)
            # Run one PPO update step
            update_runner_state, metric = _update_step(update_runner_state, None)
            _, update_steps = update_runner_state
            # update steps is 1-indexed because it was incremented at the end of the update step
            to_store = jnp.logical_or(jnp.equal(jnp.mod(update_steps-1, ckpt_and_eval_interval), 0),
                                      jnp.equal(update_steps, config["NUM_UPDATES"]))

            def store_ckpt_fn(args):
                # Write current runner_state[0].params into checkpoint_array at ckpt_idx
                # and increment ckpt_idx
                _checkpoint_array, _ckpt_idx = args
                new_checkpoint_array = jax.tree.map(
                    lambda c_arr, p: c_arr.at[_ckpt_idx].set(p),
                    _checkpoint_array,
                    update_runner_state[0][0].params
                )
                return new_checkpoint_array, _ckpt_idx + 1 
            # TODO: potential issue is that if this function is always executed regardless of whether to_store is true or false, then _ckpt_idx will be wrong

            def skip_ckpt_fn(args):
                return args  # No changes if we don't store

            checkpoint_array, ckpt_idx = jax.lax.cond(
                to_store, # if to_store, execute true function(operand). else, execute false function(operand).
                store_ckpt_fn, # true fn
                skip_ckpt_fn, # false fn
                (checkpoint_array, ckpt_idx),
            )

            runner_state = (update_runner_state, checkpoint_array, ckpt_idx)
            return runner_state, metric

        # (5) Use lax.scan over NUM_UPDATES
        rng, _rng = jax.random.split(rng)
        update_steps = 0
        init_hstate = policy.init_hstate(config["NUM_ACTORS"])
        init_done = {k: jnp.zeros((config["NUM_ENVS"]), dtype=bool) for k in env.agents + ["__all__"]}
        update_runner_state = ((train_state, env_state, obsv, init_done, init_hstate, _rng), update_steps)
        checkpoint_array = init_ckpt_array(train_state.params)
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
            "final_params": update_runner_state[0][0].params,
            "metrics": metrics,
            "checkpoints": checkpoint_array,
            "final_ckpt_idx": final_ckpt_idx,
        }
    return train

def run_ippo(config, logger):
    algorithm_config = dict(config.algorithm)
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rngs = jax.random.split(rng, algorithm_config["NUM_SEEDS"])
    
    with jax.disable_jit(False):
        train_jit = jax.jit(jax.vmap(make_train(algorithm_config, env)))
        out = train_jit(rngs)

    report_basic_training_outputs(
        config,
        out,
        logger,
        scalar_keys=[],
        print_prefix="ippo",
    )
    return out
