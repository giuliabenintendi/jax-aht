'''
Script for training an ego agent with an auxiliary partner-modeling loss.



Only supports a population of homogeneous RL partner agents.



Command to run partner-modeling ego training:

python ego_agent_training/run.py algorithm=partner_modeling_ego/lbf task=lbf label=test_partner_modeling_ego



Suggested debug command:

python ego_agent_training/run.py algorithm=partner_modeling_ego/lbf task=lbf logger.mode=disabled label=debug algorithm.TOTAL_TIMESTEPS=1e5
'''
import shutil
import time
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import hydra

from agents.population_interface import AgentPopulation
from common.plot_utils import get_stats, get_metric_names
from common.save_load_utils import save_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from common.agent_loader_from_config import initialize_rl_agent_from_config
from marl.ppo_utils import unbatchify
from ego_agent_training.auxiliary_ego_driver import train_auxiliary_ego_agent
from ego_agent_training.partner_modeling_utils import (
    PartnerModelingPolicy,
    Transition,
    initialize_partner_modeling_auxiliaries,
)
from ego_agent_training.utils import initialize_ego_agent

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_partner_modeling_ego_agent(config, env, train_rng,
                                     ego_policy, init_ego_params, n_ego_train_seeds,
                                     partner_population: AgentPopulation,
                                     partner_params):
    """Train an ego agent with an auxiliary partner-modeling loss."""

    def make_policy_tx(config, linear_schedule):
        if config["ANNEAL_LR"]:
            return optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        return optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )

    def init_rollout_runner_state(
        train_state,
        encoder_decoder_train_state,
        init_env_state,
        init_obs,
        init_done,
        init_ego_hstate,
        init_partner_hstate,
        partner_indices,
        rng,
    ):
        init_act_onehot = {
            k: jnp.zeros((config["NUM_ENVS"], env.action_space(env.agents[i]).n))
            for i, k in enumerate(env.agents)
        }
        return (
            train_state,
            encoder_decoder_train_state,
            init_env_state,
            init_obs,
            init_done,
            init_act_onehot,
            init_ego_hstate,
            init_partner_hstate,
            partner_indices,
            rng,
        )

    def env_step_fn(runner_state, unused):
        (
            train_state,
            encoder_decoder_train_state,
            env_state,
            prev_obs,
            prev_done,
            act_onehot,
            ego_hstate,
            partner_hstate,
            partner_indices,
            rng,
        ) = runner_state
        rng, actor_rng, partner_rng, step_rng = jax.random.split(rng, 4)

        avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
        avail_actions = jax.lax.stop_gradient(avail_actions)
        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

        sampled_indices_all = partner_population.sample_agent_indices(
            config["NUM_CONTROLLED_ACTORS"],
            partner_rng,
        )
        updated_partner_indices = jnp.where(
            prev_done["__all__"],
            sampled_indices_all,
            partner_indices,
        )

        act_0, val_0, pi_0, new_ego_hstate = ego_policy.get_action_value_policy(
            params={
                "encoder": encoder_decoder_train_state.params["encoder"],
                "decoder": encoder_decoder_train_state.params["decoder"],
                "policy": train_state.params,
            },
            obs=prev_obs["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
            done=prev_done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
            avail_actions=avail_actions_0,
            hstate=ego_hstate,
            rng=actor_rng,
            aux_obs=(act_onehot["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1), None, None),
        )
        logp_0 = pi_0.log_prob(act_0)
        act_0 = act_0.squeeze()
        logp_0 = logp_0.squeeze()
        val_0 = val_0.squeeze()

        act_1, new_partner_hstate = partner_population.get_actions(
            partner_params,
            updated_partner_indices,
            prev_obs["agent_1"].reshape(config["NUM_CONTROLLED_ACTORS"], 1, -1),
            prev_done["agent_1"].reshape(config["NUM_CONTROLLED_ACTORS"], 1, -1),
            avail_actions_1,
            partner_hstate,
            partner_rng,
            env_state=env_state,
            aux_obs=None,
        )
        act_1 = act_1.squeeze()

        combined_actions = jnp.concatenate([act_0, act_1], axis=0)
        env_act = unbatchify(combined_actions, env.agents, config["NUM_ENVS"], env.num_agents)
        env_act = {k: v.flatten() for k, v in env_act.items()}
        env_act_onehot = {
            k: jax.nn.one_hot(v.flatten(), env.action_space(env.agents[i]).n)
            for i, (k, v) in enumerate(env_act.items())
        }

        step_rngs = jax.random.split(step_rng, config["NUM_ENVS"])
        obs_next, env_state_next, reward, done_next, info = jax.vmap(
            env.step,
            in_axes=(0, 0, 0),
        )(step_rngs, env_state, env_act)
        info_0 = jax.tree.map(lambda x: x[:, 0], info)

        transition = Transition(
            done=done_next["agent_0"],
            action=act_0,
            value=val_0,
            reward=reward["agent_0"],
            log_prob=logp_0,
            obs=prev_obs["agent_0"],
            info=info_0,
            avail_actions=avail_actions_0,
            prev_action_onehot=act_onehot["agent_0"],
            partner_obs=prev_obs["agent_1"],
            partner_action_onehot=env_act_onehot["agent_1"],
        )
        new_runner_state = (
            train_state,
            encoder_decoder_train_state,
            env_state_next,
            obs_next,
            done_next,
            env_act_onehot,
            new_ego_hstate,
            new_partner_hstate,
            updated_partner_indices,
            rng,
        )
        return new_runner_state, transition

    def last_value_fn(train_state, encoder_decoder_train_state, runner_state):
        (
            _train_state,
            _encoder_decoder_train_state,
            env_state,
            obs,
            done,
            act_onehot,
            ego_hstate,
            _partner_hstate,
            _partner_indices,
            _rng,
        ) = runner_state
        avail_actions_0 = jax.vmap(env.get_avail_actions)(env_state.env_state)["agent_0"].astype(jnp.float32)
        _, last_val, _, _ = ego_policy.get_action_value_policy(
            params={
                "encoder": encoder_decoder_train_state.params["encoder"],
                "decoder": encoder_decoder_train_state.params["decoder"],
                "policy": train_state.params,
            },
            obs=obs["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1),
            done=done["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"]),
            avail_actions=jax.lax.stop_gradient(avail_actions_0),
            hstate=ego_hstate,
            rng=jax.random.PRNGKey(0),
            aux_obs=(act_onehot["agent_0"].reshape(1, config["NUM_CONTROLLED_ACTORS"], -1), None, None),
        )
        return last_val.squeeze()

    def aux_loss_fn(params, encoder_decoder_params, init_ego_hstate, traj_batch, gae, target_v):
        _, value, pi, recon_loss1, recon_loss2, _ = ego_policy.compute_decoder_losses(
            params={
                "encoder": encoder_decoder_params["encoder"],
                "decoder": encoder_decoder_params["decoder"],
                "policy": params,
            },
            obs=traj_batch.obs,
            done=traj_batch.done,
            avail_actions=traj_batch.avail_actions,
            hstate=init_ego_hstate,
            rng=jax.random.PRNGKey(0),
            aux_obs=(traj_batch.prev_action_onehot, None, None),
            modelled_agent_obs=traj_batch.partner_obs,
            modelled_agent_act=traj_batch.partner_action_onehot,
        )
        log_prob = pi.log_prob(traj_batch.action)
        recon_loss = recon_loss1 + recon_loss2

        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
            -config["CLIP_EPS"],
            config["CLIP_EPS"],
        )
        value_losses = jnp.square(value - target_v)
        value_losses_clipped = jnp.square(value_pred_clipped - target_v)
        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
        pg_loss_1 = ratio * gae_norm
        pg_loss_2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae_norm
        pg_loss = -jnp.mean(jnp.minimum(pg_loss_1, pg_loss_2))
        entropy = jnp.mean(pi.entropy())

        total_loss = (
            pg_loss
            + config["VF_COEF"] * value_loss
            - config["ENT_COEF"] * entropy
            + config["RECON_COEF"] * recon_loss
        )
        return total_loss, (value_loss, pg_loss, entropy, recon_loss)

    def metric_from_loss_terms(loss_terms):
        return {
            "actor_loss": loss_terms[1],
            "value_loss": loss_terms[0],
            "entropy_loss": loss_terms[2],
            "reconstruction_loss": loss_terms[3],
        }

    return train_auxiliary_ego_agent(
        config=config,
        env=env,
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=init_ego_params,
        n_ego_train_seeds=n_ego_train_seeds,
        partner_population=partner_population,
        partner_params=partner_params,
        make_policy_tx=make_policy_tx,
        init_rollout_runner_state=init_rollout_runner_state,
        env_step_fn=env_step_fn,
        last_value_fn=last_value_fn,
        aux_loss_fn=aux_loss_fn,
        metric_from_loss_terms=metric_from_loss_terms,
    )

def run_ego_training(config, wandb_logger):
    '''Run ego agent training against the population of partner agents.

    Args:
        config: dict, config for the training
    '''
    algorithm_config = dict(config["algorithm"])

    # Create only one environment instance
    env = make_env(algorithm_config["ENV_NAME"], algorithm_config["ENV_KWARGS"])
    env = LogWrapper(env)

    # Set the policy input dimension for the wrapped ego policy
    # Embedding dimension + observation dimension
    algorithm_config['POLICY_INPUT_DIM'] = algorithm_config['ENCODER_OUTPUT_DIM'] + env.observation_space(env.agents[0]).shape[0]

    rng = jax.random.PRNGKey(algorithm_config["TRAIN_SEED"])
    rng, init_partner_rng, init_ego_rng, train_rng = jax.random.split(rng, 4)


    partner_agent_config = dict(algorithm_config["partner_agent"])
    assert len(partner_agent_config) == 1, "Only supports training against one type of partner agent."

    partner0_name = list(partner_agent_config.keys())[0]
    partner0_agent_config = list(partner_agent_config.values())[0]
    partner_policy, partner_params, init_partner_params, idx_labels = initialize_rl_agent_from_config(
        partner0_agent_config, partner0_name, env, init_partner_rng)

    flattened_partner_params = jax.tree.map(lambda x, y: x.reshape((-1,) + y.shape), partner_params, init_partner_params)
    pop_size = jax.tree.leaves(flattened_partner_params)[0].shape[0]

    # Create partner population
    partner_population = AgentPopulation(
        pop_size=pop_size,
        policy_cls=partner_policy
    )

    rng, init_aux_rng, init_policy_rng = jax.random.split(init_ego_rng, 3)
    base_ego_policy, init_base_ego_params = initialize_ego_agent(
        algorithm_config,
        env,
        init_policy_rng,
    )
    encoder, decoder, init_aux_params = initialize_partner_modeling_auxiliaries(
        algorithm_config,
        env,
        init_aux_rng,
    )
    ego_policy = PartnerModelingPolicy(
        policy=base_ego_policy,
        encoder=encoder,
        decoder=decoder,
    )
    init_ego_params = {
        "encoder": init_aux_params["encoder"],
        "decoder": init_aux_params["decoder"],
        "policy": init_base_ego_params,
    }

    log.info("Starting ego agent training...")
    start_time = time.time()

    # Run the training
    out = train_partner_modeling_ego_agent(
        config=algorithm_config,
        env=env,
        train_rng=train_rng,
        ego_policy=ego_policy,
        init_ego_params=init_ego_params,
        n_ego_train_seeds=algorithm_config["NUM_EGO_TRAIN_SEEDS"],
        partner_population=partner_population,
        partner_params=flattened_partner_params
    )

    log.info(f"Training completed in {time.time() - start_time:.2f} seconds")

    # process and log metrics
    metric_names = get_metric_names(config["ENV_NAME"])
    log_metrics(config, out, wandb_logger, metric_names)

    return out["final_params"], ego_policy, init_ego_params

def log_metrics(config, train_out, logger, metric_names: tuple):
    """Process training metrics and log them using the provided logger.

    Args:
        training_logs: dict, the logs from training
        logger: Logger, instance to log metrics
        metric_names: tuple, names of metrics to extract from training logs
    """
    train_metrics = train_out["metrics"]

    #### Extract train metrics ####
    train_stats = get_stats(train_metrics, metric_names)
    # each key in train_stats is a metric name, and the value is an array of shape (num_seeds, num_updates, 2)
    # where the last dimension contains the mean and std of the metric
    train_stats = {k: np.mean(np.array(v), axis=0) for k, v in train_stats.items()}

    all_ego_value_losses = np.asarray(train_metrics["value_loss"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_actor_losses = np.asarray(train_metrics["actor_loss"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_entropy_losses = np.asarray(train_metrics["entropy_loss"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_reconstruction_losses = np.asarray(train_metrics["reconstruction_loss"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_grad_norms = np.asarray(train_metrics["avg_grad_norm"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_encoder_grad_norms = np.asarray(train_metrics["encoder_avg_grad_norm"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    all_ego_decoder_grad_norms = np.asarray(train_metrics["decoder_avg_grad_norm"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_minibatches)
    # Process eval return metrics - average across ego seeds, eval episodes,  training partners
    # and num_agents per game for each checkpoint
    all_ego_returns = np.asarray(train_metrics["eval_ep_last_info"]["returned_episode_returns"]) # shape (n_ego_train_seeds, num_updates, num_partners, num_eval_episodes, nuM_agents_per_game)
    average_ego_rets_per_iter = np.mean(all_ego_returns, axis=(0, 2, 3, 4))

    # Process loss metrics - average across ego seeds, partners and minibatches dims
    # Loss metrics shape should be (n_ego_train_seeds, num_updates, ...)
    average_ego_value_losses = np.mean(all_ego_value_losses, axis=(0, 2, 3))
    average_ego_actor_losses = np.mean(all_ego_actor_losses, axis=(0, 2, 3))
    average_ego_entropy_losses = np.mean(all_ego_entropy_losses, axis=(0, 2, 3))
    average_ego_reconstruction_losses = np.mean(all_ego_reconstruction_losses, axis=(0, 2, 3))
    average_ego_grad_norms = np.mean(all_ego_grad_norms, axis=(0, 2, 3))
    average_ego_encoder_grad_norms = np.mean(all_ego_encoder_grad_norms, axis=(0, 2, 3))
    average_ego_decoder_grad_norms = np.mean(all_ego_decoder_grad_norms, axis=(0, 2, 3))

    # Log metrics for each update step
    num_updates = len(average_ego_value_losses)
    for step in range(num_updates):
        for stat_name, stat_data in train_stats.items():
            # second dimension contains the mean and std of the metric
            stat_mean = stat_data[step, 0]
            logger.log_item(f"Train/Ego_{stat_name}", stat_mean, train_step=step, commit=True)

        logger.log_item("Eval/EgoReturn", average_ego_rets_per_iter[step], train_step=step, commit=True)
        logger.log_item("Train/EgoValueLoss", average_ego_value_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoActorLoss", average_ego_actor_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoEntropyLoss", average_ego_entropy_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoReconstructionLoss", average_ego_reconstruction_losses[step], train_step=step, commit=True)
        logger.log_item("Train/EgoGradNorm", average_ego_grad_norms[step], train_step=step, commit=True)
        logger.log_item("Train/EgoEncoderGradNorm", average_ego_encoder_grad_norms[step], train_step=step, commit=True)
        logger.log_item("Train/EgoDecoderGradNorm", average_ego_decoder_grad_norms[step], train_step=step, commit=True)
        logger.commit()

    # Saving artifacts
    savedir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    out_savepath = save_train_run(train_out, savedir, savename="ego_train_run")
    if config["logger"]["log_train_out"]:
        logger.log_artifact(name="ego_train_run", path=out_savepath, type_name="train_run")
        # Cleanup locally logged out file
    if not config["local_logger"]["save_train_out"]:
        shutil.rmtree(out_savepath)
