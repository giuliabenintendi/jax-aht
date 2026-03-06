import jax
import jax.numpy as jnp
import os
from envs.overcooked.adhoc_overcooked_visualizer import AdHocOvercookedVisualizer


def save_video(env, env_name, 
               agent_0_param, agent_0_policy, 
               agent_1_param, agent_1_policy, 
               max_episode_steps, num_eps, 
               savevideo: bool, save_dir: str, save_name: str):
    '''
    Render or save video of agent 0 and agent 1 playing against each other.
    
    Args:
        env: The environment instance
        env_name: Name of the environment ('lbf' or 'overcooked-v1')
        agent_0_param: Parameters for agent 0
        agent_0_policy: Policy for agent 0
        agent_1_param: Parameters for agent 1
        agent_1_policy: Policy for agent 1
        max_episode_steps: Maximum number of steps per episode
        num_eps: Number of episodes to run
        savevideo: Whether to save a video of the episode
        save_dir: Directory to save the video
        save_name: Name to use for the saved video
    '''
    assert env_name in ['lbf', 'lbf-reward-shaping', 'overcooked-v1'], "Supported environments are lbf or overcooked-v1"
    
    # Step 1: run the episode and generate a list of env states 
    states = []
    rng = jax.random.PRNGKey(112358)
    
    for episode in range(num_eps):
        print(f"Running episode {episode+1}/{num_eps}")
        rng, episode_rng = jax.random.split(rng)
        
        # Run a single episode and collect states
        episode_states = run_episode_with_states(
            episode_rng, env, agent_0_param, agent_0_policy,
            agent_1_param, agent_1_policy, max_episode_steps
        )
        
        states.extend(episode_states)
    
    # Step 2: render or save video
    print(f"\nSaving video with {len(states)} frames...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    savepath = f"{save_dir}/{save_name}.mp4"
    if env_name == 'lbf' or env_name == 'lbf-reward-shaping':
        anim = env.animate(states, interval=150)
        anim.save(savepath, writer="ffmpeg")
        print(f"Video saved successfully at {savepath}")

    elif env_name == 'overcooked-v1':
        viz = AdHocOvercookedVisualizer()
        # Get layout from env kwargs if available, otherwise use default
        viz.animate_mp4([s.env_state for s in states], env.agent_view_size, 
            highlight_agent_idx=0,
            filename=savepath, 
            pixels_per_tile=32, fps=25)
        print(f"MP4 saved successfully at {savepath}")
    else:
        print(f"Unknown environment: {env_name}.")

    return savepath


def run_episode_with_states(rng, env, agent_0_param, agent_0_policy,
                           agent_1_param, agent_1_policy,
                           max_episode_steps, collect_attention=False):
    '''
    Run a single episode and collect states for rendering.

    Args:
        collect_attention: if True and agents are JA, also return per-timestep
            attention maps via get_action_and_attention.

    Returns:
        ep_states when collect_attention=False,
        (ep_states, {"agent_0": [...], "agent_1": [...]}) when True.
    '''
    # Reset the env.
    rng, reset_rng = jax.random.split(rng)

    obs, env_state = env.reset(reset_rng)
    done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}

    # Initialize hidden states
    hstate_0 = agent_0_policy.init_hstate(1)
    hstate_1 = agent_1_policy.init_hstate(1)

    # Collect states for rendering
    ep_states = [env_state]
    attn_maps = {"agent_0": [], "agent_1": []}

    # Run episode until done or max steps reached
    step = 0
    while not done["__all__"] and step < max_episode_steps:
        # Get available actions for each agent
        avail_actions = env.get_avail_actions(env_state)
        avail_actions = jax.lax.stop_gradient(avail_actions)
        avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
        avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

        # Get agent obses
        obs_0, obs_1 = obs["agent_0"], obs["agent_1"]
        prev_done_0, prev_done_1 = done["agent_0"], done["agent_1"]

        # Reshape inputs for policies
        obs_0_reshaped = obs_0.reshape(1, 1, -1)
        done_0_reshaped = prev_done_0.reshape(1, 1)
        obs_1_reshaped = obs_1.reshape(1, 1, -1)
        done_1_reshaped = prev_done_1.reshape(1, 1)

        # Get actions for both agents
        rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)

        # Get ego action (optionally with attention)
        if collect_attention and hasattr(agent_0_policy, 'get_action_and_attention'):
            act_0, hstate_0, attn_0 = agent_0_policy.get_action_and_attention(
                params=agent_0_param,
                obs=obs_0_reshaped,
                done=done_0_reshaped,
                avail_actions=avail_actions_0,
                hstate=hstate_0,
                rng=act_rng,
            )
            attn_maps["agent_0"].append(attn_0)
        else:
            act_0, hstate_0 = agent_0_policy.get_action(
                params=agent_0_param,
                obs=obs_0_reshaped,
                done=done_0_reshaped,
                avail_actions=avail_actions_0,
                hstate=hstate_0,
                rng=act_rng,
            )
        act_0 = act_0.squeeze()

        # Get partner action (optionally with attention)
        if collect_attention and hasattr(agent_1_policy, 'get_action_and_attention'):
            act_1, hstate_1, attn_1 = agent_1_policy.get_action_and_attention(
                params=agent_1_param,
                obs=obs_1_reshaped,
                done=done_1_reshaped,
                avail_actions=avail_actions_1,
                hstate=hstate_1,
                rng=part_rng,
            )
            attn_maps["agent_1"].append(attn_1)
        else:
            act_1, hstate_1 = agent_1_policy.get_action(
                params=agent_1_param,
                obs=obs_1_reshaped,
                done=done_1_reshaped,
                avail_actions=avail_actions_1,
                hstate=hstate_1,
                rng=part_rng,
            )
        act_1 = act_1.squeeze()

        # Take step in environment
        both_actions = [act_0, act_1]
        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
        obs, env_state, reward, done, info = env.step(step_rng, env_state, env_act)

        # Add state to the list for rendering
        ep_states.append(env_state)

        step += 1

    if collect_attention:
        return ep_states, attn_maps
    return ep_states

def _render_single_frame(state, agent_view_size):
    """Render one env state to an RGB numpy array (H_px, W_px, 3)."""
    import numpy as np
    from envs.overcooked.rendering.overcooked_rendering import render_state
    img = render_state(state, agent_view_size=agent_view_size)
    return np.asarray(img)


def _overlay_attention(frame, attn, alpha=0.5):
    """Overlay attention heatmap on a rendered frame.

    Args:
        frame: (H_px, W_px, 3) uint8 RGB image.
        attn: (H, W) float attention weights (softmax output).
            May differ in spatial dims from frame — resized to match.
        alpha: blend factor for the heatmap overlay.

    Returns:
        (H_px, W_px, 3) uint8 RGB image with heatmap overlay.
    """
    import matplotlib.cm as cm
    from PIL import Image

    attn = np.array(attn).squeeze()
    # Normalize to [0, 1] for colormap
    a_min, a_max = attn.min(), attn.max()
    if a_max - a_min > 1e-8:
        attn_norm = (attn - a_min) / (a_max - a_min)
    else:
        attn_norm = np.zeros_like(attn)

    # Resize attention to match frame pixel dimensions
    h_px, w_px = frame.shape[:2]
    attn_resized = np.array(
        Image.fromarray(attn_norm.astype(np.float32), mode='F').resize(
            (w_px, h_px), resample=Image.NEAREST))

    # Apply colormap (hot: black -> red -> yellow -> white)
    heatmap_rgba = cm.hot(attn_resized)  # (H_px, W_px, 4) float [0,1]
    heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype(np.uint8)

    # Alpha blend
    blended = (alpha * heatmap_rgb.astype(np.float32)
               + (1 - alpha) * frame.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


def log_attention_to_wandb(attn_data, logger, step, tag_prefix="Eval",
                           commit=True, ep_states=None, agent_view_size=None):
    """Log attention heatmaps overlaid on rendered env frames to wandb.

    Args:
        attn_data: dict {"agent_0": [attn_map, ...], "agent_1": [...]},
            where each attn_map is (1, batch, H, W) from get_action_and_attention.
        logger: wandb run object (or anything with a .log method).
        step: global step for logging.
        tag_prefix: prefix for wandb log keys.
        ep_states: list of env states (WrappedEnvState) for rendering frames.
            If None, falls back to raw attention images.
        agent_view_size: env.agent_view_size, needed for rendering.
    """
    import numpy as np
    try:
        import wandb
    except ImportError:
        return

    for agent_name, maps in attn_data.items():
        if not maps:
            continue
        n = len(maps)

        # Print attention stats for diagnostics
        all_attn = np.array([np.array(m).squeeze() for m in maps])  # (T, H, W)
        h, w = all_attn.shape[1], all_attn.shape[2]
        uniform_ent = np.log(h * w)
        per_step_ent = -np.sum(all_attn * np.log(all_attn + 1e-10), axis=(1, 2))
        print(f"[attn] {agent_name}: shape=({h},{w}), "
              f"min={all_attn.min():.4f}, max={all_attn.max():.4f}, "
              f"mean_entropy={per_step_ent.mean():.3f} / {uniform_ent:.3f} (uniform)")

        indices = {"first": 0, "middle": n // 2, "last": n - 1}
        for label, idx in indices.items():
            attn = np.array(maps[idx]).squeeze()  # (H, W)

            # Overlay on rendered frame if states are available
            if ep_states is not None and agent_view_size is not None:
                # idx+1 because ep_states[0] is the initial reset state
                state_idx = min(idx + 1, len(ep_states) - 1)
                frame = _render_single_frame(
                    ep_states[state_idx].env_state, agent_view_size)
                overlay = _overlay_attention(frame, attn)
                img = wandb.Image(overlay, caption=f"{agent_name} t={idx}")
            else:
                # Fallback: normalize raw attention for visibility
                a_min, a_max = attn.min(), attn.max()
                if a_max - a_min > 1e-8:
                    attn = (attn - a_min) / (a_max - a_min)
                img = wandb.Image(attn, caption=f"{agent_name} t={idx}")

            logger.log({f"{tag_prefix}/{agent_name}_attn_{label}": img},
                       step=step, commit=commit)


def make_attention_video(ep_states, attn_data, agent_view_size, filename,
                         fps=10, agent_name="agent_1"):
    """Render an MP4 with attention heatmap overlaid on each frame.

    Args:
        ep_states: list of WrappedEnvState from run_episode_with_states.
        attn_data: dict with per-agent attention maps list.
        agent_view_size: env.agent_view_size.
        filename: output .mp4 path.
        fps: frames per second.
        agent_name: which agent's attention to overlay.
    """
    import os
    from moviepy import ImageSequenceClip

    maps = attn_data.get(agent_name, [])
    if not maps:
        print(f"[attn video] No attention maps for {agent_name}, skipping.")
        return

    frames = []
    for i, attn in enumerate(maps):
        # ep_states[0] is reset, attention maps start from step 0
        state_idx = min(i + 1, len(ep_states) - 1)
        frame = _render_single_frame(
            ep_states[state_idx].env_state, agent_view_size)
        overlay = _overlay_attention(frame, attn)
        frames.append(overlay)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(filename, fps=fps, codec='libx264', audio=False,
                         bitrate='8000k', preset='slow')
    print(f"[attn video] Saved {filename} ({len(frames)} frames)")


if __name__ == "__main__":
    from envs import make_env
    from agents.initialize_agents import initialize_mlp_agent
    from common.save_load_utils import load_checkpoints

    import sys


    if len(sys.argv) > 1: # either load from command line arguments
        # argument in the form of the raw path, include to a minimal the /results portion
        # ex: /scratch/cluster/jyliu/Documents/jax-aht/results/overcooked-v1/counter_circuit/ippo/2025-04-22_15-46-55
        import re
        ego_ckpt_path = re.findall("results/.*", sys.argv[1])[0] + "/saved_train_run"
    else: # or load the path manually
        ego_ckpt_path = "results/overcooked-v1/counter_circuit/ippo/2025-04-22_15-46-55/saved_train_run" # mlp ego agent

    ego_agent_ckpt = load_checkpoints(ego_ckpt_path)
    # ego_agent_params = jax.tree.map(lambda x: x[0, -1][np.newaxis, ...], ego_agent_ckpt)
    ego_agent_params = jax.tree.map(lambda x: x[0, -1], ego_agent_ckpt)

    # Initialize policies
    base_rng = jax.random.PRNGKey(112358)
    rng, init1_rng, init2_rng = jax.random.split(base_rng, 3)
    
    # choose env
    env_name = "lbf-reward-shaping" # "lbf" or "overcooked-v1"
    env_kwargs = { # specify the layout for overcooked 
        # "layout": "counter_circuit",
        # "random_reset": False,
        # "max_steps": 400
    }
    
    env = make_env(env_name, env_kwargs if env_name[:10] == "overcooked" else {})

    # Initialize the policies with the loaded parameters
    agent_0_policy, _ = initialize_mlp_agent({}, env, init1_rng)
    agent_1_policy, _ = initialize_mlp_agent({}, env, init2_rng)
    
    # Make sure the policies are properly initialized with the parameters
    
    save_video(env, env_name, 
        agent_0_param=ego_agent_params, agent_0_policy=agent_0_policy, 
        agent_1_param=ego_agent_params, agent_1_policy=agent_1_policy, 
        max_episode_steps=100 if env_name == "lbf" or env_name == "lbf-reward-shaping" else 400, num_eps=1, 
        savevideo=True, 
        save_dir=f"results/{env_name}/videos/", save_name="ego-vs-ego-test")


