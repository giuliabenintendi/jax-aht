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

def _render_heatmap_panel(attn, title, cmap, target_height, figwidth=3.0):
    """Render a single attention heatmap with grid, numbers, and colorbar.

    Returns an RGB uint8 array resized to match target_height.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    attn = np.array(attn).squeeze()
    h, w = attn.shape

    fig, ax = plt.subplots(1, 1, figsize=(figwidth, figwidth * h / max(w, 1)))
    im = ax.imshow(attn, cmap=cmap, vmin=0, origin="upper", aspect="equal")
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.7)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(h):
        for c in range(w):
            val = attn[r, c]
            color = "white" if val > (attn.max() * 0.6) else "black"
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(4, 36 // max(h, w)), color=color)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # Resize to match target height
    pil_img = Image.fromarray(buf)
    scale = target_height / buf.shape[0]
    new_w = int(buf.shape[1] * scale)
    pil_img = pil_img.resize((new_w, target_height), resample=Image.LANCZOS)
    return np.array(pil_img)


def _make_heatmap_figure(attn_maps, labels, cmaps, title=None, figsize=(4, 3.5)):
    """Render multi-panel attention heatmaps as a matplotlib figure (for wandb logging)."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(attn_maps)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]

    for ax, attn, label, cmap in zip(axes, attn_maps, labels, cmaps):
        attn = np.array(attn).squeeze()
        h, w = attn.shape
        im = ax.imshow(attn, cmap=cmap, vmin=0, origin="upper", aspect="equal")
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.7)
        ax.tick_params(which="minor", size=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for r in range(h):
            for c in range(w):
                val = attn[r, c]
                color = "white" if val > (attn.max() * 0.6) else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(4, 36 // max(h, w)), color=color)
        ax.set_title(label, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf


def render_episode_frames(ep_states, agent_view_size, pixels_per_tile=32):
    """Render all episode states to a list of RGB numpy arrays.

    Uses the non-JAX OvercookedVisualizer renderer (fast, no JIT overhead).

    Args:
        ep_states: list of WrappedEnvState from run_episode_with_states.
        agent_view_size: env.agent_view_size.
        pixels_per_tile: tile size for rendering.

    Returns:
        list of (H_px, W_px, 3) uint8 numpy arrays.
    """
    import numpy as np
    from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

    frames = []
    for ws in ep_states:
        state = ws.env_state
        padding = agent_view_size - 2
        grid = np.asarray(state.maze_map[padding:-padding, padding:-padding, :])
        highlight_mask = np.zeros(grid.shape[:2], dtype=bool)
        frame = OvercookedVisualizer._render_grid(
            grid,
            tile_size=pixels_per_tile,
            highlight_mask=highlight_mask,
            agent_dir_idx=state.agent_dir_idx,
            agent_inv=state.agent_inv,
        )
        frames.append(np.asarray(frame))
    return frames


def log_attention_to_wandb(attn_data, logger, step, tag_prefix="Eval",
                           commit=True, frames=None):
    """Log three-panel attention heatmaps to wandb: agent_0 (blue), agent_1 (red), overlap.

    Args:
        attn_data: dict {"agent_0": [attn_map, ...], "agent_1": [...]},
            where each attn_map is (1, batch, H, W) from get_action_and_attention.
        logger: wandb run object (or anything with a .log method).
        step: global step for logging.
        tag_prefix: prefix for wandb log keys.
        frames: list of pre-rendered RGB frames (from render_episode_frames).
    """
    import numpy as np
    try:
        import wandb
    except ImportError:
        return

    maps_0 = attn_data.get("agent_0", [])
    maps_1 = attn_data.get("agent_1", [])
    if not maps_0 or not maps_1:
        return

    n = min(len(maps_0), len(maps_1))

    # Print attention stats for diagnostics
    for agent_name, maps in attn_data.items():
        all_attn = np.array([np.array(m).squeeze() for m in maps])
        h, w = all_attn.shape[1], all_attn.shape[2]
        uniform_ent = np.log(h * w)
        per_step_ent = -np.sum(all_attn * np.log(all_attn + 1e-10), axis=(1, 2))
        print(f"[attn] {agent_name}: shape=({h},{w}), "
              f"min={all_attn.min():.4f}, max={all_attn.max():.4f}, "
              f"mean_entropy={per_step_ent.mean():.3f} / {uniform_ent:.3f} (uniform)")

    indices = {"first": 0, "middle": n // 2, "last": n - 1}
    for label, idx in indices.items():
        attn_0 = np.array(maps_0[idx]).squeeze()
        attn_1 = np.array(maps_1[idx]).squeeze()
        combined = np.minimum(attn_0, attn_1)

        panel = _make_heatmap_figure(
            [attn_0, attn_1, combined],
            ["Agent 0", "Agent 1", "Combined"],
            ["Blues", "Reds", "jet"],
            title=f"Attention t={idx}",
        )
        img = wandb.Image(panel, caption=f"t={idx}")
        logger.log({f"{tag_prefix}/attention_{label}": img},
                   step=step, commit=commit)


def _make_overlay_video(frames, attn_maps, title, cmap, filename, fps):
    """Create a single MP4 with a heatmap panel overlaid next to the game frame."""
    import os
    import numpy as np
    from moviepy import ImageSequenceClip

    n = min(len(attn_maps), len(frames) - 1)
    composite_frames = []
    for i in range(n):
        frame_idx = min(i + 1, len(frames) - 1)
        game_frame = frames[frame_idx]
        attn = np.array(attn_maps[i]).squeeze()
        panel = _render_heatmap_panel(attn, title, cmap, game_frame.shape[0])
        composite_frames.append(np.concatenate([game_frame, panel], axis=1))

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    clip = ImageSequenceClip(composite_frames, fps=fps)
    clip.write_videofile(filename, fps=fps, codec='libx264', audio=False,
                         bitrate='8000k', preset='slow')
    print(f"[attn video] Saved {filename} ({len(composite_frames)} frames)")


def make_attention_video(frames, attn_data, filename, fps=10):
    """Create three MP4s: Agent 0 (Blues), Agent 1 (Reds), Combined (jet).

    Each video shows the game frame with the corresponding heatmap panel alongside.
    Combined = element-wise minimum, highlighting where both agents attend.

    Output files: <filename>_agent0.mp4, <filename>_agent1.mp4, <filename>_combined.mp4

    Args:
        frames: list of pre-rendered RGB frames (from render_episode_frames).
        attn_data: dict with per-agent attention maps list.
        filename: base output path (extension stripped, suffixes appended).
        fps: frames per second.
    """
    import numpy as np

    maps_0 = attn_data.get("agent_0", [])
    maps_1 = attn_data.get("agent_1", [])
    if not maps_0 or not maps_1:
        print("[attn video] Missing attention maps for one or both agents, skipping.")
        return

    base = filename.rsplit(".", 1)[0] if "." in filename else filename

    n = min(len(maps_0), len(maps_1))
    combined = [np.minimum(np.array(maps_0[i]).squeeze(),
                           np.array(maps_1[i]).squeeze()) for i in range(n)]

    _make_overlay_video(frames, maps_0, "Agent 0", "Blues", f"{base}_agent0.mp4", fps)
    _make_overlay_video(frames, maps_1, "Agent 1", "Reds", f"{base}_agent1.mp4", fps)
    _make_overlay_video(frames, combined, "Combined", "jet", f"{base}_combined.mp4", fps)


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


