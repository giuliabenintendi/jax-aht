import jax
import jax.numpy as jnp
import os
from envs.overcooked.adhoc_overcooked_visualizer import AdHocOvercookedVisualizer


def _action_to_ground_truth(env, state, action):
    """Map wrapper-space actions back to the base environment action space."""
    current_env = env
    current_state = state
    true_action = action

    while True:
        if hasattr(current_env, "_invert_actions"):
            true_action = current_env._invert_actions(true_action, current_state)
        if hasattr(current_env, "_env") and hasattr(current_state, "env_state"):
            current_env = current_env._env
            current_state = current_state.env_state
        else:
            break

    return true_action


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
        episode_states, _, _ = run_episode_with_states(
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
                           max_episode_steps, collect_attention=False,
                           greedy=True, feed_other_attn_dims=None,
                           fixed_partner_attn=None):
    '''
    Run a single episode and collect states for rendering.

    Args:
        collect_attention: if True and agents are JA, also return per-timestep
            attention maps via get_action_and_attention.
        feed_other_attn_dims: if not None, a tuple (img_h, img_w, feat_h, feat_w)
            for augmenting obs with the other agent's previous attention map.

    Returns:
        ep_states when collect_attention=False,
        (ep_states, {"agent_0": [...], "agent_1": [...]}) when True.
    '''
    from agents.ja_utils import augment_obs_for_eval

    # Reset the env.
    rng, reset_rng = jax.random.split(rng)

    obs, env_state = env.reset(reset_rng)
    done = {k: jnp.zeros((1), dtype=bool) for k in env.agents + ["__all__"]}

    # Initialize hidden states
    hstate_0 = agent_0_policy.init_hstate(1)
    hstate_1 = agent_1_policy.init_hstate(1)
    use_prev_io = (
        getattr(agent_0_policy, "uses_prev_reward_action", False)
        and getattr(agent_1_policy, "uses_prev_reward_action", False)
    )
    if use_prev_io:
        prev_reward_0 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_reward_1 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_action_0 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_action_1 = jnp.zeros((1, 1), dtype=jnp.float32)

    # Initialize previous attention maps for feed_other_attn
    if feed_other_attn_dims is not None:
        _img_h, _img_w, _feat_h, _feat_w = feed_other_attn_dims
        prev_attn_0 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)
        prev_attn_1 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)

    _xattn = getattr(agent_0_policy, 'cross_agent_attn', False)
    if _xattn:
        _ed = getattr(agent_0_policy, 'xattn_embed_dim', 64)
        _pe_actor_0 = jnp.zeros((1, 1, _ed))
        _pe_actor_1 = jnp.zeros((1, 1, _ed))
        _pe_critic_0 = jnp.zeros((1, 1, _ed))
        _pe_critic_1 = jnp.zeros((1, 1, _ed))

    # Collect states and actions for rendering
    ep_states = [env_state]
    ep_actions = []
    ep_messages = []  # (msg_0, msg_1) per step, empty if no communication
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

        # Augment obs with other agent's previous attention as 4th channel
        if feed_other_attn_dims is not None:
            obs_0 = augment_obs_for_eval(obs_0, prev_attn_1, _img_h, _img_w)
            obs_1 = augment_obs_for_eval(obs_1, prev_attn_0, _img_h, _img_w)

        # Reshape inputs for policies
        obs_0_reshaped = obs_0.reshape(1, 1, -1)
        done_0_reshaped = prev_done_0.reshape(1, 1)
        obs_1_reshaped = obs_1.reshape(1, 1, -1)
        done_1_reshaped = prev_done_1.reshape(1, 1)

        # Get actions for both agents
        rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)
        has_comm = getattr(env, 'communication', False)
        num_cards = getattr(env, 'num_cards', 5)

        _xattn = getattr(agent_0_policy, 'cross_agent_attn', False)

        # Get ego action (optionally with attention)
        if collect_attention and hasattr(agent_0_policy, 'get_action_and_attention'):
            if _xattn:
                act_0, hstate_0, attn_0, own_a0, own_c0 = agent_0_policy.get_action_and_attention(
                    params=agent_0_param,
                    obs=obs_0_reshaped,
                    done=done_0_reshaped,
                    avail_actions=avail_actions_0,
                    hstate=hstate_0,
                    rng=act_rng,
                    greedy=greedy,
                    agent_id=0,
                    partner_embed_actor=_pe_actor_0,
                    partner_embed_critic=_pe_critic_0,
                )
            else:
                act_0, hstate_0, attn_0 = agent_0_policy.get_action_and_attention(
                    params=agent_0_param,
                    obs=obs_0_reshaped,
                    done=done_0_reshaped,
                    avail_actions=avail_actions_0,
                    hstate=hstate_0,
                    rng=act_rng,
                    greedy=greedy,
                    agent_id=0,
                    prev_reward=prev_reward_0 if use_prev_io else None,
                    prev_action=prev_action_0 if use_prev_io else None,
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
                greedy=greedy,
                prev_reward=prev_reward_0 if use_prev_io else None,
                prev_action=prev_action_0 if use_prev_io else None,
            )
        act_0 = act_0.squeeze()

        # Get partner action (optionally with attention)
        if collect_attention and hasattr(agent_1_policy, 'get_action_and_attention'):
            if _xattn:
                act_1, hstate_1, attn_1, own_a1, own_c1 = agent_1_policy.get_action_and_attention(
                    params=agent_1_param,
                    obs=obs_1_reshaped,
                    done=done_1_reshaped,
                    avail_actions=avail_actions_1,
                    hstate=hstate_1,
                    rng=part_rng,
                    greedy=greedy,
                    agent_id=1,
                    partner_embed_actor=_pe_actor_1,
                    partner_embed_critic=_pe_critic_1,
                )
            else:
                act_1, hstate_1, attn_1 = agent_1_policy.get_action_and_attention(
                    params=agent_1_param,
                    obs=obs_1_reshaped,
                    done=done_1_reshaped,
                    avail_actions=avail_actions_1,
                    hstate=hstate_1,
                    rng=part_rng,
                    greedy=greedy,
                    agent_id=1,
                    prev_reward=prev_reward_1 if use_prev_io else None,
                    prev_action=prev_action_1 if use_prev_io else None,
                )
            # Override agent 1's attention if fixed partner
            if fixed_partner_attn is not None:
                attn_1 = fixed_partner_attn[None, None]  # match shape
            attn_maps["agent_1"].append(attn_1)
        else:
            act_1, hstate_1 = agent_1_policy.get_action(
                params=agent_1_param,
                obs=obs_1_reshaped,
                done=done_1_reshaped,
                avail_actions=avail_actions_1,
                hstate=hstate_1,
                rng=part_rng,
                greedy=greedy,
                prev_reward=prev_reward_1 if use_prev_io else None,
                prev_action=prev_action_1 if use_prev_io else None,
            )
        act_1 = act_1.squeeze()

        # Update cross-agent partner embeddings
        if _xattn and collect_attention:
            _pe_actor_0, _pe_actor_1 = own_a1, own_a0
            _pe_critic_0, _pe_critic_1 = own_c1, own_c0

        # Update previous attention maps for feed_other_attn
        if feed_other_attn_dims is not None and collect_attention:
            prev_attn_0 = attn_0.squeeze()  # (feat_h, feat_w)
            prev_attn_1 = attn_1.squeeze()  # (feat_h, feat_w)

        # Take step in environment (joint action encodes card choice + message)
        both_actions = [act_0, act_1]
        env_act = {k: both_actions[i] for i, k in enumerate(env.agents)}
        render_act = _action_to_ground_truth(env, env_state, env_act)
        obs, env_state, reward, done, info = env.step(step_rng, env_state, env_act)
        if use_prev_io:
            prev_reward_0 = reward["agent_0"].reshape(1, 1).astype(jnp.float32)
            prev_reward_1 = reward["agent_1"].reshape(1, 1).astype(jnp.float32)
            prev_action_0 = act_0.reshape(1, 1).astype(jnp.float32)
            prev_action_1 = act_1.reshape(1, 1).astype(jnp.float32)

        # Add state and actions to the lists for rendering
        fp = getattr(env, 'fixed_partner_pos', -1)
        act_0_record = int(render_act["agent_0"])
        act_1_record = int(fp) if fp >= 0 else int(render_act["agent_1"])
        ep_states.append(env_state)
        if has_comm:
            # Decode: 0-24 = card+msg, 25-29 = msg only (deliberation)
            n_card_msg = num_cards * num_cards
            a0_int, a1_int = act_0_record, int(render_act["agent_1"])
            card_0 = a0_int // num_cards if a0_int < n_card_msg else -1
            card_1 = (act_1_record // num_cards if fp < 0 else act_1_record) if a1_int < n_card_msg else -1
            msg_0 = a0_int % num_cards if a0_int < n_card_msg else a0_int - n_card_msg
            msg_1 = a1_int % num_cards if a1_int < n_card_msg else a1_int - n_card_msg
            ep_actions.append((card_0, card_1))
            ep_messages.append((msg_0, msg_1))
        else:
            ep_actions.append((act_0_record, act_1_record))

        step += 1

    if collect_attention:
        return ep_states, attn_maps, ep_actions, ep_messages
    return ep_states, ep_actions, ep_messages

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
        padding = 4  # must match OvercookedImageWrapper._make_obs
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




INDEX_TO_OBJECT = {
    0: "unseen", 1: "floor", 2: "wall", 3: "onion",
    4: "onion_disp", 5: "plate", 6: "plate_disp",
    7: "serve", 8: "pot", 9: "dish", 10: "agent",
    11: "counter",
}
NUM_CATEGORIES = len(INDEX_TO_OBJECT)

# Indices for maze_map channel 0 (from jaxmarl OBJECT_TO_INDEX)
_WALL_IDX = 2


def _build_counter_mask(wall_map):
    """Identify counters: wall_map=True tiles adjacent to at least one walkable tile.

    Real walls are surrounded entirely by other wall_map=True tiles.
    Counters are wall_map=True tiles that border at least one walkable (False) tile.
    """
    import numpy as np

    wm = np.array(wall_map, dtype=bool)
    h, w = wm.shape
    counter = np.zeros_like(wm, dtype=bool)

    for r in range(h):
        for c in range(w):
            if not wm[r, c]:
                continue
            # Check 4-connected neighbors for walkable tiles
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not wm[nr, nc]:
                    counter[r, c] = True
                    break

    return counter


def build_coverage_map(state, feat_h, feat_w, padding=4, tile_pixels=7):
    """Build a (feat_h, feat_w, NUM_CATEGORIES) coverage tensor from env state.

    For each feature-map cell, computes the fractional area covered by each
    tile type. Uses wall_map + adjacency to distinguish walls from counters,
    and overlays dynamic agent positions.

    Args:
        state: WrappedEnvState — must have .env_state with maze_map, wall_map, agent_pos.
        feat_h: feature map height.
        feat_w: feature map width.
        padding: maze_map padding (default 4 for agent_view_size=5).
        tile_pixels: pixels per tile (default 7).

    Returns:
        (feat_h, feat_w, NUM_CATEGORIES) float array, each cell sums to 1.
    """
    import numpy as np

    env_state = state.env_state
    grid = np.array(env_state.maze_map[padding:-padding, padding:-padding, 0])
    grid_h, grid_w = grid.shape

    # Distinguish counters from walls using wall_map adjacency
    wall_map = np.array(env_state.wall_map)
    counter_mask = _build_counter_mask(wall_map)

    # Remap: wall tiles that are counters get index 11
    for r in range(grid_h):
        for c in range(grid_w):
            if grid[r, c] == _WALL_IDX and counter_mask[r, c]:
                grid[r, c] = 11  # counter

    # Overlay agent positions
    agent_pos = np.array(env_state.agent_pos)  # (num_agents, 2) — (x, y)
    for a in range(agent_pos.shape[0]):
        col, row = int(agent_pos[a, 0]), int(agent_pos[a, 1])
        if 0 <= row < grid_h and 0 <= col < grid_w:
            grid[row, col] = 10  # agent

    img_h = grid_h * tile_pixels
    img_w = grid_w * tile_pixels

    coverage = np.zeros((feat_h, feat_w, NUM_CATEGORIES), dtype=np.float32)

    for r in range(feat_h):
        for c in range(feat_w):
            # Pixel rectangle this feature cell covers
            y0 = r * img_h / feat_h
            y1 = (r + 1) * img_h / feat_h
            x0 = c * img_w / feat_w
            x1 = (c + 1) * img_w / feat_w

            # Iterate over tiles that overlap with this rectangle
            tile_r0 = max(0, int(y0 // tile_pixels))
            tile_r1 = min(grid_h, int(np.ceil(y1 / tile_pixels)))
            tile_c0 = max(0, int(x0 // tile_pixels))
            tile_c1 = min(grid_w, int(np.ceil(x1 / tile_pixels)))

            total_area = 0.0
            for tr in range(tile_r0, tile_r1):
                for tc in range(tile_c0, tile_c1):
                    # Overlap area between feature cell and tile
                    oy0 = max(y0, tr * tile_pixels)
                    oy1 = min(y1, (tr + 1) * tile_pixels)
                    ox0 = max(x0, tc * tile_pixels)
                    ox1 = min(x1, (tc + 1) * tile_pixels)
                    area = max(0.0, oy1 - oy0) * max(0.0, ox1 - ox0)
                    if area > 0:
                        obj_idx = int(grid[tr, tc])
                        if 0 <= obj_idx < NUM_CATEGORIES:
                            coverage[r, c, obj_idx] += area
                        total_area += area

            if total_area > 0:
                coverage[r, c] /= total_area

    return coverage


def render_coverage_debug(frame, coverage_map, attn_map=None, upscale=16):
    """Render a debug image showing the feature-map grid over the game frame.

    Each feature cell is labeled with its dominant category (and percentage).
    If attn_map is provided, cells below 0.05 are dimmed.

    Args:
        frame: (H_px, W_px, 3) uint8 game frame.
        coverage_map: (feat_h, feat_w, NUM_CATEGORIES) from build_coverage_map.
        attn_map: optional (feat_h, feat_w) attention weights.
        upscale: factor to enlarge the frame for readability.

    Returns:
        RGB uint8 numpy array.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    feat_h, feat_w = coverage_map.shape[:2]
    h_px, w_px = frame.shape[:2]

    # Upscale frame for readability
    out_w, out_h = w_px * upscale, h_px * upscale
    img = Image.fromarray(frame).resize((out_w, out_h), resample=Image.NEAREST)
    draw = ImageDraw.Draw(img)

    cell_h = out_h / feat_h
    cell_w = out_w / feat_w

    # Try to load a truetype font; fall back to default
    font_size = max(10, int(min(cell_h, cell_w) / 5))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Short labels for categories
    _SHORT = {
        "unseen": "?", "floor": "flr", "wall": "wal", "onion": "oni",
        "onion_disp": "o.d", "plate": "plt", "plate_disp": "p.d",
        "serve": "srv", "pot": "pot", "dish": "dsh", "agent": "agt",
        "counter": "ctr",
    }

    for r in range(feat_h):
        for c in range(feat_w):
            x0 = int(c * cell_w)
            y0 = int(r * cell_h)
            x1 = int((c + 1) * cell_w)
            y1 = int((r + 1) * cell_h)

            # Draw grid lines
            draw.rectangle([x0, y0, x1, y1], outline="white", width=1)

            # Build label: top 2 categories with percentages
            cov = coverage_map[r, c]
            top_indices = np.argsort(-cov)
            parts = []
            for idx in top_indices[:2]:
                if cov[idx] > 0.05:
                    name = INDEX_TO_OBJECT.get(idx, "?")
                    short = _SHORT.get(name, name[:3])
                    parts.append(f"{short}{int(cov[idx]*100)}")
            label = "\n".join(parts) if parts else "?"

            # Dim cells with low attention
            if attn_map is not None:
                attn_val = float(np.array(attn_map).squeeze()[r, c])
                if attn_val < 0.05:
                    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 140))
                    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                    draw = ImageDraw.Draw(img)
                    continue
                # Show attention value
                label = f"a={attn_val:.2f}\n{label}"

            # Draw text with shadow for readability
            text_x = x0 + 3
            text_y = y0 + 3
            draw.text((text_x + 1, text_y + 1), label, fill="black", font=font)
            draw.text((text_x, text_y), label, fill="yellow", font=font)

    return np.array(img)



def compute_attention_stasis(attn_data):
    """Compute per-agent attention stasis via mean consecutive JSD.

    Measures how much each agent's own attention map changes from one
    timestep to the next. Low values indicate static/fixated attention
    (potential reward hacking); higher values indicate dynamic attention
    that tracks the game state.

    Args:
        attn_data: dict {"agent_0": [attn_map, ...], "agent_1": [...]},
            where each attn_map is (1, batch, H, W) from get_action_and_attention.

    Returns:
        dict with per-agent metrics:
            "agent_0_stasis": float — mean consecutive JSD (0 = perfectly static),
            "agent_1_stasis": float,
            "agent_0_jsd_trace": list of per-step JSD values,
            "agent_1_jsd_trace": list of per-step JSD values,
    """
    import numpy as np

    eps = 1e-8

    def _jsd(p, q):
        """JSD between two 1-D probability distributions (numpy)."""
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
        kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
        return float(0.5 * kl_pm + 0.5 * kl_qm)

    metrics = {}
    for agent_name in ("agent_0", "agent_1"):
        maps = attn_data.get(agent_name, [])
        if len(maps) < 2:
            metrics[f"{agent_name}_stasis"] = float("nan")
            metrics[f"{agent_name}_jsd_trace"] = []
            continue

        flat = [np.array(m).squeeze().flatten() for m in maps]
        jsd_vals = [_jsd(flat[t], flat[t + 1]) for t in range(len(flat) - 1)]

        metrics[f"{agent_name}_stasis"] = float(np.mean(jsd_vals))
        metrics[f"{agent_name}_jsd_trace"] = jsd_vals

    return metrics


def compute_object_coverage(attn_data, ep_states, feat_h, feat_w,
                            padding=4, tile_pixels=7):
    """Compute per-agent, per-timestep attention mass on objects vs non-objects.

    For each timestep, reports the fraction of feature-map cells that cover
    task-relevant objects (semantic categories) and the attention weight on
    those cells, kept as separate quantities.

    Object categories: onion, onion_disp, plate, plate_disp, serve, pot, dish, agent.
    Non-object categories: floor, wall, counter, unseen.

    Args:
        attn_data: dict {"agent_0": [...], "agent_1": [...]}.
        ep_states: list of WrappedEnvState from run_episode_with_states.
        feat_h, feat_w: feature map spatial dimensions.
        padding: maze_map padding (default 4).
        tile_pixels: pixels per tile (default 7).

    Returns:
        dict with per-agent metrics:
            "agent_0_pct_objects": float — mean % of attention mass on object cells,
            "agent_1_pct_objects": float,
            "agent_0_pct_objects_trace": list of per-timestep values,
            "agent_1_pct_objects_trace": list of per-timestep values,
            "agent_0_category_mass": dict — mean attention mass per category,
            "agent_1_category_mass": dict — mean attention mass per category,
    """
    import numpy as np

    OBJECT_CATEGORIES = {"onion", "onion_disp", "plate", "plate_disp",
                         "serve", "pot", "dish", "agent"}

    metrics = {}
    for agent_name in ("agent_0", "agent_1"):
        maps = attn_data.get(agent_name, [])
        n_steps = min(len(maps), len(ep_states) - 1)
        if n_steps == 0:
            metrics[f"{agent_name}_pct_objects"] = float("nan")
            metrics[f"{agent_name}_pct_objects_trace"] = []
            metrics[f"{agent_name}_category_mass"] = {}
            continue

        pct_trace = []
        category_accum = {name: 0.0 for name in INDEX_TO_OBJECT.values()}

        for t in range(n_steps):
            attn = np.array(maps[t]).squeeze()  # (feat_h, feat_w)
            coverage = build_coverage_map(ep_states[t], feat_h, feat_w,
                                          padding=padding, tile_pixels=tile_pixels)

            # Per-cell: weighted coverage by attention
            # attn (H, W), coverage (H, W, NUM_CATEGORIES)
            weighted = attn[..., None] * coverage  # (H, W, NUM_CATEGORIES)
            category_mass = weighted.sum(axis=(0, 1))  # (NUM_CATEGORIES,)

            obj_mass = sum(
                float(category_mass[i])
                for i, name in INDEX_TO_OBJECT.items()
                if name in OBJECT_CATEGORIES
            )
            pct_trace.append(float(obj_mass))

            for i, name in INDEX_TO_OBJECT.items():
                category_accum[name] += float(category_mass[i])

        metrics[f"{agent_name}_pct_objects"] = float(np.mean(pct_trace))
        metrics[f"{agent_name}_pct_objects_trace"] = pct_trace
        metrics[f"{agent_name}_category_mass"] = {
            k: v / n_steps for k, v in category_accum.items()
        }

    return metrics


def _overlay_attention(frame, attn, cmap_name, alpha=0.6):
    """Blend a single attention heatmap onto a game frame.

    Args:
        frame: (H_px, W_px, 3) uint8 RGB.
        attn: (fh, fw) float attention weights.
        cmap_name: matplotlib colormap name (e.g. 'Blues', 'Reds', 'jet').
        alpha: max overlay opacity.

    Returns:
        (H_px, W_px, 3) uint8 blended frame.
    """
    import numpy as np
    import matplotlib.cm as cm
    from PIL import Image

    attn = np.array(attn).squeeze()
    a_min, a_max = attn.min(), attn.max()
    if a_max - a_min > 1e-8:
        attn_norm = (attn - a_min) / (a_max - a_min)
    else:
        attn_norm = np.zeros_like(attn)

    h_px, w_px = frame.shape[:2]
    attn_resized = np.array(
        Image.fromarray(attn_norm.astype(np.float32), mode='F').resize(
            (w_px, h_px), resample=Image.NEAREST))

    cmap = getattr(cm, cmap_name)
    heatmap_rgb = cmap(attn_resized)[..., :3]  # (H, W, 3) float [0,1]
    a = (attn_resized * alpha)[..., None]
    blended = (1 - a) * frame.astype(np.float32) + a * heatmap_rgb * 255
    return np.clip(blended, 0, 255).astype(np.uint8)


def _make_overlay_video(frames, attn_maps, cmap_name, filename, fps):
    """Create a single MP4 with a colormap heatmap overlaid on the game frame."""
    import os
    import numpy as np
    from moviepy import ImageSequenceClip

    n = min(len(attn_maps), len(frames) - 1)
    overlay_frames = []
    for i in range(n):
        frame_idx = min(i + 1, len(frames) - 1)
        overlay = _overlay_attention(frames[frame_idx], attn_maps[i], cmap_name)
        overlay_frames.append(overlay)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    clip = ImageSequenceClip(overlay_frames, fps=fps)
    clip.write_videofile(filename, fps=fps, codec='libx264', audio=False,
                         bitrate='8000k', preset='slow')
    print(f"[attn video] Saved {filename} ({len(overlay_frames)} frames)")


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

    _make_overlay_video(frames, maps_0, "Blues", f"{base}_agent0.mp4", fps)
    _make_overlay_video(frames, maps_1, "Reds", f"{base}_agent1.mp4", fps)
    _make_overlay_video(frames, combined, "jet", f"{base}_combined.mp4", fps)


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
