"""LBF-specific eval frame rendering."""


def _to_jumanji_state(st):
    # Walk wrapper layers (LogWrapper / Other-Play / LBFWrapper) to the inner
    # jumanji LBF state that owns food_items. Under OP there is an extra layer,
    # so the old hard-coded `s.env_state` landed on a WrappedEnvState.
    cur = st
    for _ in range(8):
        if hasattr(cur, "food_items"):
            return cur
        nxt = getattr(cur, "env_state", None)
        if nxt is None:
            return cur
        cur = nxt
    return cur


def _lbf_grid_size(inner_env):
    """Grid size from the wrapper or underlying jumanji env."""
    wrapper = inner_env._env if hasattr(inner_env, '_env') else inner_env
    jumanji_env = wrapper.env if hasattr(wrapper, 'env') else wrapper
    return jumanji_env._generator.grid_size


def _render_lbf_eval_frames(inner_env, ep_states):
    """Render LBF eval frames using the Jumanji matplotlib viewer for quality."""
    import matplotlib
    matplotlib.use("Agg")
    from jumanji.environments.routing.lbf.viewer import LevelBasedForagingViewer

    viewer = LevelBasedForagingViewer(grid_size=_lbf_grid_size(inner_env),
                                      render_mode="rgb_array")
    frames = []
    for s in ep_states:
        rgba = viewer.render(_to_jumanji_state(s))
        # RGBA -> RGB
        frames.append(rgba[:, :, :3].copy())
    viewer.close()
    return frames
