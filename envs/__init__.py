import copy
import numpy as np

# jumanji is imported lazily inside the 'lbf' branch of make_env. Importing it
# at module level forces a jax-array construction in jumanji/types.py at
# import time, which initialises the JAX backend before any user code runs —
# any cuDNN/GPU misconfiguration then aborts pytest collection for unrelated
# tests (e.g. Hanabi). Keeping it lazy isolates lbf's import cost to lbf.

def process_default_args(env_kwargs: dict, default_args: dict):
    '''Helper function to process generator and viewer args for Jumanji environments. 
    If env_args and default_args have any key overlap, overwrite 
    args in default_args with those in env_args, deleting those in env_args
    '''
    env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
    default_args_copy = dict(copy.deepcopy(default_args))
    for key in env_kwargs:
        if key in default_args:
            default_args_copy[key] = env_kwargs[key]
            del env_kwargs_copy[key]
    return default_args_copy, env_kwargs_copy

def make_env(env_name: str, env_kwargs: dict = {}):
    if env_name == 'lbf':
        import jumanji
        from jumanji.environments.routing.lbf.generator import RandomGenerator as LbfGenerator

        default_generator_args = {
            "grid_size": 7,
            "fov": 7,
            "num_agents": 2,
            "num_food": 3,
            "max_agent_level": 2,
            "force_coop": True,
        }
        default_viewer_args = {"highlight_agent_idx": 0} # None to disable highlighting

        from envs.lbf.lbf_wrapper import LBFWrapper
        from envs.lbf.adhoc_lbf_viewer import AdHocLBFViewer

        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        obs_type = env_kwargs_copy.pop("obs_type", "symbolic")
        op_mirror = env_kwargs_copy.pop("other_play_mirror", False)
        op_rotation = env_kwargs_copy.pop("other_play_rotation", False)
        if op_rotation:
            raise ValueError(
                "LBF rotation Other-Play has been removed; use other_play_mirror instead."
            )

        generator_args, env_kwargs_copy = process_default_args(env_kwargs_copy, default_generator_args)
        # Full observability: fov must equal grid_size
        generator_args["fov"] = generator_args["grid_size"]
        viewer_args, env_kwargs_copy = process_default_args(env_kwargs_copy, default_viewer_args)
        jumanji_env = jumanji.make('LevelBasedForaging-v0',
                            generator=LbfGenerator(**generator_args),
                            **env_kwargs_copy,
                            viewer=AdHocLBFViewer(grid_size=generator_args["grid_size"],
                                                  **viewer_args))

        if obs_type == 'image':
            from envs.lbf.lbf_image_wrapper import LBFImageWrapper
            env = LBFImageWrapper(jumanji_env, share_rewards=True)
        else:
            env = LBFWrapper(jumanji_env, share_rewards=True)

        if op_mirror:
            if obs_type != 'image':
                raise ValueError("LBF Other-Play requires obs_type=image (geometric pixel transform)")
            from envs.lbf.other_play import LBFMirrorOtherPlayWrapper
            env = LBFMirrorOtherPlayWrapper(env)
        
    elif env_name == 'overcooked-v1':
        default_env_kwargs = {
            "random_reset": True,
            "random_obj_state": False,
            "max_steps": 400
        }
        
        # preprocess env_kwargs to maintain compatibility with symmetric reward shaping
        if "reward_shaping_params" in env_kwargs:
            for param in env_kwargs["reward_shaping_params"]:
                payload = env_kwargs["reward_shaping_params"][param]
                if type(payload) == int or type(payload) == float:
                    # turn the param into symmetric form
                    env_kwargs["reward_shaping_params"][param] = [payload, payload] 
                elif type(payload) == tuple or type(payload) == list:
                    # this is the correct format
                    pass 
                else:
                    print(f"\n[Environment Instantiation Error] {type(payload)} is not valid type as a reward shaping parameter for {param}.\n")
                    exit()

        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        # add default args that are not already in env_kwargs
        for key in default_env_kwargs:
            if key not in env_kwargs:
                env_kwargs_copy[key] = default_env_kwargs[key]

        from envs.overcooked.augmented_layouts import augmented_layouts

        layout = augmented_layouts[env_kwargs['layout']]
        env_kwargs_copy["layout"] = layout

        obs_type = env_kwargs_copy.pop("obs_type", "symbolic")
        # Strip deprecated Overcooked PO/FOV kwargs so old configs still load.
        deprecated_obs_keys = (
            "po_mode",
            "fov_range",
            "fov_slope",
            "use_occlusion",
            "soft_view",
            "dist_sigma",
            "ang_sigma",
        )
        for k in deprecated_obs_keys:
            env_kwargs_copy.pop(k, None)

        if obs_type == "image":
            from envs.overcooked.overcooked_image_wrapper import OvercookedImageWrapper
            env = OvercookedImageWrapper(**env_kwargs_copy)
        else:
            from envs.overcooked.overcooked_wrapper import OvercookedWrapper
            env = OvercookedWrapper(**env_kwargs_copy)
    
    elif env_name == 'overcooked-v2':
        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        obs_type = env_kwargs_copy.pop("obs_type", "image")
        # Wrapper-only kwargs, not accepted by OvercookedV2.
        tile_size = env_kwargs_copy.pop("tile_size", None)
        do_reward_shaping = env_kwargs_copy.pop("do_reward_shaping", True)

        if obs_type != "image":
            raise NotImplementedError(
                "overcooked-v2 currently only supports obs_type=image."
            )

        from envs.overcooked_v2.overcooked_v2_image_wrapper import (
            OvercookedV2ImageWrapper,
        )
        from envs.overcooked_v2.rendering import TILE_PIXELS

        env = OvercookedV2ImageWrapper(
            tile_size=tile_size if tile_size is not None else TILE_PIXELS,
            do_reward_shaping=do_reward_shaping,
            **env_kwargs_copy,
        )
        # Other-Play: the base env samples per-agent ingredient permutations into
        # state.ingredient_permutations; the image wrapper renders each agent's
        # view with a correspondingly permuted palette, so declaring the kwarg
        # is all that activates OP.

    elif env_name == 'card-game':
        from envs.card_game.card_game import CardGameEnv
        env_kwargs = dict(env_kwargs)
        op_pos = env_kwargs.pop('other_play_position_shuffle', False)
        op_recolour = env_kwargs.pop('other_play_recolouring', False)
        if op_pos:
            env_kwargs['shuffle'] = False
        env = CardGameEnv(**env_kwargs)
        if op_pos:
            from envs.card_game.other_play import CardGamePositionShuffleWrapper
            env = CardGamePositionShuffleWrapper(env)
        if op_recolour:
            from envs.card_game.other_play import CardGameRecolouringWrapper
            env = CardGameRecolouringWrapper(env)

    elif env_name == 'hanabi':
        default_env_kwargs = {
            "num_agents": 2,
            "num_colors": 5,
            "num_ranks": 5,
            "max_info_tokens": 8,
            "max_life_tokens": 3,
            "num_cards_of_rank": np.array([3, 2, 2, 2, 1]),
        }

        env_kwargs_copy = dict(copy.deepcopy(env_kwargs))
        obs_type = env_kwargs_copy.pop("obs_type", "symbolic")
        op_recolour = env_kwargs_copy.pop("other_play_recolouring", False)
        # caller kwargs override defaults
        merged_kwargs = {**default_env_kwargs, **env_kwargs_copy}

        if obs_type == "image":
            from envs.hanabi.hanabi_image_wrapper import HanabiImageWrapper
            env = HanabiImageWrapper(**merged_kwargs)
        else:
            from envs.hanabi.hanabi_wrapper import HanabiWrapper
            env = HanabiWrapper(**merged_kwargs)
        if op_recolour:
            if obs_type != "image":
                raise ValueError(
                    "hanabi other_play_recolouring requires obs_type=image "
                    "(pixel-based recolouring; the symbolic obs has no colour pixels)."
                )
            from envs.hanabi.other_play import HanabiColourPermutationWrapper
            env = HanabiColourPermutationWrapper(env)

    else:
        raise NotImplementedError(f"Environment {env_name} not implemented in make_env.")
    
    return env

if __name__ == "__main__":
    # sanity check: test environment creation
    env = make_env('lbf', {'num_agents': 3, 'grid_size': 9})
    print(env)
    env = make_env('overcooked-v1', {'layout': 'cramped_room'})
    print(env)
    env = make_env('hanabi', {'num_agents': 2})
    print(env)
