"""Visual debugger for scripted-speaker experiments on the card game.

Runs a small number of scripted-speaker episodes (one agent forced to emit a
constant message X, the other plays organically) and renders each agent's
own-frame view across the deliberation+decision steps. Annotates each
episode with the scripted X (speaker frame), the GT message after
inverse-mapping through OP recolouring, the listener's pick (own frame +
GT), and whether canonical coordination was achieved.

Use this to eyeball what the listener actually does when the speaker emits a
fixed token, instead of staring at aggregate statistics.

Usage:
    ./run_gpu.sh 7 evaluation.visualize_scripted_speaker \\
        --checkpoint /path/to/saved_train_run \\
        --seed-idx 0 --speaker-idx 0 \\
        --x-values 0,1,2,3,4 \\
        --num-episodes 3 \\
        --output-dir plots/scripted_viz/<run_name>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import NUM_CARDS
from envs.log_wrapper import LogWrapper
from evaluation.analyze_attention import _render_obs_sequence


def _get_obs_type(alg_config):
    return alg_config.get(
        "OBS_TYPE",
        alg_config.get("ENV_KWARGS", {}).get("obs_type", "symbolic"),
    )


def _greedy_action(policy, params, obs_dict, agent_id, hstate, avail, rng):
    obs = obs_dict[f"agent_{agent_id}"].reshape(1, 1, -1)
    done = jnp.zeros((1, 1), dtype=bool)
    avail_a = avail[f"agent_{agent_id}"].astype(jnp.float32)
    action, new_hstate = policy.get_action(
        params=params, obs=obs, done=done, avail_actions=avail_a,
        hstate=hstate, rng=rng, greedy=True,
    )
    return int(action.squeeze()), new_hstate


def _run_and_collect(env, policy, params, reset_rng, speaker_idx, script,
                     max_steps):
    """Run a scripted-speaker episode, return ep_states, ep_actions, ep_messages
    plus the OP wrapper inverse perms captured at episode start.
    """
    listener_idx = 1 - speaker_idx
    rng = jax.random.fold_in(reset_rng, 1)
    obs, state = env.reset(reset_rng)

    # Capture per-agent recolouring inverse perms from the wrapper state.
    listener_inv = state.per_agent_inv_recolouring[f"agent_{listener_idx}"]
    speaker_inv = state.per_agent_inv_recolouring[f"agent_{speaker_idx}"]

    h_speaker = policy.init_hstate(1)
    h_listener = policy.init_hstate(1)

    ep_states = [state]
    ep_actions: list = []
    ep_messages: list = []

    listener_pick_visual = -1
    coord = 0

    for c in range(max_steps):
        avail = env.get_avail_actions(state)
        rng, k_s, k_l, k_step = jax.random.split(rng, 4)

        a_s_sampled, h_speaker = _greedy_action(
            policy, params, obs, speaker_idx, h_speaker, avail, k_s,
        )
        a_l, h_listener = _greedy_action(
            policy, params, obs, listener_idx, h_listener, avail, k_l,
        )

        if c < max_steps - 1:
            a_s = int(script[c])
        else:
            a_s = a_s_sampled
            listener_pick_visual = a_l

        env_act = {
            f"agent_{speaker_idx}": jnp.int32(a_s),
            f"agent_{listener_idx}": jnp.int32(a_l),
        }

        # Per-agent action arrays for the rendering helpers (speaker first slot
        # = agent 0, listener slot = agent 1, mapped via index).
        action_pair = [0, 0]
        action_pair[speaker_idx] = a_s
        action_pair[listener_idx] = a_l

        if c < max_steps - 1:
            ep_messages.append(tuple(action_pair))
        else:
            ep_actions.append(tuple(action_pair))

        obs, state, reward, _d, info = env.step(k_step, state, env_act)
        ep_states.append(state)

        if c == max_steps - 1:
            coord = int(float(info["base_reward"][0]) > 0)

    # Pad ep_actions with the final pick at all post-deliberation indices so
    # _render_obs_sequence can index the decision step uniformly.
    # Existing _render_obs_sequence uses ep_actions[t] only on the decision step
    # (t == num_steps - 1) and ep_messages[t] otherwise.
    return {
        "ep_states": ep_states,
        "ep_actions": ep_actions,
        "ep_messages": ep_messages,
        "listener_pick_visual": listener_pick_visual,
        "listener_inv": np.asarray(listener_inv),
        "speaker_inv": np.asarray(speaker_inv),
        "coord": coord,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--speaker-idx", type=int, default=0,
                        choices=(0, 1))
    parser.add_argument("--x-values", default="0,1,2,3,4",
                        help="Comma-separated speaker-frame action ints to script.")
    parser.add_argument("--num-episodes", type=int, default=3,
                        help="Episodes per X value.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-rng-base", type=int, default=8000)
    args = parser.parse_args()

    x_values = [int(x) for x in args.x_values.split(",") if x.strip()]
    listener_idx = 1 - args.speaker_idx

    run_dir = os.path.dirname(args.checkpoint)
    cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml")),
        resolve=True,
    )
    alg_config = cfg["algorithm"]
    env_kwargs = dict(alg_config.get("ENV_KWARGS", {}))
    if alg_config.get("COMMUNICATION", False):
        env_kwargs["communication"] = True
    if not env_kwargs.get("communication", False):
        raise SystemExit(
            "Checkpoint trained without communication; scripted-speaker probe undefined."
        )
    env_kwargs["scramble_partner_msg"] = False

    env = make_env(alg_config["ENV_NAME"], env_kwargs)
    inner_env = env
    env_wrapped = LogWrapper(env)

    obs_type = _get_obs_type(alg_config)
    init_fn = initialize_ja_image_agent if obs_type in ("image", "fov") else initialize_ja_agent
    policy, _ = init_fn(alg_config, env_wrapped, jax.random.PRNGKey(0))

    run_data = load_train_run(args.checkpoint)
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    if args.seed_idx >= num_seeds:
        raise ValueError(
            f"seed_idx={args.seed_idx} out of range for {num_seeds} seeds in checkpoint"
        )
    params = jax.tree.map(lambda x: x[args.seed_idx], final_params)

    max_steps = int(env_kwargs.get("max_steps", 8))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing scripted-speaker episodes")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  seed_idx: {args.seed_idx}/{num_seeds - 1}")
    print(f"  speaker: agent_{args.speaker_idx}, listener: agent_{listener_idx}")
    print(f"  x_values: {x_values}")
    print(f"  num_episodes per X: {args.num_episodes}")
    print(f"  output: {output_root.resolve()}")

    for X in x_values:
        x_dir = output_root / f"X={X}"
        x_dir.mkdir(parents=True, exist_ok=True)
        script = np.full(max_steps - 1, X, dtype=np.int32)

        print(f"\n[X={X}]")
        for ep in range(args.num_episodes):
            reset_rng = jax.random.PRNGKey(args.episode_rng_base + ep + X * 10007)
            result = _run_and_collect(
                inner_env, policy, params, reset_rng,
                speaker_idx=args.speaker_idx, script=script, max_steps=max_steps,
            )

            gt_speaker_msg = int(result["speaker_inv"][X])
            gt_listener_pick = int(result["listener_inv"][result["listener_pick_visual"]])
            follow = int(gt_listener_pick == gt_speaker_msg)
            coord = result["coord"]

            ep_dir = x_dir / f"episode_{ep}"
            ep_dir.mkdir(exist_ok=True)

            # Render each agent's own-frame view sequence with own-action dot.
            _render_obs_sequence(
                result["ep_states"], result["ep_actions"], result["ep_messages"],
                args.speaker_idx,
                output_path=ep_dir / "speaker_view.png",
                max_steps=max_steps,
            )
            _render_obs_sequence(
                result["ep_states"], result["ep_actions"], result["ep_messages"],
                listener_idx,
                output_path=ep_dir / "listener_view.png",
                max_steps=max_steps,
            )

            with (ep_dir / "summary.txt").open("w") as f:
                f.write(f"Scripted-speaker episode (X={X})\n")
                f.write(f"  speaker_idx: {args.speaker_idx}\n")
                f.write(f"  listener_idx: {listener_idx}\n")
                f.write(f"  scripted X (speaker frame): {X}\n")
                f.write(f"  GT speaker message (after inverse-map): {gt_speaker_msg}\n")
                f.write(f"  listener pick (listener frame): {result['listener_pick_visual']}\n")
                f.write(f"  GT listener pick (after inverse-map): {gt_listener_pick}\n")
                f.write(f"  listener follows speaker (GT match): {bool(follow)}\n")
                f.write(f"  canonical coordination (pick_0==pick_1 in GT): {bool(coord)}\n")
                f.write(f"  speaker_recolouring_inv (full): {result['speaker_inv'].tolist()}\n")
                f.write(f"  listener_recolouring_inv (full): {result['listener_inv'].tolist()}\n")
                f.write(f"  ep_messages (per-step (a0, a1) in agent's own frame):\n")
                for t, (a0, a1) in enumerate(result["ep_messages"]):
                    f.write(f"    step {t}: a0={a0}  a1={a1}\n")
                f.write(f"  ep_actions (decision step (a0, a1) in agent's own frame):\n")
                for t, (a0, a1) in enumerate(result["ep_actions"]):
                    f.write(f"    decision step: a0={a0}  a1={a1}\n")

            tag = "FOLLOW" if follow else "no follow"
            ctag = "COORD" if coord else "no coord"
            print(f"  ep {ep}: GT_msg={gt_speaker_msg}  GT_pick={gt_listener_pick}  "
                  f"[{tag}, {ctag}]  -> {ep_dir.relative_to(output_root)}/")

    print(f"\nDone. Figures in {output_root.resolve()}/")


if __name__ == "__main__":
    main()
