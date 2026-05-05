"""Visual debugger for scripted-speaker experiments on the card game.

Runs scripted-speaker episodes (one agent forced to emit a fixed token
schedule, the other plays organically) and produces, **per episode, a single
combined figure** with:

  - Top row: speaker's own-frame view at every step, with a dot on the
    speaker's own action.
  - Bottom row: listener's own-frame view at every step, with a dot on the
    listener's own action.
  - Header: scripted condition (constant X or switch X->Y at slot k),
    canonical messages (after inverse-mapping through OP recolouring),
    listener's pick (own-frame and canonical), FOLLOW (listener pick ==
    speaker canonical message), COORD (env's canonical-frame coordination
    reward).
  - Per-step labels below each column: speaker's emitted/picked value and
    listener's emitted/picked value.
  - In switch mode: red marker on column k of the speaker row ("switches to
    Y here") and on column k+1 of the listener row ("first sees Y").

Two modes:
  - constant (default): script = [X] * (max_steps - 1) for each X in --x-values.
  - switch: script = [X]*k + [Y]*(K-k), enabled by --switch X,Y,k.

Usage examples:
    # Constant: 3 episodes for each X in {0,1,2,3,4}, agent 0 as speaker
    ./run_gpu.sh 7 evaluation.visualize_scripted_speaker \\
        --checkpoint /path/to/saved_train_run \\
        --seed-idx 0 --speaker-idx 0 \\
        --x-values 0,1,2,3,4 --num-episodes 3 \\
        --output-dir plots/scripted_viz/<run>/seed_0

    # Switch: 3 episodes of speaker switching from X=2 to Y=4 at slot k=3
    ./run_gpu.sh 7 evaluation.visualize_scripted_speaker \\
        --checkpoint /path/to/saved_train_run \\
        --seed-idx 0 --speaker-idx 0 \\
        --switch 2,4,3 --num-episodes 3 \\
        --output-dir plots/scripted_viz/<run>/seed_0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
import numpy as np
from omegaconf import OmegaConf

from agents.initialize_agents import initialize_ja_agent, initialize_ja_image_agent
from common.save_load_utils import load_train_run
from envs import make_env
from envs.card_game.rendering import NUM_CARDS
from envs.log_wrapper import LogWrapper
from evaluation.analyze_attention import _get_per_agent_info, _render_own_frame


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
    """Run an episode with the speaker scripted; return per-step info."""
    listener_idx = 1 - speaker_idx
    rng = jax.random.fold_in(reset_rng, 1)
    obs, state = env.reset(reset_rng)

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

        action_pair = [0, 0]
        action_pair[speaker_idx] = a_s
        action_pair[listener_idx] = a_l

        if c < max_steps - 1:
            ep_messages.append(tuple(action_pair))
        else:
            ep_actions.append(tuple(action_pair))

        obs, state, _r, _d, info = env.step(k_step, state, env_act)
        ep_states.append(state)

        if c == max_steps - 1:
            coord = int(float(info["base_reward"][0]) > 0)

    return {
        "ep_states": ep_states,
        "ep_actions": ep_actions,
        "ep_messages": ep_messages,
        "listener_pick_visual": listener_pick_visual,
        "listener_inv": np.asarray(listener_inv),
        "speaker_inv": np.asarray(speaker_inv),
        "coord": coord,
    }


def _render_combined(
    result, speaker_idx, listener_idx,
    mode, scripted_X, scripted_Y, switch_k,
    gt_speaker_X, gt_speaker_Y, gt_listener_pick,
    follow, coord,
    output_path, max_steps,
):
    """One figure: speaker view (top row) + listener view (bottom row),
    paired column-by-column with annotations."""
    num_steps = min(len(result["ep_states"]) - 1, max_steps)
    if num_steps == 0:
        return

    fig, axes = plt.subplots(
        2, num_steps,
        figsize=(num_steps * 2.0, 4.6),
    )
    if num_steps == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for t in range(num_steps):
        is_decision = (t == num_steps - 1)
        state = result["ep_states"][t]

        if is_decision:
            speaker_value = int(result["ep_actions"][0][speaker_idx])
            listener_value = int(result["ep_actions"][0][listener_idx])
        else:
            speaker_value = int(result["ep_messages"][t][speaker_idx])
            listener_value = int(result["ep_messages"][t][listener_idx])

        # Speaker (top)
        cps, pps, recs = _get_per_agent_info(state, speaker_idx)
        img_s = _render_own_frame(cps, pps, recs, speaker_idx, speaker_value, is_decision)
        axes[0, t].imshow(img_s, interpolation="nearest")
        axes[0, t].set_xticks([])
        axes[0, t].set_yticks([])

        # Listener (bottom)
        cpl, ppl, recl = _get_per_agent_info(state, listener_idx)
        img_l = _render_own_frame(cpl, ppl, recl, listener_idx, listener_value, is_decision)
        axes[1, t].imshow(img_l, interpolation="nearest")
        axes[1, t].set_xticks([])
        axes[1, t].set_yticks([])

        step_label = "decision" if is_decision else f"step {t}"
        axes[0, t].set_title(step_label, fontsize=9)

        # Bottom labels
        sent_label = (
            f"sent: {speaker_value}" if not is_decision
            else f"pick: {speaker_value}"
        )
        if mode == "switch" and not is_decision:
            tag = "X" if t < switch_k else "Y"
            sent_label = f"sent: {speaker_value} ({tag})"
        axes[0, t].set_xlabel(sent_label, fontsize=9)

        listener_label = (
            f"sent: {listener_value}" if not is_decision
            else f"pick: {listener_value}"
        )
        axes[1, t].set_xlabel(listener_label, fontsize=9, fontweight=(
            "bold" if is_decision else "normal"
        ))

    # Switch markers
    if mode == "switch":
        # Speaker row: red border around column switch_k (where speaker first emits Y)
        if 0 <= switch_k < num_steps:
            ax = axes[0, switch_k]
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
            ax.set_title(f"step {switch_k}\nspeaker switches X->Y",
                         fontsize=9, color="red")
        # Listener row: red border on column switch_k + 1 (where listener first sees Y)
        first_listener_seen = switch_k + 1
        if 0 <= first_listener_seen < num_steps:
            ax = axes[1, first_listener_seen]
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)
            existing_label = ax.get_xlabel()
            ax.set_xlabel(
                f"{existing_label}\nlistener first sees Y",
                fontsize=9, color="red",
            )

    # Row labels using figtext (axes have ticks off)
    fig.text(
        0.005, 0.66, f"speaker (agent_{speaker_idx})",
        rotation=90, va="center", ha="left", fontsize=10, fontweight="bold",
    )
    fig.text(
        0.005, 0.30, f"listener (agent_{listener_idx})",
        rotation=90, va="center", ha="left", fontsize=10, fontweight="bold",
    )

    # Header: condition + outcome
    if mode == "constant":
        cond = (
            f"CONSTANT — speaker forced to say X={scripted_X} every slot  "
            f"|  canonical msg = {gt_speaker_X}"
        )
    else:
        cond = (
            f"SWITCH — X={scripted_X} (canonical {gt_speaker_X}) for slots 0..{switch_k - 1}, "
            f"then Y={scripted_Y} (canonical {gt_speaker_Y}) for slots {switch_k}..{max_steps - 2}"
        )

    follow_str = "✓ FOLLOW" if follow else "✗ no follow"
    coord_str = "✓ COORD" if coord else "✗ no coord"
    out = (
        f"listener pick = {result['listener_pick_visual']} (canonical {gt_listener_pick})  "
        f"|  {follow_str}  |  {coord_str}"
    )

    fig.suptitle(f"{cond}\n{out}", fontsize=10, y=0.995)

    fig.subplots_adjust(left=0.04, right=0.99, top=0.86, bottom=0.10, hspace=0.30, wspace=0.06)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _build_script(mode, X, Y, k, K):
    if mode == "constant":
        return np.full(K, X, dtype=np.int32)
    return np.concatenate([np.full(k, X, dtype=np.int32),
                           np.full(K - k, Y, dtype=np.int32)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed-idx", type=int, default=0)
    parser.add_argument("--speaker-idx", type=int, default=0, choices=(0, 1))
    parser.add_argument("--x-values", default="0,1,2,3,4",
                        help="Constant mode: comma-separated speaker-frame X values.")
    parser.add_argument("--switch", default=None,
                        help="Switch mode: 'X,Y,k' (overrides --x-values).")
    parser.add_argument("--num-episodes", type=int, default=3,
                        help="Episodes per condition.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-rng-base", type=int, default=8000)
    args = parser.parse_args()

    listener_idx = 1 - args.speaker_idx

    if args.switch is not None:
        parts = [int(p) for p in args.switch.split(",")]
        if len(parts) != 3:
            raise SystemExit("--switch must be 'X,Y,k' (three ints)")
        switch_X, switch_Y, switch_k = parts
        mode = "switch"
        x_values = [None]  # one config; loop over episodes only
    else:
        switch_X = switch_Y = switch_k = None
        mode = "constant"
        x_values = [int(x) for x in args.x_values.split(",") if x.strip()]

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
    init_fn = (
        initialize_ja_image_agent if obs_type in ("image", "fov")
        else initialize_ja_agent
    )
    policy, _ = init_fn(alg_config, env_wrapped, jax.random.PRNGKey(0))

    run_data = load_train_run(args.checkpoint)
    final_params = run_data["final_params"]
    num_seeds = jax.tree.leaves(final_params)[0].shape[0]
    if args.seed_idx >= num_seeds:
        raise ValueError(
            f"seed_idx={args.seed_idx} out of range for {num_seeds} seeds"
        )
    params = jax.tree.map(lambda x: x[args.seed_idx], final_params)

    max_steps = int(env_kwargs.get("max_steps", 8))
    K = max_steps - 1  # deliberation slots
    if mode == "switch" and not (1 <= switch_k <= K - 1):
        raise SystemExit(
            f"--switch k={switch_k} out of range; valid range is 1..{K - 1}"
        )

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Visualizing scripted-speaker episodes ({mode} mode)")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  seed_idx: {args.seed_idx}/{num_seeds - 1}")
    print(f"  speaker: agent_{args.speaker_idx}, listener: agent_{listener_idx}")
    if mode == "constant":
        print(f"  x_values: {x_values}")
    else:
        print(f"  switch: X={switch_X}, Y={switch_Y}, k={switch_k}")
    print(f"  num_episodes per condition: {args.num_episodes}")
    print(f"  output: {output_root.resolve()}")

    for X_or_none in x_values:
        if mode == "constant":
            X = X_or_none
            cond_dir = output_root / f"X={X}"
        else:
            X = switch_X
            cond_dir = output_root / f"switch_X{switch_X}_Y{switch_Y}_k{switch_k}"
        cond_dir.mkdir(parents=True, exist_ok=True)
        script = _build_script(mode, X, switch_Y, switch_k, K)

        print(f"\n[{cond_dir.name}]")
        for ep in range(args.num_episodes):
            seed_offset = (X if mode == "constant" else
                           switch_X * 1009 + switch_Y * 53 + switch_k)
            reset_rng = jax.random.PRNGKey(args.episode_rng_base + ep + seed_offset * 10007)

            result = _run_and_collect(
                inner_env, policy, params, reset_rng,
                speaker_idx=args.speaker_idx, script=script, max_steps=max_steps,
            )

            gt_speaker_X = int(result["speaker_inv"][X])
            gt_speaker_Y = (
                int(result["speaker_inv"][switch_Y]) if mode == "switch" else None
            )
            gt_listener_pick = int(
                result["listener_inv"][result["listener_pick_visual"]]
            )
            # FOLLOW for switch is "listener picks the LATEST speaker message" (Y); for
            # constant it is "listener picks X".
            gt_msg_for_follow = (
                gt_speaker_Y if mode == "switch" else gt_speaker_X
            )
            follow = int(gt_listener_pick == gt_msg_for_follow)
            coord = result["coord"]

            ep_dir = cond_dir / f"episode_{ep}"
            ep_dir.mkdir(exist_ok=True)

            _render_combined(
                result,
                speaker_idx=args.speaker_idx, listener_idx=listener_idx,
                mode=mode, scripted_X=X, scripted_Y=switch_Y, switch_k=switch_k,
                gt_speaker_X=gt_speaker_X, gt_speaker_Y=gt_speaker_Y,
                gt_listener_pick=gt_listener_pick,
                follow=follow, coord=coord,
                output_path=ep_dir / "episode.png",
                max_steps=max_steps,
            )

            with (ep_dir / "summary.txt").open("w") as f:
                f.write(f"Scripted-speaker episode (mode={mode})\n")
                f.write(f"  speaker_idx: {args.speaker_idx}\n")
                f.write(f"  listener_idx: {listener_idx}\n")
                if mode == "constant":
                    f.write(f"  scripted X (speaker frame): {X}\n")
                    f.write(f"  GT speaker message: {gt_speaker_X}\n")
                else:
                    f.write(f"  scripted X={switch_X}, Y={switch_Y}, k={switch_k}\n")
                    f.write(f"  GT msg X: {gt_speaker_X}\n")
                    f.write(f"  GT msg Y: {gt_speaker_Y}\n")
                f.write(f"  listener pick (own frame): {result['listener_pick_visual']}\n")
                f.write(f"  GT listener pick: {gt_listener_pick}\n")
                f.write(f"  FOLLOW (canonical match): {bool(follow)}\n")
                f.write(f"  COORD (canonical pick agreement): {bool(coord)}\n")
                f.write(f"  speaker_recolouring_inv: {result['speaker_inv'].tolist()}\n")
                f.write(f"  listener_recolouring_inv: {result['listener_inv'].tolist()}\n")
                f.write(f"  ep_messages (deliberation, own frame per agent):\n")
                for t, (a0, a1) in enumerate(result["ep_messages"]):
                    f.write(f"    step {t}: a0={a0}  a1={a1}\n")
                for (a0, a1) in result["ep_actions"]:
                    f.write(f"  decision: a0={a0}  a1={a1}\n")

            tag = "FOLLOW" if follow else "no follow"
            ctag = "COORD" if coord else "no coord"
            print(
                f"  ep {ep}: GT_msg={gt_msg_for_follow}  "
                f"GT_pick={gt_listener_pick}  [{tag}, {ctag}]  "
                f"-> {ep_dir.relative_to(output_root)}/episode.png"
            )

    print(f"\nDone. Figures in {output_root.resolve()}/")


if __name__ == "__main__":
    main()
