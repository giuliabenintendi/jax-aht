"""Per-step card-level JSD trajectories for card-game checkpoints.

For a chosen cross-play (or self-play) pairing, rolls out `num_episodes`
episodes and records, at every deliberation/decision step, the
Jensen-Shannon divergence between the two agents' attention distributions
over the five cards.

The two agents see different Other-Play scenes, so the comparison is done in
the shared ground-truth card-identity frame: each agent's spatial attention is
pooled to a 5-card mass via the fixed card masks, mapped to GT-card space
through that agent's OP position permutation (matching the training-time
`JA_CARD_*` bookkeeping in `run_episode_with_states`), then renormalized over
the five cards. Convergent joint attention drives the JSD toward 0; agents
that lock onto different cards sit near ln 2.

Writes `<name>.npz` (`jsd` of shape `(num_episodes, max_steps)`) for
`plot_card_jsd.py` to render.

Usage:
    ./run_gpu.sh 0 evaluation.card_game.eval_card_jsd \\
        --checkpoint <path_to_saved_train_run> \\
        --name op_only --xp-pair 0,1 --num-episodes 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.ja_utils import build_card_masks, jsd_divergence
from evaluation.card_game._card_game_utils import load_card_game_eval
from evaluation.card_game.eval_time_to_agree import (
    FEAT_H,
    FEAT_W,
    IMG_H,
    IMG_W,
    setup_card_feed,
)
from evaluation.vis_episodes import (
    _action_to_ground_truth,
    _get_card_game_position_perm,
)

NUM_CARDS = 5


def _build_jsd_runner(ev, greedy: bool, feed_dim: int, feed_active: bool):
    """Compiled batched rollout returning per-step card-level JSD.

    Attention is always collected (both agents are JA networks), independently
    of whether the policy is fed the partner's card attention: `feed_active`
    only controls the obs augmentation the MATE policy expects.
    """
    T = ev.max_steps
    card_masks = jnp.asarray(build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W))
    needs_prev_io = bool(getattr(ev.policy, "uses_prev_reward_action", False))
    if not hasattr(ev.policy, "get_action_and_attention"):
        raise ValueError("Card-JSD eval requires a policy with get_action_and_attention().")

    def _run_one_episode(rng, params_a, params_b):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = ev.env.reset(reset_rng)
        done = {k: jnp.zeros((), dtype=bool) for k in ev.env.agents + ["__all__"]}

        hstate_0 = ev.policy.init_hstate(1)
        hstate_1 = ev.policy.init_hstate(1)
        prev_reward_0 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_reward_1 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_action_0 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_action_1 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_pca_0 = jnp.zeros((feed_dim,), dtype=jnp.float32)
        prev_pca_1 = jnp.zeros((feed_dim,), dtype=jnp.float32)

        init_carry = (
            obs, env_state, done, rng, hstate_0, hstate_1,
            prev_reward_0, prev_reward_1, prev_action_0, prev_action_1,
            prev_pca_0, prev_pca_1,
        )

        def _take_step(carry, _step_idx):
            (
                obs, env_state, done, rng, hstate_0, hstate_1,
                prev_reward_0, prev_reward_1, prev_action_0, prev_action_1,
                prev_pca_0, prev_pca_1,
            ) = carry

            avail_actions = jax.lax.stop_gradient(ev.env.get_avail_actions(env_state))
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            obs_0 = obs["agent_0"]
            obs_1 = obs["agent_1"]
            done_0 = done["agent_0"].reshape(1, 1)
            done_1 = done["agent_1"].reshape(1, 1)

            if feed_active:
                obs_0 = jnp.concatenate([obs_0, prev_pca_0])
                obs_1 = jnp.concatenate([obs_1, prev_pca_1])
            obs_0 = obs_0.reshape(1, 1, -1)
            obs_1 = obs_1.reshape(1, 1, -1)

            rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)

            act_0, hstate_0_next, attn_0 = ev.policy.get_action_and_attention(
                params=params_a, obs=obs_0, done=done_0,
                avail_actions=avail_actions_0, hstate=hstate_0, rng=act_rng,
                greedy=greedy, agent_id=0,
                prev_reward=prev_reward_0 if needs_prev_io else None,
                prev_action=prev_action_0 if needs_prev_io else None,
            )
            act_1, hstate_1_next, attn_1 = ev.policy.get_action_and_attention(
                params=params_b, obs=obs_1, done=done_1,
                avail_actions=avail_actions_1, hstate=hstate_1, rng=part_rng,
                greedy=greedy, agent_id=1,
                prev_reward=prev_reward_1 if needs_prev_io else None,
                prev_action=prev_action_1 if needs_prev_io else None,
            )

            act_0 = act_0.squeeze()
            act_1 = act_1.squeeze()
            env_act = {"agent_0": act_0, "agent_1": act_1}

            # Pool spatial attention to per-card mass, then map both agents into
            # the shared GT-card frame via their OP position permutations.
            p0 = _get_card_game_position_perm(env_state, "agent_0")
            p1 = _get_card_game_position_perm(env_state, "agent_1")
            ca0 = jnp.einsum("hw,chw->c", attn_0.squeeze(), card_masks)
            ca1 = jnp.einsum("hw,chw->c", attn_1.squeeze(), card_masks)
            ph0 = jnp.zeros(NUM_CARDS).at[p0].set(ca0)
            ph1 = jnp.zeros(NUM_CARDS).at[p1].set(ca1)
            d0 = ph0 / jnp.maximum(ph0.sum(), 1e-8)
            d1 = ph1 / jnp.maximum(ph1.sum(), 1e-8)
            jsd = jsd_divergence(d0.reshape(1, NUM_CARDS), d1.reshape(1, NUM_CARDS)).reshape(())

            obs_next, env_state_next, reward, done_next, _ = ev.env.step(
                step_rng, env_state, env_act
            )

            if needs_prev_io:
                prev_reward_0_next = reward["agent_0"].reshape(1, 1).astype(jnp.float32)
                prev_reward_1_next = reward["agent_1"].reshape(1, 1).astype(jnp.float32)
                prev_action_0_next = act_0.reshape(1, 1).astype(jnp.float32)
                prev_action_1_next = act_1.reshape(1, 1).astype(jnp.float32)
            else:
                prev_reward_0_next, prev_reward_1_next = prev_reward_0, prev_reward_1
                prev_action_0_next, prev_action_1_next = prev_action_0, prev_action_1

            if feed_active:
                prev_pca_0_next = ph1[p0]
                prev_pca_1_next = ph0[p1]
            else:
                prev_pca_0_next, prev_pca_1_next = prev_pca_0, prev_pca_1

            next_carry = (
                obs_next, env_state_next, done_next, rng, hstate_0_next, hstate_1_next,
                prev_reward_0_next, prev_reward_1_next, prev_action_0_next,
                prev_action_1_next, prev_pca_0_next, prev_pca_1_next,
            )
            return next_carry, jsd

        def _scan_step(carry, step_idx):
            done_all = carry[2]["__all__"].reshape(())
            return jax.lax.cond(
                done_all,
                lambda c: (c, jnp.nan),
                lambda c: _take_step(c, step_idx),
                carry,
            )

        _, jsd_trace = jax.lax.scan(_scan_step, init_carry, jnp.arange(T))
        return jsd_trace

    return jax.jit(jax.vmap(_run_one_episode, in_axes=(0, None, None)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--xp-pair", default="0,1",
                        help="'i,j' — agent 0 uses seed i, agent 1 uses seed j. "
                             "Pass 'i,i' for self-play.")
    parser.add_argument("--name", default=None,
                        help="Output basename. Defaults to the run label slug.")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to evaluation/card_game/card_jsd_data/")
    parser.add_argument("--rng-base", type=int, default=7_000_000)
    parser.add_argument("--sampled", action="store_true",
                        help="Sampled actions (default greedy).")
    args = parser.parse_args()

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ev = load_card_game_eval(args.checkpoint)
    greedy = not args.sampled
    T = ev.max_steps
    scalar_dim = int(getattr(ev.policy.network, "scalar_dim", 0))
    feed_active = scalar_dim > 0
    feed_dim, _ = setup_card_feed(ev)

    i, j = (int(x) for x in args.xp_pair.split(","))
    print(
        f"\nCheckpoint: {args.checkpoint}"
        f"\n  label={ev.label}  seeds={ev.num_seeds}  max_steps={T}"
        f"  greedy={greedy}  feed_active={feed_active}"
        f"\n  pair=({i},{j})  episodes={args.num_episodes}"
    )

    runner = _build_jsd_runner(ev, greedy, feed_dim, feed_active)
    params_i = jax.tree.map(lambda x: x[i], ev.params)
    params_j = jax.tree.map(lambda x: x[j], ev.params)
    rngs = jax.random.split(jax.random.PRNGKey(args.rng_base), args.num_episodes)
    jsd = np.asarray(runner(rngs, params_i, params_j))  # (num_episodes, T)

    mean = np.nanmean(jsd, axis=0)
    sem = np.nanstd(jsd, axis=0) / np.sqrt(np.sum(~np.isnan(jsd), axis=0))
    print("\n  step:     " + "  ".join(f"k{k}" for k in range(T)))
    print("  mean JSD: " + "  ".join(f"{v:.3f}" for v in mean))
    print("  sem:      " + "  ".join(f"{v:.3f}" for v in sem))

    name = args.name or ev.label.replace("/", "_").replace(" ", "_")
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(__file__).parent / "card_jsd_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.npz"
    np.savez(out_path, jsd=jsd, max_steps=T, pair=np.array([i, j]), label=ev.label)
    print(f"\nSaved {out_path}")


## Tests

def _test_jsd_endpoints() -> None:
    """Same-card distributions give JSD 0; disjoint single cards give ln 2."""
    same = jsd_divergence(
        jnp.array([[0.0, 1.0, 0.0, 0.0, 0.0]]),
        jnp.array([[0.0, 1.0, 0.0, 0.0, 0.0]]),
    ).reshape(())
    disjoint = jsd_divergence(
        jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        jnp.array([[0.0, 1.0, 0.0, 0.0, 0.0]]),
    ).reshape(())
    assert float(same) < 1e-5
    assert abs(float(disjoint) - float(jnp.log(2.0))) < 1e-3


if __name__ == "__main__":
    main()
