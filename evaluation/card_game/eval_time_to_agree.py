"""Time-to-agree analysis for card-game checkpoints.

Measures how many deliberation steps the two agents take to converge on a
shared card, in self-play (SP) vs cross-play (XP). Works for both the
implicit-communication runs (deliberation actions are emitted but unseen by
the partner) and the explicit-communication run (deliberation actions are
messages drawn into the partner's observation).

For each episode the two agents have a per-step *intended card* in the
canonical (un-OP'd) frame. The episode's `settle step` is the earliest
deliberation step after which the two intentions agree and stay agreed
through the decision step; a decision-step disagreement is a coordination
failure (settle step = max_steps, i.e. "never").

Writes `<name>.npz` (per-episode agreement trajectories + settle steps,
split SP/XP) for `plot_time_to_agree.py` to render.

Usage:
    ./run_gpu.sh 0 evaluation.card_game.eval_time_to_agree \\
        --checkpoint <path_to_saved_train_run> \\
        --name op_ja_shaping \\
        --num-episodes 128
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.ja_utils import build_card_masks
from evaluation.card_game._card_game_utils import load_card_game_eval
from evaluation.vis_episodes import (
    _action_to_ground_truth,
    _get_card_game_position_perm,
)

IMG_H, IMG_W = 21, 35
FEAT_H, FEAT_W = 6, 9


def setup_card_feed(ev) -> tuple[int, jnp.ndarray | None]:
    """Return `(partner_feed_dim, card_masks)` for `run_episode_with_states`.

    Card-attention augmentation is applied only when the policy network has a
    scalar-embedding head (`scalar_dim > 0`); otherwise the observation must
    be left unaugmented or the policy input shape will not match.
    """
    scalar_dim = int(getattr(ev.policy.network, "scalar_dim", 0))
    partner_feed_dim = scalar_dim if scalar_dim > 0 else 5
    card_masks = (
        jnp.asarray(build_card_masks(IMG_H, IMG_W, FEAT_H, FEAT_W))
        if scalar_dim > 0 else None
    )
    return partner_feed_dim, card_masks


def extract_intentions(
    ep_actions: list[tuple[int, int]],
    ep_messages: list[tuple[int, int]],
    max_steps: int,
) -> np.ndarray:
    """Per-step canonical intended card for both agents, shape `(max_steps, 2)`.

    Unifies the two recording conventions of `run_episode_with_states`:
    no-comm runs put every step's emitted card in `ep_actions`; the comm run
    puts the deliberation cards in `ep_messages` and only the decision pick in
    `ep_actions`. Both are already in the canonical (un-OP'd) frame.
    """
    T = max_steps
    intents = np.full((T, 2), -1, dtype=np.int64)
    if ep_messages:
        for k in range(min(T - 1, len(ep_messages))):
            intents[k] = ep_messages[k]
        intents[T - 1] = ep_actions[T - 1]
    else:
        for k in range(min(T, len(ep_actions))):
            intents[k] = ep_actions[k]
    return intents


def episode_agreement(intents: np.ndarray) -> tuple[np.ndarray, int]:
    """Return `(agree, settle_step)` for one episode.

    `agree` is a `(T,)` bool array — do the two intentions match at each step.
    `settle_step` is the earliest k after which `agree` holds through the
    decision step, or T ("never") when the decision step disagrees.
    """
    a0, a1 = intents[:, 0], intents[:, 1]
    agree = (a0 == a1) & (a0 >= 0) & (a1 >= 0)
    T = agree.shape[0]
    settle = T
    for k in range(T - 1, -1, -1):
        if agree[k]:
            settle = k
        else:
            break
    return agree, settle


def _build_pair_runner(ev, greedy: bool, partner_feed_dim: int, card_masks):
    """Build a compiled batched rollout that returns per-step intentions.

    The analysis only needs the canonical intended card per step, so this
    skips the rendering-oriented Python bookkeeping in `run_episode_with_states`
    and returns a dense `(num_episodes, T, 2)` array directly.
    """
    T = ev.max_steps
    needs_prev_io = (
        getattr(ev.policy, "uses_prev_reward_action", False)
        and getattr(ev.policy, "uses_prev_reward_action", False)
    )
    needs_card_attention = card_masks is not None
    has_attention_api = hasattr(ev.policy, "get_action_and_attention")
    if needs_card_attention and not has_attention_api:
        raise ValueError("JA card-attention eval requires get_action_and_attention().")

    def _run_one_episode(rng, params_a, params_b):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = ev.env.reset(reset_rng)
        done = {k: jnp.zeros((1,), dtype=bool) for k in ev.env.agents + ["__all__"]}

        hstate_0 = ev.policy.init_hstate(1)
        hstate_1 = ev.policy.init_hstate(1)
        prev_reward_0 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_reward_1 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_action_0 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_action_1 = jnp.zeros((1, 1), dtype=jnp.float32)
        prev_pca_0 = jnp.zeros((partner_feed_dim,), dtype=jnp.float32)
        prev_pca_1 = jnp.zeros((partner_feed_dim,), dtype=jnp.float32)

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

            avail_actions = ev.env.get_avail_actions(env_state)
            avail_actions = jax.lax.stop_gradient(avail_actions)
            avail_actions_0 = avail_actions["agent_0"].astype(jnp.float32)
            avail_actions_1 = avail_actions["agent_1"].astype(jnp.float32)

            obs_0 = obs["agent_0"]
            obs_1 = obs["agent_1"]
            done_0 = done["agent_0"].reshape(1, 1)
            done_1 = done["agent_1"].reshape(1, 1)

            if needs_card_attention:
                obs_0 = jnp.concatenate([obs_0, prev_pca_0])
                obs_1 = jnp.concatenate([obs_1, prev_pca_1])

            obs_0 = obs_0.reshape(1, 1, -1)
            obs_1 = obs_1.reshape(1, 1, -1)

            rng, act_rng, part_rng, step_rng = jax.random.split(rng, 4)

            if needs_card_attention:
                act_0, hstate_0_next, attn_0 = ev.policy.get_action_and_attention(
                    params=params_a,
                    obs=obs_0,
                    done=done_0,
                    avail_actions=avail_actions_0,
                    hstate=hstate_0,
                    rng=act_rng,
                    greedy=greedy,
                    agent_id=0,
                    prev_reward=prev_reward_0 if needs_prev_io else None,
                    prev_action=prev_action_0 if needs_prev_io else None,
                )
                act_1, hstate_1_next, attn_1 = ev.policy.get_action_and_attention(
                    params=params_b,
                    obs=obs_1,
                    done=done_1,
                    avail_actions=avail_actions_1,
                    hstate=hstate_1,
                    rng=part_rng,
                    greedy=greedy,
                    agent_id=1,
                    prev_reward=prev_reward_1 if needs_prev_io else None,
                    prev_action=prev_action_1 if needs_prev_io else None,
                )
            else:
                extra_kwargs_0 = (
                    {"prev_reward": prev_reward_0, "prev_action": prev_action_0}
                    if needs_prev_io else {}
                )
                extra_kwargs_1 = (
                    {"prev_reward": prev_reward_1, "prev_action": prev_action_1}
                    if needs_prev_io else {}
                )
                act_0, hstate_0_next = ev.policy.get_action(
                    params=params_a,
                    obs=obs_0,
                    done=done_0,
                    avail_actions=avail_actions_0,
                    hstate=hstate_0,
                    rng=act_rng,
                    greedy=greedy,
                    **extra_kwargs_0,
                )
                act_1, hstate_1_next = ev.policy.get_action(
                    params=params_b,
                    obs=obs_1,
                    done=done_1,
                    avail_actions=avail_actions_1,
                    hstate=hstate_1,
                    rng=part_rng,
                    greedy=greedy,
                    **extra_kwargs_1,
                )

            act_0 = act_0.squeeze()
            act_1 = act_1.squeeze()
            env_act = {"agent_0": act_0, "agent_1": act_1}
            render_act = _action_to_ground_truth(ev.env, env_state, env_act)
            intents = jnp.stack(
                [render_act["agent_0"], render_act["agent_1"]], axis=0
            ).astype(jnp.int32)

            obs_next, env_state_next, reward, done_next, _ = ev.env.step(
                step_rng, env_state, env_act
            )

            if needs_prev_io:
                prev_reward_0_next = reward["agent_0"].reshape(1, 1).astype(jnp.float32)
                prev_reward_1_next = reward["agent_1"].reshape(1, 1).astype(jnp.float32)
                prev_action_0_next = act_0.reshape(1, 1).astype(jnp.float32)
                prev_action_1_next = act_1.reshape(1, 1).astype(jnp.float32)
            else:
                prev_reward_0_next = prev_reward_0
                prev_reward_1_next = prev_reward_1
                prev_action_0_next = prev_action_0
                prev_action_1_next = prev_action_1

            if needs_card_attention:
                attn_0_sq = attn_0.squeeze()
                attn_1_sq = attn_1.squeeze()
                p0 = _get_card_game_position_perm(env_state, "agent_0")
                p1 = _get_card_game_position_perm(env_state, "agent_1")
                if partner_feed_dim > 5:
                    cpa_0 = jnp.einsum("hwk,chw->ck", attn_0_sq, card_masks)
                    cpa_1 = jnp.einsum("hwk,chw->ck", attn_1_sq, card_masks)
                    num_heads = cpa_0.shape[-1]
                    phys_0 = jnp.zeros((5, num_heads)).at[p0].set(cpa_0)
                    phys_1 = jnp.zeros((5, num_heads)).at[p1].set(cpa_1)
                    prev_pca_0_next = phys_1[p0].reshape(-1)
                    prev_pca_1_next = phys_0[p1].reshape(-1)
                else:
                    attn_0_2d = (
                        attn_0_sq.mean(axis=-1) if attn_0_sq.ndim == 3 else attn_0_sq
                    )
                    attn_1_2d = (
                        attn_1_sq.mean(axis=-1) if attn_1_sq.ndim == 3 else attn_1_sq
                    )
                    ca0 = jnp.einsum("hw,chw->c", attn_0_2d, card_masks)
                    ca1 = jnp.einsum("hw,chw->c", attn_1_2d, card_masks)
                    ph0 = jnp.zeros(5).at[p0].set(ca0)
                    ph1 = jnp.zeros(5).at[p1].set(ca1)
                    prev_pca_0_next = ph1[p0]
                    prev_pca_1_next = ph0[p1]
            else:
                prev_pca_0_next = prev_pca_0
                prev_pca_1_next = prev_pca_1

            next_carry = (
                obs_next, env_state_next, done_next, rng, hstate_0_next, hstate_1_next,
                prev_reward_0_next, prev_reward_1_next, prev_action_0_next,
                prev_action_1_next, prev_pca_0_next, prev_pca_1_next,
            )
            return next_carry, intents

        def _scan_step(carry, step_idx):
            done_all = carry[2]["__all__"].reshape(())
            return jax.lax.cond(
                done_all,
                lambda c: (c, jnp.full((2,), -1, dtype=jnp.int32)),
                lambda c: _take_step(c, step_idx),
                carry,
            )

        _, intents = jax.lax.scan(_scan_step, init_carry, jnp.arange(T))
        return intents

    return jax.jit(jax.vmap(_run_one_episode, in_axes=(0, None, None)))


def _collect_pair(
    runner, params_a, params_b, num_episodes: int, rng_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run `num_episodes` for one pairing. Returns `(agree, settle)`."""
    rngs = jax.random.split(jax.random.PRNGKey(rng_base), num_episodes)
    intents_all = np.asarray(runner(rngs, params_a, params_b))
    agree_all = (
        (intents_all[:, :, 0] == intents_all[:, :, 1])
        & (intents_all[:, :, 0] >= 0)
        & (intents_all[:, :, 1] >= 0)
    )
    stable_suffix = np.flip(
        np.logical_and.accumulate(np.flip(agree_all, axis=1), axis=1),
        axis=1,
    )
    settle_all = np.where(
        agree_all[:, -1],
        stable_suffix.argmax(axis=1),
        agree_all.shape[1],
    ).astype(np.int64)
    return agree_all, settle_all


def _mean_settled(settle: np.ndarray, max_steps: int) -> float:
    """Mean settle step over episodes that did settle (decision-step success)."""
    s = settle[settle < max_steps]
    return float(s.mean()) if s.size else float("nan")


def _settled_cdf(settle: np.ndarray, max_steps: int) -> np.ndarray:
    """Fraction of episodes permanently settled by each step k (k=0..T-1)."""
    return np.array([float(np.mean(settle <= k)) for k in range(max_steps)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-episodes", type=int, default=128,
                        help="Episodes per seed (SP) and per seed-pair (XP).")
    parser.add_argument("--name", default=None,
                        help="Output basename (e.g. op_ja_shaping). "
                             "Defaults to the run label slug.")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to evaluation/card_game/time_to_agree_data/")
    parser.add_argument("--sampled", action="store_true",
                        help="Sampled actions (default greedy).")
    parser.add_argument("--max-xp-pairs", type=int, default=None,
                        help="Cap on the number of XP seed pairs (quick runs).")
    args = parser.parse_args()

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ev = load_card_game_eval(args.checkpoint)
    greedy = not args.sampled
    T = ev.max_steps
    has_comm = bool(ev.env_kwargs.get("communication", False))
    partner_feed_dim, card_masks = setup_card_feed(ev)
    pair_runner = _build_pair_runner(ev, greedy, partner_feed_dim, card_masks)

    print(
        f"\nCheckpoint: {args.checkpoint}"
        f"\n  label={ev.label}  comm={has_comm}  seeds={ev.num_seeds}"
        f"  max_steps={T}  episodes/pairing={args.num_episodes}  greedy={greedy}"
    )

    sp_agree, sp_settle = [], []
    for s in range(ev.num_seeds):
        params = jax.tree.map(lambda x: x[s], ev.params)
        a, st = _collect_pair(
            pair_runner, params, params, args.num_episodes,
            rng_base=1_000_000 + s * 10_000,
        )
        sp_agree.append(a)
        sp_settle.append(st)
        print(f"  SP seed {s:2d}: success={np.mean(st < T):.3f}  "
              f"mean settle(settled)={_mean_settled(st, T):.2f}")
    sp_agree = np.concatenate(sp_agree)
    sp_settle = np.concatenate(sp_settle)

    pairs = list(itertools.combinations(range(ev.num_seeds), 2))
    if args.max_xp_pairs is not None:
        pairs = pairs[: args.max_xp_pairs]
    xp_agree, xp_settle = [], []
    for idx, (i, j) in enumerate(pairs):
        params_i = jax.tree.map(lambda x: x[i], ev.params)
        params_j = jax.tree.map(lambda x: x[j], ev.params)
        a, st = _collect_pair(
            pair_runner, params_i, params_j, args.num_episodes,
            rng_base=2_000_000 + idx * 10_000,
        )
        xp_agree.append(a)
        xp_settle.append(st)
        if idx == 0 or (idx + 1) % 5 == 0 or idx + 1 == len(pairs):
            print(
                f"  XP pair {idx + 1:2d}/{len(pairs)} ({i},{j}): "
                f"success={np.mean(st < T):.3f}  "
                f"mean settle(settled)={_mean_settled(st, T):.2f}"
            )
    xp_agree = np.concatenate(xp_agree)
    xp_settle = np.concatenate(xp_settle)

    print(f"\nAggregate ({len(sp_settle)} SP, {len(xp_settle)} XP episodes):")
    for tag, st in (("SP", sp_settle), ("XP", xp_settle)):
        print(f"  {tag}: success={np.mean(st < T):.3f}  "
              f"mean settle(settled)={_mean_settled(st, T):.2f}")
    print("  settled-by-step CDF:")
    print("    step:  " + "  ".join(f"k{k}" for k in range(T)))
    print("    SP:    " + "  ".join(f"{v:.2f}" for v in _settled_cdf(sp_settle, T)))
    print("    XP:    " + "  ".join(f"{v:.2f}" for v in _settled_cdf(xp_settle, T)))

    name = args.name or ev.label.replace("/", "_").replace(" ", "_")
    out_dir = (Path(args.output_dir) if args.output_dir
               else Path(__file__).parent / "time_to_agree_data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.npz"
    np.savez(
        out_path,
        sp_agree=sp_agree, sp_settle=sp_settle,
        xp_agree=xp_agree, xp_settle=xp_settle,
        max_steps=T, has_comm=has_comm, label=ev.label,
    )
    print(f"\nSaved {out_path}")


## Tests

def _test_extract_intentions_no_comm() -> None:
    ep_actions = [(0, 1), (0, 2), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
    intents = extract_intentions(ep_actions, [], 8)
    assert intents.shape == (8, 2)
    assert intents[0].tolist() == [0, 1]
    assert intents[7].tolist() == [3, 3]


def _test_extract_intentions_comm() -> None:
    ep_messages = [(1, 2), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (-1, -1)]
    ep_actions = [(-1, -1)] * 7 + [(4, 4)]
    intents = extract_intentions(ep_actions, ep_messages, 8)
    assert intents[0].tolist() == [1, 2]
    assert intents[6].tolist() == [1, 1]
    # decision step takes the pick, never the -1 message slot
    assert intents[7].tolist() == [4, 4]


def _test_episode_agreement_late_settle() -> None:
    intents = np.array([[0, 1], [2, 2], [0, 1], [3, 3], [3, 3],
                        [3, 3], [3, 3], [3, 3]])
    agree, settle = episode_agreement(intents)
    assert settle == 3
    assert agree.tolist() == [False, True, False, True, True, True, True, True]


def _test_episode_agreement_never() -> None:
    intents = np.array([[3, 3]] * 7 + [[3, 4]])
    _, settle = episode_agreement(intents)
    assert settle == 8  # decision-step disagreement = coordination failure


def _test_episode_agreement_always() -> None:
    intents = np.array([[2, 2]] * 8)
    _, settle = episode_agreement(intents)
    assert settle == 0


if __name__ == "__main__":
    main()
