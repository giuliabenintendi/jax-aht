"""Self-play behavior traces for the demo_cook recipe-convention analysis.

Tests how the recipe information asymmetry is resolved: only the agent in the
compartment with the recipe indicators can read the recipe, so either the
observer initiates every cook (information-respecting play), the non-observer
blind-pours a default ingredient (H1), or the observer signals through movement
and the non-observer decodes it (H2). The three cases separate on two numbers:
who inserts into the pot first, and whether the non-observer's first insert
matches the recipe at chance (H1) or above it (H2).

Runs same-seed self-play episodes per checkpoint, dumps per-step traces to
.npz, and prints first-insert metrics. The episode loop mirrors
run_xp_seeds.run_single_episode_with_jsd exactly on the feed semantics
(per-receiver carry, uniform init, zero visibility mask at t=0, egocentric
reframe) so MATE checkpoints behave as in the XP evals.

Usage:
    uv run python -m evaluation.eval_first_insert \
        --task overcooked-v2/demo_cook_simple-ego \
        --checkpoints results/.../saved_train_run [...] \
        --episodes 128 --out ~/first_insert_traces
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

from agents.initialize_agents import initialize_ja_image_agent
from agents.ja_utils import augment_obs_for_eval
from common.save_load_utils import load_train_run
from envs import make_env
from envs.log_wrapper import LogWrapper
from envs.overcooked_v2.common import Actions, StaticObject
from evaluation.run_xp_seeds import (
    _load_hydra_config,
    _select_xp_params,
    load_task_config,
)

EVAL_SEED = 34957
NUM_INGREDIENTS = 3  # demo_cook_simple: 0=onion, 1=broccoli, 2=distractor


def _base_state(state):
    """Unwrap LogEnvState/WrappedEnvState chains down to the ocv2 State."""
    while not hasattr(state, "grid"):
        state = state.env_state
    return state


def build_feed_ctx(algo_cfg, env):
    """Feed-channel setup copied from run_xp_seeds.run_xp_multi_checkpoint."""
    feed_attn_dims = None
    feed_mask_fn = None
    feed_reframe_fn = None
    if not algo_cfg.get("FEED_OTHER_ATTN", False):
        return feed_attn_dims, feed_mask_fn, feed_reframe_fn
    from agents.initialize_agents import _get_image_dims
    from agents.ja_actor_critic import _compute_resnet_output_dims
    from agents.overcooked_v2.ja_overcooked_v2_attention import (
        make_visibility_mask_fn,
        overcooked_v2_object_ctx,
        reframe_partner_attention_for_eval,
    )
    img_h, img_w, _ = _get_image_dims(env)
    feat_h, feat_w = _compute_resnet_output_dims(
        img_h, img_w,
        stride=algo_cfg.get("CONV_STRIDE", 2),
        kernel_size=algo_cfg.get("CONV_KERNEL_SIZE", 3),
        padding=algo_cfg.get("CONV_PADDING", "SAME"),
        num_blocks=algo_cfg.get("CONV_NUM_BLOCKS", 4),
    )
    feed_attn_dims = (img_h, img_w, feat_h, feat_w)
    ocv2_ctx = overcooked_v2_object_ctx(algo_cfg, env)
    if algo_cfg.get("JA_VISIBILITY_GATING", False):
        feed_mask_fn = make_visibility_mask_fn(ocv2_ctx)
    if ocv2_ctx.get("egocentric", False):
        feed_reframe_fn = lambda a0, a1, s0, s1: reframe_partner_attention_for_eval(
            a0, a1, s0, s1, ocv2_ctx
        )
    return feed_attn_dims, feed_mask_fn, feed_reframe_fn


def make_episode_fn(env, policy, max_steps, action_sizes, feed_attn_dims,
                    feed_mask_fn, feed_reframe_fn, greedy):
    """One SP episode returning per-step traces (see ys dict for fields)."""
    _img_h = _img_w = _feat_h = _feat_w = None
    if feed_attn_dims is not None:
        _img_h, _img_w, _feat_h, _feat_w = feed_attn_dims

    def _act(params, obs, done, avail, hstate, rng):
        act, hs, attn = policy.get_action_and_attention(
            params=params, obs=obs.reshape(1, 1, -1),
            done=done.reshape(1, 1), avail_actions=avail,
            hstate=hstate, rng=rng, greedy=greedy,
        )
        return act.squeeze(), hs, attn

    # params0/params1 per agent so the same runner covers SP and XP pairs.
    def _trace(state_pre):
        s = _base_state(state_pre)
        return {
            "recipe": jnp.asarray(s.recipe, jnp.int32),
            "pot": jnp.zeros((), jnp.int32),  # filled by caller via pot_yx
            "ax": jnp.ravel(s.agents.pos.x)[:2].astype(jnp.int32),
            "ay": jnp.ravel(s.agents.pos.y)[:2].astype(jnp.int32),
            "inv": jnp.ravel(s.agents.inventory)[:2].astype(jnp.int32),
        }

    def episode(rng, params0, params1, pot_yx):
        rng, reset_rng = jax.random.split(rng)
        obs, state = env.reset(reset_rng)
        # env.step returns scalar dones; init must match for the scan carry.
        done = {k: jnp.zeros((), dtype=bool) for k in env.agents + ["__all__"]}
        hs0 = policy.init_hstate(1, aux_info={"agent_id": 0})
        hs1 = policy.init_hstate(1, aux_info={"agent_id": 1})
        if feed_attn_dims is not None:
            feed0 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)
            feed1 = jnp.ones((_feat_h, _feat_w)) / (_feat_h * _feat_w)
        else:
            feed0 = feed1 = jnp.zeros(())

        def body(carry, t):
            state, obs, rng, done, hs0, hs1, feed0, feed1 = carry

            def frozen(_):
                z = {
                    "recipe": jnp.zeros((), jnp.int32),
                    "pot": jnp.zeros((), jnp.int32),
                    "ax": jnp.zeros(2, jnp.int32), "ay": jnp.zeros(2, jnp.int32),
                    "inv": jnp.zeros(2, jnp.int32),
                    "acts": jnp.zeros(2, jnp.int32),
                    "rew": jnp.zeros(()), "delivery": jnp.zeros((), bool),
                    "valid": jnp.zeros((), bool),
                }
                return carry, z

            def step(_):
                avail = jax.lax.stop_gradient(env.get_avail_actions(state))
                a0_avail = avail["agent_0"].astype(jnp.float32)
                a1_avail = avail["agent_1"].astype(jnp.float32)
                rng_, k0, k1, ks = jax.random.split(rng, 4)

                o0, o1 = obs["agent_0"], obs["agent_1"]
                if feed_attn_dims is not None:
                    m0 = m1 = None
                    if feed_mask_fn is not None:
                        # Training starts fully masked (positions unknown
                        # before the first step); mirror run_xp_seeds.
                        vm0, vm1 = feed_mask_fn(state)
                        zero = jnp.zeros((_feat_h, _feat_w))
                        m0 = jnp.where(t == 0, zero, vm0)
                        m1 = jnp.where(t == 0, zero, vm1)
                    o0 = augment_obs_for_eval(o0, feed0, _img_h, _img_w, mask=m0)
                    o1 = augment_obs_for_eval(o1, feed1, _img_h, _img_w, mask=m1)

                act0, hs0_n, attn0 = _act(params0, o0, done["agent_0"], a0_avail, hs0, k0)
                act1, hs1_n, attn1 = _act(params1, o1, done["agent_1"], a1_avail, hs1, k1)

                env_act = {k: [act0, act1][i] for i, k in enumerate(env.agents)}
                obs_n, state_n, reward, done_n, _ = env.step(ks, state, env_act)

                if feed_attn_dims is not None and feed_reframe_fn is not None:
                    feed0_n, feed1_n = feed_reframe_fn(
                        attn0.squeeze(), attn1.squeeze(), state, state_n
                    )
                elif feed_attn_dims is not None:
                    feed0_n, feed1_n = attn1.squeeze(), attn0.squeeze()
                else:
                    feed0_n, feed1_n = feed0, feed1

                ys = _trace(state)
                ys["pot"] = _base_state(state).grid[pot_yx[0], pot_yx[1], 1].astype(jnp.int32)
                ys["acts"] = jnp.stack([act0, act1]).astype(jnp.int32)
                ys["rew"] = reward["agent_0"].squeeze().astype(jnp.float32)
                ys["delivery"] = jnp.asarray(_base_state(state_n).new_correct_delivery, bool)
                ys["valid"] = jnp.ones((), bool)
                return (state_n, obs_n, rng_, done_n, hs0_n, hs1_n, feed0_n, feed1_n), ys

            return jax.lax.cond(done["__all__"].squeeze(), frozen, step, operand=None)

        carry0 = (state, obs, rng, done, hs0, hs1, feed0, feed1)
        _, traces = jax.lax.scan(body, carry0, jnp.arange(max_steps))
        return traces

    return episode


def ingredient_counts(pot_vals):
    """(..., ) int pot/inventory encodings -> (..., NUM_INGREDIENTS) counts."""
    pot_vals = np.asarray(pot_vals)[..., None]
    shifts = 2 + 2 * np.arange(NUM_INGREDIENTS)
    return (pot_vals >> shifts) & 0x3


def analyze(traces, pot_x, observer_side_left):
    """First-insert metrics from stacked traces (episodes, steps, ...)."""
    valid = np.asarray(traces["valid"])
    pot_cnt = ingredient_counts(traces["pot"]).sum(-1)  # (E, T)
    E = valid.shape[0]
    rows = []
    for e in range(E):
        T = int(valid[e].sum())
        if T < 2:
            continue
        diff = np.diff(pot_cnt[e, :T])
        ins_steps = np.nonzero(diff > 0)[0]
        if len(ins_steps) == 0:
            rows.append(dict(insert=False, ret=float(traces["rew"][e, :T].sum())))
            continue
        t = int(ins_steps[0])
        per_ing = np.diff(ingredient_counts(traces["pot"][e, :T]), axis=0)[t]
        ing = int(per_ing.argmax())
        # Attribute: interacting agent adjacent to the pot holding that ingredient.
        who = -1
        for a in range(2):
            held = ingredient_counts(traces["inv"][e, t, a])[..., ing] > 0
            if traces["acts"][e, t, a] == Actions.interact and held:
                who = a if who == -1 else 2  # 2 = ambiguous double insert
        left = traces["ax"][e, t, who if who in (0, 1) else 0] < pot_x
        role_obs = bool(left) == bool(observer_side_left)
        recipe_has = ingredient_counts(traces["recipe"][e, t])[..., ing] > 0
        rows.append(dict(
            insert=True, t=t, who=who, ing=ing,
            by_observer=role_obs if who in (0, 1) else None,
            correct=bool(recipe_has),
            ret=float(traces["rew"][e, :T].sum()),
        ))
    ins = [r for r in rows if r.get("insert")]
    attributed = [r for r in ins if r["by_observer"] is not None]
    nonobs = [r for r in attributed if not r["by_observer"]]
    obs_first = [r for r in attributed if r["by_observer"]]
    out = {
        "episodes": len(rows),
        "with_insert": len(ins),
        "mean_return": float(np.mean([r["ret"] for r in rows])) if rows else float("nan"),
        "p_nonobserver_first": len(nonobs) / max(len(attributed), 1),
        "p_correct_nonobs_first": (np.mean([r["correct"] for r in nonobs])
                                   if nonobs else float("nan")),
        "p_correct_obs_first": (np.mean([r["correct"] for r in obs_first])
                                if obs_first else float("nan")),
        "mean_t_first_insert": float(np.mean([r["t"] for r in ins])) if ins else float("nan"),
    }
    return out, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--sampled", action="store_true")
    p.add_argument("--xp", action="store_true",
                   help="run every ordered cross pair (i, j), i != j, instead of SP")
    p.add_argument("--out", default="first_insert_traces")
    args = p.parse_args()

    task_cfg = load_task_config(args.task)
    os.makedirs(args.out, exist_ok=True)

    hydra_cfg = _load_hydra_config(args.checkpoints[0])
    algo_cfg = hydra_cfg["algorithm"]
    algo_cfg["ENV_NAME"] = task_cfg["ENV_NAME"]
    algo_cfg["ENV_KWARGS"] = dict(task_cfg["ENV_KWARGS"])
    env = LogWrapper(make_env(task_cfg["ENV_NAME"], dict(task_cfg["ENV_KWARGS"])))

    rng = jax.random.PRNGKey(EVAL_SEED)
    rng, init_rng = jax.random.split(rng)
    policy, _ = initialize_ja_image_agent(algo_cfg, env, init_rng)
    action_sizes = {k: int(env.action_space(k).n) for k in env.agents}
    max_steps = int(task_cfg["ENV_KWARGS"].get("max_steps") or task_cfg["ROLLOUT_LENGTH"])

    # Static geometry from a concrete reset: pot cell, recipe-indicator side.
    _, s0 = env.reset(jax.random.PRNGKey(0))
    static = np.asarray(_base_state(s0).grid[:, :, 0])
    pot_yx = tuple(int(v) for v in np.argwhere(static == StaticObject.POT)[0])
    r_pos = np.argwhere(static == StaticObject.RECIPE_INDICATOR)
    observer_side_left = bool(r_pos[:, 1].mean() < pot_yx[1])
    print(f"[first_insert] pot at (y,x)={pot_yx}, recipe indicators at "
          f"{r_pos.tolist()}, observer side = {'left' if observer_side_left else 'right'}")

    feed_ctx = build_feed_ctx(algo_cfg, env)
    episode_fn = make_episode_fn(env, policy, max_steps, action_sizes,
                                 *feed_ctx, greedy=not args.sampled)
    batched = jax.jit(jax.vmap(episode_fn, in_axes=(0, None, None, None)),
                      static_argnums=())

    mode = "greedy" if not args.sampled else "sampled"
    print(f"[first_insert] {len(args.checkpoints)} checkpoints, "
          f"{args.episodes} episodes each, {mode} decoding, "
          f"{'XP pairs' if args.xp else 'SP'}")
    pot_arr = jnp.asarray(pot_yx)

    seed_params, labels = [], []
    for ckpt in args.checkpoints:
        run_data = load_train_run(ckpt)
        params, key = _select_xp_params(run_data)
        seed_params.append(jax.tree.map(lambda x: x[0], params))
        # .../<label>_30M/<timestamp>/saved_train_run -> <label>_30M
        labels.append(ckpt.rstrip("/").split("/")[-3])

    if args.xp:
        pairs = [(i, j) for i in range(len(labels))
                 for j in range(len(labels)) if i != j]
    else:
        pairs = [(i, i) for i in range(len(labels))]

    for i, j in pairs:
        rng, ep_rng = jax.random.split(rng)
        rngs = jax.random.split(ep_rng, args.episodes)
        traces = jax.device_get(batched(rngs, seed_params[i], seed_params[j], pot_arr))
        name = labels[i] if i == j else f"{labels[i]}__X__{labels[j]}"
        np.savez_compressed(
            os.path.join(args.out, f"{name}_{mode}.npz"),
            pot_yx=np.asarray(pot_yx), observer_side_left=observer_side_left,
            r_pos=r_pos, **{k: np.asarray(v) for k, v in traces.items()},
        )
        metrics, _ = analyze(traces, pot_yx[1], observer_side_left)
        print(f"  {name}: " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()))


if __name__ == "__main__":
    main()
