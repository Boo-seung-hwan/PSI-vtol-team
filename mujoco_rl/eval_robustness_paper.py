#!/usr/bin/env python3
"""Paper-oriented robustness evaluation for PPO residual UAV landing.

Default experiment matrix
-------------------------
A_nominal : 1-step action/obs delay, clean target, no wind
B_delay   : 5-step action/obs delay, clean target, no wind
C_target  : 1-step delay, target dropout/stale/outlier/noise, no wind
D_wind    : 1-step delay, clean target, wind disturbance
E_mixed   : 1-5 step delay + target uncertainty + wind disturbance

For every selected case, PID-only and deterministic PPO residual are evaluated
with exactly the same episode seeds. A random-residual baseline is optionally
run for E_mixed (enabled by default).

Outputs
-------
<output-dir>/robustness_summary.csv
<output-dir>/robustness_episodes.csv
<output-dir>/robustness_paired_pid_vs_ppo.csv
<output-dir>/robustness_config.txt

The paired CSV supports episode-wise PID-vs-PPO analysis and exact McNemar
(binomial) testing. Touchdown-speed means reported in the summary are computed
on successful landing episodes only, so timeout zero-values cannot bias them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.env_prototype import LandingEnv, LandingConfig


DEFAULT_MODEL = "./runs/ppo_landing_residual_v4_stage2_contact_final.zip"
DEFAULT_VECNORM = "./runs/vecnormalize_v4_stage2_contact.pkl"
DEFAULT_OUTPUT_DIR = "./runs/paper_eval"
DEFAULT_N_EVAL = 200
DEFAULT_SEED_START = 7000


# -----------------------------------------------------------------------------
# Experiment definitions
# -----------------------------------------------------------------------------

COMMON_CONFIG: Dict[str, Any] = dict(
    max_steps=500,
    residual_xy_mps=0.25,
    residual_z_mps=0.05,
    init_xy_range_m=6.0,
    init_altitude_min_m=2.0,
    init_altitude_max_m=7.0,
    init_vel_range_mps=0.60,
    dt_jitter_std=0.010,
    ground_effect_gain=0.005,
    landing_descent_bias_mps=0.14,
    contact_enabled=True,
    success_requires_contact=True,
    init_yaw_range_rad=0.5,
    w_yaw=0.05,
    # Keep the final-stage outlier magnitude explicit rather than relying on a
    # dataclass default. It matters only in cases where outlier_prob > 0.
    target_outlier_xy_m=1.5,
)

CASE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "A_nominal": dict(
        action_delay_steps_min=1,
        action_delay_steps_max=1,
        obs_delay_steps_min=1,
        obs_delay_steps_max=1,
        target_dropout_prob=0.00,
        target_outlier_prob=0.00,
        target_stale_prob=0.00,
        target_noise_xy_std_m=0.00,
        target_noise_z_std_m=0.00,
        wind_accel_xy_max_mps2=0.00,
        wind_accel_z_max_mps2=0.00,
    ),
    "B_delay": dict(
        action_delay_steps_min=5,
        action_delay_steps_max=5,
        obs_delay_steps_min=5,
        obs_delay_steps_max=5,
        target_dropout_prob=0.00,
        target_outlier_prob=0.00,
        target_stale_prob=0.00,
        target_noise_xy_std_m=0.00,
        target_noise_z_std_m=0.00,
        wind_accel_xy_max_mps2=0.00,
        wind_accel_z_max_mps2=0.00,
    ),
    "C_target": dict(
        action_delay_steps_min=1,
        action_delay_steps_max=1,
        obs_delay_steps_min=1,
        obs_delay_steps_max=1,
        target_dropout_prob=0.05,
        target_outlier_prob=0.02,
        target_stale_prob=0.08,
        target_noise_xy_std_m=0.06,
        target_noise_z_std_m=0.04,
        wind_accel_xy_max_mps2=0.00,
        wind_accel_z_max_mps2=0.00,
    ),
    "D_wind": dict(
        action_delay_steps_min=1,
        action_delay_steps_max=1,
        obs_delay_steps_min=1,
        obs_delay_steps_max=1,
        target_dropout_prob=0.00,
        target_outlier_prob=0.00,
        target_stale_prob=0.00,
        target_noise_xy_std_m=0.00,
        target_noise_z_std_m=0.00,
        wind_accel_xy_max_mps2=0.10,
        wind_accel_z_max_mps2=0.02,
    ),
    "E_mixed": dict(
        action_delay_steps_min=1,
        action_delay_steps_max=5,
        obs_delay_steps_min=1,
        obs_delay_steps_max=5,
        target_dropout_prob=0.05,
        target_outlier_prob=0.02,
        target_stale_prob=0.08,
        target_noise_xy_std_m=0.06,
        target_noise_z_std_m=0.04,
        wind_accel_xy_max_mps2=0.10,
        wind_accel_z_max_mps2=0.02,
    ),
}


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def make_config(case_name: str) -> LandingConfig:
    if case_name not in CASE_OVERRIDES:
        raise KeyError(f"Unknown case: {case_name}")
    values = dict(COMMON_CONFIG)
    values.update(CASE_OVERRIDES[case_name])
    return LandingConfig(**values)


def make_case_env(case_name: str):
    return Monitor(LandingEnv(make_config(case_name)))


def make_eval_env(case_name: str, vecnormalize_path: str, seed: int):
    env = DummyVecEnv([lambda: make_case_env(case_name)])
    env = VecNormalize.load(vecnormalize_path, env)
    env.training = False
    env.norm_reward = False
    env.seed(seed)
    return env


def safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x


def nanmean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def fmt_float(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{digits}f}"


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (
        z
        * math.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n)))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar_p(n_pid_fail_ppo_success: int, n_pid_success_ppo_fail: int) -> float:
    """Two-sided exact McNemar p-value via Binomial(n, 0.5)."""
    b = int(n_pid_fail_ppo_success)
    c = int(n_pid_success_ppo_fail)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def print_case_header(case_name: str, n_eval: int, seed_start: int) -> None:
    cfg = make_config(case_name)
    print()
    print("=" * 78)
    print(f"CASE {case_name}")
    print("-" * 78)
    print(
        f"action delay : {cfg.action_delay_steps_min} - {cfg.action_delay_steps_max} step(s)\n"
        f"obs delay    : {cfg.obs_delay_steps_min} - {cfg.obs_delay_steps_max} step(s)\n"
        f"target       : dropout={cfg.target_dropout_prob:.3f}, "
        f"stale={cfg.target_stale_prob:.3f}, outlier={cfg.target_outlier_prob:.3f}\n"
        f"target noise : xy={cfg.target_noise_xy_std_m:.3f} m, "
        f"z={cfg.target_noise_z_std_m:.3f} m, outlier_xy=+/-{cfg.target_outlier_xy_m:.3f} m\n"
        f"wind         : xy=+/-{cfg.wind_accel_xy_max_mps2:.3f} m/s^2, "
        f"z=+/-{cfg.wind_accel_z_max_mps2:.3f} m/s^2\n"
        f"ground effect: {cfg.ground_effect_gain:.4f}\n"
        f"episodes     : {n_eval}\n"
        f"seeds        : {seed_start} - {seed_start + n_eval - 1}"
    )
    print("=" * 78)


def kinematic_success_from_info(info: Dict[str, Any], cfg: LandingConfig) -> bool:
    xy = safe_float(info.get("xy_error", np.nan))
    z = safe_float(info.get("z_error", np.nan))
    vxy = safe_float(info.get("vxy", np.nan))
    vz = safe_float(info.get("vz_abs", np.nan))
    tilt = safe_float(info.get("tilt", np.nan))
    yaw = safe_float(info.get("yaw_error_abs", 0.0))

    vals = [xy, z, vxy, vz, tilt, yaw]
    if any(not np.isfinite(v) for v in vals):
        return False

    return bool(
        xy < cfg.success_xy_m
        and z < cfg.success_altitude_m
        and vxy < cfg.success_vxy_mps
        and vz < cfg.success_vz_mps
        and tilt < cfg.success_tilt_rad
        and yaw < cfg.success_yaw_error_rad
    )


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def run_eval(
    *,
    case_name: str,
    controller: str,
    model: PPO,
    vecnormalize_path: str,
    n_eval: int,
    seed_start: int,
    print_timeouts: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one controller on one case and return summary + episode rows."""

    cfg = make_config(case_name)
    env = make_eval_env(case_name, vecnormalize_path, seed_start)

    successes = 0
    failures = 0
    timeouts = 0
    failure_reasons: Counter = Counter()

    final_xy_errors: List[float] = []
    final_z_errors: List[float] = []
    steps_list: List[int] = []
    rewards: List[float] = []

    td_vz_success: List[float] = []
    td_vxy_success: List[float] = []
    td_vz_contact: List[float] = []
    td_vxy_contact: List[float] = []

    episode_rows: List[Dict[str, Any]] = []
    timeout_rows: List[Dict[str, Any]] = []

    for ep in range(n_eval):
        episode_seed = seed_start + ep
        env.seed(episode_seed)
        obs = env.reset()

        # Random residual gets an episode-local RNG. This makes re-runs exactly
        # reproducible and avoids dependence on how long prior episodes lasted.
        random_rng: Optional[np.random.Generator] = None
        if controller == "Random":
            random_rng = np.random.default_rng(100_000 + episode_seed)

        done = False
        total_reward = 0.0
        last_info: Dict[str, Any] = {}
        steps = 0

        while not done:
            if controller == "PID":
                action = np.zeros((1, 3), dtype=np.float32)
            elif controller == "PPO":
                action, _ = model.predict(obs, deterministic=True)
            elif controller == "Random":
                assert random_rng is not None
                action = random_rng.uniform(-1.0, 1.0, size=(1, 3)).astype(np.float32)
            else:
                raise ValueError(f"Unknown controller: {controller}")

            obs, reward, done_array, infos = env.step(action)
            done = bool(done_array[0])
            total_reward += float(reward[0])
            last_info = infos[0]
            steps += 1

        success = bool(last_info.get("success", False))
        failed = bool(last_info.get("failed", False))
        timeout = (not success) and (not failed)

        successes += int(success)
        failures += int(failed)
        timeouts += int(timeout)

        xy_error = safe_float(last_info.get("xy_error", np.nan))
        z_error = safe_float(last_info.get("z_error", np.nan))
        impact_vz = safe_float(last_info.get("last_impact_vz", np.nan))
        touchdown_vxy = safe_float(last_info.get("last_touchdown_vxy", np.nan))
        ground_contact = bool(last_info.get("ground_contact", False))

        final_xy_errors.append(xy_error)
        final_z_errors.append(z_error)
        steps_list.append(steps)
        rewards.append(total_reward)

        if success:
            td_vz_success.append(impact_vz)
            td_vxy_success.append(touchdown_vxy)

        if ground_contact:
            td_vz_contact.append(impact_vz)
            td_vxy_contact.append(touchdown_vxy)

        failure_reason = str(last_info.get("failure_reason", "none"))
        if failed:
            failure_reasons[failure_reason] += 1

        kinematic_success = kinematic_success_from_info(last_info, cfg)
        kin_success_no_contact = bool(timeout and kinematic_success and not ground_contact)

        wind = np.asarray(last_info.get("wind_accel", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
        if wind.size < 3:
            wind = np.pad(wind, (0, 3 - wind.size), constant_values=np.nan)

        row = {
            "case": case_name,
            "controller": controller,
            "episode": ep,
            "seed": episode_seed,
            "success": int(success),
            "failed": int(failed),
            "timeout": int(timeout),
            "kinematic_success": int(kinematic_success),
            "kinematic_success_no_contact": int(kin_success_no_contact),
            "xy_error_m": xy_error,
            "z_error_m": z_error,
            "vxy_mps": safe_float(last_info.get("vxy", np.nan)),
            "vz_abs_mps": safe_float(last_info.get("vz_abs", np.nan)),
            "tilt_rad": safe_float(last_info.get("tilt", np.nan)),
            "steps": steps,
            "reward": total_reward,
            "failure_reason": failure_reason,
            "target_valid": last_info.get("target_valid", None),
            "target_mode": last_info.get("target_mode", None),
            "ground_contact": int(ground_contact),
            "contact_event": int(bool(last_info.get("contact_event", False))),
            "soft_contact": int(bool(last_info.get("soft_contact", False))),
            "hard_contact": int(bool(last_info.get("hard_contact", False))),
            "bounced": int(bool(last_info.get("bounced", False))),
            "touchdown_quality": last_info.get("touchdown_quality", "none"),
            "impact_vz_mps": impact_vz,
            "touchdown_vxy_mps": touchdown_vxy,
            "bounce_count": last_info.get("bounce_count", 0),
            "action_delay_steps": last_info.get("action_delay_steps", np.nan),
            "obs_delay_steps": last_info.get("obs_delay_steps", np.nan),
            "wind_n_mps2": safe_float(wind[0]),
            "wind_e_mps2": safe_float(wind[1]),
            "wind_d_mps2": safe_float(wind[2]),
        }
        episode_rows.append(row)

        if timeout:
            timeout_rows.append(row)

    env.close()

    ci_low, ci_high = wilson_interval(successes, n_eval)
    summary = {
        "case": case_name,
        "controller": controller,
        "n_eval": n_eval,
        "seed_start": seed_start,
        "seed_end": seed_start + n_eval - 1,
        "success_count": successes,
        "success_rate_pct": 100.0 * successes / n_eval,
        "success_ci95_low_pct": 100.0 * ci_low,
        "success_ci95_high_pct": 100.0 * ci_high,
        "failure_count": failures,
        "failure_rate_pct": 100.0 * failures / n_eval,
        "timeout_count": timeouts,
        "timeout_rate_pct": 100.0 * timeouts / n_eval,
        "mean_final_xy_error_m": nanmean_or_nan(final_xy_errors),
        "mean_final_z_error_m": nanmean_or_nan(final_z_errors),
        "mean_steps": float(np.mean(steps_list)),
        "mean_reward": float(np.mean(rewards)),
        "mean_touchdown_vz_success_mps": nanmean_or_nan(td_vz_success),
        "mean_touchdown_vxy_success_mps": nanmean_or_nan(td_vxy_success),
        "mean_touchdown_vz_contact_mps": nanmean_or_nan(td_vz_contact),
        "mean_touchdown_vxy_contact_mps": nanmean_or_nan(td_vxy_contact),
        "kinematic_success_no_contact_count": int(
            sum(int(r["kinematic_success_no_contact"]) for r in episode_rows)
        ),
        "failure_reasons": json.dumps(dict(failure_reasons), sort_keys=True),
    }

    print()
    print(f"===== {case_name} | {controller} =====")
    print(
        f"Success rate:        {successes}/{n_eval} = {summary['success_rate_pct']:.1f}% "
        f"(95% CI {summary['success_ci95_low_pct']:.1f}-{summary['success_ci95_high_pct']:.1f}%)"
    )
    print(f"Failure count:       {failures}/{n_eval}")
    print(f"Timeout count:       {timeouts}/{n_eval}")
    print(f"Mean final XY error: {summary['mean_final_xy_error_m']:.4f} m")
    print(f"Mean final Z error:  {summary['mean_final_z_error_m']:.4f} m")
    print(f"Mean steps:          {summary['mean_steps']:.1f}")
    print(f"Mean reward:         {summary['mean_reward']:.1f}")
    print(
        "TD speed (success): "
        f"vz={fmt_float(summary['mean_touchdown_vz_success_mps'])} m/s, "
        f"vxy={fmt_float(summary['mean_touchdown_vxy_success_mps'])} m/s"
    )
    print(
        f"Kinematic success but no contact timeout: "
        f"{summary['kinematic_success_no_contact_count']}"
    )
    print(f"Failure reasons:     {dict(failure_reasons)}")

    if print_timeouts > 0 and timeout_rows:
        n_show = min(print_timeouts, len(timeout_rows))
        print(f"--- Timeout details: first {n_show}/{len(timeout_rows)} ---")
        for item in timeout_rows[:n_show]:
            print(
                f"[ep {int(item['episode']):03d} seed={int(item['seed'])}] "
                f"xy={item['xy_error_m']:.4f}, z={item['z_error_m']:.4f}, "
                f"vxy={item['vxy_mps']:.4f}, vz={item['vz_abs_mps']:.4f}, "
                f"kin_no_contact={bool(item['kinematic_success_no_contact'])}, "
                f"target={item['target_mode']}"
            )

    return summary, episode_rows


# -----------------------------------------------------------------------------
# Pairing + persistence
# -----------------------------------------------------------------------------

def build_paired_rows(
    case_name: str,
    pid_rows: Sequence[Dict[str, Any]],
    ppo_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pid_by_seed = {int(r["seed"]): r for r in pid_rows}
    ppo_by_seed = {int(r["seed"]): r for r in ppo_rows}

    seeds = sorted(set(pid_by_seed).intersection(ppo_by_seed))
    rows: List[Dict[str, Any]] = []
    counts = Counter()

    for seed in seeds:
        pid = pid_by_seed[seed]
        ppo = ppo_by_seed[seed]
        ps = bool(int(pid["success"]))
        qs = bool(int(ppo["success"]))

        if ps and qs:
            pair_type = "both_success"
        elif (not ps) and (not qs):
            pair_type = "both_unsuccessful"
        elif (not ps) and qs:
            pair_type = "ppo_rescue"
        else:
            pair_type = "ppo_regression"
        counts[pair_type] += 1

        rows.append(
            {
                "case": case_name,
                "seed": seed,
                "pid_success": int(ps),
                "ppo_success": int(qs),
                "pair_type": pair_type,
                "pid_xy_error_m": pid["xy_error_m"],
                "ppo_xy_error_m": ppo["xy_error_m"],
                "delta_xy_ppo_minus_pid_m": safe_float(ppo["xy_error_m"]) - safe_float(pid["xy_error_m"]),
                "pid_reward": pid["reward"],
                "ppo_reward": ppo["reward"],
                "delta_reward_ppo_minus_pid": safe_float(ppo["reward"]) - safe_float(pid["reward"]),
                "pid_impact_vz_mps": pid["impact_vz_mps"],
                "ppo_impact_vz_mps": ppo["impact_vz_mps"],
                "pid_touchdown_vxy_mps": pid["touchdown_vxy_mps"],
                "ppo_touchdown_vxy_mps": ppo["touchdown_vxy_mps"],
            }
        )

    b = counts["ppo_rescue"]
    c = counts["ppo_regression"]
    p_exact = exact_mcnemar_p(b, c)

    pair_summary = {
        "case": case_name,
        "n_pairs": len(seeds),
        "both_success": counts["both_success"],
        "both_unsuccessful": counts["both_unsuccessful"],
        "ppo_rescue": b,
        "ppo_regression": c,
        "net_ppo_rescue": b - c,
        "mcnemar_exact_p": p_exact,
    }

    print()
    print(f"===== Paired PID vs PPO | {case_name} =====")
    print(f"Both success         : {pair_summary['both_success']}")
    print(f"Both unsuccessful    : {pair_summary['both_unsuccessful']}")
    print(f"PID fail -> PPO succ : {pair_summary['ppo_rescue']}")
    print(f"PID succ -> PPO fail : {pair_summary['ppo_regression']}")
    print(f"Net PPO rescue       : {pair_summary['net_ppo_rescue']:+d} episode(s)")
    print(f"Exact McNemar p      : {pair_summary['mcnemar_exact_p']:.6f}")

    return rows, pair_summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_config_report(
    path: Path,
    *,
    args: argparse.Namespace,
    cases: Sequence[str],
    pair_summaries: Sequence[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("PPO residual landing robustness evaluation\n")
    lines.append(f"model={args.model}\n")
    lines.append(f"vecnormalize={args.vecnormalize}\n")
    lines.append(f"n_eval={args.n_eval}\n")
    lines.append(f"seed_start={args.seed_start}\n")
    lines.append(f"seed_end={args.seed_start + args.n_eval - 1}\n")
    lines.append(f"random_mixed={not args.skip_random_mixed}\n")
    lines.append("\nCOMMON_CONFIG\n")
    for key, value in COMMON_CONFIG.items():
        lines.append(f"{key}={value}\n")

    for case_name in cases:
        lines.append(f"\n[{case_name}]\n")
        cfg = make_config(case_name)
        for key in sorted(CASE_OVERRIDES[case_name].keys()):
            lines.append(f"{key}={getattr(cfg, key)}\n")

    lines.append("\nPAIRED_PID_VS_PPO\n")
    for item in pair_summaries:
        lines.append(json.dumps(item, sort_keys=True) + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-oriented PID-vs-PPO robustness evaluation."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--vecnormalize", default=DEFAULT_VECNORM)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-eval", type=int, default=DEFAULT_N_EVAL)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=list(CASE_OVERRIDES.keys()),
        default=list(CASE_OVERRIDES.keys()),
        help="Cases to run; default runs A-E in order.",
    )
    parser.add_argument(
        "--skip-random-mixed",
        action="store_true",
        help="Do not evaluate Random residual in E_mixed. PID/PPO always run.",
    )
    parser.add_argument(
        "--print-timeouts",
        type=int,
        default=0,
        metavar="N",
        help="Print at most N timeout rows per controller/case (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.n_eval <= 0:
        raise ValueError("--n-eval must be positive")

    model_path = Path(args.model)
    vec_path = Path(args.vecnormalize)
    if not model_path.exists():
        raise FileNotFoundError(f"PPO model not found: {model_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vec_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PPO model...")
    model = PPO.load(str(model_path), device="cpu")
    print(f"model        : {model_path}")
    print(f"vecnormalize : {vec_path}")
    print(f"output dir   : {out_dir}")

    all_summaries: List[Dict[str, Any]] = []
    all_episode_rows: List[Dict[str, Any]] = []
    all_paired_rows: List[Dict[str, Any]] = []
    pair_summaries: List[Dict[str, Any]] = []

    for case_name in args.cases:
        print_case_header(case_name, args.n_eval, args.seed_start)

        per_controller: Dict[str, List[Dict[str, Any]]] = {}
        controllers = ["PID", "PPO"]
        if case_name == "E_mixed" and not args.skip_random_mixed:
            controllers.append("Random")

        for controller in controllers:
            summary, episode_rows = run_eval(
                case_name=case_name,
                controller=controller,
                model=model,
                vecnormalize_path=str(vec_path),
                n_eval=args.n_eval,
                seed_start=args.seed_start,
                print_timeouts=args.print_timeouts,
            )
            all_summaries.append(summary)
            all_episode_rows.extend(episode_rows)
            per_controller[controller] = episode_rows

        paired_rows, pair_summary = build_paired_rows(
            case_name,
            per_controller["PID"],
            per_controller["PPO"],
        )
        all_paired_rows.extend(paired_rows)
        pair_summaries.append(pair_summary)

    # Add paired-case summary columns to the PID/PPO summary rows so the main
    # summary CSV is immediately useful for a paper table.
    pair_by_case = {x["case"]: x for x in pair_summaries}
    for row in all_summaries:
        pair = pair_by_case.get(row["case"], {})
        row["paired_ppo_rescue"] = pair.get("ppo_rescue", "")
        row["paired_ppo_regression"] = pair.get("ppo_regression", "")
        row["paired_net_ppo_rescue"] = pair.get("net_ppo_rescue", "")
        row["paired_mcnemar_exact_p"] = pair.get("mcnemar_exact_p", "")

    summary_path = out_dir / "robustness_summary.csv"
    episodes_path = out_dir / "robustness_episodes.csv"
    paired_path = out_dir / "robustness_paired_pid_vs_ppo.csv"
    config_path = out_dir / "robustness_config.txt"

    write_csv(summary_path, all_summaries)
    write_csv(episodes_path, all_episode_rows)
    write_csv(paired_path, all_paired_rows)
    write_config_report(
        config_path,
        args=args,
        cases=args.cases,
        pair_summaries=pair_summaries,
    )

    print()
    print("=" * 78)
    print("DONE")
    print(f"summary : {summary_path}")
    print(f"episodes: {episodes_path}")
    print(f"paired  : {paired_path}")
    print(f"config  : {config_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
