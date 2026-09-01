import numpy as np
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.env_prototype import LandingEnv, LandingConfig


def make_config():
    return LandingConfig(
        # Stage 1 config
        max_steps=500,

        residual_xy_mps = 0.25,
        residual_z_mps=0.05,

        init_xy_range_m=6.0,
        init_altitude_min_m=2.0,
        init_altitude_max_m=7.0,
        init_vel_range_mps=0.60,

        dt_jitter_std=0.010,

        action_delay_steps_min=1,
        action_delay_steps_max=1,
        obs_delay_steps_min=1,
        obs_delay_steps_max=1,

        target_dropout_prob=0.00,
        target_outlier_prob=0.00,
        target_stale_prob=0.0,
        target_noise_xy_std_m=0.00,
        target_noise_z_std_m=0.0,

        wind_accel_xy_max_mps2=0.1,
        wind_accel_z_max_mps2=0.02,
        ground_effect_gain= 0.005,
        landing_descent_bias_mps=0.14,


        contact_enabled=True,
        success_requires_contact=True,

        init_yaw_range_rad=0.5,
        w_yaw=0.05,
    )


def make_stage1_env():
    return Monitor(LandingEnv(make_config()))


def make_eval_env(seed=0):
    env = DummyVecEnv([make_stage1_env])

    env = VecNormalize.load(
        "./runs/vecnormalize_v4_stage2_contact.pkl",
        env,
    )

    env.training = False
    env.norm_reward = False
    env.seed(seed)

    return env


def fmt_vec(x):
    if x is None:
        return "None"
    x = np.asarray(x, dtype=float).reshape(-1)
    return "[" + ", ".join(f"{v:+.3f}" for v in x) + "]"


def run_eval(name, policy_fn, n_eval=200, print_timeouts=False, max_timeout_print=15):
    env = make_eval_env(seed=5000)

    successes = 0
    failures = 0
    truncations = 0

    final_xy_errors = []
    final_z_errors = []
    steps_list = []
    rewards = []
    failure_reasons = Counter()

    timeout_infos = []

    for ep in range(n_eval):
        env.seed(5000 + ep)
        obs = env.reset()

        done = False
        total_reward = 0.0
        last_info = {}
        steps = 0

        while not done:
            action = policy_fn(obs)

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
        truncations += int(timeout)

        final_xy_errors.append(last_info.get("xy_error", np.nan))
        final_z_errors.append(last_info.get("z_error", np.nan))
        steps_list.append(steps)
        rewards.append(total_reward)

        reason = last_info.get("failure_reason", "none")
        if failed:
            failure_reasons[reason] += 1

        if timeout:
            timeout_infos.append({
                "ep": ep,
                "steps": steps,
                "reward": total_reward,
                "xy_error": last_info.get("xy_error", np.nan),
                "z_error": last_info.get("z_error", np.nan),
                "vxy": last_info.get("vxy", np.nan),
                "vz_abs": last_info.get("vz_abs", np.nan),
                "altitude_agl": last_info.get("altitude_agl", np.nan),
                "tilt": last_info.get("tilt", np.nan),
                "target_valid": last_info.get("target_valid", None),
                "target_mode": last_info.get("target_mode", None),
                "raw_action": last_info.get("raw_action", None),
                "applied_action": last_info.get("applied_action", None),
                "v_pid": last_info.get("v_pid", None),
                "v_residual": last_info.get("v_residual", None),
                "v_cmd": last_info.get("v_cmd", None),
                "pos": last_info.get("pos", None),
                "vel": last_info.get("vel", None),
                "ground_contact": last_info.get("ground_contact", None),
                "contact_event": last_info.get("contact_event", None),
                "contact_count": last_info.get("contact_count", None),
                "soft_contact": last_info.get("soft_contact", None),
                "hard_contact": last_info.get("hard_contact", None),
                "bounced": last_info.get("bounced", None),
                "touchdown_quality": last_info.get("touchdown_quality", None),
                "last_impact_vz": last_info.get("last_impact_vz", np.nan),
                "last_touchdown_vxy": last_info.get("last_touchdown_vxy", np.nan),
                "bounce_count": last_info.get("bounce_count", None),
            })

    env.close()

    print()
    print(f"===== {name} =====")
    print(f"Success rate:       {successes}/{n_eval} = {successes / n_eval * 100:.1f}%")
    print(f"Failure count:      {failures}/{n_eval}")
    print(f"Timeout count:      {truncations}/{n_eval}")
    print(f"Mean final XY error:{np.nanmean(final_xy_errors):.4f} m")
    print(f"Mean final Z error: {np.nanmean(final_z_errors):.4f} m")
    print(f"Mean steps:         {np.mean(steps_list):.1f}")
    print(f"Mean reward:        {np.mean(rewards):.1f}")
    print(f"Failure reasons:    {dict(failure_reasons)}")

    if print_timeouts and timeout_infos:
        print()
        print(f"--- Timeout details: first {min(max_timeout_print, len(timeout_infos))}/{len(timeout_infos)} ---")

        for item in timeout_infos[:max_timeout_print]:
            print(
                f"[ep {item['ep']:03d}] "
                f"steps={item['steps']}, "
                f"R={item['reward']:.1f}, "
                f"xy={item['xy_error']:.4f}, "
                f"z={item['z_error']:.4f}, "
                f"alt_agl={item['altitude_agl']:.4f}, "
                f"vxy={item['vxy']:.4f}, "
                f"vz_abs={item['vz_abs']:.4f}, "
                f"tilt={item['tilt']:.4f}, "
                f"valid={item['target_valid']}, "
                f"mode={item['target_mode']}"
            )
            print(
                f"    contact    = ground={item['ground_contact']}, "
                f"event={item['contact_event']}, "
                f"count={item['contact_count']}, "
                f"soft={item['soft_contact']}, "
                f"hard={item['hard_contact']}, "
                f"bounced={item['bounced']}, "
                f"quality={item['touchdown_quality']}, "
                f"impact_vz={item['last_impact_vz']:.4f}, "
                f"td_vxy={item['last_touchdown_vxy']:.4f}, "
                f"bounce_count={item['bounce_count']}"
            )
            print(f"    pos        = {fmt_vec(item['pos'])}")
            print(f"    vel        = {fmt_vec(item['vel'])}")
            print(f"    raw_action = {fmt_vec(item['raw_action'])}")
            print(f"    applied    = {fmt_vec(item['applied_action'])}")
            print(f"    v_pid      = {fmt_vec(item['v_pid'])}")
            print(f"    v_residual = {fmt_vec(item['v_residual'])}")
            print(f"    v_cmd      = {fmt_vec(item['v_cmd'])}")


def main():
    model = PPO.load(
        "./runs/ppo_landing_residual_v4_stage2_contact_final.zip",
        device="cpu",
    )

    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    def pid_only_policy(obs):
        return np.zeros((1, 3), dtype=np.float32)

    rng = np.random.default_rng(0)

    def random_residual_policy(obs):
        return rng.uniform(-1.0, 1.0, size=(1, 3)).astype(np.float32)

    run_eval("PID only stage1 env", pid_only_policy, n_eval=200, print_timeouts=False)
    run_eval("Random residual stage1 env", random_residual_policy, n_eval=200, print_timeouts=False)
    run_eval("PPO residual stage1 env", ppo_policy, n_eval=200, print_timeouts=True)


if __name__ == "__main__":
    main()