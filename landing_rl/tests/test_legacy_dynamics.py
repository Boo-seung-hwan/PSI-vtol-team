"""Focused tests for the Phase 13C-1 VehicleState + LegacyVehicleDynamics
extraction.

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py`` and the checkpoint gate in
``test_checkpoint_compatibility.py``, both of which remain authoritative.

``LegacyVehicleDynamics.advance_free_flight`` holds the free-flight / PX4-proxy
portion (D4-D25) of ``LandingEnv._update_rigid_body_dynamics`` lifted verbatim
from ``mujoco_rl/envs/env_prototype.py``. These tests compare it against a
verbatim in-file replica of that legacy span (operating on a plain holder
object, drawing process noise with raw ``rng.normal`` -- a genuinely
independent reference), and separately assert the env-level VehicleState
compatibility-mirror contract and the structural ownership constraints.

Comparison is exact (``np.array_equal`` / ``==`` / identical RNG state). No
tolerance.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from test_legacy_regression_contract import (
    NewLandingConfig,
    NewLandingEnv,
    OldLandingConfig,
    OldLandingEnv,
    _rng_state_equal,
    make_action_sequence,
)

from landing_rl.dynamics import LegacyVehicleDynamics, ProcessNoiseSampler, VehicleState


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------------------
# Verbatim replica of the legacy free-flight span (D4-D25).
#
# Copied character-for-character from mujoco_rl/envs/env_prototype.py:
#   _velocity_command_to_inner_loop_setpoints  (lines ~664-723)
#   _rotation_body_to_ned                        (lines ~647-662)
#   _compute_ground_effect_factor + _altitude_agl
#   _update_rigid_body_dynamics lines 733-735 + 739-849
# with `self.<x>` kept as holder attributes. The three process-noise draws use
# raw `self.np_random.normal(...)` exactly as the legacy code does -- this
# replica does NOT go through ProcessNoiseSampler.
# ---------------------------------------------------------------------------
class _LegacyFreeFlightReplica:
    def __init__(self, cfg):
        self.cfg = cfg

    def _altitude_agl(self) -> float:
        return max(0.0, float(self.cfg.ground_z_m - self.pos[2]))

    def _compute_ground_effect_factor(self) -> float:
        h = self._altitude_agl()
        height = max(float(self.cfg.ground_effect_height_m), 1e-6)
        if h >= height or self.ground_contact:
            return 1.0
        closeness = 1.0 - h / height
        factor = 1.0 + float(self.cfg.ground_effect_gain) * closeness * closeness
        return float(np.clip(factor, 1.0, self.cfg.ground_effect_max_factor))

    def _rotation_body_to_ned(self) -> np.ndarray:
        roll, pitch, yaw = [float(v) for v in self.attitude]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ],
            dtype=np.float64,
        )

    def _velocity_command_to_inner_loop_setpoints(self, v_cmd, dt):
        dt_safe = max(float(dt), 1e-6)
        g = float(self.cfg.gravity_mps2)
        accel_cmd = (np.asarray(v_cmd, dtype=np.float64) - self.vel) / max(
            self.cfg.vel_cmd_tau_s,
            dt_safe,
        )
        axy_norm = float(np.linalg.norm(accel_cmd[:2]))
        if axy_norm > self.cfg.max_cmd_accel_xy_mps2:
            accel_cmd[:2] *= self.cfg.max_cmd_accel_xy_mps2 / (axy_norm + 1e-9)
        accel_cmd[2] = float(np.clip(
            accel_cmd[2],
            -self.cfg.max_cmd_accel_z_mps2,
            self.cfg.max_cmd_accel_z_mps2,
        ))
        denom = max(g - float(accel_cmd[2]), 1e-3)
        roll_sp = math.atan2(float(accel_cmd[1]), denom)
        pitch_sp = math.atan2(-float(accel_cmd[0]), denom)
        roll_sp = float(np.clip(
            roll_sp, -self.cfg.max_tilt_target_rad, self.cfg.max_tilt_target_rad,
        ))
        pitch_sp = float(np.clip(
            pitch_sp, -self.cfg.max_tilt_target_rad, self.cfg.max_tilt_target_rad,
        ))
        yaw_sp = float(self.target_yaw)
        thrust_sp = float(np.clip(
            g - float(accel_cmd[2]),
            self.cfg.min_thrust_accel_mps2,
            self.cfg.max_thrust_accel_mps2,
        ))
        attitude_sp = np.array([roll_sp, pitch_sp, yaw_sp], dtype=np.float64)
        return attitude_sp, thrust_sp, accel_cmd

    def free_flight(self, v_cmd, dt):
        """Verbatim _update_rigid_body_dynamics lines 733-735 + 739-849
        (i.e. the D4-D25 span, excluding _reset_contact_step_flags / vel_before /
        _apply_ground_contact / final accel recompute)."""
        dt_safe = max(float(dt), 1e-6)
        g = float(self.cfg.gravity_mps2)

        attitude_sp, thrust_sp, accel_cmd = self._velocity_command_to_inner_loop_setpoints(v_cmd, dt_safe)
        if self.motor_cutoff:
            thrust_sp = float(self.cfg.motor_cutoff_thrust_accel_mps2)

        self.attitude_setpoint = attitude_sp.copy()
        self.thrust_accel_setpoint = float(thrust_sp)
        self.accel_cmd = accel_cmd.copy()

        att_error = np.array(
            [
                attitude_sp[0] - self.attitude[0],
                attitude_sp[1] - self.attitude[1],
                _wrap_pi(float(attitude_sp[2] - self.attitude[2])),
            ],
            dtype=np.float64,
        )

        rate_cmd = att_error / max(self.cfg.attitude_time_constant_s, dt_safe)
        rate_cmd[0] = float(np.clip(
            rate_cmd[0],
            -self.cfg.max_roll_pitch_rate_radps,
            self.cfg.max_roll_pitch_rate_radps,
        ))
        rate_cmd[1] = float(np.clip(
            rate_cmd[1],
            -self.cfg.max_roll_pitch_rate_radps,
            self.cfg.max_roll_pitch_rate_radps,
        ))
        rate_cmd[2] = float(np.clip(
            self.cfg.yaw_align_kp * att_error[2],
            -self.cfg.max_yaw_rate_response_radps,
            self.cfg.max_yaw_rate_response_radps,
        ))

        rate_alpha_eff = 1.0 - np.power(
            1.0 - np.clip(self.body_rate_response_alpha, 1e-4, 0.9999),
            dt_safe / max(self.cfg.dt, 1e-6),
        )
        rate_alpha_eff = np.clip(rate_alpha_eff, 0.0, 1.0)

        self.body_rates = self.body_rates + rate_alpha_eff * (rate_cmd - self.body_rates)

        noise_scale = math.sqrt(dt_safe / max(self.cfg.dt, 1e-6))
        self.body_rates += self.np_random.normal(
            0.0,
            self.cfg.body_rate_process_noise_std_radps * noise_scale,
            size=3,
        )

        self.attitude[0] = float(np.clip(
            self.attitude[0] + self.body_rates[0] * dt_safe, -math.pi, math.pi,
        ))
        self.attitude[1] = float(np.clip(
            self.attitude[1] + self.body_rates[1] * dt_safe, -math.pi, math.pi,
        ))
        self.attitude[2] = _wrap_pi(float(self.attitude[2] + self.body_rates[2] * dt_safe))
        self.yaw_rate = float(self.body_rates[2])

        thrust_alpha_eff = 1.0 - (1.0 - np.clip(self.thrust_response_alpha, 1e-4, 0.9999)) ** (
            dt_safe / max(self.cfg.dt, 1e-6)
        )
        thrust_alpha_eff = float(np.clip(thrust_alpha_eff, 0.0, 1.0))
        self.thrust_accel = float(
            self.thrust_accel
            + thrust_alpha_eff * (thrust_sp - self.thrust_accel)
            + self.np_random.normal(0.0, self.cfg.thrust_process_noise_std_mps2 * noise_scale)
        )
        thrust_min = (
            float(self.cfg.motor_cutoff_thrust_accel_mps2)
            if self.motor_cutoff
            else float(self.cfg.min_thrust_accel_mps2)
        )
        self.thrust_accel = float(np.clip(
            self.thrust_accel, thrust_min, self.cfg.max_thrust_accel_mps2,
        ))

        r_bn = self._rotation_body_to_ned()
        body_z_in_ned = r_bn[:, 2]
        gravity_accel = np.array([0.0, 0.0, g], dtype=np.float64)

        self.ground_effect_factor = self._compute_ground_effect_factor()
        effective_thrust_accel = self.thrust_accel * self.ground_effect_factor
        thrust_accel_ned = -effective_thrust_accel * body_z_in_ned
        drag_accel = -np.array(
            [
                self.cfg.linear_drag_xy * self.vel[0],
                self.cfg.linear_drag_xy * self.vel[1],
                self.cfg.linear_drag_z * self.vel[2],
            ],
            dtype=np.float64,
        )
        process_accel_noise = self.np_random.normal(
            0.0,
            self.cfg.process_noise_vel_std_mps * noise_scale / max(dt_safe, 1e-6),
            size=3,
        )

        self.prev_accel = self.accel.copy()
        self.accel = gravity_accel + thrust_accel_ned + drag_accel + self.wind_accel + process_accel_noise

        self.vel = self.vel + self.accel * dt_safe
        self.pos = self.pos + self.vel * dt_safe


_VS_FIELDS = tuple(VehicleState.__dataclass_fields__)
_ARR_FIELDS = ("pos", "vel", "accel", "prev_accel", "attitude", "body_rates",
               "attitude_setpoint", "accel_cmd")
_SCALAR_FIELDS = ("yaw_rate", "thrust_accel", "thrust_accel_setpoint",
                  "ground_effect_factor")


# scenario name -> dict of initial VehicleState field seeds (the rest default).
_INIT_STATES = {
    "hover": dict(),
    "descending": dict(pos=[0.2, -0.3, -1.5], vel=[0.1, 0.05, 0.9]),
    "tilted_rolling": dict(
        pos=[1.0, 2.0, -3.0], vel=[0.4, -0.2, 0.3],
        attitude=[0.08, -0.05, 0.2], body_rates=[0.3, -0.1, 0.05],
        thrust_accel=11.0,
    ),
    "near_ground": dict(pos=[0.0, 0.0, -0.4], vel=[0.05, 0.0, 0.6],
                        attitude=[0.02, 0.01, 0.0]),
    "fast_lateral": dict(pos=[-2.0, 1.5, -2.0], vel=[1.2, -0.9, 0.1],
                         attitude=[-0.1, 0.12, -0.3], body_rates=[-0.2, 0.4, 0.0]),
}


def _make_state(seed_fields: dict) -> VehicleState:
    d = dict(
        pos=[0.0, 0.0, -2.0], vel=[0.0, 0.0, 0.0], accel=[0.0, 0.0, 0.0],
        prev_accel=[0.0, 0.0, 0.0], attitude=[0.0, 0.0, 0.0],
        body_rates=[0.0, 0.0, 0.0], attitude_setpoint=[0.0, 0.0, 0.0],
        accel_cmd=[0.0, 0.0, 0.0],
    )
    d.update({k: v for k, v in seed_fields.items()})
    g = 9.8065
    return VehicleState(
        pos=np.array(d["pos"], dtype=np.float64),
        vel=np.array(d["vel"], dtype=np.float64),
        accel=np.array(d["accel"], dtype=np.float64),
        prev_accel=np.array(d["prev_accel"], dtype=np.float64),
        attitude=np.array(d["attitude"], dtype=np.float64),
        yaw_rate=float(seed_fields.get("yaw_rate", 0.0)),
        body_rates=np.array(d["body_rates"], dtype=np.float64),
        thrust_accel=float(seed_fields.get("thrust_accel", g)),
        attitude_setpoint=np.array(d["attitude_setpoint"], dtype=np.float64),
        thrust_accel_setpoint=float(seed_fields.get("thrust_accel_setpoint", g)),
        accel_cmd=np.array(d["accel_cmd"], dtype=np.float64),
        ground_effect_factor=float(seed_fields.get("ground_effect_factor", 1.0)),
    )


def _make_replica(cfg, state: VehicleState, *, target_yaw, wind_accel,
                  body_rate_alpha, thrust_alpha, motor_cutoff, ground_contact,
                  rng):
    rep = _LegacyFreeFlightReplica(cfg)
    rep.pos = state.pos.copy()
    rep.vel = state.vel.copy()
    rep.accel = state.accel.copy()
    rep.prev_accel = state.prev_accel.copy()
    rep.attitude = state.attitude.copy()
    rep.yaw_rate = state.yaw_rate
    rep.body_rates = state.body_rates.copy()
    rep.thrust_accel = state.thrust_accel
    rep.attitude_setpoint = state.attitude_setpoint.copy()
    rep.thrust_accel_setpoint = state.thrust_accel_setpoint
    rep.accel_cmd = state.accel_cmd.copy()
    rep.ground_effect_factor = state.ground_effect_factor
    rep.target_yaw = target_yaw
    rep.wind_accel = np.array(wind_accel, dtype=np.float64)
    rep.body_rate_response_alpha = np.array(body_rate_alpha, dtype=np.float64)
    rep.thrust_response_alpha = thrust_alpha
    rep.motor_cutoff = motor_cutoff
    rep.ground_contact = ground_contact
    rep.np_random = rng
    return rep


class LegacyDynamicsReplicaParityTest(unittest.TestCase):
    maxDiff = None

    _WINDS = ([0.0, 0.0, 0.0], [0.08, -0.03, 0.02])
    _DTS = (0.03, 0.05, 0.08)
    _SEEDS = (0, 1, 2, 7, 5000)

    def _cfgs(self):
        return {
            "default": NewLandingConfig(),
            "zero_process_noise": NewLandingConfig(
                body_rate_process_noise_std_radps=0.0,
                thrust_process_noise_std_mps2=0.0,
                process_noise_vel_std_mps=0.0,
            ),
            "ground_effect_on": NewLandingConfig(ground_effect_gain=0.18),
        }

    def test_advance_free_flight_matches_legacy_replica(self):
        for cname, cfg in self._cfgs().items():
            pn = ProcessNoiseSampler(cfg)
            lvd = LegacyVehicleDynamics(cfg, pn)
            for sname, seed_fields in _INIT_STATES.items():
                for motor_cutoff in (False, True):
                    for ground_contact in (False, True):
                        for wind in self._WINDS:
                            for dt in self._DTS:
                                for seed in self._SEEDS:
                                    with self.subTest(cfg=cname, state=sname,
                                                      motor_cutoff=motor_cutoff,
                                                      ground_contact=ground_contact,
                                                      wind=wind, dt=dt, seed=seed):
                                        self._one(lvd, cfg, seed_fields, motor_cutoff,
                                                  ground_contact, wind, dt, seed)

    def _one(self, lvd, cfg, seed_fields, motor_cutoff, ground_contact, wind, dt, seed):
        v_cmd = np.array([0.15, -0.20, 0.35], dtype=np.float64)
        target_yaw = 0.0
        bra = [0.4, 0.4, 0.4]
        ta = 0.30

        st = _make_state(seed_fields)
        rng_new = np.random.default_rng(seed)
        lvd.advance_free_flight(
            st, v_cmd, dt, rng_new, np.array(wind, dtype=np.float64),
            target_yaw, np.array(bra, dtype=np.float64), ta,
            motor_cutoff, ground_contact,
        )

        rng_old = np.random.default_rng(seed)
        rep = _make_replica(
            cfg, _make_state(seed_fields), target_yaw=target_yaw, wind_accel=wind,
            body_rate_alpha=bra, thrust_alpha=ta, motor_cutoff=motor_cutoff,
            ground_contact=ground_contact, rng=rng_old,
        )
        rep.free_flight(v_cmd, dt)

        for f in _ARR_FIELDS:
            self.assertTrue(
                np.array_equal(getattr(st, f), getattr(rep, f)),
                f"{f}: new={getattr(st, f)!r} old={getattr(rep, f)!r}",
            )
        for f in _SCALAR_FIELDS:
            self.assertEqual(getattr(st, f), getattr(rep, f), f)
        self.assertTrue(
            _rng_state_equal(rng_new.bit_generator.state, rng_old.bit_generator.state),
            "RNG state after free-flight",
        )

    def test_exactly_three_normal_draws(self):
        cfg = NewLandingConfig()
        lvd = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
        st = _make_state(_INIT_STATES["descending"])
        rng = np.random.default_rng(0)
        lvd.advance_free_flight(
            st, np.zeros(3), 0.05, rng, np.zeros(3), 0.0,
            np.array([0.4, 0.4, 0.4]), 0.30, False, False,
        )
        ref = np.random.default_rng(0)
        ref.normal(0.0, 1.0, size=3)   # body-rate
        ref.normal(0.0, 1.0)           # thrust (scalar)
        ref.normal(0.0, 1.0, size=3)   # translational
        self.assertTrue(_rng_state_equal(rng.bit_generator.state, ref.bit_generator.state))


class LegacyDynamicsStructuralTest(unittest.TestCase):
    """Section 19: no component-owned RNG, no env reference."""

    def test_backend_owns_only_cfg_and_process_noise(self):
        cfg = NewLandingConfig()
        lvd = LegacyVehicleDynamics(cfg, ProcessNoiseSampler(cfg))
        self.assertEqual(set(vars(lvd)), {"cfg", "process_noise"})
        for banned in ("np_random", "rng", "state", "env", "_env", "contact"):
            self.assertFalse(hasattr(lvd, banned), banned)

    def test_vehicle_state_has_no_rng_or_cfg(self):
        st = _make_state({})
        names = set(vars(st))
        self.assertEqual(names, set(_VS_FIELDS))
        for banned in ("rng", "np_random", "cfg", "motor_cutoff", "ground_contact",
                       "target_yaw", "wind_accel", "contact_count", "bounce_count"):
            self.assertNotIn(banned, names)

    def test_vehicle_state_field_set_is_frozen(self):
        self.assertEqual(
            _VS_FIELDS,
            ("pos", "vel", "accel", "prev_accel", "attitude", "yaw_rate",
             "body_rates", "thrust_accel", "attitude_setpoint",
             "thrust_accel_setpoint", "accel_cmd", "ground_effect_factor"),
        )


class VehicleStateMirrorContractTest(unittest.TestCase):
    """Section 16: the env exposes no stale vehicle mirror.

    Contract under test: after ``reset()`` and after every
    ``_update_rigid_body_dynamics`` (i.e. every ``step``), each ndarray mirror
    IS the same object as the corresponding ``VehicleState`` field (object
    identity), and each scalar mirror is value-equal.
    """

    maxDiff = None

    def _check(self, env, where):
        st = env._vehicle_state
        for f in _ARR_FIELDS:
            self.assertIs(getattr(env, f), getattr(st, f), f"{where}: {f} identity")
            self.assertTrue(np.array_equal(getattr(env, f), getattr(st, f)),
                            f"{where}: {f} value")
        for f in _SCALAR_FIELDS:
            self.assertEqual(getattr(env, f), getattr(st, f), f"{where}: {f}")

    def test_mirrors_never_stale_across_reset_and_steps(self):
        for pattern in ("zero", "pseudo_random", "saturation", "descend_z"):
            for seed in (0, 1, 5000):
                with self.subTest(pattern=pattern, seed=seed):
                    env = NewLandingEnv(NewLandingConfig())
                    try:
                        env.reset(seed=seed)
                        self._check(env, f"post-reset[{pattern},{seed}]")
                        actions = make_action_sequence(pattern, 200)
                        for i in range(200):
                            _, _, term, trunc, _ = env.step(actions[i].copy())
                            self._check(env, f"post-step {i}[{pattern},{seed}]")
                            if term or trunc:
                                break
                    finally:
                        env.close()


class FreeFlightContactIntegrationTest(unittest.TestCase):
    """Section 18: LegacyVehicleDynamics -> ContactModel -> final accel recompute
    matches OLD exactly through soft contact, hard contact, bounce, and the
    motor-cutoff feed-forward. OLD env vs NEW env, lockstep."""

    maxDiff = None

    _BOUNCE = dict(
        contact_enabled=True, init_altitude_min_m=2.0, init_altitude_max_m=2.5,
        bounce_vz_threshold_mps=0.05, touchdown_vz_soft_mps=0.01,
    )

    def _run(self, overrides, seeds):
        seen = {"soft": False, "hard": False, "bounce": False, "cutoff_multistep": False}
        for seed in seeds:
            with self.subTest(seed=seed, overrides=tuple(sorted(overrides))):
                old_env = OldLandingEnv(OldLandingConfig(**overrides))
                new_env = NewLandingEnv(NewLandingConfig(**overrides))
                try:
                    old_env.reset(seed=seed)
                    new_env.reset(seed=seed)
                    actions = make_action_sequence("descend_z", 420)
                    cutoff_run = 0
                    for i in range(420):
                        a = actions[i].copy()
                        o = old_env.step(a.copy())
                        n = new_env.step(a.copy())
                        oi, ni = o[4], n[4]
                        for key in ("pos", "vel", "accel", "prev_accel"):
                            self.assertTrue(
                                np.array_equal(oi[key], ni[key]),
                                f"[seed={seed} step={i}] {key}: old={oi[key]!r} new={ni[key]!r}",
                            )
                        for key in ("thrust_accel", "thrust_accel_setpoint",
                                    "ground_effect_factor", "motor_cutoff",
                                    "soft_contact", "hard_contact", "bounced",
                                    "touchdown_quality", "last_bounce_speed",
                                    "bounce_count", "jerk_mag"):
                            self.assertEqual(oi[key], ni[key],
                                             f"[seed={seed} step={i}] {key}")
                        self.assertEqual(float(o[1]), float(n[1]),
                                         f"[seed={seed} step={i}] reward")
                        self.assertEqual(bool(o[2]), bool(n[2]))
                        self.assertEqual(bool(o[3]), bool(n[3]))
                        self.assertTrue(_rng_state_equal(
                            old_env.np_random.bit_generator.state,
                            new_env.np_random.bit_generator.state),
                            f"[seed={seed} step={i}] RNG")
                        term = o[2] or o[3]
                        if ni["soft_contact"]:
                            seen["soft"] = True
                        if ni["hard_contact"]:
                            seen["hard"] = True
                        if ni["bounced"]:
                            seen["bounce"] = True
                        if ni["motor_cutoff"] and not term:
                            cutoff_run += 1
                            if cutoff_run >= 2:
                                seen["cutoff_multistep"] = True
                        elif not ni["motor_cutoff"]:
                            cutoff_run = 0
                        if term:
                            break
                finally:
                    old_env.close()
                    new_env.close()
        return seen

    def test_default_config_soft_and_cutoff(self):
        seen = self._run({}, (0, 1, 2, 3, 5))
        self.assertTrue(seen["soft"], "no soft contact exercised")
        self.assertTrue(seen["cutoff_multistep"],
                        "no >=2-step motor-cutoff window exercised")

    def test_forced_bounce_config(self):
        seen = self._run(self._BOUNCE, (0, 1, 2, 3, 4))
        self.assertTrue(seen["bounce"], "forced-bounce config never bounced")


if __name__ == "__main__":
    unittest.main()
