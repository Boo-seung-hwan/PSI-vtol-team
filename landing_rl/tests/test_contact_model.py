"""Focused tests for the Phase 13B ContactModel extraction.

Secondary to the OLD-vs-NEW environment regression in
``test_legacy_regression_contract.py`` and the checkpoint gate in
``test_checkpoint_compatibility.py``, both of which remain authoritative.

ContactModel holds the legacy ground-contact logic lifted verbatim from
``mujoco_rl/envs/env_prototype.py`` (``_reset_contact_step_flags`` and
``_apply_ground_contact``). These tests compare it against a verbatim in-file
replica of that legacy method and, for the RNG-order gate, against the real
OLD environment stepped in lockstep with the NEW one.

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

from landing_rl.contact import ContactModel, ContactResult, ContactState


# ---------------------------------------------------------------------------
# Verbatim replica of the legacy contact code (self.<x> -> holder attribute).
# Copied character-for-character from mujoco_rl/envs/env_prototype.py
# _reset_contact_step_flags / _apply_ground_contact.
# ---------------------------------------------------------------------------
class _LegacyContactReplica:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.attitude = np.zeros(3, dtype=np.float64)
        self.thrust_accel = float(cfg.gravity_mps2)
        self.thrust_accel_setpoint = float(cfg.gravity_mps2)
        self.np_random = np.random.default_rng(0)

        self.ground_contact = False
        self.contact_count = 0
        self.bounce_count = 0
        self.motor_cutoff = False
        self.contact_event = False
        self.soft_contact = False
        self.hard_contact = False
        self.bounced = False
        self.touchdown_quality = "none"
        self.last_impact_vz = 0.0
        self.last_touchdown_vxy = 0.0
        self.last_bounce_speed = 0.0

    def _reset_contact_step_flags(self) -> None:
        self.contact_event = False
        self.soft_contact = False
        self.hard_contact = False
        self.bounced = False
        self.touchdown_quality = "none"
        self.last_impact_vz = 0.0
        self.last_touchdown_vxy = 0.0
        self.last_bounce_speed = 0.0

    def _apply_ground_contact(self, vel_before_step: np.ndarray) -> None:
        if not self.cfg.contact_enabled:
            return

        ground_z = float(self.cfg.ground_z_m)
        if self.pos[2] < ground_z:
            self.ground_contact = False
            self.contact_count = 0
            return

        self.contact_event = True
        self.ground_contact = True
        self.contact_count += 1
        self.pos[2] = ground_z

        impact_vz = max(float(self.vel[2]), float(vel_before_step[2]), 0.0)
        touchdown_vxy = float(np.linalg.norm(self.vel[:2]))
        tilt = float(np.linalg.norm(self.attitude[:2]))

        self.last_impact_vz = impact_vz
        self.last_touchdown_vxy = touchdown_vxy

        self.hard_contact = (
            impact_vz > self.cfg.hard_touchdown_vz_mps
            or touchdown_vxy > self.cfg.hard_touchdown_vxy_mps
            or tilt > self.cfg.hard_touchdown_tilt_rad
        )

        self.soft_contact = (
            impact_vz <= self.cfg.touchdown_vz_soft_mps
            and touchdown_vxy <= self.cfg.touchdown_vxy_soft_mps
            and tilt <= self.cfg.touchdown_tilt_soft_rad
            and not self.hard_contact
        )

        if self.hard_contact:
            self.touchdown_quality = "hard"
        elif self.soft_contact:
            self.touchdown_quality = "soft"
        else:
            self.touchdown_quality = "rough"

        self.vel[:2] *= float(np.clip(1.0 - self.cfg.ground_friction_xy, 0.0, 1.0))

        should_bounce = (
            impact_vz > self.cfg.bounce_vz_threshold_mps
            and not self.soft_contact
        )

        if should_bounce:
            restitution = float(self.np_random.uniform(
                self.cfg.bounce_restitution_min,
                self.cfg.bounce_restitution_max,
            ))
            bounce_speed = restitution * impact_vz
            self.vel[2] = -bounce_speed
            self.bounced = True
            self.bounce_count += 1
            self.contact_count = 0
            self.last_bounce_speed = bounce_speed
            if not self.hard_contact:
                self.touchdown_quality = "bounce"
        else:
            self.vel[2] = 0.0
            self.last_bounce_speed = 0.0

        if self.soft_contact and self.cfg.motor_cutoff_on_soft_contact:
            self.motor_cutoff = True
            self.thrust_accel_setpoint = float(self.cfg.motor_cutoff_thrust_accel_mps2)
            self.thrust_accel = float(self.cfg.motor_cutoff_thrust_accel_mps2)


_PERSIST = ("ground_contact", "contact_count", "bounce_count", "motor_cutoff")
_TRANSIENT = (
    "contact_event", "soft_contact", "hard_contact", "bounced",
    "touchdown_quality", "last_impact_vz", "last_touchdown_vxy",
    "last_bounce_speed",
)


def _cfg(**kw):
    return NewLandingConfig(**kw)


def _run_new(cfg, pos, vel, attitude, vel_before, seed):
    """Run ContactModel.begin_step()+apply() once; return an outcome dict."""
    cm = ContactModel(cfg)
    cm.begin_step()
    p = np.array(pos, dtype=np.float64)
    v = np.array(vel, dtype=np.float64)
    a = np.array(attitude, dtype=np.float64)
    rng = np.random.default_rng(seed)
    result, th, thsp = cm.apply(
        pos=p, vel=v, attitude=a,
        vel_before_step=np.array(vel_before, dtype=np.float64),
        thrust_accel=float(cfg.gravity_mps2),
        thrust_accel_setpoint=float(cfg.gravity_mps2),
        rng=rng,
    )
    out = {"pos": p, "vel": v, "thrust_accel": th, "thrust_accel_setpoint": thsp,
           "rng_state": rng.bit_generator.state}
    for k in _PERSIST:
        out[k] = getattr(cm.state, k)
    for k in _TRANSIENT:
        out[k] = getattr(result, k)
    return out, cm


def _run_replica(cfg, pos, vel, attitude, vel_before, seed):
    rep = _LegacyContactReplica(cfg)
    rep.pos = np.array(pos, dtype=np.float64)
    rep.vel = np.array(vel, dtype=np.float64)
    rep.attitude = np.array(attitude, dtype=np.float64)
    rep.np_random = np.random.default_rng(seed)
    rep._reset_contact_step_flags()
    rep._apply_ground_contact(np.array(vel_before, dtype=np.float64))
    out = {"pos": rep.pos, "vel": rep.vel,
           "thrust_accel": rep.thrust_accel,
           "thrust_accel_setpoint": rep.thrust_accel_setpoint,
           "rng_state": rep.np_random.bit_generator.state}
    for k in _PERSIST + _TRANSIENT:
        out[k] = getattr(rep, k)
    return out


# scenario -> (pos, vel, attitude, vel_before)
_SCENARIOS = {
    "airborne":   ([0.3, -0.4, -1.0], [0.1, 0.0, 0.2],  [0.02, 0.0, 0.0], [0.0, 0.0, 0.2]),
    "soft":       ([0.0, 0.0, 0.01],  [0.1, 0.1, 0.15], [0.05, 0.05, 0.0], [0.0, 0.0, 0.15]),
    "rough_slow": ([0.0, 0.0, 0.02],  [0.5, 0.0, 0.20], [0.02, 0.0, 0.0],  [0.0, 0.0, 0.20]),
    "hard":       ([0.0, 0.0, 0.05],  [0.0, 0.0, 1.00], [0.0, 0.0, 0.0],   [0.0, 0.0, 1.00]),
    "bounce":     ([1.5, -2.3, 0.05], [0.5, 0.3, 0.40], [0.03, 0.01, 0.0], [0.0, 0.0, 0.40]),
}


class ContactModelUnitTest(unittest.TestCase):
    maxDiff = None

    # -- A. constructor + reset consume zero RNG -----------------------
    def test_constructor_and_reset_consume_zero_rng(self):
        rng = np.random.default_rng(123)
        before = rng.bit_generator.state
        cm = ContactModel(_cfg())
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        cm.reset()
        self.assertTrue(_rng_state_equal(before, rng.bit_generator.state))
        # ownership: cfg + state + result only
        self.assertEqual(set(vars(cm)), {"cfg", "state", "result"})
        self.assertFalse(hasattr(cm, "np_random"))
        self.assertFalse(hasattr(cm, "rng"))

    def test_reset_restores_exact_legacy_defaults(self):
        cm = ContactModel(_cfg())
        cm.state.ground_contact = True
        cm.state.contact_count = 7
        cm.state.bounce_count = 3
        cm.state.motor_cutoff = True
        cm.result.contact_event = True
        cm.result.touchdown_quality = "hard"
        cm.result.last_impact_vz = 9.0
        cm.reset()
        self.assertEqual(
            (cm.state.ground_contact, cm.state.contact_count,
             cm.state.bounce_count, cm.state.motor_cutoff),
            (False, 0, 0, False),
        )
        self.assertEqual(vars(cm.result), vars(ContactResult()))

    # -- B. begin_step resets 8 transient, no persistent -------------
    def test_begin_step_resets_only_transient(self):
        cm = ContactModel(_cfg())
        cm.state.ground_contact = True
        cm.state.contact_count = 4
        cm.state.bounce_count = 2
        cm.state.motor_cutoff = True
        for k in _TRANSIENT:
            setattr(cm.result, k, ("x" if k == "touchdown_quality" else
                                   (True if isinstance(getattr(cm.result, k), bool) else 5.0)))
        cm.begin_step()
        self.assertEqual(vars(cm.result), vars(ContactResult()))  # all 8 back to legacy defaults
        # persistent untouched
        self.assertEqual(
            (cm.state.ground_contact, cm.state.contact_count,
             cm.state.bounce_count, cm.state.motor_cutoff),
            (True, 4, 2, True),
        )

    # -- C. contact_enabled=False: nothing happens, no RNG ----------
    def test_contact_disabled_is_a_noop(self):
        cfg = _cfg(contact_enabled=False)
        cm = ContactModel(cfg)
        cm.begin_step()
        cm.state.ground_contact = True      # pre-existing non-default persistent
        cm.state.contact_count = 3
        cm.state.bounce_count = 1
        cm.state.motor_cutoff = True
        p = np.array([0.1, 0.2, 0.05], dtype=np.float64)
        v = np.array([0.4, 0.3, 0.6], dtype=np.float64)
        p0, v0 = p.copy(), v.copy()
        rng = np.random.default_rng(9)
        st_before = rng.bit_generator.state
        result, th, thsp = cm.apply(
            pos=p, vel=v, attitude=np.zeros(3),
            vel_before_step=np.array([0.0, 0.0, 0.6]),
            thrust_accel=1.23, thrust_accel_setpoint=4.56, rng=rng,
        )
        self.assertTrue(np.array_equal(p, p0))            # pos untouched
        self.assertTrue(np.array_equal(v, v0))            # vel untouched
        self.assertEqual((th, thsp), (1.23, 4.56))        # scalars passed through
        self.assertTrue(_rng_state_equal(st_before, rng.bit_generator.state))  # no draw
        self.assertEqual(
            (cm.state.ground_contact, cm.state.contact_count,
             cm.state.bounce_count, cm.state.motor_cutoff),
            (True, 3, 1, True),                            # persistent untouched
        )
        self.assertEqual(vars(result), vars(ContactResult()))  # transient still default

    # -- D. airborne branch ----------------------------------------
    def test_airborne_branch(self):
        cfg = _cfg()
        cm = ContactModel(cfg)
        cm.begin_step()
        cm.state.ground_contact = True
        cm.state.contact_count = 5
        cm.state.bounce_count = 2
        cm.state.motor_cutoff = True
        p = np.array([0.0, 0.0, -0.5], dtype=np.float64)   # 0.5 m above ground
        v = np.array([0.1, 0.1, 0.2], dtype=np.float64)
        p0, v0 = p.copy(), v.copy()
        rng = np.random.default_rng(3)
        st = rng.bit_generator.state
        result, th, thsp = cm.apply(
            pos=p, vel=v, attitude=np.zeros(3),
            vel_before_step=np.array([0.0, 0.0, 0.2]),
            thrust_accel=9.0, thrust_accel_setpoint=9.0, rng=rng,
        )
        self.assertEqual((cm.state.ground_contact, cm.state.contact_count), (False, 0))
        self.assertEqual((cm.state.bounce_count, cm.state.motor_cutoff), (2, True))  # unchanged
        self.assertTrue(np.array_equal(p, p0))
        self.assertTrue(np.array_equal(v, v0))
        self.assertTrue(_rng_state_equal(st, rng.bit_generator.state))
        self.assertEqual(vars(result), vars(ContactResult()))

    # -- E/F/G/H. classification branches vs replica --------------
    def test_scenarios_match_legacy_replica(self):
        for name, (pos, vel, att, vb) in _SCENARIOS.items():
            for seed in (0, 1, 7, 5000):
                with self.subTest(scenario=name, seed=seed):
                    new_out, _ = _run_new(_cfg(), pos, vel, att, vb, seed)
                    old_out = _run_replica(_cfg(), pos, vel, att, vb, seed)
                    self.assertTrue(np.array_equal(new_out["pos"], old_out["pos"]), "pos")
                    self.assertTrue(np.array_equal(new_out["vel"], old_out["vel"]), "vel")
                    self.assertEqual(new_out["thrust_accel"], old_out["thrust_accel"])
                    self.assertEqual(new_out["thrust_accel_setpoint"],
                                     old_out["thrust_accel_setpoint"])
                    for k in _PERSIST + _TRANSIENT:
                        self.assertEqual(new_out[k], old_out[k], k)
                    self.assertTrue(_rng_state_equal(new_out["rng_state"],
                                                     old_out["rng_state"]), "rng state")

    def test_soft_touchdown_specifics(self):
        cfg = _cfg()
        out, cm = _run_new(cfg, *_SCENARIOS["soft"], seed=0)
        self.assertTrue(out["soft_contact"])
        self.assertFalse(out["hard_contact"])
        self.assertEqual(out["touchdown_quality"], "soft")
        self.assertEqual(out["contact_count"], 1)
        self.assertEqual(float(out["vel"][2]), 0.0)               # no bounce -> vz = 0
        self.assertAlmostEqual(float(out["vel"][0]), 0.1 * 0.45, places=15)  # friction
        self.assertEqual(out["last_impact_vz"], 0.15)
        self.assertTrue(cm.state.motor_cutoff)                    # soft cutoff latched

    def test_rough_touchdown_specifics(self):
        out, _ = _run_new(_cfg(), *_SCENARIOS["rough_slow"], seed=0)
        self.assertFalse(out["soft_contact"])
        self.assertFalse(out["hard_contact"])
        self.assertFalse(out["bounced"])
        self.assertEqual(out["touchdown_quality"], "rough")
        self.assertEqual(float(out["vel"][2]), 0.0)

    def test_hard_touchdown_specifics(self):
        out, _ = _run_new(_cfg(), *_SCENARIOS["hard"], seed=0)
        self.assertTrue(out["hard_contact"])
        self.assertFalse(out["soft_contact"])
        self.assertTrue(out["bounced"])                          # impact 1.0 > bounce thr
        self.assertEqual(out["touchdown_quality"], "hard")       # NOT overwritten to "bounce"
        self.assertEqual(out["bounce_count"], 1)
        self.assertEqual(out["contact_count"], 0)                # reset on bounce

    def test_bounce_specifics(self):
        cfg = _cfg()
        out, _ = _run_new(cfg, *_SCENARIOS["bounce"], seed=0)
        self.assertTrue(out["bounced"])
        self.assertFalse(out["hard_contact"])
        self.assertEqual(out["touchdown_quality"], "bounce")
        self.assertEqual(out["bounce_count"], 1)
        self.assertEqual(out["contact_count"], 0)
        # vel[2] == -restitution * impact_vz  (impact_vz == 0.40)
        rng = np.random.default_rng(0)
        restit = float(rng.uniform(cfg.bounce_restitution_min, cfg.bounce_restitution_max))
        self.assertAlmostEqual(float(out["vel"][2]), -(restit * 0.40), places=15)
        self.assertAlmostEqual(out["last_bounce_speed"], restit * 0.40, places=15)

    # -- I. no-bounce contact consumes no uniform draw ------------
    def test_no_bounce_consumes_no_rng(self):
        for name in ("soft", "rough_slow"):
            with self.subTest(scenario=name):
                cfg = _cfg()
                cm = ContactModel(cfg)
                cm.begin_step()
                rng = np.random.default_rng(11)
                st = rng.bit_generator.state
                pos, vel, att, vb = _SCENARIOS[name]
                cm.apply(
                    pos=np.array(pos, float), vel=np.array(vel, float),
                    attitude=np.array(att, float),
                    vel_before_step=np.array(vb, float),
                    thrust_accel=9.0, thrust_accel_setpoint=9.0, rng=rng,
                )
                self.assertTrue(_rng_state_equal(st, rng.bit_generator.state))

    def test_bounce_consumes_exactly_one_uniform_draw(self):
        cfg = _cfg()
        cm = ContactModel(cfg)
        cm.begin_step()
        rng = np.random.default_rng(4)
        ref = np.random.default_rng(4)
        pos, vel, att, vb = _SCENARIOS["bounce"]
        cm.apply(pos=np.array(pos, float), vel=np.array(vel, float),
                 attitude=np.array(att, float), vel_before_step=np.array(vb, float),
                 thrust_accel=9.0, thrust_accel_setpoint=9.0, rng=rng)
        ref.uniform(cfg.bounce_restitution_min, cfg.bounce_restitution_max)
        self.assertTrue(_rng_state_equal(rng.bit_generator.state, ref.bit_generator.state))

    # -- J. lateral friction touches only vel[:2] ----------------
    def test_friction_only_affects_vel_xy(self):
        # rough, non-bounce so vz is set to 0 deterministically; check xy scaling
        cfg = _cfg()
        pos = [0.0, 0.0, 0.02]
        vel = [0.5, 0.3, 0.20]
        out, _ = _run_new(cfg, pos, vel, [0.0, 0.0, 0.0], [0.0, 0.0, 0.20], seed=0)
        k = float(np.clip(1.0 - cfg.ground_friction_xy, 0.0, 1.0))
        self.assertAlmostEqual(float(out["vel"][0]), 0.5 * k, places=15)
        self.assertAlmostEqual(float(out["vel"][1]), 0.3 * k, places=15)
        self.assertEqual(float(out["vel"][2]), 0.0)

    # -- K. position clamp touches only pos[2] ------------------
    def test_position_clamp_only_z(self):
        cfg = _cfg()
        out, _ = _run_new(cfg, [1.5, -2.3, 0.08], [0.0, 0.0, 0.1],
                          [0.0, 0.0, 0.0], [0.0, 0.0, 0.1], seed=0)
        self.assertEqual(float(out["pos"][0]), 1.5)
        self.assertEqual(float(out["pos"][1]), -2.3)
        self.assertEqual(float(out["pos"][2]), float(cfg.ground_z_m))

    # -- L. soft-contact motor cutoff returns cutoff scalars ----
    def test_soft_contact_motor_cutoff_scalars(self):
        cfg = _cfg()
        cm = ContactModel(cfg)
        cm.begin_step()
        pos, vel, att, vb = _SCENARIOS["soft"]
        result, th, thsp = cm.apply(
            pos=np.array(pos, float), vel=np.array(vel, float),
            attitude=np.array(att, float), vel_before_step=np.array(vb, float),
            thrust_accel=9.8065, thrust_accel_setpoint=9.8065,
            rng=np.random.default_rng(0),
        )
        self.assertTrue(cm.state.motor_cutoff)
        self.assertEqual(th, float(cfg.motor_cutoff_thrust_accel_mps2))
        self.assertEqual(thsp, float(cfg.motor_cutoff_thrust_accel_mps2))

    # -- M. motor_cutoff stays latched over later calls --------
    def test_motor_cutoff_latched_until_reset(self):
        cfg = _cfg()
        cm = ContactModel(cfg)
        cm.begin_step()
        pos, vel, att, vb = _SCENARIOS["soft"]
        cm.apply(pos=np.array(pos, float), vel=np.array(vel, float),
                 attitude=np.array(att, float), vel_before_step=np.array(vb, float),
                 thrust_accel=9.8, thrust_accel_setpoint=9.8, rng=np.random.default_rng(0))
        self.assertTrue(cm.state.motor_cutoff)
        # a later airborne call must NOT clear it
        cm.begin_step()
        cm.apply(pos=np.array([0.0, 0.0, -1.0]), vel=np.array([0.0, 0.0, 0.1]),
                 attitude=np.zeros(3), vel_before_step=np.array([0.0, 0.0, 0.1]),
                 thrust_accel=9.8, thrust_accel_setpoint=9.8, rng=np.random.default_rng(0))
        self.assertTrue(cm.state.motor_cutoff)
        # only reset() clears it
        cm.reset()
        self.assertFalse(cm.state.motor_cutoff)

    # -- N. broad legacy-replica matrix incl. RNG state -------
    def test_replica_matrix_exact_including_rng_state(self):
        cfgs = {
            "default": _cfg(),
            "no_cutoff": _cfg(motor_cutoff_on_soft_contact=False),
            "low_friction": _cfg(ground_friction_xy=0.05),
            "low_bounce_thr": _cfg(bounce_vz_threshold_mps=0.05,
                                   touchdown_vz_soft_mps=0.01),
        }
        for cname, cfg in cfgs.items():
            for sname, (pos, vel, att, vb) in _SCENARIOS.items():
                for seed in (0, 2, 42, 777):
                    with self.subTest(cfg=cname, scenario=sname, seed=seed):
                        new_out, _ = _run_new(cfg, pos, vel, att, vb, seed)
                        old_out = _run_replica(cfg, pos, vel, att, vb, seed)
                        self.assertTrue(np.array_equal(new_out["pos"], old_out["pos"]))
                        self.assertTrue(np.array_equal(new_out["vel"], old_out["vel"]))
                        self.assertEqual(new_out["thrust_accel"], old_out["thrust_accel"])
                        self.assertEqual(new_out["thrust_accel_setpoint"],
                                         old_out["thrust_accel_setpoint"])
                        for k in _PERSIST + _TRANSIENT:
                            self.assertEqual(new_out[k], old_out[k], k)
                        self.assertTrue(_rng_state_equal(
                            new_out["rng_state"], old_out["rng_state"]))


# ---------------------------------------------------------------------------
# Section 21: forced-bounce RNG-order gate (OLD env vs NEW env, lockstep).
# ---------------------------------------------------------------------------
_BOUNCE_OVERRIDES = dict(
    contact_enabled=True,
    init_altitude_min_m=2.0,
    init_altitude_max_m=2.5,
    bounce_vz_threshold_mps=0.05,
    touchdown_vz_soft_mps=0.01,
)


class ContactRngOrderGateTest(unittest.TestCase):
    maxDiff = None

    def test_forced_bounce_old_vs_new_lockstep(self):
        seeds = (0, 1, 2, 3, 4)
        any_bounce = False
        for seed in seeds:
            with self.subTest(seed=seed):
                old_env = OldLandingEnv(OldLandingConfig(**_BOUNCE_OVERRIDES))
                new_env = NewLandingEnv(NewLandingConfig(**_BOUNCE_OVERRIDES))
                try:
                    old_env.reset(seed=seed)
                    new_env.reset(seed=seed)
                    self.assertTrue(_rng_state_equal(
                        old_env.np_random.bit_generator.state,
                        new_env.np_random.bit_generator.state), "post-reset RNG")
                    actions = make_action_sequence("descend_z", 420)
                    saw_bounce = False
                    for i in range(420):
                        a = actions[i].copy()
                        o = old_env.step(a.copy())
                        n = new_env.step(a.copy())
                        # RNG state exact every step -> the 3 process-noise
                        # normals AND the conditional restitution uniform were
                        # consumed in the same order and count.
                        self.assertTrue(_rng_state_equal(
                            old_env.np_random.bit_generator.state,
                            new_env.np_random.bit_generator.state),
                            f"[seed={seed} step={i}] RNG diverged")
                        oi, ni = o[4], n[4]
                        self.assertEqual(bool(oi["bounced"]), bool(ni["bounced"]),
                                         f"[seed={seed} step={i}] bounced flag")
                        self.assertTrue(np.array_equal(oi["vel"], ni["vel"]),
                                        f"[seed={seed} step={i}] post-contact vel")
                        self.assertTrue(np.array_equal(oi["accel"], ni["accel"]),
                                        f"[seed={seed} step={i}] final accel")
                        self.assertEqual(oi["last_bounce_speed"], ni["last_bounce_speed"])
                        self.assertEqual(oi["bounce_count"], ni["bounce_count"])
                        self.assertEqual(float(o[1]), float(n[1]),
                                         f"[seed={seed} step={i}] reward")
                        if oi["bounced"]:
                            saw_bounce = True
                        if o[2] or o[3]:
                            break
                    any_bounce = any_bounce or saw_bounce
                finally:
                    old_env.close()
                    new_env.close()
        self.assertTrue(any_bounce,
                        "forced-bounce config never actually bounced across seeds "
                        "0-4; the RNG-order gate did not exercise the restitution "
                        "uniform")


class MotorCutoffFeedbackTest(unittest.TestCase):
    """Section 17: multi-step motor-cutoff feedback loop.

    Step N (D26): ContactModel latches ``state.motor_cutoff = True`` on a soft
    touchdown and forces the thrust scalars to the cutoff value.
    Step N+1 (D5/D16): ``_update_rigid_body_dynamics`` reads the synchronized
    ``self.motor_cutoff`` mirror and clamps ``thrust_sp`` / ``thrust_min`` to
    the cutoff. If the mirror or the latch broke, NEW would diverge from OLD on
    step N+1. We assert full OLD-vs-NEW parity every step AND that a
    genuine multi-step latched window (cutoff True, episode still running) is
    exercised.
    """

    maxDiff = None

    def test_cutoff_latches_and_feeds_forward_old_vs_new(self):
        observed_multistep_window = False
        for seed in (0, 1, 2, 3, 5):
            with self.subTest(seed=seed):
                old_env = OldLandingEnv(OldLandingConfig())
                new_env = NewLandingEnv(NewLandingConfig())
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
                        self.assertEqual(bool(oi["motor_cutoff"]),
                                         bool(ni["motor_cutoff"]),
                                         f"[seed={seed} step={i}] motor_cutoff")
                        self.assertEqual(oi["thrust_accel"], ni["thrust_accel"],
                                         f"[seed={seed} step={i}] thrust_accel")
                        self.assertEqual(oi["thrust_accel_setpoint"],
                                         ni["thrust_accel_setpoint"],
                                         f"[seed={seed} step={i}] thrust_accel_setpoint")
                        self.assertTrue(np.array_equal(oi["vel"], ni["vel"]),
                                        f"[seed={seed} step={i}] vel")
                        self.assertTrue(np.array_equal(oi["accel"], ni["accel"]),
                                        f"[seed={seed} step={i}] accel")
                        self.assertEqual(float(o[1]), float(n[1]),
                                         f"[seed={seed} step={i}] reward")
                        self.assertTrue(_rng_state_equal(
                            old_env.np_random.bit_generator.state,
                            new_env.np_random.bit_generator.state),
                            f"[seed={seed} step={i}] RNG")
                        terminated = o[2] or o[3]
                        if ni["motor_cutoff"] and not terminated:
                            cutoff_run += 1
                            if cutoff_run >= 2:
                                observed_multistep_window = True
                        if terminated:
                            break
                finally:
                    old_env.close()
                    new_env.close()
        self.assertTrue(
            observed_multistep_window,
            "no seed produced a >=2-step window with motor_cutoff latched and "
            "the episode still running; the D26->D5/D16 feed-forward was not "
            "exercised across steps",
        )


if __name__ == "__main__":
    unittest.main()
