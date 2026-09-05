"""Plant orchestration: free-flight dynamics + ground contact.

Phase 13C-2 extraction. This module lifts the orchestration that previously
lived directly in ``LandingEnv._update_rigid_body_dynamics`` -- the D2/D3/D26/
D27 span surrounding the already-extracted ``LegacyVehicleDynamics`` (Phase
13C-1) and ``ContactModel`` (Phase 13B) collaborators -- into a single seam:

    PlantModel.step(state, v_cmd, dt, rng, wind_accel, target_yaw,
                     body_rate_response_alpha, thrust_response_alpha)

        D2   contact.begin_step()
        D3   vel_before = state.vel.copy()
        D4-D25  dynamics.advance_free_flight(...)      (free flight)
        D26  contact.apply(...)                        (ground contact)
             state.thrust_accel / thrust_accel_setpoint write-back
        D27  state.accel = (state.vel - vel_before) / dt_safe

No behavior change and no new physics. ``LegacyVehicleDynamics`` arithmetic
and ``ContactModel`` logic are used unmodified and unchanged; ``PlantModel``
only reproduces the existing execution order that used to live inline in
``LandingEnv``.

Ownership
---------
Owns ONLY ``cfg``, ``dynamics`` (a ``LegacyVehicleDynamics``), and ``contact``
(a ``ContactModel``). Does NOT own an RNG, a ``VehicleState``, a ``LandingEnv``
reference, wind state, target state, or response-alpha state. ``rng`` must be
``LandingEnv.np_random``; ``PlantModel`` passes it straight through to
``dynamics`` and ``contact`` and never draws from it directly, never creates
its own generator, and never pre-samples / caches / reorders anything.

``motor_cutoff`` / ``ground_contact`` handed to
``dynamics.advance_free_flight`` are read from ``contact.state`` -- the
PERSISTENT contact state as it stands before this step's ``contact.apply()``
call. ``contact.begin_step()`` only clears the eight transient
``ContactResult`` fields, so ``contact.state.motor_cutoff`` /
``contact.state.ground_contact`` are still last step's latched values at that
point -- exactly the values the pre-extraction ``LandingEnv`` passed in (there
via the ``self.motor_cutoff`` / ``self.ground_contact`` compatibility mirrors,
which were synced immediately after ``begin_step()`` and therefore held the
identical persistent value at the point they were read).

``step()`` returns ``contact.result`` -- the same ``ContactResult`` object
``contact`` already owns, not a new object -- for a caller that wants the
transient contact outcome directly. ``LandingEnv`` does not use the return
value; it re-syncs its own flat compatibility mirrors from ``contact.state`` /
``contact.result`` instead, exactly as before.
"""

from __future__ import annotations


class PlantModel:
    """Orchestrates one control step of free-flight dynamics + ground contact.

    Stateless beyond ``cfg``, ``dynamics``, and ``contact``. Mutates the
    ``VehicleState`` handed to ``step()`` in place, and drives ``contact``'s
    own ``ContactState`` / ``ContactResult`` mutation (via ``contact``'s own
    ``begin_step()`` / ``apply()`` methods) in the same order the
    pre-extraction ``LandingEnv`` orchestration used.
    """

    def __init__(self, cfg, dynamics, contact):
        self.cfg = cfg
        self.dynamics = dynamics
        self.contact = contact

    def step(
        self,
        state,
        v_cmd,
        dt,
        rng,
        wind_accel,
        target_yaw,
        body_rate_response_alpha,
        thrust_response_alpha,
    ):
        """Advance ``state`` by one control step: free flight, then ground
        contact, then the final post-contact acceleration recompute.

        ``state`` (a ``VehicleState``) is mutated in place. ``rng`` must be
        ``LandingEnv.np_random``. Returns ``contact.result``.
        """
        dt_safe = max(float(dt), 1e-6)

        # D2: clear the eight transient contact flags. Touches no persistent
        # ContactState field.
        self.contact.begin_step()

        # D3: free-flight-before snapshot. Used by ground effect (inside
        # advance_free_flight, via contact.state.ground_contact) and by the
        # D27 recompute below.
        vel_before = state.vel.copy()

        # D4-D25: free-flight rigid-body integration. motor_cutoff /
        # ground_contact are the PERSISTENT contact state from before this
        # step's contact resolution (see module docstring).
        self.dynamics.advance_free_flight(
            state,
            v_cmd,
            dt_safe,
            rng,
            wind_accel,
            target_yaw,
            body_rate_response_alpha,
            thrust_response_alpha,
            self.contact.state.motor_cutoff,
            self.contact.state.ground_contact,
        )

        # D26: ground-contact resolution (mutates state.pos/vel/attitude in
        # place; may overwrite the thrust scalars on a soft touchdown).
        result, thrust_accel, thrust_accel_setpoint = self.contact.apply(
            pos=state.pos,
            vel=state.vel,
            attitude=state.attitude,
            vel_before_step=vel_before,
            thrust_accel=state.thrust_accel,
            thrust_accel_setpoint=state.thrust_accel_setpoint,
            rng=rng,
        )
        state.thrust_accel = thrust_accel
        state.thrust_accel_setpoint = thrust_accel_setpoint

        # D27: final observed acceleration, computed AFTER contact -- the
        # value the policy observes as ax/ay/az.
        state.accel = (state.vel - vel_before) / dt_safe

        return result
