"""VehicleDynamicsBackend -- the free-flight physics contract ``PlantModel``
depends on, instead of the concrete ``LegacyVehicleDynamics`` class.

This is the minimal interface seam described in the MuJoCo-integration
planning work: it exists so a future ``MuJoCoDynamics`` backend can be handed
to ``PlantModel`` in place of ``LegacyVehicleDynamics`` without touching
``PlantModel``, ``LandingEnv``, or any other collaborator. It is NOT a
MuJoCo implementation, and it changes NO runtime behavior by itself --
``PlantModel`` already accepted any object exposing a matching
``advance_free_flight`` method (Python duck typing), and
``landing_rl/tests/test_plant_model.py::PlantModelOrchestrationOrderTest``
already injects a hand-written fake dynamics object today. This module only
gives that existing, already-relied-upon contract a name, a single canonical
signature definition, and a structural (``isinstance``-checkable) type.

Contract
--------
Exactly the current signature and side effects of
``LegacyVehicleDynamics.advance_free_flight`` -- lifted verbatim, not
redesigned:

    * ``state`` (a ``VehicleState``) is mutated in place: this call is
      responsible for ``attitude_setpoint``, ``thrust_accel_setpoint``,
      ``accel_cmd``, ``body_rates``, ``attitude``, ``yaw_rate``,
      ``thrust_accel``, ``ground_effect_factor``, ``prev_accel``, ``accel``,
      ``vel``, and ``pos``. Field names, shapes, units, and the NED
      convention are exactly ``VehicleState``'s existing contract
      (unchanged by this module).
    * ground contact is explicitly OUT OF SCOPE for this call -- ``PlantModel``
      resolves it separately, immediately afterward, via
      ``ContactModel.apply()``. A backend must not resolve or mutate contact
      state itself.
    * ``rng`` is consumed however the concrete backend's own process-noise
      model requires. The *number and order* of RNG draws is part of each
      concrete backend's own determinism contract, not of this Protocol --
      ``LegacyVehicleDynamics`` consumes exactly three draws (body-rate,
      thrust, translational, in that order); a future backend may consume a
      different number/order, and no caller may assume otherwise.
    * returns ``None``.

``motor_cutoff`` / ``ground_contact`` are the PERSISTENT contact-state
booleans from BEFORE this step's contact resolution (see
``PlantModel.step``'s docstring) -- inputs only, never written by this call.

Why ``typing.Protocol`` (not ``abc.ABC``, not left as plain duck typing)
-------------------------------------------------------------------------
No file under ``landing_rl/`` uses ``abc``/``abstractmethod`` anywhere
(verified by repository-wide grep) -- every existing component is a plain
class with no base-class hierarchy, and every module already uses
``from __future__ import annotations`` for lazy type hints. An ``ABC`` base
class would force ``LegacyVehicleDynamics`` (and any future backend) to
inherit from a new base purely for typing purposes -- an actual class-
hierarchy change to a component ``test_structural_freeze.py`` already
freezes the shape of. A bare, undocumented duck-typing convention (the
status quo before this file existed) works at runtime but gives
``PlantModel``'s dependency no discoverable name, no single canonical
signature definition, and no way to assert "does this object satisfy the
contract" in a test. ``typing.Protocol`` (with ``@runtime_checkable``) gets
the discoverable name and the ``isinstance()``-checkable contract with ZERO
inheritance change to ``LegacyVehicleDynamics`` -- it already satisfies this
Protocol structurally, with no code change of its own. This is the smallest
seam that satisfies the stated goal ("PlantModel depends on the contract,
not the concrete class") without adding a class hierarchy, a registry, a
factory, or any other framework machinery.

Not in scope for this module (see the migration report / CLAUDE.md's
structural-freeze rule): no MuJoCo import, no contact-model change, no new
physics parameter, no change to ``VehicleState`` or ``ContactModel``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from landing_rl.dynamics.vehicle_state import VehicleState


@runtime_checkable
class VehicleDynamicsBackend(Protocol):
    """Structural contract for one control-step free-flight physics update.

    ``LegacyVehicleDynamics`` is currently the ONLY implementation and
    remains the default/only backend actually wired into ``PlantModel`` by
    ``LandingEnv``. A future ``MuJoCoDynamics`` would implement this same
    single method to become a drop-in alternative.
    """

    def advance_free_flight(
        self,
        state: VehicleState,
        v_cmd: np.ndarray,
        dt: float,
        rng,
        wind_accel: np.ndarray,
        target_yaw: float,
        body_rate_response_alpha: np.ndarray,
        thrust_response_alpha: float,
        motor_cutoff: bool,
        ground_contact: bool,
    ) -> None:
        """Advance ``state`` by one control step of free-flight physics (no
        ground contact). See the module docstring for the full contract."""
        ...
