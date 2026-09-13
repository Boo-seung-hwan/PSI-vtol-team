"""Simulation timing configuration.

Kept separate from ``UAVPhysicalParams`` (physical/vehicle data) and from
``landing_rl.envs.landing_env.LandingConfig`` (RL/control tuning) -- this is
purely the physics-integration schedule.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationConfig:
    control_dt: float = 0.05     # matches landing_rl LandingConfig.dt (20 Hz)
    physics_dt: float = 0.002    # MuJoCo integration step, must be < control_dt
    deterministic_physics: bool = False

    def __post_init__(self) -> None:
        if self.physics_dt <= 0:
            raise ValueError(f"physics_dt must be > 0, got {self.physics_dt}")
        if self.control_dt <= 0:
            raise ValueError(f"control_dt must be > 0, got {self.control_dt}")
        if self.physics_dt > self.control_dt:
            raise ValueError(
                f"physics_dt ({self.physics_dt}) must not exceed control_dt "
                f"({self.control_dt}) -- MuJoCo must integrate faster than the "
                "control loop."
            )

    def physics_substeps(self, dt: float | None = None) -> int:
        """Number of physics_dt-sized substeps needed to cover one control
        step of duration ``dt`` (defaults to ``control_dt``).

        ``control_dt / physics_dt`` is not required to be an exact integer --
        the last substep is shortened to land exactly on ``dt`` so the
        integration schedule is always well-defined for jittered dt (the
        existing environment samples per-step dt with jitter)."""
        step_dt = self.control_dt if dt is None else dt
        return _n_substeps(step_dt, self.physics_dt)


def _n_substeps(dt: float, physics_dt: float) -> int:
    import math

    return max(1, math.ceil(dt / physics_dt))
