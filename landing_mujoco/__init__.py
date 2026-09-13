"""MuJoCo-backed UAV precision-landing plant.

This package is additive: it does not modify ``landing_rl/`` or
``mujoco_rl/``. It reuses the existing controller / perception / reward /
success-failure logic from ``landing_rl`` unchanged, and replaces only the
rigid-body plant + contact model with a MuJoCo simulation whose physical
parameters are loaded from an explicit, provenance-tagged YAML config (see
``landing_mujoco/configs``).

See ``MUJOCO_MODEL.md`` (repo root) for the coordinate convention, parameter
table, and provisional-vs-measured parameter policy.
"""
