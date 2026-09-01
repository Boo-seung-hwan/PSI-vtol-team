"""landing_rl controllers package.

Phase 2 of the structural migration: the deterministic velocity-control
arithmetic extracted from ``LandingEnv`` (see ``baseline_controller``). No
behavior change; the OLD-vs-NEW exact parity gate remains authoritative.
"""

from landing_rl.controllers.baseline_controller import BaselineController

__all__ = ["BaselineController"]
