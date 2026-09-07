"""UGRP precision-landing system-identification package.

Current state (see PROJECT_HANDOVER.md):

    system_id/preprocessing/   IMPLEMENTED (this phase) -- raw ULog -> masked,
                               frame-consistent, per-loop datasets + provenance
                               + data-quality report. NO parameter fitting.

    system_id/identification/  NOT IMPLEMENTED
    system_id/validation/      NOT IMPLEMENTED
    system_id/results/         NO IDENTIFIED UGRP PARAMETERS EXIST

No numeric value in this package has been identified for the UGRP vehicle.
The only ULogs currently available are OTHER-PROJECT sample logs used strictly
for pipeline validation; nothing derived from them is a UGRP parameter.
"""

__all__ = ["preprocessing"]
