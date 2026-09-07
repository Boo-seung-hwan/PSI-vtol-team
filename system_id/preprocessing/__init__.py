"""ULog -> per-loop SI dataset preprocessing pipeline.

Pipeline stages (see ``pipeline.preprocess_log``):

    raw .ulg
      -> ulog_loader.load_ulog        validated topic/field/param extraction
      -> timebase                     per-loop timebase + ZOH resampling
      -> frames                       quaternion->Euler, tilt, thrust magnitude
      -> masks.compute_masks          armed/offboard/airborne/saturation/sentinel
      -> segments.split_segments      contiguous SI segments
      -> dataset.build_all            velocity / attitude / rate / translational
      -> provenance.build_provenance  SHA256 + PX4 version + git state + roles
      -> report.usability_report      data-quality / excitation gate (NOT a fit)

This package performs NO system-identification parameter fitting. It never
writes into ``system_id/results/`` or any UGRP config directory. Derived
sample data goes to ``system_id/derived/sample_other_project/`` (gitignored).
"""

from system_id.preprocessing.schema import SCHEMA_VERSION, PREPROCESSING_VERSION

__all__ = ["SCHEMA_VERSION", "PREPROCESSING_VERSION"]
