"""Aim 1 — locked KRAS mutant-vs-wild-type model and its dependency analyses.

This package implements ``Aim1_Setup.md`` (Experiments 1A / 1B / 1C). It is
deliberately separate from ``oceanpath.kras`` (the frozen allele-specific
pre-registration, ``Experimental_Setup.md`` v1.1): Aim 1 has its own eligible
population, its own multi-marginal balanced fold assignment, and its own
locked calibration and thresholds. Nothing here reads or mutates the allele
study's frozen manifests, splits, or checkpoints.

Module layout::

    paths        frozen source paths + output layout
    technical    label-independent per-slide technical descriptors (§4.5)
    population   the eligible population and every frozen manifest (§4.2/§5.1)
    balance      deterministic multi-marginal 5-fold assignment (§4.4)
    calibration  cross-fitted Platt calibration + locked thresholds (§4.6)
    evaluate     discrimination / calibration / threshold / triage (§4.7)
    baselines    clinical-only, technical-only, late fusion, adjusted OR (§5.5-5.6)
    scoring      frozen fold-ensemble inference over a manifest (§4.5)
    interpretation  prespecified bands + pooled external summary (§7)
"""

from __future__ import annotations

__all__ = [
    "balance",
    "baselines",
    "calibration",
    "evaluate",
    "interpretation",
    "paths",
    "population",
    "registry",
    "scoring",
    "technical",
]
