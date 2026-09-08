"""KRAS allele-specific histomorphology study (Experimental_Setup.md).

``labels``    — the frozen §5 label rules (token parsing, class partitions)
``registry``  — the §4.2 experiment registry (row filters + label partitions)
``study``     — protocol constants, aggregation, and ensemble scoring
"""

from oceanpath.kras.labels import (
    KRAS_CLASSES,
    NAMED_ALLELES,
    add_kras_class_columns,
    has_token,
    kras_class,
    subvariant_tokens,
)
from oceanpath.kras.registry import (
    EXPERIMENTS,
    GATEKEEPING_SEQUENCE,
    T1_POOLED_EXPERIMENTS,
    TRAINING_ORDER,
    KrasExperiment,
    get_experiment,
)

__all__ = [
    "EXPERIMENTS",
    "GATEKEEPING_SEQUENCE",
    "KRAS_CLASSES",
    "NAMED_ALLELES",
    "T1_POOLED_EXPERIMENTS",
    "TRAINING_ORDER",
    "KrasExperiment",
    "add_kras_class_columns",
    "get_experiment",
    "has_token",
    "kras_class",
    "subvariant_tokens",
]
