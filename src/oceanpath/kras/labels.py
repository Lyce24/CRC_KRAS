"""Frozen KRAS label rules (Experimental_Setup.md §5, §6.4 strata).

Everything that turns ``kras_final.csv`` rows into experiment labels lives
here — this module is part of the freeze snapshot (§6.5), so any change is a
logged amendment.

The rules, verbatim from §5:

1. Positive class = target-allele TOKEN PRESENT in ``kras_subvariant``
   (split on ';', never string equality). ``G12V;G12C`` is positive for both
   the G12V and G12C models; ``G13D;R164Q`` and ``A146T;A59T;G13D`` are
   G13D-positive.
2. Negative class (allele models) = ``kras == mutant`` WITHOUT the target
   token.
3. Exclusions: ``kras == unknown`` everywhere (non-random, cohort-specific
   meaning). Mutants with no recorded variant (RIH SL-102 is the only one)
   are excluded from allele/codon/B2 models but stay in B1.
4. All metrics are patient-level; splits are by ``patient_uid`` only.
"""

from __future__ import annotations

import re

import pandas as pd

# Named alleles of the study, in the fixed priority used ONLY for the 6-class
# stratification key (multi-token slides take the highest-priority named
# token). Experiment labels never use this priority — they use token presence.
NAMED_ALLELES = ("G12D", "G12V", "G13D", "G12C")

# 6-class stratification partition (§6.4): fold assignment is stratified by
# cohort x KRAS class over the FULL development population, so every
# experiment's row subset inherits approximately balanced folds.
KRAS_CLASSES = (*NAMED_ALLELES, "other_mutant", "wild_type")

_CODON12_RE = re.compile(r"G12[A-Z]")
_A146_RE = re.compile(r"A146[A-Z]")


def subvariant_tokens(value: object) -> list[str]:
    """Split a ``kras_subvariant`` cell into clean HGVS-short tokens.

    Multi-variant values are ';'-joined at build time; whitespace and empty
    fragments are discarded. NaN/empty → no tokens (the SL-102 case).
    """
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return []
    return [token.strip() for token in str(value).split(";") if token.strip()]


def has_token(value: object, allele: str) -> bool:
    """§5 rule 1: token membership, never substring/equality on the raw cell."""
    return allele in subvariant_tokens(value)


def is_codon12(value: object) -> bool:
    """Any G12x token (exact codon match, not ``startswith``)."""
    return any(_CODON12_RE.fullmatch(token) for token in subvariant_tokens(value))


def has_a146(value: object) -> bool:
    """A146x token present (the R3 sensitivity exclusion, §7 issue 2)."""
    return any(_A146_RE.fullmatch(token) for token in subvariant_tokens(value))


def kras_class(kras: str, subvariant: object) -> str:
    """6-class stratification label for one slide (NOT an experiment label).

    wild_type → wild_type; mutant → highest-priority named token, else
    other_mutant. Unknown must be excluded before calling.
    """
    if kras == "wild_type":
        return "wild_type"
    if kras != "mutant":
        raise ValueError(f"kras_class needs mutant/wild_type rows, got {kras!r}")
    tokens = set(subvariant_tokens(subvariant))
    for allele in NAMED_ALLELES:
        if allele in tokens:
            return allele
    return "other_mutant"


def b2_class_set(subvariant: object) -> set[str]:
    """B2 4-way class memberships of a mutant slide: {G12D, G12V, G13D, other}.

    A slide belongs to every named class whose token it carries, plus
    ``other`` if it carries any non-named token. Slides with more than one
    membership are multi-assignment and are EXCLUDED from B2 (§5 rule 1:
    TCGA-G4-6320 ``G12D;G13D``; the same rule catches ``G12C;G12V`` because
    G12C belongs to ``other``).
    """
    tokens = subvariant_tokens(subvariant)
    memberships = {token for token in tokens if token in ("G12D", "G12V", "G13D")}
    if any(token not in ("G12D", "G12V", "G13D") for token in tokens):
        memberships.add("other")
    return memberships


B2_CLASS_NAMES = ("G12D", "G12V", "G13D", "other")


def add_kras_class_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ``kras_class`` and the composite stratification key.

    Requires ``kras`` ∈ {mutant, wild_type} and a ``cohort_group`` column.
    """
    out = df.copy()
    out["kras_class"] = [
        kras_class(k, sub) for k, sub in zip(out["kras"], out["kras_subvariant"], strict=True)
    ]
    out["strat_kras_class_cohort"] = out["cohort_group"] + "|" + out["kras_class"]
    return out


def mutant_with_variant_mask(df: pd.DataFrame) -> pd.Series:
    """Rows eligible for allele/codon/B2 models: mutant AND variant recorded.

    Mutants without a recorded variant (RIH SL-102) stay in B1 only.
    """
    has_variant = df["kras_subvariant"].map(lambda v: len(subvariant_tokens(v)) > 0)
    return (df["kras"] == "mutant") & has_variant
