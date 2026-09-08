"""The KRAS allele-study experiment registry (Experimental_Setup.md §4.2).

Every experiment shares one recipe (encoder, ABMIL family, frozen
hyperparameter grid, CV protocol) and differs ONLY in its training rows and
label partition — this module is the single place where those row filters
and label rules are defined. Phase 3 materializes them into per-experiment
manifests; phases 4-7 consume the same registry so the definitions can never
drift between splitting, training, and scoring.

T1 (primary→metastatic transfer) is deliberately absent: it trains nothing
and simply scores the frozen P1/S1/S2 ensembles on EXT-M (phase 7).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from oceanpath.kras.labels import (
    B2_CLASS_NAMES,
    b2_class_set,
    has_token,
    is_codon12,
    mutant_with_variant_mask,
)


@dataclass(frozen=True)
class KrasExperiment:
    """One row-filter + label-partition over an arm of the study population."""

    exp_id: str
    title: str
    # select_and_label(arm_df) -> rows with an integer ``target_label`` column.
    # arm_df carries kras ∈ {mutant, wild_type}, kras_subvariant, kras_class.
    select_and_label: Callable[[pd.DataFrame], pd.DataFrame]
    num_classes: int = 2
    class_names: tuple[str, ...] = ("negative", "positive")
    loss: str = "bce"  # bce → single-logit binary head; ce → softmax head
    # Arms the FROZEN model is scored on ("ext_p", "ext_m"); () = DEV-only,
    # pooled OOF is the reported result (M-series, §6.4).
    external_arms: tuple[str, ...] = ()
    # X1's inner-validation splits cannot support early stopping (§6.4 small-
    # class rule); such experiments need P1's median best-epoch budget.
    requires_epoch_budget: bool = False
    # Label-source-level expectations from the plan (§3/§4.2), verified by the
    # phase 3 builder BEFORE feature-availability filtering. None = report-only.
    expected_dev: dict[str, int] | None = None
    notes: str = ""
    extra_label_columns: tuple[str, ...] = field(default=())


def _allele_vs_other_mutant(allele: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """P1/S1/S2/X1: target-allele token vs every other variant-known mutant."""

    def build(df: pd.DataFrame) -> pd.DataFrame:
        rows = df[mutant_with_variant_mask(df)].copy()
        rows["target_label"] = (
            rows["kras_subvariant"].map(lambda v, a=allele: has_token(v, a)).astype(int)
        )
        return rows

    return build


def _mutant_vs_wild_type(df: pd.DataFrame) -> pd.DataFrame:
    """B1: every kras-known slide; unknown was excluded upstream. SL-102
    (mutant, no variant recorded) stays in — B1 needs no variant."""
    rows = df[df["kras"].isin(["mutant", "wild_type"])].copy()
    rows["target_label"] = (rows["kras"] == "mutant").astype(int)
    return rows


def _b2_four_class(df: pd.DataFrame) -> pd.DataFrame:
    """B2: single-assignment mutants, 4-way G12D/G12V/G13D/other."""
    rows = df[mutant_with_variant_mask(df)].copy()
    memberships = rows["kras_subvariant"].map(b2_class_set)
    rows = rows[memberships.map(len) == 1].copy()
    class_index = {name: index for index, name in enumerate(B2_CLASS_NAMES)}
    rows["target_label"] = [
        class_index[next(iter(b2_class_set(v)))] for v in rows["kras_subvariant"]
    ]
    return rows


def _codon12_control(allele: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """M1/M2: target allele vs the other codon-12 mutants only."""

    def build(df: pd.DataFrame) -> pd.DataFrame:
        rows = df[mutant_with_variant_mask(df)].copy()
        rows = rows[rows["kras_subvariant"].map(is_codon12)].copy()
        rows["target_label"] = (
            rows["kras_subvariant"].map(lambda v, a=allele: has_token(v, a)).astype(int)
        )
        return rows

    return build


def _g12d_vs_g12v(df: pd.DataFrame) -> pd.DataFrame:
    """M3 head-to-head. A slide carrying BOTH tokens would be ambiguous and is
    excluded (none exists in DEV; the guard keeps external reuse safe)."""
    rows = df[mutant_with_variant_mask(df)].copy()
    has_d = rows["kras_subvariant"].map(lambda v: has_token(v, "G12D"))
    has_v = rows["kras_subvariant"].map(lambda v: has_token(v, "G12V"))
    rows = rows[(has_d | has_v) & ~(has_d & has_v)].copy()
    rows["target_label"] = rows["kras_subvariant"].map(lambda v: int(has_token(v, "G12D")))
    return rows


def _multihead_labels(df: pd.DataFrame) -> pd.DataFrame:
    """R4: one row per variant-known mutant, four binary label columns.

    ``target_label`` mirrors P1 (G12D) so shared tooling stays valid; the
    per-head labels live in label_g12d/…/label_g12c. Masks are implicit: a
    head's label is defined on every row of this manifest (variant-known
    mutants), which is why SL-102-style rows are excluded here too.
    """
    rows = df[mutant_with_variant_mask(df)].copy()
    for allele in ("G12D", "G12V", "G13D", "G12C"):
        rows[f"label_{allele.lower()}"] = (
            rows["kras_subvariant"].map(lambda v, a=allele: has_token(v, a)).astype(int)
        )
    rows["target_label"] = rows["label_g12d"]
    return rows


EXPERIMENTS: dict[str, KrasExperiment] = {
    experiment.exp_id: experiment
    for experiment in (
        KrasExperiment(
            exp_id="p1",
            title="G12D vs other KRAS-mutant (primary experiment)",
            select_and_label=_allele_vs_other_mutant("G12D"),
            class_names=("other_mutant", "g12d"),
            external_arms=("ext_p", "ext_m"),
            expected_dev={"n": 363, "pos": 110},
            notes="Confirmatory: pooled EXT-P patient AUROC, then per-site.",
        ),
        KrasExperiment(
            exp_id="s1",
            title="G12V vs other KRAS-mutant",
            select_and_label=_allele_vs_other_mutant("G12V"),
            class_names=("other_mutant", "g12v"),
            external_arms=("ext_p", "ext_m"),
            expected_dev={"n": 363, "pos": 78},
        ),
        KrasExperiment(
            exp_id="s2",
            title="G13D vs other KRAS-mutant",
            select_and_label=_allele_vs_other_mutant("G13D"),
            class_names=("other_mutant", "g13d"),
            external_arms=("ext_p", "ext_m"),
            expected_dev={"n": 363, "pos": 55},
            notes="RIH-P has n=10 positives — exact CIs at that site.",
        ),
        KrasExperiment(
            exp_id="x1",
            title="G12C vs other KRAS-mutant (exploratory)",
            select_and_label=_allele_vs_other_mutant("G12C"),
            class_names=("other_mutant", "g12c"),
            external_arms=("ext_p",),
            requires_epoch_budget=True,
            expected_dev={"n": 363, "pos": 31},
            notes="~4 inner-val positives → fixed epoch budget by construction.",
        ),
        KrasExperiment(
            exp_id="b1",
            title="KRAS mutant vs wild-type (benchmark)",
            select_and_label=_mutant_vs_wild_type,
            class_names=("wild_type", "mutant"),
            external_arms=("ext_p", "ext_m"),
            expected_dev={"n": 938, "pos": 363},
        ),
        KrasExperiment(
            exp_id="b2",
            title="4-class G12D/G12V/G13D/other (benchmark)",
            select_and_label=_b2_four_class,
            num_classes=4,
            class_names=B2_CLASS_NAMES,
            loss="ce",
            external_arms=("ext_p",),
            expected_dev={"n": 361},
            notes="Multi-assignment mutants excluded (TCGA-G4-6320, TCGA-AG-4008).",
        ),
        KrasExperiment(
            exp_id="m1",
            title="G12D vs other codon-12 (codon control)",
            select_and_label=_codon12_control("G12D"),
            class_names=("other_codon12", "g12d"),
            expected_dev={"n": 256, "pos": 110},
        ),
        KrasExperiment(
            exp_id="m2",
            title="G12V vs other codon-12 (codon control)",
            select_and_label=_codon12_control("G12V"),
            class_names=("other_codon12", "g12v"),
            expected_dev={"n": 256, "pos": 78},
        ),
        KrasExperiment(
            exp_id="m3",
            title="G12D vs G12V head-to-head",
            select_and_label=_g12d_vs_g12v,
            class_names=("g12v", "g12d"),
            expected_dev={"n": 188, "pos": 110},
        ),
        KrasExperiment(
            exp_id="r4",
            title="Multihead robustness (shared trunk, 4 masked binary heads)",
            select_and_label=_multihead_labels,
            class_names=("other_mutant", "g12d"),
            external_arms=("ext_p",),
            expected_dev={"n": 363, "pos": 110},
            extra_label_columns=("label_g12d", "label_g12v", "label_g13d", "label_g12c"),
            notes="Robustness only — conclusions cite the separate models.",
        ),
    )
}

# The fixed-sequence gatekeeping order of §6.7: pooled P1 → P1-at-RIH → S1 →
# S2. Everything after the first failure is reported descriptively.
GATEKEEPING_SEQUENCE = (
    ("p1", "pooled"),
    ("p1", "RIH"),
    ("s1", "pooled"),
    ("s2", "pooled"),
)

# T1's powered claim pools these alleles' ΔAUROC (§6.6); X1 is descriptive.
T1_POOLED_EXPERIMENTS = ("p1", "s1", "s2")

# Experiments trained by phase 5 in dependency order: P1 first, because its
# fold-runs define both the frozen hyperparameters (phase 4) and the fixed
# epoch budget the small-class fallback substitutes for early stopping.
TRAINING_ORDER = ("p1", "s1", "s2", "x1", "b1", "b2", "m1", "m2", "m3")


def get_experiment(exp_id: str) -> KrasExperiment:
    try:
        return EXPERIMENTS[exp_id]
    except KeyError:
        raise SystemExit(
            f"Unknown experiment '{exp_id}'. Known: {', '.join(EXPERIMENTS)}"
        ) from None
