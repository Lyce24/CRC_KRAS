"""Frozen Stage-B evaluation for the G12D marginal/conditional experiment.

Stage A trains G12D versus every KRAS-known non-G12D patient.  Stage B does
not train a model: it evaluates those frozen Stage-A scores on the exact
Stage-C population (G12D versus other variant-known KRAS mutants).

The evaluator deliberately validates each prediction artifact against its
*full* Stage-A held-out cohort before applying the Stage-B restriction.  This
prevents an incomplete prediction file containing only conditional patients
from passing as a valid frozen marginal evaluation.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score

from oceanpath.eval.pooled_loco import RotationRecord

G12D_CONTEXT = "G12D"
OTHER_KRAS_CONTEXT = "other_KRAS_mutant"
KRAS_WILD_TYPE_CONTEXT = "KRAS_wild_type"
CONTEXTS: tuple[str, ...] = (
    G12D_CONTEXT,
    OTHER_KRAS_CONTEXT,
    KRAS_WILD_TYPE_CONTEXT,
)
CONDITIONAL_CONTEXTS: tuple[str, ...] = (G12D_CONTEXT, OTHER_KRAS_CONTEXT)
NEGATIVE_CONTEXTS: tuple[str, ...] = (
    OTHER_KRAS_CONTEXT,
    KRAS_WILD_TYPE_CONTEXT,
)

_PROBABILITY_CONSISTENCY_RTOL = 1e-6
_PROBABILITY_CONSISTENCY_ATOL = 1e-7
_COMPLETION_FILENAME = "evaluation_completion.json"


class G12DFrozenSubsetValidationError(ValueError):
    """Raised when an input violates the frozen Stage-B contract."""


@dataclass
class G12DFrozenSubsetEvaluation:
    """In-memory predictions, metrics, and report for frozen Stage B."""

    stage_a_slide_predictions: pd.DataFrame
    stage_a_patient_predictions: pd.DataFrame
    stage_b_slide_predictions: pd.DataFrame
    stage_b_patient_predictions: pd.DataFrame
    metrics: dict[str, Any]
    report_markdown: str

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        """Atomically publish canonical artifacts, with completion last."""
        destination = Path(output_dir)
        if destination.exists() and not destination.is_dir():
            raise NotADirectoryError(f"evaluation output is not a directory: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        paths = {
            "stage_a_slides": destination / "stage_a_slide_predictions.parquet",
            "stage_a_patients": destination / "stage_a_patient_predictions.parquet",
            "stage_b_slides": destination / "stage_b_slide_predictions.parquet",
            "stage_b_patients": destination / "stage_b_patient_predictions.parquet",
            "metrics": destination / "metrics.json",
            "report": destination / "report.md",
            "completion": destination / _COMPLETION_FILENAME,
        }
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.g12d-stage-b-publish-",
                dir=destination.parent,
            )
        )
        staged = {key: staging / path.name for key, path in paths.items()}
        artifact_keys = tuple(key for key in paths if key != "completion")
        try:
            self.stage_a_slide_predictions.to_parquet(staged["stage_a_slides"], index=False)
            self.stage_a_patient_predictions.to_parquet(
                staged["stage_a_patients"], index=False
            )
            self.stage_b_slide_predictions.to_parquet(staged["stage_b_slides"], index=False)
            self.stage_b_patient_predictions.to_parquet(
                staged["stage_b_patients"], index=False
            )
            staged["metrics"].write_text(
                json.dumps(self.metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            staged["report"].write_text(self.report_markdown, encoding="utf-8")
            evidence = {
                staged[key].name: _artifact_evidence(staged[key]) for key in artifact_keys
            }
            completion = {
                "schema_version": 1,
                "status": "complete",
                "analysis": "g12d_frozen_conditional_stage_b",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": evidence,
            }
            staged["completion"].write_text(
                json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            for filename, expected in evidence.items():
                observed = _artifact_evidence(staging / filename)
                if observed != expected:
                    raise RuntimeError(f"staged artifact changed before publication: {filename}")

            # An old completion marker must never describe partially replaced files.
            paths["completion"].unlink(missing_ok=True)
            for key in artifact_keys:
                staged[key].replace(paths[key])
            staged["completion"].replace(paths["completion"])
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        return paths


def evaluate_g12d_frozen_subset(
    rotations: Sequence[RotationRecord],
    stage_a_manifest: pd.DataFrame | str | Path,
    stage_c_manifest: pd.DataFrame | str | Path,
    *,
    output_dir: str | Path | None = None,
    stage_a_label_column: str = "target_label",
    stage_c_label_column: str = "target_label",
    context_column: str = "kras_context",
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
    n_matched: int = 1000,
    matched_seed: int = 1729,
) -> G12DFrozenSubsetEvaluation:
    """Validate and evaluate frozen Stage-A scores on the exact Stage-C set.

    Parameters
    ----------
    rotations:
        One :class:`~oceanpath.eval.pooled_loco.RotationRecord` per held-out
        cohort and seed. Every prediction file must cover the complete Stage-A
        held-out cohort and contain ``slide_id``, ``label``, raw ``logit``, and
        ``prob_1``.
    stage_a_manifest:
        Marginal manifest with slide/patient/cohort/label plus ``kras_context``.
    stage_c_manifest:
        Frozen conditional manifest. It must equal exactly the G12D and
        other-KRAS rows of the Stage-A manifest on slide, patient, cohort, and
        label.
    n_bootstrap:
        Patient bootstrap replicates for point-metric intervals and the nested
        shared Stage-A-minus-Stage-B AUROC contrast.
    n_matched:
        Frozen marginal subsamples. Defaults to the preregistered 1,000.
    """
    _validate_parameters(
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        n_matched=n_matched,
        matched_seed=matched_seed,
    )
    if not rotations:
        raise G12DFrozenSubsetValidationError("at least one rotation record is required")

    stage_a, stage_a_provenance = _prepare_stage_a_manifest(
        stage_a_manifest,
        label_column=stage_a_label_column,
        context_column=context_column,
    )
    stage_c, stage_c_provenance = _prepare_stage_c_manifest(
        stage_c_manifest,
        label_column=stage_c_label_column,
    )
    _validate_exact_conditional_subset(stage_a, stage_c)
    records = _normalize_rotation_records(rotations, stage_a)

    stage_a_slides = _load_and_ensemble_full_stage_a(records, stage_a)
    conditional_slide_ids = set(stage_c["slide_id"])
    stage_b_slides = (
        stage_a_slides.loc[stage_a_slides["slide_id"].isin(conditional_slide_ids)]
        .sort_values(["cohort", "patient_id", "slide_id"], kind="stable")
        .reset_index(drop=True)
    )
    if set(stage_b_slides["slide_id"]) != conditional_slide_ids:
        raise G12DFrozenSubsetValidationError(
            "validated Stage-A predictions did not yield exact Stage-C slide coverage"
        )

    stage_a_patients = _aggregate_patients(stage_a_slides)
    stage_b_patients = _aggregate_patients(stage_b_slides)
    _validate_stage_b_patients(stage_b_patients, stage_c)

    stage_a_metrics = _evaluate_population(
        stage_a_patients,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=_derived_seed(bootstrap_seed, "stage_a"),
    )
    stage_b_metrics = _evaluate_population(
        stage_b_patients,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=_derived_seed(bootstrap_seed, "stage_b"),
    )
    delta_context = _nested_context_delta(
        stage_a_patients,
        stage_b_patients,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=_derived_seed(bootstrap_seed, "delta_context"),
    )
    matched = _matched_marginal_controls(
        stage_a_patients,
        stage_b_patients,
        n_repeats=n_matched,
        seed=matched_seed,
    )

    seed_ids = sorted({str(record.seed) for record in records})
    metrics: dict[str, Any] = {
        "analysis": "g12d_frozen_conditional_stage_b",
        "definition": {
            "stage_a": "frozen G12D versus all KRAS-known non-G12D held-out patients",
            "stage_b": "same frozen Stage-A scores restricted to exact Stage-C patients",
            "delta_context": "AUROC_A - AUROC_B",
        },
        "stage_a_manifest": stage_a_provenance,
        "stage_c_manifest": stage_c_provenance,
        "subset_audit": {
            "status": "exact",
            "conditional_contexts": list(CONDITIONAL_CONTEXTS),
            "stage_c_slides": int(len(stage_c)),
            "stage_c_patients": int(stage_c["patient_id"].nunique()),
        },
        "aggregation": {
            "score_source": "stored raw prediction logit",
            "across_seeds": "mean raw logit per slide",
            "within_patient": "sigmoid(mean ensembled slide logits)",
            "probability_check": {
                "definition": "prob_1 approximately equals sigmoid(logit)",
                "relative_tolerance": _PROBABILITY_CONSISTENCY_RTOL,
                "absolute_tolerance": _PROBABILITY_CONSISTENCY_ATOL,
            },
        },
        "bootstrap": {
            "method": "patient-level percentile",
            "confidence_level": 0.95,
            "n_resamples": n_bootstrap,
            "seed": bootstrap_seed,
            "point_metric_stratification": "held-out cohort x label (label within cohort)",
            "delta_context_stratification": "held-out cohort x kras_context",
            "delta_context_shared_samples": True,
        },
        "n_seeds": len(seed_ids),
        "seeds": seed_ids,
        "rotations": [_rotation_provenance(record) for record in records],
        "stage_a": stage_a_metrics,
        "stage_b": stage_b_metrics,
        "delta_context_auc_a_minus_b": delta_context,
        "matched_marginal": matched,
    }
    report = _render_markdown(metrics)
    result = G12DFrozenSubsetEvaluation(
        stage_a_slide_predictions=stage_a_slides,
        stage_a_patient_predictions=stage_a_patients,
        stage_b_slide_predictions=stage_b_slides,
        stage_b_patient_predictions=stage_b_patients,
        metrics=metrics,
        report_markdown=report,
    )
    if output_dir is not None:
        result.write(output_dir)
    return result


def _validate_parameters(
    *, n_bootstrap: int, bootstrap_seed: int, n_matched: int, matched_seed: int
) -> None:
    for name, value in (("n_bootstrap", n_bootstrap), ("n_matched", n_matched)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise G12DFrozenSubsetValidationError(f"{name} must be a positive integer")
    for name, value in (("bootstrap_seed", bootstrap_seed), ("matched_seed", matched_seed)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise G12DFrozenSubsetValidationError(f"{name} must be an integer")


def _read_table(table: pd.DataFrame | str | Path, *, name: str) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    path = Path(table)
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    suffix = path.suffix.casefold()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
    raise G12DFrozenSubsetValidationError(
        f"unsupported {name} format {path.suffix!r}; use CSV, TSV, or Parquet"
    )


def _prepare_stage_a_manifest(
    table: pd.DataFrame | str | Path,
    *,
    label_column: str,
    context_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_table(table, name="Stage-A manifest")
    required = ["slide_id", "patient_id", "cohort", label_column, context_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise G12DFrozenSubsetValidationError(
            f"Stage-A manifest is missing columns: {missing}"
        )
    output = frame[required].rename(
        columns={label_column: "label", context_column: "kras_context"}
    )
    output = _canonicalize_manifest(output, name="Stage-A manifest", with_context=True)
    observed_contexts = set(output["kras_context"])
    if observed_contexts != set(CONTEXTS):
        raise G12DFrozenSubsetValidationError(
            "Stage-A kras_context values must be exactly "
            f"{list(CONTEXTS)}; observed={sorted(observed_contexts)}"
        )
    expected_labels = output["kras_context"].eq(G12D_CONTEXT).astype(np.int8)
    if not np.array_equal(output["label"].to_numpy(dtype=np.int8), expected_labels):
        bad = output.loc[output["label"].to_numpy() != expected_labels, "slide_id"]
        raise G12DFrozenSubsetValidationError(
            "Stage-A labels disagree with kras_context: " + _preview(bad.tolist())
        )
    _require_contexts_per_cohort(output)
    return output, _manifest_provenance(table, output)


def _prepare_stage_c_manifest(
    table: pd.DataFrame | str | Path,
    *,
    label_column: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = _read_table(table, name="Stage-C manifest")
    required = ["slide_id", "patient_id", "cohort", label_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise G12DFrozenSubsetValidationError(
            f"Stage-C manifest is missing columns: {missing}"
        )
    output = frame[required].rename(columns={label_column: "label"})
    output = _canonicalize_manifest(output, name="Stage-C manifest", with_context=False)
    return output, _manifest_provenance(table, output)


def _canonicalize_manifest(
    frame: pd.DataFrame, *, name: str, with_context: bool
) -> pd.DataFrame:
    output = frame.copy()
    for column in ("slide_id", "patient_id", "cohort"):
        output[column] = _identifier_values(output[column], context=f"{name} {column}")
    if with_context:
        output["kras_context"] = _identifier_values(
            output["kras_context"], context=f"{name} kras_context"
        )
    duplicates = sorted(output.loc[output["slide_id"].duplicated(False), "slide_id"].unique())
    if duplicates:
        raise G12DFrozenSubsetValidationError(
            f"{name} has duplicate slide_id values: {_preview(duplicates)}"
        )
    output["label"] = _binary_labels(output["label"], context=f"{name} label")
    patient_columns = ["patient_id", "cohort", "label"]
    if with_context:
        patient_columns.append("kras_context")
    patient_metadata = output[patient_columns].drop_duplicates()
    conflicts = patient_metadata.groupby("patient_id", sort=False).size()
    conflicts = conflicts[conflicts != 1]
    if not conflicts.empty:
        raise G12DFrozenSubsetValidationError(
            f"{name} has inconsistent patient metadata: {_preview(conflicts.index.tolist())}"
        )
    columns = ["slide_id", "patient_id", "cohort", "label"]
    if with_context:
        columns.append("kras_context")
    return output.sort_values("slide_id", kind="stable")[columns].reset_index(drop=True)


def _require_contexts_per_cohort(stage_a: pd.DataFrame) -> None:
    for cohort, group in stage_a.groupby("cohort", sort=True):
        observed = set(group["kras_context"])
        if observed != set(CONTEXTS):
            raise G12DFrozenSubsetValidationError(
                f"Stage-A cohort {cohort!r} must contain all contexts; "
                f"observed={sorted(observed)}"
            )


def _validate_exact_conditional_subset(stage_a: pd.DataFrame, stage_c: pd.DataFrame) -> None:
    conditional = stage_a.loc[
        stage_a["kras_context"].isin(CONDITIONAL_CONTEXTS),
        ["slide_id", "patient_id", "cohort", "label"],
    ].sort_values("slide_id", kind="stable").reset_index(drop=True)
    expected = stage_c.sort_values("slide_id", kind="stable").reset_index(drop=True)
    if conditional.equals(expected):
        return
    conditional_slides = set(conditional["slide_id"])
    stage_c_slides = set(expected["slide_id"])
    missing = sorted(conditional_slides - stage_c_slides)
    extra = sorted(stage_c_slides - conditional_slides)
    shared = conditional_slides & stage_c_slides
    left = conditional.set_index("slide_id").loc[sorted(shared)]
    right = expected.set_index("slide_id").loc[sorted(shared)]
    mismatched = [
        slide_id
        for slide_id in sorted(shared)
        if not left.loc[slide_id].equals(right.loc[slide_id])
    ]
    raise G12DFrozenSubsetValidationError(
        "Stage-C manifest is not the exact conditional subset of Stage A; "
        f"missing={_preview(missing)}, extra={_preview(extra)}, "
        f"metadata_mismatch={_preview(mismatched)}"
    )


def _normalize_rotation_records(
    rotations: Sequence[RotationRecord], stage_a: pd.DataFrame
) -> tuple[RotationRecord, ...]:
    cohorts = sorted(stage_a["cohort"].unique())
    normalized: list[RotationRecord] = []
    keys: set[tuple[str, str]] = set()
    paths: set[Path] = set()
    for record in rotations:
        cohort = str(record.heldout_cohort).strip()
        seed = str(record.seed)
        if cohort not in cohorts:
            raise G12DFrozenSubsetValidationError(
                f"rotation heldout_cohort {cohort!r} is not in Stage A"
            )
        path = Path(record.prediction_path)
        if not path.is_file():
            raise FileNotFoundError(f"prediction file not found: {path}")
        resolved = path.resolve()
        key = (seed, cohort)
        if key in keys:
            raise G12DFrozenSubsetValidationError(
                f"duplicate rotation for seed={seed!r}, heldout_cohort={cohort!r}"
            )
        if resolved in paths:
            raise G12DFrozenSubsetValidationError(
                f"prediction path is reused by multiple rotations: {path}"
            )
        keys.add(key)
        paths.add(resolved)
        normalized.append(
            RotationRecord(
                heldout_cohort=cohort,
                prediction_path=path,
                seed=seed,
                fold=record.fold,
                training_fingerprint=record.training_fingerprint,
            )
        )
    seeds = sorted({str(record.seed) for record in normalized})
    for seed in seeds:
        observed = {record.heldout_cohort for record in normalized if str(record.seed) == seed}
        if observed != set(cohorts):
            raise G12DFrozenSubsetValidationError(
                f"seed {seed!r} must have exactly one rotation per Stage-A cohort; "
                f"expected={cohorts}, observed={sorted(observed)}"
            )
    return tuple(
        sorted(normalized, key=lambda record: (str(record.seed), record.heldout_cohort))
    )


def _load_and_ensemble_full_stage_a(
    records: Sequence[RotationRecord], stage_a: pd.DataFrame
) -> pd.DataFrame:
    expected_by_cohort = {
        str(cohort): group for cohort, group in stage_a.groupby("cohort", sort=False)
    }
    frames: list[pd.DataFrame] = []
    for record in records:
        raw = pd.read_parquet(record.prediction_path)
        frames.append(
            _prepare_full_prediction_frame(
                raw,
                expected=expected_by_cohort[record.heldout_cohort],
                context=(
                    f"seed={record.seed!r}, heldout_cohort={record.heldout_cohort!r}"
                ),
            ).assign(seed=str(record.seed))
        )
    stacked = pd.concat(frames, ignore_index=True)
    n_seeds = len({str(record.seed) for record in records})
    counts = stacked.groupby("slide_id", sort=False)["seed"].nunique()
    invalid = counts[counts != n_seeds]
    if not invalid.empty:
        raise G12DFrozenSubsetValidationError(
            "Stage-A slides do not have exactly one prediction from every seed: "
            + _preview(invalid.index.tolist())
        )
    mean_logits = stacked.groupby("slide_id", sort=False)["logit"].mean().rename("mean_logit")
    output = stage_a.merge(mean_logits, on="slide_id", how="left", validate="one_to_one")
    if output["mean_logit"].isna().any():
        missing = output.loc[output["mean_logit"].isna(), "slide_id"].tolist()
        raise G12DFrozenSubsetValidationError(
            "ensembled Stage-A predictions are incomplete: " + _preview(missing)
        )
    output["prob_1"] = _sigmoid(output["mean_logit"].to_numpy(dtype=float))
    output["n_seeds"] = n_seeds
    columns = [
        "slide_id",
        "patient_id",
        "cohort",
        "label",
        "kras_context",
        "mean_logit",
        "prob_1",
        "n_seeds",
    ]
    return output.sort_values(["cohort", "patient_id", "slide_id"], kind="stable")[
        columns
    ].reset_index(drop=True)


def _prepare_full_prediction_frame(
    frame: pd.DataFrame, *, expected: pd.DataFrame, context: str
) -> pd.DataFrame:
    required = ["slide_id", "label", "logit", "prob_1"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise G12DFrozenSubsetValidationError(
            f"predictions for {context} are missing columns: {missing}"
        )
    output = frame.copy()
    output["slide_id"] = _identifier_values(
        output["slide_id"], context=f"prediction slide_id ({context})"
    )
    duplicates = sorted(output.loc[output["slide_id"].duplicated(False), "slide_id"].unique())
    if duplicates:
        raise G12DFrozenSubsetValidationError(
            f"duplicate prediction slide_id values for {context}: {_preview(duplicates)}"
        )

    # This check intentionally happens against the complete marginal cohort,
    # before any Stage-B filtering.
    expected_slides = set(expected["slide_id"])
    observed_slides = set(output["slide_id"])
    if expected_slides != observed_slides:
        raise G12DFrozenSubsetValidationError(
            f"full Stage-A held-out slide coverage mismatch for {context}; "
            f"missing={_preview(sorted(expected_slides - observed_slides))}, "
            f"extra={_preview(sorted(observed_slides - expected_slides))}"
        )

    output["label"] = _binary_labels(
        output["label"], context=f"prediction label ({context})"
    )
    probabilities = _numeric_values(output["prob_1"], context=f"prob_1 ({context})")
    if bool(((probabilities < 0.0) | (probabilities > 1.0)).any()):
        raise G12DFrozenSubsetValidationError(
            f"prediction prob_1 contains values outside [0, 1] for {context}"
        )
    logits = _numeric_values(output["logit"], context=f"logit ({context})")
    from_logits = _sigmoid(logits)
    consistent = np.isclose(
        probabilities,
        from_logits,
        rtol=_PROBABILITY_CONSISTENCY_RTOL,
        atol=_PROBABILITY_CONSISTENCY_ATOL,
    )
    if not bool(consistent.all()):
        mismatched = output.loc[~consistent, "slide_id"].tolist()
        maximum_error = float(np.max(np.abs(probabilities[~consistent] - from_logits[~consistent])))
        raise G12DFrozenSubsetValidationError(
            f"prediction prob_1 is inconsistent with sigmoid(logit) for {context}; "
            f"max_abs_error={maximum_error:.3g}, slides={_preview(mismatched)}"
        )

    canonical = expected.set_index("slide_id").loc[output["slide_id"]]
    expected_labels = canonical["label"].to_numpy(dtype=np.int8)
    predicted_labels = output["label"].to_numpy(dtype=np.int8)
    if not np.array_equal(predicted_labels, expected_labels):
        mismatched = output.loc[predicted_labels != expected_labels, "slide_id"].tolist()
        raise G12DFrozenSubsetValidationError(
            f"prediction labels disagree with Stage-A manifest for {context}: "
            + _preview(mismatched)
        )
    for column in ("patient_id", "cohort"):
        if column in output.columns:
            values = _identifier_values(
                output[column], context=f"prediction {column} ({context})"
            ).to_numpy()
            expected_values = canonical[column].to_numpy(dtype=str)
            if not np.array_equal(values, expected_values):
                mismatched = output.loc[values != expected_values, "slide_id"].tolist()
                raise G12DFrozenSubsetValidationError(
                    f"prediction {column} disagrees with Stage-A manifest for {context}: "
                    + _preview(mismatched)
                )
    return pd.DataFrame(
        {
            "slide_id": output["slide_id"].to_numpy(),
            "label": predicted_labels,
            "logit": logits,
            "prob_1": probabilities,
        }
    )


def _aggregate_patients(slides: pd.DataFrame) -> pd.DataFrame:
    patients = (
        slides.groupby("patient_id", sort=False)
        .agg(
            cohort=("cohort", "first"),
            label=("label", "first"),
            kras_context=("kras_context", "first"),
            mean_logit=("mean_logit", "mean"),
            n_slides=("slide_id", "size"),
            n_labels=("label", "nunique"),
            n_cohorts=("cohort", "nunique"),
            n_contexts=("kras_context", "nunique"),
        )
        .reset_index()
    )
    if bool(
        (patients["n_labels"] != 1).any()
        or (patients["n_cohorts"] != 1).any()
        or (patients["n_contexts"] != 1).any()
    ):
        raise G12DFrozenSubsetValidationError(
            "patient metadata became inconsistent during aggregation"
        )
    patients["prob_1"] = _sigmoid(patients["mean_logit"].to_numpy(dtype=float))
    columns = [
        "patient_id",
        "cohort",
        "label",
        "kras_context",
        "mean_logit",
        "n_slides",
        "prob_1",
    ]
    return patients.sort_values(["cohort", "patient_id"], kind="stable")[columns].reset_index(
        drop=True
    )


def _validate_stage_b_patients(stage_b: pd.DataFrame, stage_c: pd.DataFrame) -> None:
    expected = (
        stage_c.groupby("patient_id", sort=False)
        .agg(cohort=("cohort", "first"), label=("label", "first"), n_slides=("slide_id", "size"))
        .reset_index()
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )
    observed = stage_b[["patient_id", "cohort", "label", "n_slides"]].sort_values(
        "patient_id", kind="stable"
    ).reset_index(drop=True)
    if not observed.equals(expected):
        raise G12DFrozenSubsetValidationError(
            "Stage-B patient population does not exactly match Stage C"
        )


def _evaluate_population(
    patients: pd.DataFrame, *, n_bootstrap: int, bootstrap_seed: int
) -> dict[str, Any]:
    per_cohort: dict[str, Any] = {}
    for cohort, group in patients.groupby("cohort", sort=True):
        cohort_frame = group.reset_index(drop=True)
        cohort_strata = [
            label_group.index.to_numpy(dtype=int)
            for _, label_group in cohort_frame.groupby("label", sort=True)
        ]
        per_cohort[str(cohort)] = _metric_block(
            cohort_frame,
            n_bootstrap=n_bootstrap,
            seed=_derived_seed(bootstrap_seed, f"cohort:{cohort}"),
            strata=cohort_strata,
        )
    pooled_frame = patients.reset_index(drop=True)
    pooled_strata = [
        group.index.to_numpy(dtype=int)
        for _, group in pooled_frame.groupby(["cohort", "label"], sort=True)
    ]
    pooled = _metric_block(
        pooled_frame,
        n_bootstrap=n_bootstrap,
        seed=_derived_seed(bootstrap_seed, "pooled"),
        strata=pooled_strata,
    )
    return {"per_cohort": per_cohort, "pooled_cohort_stratified": pooled}


def _metric_block(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    strata: Sequence[NDArray[np.int64]] | None,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int8)
    scores = frame["prob_1"].to_numpy(dtype=float)
    point = _discrimination_metrics(labels, scores)
    if point["auroc"] is None:
        raise G12DFrozenSubsetValidationError("each reported population must contain both labels")
    if strata is None:
        strata = [np.arange(len(frame), dtype=np.int64)]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"auroc": [], "auprc": []}
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata]
        )
        values = _discrimination_metrics(labels[indices], scores[indices])
        for metric in samples:
            if values[metric] is not None:
                samples[metric].append(float(values[metric]))
    return {
        "n_patients": int(len(frame)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": _metric_summary(float(point["auroc"]), samples["auroc"]),
        "auprc": _metric_summary(float(point["auprc"]), samples["auprc"]),
    }


def _nested_context_delta(
    stage_a: pd.DataFrame,
    stage_b: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    per_cohort: dict[str, Any] = {}
    cohorts = sorted(stage_a["cohort"].unique())
    for cohort in cohorts:
        marginal = stage_a.loc[stage_a["cohort"] == cohort].reset_index(drop=True)
        conditional = stage_b.loc[stage_b["cohort"] == cohort].reset_index(drop=True)
        point = _auc(marginal) - _auc(conditional)
        samples = _nested_delta_samples(
            marginal,
            n_bootstrap=n_bootstrap,
            seed=_derived_seed(bootstrap_seed, f"cohort:{cohort}"),
            pooled=False,
        )
        per_cohort[str(cohort)] = _metric_summary(point, samples)

    marginal_pooled = stage_a.reset_index(drop=True)
    conditional_pooled = stage_b.reset_index(drop=True)
    pooled_point = _auc(marginal_pooled) - _auc(conditional_pooled)
    pooled_samples = _nested_delta_samples(
        marginal_pooled,
        n_bootstrap=n_bootstrap,
        seed=_derived_seed(bootstrap_seed, "pooled"),
        pooled=True,
    )
    return {
        "metric": "raw patient AUROC",
        "direction": "Stage A minus Stage B",
        "bootstrap": "shared patient resampling within held-out cohort x kras_context",
        "per_cohort": per_cohort,
        "pooled_cohort_stratified": _metric_summary(pooled_point, pooled_samples),
    }


def _nested_delta_samples(
    marginal: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    pooled: bool,
) -> list[float]:
    group_columns = ["kras_context"]
    if pooled:
        group_columns.insert(0, "cohort")
    groups = [group.reset_index(drop=True) for _, group in marginal.groupby(group_columns, sort=True)]
    expected_groups = len(CONTEXTS) * (marginal["cohort"].nunique() if pooled else 1)
    if len(groups) != expected_groups:
        raise G12DFrozenSubsetValidationError(
            "nested context bootstrap is missing a cohort x kras_context stratum"
        )
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        sampled = pd.concat(
            [group.iloc[rng.integers(0, len(group), size=len(group))] for group in groups],
            ignore_index=True,
        )
        conditional = sampled.loc[sampled["kras_context"].isin(CONDITIONAL_CONTEXTS)]
        samples.append(_auc(sampled) - _auc(conditional))
    return samples


def _matched_marginal_controls(
    stage_a: pd.DataFrame,
    stage_b: pd.DataFrame,
    *,
    n_repeats: int,
    seed: int,
) -> dict[str, Any]:
    cohorts = sorted(stage_a["cohort"].unique())
    cohort_inputs: dict[str, dict[str, Any]] = {}
    for cohort in cohorts:
        marginal = stage_a.loc[stage_a["cohort"] == cohort].reset_index(drop=True)
        conditional = stage_b.loc[stage_b["cohort"] == cohort].reset_index(drop=True)
        positives = marginal.loc[marginal["kras_context"] == G12D_CONTEXT]
        b_positives = conditional.loc[conditional["label"] == 1]
        if set(positives["patient_id"]) != set(b_positives["patient_id"]):
            raise G12DFrozenSubsetValidationError(
                f"Stage A and B do not contain identical G12D patients in {cohort}"
            )
        target_negative = int((conditional["label"] == 0).sum())
        pools = {
            context: marginal.loc[marginal["kras_context"] == context].reset_index(drop=True)
            for context in NEGATIVE_CONTEXTS
        }
        counts = {context: len(pool) for context, pool in pools.items()}
        quotas = _proportional_quotas(counts, target_negative)
        cohort_inputs[str(cohort)] = {
            "positives": positives.reset_index(drop=True),
            "pools": pools,
            "population": counts,
            "quotas": quotas,
            "observed_b": conditional,
        }

    rng = np.random.default_rng(seed)
    samples: dict[str, dict[str, list[float]]] = {
        cohort: {"auroc": [], "auprc": []} for cohort in cohorts
    }
    pooled_samples: dict[str, list[float]] = {"auroc": [], "auprc": []}
    for _ in range(n_repeats):
        pooled_parts: list[pd.DataFrame] = []
        for cohort in cohorts:
            inputs = cohort_inputs[str(cohort)]
            selected = [inputs["positives"]]
            for context in NEGATIVE_CONTEXTS:
                pool = inputs["pools"][context]
                quota = inputs["quotas"][context]
                indices = rng.choice(len(pool), size=quota, replace=False)
                selected.append(pool.iloc[indices])
            matched = pd.concat(selected, ignore_index=True)
            values = _discrimination_metrics(
                matched["label"].to_numpy(dtype=np.int8),
                matched["prob_1"].to_numpy(dtype=float),
            )
            for metric in samples[str(cohort)]:
                samples[str(cohort)][metric].append(float(values[metric]))
            pooled_parts.append(matched)
        pooled = pd.concat(pooled_parts, ignore_index=True)
        values = _discrimination_metrics(
            pooled["label"].to_numpy(dtype=np.int8),
            pooled["prob_1"].to_numpy(dtype=float),
        )
        for metric in pooled_samples:
            pooled_samples[metric].append(float(values[metric]))

    per_cohort: dict[str, Any] = {}
    for cohort in cohorts:
        inputs = cohort_inputs[str(cohort)]
        observed = _discrimination_metrics(
            inputs["observed_b"]["label"].to_numpy(dtype=np.int8),
            inputs["observed_b"]["prob_1"].to_numpy(dtype=float),
        )
        per_cohort[str(cohort)] = {
            "n_positive": int(len(inputs["positives"])),
            "n_negative": int(sum(inputs["quotas"].values())),
            "negative_context_population": inputs["population"],
            "negative_context_quota": inputs["quotas"],
            "auroc": _matched_summary(float(observed["auroc"]), samples[str(cohort)]["auroc"]),
            "auprc": _matched_summary(float(observed["auprc"]), samples[str(cohort)]["auprc"]),
        }

    observed_pooled = _discrimination_metrics(
        stage_b["label"].to_numpy(dtype=np.int8),
        stage_b["prob_1"].to_numpy(dtype=float),
    )
    return {
        "method": (
            "retain all marginal G12D patients; draw the Stage-B negative count "
            "without replacement using fixed proportional quotas from the marginal "
            "other-KRAS and KRAS-wild-type negative pools"
        ),
        "sampling_unit": "patient",
        "n_repeats": n_repeats,
        "seed": seed,
        "without_replacement": True,
        "reference_interval": "2.5th to 97.5th percentiles across frozen marginal draws",
        "per_cohort": per_cohort,
        "pooled_cohort_stratified": {
            "auroc": _matched_summary(float(observed_pooled["auroc"]), pooled_samples["auroc"]),
            "auprc": _matched_summary(float(observed_pooled["auprc"]), pooled_samples["auprc"]),
        },
    }


def _proportional_quotas(counts: Mapping[str, int], total: int) -> dict[str, int]:
    population = sum(counts.values())
    if total <= 0 or total > population:
        raise G12DFrozenSubsetValidationError(
            f"cannot draw {total} matched negatives from population {population}"
        )
    if any(count <= 0 for count in counts.values()):
        raise G12DFrozenSubsetValidationError(
            "matched marginal control requires both negative contexts"
        )
    raw = {context: total * count / population for context, count in counts.items()}
    quotas = {context: int(np.floor(value)) for context, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda context: (
            -(raw[context] - quotas[context]),
            -counts[context],
            context,
        ),
    )
    for context in order[:remainder]:
        quotas[context] += 1
    if sum(quotas.values()) != total or any(quotas[key] > counts[key] for key in counts):
        raise G12DFrozenSubsetValidationError("matched marginal quota allocation failed")
    return quotas


def _discrimination_metrics(
    labels: NDArray[np.int8], scores: NDArray[np.float64]
) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _auc(frame: pd.DataFrame) -> float:
    return float(
        roc_auc_score(
            frame["label"].to_numpy(dtype=np.int8),
            frame["prob_1"].to_numpy(dtype=float),
        )
    )


def _metric_summary(value: float, samples: Sequence[float]) -> dict[str, Any]:
    interval = np.percentile(np.asarray(samples, dtype=float), [2.5, 97.5])
    return {
        "value": float(value),
        "ci95": [float(interval[0]), float(interval[1])],
        "n_bootstrap_valid": len(samples),
    }


def _matched_summary(observed_b: float, samples: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(samples, dtype=float)
    interval = np.percentile(values, [2.5, 97.5])
    median = float(np.median(values))
    percentile = 100.0 * float(
        ((values < observed_b).sum() + 0.5 * (values == observed_b).sum()) / len(values)
    )
    return {
        "observed_stage_b": float(observed_b),
        "matched_marginal_median": median,
        "reference_interval95": [float(interval[0]), float(interval[1])],
        "stage_b_percentile": percentile,
        "matched_median_minus_stage_b": median - float(observed_b),
        "n_repeats_valid": len(values),
    }


def _render_markdown(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# G12D frozen conditional Stage-B evaluation",
        "",
        "Stage B uses the frozen Stage-A models and the exact Stage-C patient set. "
        "Scores use stored raw slide logits: seeds are averaged within slide, then "
        "slides within patient. No calibration or threshold was fitted on held-out labels.",
        "",
        "## Discrimination",
        "",
        "| Stage | Cohort | Patients (+/−) | Prevalence | AUROC (95% CI) | AUPRC (95% CI) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for stage in ("stage_a", "stage_b"):
        block = metrics[stage]
        for cohort, values in block["per_cohort"].items():
            lines.append(_metric_row(stage, cohort, values))
        lines.append(_metric_row(stage, "Pooled", block["pooled_cohort_stratified"]))

    lines.extend(
        [
            "",
            "## Context contrast",
            "",
            "Positive values mean the marginal Stage-A task scores higher than frozen "
            "conditional Stage B. Intervals use shared patient resampling within "
            "held-out cohort × KRAS context.",
            "",
            "| Cohort | AUROC_A − AUROC_B (95% CI) |",
            "|---|---:|",
        ]
    )
    delta = metrics["delta_context_auc_a_minus_b"]
    for cohort, summary in delta["per_cohort"].items():
        lines.append(f"| {cohort} | {_format_summary(summary)} |")
    lines.append(f"| Pooled | {_format_summary(delta['pooled_cohort_stratified'])} |")

    lines.extend(
        [
            "",
            "## Matched-marginal control",
            "",
            "Each frozen marginal draw matches the Stage-B positive and negative counts "
            "while preserving the marginal other-KRAS / KRAS-wild-type negative mix. "
            "The interval is a repeated-subsampling reference interval, not a confidence "
            "interval.",
            "",
            "| Cohort | Stage-B AUROC | Matched median (95% reference interval) | "
            "Stage-B percentile |",
            "|---|---:|---:|---:|",
        ]
    )
    matched = metrics["matched_marginal"]
    for cohort, values in matched["per_cohort"].items():
        lines.append(_matched_row(cohort, values["auroc"]))
    lines.append(_matched_row("Pooled", matched["pooled_cohort_stratified"]["auroc"]))
    lines.append("")
    return "\n".join(lines)


def _metric_row(stage: str, cohort: str, block: Mapping[str, Any]) -> str:
    return (
        f"| {stage.replace('_', ' ').title()} | {cohort} | "
        f"{block['n_patients']} ({block['n_positive']}/{block['n_negative']}) | "
        f"{block['prevalence']:.3f} | {_format_summary(block['auroc'])} | "
        f"{_format_summary(block['auprc'])} |"
    )


def _matched_row(cohort: str, summary: Mapping[str, Any]) -> str:
    interval = summary["reference_interval95"]
    return (
        f"| {cohort} | {summary['observed_stage_b']:.3f} | "
        f"{summary['matched_marginal_median']:.3f} "
        f"({interval[0]:.3f}–{interval[1]:.3f}) | "
        f"{summary['stage_b_percentile']:.1f}% |"
    )


def _format_summary(summary: Mapping[str, Any]) -> str:
    interval = summary["ci95"]
    return f"{summary['value']:.3f} ({interval[0]:.3f}–{interval[1]:.3f})"


def _manifest_provenance(
    source: pd.DataFrame | str | Path, canonical: pd.DataFrame
) -> dict[str, Any]:
    canonical_bytes = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    output: dict[str, Any] = {
        "canonical_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "n_slides": int(len(canonical)),
        "n_patients": int(canonical["patient_id"].nunique()),
        "cohorts": sorted(canonical["cohort"].unique()),
    }
    if not isinstance(source, pd.DataFrame):
        path = Path(source)
        output.update(
            {
                "source_path": str(path),
                "source_sha256": _file_sha256(path),
                "source_size_bytes": path.stat().st_size,
            }
        )
    else:
        output["source_path"] = None
    return output


def _rotation_provenance(record: RotationRecord) -> dict[str, Any]:
    path = Path(record.prediction_path)
    return {
        "seed": str(record.seed),
        "fold": record.fold,
        "heldout_cohort": record.heldout_cohort,
        "prediction_path": str(path),
        "prediction_sha256": _file_sha256(path),
        "prediction_size_bytes": path.stat().st_size,
        "training_fingerprint": record.training_fingerprint,
        "validated_against": "full Stage-A held-out cohort before Stage-B filtering",
    }


def _identifier_values(series: pd.Series, *, context: str) -> pd.Series:
    values = series.astype("string").str.strip()
    if bool(values.isna().any() or values.eq("").any()):
        raise G12DFrozenSubsetValidationError(f"{context} contains missing or blank values")
    return values.astype(str)


def _binary_labels(series: pd.Series, *, context: str) -> pd.Series:
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise G12DFrozenSubsetValidationError(f"{context} must be numeric binary labels") from exc
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not bool(np.isin(values, [0.0, 1.0]).all()):
        raise G12DFrozenSubsetValidationError(f"{context} must contain only 0 and 1")
    return pd.Series(values.astype(np.int8), index=series.index)


def _numeric_values(series: pd.Series, *, context: str) -> NDArray[np.float64]:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise G12DFrozenSubsetValidationError(f"prediction {context} must be numeric") from exc
    if not np.isfinite(values).all():
        raise G12DFrozenSubsetValidationError(
            f"prediction {context} contains nonfinite values"
        )
    return values


def _sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    output = np.empty_like(values)
    nonnegative = values >= 0
    output[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    output[~nonnegative] = exponent / (1.0 + exponent)
    return output


def _derived_seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_evidence(path: Path) -> dict[str, Any]:
    return {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}


def _preview(values: Sequence[Any], *, limit: int = 10) -> str:
    rendered = [str(value) for value in values[:limit]]
    if len(values) > limit:
        rendered.append(f"... (+{len(values) - limit} more)")
    return repr(rendered)


__all__ = [
    "G12DFrozenSubsetEvaluation",
    "G12DFrozenSubsetValidationError",
    "evaluate_g12d_frozen_subset",
]
