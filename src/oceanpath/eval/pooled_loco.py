"""Strict patient-level evaluation for pooled leave-one-cohort-out runs.

Each LOCO rotation contributes predictions only for its named held-out cohort.
Across repeated seeds, slide logits are averaged first.  The resulting slide
logits are then averaged within patients and transformed back to probabilities.
The pooled confidence interval resamples patients independently within cohort,
so the cohort composition of every bootstrap sample is fixed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, roc_auc_score


class PooledLocoValidationError(ValueError):
    """Raised when LOCO inputs do not satisfy the evaluation contract."""


@dataclass(frozen=True)
class RotationRecord:
    """One held-out cohort prediction artifact from one random seed."""

    heldout_cohort: str
    prediction_path: str | Path
    seed: str | int = "seed_0"
    fold: int | None = None
    training_fingerprint: str | None = None


@dataclass
class PooledLocoResult:
    """In-memory pooled LOCO outputs."""

    slide_predictions: pd.DataFrame
    patient_predictions: pd.DataFrame
    metrics: dict[str, object]
    report_markdown: str

    def write(self, output_dir: str | Path) -> dict[str, Path]:
        """Write the four canonical pooled LOCO artifacts."""
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        paths = {
            "slide_predictions": destination / "slide_predictions.parquet",
            "patient_predictions": destination / "patient_predictions.parquet",
            "metrics": destination / "metrics.json",
            "report": destination / "report.md",
        }
        self.slide_predictions.to_parquet(paths["slide_predictions"], index=False)
        self.patient_predictions.to_parquet(paths["patient_predictions"], index=False)
        paths["metrics"].write_text(
            json.dumps(self.metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        paths["report"].write_text(self.report_markdown)
        return paths


def evaluate_pooled_loco(
    rotations: Sequence[RotationRecord],
    manifest: pd.DataFrame | str | Path,
    *,
    output_dir: str | Path | None = None,
    manifest_slide_column: str = "slide_id",
    manifest_patient_column: str = "patient_id",
    manifest_cohort_column: str = "cohort",
    manifest_label_column: str = "label",
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 17,
) -> PooledLocoResult:
    """Validate, ensemble, aggregate, and score explicit LOCO rotations.

    The manifest is the source of truth for held-out membership, labels,
    patients, and cohorts.  Every seed must have one rotation for every cohort
    present in the manifest, and each rotation must predict exactly that
    cohort's slides once.
    """
    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap <= 0:
        raise PooledLocoValidationError("n_bootstrap must be a positive integer")
    if not isinstance(bootstrap_seed, int) or isinstance(bootstrap_seed, bool):
        raise PooledLocoValidationError("bootstrap_seed must be an integer")
    if not rotations:
        raise PooledLocoValidationError("at least one rotation record is required")

    canonical_manifest = _prepare_manifest(
        manifest,
        slide_column=manifest_slide_column,
        patient_column=manifest_patient_column,
        cohort_column=manifest_cohort_column,
        label_column=manifest_label_column,
    )
    manifest_provenance = _manifest_provenance(manifest, canonical_manifest)
    normalized_records = _validate_rotation_records(rotations, canonical_manifest)
    slide_predictions = _load_and_ensemble_predictions(
        normalized_records,
        canonical_manifest,
    )
    patient_predictions = _aggregate_patients(slide_predictions)
    metrics = _compute_reported_metrics(
        patient_predictions,
        n_seeds=len({record.seed for record in normalized_records}),
        rotations=normalized_records,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
        manifest_provenance=manifest_provenance,
    )
    report = _render_markdown(metrics)
    result = PooledLocoResult(
        slide_predictions=slide_predictions,
        patient_predictions=patient_predictions,
        metrics=metrics,
        report_markdown=report,
    )
    if output_dir is not None:
        result.write(output_dir)
    return result


def _read_table(table: pd.DataFrame | str | Path, *, name: str) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    path = Path(table)
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    if path.suffix.casefold() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.casefold() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.casefold() == ".tsv" else ","
        return pd.read_csv(path, sep=separator)
    raise PooledLocoValidationError(
        f"unsupported {name} format {path.suffix!r}; use CSV, TSV, or Parquet"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_provenance(
    manifest: pd.DataFrame | str | Path,
    canonical_manifest: pd.DataFrame,
) -> dict[str, object]:
    canonical_csv = canonical_manifest.to_csv(index=False, lineterminator="\n").encode()
    provenance: dict[str, object] = {
        "canonical_sha256": hashlib.sha256(canonical_csv).hexdigest(),
        "canonical_hash_definition": (
            "sha256 of normalized slide_id,patient_id,cohort,label CSV sorted by slide_id"
        ),
        "n_slides": int(len(canonical_manifest)),
        "n_patients": int(canonical_manifest["patient_id"].nunique()),
    }
    if isinstance(manifest, pd.DataFrame):
        provenance["source"] = "in_memory_dataframe"
        provenance["source_sha256"] = None
    else:
        path = Path(manifest)
        provenance["source"] = str(path)
        provenance["source_sha256"] = _file_sha256(path)
    return provenance


def _prepare_manifest(
    manifest: pd.DataFrame | str | Path,
    *,
    slide_column: str,
    patient_column: str,
    cohort_column: str,
    label_column: str,
) -> pd.DataFrame:
    frame = _read_table(manifest, name="manifest")
    requested = [slide_column, patient_column, cohort_column, label_column]
    missing_columns = [column for column in requested if column not in frame.columns]
    if missing_columns:
        raise PooledLocoValidationError(
            f"manifest is missing required columns: {missing_columns}"
        )
    frame = frame[requested].rename(
        columns={
            slide_column: "slide_id",
            patient_column: "patient_id",
            cohort_column: "cohort",
            label_column: "label",
        }
    )
    for column in ("slide_id", "patient_id", "cohort"):
        _validate_identifier_column(frame[column], context=f"manifest {column}")
        frame[column] = frame[column].astype(str).str.strip()
    frame["label"] = _binary_labels(frame["label"], context="manifest label")

    duplicate_slides = sorted(frame.loc[frame["slide_id"].duplicated(False), "slide_id"].unique())
    if duplicate_slides:
        raise PooledLocoValidationError(
            f"manifest contains duplicate slide_id values: {_preview(duplicate_slides)}"
        )
    patient_label_counts = frame.groupby("patient_id", sort=False)["label"].nunique()
    conflicting_labels = sorted(patient_label_counts[patient_label_counts > 1].index)
    if conflicting_labels:
        raise PooledLocoValidationError(
            "manifest has patient label conflicts: " f"{_preview(conflicting_labels)}"
        )
    patient_cohort_counts = frame.groupby("patient_id", sort=False)["cohort"].nunique()
    conflicting_cohorts = sorted(patient_cohort_counts[patient_cohort_counts > 1].index)
    if conflicting_cohorts:
        raise PooledLocoValidationError(
            "manifest has patient cohort conflicts: " f"{_preview(conflicting_cohorts)}"
        )
    if frame.empty:
        raise PooledLocoValidationError("manifest must contain at least one slide")
    return frame.sort_values("slide_id", kind="stable").reset_index(drop=True)


def _validate_identifier_column(series: pd.Series, *, context: str) -> None:
    missing = series.isna()
    blank = series.astype("string").str.strip().eq("").fillna(False)
    if bool((missing | blank).any()):
        raise PooledLocoValidationError(f"{context} contains missing or blank values")


def _binary_labels(series: pd.Series, *, context: str) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise PooledLocoValidationError(f"{context} must contain only binary 0/1 values") from exc
    array = values.to_numpy(dtype=float)
    if not np.isfinite(array).all() or not np.isin(array, [0.0, 1.0]).all():
        raise PooledLocoValidationError(f"{context} must contain only binary 0/1 values")
    return values.astype(np.int8)


def _validate_rotation_records(
    rotations: Sequence[RotationRecord],
    manifest: pd.DataFrame,
) -> tuple[RotationRecord, ...]:
    normalized: list[RotationRecord] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_paths: set[Path] = set()
    manifest_cohorts = set(manifest["cohort"])
    for raw_record in rotations:
        if not isinstance(raw_record, RotationRecord):
            raise TypeError("rotations must contain RotationRecord instances")
        seed = str(raw_record.seed).strip()
        cohort = str(raw_record.heldout_cohort).strip()
        if not seed or not cohort:
            raise PooledLocoValidationError("rotation seed and heldout_cohort must not be blank")
        if cohort not in manifest_cohorts:
            raise PooledLocoValidationError(
                f"rotation names unknown held-out cohort {cohort!r}; "
                f"manifest cohorts={sorted(manifest_cohorts)}"
            )
        path = Path(raw_record.prediction_path)
        if not path.is_file():
            raise FileNotFoundError(f"held-out predictions not found: {path}")
        resolved_path = path.resolve()
        key = (seed, cohort)
        if key in seen_keys:
            raise PooledLocoValidationError(
                f"duplicate rotation for seed={seed!r}, heldout_cohort={cohort!r}"
            )
        if resolved_path in seen_paths:
            raise PooledLocoValidationError(
                f"prediction artifact is reused by multiple rotations: {path}"
            )
        seen_keys.add(key)
        seen_paths.add(resolved_path)
        normalized.append(
            RotationRecord(
                heldout_cohort=cohort,
                prediction_path=path,
                seed=seed,
                fold=raw_record.fold,
                training_fingerprint=_normalize_optional_fingerprint(
                    raw_record.training_fingerprint,
                    context=f"seed={seed!r}, heldout_cohort={cohort!r}",
                ),
            )
        )

    cohorts_by_seed: dict[str, set[str]] = {}
    for record in normalized:
        cohorts_by_seed.setdefault(str(record.seed), set()).add(record.heldout_cohort)
    for seed, observed_cohorts in sorted(cohorts_by_seed.items()):
        missing = manifest_cohorts - observed_cohorts
        extra = observed_cohorts - manifest_cohorts
        if missing or extra:
            raise PooledLocoValidationError(
                f"seed {seed!r} rotations do not cover every manifest cohort; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
    return tuple(
        sorted(
            normalized,
            key=lambda record: (
                str(record.seed),
                record.heldout_cohort,
                -1 if record.fold is None else record.fold,
            ),
        )
    )


def _normalize_optional_fingerprint(value: str | None, *, context: str) -> str | None:
    if value is None:
        return None
    fingerprint = str(value).strip()
    if not fingerprint:
        raise PooledLocoValidationError(
            f"training_fingerprint must not be blank for {context}"
        )
    return fingerprint


def _load_and_ensemble_predictions(
    rotations: Sequence[RotationRecord],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    expected_by_cohort = {
        cohort: set(group["slide_id"])
        for cohort, group in manifest.groupby("cohort", sort=False)
    }
    prediction_frames: list[pd.DataFrame] = []
    for record in rotations:
        raw = pd.read_parquet(record.prediction_path)
        prediction_frames.append(
            _validate_prediction_frame(
                raw,
                record=record,
                manifest=manifest,
                expected_slides=expected_by_cohort[record.heldout_cohort],
            )
        )

    stacked = pd.concat(prediction_frames, ignore_index=True)
    n_seeds = len({str(record.seed) for record in rotations})
    counts = stacked.groupby("slide_id", sort=False)["seed"].nunique()
    invalid_counts = counts[counts != n_seeds]
    if not invalid_counts.empty:
        raise PooledLocoValidationError(
            "slides do not have exactly one prediction from every seed: "
            f"{_preview(sorted(invalid_counts.index))}"
        )
    mean_logits = (
        stacked.groupby("slide_id", sort=False)["seed_logit"].mean().rename("mean_logit")
    )
    output = manifest.merge(mean_logits, on="slide_id", how="left", validate="one_to_one")
    if output["mean_logit"].isna().any():
        missing = sorted(output.loc[output["mean_logit"].isna(), "slide_id"])
        raise PooledLocoValidationError(
            f"pooled slide predictions are incomplete: {_preview(missing)}"
        )
    output["prob_1"] = _sigmoid(output["mean_logit"].to_numpy(dtype=float))
    output["n_seeds"] = n_seeds
    columns = [
        "slide_id",
        "patient_id",
        "cohort",
        "label",
        "mean_logit",
        "prob_1",
        "n_seeds",
    ]
    return output.sort_values(["cohort", "patient_id", "slide_id"], kind="stable")[
        columns
    ].reset_index(drop=True)


def _validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    record: RotationRecord,
    manifest: pd.DataFrame,
    expected_slides: set[str],
) -> pd.DataFrame:
    context = f"seed={record.seed!r}, heldout_cohort={record.heldout_cohort!r}"
    required = ["slide_id", "label", "prob_1"]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise PooledLocoValidationError(
            f"prediction file for {context} is missing columns: {missing_columns}"
        )
    output = frame.copy()
    _validate_identifier_column(output["slide_id"], context=f"prediction slide_id ({context})")
    output["slide_id"] = output["slide_id"].astype(str).str.strip()
    duplicates = sorted(output.loc[output["slide_id"].duplicated(False), "slide_id"].unique())
    if duplicates:
        raise PooledLocoValidationError(
            f"duplicate prediction slide_id values for {context}: {_preview(duplicates)}"
        )
    observed_slides = set(output["slide_id"])
    missing = expected_slides - observed_slides
    extra = observed_slides - expected_slides
    if missing or extra:
        raise PooledLocoValidationError(
            f"held-out slide coverage mismatch for {context}; "
            f"missing={_preview(sorted(missing))}, extra={_preview(sorted(extra))}"
        )

    output["label"] = _binary_labels(output["label"], context=f"prediction label ({context})")
    try:
        probabilities = pd.to_numeric(output["prob_1"], errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise PooledLocoValidationError(f"prob_1 must be numeric for {context}") from exc
    if not np.isfinite(probabilities).all():
        raise PooledLocoValidationError(f"prob_1 contains nonfinite values for {context}")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise PooledLocoValidationError(f"prob_1 contains values outside [0, 1] for {context}")

    canonical = manifest.set_index("slide_id").loc[output["slide_id"]]
    canonical_labels = canonical["label"].to_numpy(dtype=np.int8)
    predicted_labels = output["label"].to_numpy(dtype=np.int8)
    if not np.array_equal(predicted_labels, canonical_labels):
        mismatched = output.loc[predicted_labels != canonical_labels, "slide_id"].tolist()
        raise PooledLocoValidationError(
            f"prediction labels disagree with manifest for {context}: {_preview(mismatched)}"
        )
    if "patient_id" in output.columns:
        _validate_metadata_match(
            output,
            canonical,
            column="patient_id",
            context=context,
        )
    if "cohort" in output.columns:
        _validate_metadata_match(output, canonical, column="cohort", context=context)

    return pd.DataFrame(
        {
            "slide_id": output["slide_id"].to_numpy(),
            "seed": str(record.seed),
            "seed_logit": _logit(probabilities),
        }
    )


def _validate_metadata_match(
    predictions: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    column: str,
    context: str,
) -> None:
    _validate_identifier_column(
        predictions[column],
        context=f"prediction {column} ({context})",
    )
    observed = predictions[column].astype(str).str.strip().to_numpy()
    expected = canonical[column].astype(str).to_numpy()
    if not np.array_equal(observed, expected):
        mismatched = predictions.loc[observed != expected, "slide_id"].tolist()
        raise PooledLocoValidationError(
            f"prediction {column} disagrees with manifest for {context}: "
            f"{_preview(mismatched)}"
        )


def _logit(probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probabilities.astype(np.float64), epsilon, 1.0 - epsilon)
    result: NDArray[np.float64] = np.log(clipped) - np.log1p(-clipped)
    return result


def _sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _aggregate_patients(slides: pd.DataFrame) -> pd.DataFrame:
    conflicts = slides.groupby("patient_id", sort=False).agg(
        n_labels=("label", "nunique"),
        n_cohorts=("cohort", "nunique"),
    )
    if bool((conflicts["n_labels"] > 1).any()):
        patient_ids = sorted(conflicts.index[conflicts["n_labels"] > 1])
        raise PooledLocoValidationError(
            f"patient label conflicts in pooled predictions: {_preview(patient_ids)}"
        )
    if bool((conflicts["n_cohorts"] > 1).any()):
        patient_ids = sorted(conflicts.index[conflicts["n_cohorts"] > 1])
        raise PooledLocoValidationError(
            f"patient cohort conflicts in pooled predictions: {_preview(patient_ids)}"
        )
    patients = (
        slides.groupby("patient_id", sort=False)
        .agg(
            cohort=("cohort", "first"),
            label=("label", "first"),
            mean_logit=("mean_logit", "mean"),
            n_slides=("slide_id", "size"),
            n_seeds=("n_seeds", "first"),
        )
        .reset_index()
    )
    patients["prob_1"] = _sigmoid(patients["mean_logit"].to_numpy(dtype=float))
    columns = [
        "patient_id",
        "cohort",
        "label",
        "mean_logit",
        "prob_1",
        "n_slides",
        "n_seeds",
    ]
    return patients.sort_values(["cohort", "patient_id"], kind="stable")[columns].reset_index(
        drop=True
    )


def _compute_reported_metrics(
    patients: pd.DataFrame,
    *,
    n_seeds: int,
    rotations: Sequence[RotationRecord],
    n_bootstrap: int,
    bootstrap_seed: int,
    manifest_provenance: Mapping[str, object],
) -> dict[str, object]:
    per_cohort: dict[str, object] = {}
    for cohort, group in patients.groupby("cohort", sort=True):
        per_cohort[str(cohort)] = _metric_block(
            group,
            n_bootstrap=n_bootstrap,
            seed=_derived_seed(bootstrap_seed, f"cohort:{cohort}"),
            stratify_by_cohort=False,
        )
    pooled = _metric_block(
        patients,
        n_bootstrap=n_bootstrap,
        seed=_derived_seed(bootstrap_seed, "pooled"),
        stratify_by_cohort=True,
    )
    seed_ids = sorted({str(record.seed) for record in rotations})
    rotation_summary = [
        {
            "seed": str(record.seed),
            "fold": record.fold,
            "heldout_cohort": record.heldout_cohort,
            "prediction_path": str(record.prediction_path),
            "prediction_sha256": _file_sha256(Path(record.prediction_path)),
            "training_fingerprint": record.training_fingerprint,
        }
        for record in rotations
    ]
    return {
        "analysis": "pooled_loco",
        "manifest": dict(manifest_provenance),
        "aggregation": {
            "across_seeds": "mean slide logit",
            "within_patient": "sigmoid(mean slide logit)",
        },
        "bootstrap": {
            "method": "percentile",
            "confidence_level": 0.95,
            "n_resamples": n_bootstrap,
            "seed": bootstrap_seed,
            "unit": "patient",
            "pooled_stratification": "cohort",
        },
        "n_seeds": n_seeds,
        "seeds": seed_ids,
        "rotations": rotation_summary,
        "per_cohort": per_cohort,
        "pooled": pooled,
    }


def _metric_block(
    patients: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    stratify_by_cohort: bool,
) -> dict[str, object]:
    labels = patients["label"].to_numpy(dtype=np.int8)
    probabilities = patients["prob_1"].to_numpy(dtype=float)
    point = _discrimination_metrics(labels, probabilities)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"auroc": [], "auprc": []}
    if stratify_by_cohort:
        strata = [
            group.index.to_numpy(dtype=int)
            for _, group in patients.reset_index(drop=True).groupby("cohort", sort=True)
        ]
    else:
        strata = [np.arange(len(patients), dtype=int)]
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [rng.choice(stratum, size=len(stratum), replace=True) for stratum in strata]
        )
        resampled = _discrimination_metrics(labels[indices], probabilities[indices])
        for metric_name in samples:
            value = resampled[metric_name]
            if value is not None:
                samples[metric_name].append(value)

    ci95: dict[str, list[float] | None] = {}
    valid_counts: dict[str, int] = {}
    for metric_name, values in samples.items():
        valid_counts[metric_name] = len(values)
        if values:
            ci95[metric_name] = [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ]
        else:
            ci95[metric_name] = None
    return {
        "n_patients": int(len(patients)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "prevalence": float(labels.mean()),
        "auroc": point["auroc"],
        "auprc": point["auprc"],
        "ci95": ci95,
        "n_bootstrap_valid": valid_counts,
    }


def _discrimination_metrics(
    labels: NDArray[np.int8],
    probabilities: NDArray[np.float64],
) -> dict[str, float | None]:
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
    }


def _derived_seed(base_seed: int, namespace: str) -> int:
    payload = f"{base_seed}:{namespace}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _render_markdown(metrics: Mapping[str, object]) -> str:
    per_cohort = metrics["per_cohort"]
    pooled = metrics["pooled"]
    if not isinstance(per_cohort, Mapping) or not isinstance(pooled, Mapping):
        raise TypeError("invalid metrics structure")
    lines = [
        "# Pooled LOCO evaluation",
        "",
        "Scores average slide logits across seeds, then average slide logits "
        "within each patient. Pooled bootstrap resampling is patient-level and "
        "stratified by cohort.",
        "",
        "| Scope | Patients | Positive | AUROC (95% CI) | AUPRC (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort, block in per_cohort.items():
        if not isinstance(block, Mapping):
            raise TypeError("invalid per-cohort metrics structure")
        lines.append(_markdown_metric_row(str(cohort), block))
    lines.append(_markdown_metric_row("Pooled", pooled))
    lines.append("")
    return "\n".join(lines)


def _markdown_metric_row(name: str, block: Mapping[str, object]) -> str:
    ci95 = block["ci95"]
    if not isinstance(ci95, Mapping):
        raise TypeError("invalid confidence interval structure")
    auroc_ci = ci95["auroc"]
    auprc_ci = ci95["auprc"]
    return (
        f"| {name} | {block['n_patients']} | {block['n_positive']} | "
        f"{_format_estimate(block['auroc'], auroc_ci)} | "
        f"{_format_estimate(block['auprc'], auprc_ci)} |"
    )


def _format_estimate(estimate: object, interval: object) -> str:
    if estimate is None:
        return "not estimable"
    if not isinstance(estimate, (int, float)):
        raise TypeError("metric estimate must be numeric or None")
    value = float(estimate)
    if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)):
        return f"{value:.3f}"
    if len(interval) != 2:
        return f"{value:.3f}"
    return f"{value:.3f} ({float(interval[0]):.3f}–{float(interval[1]):.3f})"


def _preview(values: Sequence[object], limit: int = 5) -> str:
    shown = [str(value) for value in values[:limit]]
    suffix = "..." if len(values) > limit else ""
    return f"[{', '.join(shown)}{suffix}]"
