#!/usr/bin/env python3
"""Independent verifier for the final-v8 E2-CPHT existing-model Orion run.

The verifier never performs model inference and never changes a prediction or
analysis artifact.  It rehashes the sealed inputs and checkpoints, validates
the complete per-model prediction roster, independently repeats slide-to-
patient aggregation and the 10,000-draw stratified bootstrap, and writes the
verification record followed by the bundle receipt.

Usage::

    python aim2_cross_protocol_transfer_verify.py
    python aim2_cross_protocol_transfer_verify.py --read-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_cross_protocol_transfer as runner  # noqa: E402
from oceanpath.aim1 import lineage  # noqa: E402
from oceanpath.datasets.packed import feature_inventory_sha256, validate_packed_dir  # noqa: E402
from oceanpath.eval.core import compute_calibration_intercept_slope  # noqa: E402

DEFAULT_OUTPUT_ROOT = runner.DEFAULT_OUTPUT_ROOT
ATOL = 1e-12


class VerificationError(RuntimeError):
    """A sealed artifact or reported value cannot be independently reproduced."""


def _artifact(path: Path) -> dict[str, Any]:
    return lineage.artifact_identity(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VerificationError(f"Expected JSON object: {path}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"


def _publish_json_once(path: Path, value: Any) -> None:
    text = _canonical_json(value)
    if path.is_file():
        existing = _read_json(path)
        # Timestamps are intentionally not part of the reproducibility check.
        comparable_existing = {key: item for key, item in existing.items() if key != "created_utc"}
        comparable_new = {key: item for key, item in value.items() if key != "created_utc"}
        if comparable_existing != comparable_new:
            raise FileExistsError(f"Refusing to replace non-identical verification: {path}")
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _require_equal(observed: Any, expected: Any, description: str) -> None:
    if observed != expected:
        raise VerificationError(
            f"{description} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _require_close(observed: Any, expected: Any, description: str) -> None:
    if observed is None or expected is None:
        if observed is not expected:
            raise VerificationError(
                f"{description} mismatch: observed={observed!r}, expected={expected!r}"
            )
        return
    if not np.isclose(float(observed), float(expected), rtol=0.0, atol=ATOL):
        raise VerificationError(
            f"{description} mismatch: observed={observed!r}, expected={expected!r}"
        )


def _expected_manifest(label_source: Path, pack_dir: Path) -> pd.DataFrame:
    """Reconstruct only the non-molecular fields used before inference."""

    master = pd.read_csv(label_source)
    rows = master.loc[
        master["cohort"].eq("Orion")
        & master["include"].astype(str).str.lower().eq("yes")
        & master["available"].astype(str).str.lower().eq("yes")
        & master["used_kras"].astype(str).str.lower().eq("yes")
        & master["qc_slides"].astype(str).str.lower().eq("pass")
    ].sort_values("output_id", kind="stable")
    index = pd.read_parquet(pack_dir / "index.parquet", columns=["slide_id", "n_patches"])
    patch_count = dict(
        zip(index["slide_id"].astype(str), index["n_patches"].astype(int), strict=True)
    )
    return pd.DataFrame(
        {
            "slide_id": rows["output_id"].astype(str),
            "patient_id": rows["patient_uid"].astype(str),
            "cohort": rows["cohort"].astype(str),
            "subcohort": rows["subcohort"].astype(str),
            "specimen_role": rows["specimen_role"].astype(str),
            "mpp": pd.to_numeric(rows["mpp"], errors="raise"),
            "mpp_source": rows["mpp_source"].astype(str),
            "patch_count": rows["output_id"].astype(str).map(patch_count).astype(int),
            "exclude_neoadjuvant": rows["qc_flags"].fillna("").astype(str).eq("treatment"),
            "exclude_ambiguous_crc15": rows["patient_uid"].astype(str).eq("ORION:C15"),
        }
    ).reset_index(drop=True)


def _verify_contract(root: Path, label_source: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = root / "inputs/orion_primary.csv"
    contract_path = root / "inputs/preoutcome_contract.json"
    manifest = pd.read_csv(manifest_path)
    contract = _read_json(contract_path)
    _require_equal(contract.get("preoutcome"), True, "contract preoutcome flag")
    _require_equal(contract.get("label_join_permitted"), False, "contract label-join flag")
    leaked = runner.FORBIDDEN_PREOUTCOME_COLUMNS & set(manifest.columns)
    if leaked:
        raise VerificationError(f"Pre-outcome manifest leaks columns: {sorted(leaked)}")
    pack_dir = Path(contract["feature_store"]["path"])
    feature_dir = Path(contract["feature_store"]["source_dir"])
    expected = _expected_manifest(label_source, pack_dir)
    assert_frame_equal(manifest, expected, check_dtype=False, check_exact=True)

    live_inventory = feature_inventory_sha256(feature_dir)
    meta = validate_packed_dir(pack_dir, verify_source=live_inventory)
    _require_equal(meta.n_slides, runner.EXPECTED_PACK_SLIDES, "packed slide count")
    _require_equal(meta.feat_dim, runner.EXPECTED_FEATURE_DIM, "packed feature dimension")
    _require_equal(
        contract["feature_store"]["source_inventory_sha256"],
        live_inventory,
        "packed source inventory",
    )
    _require_equal(contract["feature_store"]["meta"], _artifact(pack_dir / "meta.json"), "meta")
    _require_equal(
        contract["feature_store"]["index"],
        _artifact(pack_dir / "index.parquet"),
        "packed index",
    )
    _require_equal(
        contract["feature_store"]["features"],
        _artifact(pack_dir / "features.bin"),
        "packed feature payload",
    )
    _require_equal(
        contract["feature_store"]["coords"],
        _artifact(pack_dir / "coords.bin"),
        "packed coordinate payload",
    )
    _require_equal(
        contract["label_source_identity_not_opened_by_score"],
        _artifact(label_source),
        "label-source identity",
    )
    _require_equal(
        contract["inference"]["environment"],
        runner.inference_environment(),
        "inference implementation/runtime identity",
    )
    _require_equal(
        contract["statistics"]["analysis_environment"],
        runner.analysis_environment(),
        "analysis implementation/runtime identity",
    )

    aim1 = contract["aim1_outer15"]["models_by_seed"]
    _require_equal(sorted(map(int, aim1)), list(runner.SEEDS), "Aim1 seed roster")
    checkpoint_count = 0
    for seed in runner.SEEDS:
        entries = aim1[str(seed)]
        _require_equal([int(entry["fold"]) for entry in entries], list(runner.FOLDS), "Aim1 folds")
        for entry in entries:
            _require_equal(
                entry["checkpoint"],
                _artifact(Path(entry["checkpoint"]["path"])),
                "Aim1 checkpoint identity",
            )
            _require_equal(
                entry["fold_completion"],
                _artifact(Path(entry["fold_completion"]["path"])),
                "Aim1 fold receipt identity",
            )
            for key in (
                "fold_metrics",
                "training_identity",
                "training_completion",
                "resolved_config",
            ):
                _require_equal(
                    entry[key],
                    _artifact(Path(entry[key]["path"])),
                    f"Aim1 {seed}/{entry['fold']} {key}",
                )
            checkpoint_count += 1
    _require_equal(checkpoint_count, 15, "Aim1 checkpoint count")

    loco = contract["family_loco"]
    _require_equal(loco["complete_eight_model_matrix"], False, "complete LOCO matrix flag")
    _require_equal(
        loco["available_held_out_families"], list(runner.FAMILY_LOCO_TARGETS), "LOCO roster"
    )
    loco_count = 0
    for target in runner.FAMILY_LOCO_TARGETS:
        entry = loco["models"][target]
        for seed in runner.SEEDS:
            model = entry[str(seed)]
            _require_equal(model["optimizer_steps"], runner.LOCO_STEP_BUDGET, "LOCO steps")
            for key in (
                "checkpoint",
                "resolved_config",
                "fit_summary",
                "run_request",
                "source_manifest",
                "splits",
            ):
                _require_equal(
                    model[key], _artifact(Path(model[key]["path"])), f"{target} {seed} {key}"
                )
            loco_count += 1
        calibration = entry["calibrator"]
        _require_equal(
            calibration["artifact"],
            _artifact(Path(calibration["artifact"]["path"])),
            f"{target} calibrator",
        )
        _require_equal(
            sorted(map(int, calibration["source_cv_inputs"])),
            list(runner.SEEDS),
            f"{target} calibrator source-CV roster",
        )
        for seed in runner.SEEDS:
            identity = calibration["source_cv_inputs"][str(seed)]
            _require_equal(
                identity,
                _artifact(Path(identity["path"])),
                f"{target} calibrator source-CV seed {seed}",
            )
    _require_equal(loco_count, 12, "LOCO checkpoint count")
    return manifest, contract


def _expected_score_spec(
    contract: dict[str, Any], path: Path
) -> tuple[int, set[int], set[int], list[dict[str, Any]]]:
    stem = path.stem
    seed = int(stem.split("_seed", maxsplit=1)[1].split("_", maxsplit=1)[0])
    if stem.startswith("aim1_"):
        entries = contract["aim1_outer15"]["models_by_seed"][str(seed)]
        return runner.EXPECTED_SLIDES * 5, {seed}, set(runner.FOLDS), [
            {"seed": seed, "fold": int(entry["fold"]), "checkpoint": entry["checkpoint"]}
            for entry in entries
        ]
    target = stem.removeprefix("loco_").split("_seed", maxsplit=1)[0]
    canonical = next(item for item in runner.FAMILY_LOCO_TARGETS if item.lower() == target)
    entry = contract["family_loco"]["models"][canonical][str(seed)]
    return runner.EXPECTED_SLIDES, {seed}, {0}, [
        {"seed": seed, "fold": 0, "checkpoint": entry["checkpoint"]}
    ]


def _verify_scores(
    root: Path, manifest: pd.DataFrame, contract: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    manifest_identity = _artifact(root / "inputs/orion_primary.csv")
    contract_identity = _artifact(root / "inputs/preoutcome_contract.json")
    slide_ids = set(manifest["slide_id"].astype(str))
    frames: dict[str, pd.DataFrame] = {}
    for path in runner.expected_score_paths(root):
        receipt_path = path.with_suffix(".receipt.json")
        frame = pd.read_parquet(path)
        receipt = _read_json(receipt_path)
        n_rows, seeds, folds, checkpoints = _expected_score_spec(contract, path)
        _require_equal(
            list(frame.columns), ["slide_id", "seed", "fold", "logit"], f"{path.name} schema"
        )
        _require_equal(len(frame), n_rows, f"{path.name} row count")
        _require_equal(set(frame["slide_id"].astype(str)), slide_ids, f"{path.name} slides")
        _require_equal(set(frame["seed"].astype(int)), seeds, f"{path.name} seeds")
        _require_equal(set(frame["fold"].astype(int)), folds, f"{path.name} folds")
        if frame.duplicated(["slide_id", "seed", "fold"]).any():
            raise VerificationError(f"Duplicate model/slide score rows: {path}")
        for _, model_rows in frame.groupby(["seed", "fold"]):
            _require_equal(
                set(model_rows["slide_id"].astype(str)),
                slide_ids,
                f"{path.name} per-model slide roster",
            )
            _require_equal(
                len(model_rows), runner.EXPECTED_SLIDES, f"{path.name} per-model row count"
            )
        if not np.isfinite(frame["logit"].to_numpy()).all():
            raise VerificationError(f"Non-finite native logits: {path}")
        leaked = runner.FORBIDDEN_PREOUTCOME_COLUMNS & set(frame.columns)
        if leaked:
            raise VerificationError(f"Outcome columns in pre-outcome score file: {path}")
        _require_equal(receipt["preoutcome"], True, f"{path.name} preoutcome flag")
        _require_equal(
            receipt["contains_target_outcomes"], False, f"{path.name} outcome flag"
        )
        _require_equal(receipt["artifact"], _artifact(path), f"{path.name} artifact")
        _require_equal(receipt["inputs"]["manifest"], manifest_identity, "score manifest")
        _require_equal(receipt["inputs"]["contract"], contract_identity, "score contract")
        _require_equal(receipt["inputs"]["checkpoints"], checkpoints, "score checkpoints")
        _require_equal(
            receipt["inputs"]["inference_environment"],
            contract["inference"]["environment"],
            "score inference environment",
        )
        frames[path.name] = frame

    seal_path = root / "inference_seal.json"
    seal = _read_json(seal_path)
    _require_equal(seal["status"], "sealed_before_outcome_join", "inference seal status")
    _require_equal(seal["score_artifact_count"], 15, "inference score count")
    _require_equal(seal["target_outcomes_present"], False, "seal outcome flag")
    _require_equal(seal["confirmatory_cpht_model_present"], False, "confirmatory flag")
    _require_equal(seal["complete_eight_model_matrix"], False, "matrix completeness")
    _require_equal(seal["manifest"], manifest_identity, "sealed manifest")
    _require_equal(seal["contract"], contract_identity, "sealed contract")
    _require_equal(
        seal["inference_environment"],
        contract["inference"]["environment"],
        "sealed inference environment",
    )
    records = seal["score_artifacts"]
    _require_equal(len(records), 15, "sealed artifact roster")
    for path, record in zip(runner.expected_score_paths(root), records, strict=True):
        _require_equal(record["score"], _artifact(path), "sealed prediction")
        _require_equal(record["receipt"], _artifact(path.with_suffix(".receipt.json")), "seal")
    return frames


def _labels(label_source: Path) -> pd.DataFrame:
    master = pd.read_csv(label_source)
    rows = master.loc[master["cohort"].eq("Orion"), ["patient_uid", "kras"]].copy()
    rows["label"] = rows["kras"].eq("mutant").astype(int)
    if not rows.groupby("patient_uid")["label"].nunique().eq(1).all():
        raise VerificationError("Patient-level KRAS labels are inconsistent")
    return (
        rows.drop_duplicates("patient_uid")[["patient_uid", "label"]]
        .rename(columns={"patient_uid": "patient_id"})
        .sort_values("patient_id", kind="stable")
        .reset_index(drop=True)
    )


def _aggregate(
    frame: pd.DataFrame, manifest: pd.DataFrame, labels: pd.DataFrame, expected_models: int
) -> pd.DataFrame:
    counts = frame.groupby("slide_id").size()
    if len(counts) != runner.EXPECTED_SLIDES or not counts.eq(expected_models).all():
        raise VerificationError("Model count per slide is incorrect")
    slides = (
        frame.groupby("slide_id", as_index=False)["logit"]
        .mean()
        .rename(columns={"logit": "slide_logit"})
        .merge(manifest, on="slide_id", validate="one_to_one")
    )
    patients = (
        slides.sort_values("slide_id", kind="stable")
        .groupby("patient_id", as_index=False, sort=True)
        .agg(
            mean_logit=("slide_logit", "mean"),
            n_slides=("slide_id", "count"),
            exclude_neoadjuvant=("exclude_neoadjuvant", "max"),
            exclude_ambiguous_crc15=("exclude_ambiguous_crc15", "max"),
        )
        .merge(labels, on="patient_id", validate="one_to_one")
    )
    c33 = patients.loc[patients["patient_id"].eq("ORION:C33"), "n_slides"]
    _require_equal(c33.tolist(), [2], "C33 slide aggregation")
    return patients


def _bootstrap_indices(labels: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(runner.BOOTSTRAP_SEED)
    zero = np.flatnonzero(labels == 0)
    one = np.flatnonzero(labels == 1)
    return np.concatenate(
        [
            rng.choice(zero, size=(runner.N_BOOTSTRAP, len(zero)), replace=True),
            rng.choice(one, size=(runner.N_BOOTSTRAP, len(one)), replace=True),
        ],
        axis=1,
    )


def _rank_samples(labels: np.ndarray, score: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    draw_y = labels[indices]
    draw_score = score[indices]
    n_zero = int((labels == 0).sum())
    negative = draw_score[:, :n_zero]
    positive = draw_score[:, n_zero:]
    differences = positive[:, :, None] - negative[:, None, :]
    auc = (
        (differences > 0).sum((1, 2))
        + 0.5 * (differences == 0).sum((1, 2))
    ) / ((labels == 0).sum() * (labels == 1).sum())
    order = np.argsort(-draw_score, axis=1, kind="stable")
    ranked_y = np.take_along_axis(draw_y, order, axis=1)
    ranked_score = np.take_along_axis(draw_score, order, axis=1)
    cumulative = np.cumsum(ranked_y, axis=1)
    tie_end = np.ones_like(ranked_y, dtype=bool)
    tie_end[:, :-1] = ranked_score[:, :-1] != ranked_score[:, 1:]
    ap = np.zeros(len(indices), dtype=float)
    previous_tp = np.zeros(len(indices), dtype=float)
    n_positive = ranked_y.sum(axis=1)
    for column in range(ranked_y.shape[1]):
        end = tie_end[:, column]
        current_tp = cumulative[:, column].astype(float)
        increment = current_tp - previous_tp
        precision = current_tp / float(column + 1)
        ap += np.where(end, increment / n_positive * precision, 0.0)
        previous_tp = np.where(end, current_tp, previous_tp)
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auroc_samples": auc,
        "auprc": float(average_precision_score(labels, score)),
        "auprc_samples": ap,
    }


def _calibrated_probability(eta: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    z = float(calibration["a"]) + float(calibration["b"]) * eta
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _verify_metric_block(
    reported: dict[str, Any],
    labels: np.ndarray,
    score: np.ndarray,
    indices: np.ndarray,
    description: str,
    probability: np.ndarray | None,
) -> np.ndarray:
    rank = _rank_samples(labels, score, indices)
    expected_auc_ci = np.percentile(rank["auroc_samples"], [2.5, 97.5])
    expected_ap_ci = np.percentile(rank["auprc_samples"], [2.5, 97.5])
    _require_close(reported["auroc"], rank["auroc"], f"{description} AUROC")
    for index, endpoint in enumerate(("low", "high")):
        _require_close(
            reported["auroc_ci95"][index],
            expected_auc_ci[index],
            f"{description} AUROC CI {endpoint}",
        )
        _require_close(
            reported["auprc_ci95"][index],
            expected_ap_ci[index],
            f"{description} AUPRC CI {endpoint}",
        )
    _require_close(reported["auprc"], rank["auprc"], f"{description} AUPRC")
    _require_equal(reported["n"], len(labels), f"{description} n")
    _require_equal(reported["n_mutant"], int(labels.sum()), f"{description} mutants")

    if probability is None:
        if "source_calibrated" in reported:
            raise VerificationError(f"Rank-only Aim1 block contains calibration: {description}")
        return rank["auroc_samples"]
    calibrated = reported["source_calibrated"]
    draw_y = labels[indices]
    draw_p = probability[indices]
    brier = np.mean((draw_p - draw_y) ** 2, axis=1)
    loss = -np.mean(draw_y * np.log(draw_p) + (1 - draw_y) * np.log(1 - draw_p), axis=1)
    _require_close(
        calibrated["brier"], brier_score_loss(labels, probability), f"{description} Brier"
    )
    _require_close(
        calibrated["log_loss"],
        log_loss(labels, probability, labels=[0, 1]),
        f"{description} log loss",
    )
    for index in (0, 1):
        _require_close(
            calibrated["brier_ci95"][index],
            np.percentile(brier, [2.5, 97.5])[index],
            f"{description} Brier CI",
        )
        _require_close(
            calibrated["log_loss_ci95"][index],
            np.percentile(loss, [2.5, 97.5])[index],
            f"{description} log-loss CI",
        )
    calibration = compute_calibration_intercept_slope(labels, probability)
    for key in (
        "calibration_intercept",
        "calibration_slope",
        "slope_model_intercept",
        "slope_converged",
    ):
        if isinstance(calibration[key], (bool, np.bool_)):
            _require_equal(calibrated[key], bool(calibration[key]), f"{description} {key}")
        else:
            expected = calibration[key]
            expected = float(expected) if np.isfinite(expected) else None
            _require_close(calibrated[key], expected, f"{description} {key}")
    return rank["auroc_samples"]


def _verify_analysis(
    root: Path,
    manifest: pd.DataFrame,
    contract: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    label_source: Path,
) -> None:
    results_path = root / "analysis/results.json"
    patient_path = root / "analysis/orion_patient_scores.parquet"
    table_path = root / "analysis/results.csv"
    receipt_path = root / "analysis/receipt.json"
    results = _read_json(results_path)
    reported_patients = pd.read_parquet(patient_path)
    labels = _labels(label_source)

    scorers = ["aim1_outer15"] + [
        f"loco_heldout_{target.lower()}" for target in runner.FAMILY_LOCO_TARGETS
    ]
    source_frames: dict[str, pd.DataFrame] = {
        "aim1_outer15": pd.concat(
            [frames[f"aim1_seed{seed}_orion.parquet"] for seed in runner.SEEDS],
            ignore_index=True,
        )
    }
    for target in runner.FAMILY_LOCO_TARGETS:
        source_frames[f"loco_heldout_{target.lower()}"] = pd.concat(
            [
                frames[f"loco_{target.lower()}_seed{seed}_orion.parquet"]
                for seed in runner.SEEDS
            ],
            ignore_index=True,
        )

    patients: dict[str, pd.DataFrame] = {}
    expected_patient_rows: list[pd.DataFrame] = []
    for scorer in scorers:
        frame = _aggregate(
            source_frames[scorer], manifest, labels, 15 if scorer == "aim1_outer15" else 3
        )
        frame["prob_raw"] = 1.0 / (
            1.0 + np.exp(-np.clip(frame["mean_logit"].to_numpy(), -500, 500))
        )
        frame["scorer"] = scorer
        if scorer == "aim1_outer15":
            frame["prob_source_calibrated"] = np.nan
        else:
            target = scorer.removeprefix("loco_heldout_")
            canonical = next(
                item for item in runner.FAMILY_LOCO_TARGETS if item.lower() == target
            )
            calibration = contract["family_loco"]["models"][canonical]["calibrator"]
            frame["prob_source_calibrated"] = _calibrated_probability(
                frame["mean_logit"].to_numpy(), calibration
            )
        patients[scorer] = frame
        expected_patient_rows.append(frame)

    expected_patients = pd.concat(expected_patient_rows, ignore_index=True)[
        reported_patients.columns
    ]
    assert_frame_equal(
        reported_patients.reset_index(drop=True),
        expected_patients.reset_index(drop=True),
        check_dtype=False,
        rtol=0.0,
        atol=ATOL,
    )

    masks = {
        "all_40": np.ones(runner.EXPECTED_PATIENTS, dtype=bool),
        "exclude_neoadjuvant": ~patients["aim1_outer15"][
            "exclude_neoadjuvant"
        ].to_numpy(bool),
        "exclude_ambiguous_crc15": ~patients["aim1_outer15"][
            "exclude_ambiguous_crc15"
        ].to_numpy(bool),
    }
    for population, mask in masks.items():
        y = patients["aim1_outer15"].loc[mask, "label"].to_numpy(int)
        indices = _bootstrap_indices(y)
        block = results["populations"][population]
        _require_equal(block["n"], len(y), f"{population} n")
        _require_equal(block["n_mutant"], int(y.sum()), f"{population} mutants")
        expected_hash = runner._sha256_bytes(indices.astype("<i8").tobytes())  # noqa: SLF001
        _require_equal(
            block["bootstrap_indices_sha256"], expected_hash, f"{population} bootstrap hash"
        )
        auc_samples: dict[str, np.ndarray] = {}
        for scorer in scorers:
            patient = patients[scorer].loc[mask].reset_index(drop=True)
            probability = (
                None
                if scorer == "aim1_outer15"
                else patient["prob_source_calibrated"].to_numpy(float)
            )
            auc_samples[scorer] = _verify_metric_block(
                block["metrics"][scorer],
                y,
                patient["mean_logit"].to_numpy(float),
                indices,
                f"{population}/{scorer}",
                probability,
            )
        for left_index, left in enumerate(scorers):
            for right in scorers[left_index + 1 :]:
                name = f"{right}_minus_{left}"
                reported = block["paired_auroc_contrasts"][name]
                expected_delta = (
                    block["metrics"][right]["auroc"] - block["metrics"][left]["auroc"]
                )
                distribution = auc_samples[right] - auc_samples[left]
                _require_close(
                    reported["delta_auroc"], expected_delta, f"{population}/{name} point"
                )
                for endpoint in (0, 1):
                    _require_close(
                        reported["delta_auroc_ci95"][endpoint],
                        np.percentile(distribution, [2.5, 97.5])[endpoint],
                        f"{population}/{name} CI",
                    )

    compact = pd.read_csv(table_path)
    _require_equal(len(compact), 15, "compact result row count")
    receipt = _read_json(receipt_path)
    _require_equal(
        receipt["status"],
        "analysis_complete_pending_independent_verification",
        "analysis receipt status",
    )
    _require_equal(receipt["inference_seal"], _artifact(root / "inference_seal.json"), "seal")
    _require_equal(receipt["label_source_joined_after_seal"], _artifact(label_source), "labels")
    _require_equal(receipt["outputs"]["patient_scores"], _artifact(patient_path), "patients")
    _require_equal(receipt["outputs"]["results"], _artifact(results_path), "results")
    _require_equal(receipt["outputs"]["compact_table"], _artifact(table_path), "table")


def verify(root: Path, label_source: Path) -> dict[str, Any]:
    manifest, contract = _verify_contract(root, label_source)
    frames = _verify_scores(root, manifest, contract)
    _verify_analysis(root, manifest, contract, frames, label_source)
    return {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "experiment": "E2-CPHT existing-model Orion sensitivities",
        "checks": {
            "label_blind_manifest_exact": True,
            "packed_store_current_and_structurally_valid": True,
            "aim1_selected_checkpoint_hashes": "15/15",
            "family_loco_refit_hashes_and_contracts": "12/12",
            "preoutcome_prediction_artifacts": "15/15",
            "native_logits_finite_unique_and_complete": True,
            "c33_two_slide_native_logit_aggregation": True,
            "patient_scores_independently_reproduced": "200/200",
            "stratified_bootstrap_metrics_independently_reproduced": True,
            "paired_model_contrasts_independently_reproduced": True,
            "confirmatory_cpht_not_misrepresented": True,
            "complete_eight_model_matrix_not_misrepresented": True,
        },
        "verified_inputs": {
            "manifest": _artifact(root / "inputs/orion_primary.csv"),
            "contract": _artifact(root / "inputs/preoutcome_contract.json"),
            "inference_seal": _artifact(root / "inference_seal.json"),
            "analysis_receipt": _artifact(root / "analysis/receipt.json"),
        },
    }


def cmd_verify(args: argparse.Namespace) -> None:
    result = verify(args.output_root, args.label_source)
    verification_path = args.output_root / "verification.json"
    receipt_path = args.output_root / "receipt.json"
    if args.read_only:
        print(_canonical_json(result), end="")
        return
    if receipt_path.exists() and not verification_path.is_file():
        raise VerificationError("Final receipt exists without verification artifact")
    _publish_json_once(verification_path, result)
    receipt = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "existing-model Orion sensitivities only",
        "confirmatory_cpht_completed": False,
        "complete_eight_model_matrix": False,
        "artifacts": {
            "preoutcome_contract": _artifact(args.output_root / "inputs/preoutcome_contract.json"),
            "inference_seal": _artifact(args.output_root / "inference_seal.json"),
            "analysis_receipt": _artifact(args.output_root / "analysis/receipt.json"),
            "verification": _artifact(verification_path),
        },
        "next_required": (
            "train and seal the new three-seed 6,060-step all-conventional CPHT refit; "
            "train four sibling-stratum LOCO directions for the complete eight-model matrix"
        ),
    }
    _publish_json_once(receipt_path, receipt)
    print(f"PASS: independently verified E2-CPHT sensitivity run at {args.output_root}")
    print(f"final receipt: {receipt_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-source", type=Path, default=runner.LABEL_SOURCE)
    parser.add_argument("--read-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        cmd_verify(args)
    except (
        VerificationError,
        FileNotFoundError,
        FileExistsError,
        AssertionError,
        KeyError,
        ValueError,
    ) as exc:
        print(f"E2-CPHT VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
