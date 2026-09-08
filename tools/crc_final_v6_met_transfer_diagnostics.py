#!/usr/bin/env python3
"""Post-hoc diagnostics of primary-to-metastatic KRAS-score transfer.

The analysis adds the user-supplied ``source_type`` field from ``crc_final_v6`` to
already frozen patient scores. It does not refit the MIL models. Its purpose
is to describe post-outcome associations between frozen-score rankings and
metastatic site, source-type composition, recorded case mix, and audit-only
averaged-feature decodability. It does not distinguish causal explanations.

The script is deliberately fail-closed.  It verifies that v6 is exactly v5
plus one field, joins score rosters at slide/patient-role grain, uses disjoint
role populations for contrasts, and publishes a hash receipt for every input
and output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import stat
import tempfile
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:  # noqa: E402
    from tools.final_v12_bundle_receipt import (  # type: ignore[no-redef]
        verify_published_receipt as verify_final_v12_receipt,
    )
except ModuleNotFoundError:  # direct ``python tools/...`` execution
    from final_v12_bundle_receipt import (  # type: ignore[no-redef]
        verify_published_receipt as verify_final_v12_receipt,
    )
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_ROOT = Path("/mnt/d/YC.Liu/manifests/colon")
V5 = MANIFEST_ROOT / "crc_final_v5.csv"
V6 = MANIFEST_ROOT / "crc_final_v6.csv"
V6_XLSX = MANIFEST_ROOT / "crc_final_v6.xlsx"

LOCO_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/final_v9_mil_5seed_expansion_v1_20260823/aim2_loco"
)
LOCO_PRIMARY_SCORES = LOCO_ROOT / "analysis/primary_patient_scores_five_seed.parquet"
LOCO_MET_SCORES = LOCO_ROOT / "analysis/e2met_patient_scores_five_seed.parquet"
LOCO_RESULTS_RECEIPT = LOCO_ROOT / "analysis/results_five_seed.receipt.json"
LOCO_CONTRACT = LOCO_ROOT / "contract.json"
LOCO_INFERENCE_SEAL = LOCO_ROOT / "inference/inference_seal.json"

CONTINUATION_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_tcga_surgen_two_encoder_v1_20260824/downstream_v2/continuation_v3"
)
TWO_ENCODER_SCORES = CONTINUATION_ROOT / "analysis/patient_native_logits.parquet"
TWO_ENCODER_RESULTS_RECEIPT = CONTINUATION_ROOT / "analysis/analysis_completion_receipt.json"
FINAL_V12_RECEIPT = REPO_ROOT / "reports/final_v12/report_bundle_receipt.json"
FINAL_V12_MANIFEST = REPO_ROOT / "reports/final_v12/source_manifest.json"
FINAL_V12_VERIFIER = REPO_ROOT / "tools/final_v12_bundle_receipt.py"

SLIDE_MEANS = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides/slide_means_univ1.npz"
)
COORDINATES = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides/coordinates_univ1.parquet"
)

ROSTERS = {
    ("RIH", "primary"): LOCO_ROOT / "inputs/label_blind/primary/family_rih.csv",
    ("RIH", "metastatic"): LOCO_ROOT / "inputs/label_blind/e2met/rih_m.csv",
    ("SR1482", "primary"): LOCO_ROOT / "inputs/label_blind/primary/family_surgen.csv",
    ("SR1482", "metastatic"): LOCO_ROOT / "inputs/label_blind/e2met/sr1482_m.csv",
}

DEFAULT_OUTPUT = REPO_ROOT / "reports/reruns/crc_final_v6_met_transfer_diagnostics_20260827"
DEFAULT_BOOTSTRAPS = 10_000
DEFAULT_CV_REPEATS = 20
SECONDARY_BOOTSTRAPS = 2_000
RNG_SEED = 20260827
MIN_CELL_PER_CLASS = 10
PROPENSITY_CLIP = (0.02, 0.98)
SOURCE_LEVELS = ("biopsy", "resection", "unknown")
SUBTYPE_LEVELS = ("G12D", "G12V", "G13D", "other_or_unknown_mutant")
CASE_MIX_BLOCKS = {
    "molecular": ("msi_dmmr", "braf"),
    "clinical_molecular": ("age", "sex", "msi_dmmr", "braf"),
    "joint_source_clinical_molecular": (
        "source_type",
        "age",
        "sex",
        "msi_dmmr",
        "braf",
    ),
}
EXPECTED_OUTPUTS = {
    "analysis_table.parquet",
    "case_mix_standardization.csv",
    "contrasts.csv",
    "decomposition.csv",
    "met_transfer_diagnostics.pdf",
    "met_transfer_diagnostics.png",
    "paired_role_scores.csv",
    "pairwise_source_cells.csv",
    "patient_embeddings.parquet",
    "performance_strata.csv",
    "population.csv",
    "report.md",
    "results.json",
    "score_structure.csv",
    "site_source_standardization.csv",
    "source_alignment.csv",
    "structural_cv.csv",
}
EXPECTED_INPUTS = {
    "analysis_script",
    "crc_final_v5_csv",
    "crc_final_v6_csv",
    "crc_final_v6_xlsx",
    "final_v12_receipt",
    "final_v12_source_manifest",
    "final_v12_verifier",
    "loco_contract",
    "loco_inference_seal",
    "loco_metastatic_scores",
    "loco_primary_scores",
    "loco_results_receipt",
    "roster_rih_metastatic",
    "roster_rih_primary",
    "roster_sr1482_metastatic",
    "roster_sr1482_primary",
    "two_encoder_patient_scores",
    "two_encoder_results_receipt",
    "univ1_coordinates",
    "univ1_slide_means",
}
EXPECTED_CHECKS = {
    "audit_only_embedding_provenance_declared",
    "exact_output_roster",
    "exact_target_roster_joins",
    "final_v12_parent_chain_authenticated",
    "frozen_models_not_refit",
    "inputs_unchanged_during_analysis",
    "patient_level_bootstrap",
    "patient_role_metadata_consistency",
    "post_outcome_noncausal_status_declared",
    "role_overlap_removed_from_contrasts",
    "score_artifacts_match_parent_receipts",
    "score_label_identity",
    "source_case_mix_overlap_gates",
    "strict_json_no_nonfinite",
    "unknown_source_retained_explicitly",
    "v6_csv_xlsx_cell_equality",
    "v6_is_exact_v5_plus_source_type",
}


@dataclass(frozen=True)
class View:
    lineage: str
    encoder: str
    cohort: str
    comparison_status: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise ValueError(f"Symlink is not allowed in artifact path: {path}")
    try:
        before = path.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(path) from exc
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Artifact is not a regular file: {path}")
    digest = _sha256(path)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"Artifact changed while hashing: {path}")
    return {
        "path": str(path),
        "size_bytes": before.st_size,
        "sha256": digest,
    }


def _stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _sanitize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_sanitize_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON token {value!r} in {path}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_keys,
    )

    def reject_nonfinite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite numeric JSON value in {path}")
        if isinstance(value, dict):
            for item in value.values():
                reject_nonfinite(item)
        elif isinstance(value, list):
            for item in value:
                reject_nonfinite(item)

    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    reject_nonfinite(payload)
    return payload


def _assert_receipt_record(record: dict[str, Any], expected_path: Path) -> None:
    expected = _artifact(expected_path)
    if record != expected:
        raise ValueError(
            f"Parent receipt does not authenticate {expected_path}: "
            f"expected={expected}, observed={record}"
        )


def validate_parent_score_receipts() -> dict[str, Any]:
    # Rebuild FINAL-v12 from sealed FINAL-v11, rehash all 139 governed sources,
    # and byte-compare the published receipt before trusting its source graph.
    final_receipt = verify_final_v12_receipt()
    if final_receipt.get("status") != "SEALED_FINAL_V12_INTEGRATED_PAPER_SELECTION":
        raise ValueError("FINAL-v12 parent is not sealed")
    manifest_record = final_receipt.get("source_manifest", {})
    manifest_artifact = _artifact(FINAL_V12_MANIFEST)
    if (
        manifest_record.get("sha256") != manifest_artifact["sha256"]
        or manifest_record.get("size_bytes") != manifest_artifact["size_bytes"]
    ):
        raise ValueError("FINAL-v12 receipt does not authenticate its source manifest")
    final_manifest = _strict_json_load(FINAL_V12_MANIFEST)
    if (
        final_manifest.get("bundle") != "final_v12"
        or len(final_manifest.get("artifacts", [])) != 139
    ):
        raise ValueError("Unexpected FINAL-v12 source manifest topology")
    manifest_by_path = {str(record["path"]): record for record in final_manifest["artifacts"]}
    governed_ids = {
        "aim2-loco-five-seed-report-receipt": LOCO_RESULTS_RECEIPT,
        "aim2-loco-five-seed-contract": LOCO_CONTRACT,
        "aim2-loco-five-seed-inference-seal": LOCO_INFERENCE_SEAL,
        "aim2-loco-five-seed-primary-patients": LOCO_PRIMARY_SCORES,
        "aim2-loco-five-seed-metastatic-patients": LOCO_MET_SCORES,
        "aim1-tcga-surgen-two-encoder-downstream-v3-analysis-completion": (
            TWO_ENCODER_RESULTS_RECEIPT
        ),
        "aim1-tcga-surgen-two-encoder-downstream-v3-patient-native-logits": (TWO_ENCODER_SCORES),
    }
    governed_by_id = {
        str(record["id"]): record for record in final_receipt["authoritative_sources"]
    }
    for source_id, path in governed_ids.items():
        id_record = governed_by_id.get(source_id)
        if id_record is None or str(id_record.get("path")) != str(path):
            raise ValueError(f"FINAL-v12 authoritative source ID drift: {source_id}")
        record = manifest_by_path.get(str(path))
        if record is None:
            raise ValueError(f"FINAL-v12 manifest does not contain {path}")
        actual = _artifact(path)
        for governed_record in (id_record, record):
            if (
                governed_record.get("sha256") != actual["sha256"]
                or governed_record.get("size_bytes") != actual["size_bytes"]
            ):
                raise ValueError(f"FINAL-v12 source identity mismatch for {source_id}")
        if id_record != record:
            raise ValueError(f"FINAL-v12 receipt/manifest record mismatch for {source_id}")

    loco = _strict_json_load(LOCO_RESULTS_RECEIPT)
    if loco.get("status") != "sealed_five_seed_results":
        raise ValueError("Unexpected five-seed LOCO parent receipt status")
    _assert_receipt_record(loco["artifacts"]["primary_patients"], LOCO_PRIMARY_SCORES)
    _assert_receipt_record(loco["artifacts"]["met_patients"], LOCO_MET_SCORES)
    _assert_receipt_record(loco["contract"], LOCO_CONTRACT)
    _assert_receipt_record(loco["inference_seal"], LOCO_INFERENCE_SEAL)
    if loco.get("bootstrap_unit") != "patient" or not loco.get(
        "model_seeds_are_not_inference_units"
    ):
        raise ValueError("Five-seed LOCO receipt does not preserve patient inference")
    if not loco.get("target_outcomes_opened_only_after_inference_seal"):
        raise ValueError("Five-seed LOCO receipt does not preserve inference-before-outcomes")
    inference_seal = _strict_json_load(LOCO_INFERENCE_SEAL)
    if (
        inference_seal.get("status") != "sealed_before_outcome_join"
        or inference_seal.get("target_outcomes_present") is not False
        or inference_seal.get("five_seed_loco_complete") is not True
    ):
        raise ValueError("Five-seed LOCO inference seal is not outcome-blind and complete")
    contract = _strict_json_load(LOCO_CONTRACT)
    roster_keys = {
        ("RIH", "primary"): "primary/family_rih",
        ("RIH", "metastatic"): "e2met/rih_m",
        ("SR1482", "primary"): "primary/family_surgen",
        ("SR1482", "metastatic"): "e2met/sr1482_m",
    }
    for key, contract_key in roster_keys.items():
        _assert_receipt_record(
            contract["label_blind_inputs"][contract_key]["sealed_manifest"],
            ROSTERS[key],
        )

    continuation = _strict_json_load(TWO_ENCODER_RESULTS_RECEIPT)
    if continuation.get("status") != "complete_and_verified":
        raise ValueError("Unexpected two-encoder parent receipt status")
    _assert_receipt_record(continuation["artifacts"]["patient_native_logits"], TWO_ENCODER_SCORES)
    if not continuation.get("outcomes_opened_after_inference_seal"):
        raise ValueError("Two-encoder receipt does not assert inference-before-outcomes")
    if any(
        continuation.get(field) != 0
        for field in ("target_calibrations", "target_model_selections", "target_refits")
    ):
        raise ValueError("Two-encoder receipt contains a target-label model operation")
    return {
        "five_seed_loco_status": loco["status"],
        "five_seed_loco_receipt_sha256": _sha256(LOCO_RESULTS_RECEIPT),
        "two_encoder_status": continuation["status"],
        "two_encoder_receipt_sha256": _sha256(TWO_ENCODER_RESULTS_RECEIPT),
        "final_v12_receipt_sha256": _sha256(FINAL_V12_RECEIPT),
        "final_v12_manifest_sha256": _sha256(FINAL_V12_MANIFEST),
        "score_artifacts_match_parent_receipts": True,
        "score_receipts_and_governed_artifacts_match_final_v12_manifest": True,
        "target_rosters_match_loco_contract": True,
    }


def _one_value(values: pd.Series, *, field: str, key: str) -> Any:
    present = values.dropna().unique()
    if len(present) != 1:
        raise ValueError(f"{key}: expected one {field}, observed {present.tolist()}")
    return present[0]


def validate_v6() -> tuple[pd.DataFrame, dict[str, Any]]:
    old = pd.read_csv(V5, dtype=str, keep_default_na=False)
    new = pd.read_csv(V6, dtype=str, keep_default_na=False)
    if len(new) != len(old) or list(new.columns[:-1]) != list(old.columns):
        raise ValueError("crc_final_v6 is not a one-column extension of crc_final_v5")
    if new.columns[-1] != "source_type":
        raise ValueError("source_type is not the final crc_final_v6 column")
    if not old.equals(new[old.columns]):
        raise ValueError("At least one inherited crc_final_v5 cell changed in v6")
    if new["output_id"].duplicated().any() or new["slide_uid"].duplicated().any():
        raise ValueError("crc_final_v6 slide identifiers are not unique")
    allowed = {"", *SOURCE_LEVELS}
    observed = set(new["source_type"].unique())
    if not observed <= allowed:
        raise ValueError(f"Unexpected source_type levels: {sorted(observed - allowed)}")

    # The XLSX is an independent user-provided representation.  We check cell
    # equality but never rewrite it.
    workbook = pd.read_excel(V6_XLSX, sheet_name="crc_final_v6", dtype=str).fillna("")
    workbook = workbook.astype(str)
    if list(workbook.columns) != list(new.columns) or not workbook.equals(new):
        raise ValueError("crc_final_v6 CSV/XLSX cell values differ")

    typed = new["source_type"].replace("", pd.NA)
    validation = {
        "rows": int(len(new)),
        "columns": int(len(new.columns)),
        "v5_cells_preserved_exactly": True,
        "v6_csv_xlsx_equal": True,
        "source_type_counts": {
            str(key): int(value) for key, value in typed.fillna("blank").value_counts().items()
        },
        "allowed_levels": sorted(allowed),
        "provenance_boundary": (
            "The supplied v6 files have no colocated builder, codebook, or provenance "
            "sidecar. This analysis verifies file identity and internal consistency; it "
            "does not reconstruct how each source_type value was adjudicated."
        ),
    }
    return pd.read_csv(V6, low_memory=False), validation


def _subtype(label: int, value: Any) -> str:
    if label == 0:
        return "wild_type"
    text = "" if pd.isna(value) else str(value)
    if text in {"G12D", "G12V", "G13D"}:
        return text
    return "other_or_unknown_mutant"


def build_patient_metadata(v6: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_rows: list[dict[str, Any]] = []
    slide_rows: list[pd.DataFrame] = []
    for (cohort, role), roster_path in ROSTERS.items():
        roster = pd.read_csv(roster_path)
        if cohort == "SR1482" and role == "primary":
            roster = roster[roster["subcohort"].eq("SR1482")].copy()
        if roster["slide_id"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate slide_id in {roster_path}")
        joined = roster.merge(
            v6,
            left_on="slide_id",
            right_on="output_id",
            how="left",
            validate="one_to_one",
            indicator=True,
            suffixes=("_roster", ""),
        )
        if not joined["_merge"].eq("both").all():
            missing = joined.loc[joined["_merge"].ne("both"), "slide_id"].tolist()
            raise ValueError(f"Unmatched v6 roster slides: {missing[:5]}")
        if not joined["patient_id"].astype(str).eq(joined["patient_uid"].astype(str)).all():
            raise ValueError(f"Patient identity mismatch in {roster_path}")
        if not joined["specimen_role"].eq(role).all():
            raise ValueError(f"Role mismatch in {roster_path}")
        expected_subcohort = "RIH-Colon" if cohort == "RIH" else "SR1482"
        if not joined["subcohort"].eq(expected_subcohort).all():
            raise ValueError(f"Subcohort mismatch in {roster_path}")
        joined["analysis_cohort"] = cohort
        joined["analysis_role"] = role
        slide_rows.append(joined.copy())

        for patient_id, group in joined.groupby("patient_id", sort=True):
            key = f"{cohort}/{role}/{patient_id}"
            label_text = _one_value(group["kras"], field="kras", key=key)
            if label_text not in {"mutant", "wild_type"}:
                raise ValueError(f"{key}: unknown KRAS label entered frozen roster")
            label = int(label_text == "mutant")
            source_type = _one_value(group["source_type"], field="source_type", key=key)
            if pd.isna(source_type) or source_type == "":
                source_type = "blank"
            site = (
                _one_value(group["metastatic_site_group"], field="site", key=key)
                if role == "metastatic"
                else "primary"
            )
            site_binary = (
                "liver" if site == "liver" else ("non_liver" if role == "metastatic" else "primary")
            )
            msi = _one_value(group["msi_dmmr"], field="msi_dmmr", key=key)
            braf = _one_value(group["braf"], field="braf", key=key)
            subtype_raw = (
                _one_value(
                    group.loc[group["kras_subvariant"].notna(), "kras_subvariant"],
                    field="kras_subvariant",
                    key=key,
                )
                if label and group["kras_subvariant"].notna().any()
                else np.nan
            )
            age = pd.to_numeric(group["age_at_diagnosis"], errors="coerce").dropna()
            sex = group["sex"].dropna().unique()
            patient_rows.append(
                {
                    "patient_id": str(patient_id),
                    "cohort": cohort,
                    "role": role,
                    "label": label,
                    "source_type": str(source_type),
                    "metastatic_site": str(site),
                    "site_binary": site_binary,
                    "msi_dmmr": str(msi),
                    "braf": str(braf),
                    "set_d": bool(msi == "MSS/pMMR" and braf == "wild_type"),
                    "kras_subvariant": subtype_raw,
                    "subtype_group": _subtype(label, subtype_raw),
                    "age": float(age.iloc[0]) if len(age) else np.nan,
                    "sex": str(sex[0]) if len(sex) == 1 else "missing",
                    "n_slides": int(len(group)),
                }
            )
    patients = pd.DataFrame(patient_rows)
    if patients.duplicated(["cohort", "role", "patient_id"]).any():
        raise ValueError("Patient metadata keys are not unique")
    return patients, pd.concat(slide_rows, ignore_index=True)


def _attach_metadata(scores: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    joined = scores.merge(
        metadata,
        on=["cohort", "role", "patient_id"],
        how="left",
        validate="many_to_one",
        suffixes=("_score", ""),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"].ne("both"), "patient_id"].tolist()
        raise ValueError(f"Scores missing patient-role metadata: {missing[:5]}")
    if not joined["label_score"].astype(int).eq(joined["label"].astype(int)).all():
        raise ValueError("Score and crc_final_v6 KRAS labels disagree")
    return joined.drop(columns=["_merge", "label_score"])


def build_score_table(metadata: pd.DataFrame) -> pd.DataFrame:
    primary = pd.read_parquet(LOCO_PRIMARY_SCORES)
    metastatic = pd.read_parquet(LOCO_MET_SCORES)
    rows: list[pd.DataFrame] = []
    for cohort, arm, subcohort, met_target in (
        ("RIH", "family_rih", "RIH-Colon", "rih_m"),
        ("SR1482", "family_surgen", "SR1482", "sr1482_m"),
    ):
        p = primary[
            primary["arm"].eq(arm)
            & primary["target"].eq("primary")
            & primary["subcohort"].eq(subcohort)
        ].copy()
        m = metastatic[metastatic["arm"].eq(arm) & metastatic["target"].eq(met_target)].copy()
        for frame, role in ((p, "primary"), (m, "metastatic")):
            frame = frame.rename(columns={"label": "label_score", "mean_logit": "score"})
            frame["cohort"] = cohort
            frame["role"] = role
            frame["lineage"] = "family_naive_univ1_5seed"
            frame["encoder"] = "UNI-v1"
            frame["comparison_status"] = "conformant_same_family_held_out_model"
            rows.append(
                frame[
                    [
                        "patient_id",
                        "cohort",
                        "role",
                        "label_score",
                        "score",
                        "lineage",
                        "encoder",
                        "comparison_status",
                    ]
                ]
            )

    two = pd.read_parquet(TWO_ENCODER_SCORES)
    for encoder, label in (("univ1", "UNI-v1"), ("virchow2_cls", "Virchow2-CLS")):
        for cohort, subcohort, met_dataset in (
            ("RIH", "RIH-Colon", "rih_metastatic"),
            ("SR1482", "SR1482", "sr1482_metastatic"),
        ):
            if cohort == "RIH":
                p = two[
                    two["analysis_family"].eq("target_refit_zero_shot_or_sensitivity")
                    & two["dataset"].eq("rih_primary")
                    & two["encoder"].eq(encoder)
                ].copy()
                status = "same_tcga_surgen_target_refits"
            else:
                p = two[
                    two["analysis_family"].eq("source_restricted_tcga_surgen_oof")
                    & two["dataset"].eq("tcga_surgen_source")
                    & two["subcohort"].eq(subcohort)
                    & two["encoder"].eq(encoder)
                ].copy()
                status = "nonconformant_source_oof_vs_source_exposed_target_refit"
            m = two[
                two["analysis_family"].eq("target_refit_zero_shot_or_sensitivity")
                & two["dataset"].eq(met_dataset)
                & two["encoder"].eq(encoder)
            ].copy()
            for frame, role in ((p, "primary"), (m, "metastatic")):
                frame = frame.rename(
                    columns={
                        "patient_id": "patient_id",
                        "label": "label_score",
                        "mean_logit_5seed": "score",
                    }
                )
                frame["cohort"] = cohort
                frame["role"] = role
                frame["lineage"] = f"tcga_surgen_{encoder}_5seed"
                frame["encoder"] = label
                frame["comparison_status"] = status
                rows.append(
                    frame[
                        [
                            "patient_id",
                            "cohort",
                            "role",
                            "label_score",
                            "score",
                            "lineage",
                            "encoder",
                            "comparison_status",
                        ]
                    ]
                )

    scores = pd.concat(rows, ignore_index=True)
    if scores.duplicated(["lineage", "cohort", "role", "patient_id"]).any():
        raise ValueError("Score keys are not unique")
    scores = _attach_metadata(scores, metadata)
    expected = {
        ("RIH", "primary"): 153,
        ("RIH", "metastatic"): 85,
        ("SR1482", "primary"): 324,
        ("SR1482", "metastatic"): 74,
    }
    for lineage in scores["lineage"].unique():
        counts = (
            scores[scores["lineage"].eq(lineage)]
            .groupby(["cohort", "role"])["patient_id"]
            .nunique()
            .to_dict()
        )
        if counts != expected:
            raise ValueError(f"Unexpected score counts for {lineage}: {counts}")
    return scores.sort_values(["lineage", "cohort", "role", "patient_id"])


def _auc(frame: pd.DataFrame, score_column: str = "score") -> float:
    if frame["label"].nunique() != 2:
        return float("nan")
    return float(roc_auc_score(frame["label"], frame[score_column]))


def _auprc(frame: pd.DataFrame, score_column: str = "score") -> float:
    if frame["label"].nunique() != 2:
        return float("nan")
    return float(average_precision_score(frame["label"], frame[score_column]))


def _resample_stratified(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    strata: Iterable[str] = ("label",),
) -> pd.DataFrame:
    pieces = []
    strata_list = list(strata)
    grouper: str | list[str] = strata_list[0] if len(strata_list) == 1 else strata_list
    for _, group in frame.groupby(grouper, dropna=False, sort=False):
        take = rng.integers(0, len(group), size=len(group))
        pieces.append(group.iloc[take])
    return pd.concat(pieces, ignore_index=True)


def _percentile(values: list[float]) -> tuple[float, float, int]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(array):
        return float("nan"), float("nan"), 0
    lower, upper = np.quantile(array, [0.025, 0.975])
    return float(lower), float(upper), int(len(array))


def _bootstrap_metric_draws(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
    include_auprc: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fast label-stratified patient bootstrap using multinomial counts."""
    positive = frame.loc[frame["label"].eq(1), "score"].to_numpy(dtype=float)
    negative = frame.loc[frame["label"].eq(0), "score"].to_numpy(dtype=float)
    concordance = (positive[:, None] > negative[None, :]).astype(float)
    concordance += 0.5 * (positive[:, None] == negative[None, :]).astype(float)

    pos_group_map: np.ndarray | None = None
    neg_group_map: np.ndarray | None = None
    if include_auprc:
        scores = np.concatenate([positive, negative])
        _, ascending_group = np.unique(scores, return_inverse=True)
        n_groups = int(ascending_group.max()) + 1
        descending_group = n_groups - 1 - ascending_group
        pos_group_map = np.zeros((len(positive), n_groups), dtype=float)
        neg_group_map = np.zeros((len(negative), n_groups), dtype=float)
        pos_group_map[np.arange(len(positive)), descending_group[: len(positive)]] = 1.0
        neg_group_map[np.arange(len(negative)), descending_group[len(positive) :]] = 1.0

    auc_chunks: list[np.ndarray] = []
    ap_chunks: list[np.ndarray] = []
    batch_size = 512
    for start in range(0, n_bootstrap, batch_size):
        size = min(batch_size, n_bootstrap - start)
        pos_counts = rng.multinomial(
            len(positive), np.full(len(positive), 1 / len(positive)), size=size
        ).astype(float)
        neg_counts = rng.multinomial(
            len(negative), np.full(len(negative), 1 / len(negative)), size=size
        ).astype(float)
        auc_chunks.append(
            np.einsum(
                "bi,ij,bj->b",
                pos_counts,
                concordance,
                neg_counts,
                optimize=True,
            )
            / (len(positive) * len(negative))
        )
        if include_auprc:
            if pos_group_map is None or neg_group_map is None:
                raise AssertionError("AUPRC score-group maps were not initialized")
            grouped_pos = pos_counts @ pos_group_map
            grouped_neg = neg_counts @ neg_group_map
            cumulative_pos = np.cumsum(grouped_pos, axis=1)
            cumulative_total = np.cumsum(grouped_pos + grouped_neg, axis=1)
            precision = np.divide(
                cumulative_pos,
                cumulative_total,
                out=np.zeros_like(cumulative_pos),
                where=cumulative_total > 0,
            )
            ap_chunks.append(np.sum(precision * grouped_pos, axis=1) / len(positive))
    auc = np.concatenate(auc_chunks)
    ap = np.concatenate(ap_chunks) if include_auprc else None
    return auc, ap


def metric_block(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed_key: str,
) -> dict[str, Any]:
    n_mutant = int(frame["label"].sum())
    n_wild_type = int(len(frame) - n_mutant)
    result = {
        "n": int(len(frame)),
        "n_mutant": n_mutant,
        "n_wild_type": n_wild_type,
        "prevalence": float(frame["label"].mean()) if len(frame) else None,
        "auroc": _auc(frame),
        "auprc": _auprc(frame),
        "inferential_support": bool(len(frame) >= 20 and min(n_mutant, n_wild_type) >= 10),
        "support_rule": "n>=20 and >=10 patients in each KRAS class",
    }
    if not n_mutant or not n_wild_type:
        result.update(
            {
                "auroc_ci_low": None,
                "auroc_ci_high": None,
                "auprc_ci_low": None,
                "auprc_ci_high": None,
                "n_bootstrap_valid": 0,
            }
        )
        return result
    rng = np.random.default_rng(_stable_seed(RNG_SEED, seed_key))
    auc_draws, ap_draws = _bootstrap_metric_draws(
        frame, n_bootstrap=n_bootstrap, rng=rng, include_auprc=True
    )
    if ap_draws is None:
        raise AssertionError("AUPRC bootstrap was requested but not returned")
    auc_low, auc_high, n_valid = _percentile(auc_draws.tolist())
    ap_low, ap_high, _ = _percentile(ap_draws.tolist())
    result.update(
        {
            "auroc_ci_low": auc_low,
            "auroc_ci_high": auc_high,
            "auprc_ci_low": ap_low,
            "auprc_ci_high": ap_high,
            "n_bootstrap_valid": n_valid,
            "bootstrap_method": "patient resampling within KRAS class",
        }
    )
    return result


def performance_strata(scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(group: pd.DataFrame, dimension: str, level: str) -> None:
        first = group.iloc[0]
        block = metric_block(
            group,
            n_bootstrap=n_bootstrap,
            seed_key=(f"metric/{first.lineage}/{first.cohort}/{first.role}/{dimension}/{level}"),
        )
        rows.append(
            {
                "lineage": first.lineage,
                "encoder": first.encoder,
                "cohort": first.cohort,
                "role": first.role,
                "comparison_status": first.comparison_status,
                "dimension": dimension,
                "level": level,
                **block,
            }
        )

    keys = ["lineage", "cohort", "role"]
    for _, group in scores.groupby(keys, sort=True):
        add(group, "overall", "all")
        for level, subset in group.groupby("source_type", dropna=False, sort=True):
            add(subset, "source_type", str(level))
        if group.iloc[0]["role"] == "metastatic":
            for level, subset in group.groupby("site_binary", sort=True):
                add(subset, "site_binary", str(level))
            for level, subset in group.groupby("metastatic_site", sort=True):
                add(subset, "metastatic_site", str(level))
            for (source, site), subset in group.groupby(["source_type", "site_binary"], sort=True):
                add(subset, "source_type_x_site", f"{source}/{site}")
    return pd.DataFrame(rows)


def _contrast(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed_key: str,
    strata: Iterable[str] = ("label",),
) -> dict[str, Any]:
    point = _auc(first) - _auc(second)
    rng = np.random.default_rng(_stable_seed(RNG_SEED, seed_key))
    if tuple(strata) != ("label",):
        raise ValueError("Fast contrast bootstrap only supports label stratification")
    first_draws, _ = _bootstrap_metric_draws(
        first, n_bootstrap=n_bootstrap, rng=rng, include_auprc=False
    )
    second_draws, _ = _bootstrap_metric_draws(
        second, n_bootstrap=n_bootstrap, rng=rng, include_auprc=False
    )
    low, high, valid = _percentile((first_draws - second_draws).tolist())
    return {
        "delta_auroc": point,
        "ci_low": low,
        "ci_high": high,
        "n_bootstrap_valid": valid,
    }


def _disjoint_roles(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    primary = frame[frame["role"].eq("primary")].copy()
    metastatic = frame[frame["role"].eq("metastatic")].copy()
    shared = set(primary["patient_id"]) & set(metastatic["patient_id"])
    primary = primary[~primary["patient_id"].isin(shared)].copy()
    metastatic = metastatic[~metastatic["patient_id"].isin(shared)].copy()
    return primary, metastatic, len(shared)


def contrast_table(scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (lineage, cohort), frame in scores.groupby(["lineage", "cohort"], sort=True):
        met = frame[frame["role"].eq("metastatic")]
        first = frame.iloc[0]
        for dimension, positive, negative in (
            ("source_type", "biopsy", "resection"),
            ("site_binary", "liver", "non_liver"),
        ):
            a = met[met[dimension].eq(positive)]
            b = met[met[dimension].eq(negative)]
            if a["label"].nunique() != 2 or b["label"].nunique() != 2:
                continue
            block = _contrast(
                a,
                b,
                n_bootstrap=n_bootstrap,
                seed_key=f"contrast/{lineage}/{cohort}/{dimension}",
            )
            rows.append(
                {
                    "lineage": lineage,
                    "encoder": first.encoder,
                    "cohort": cohort,
                    "contrast": f"metastatic_{positive}_minus_{negative}",
                    "first_n": len(a),
                    "second_n": len(b),
                    "first_n_mutant": int(a["label"].sum()),
                    "second_n_mutant": int(b["label"].sum()),
                    "first_auroc": _auc(a),
                    "second_auroc": _auc(b),
                    "strict_support": bool(
                        min(int(a["label"].sum()), int((1 - a["label"]).sum())) >= 10
                        and min(int(b["label"].sum()), int((1 - b["label"]).sum())) >= 10
                    ),
                    "overlap_removed": 0,
                    **block,
                }
            )

        primary, metastatic, n_overlap = _disjoint_roles(frame)
        comparisons = [("all", primary, metastatic)]
        for source in SOURCE_LEVELS:
            comparisons.append(
                (
                    source,
                    primary[primary["source_type"].eq(source)],
                    metastatic[metastatic["source_type"].eq(source)],
                )
            )
        comparisons.append(("set_d", primary[primary["set_d"]], metastatic[metastatic["set_d"]]))
        for level, p_group, m_group in comparisons:
            if p_group["label"].nunique() != 2 or m_group["label"].nunique() != 2:
                continue
            block = _contrast(
                m_group,
                p_group,
                n_bootstrap=n_bootstrap,
                seed_key=f"role/{lineage}/{cohort}/{level}",
            )
            rows.append(
                {
                    "lineage": lineage,
                    "encoder": first.encoder,
                    "cohort": cohort,
                    "contrast": f"metastatic_minus_primary/{level}",
                    "first_n": len(m_group),
                    "second_n": len(p_group),
                    "first_n_mutant": int(m_group["label"].sum()),
                    "second_n_mutant": int(p_group["label"].sum()),
                    "first_auroc": _auc(m_group),
                    "second_auroc": _auc(p_group),
                    "strict_support": bool(
                        min(
                            int(m_group["label"].sum()),
                            int((1 - m_group["label"]).sum()),
                        )
                        >= 10
                        and min(
                            int(p_group["label"].sum()),
                            int((1 - p_group["label"]).sum()),
                        )
                        >= 10
                    ),
                    "overlap_removed": n_overlap,
                    **block,
                }
            )
    return pd.DataFrame(rows)


def _cross_group_auc(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    score_column: str = "score",
) -> float:
    pos = positives[score_column].to_numpy(dtype=float)[:, None]
    neg = negatives[score_column].to_numpy(dtype=float)[None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def pairwise_source_cells(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metastatic = scores[scores["role"].eq("metastatic")]
    for (lineage, cohort), frame in metastatic.groupby(["lineage", "cohort"], sort=True):
        for pos_source, pos in frame[frame["label"].eq(1)].groupby("source_type"):
            for neg_source, neg in frame[frame["label"].eq(0)].groupby("source_type"):
                rows.append(
                    {
                        "lineage": lineage,
                        "encoder": frame.iloc[0].encoder,
                        "cohort": cohort,
                        "positive_source_type": pos_source,
                        "negative_source_type": neg_source,
                        "n_positive": len(pos),
                        "n_negative": len(neg),
                        "pair_weight": len(pos) * len(neg),
                        "pairwise_concordance": _cross_group_auc(pos, neg),
                    }
                )
    return pd.DataFrame(rows)


def _standardized_auc(
    frame: pd.DataFrame,
    category: str,
    positive_weights: dict[str, float],
    negative_weights: dict[str, float] | None = None,
) -> float:
    if negative_weights is None:
        negative_weights = {"all": 1.0}
    total = 0.0
    for pos_level, pos_weight in positive_weights.items():
        pos = frame[(frame["label"].eq(1)) & (frame[category].eq(pos_level))]
        for neg_level, neg_weight in negative_weights.items():
            neg = frame[frame["label"].eq(0)]
            if neg_level != "all":
                neg = neg[neg[category].eq(neg_level)]
            if len(pos) == 0 or len(neg) == 0:
                return float("nan")
            total += pos_weight * neg_weight * _cross_group_auc(pos, neg)
    return float(total)


def _within_category_standardized_auc(
    frame: pd.DataFrame,
    category: str,
    weights: dict[str, float],
) -> float:
    """Weighted mean of AUROCs formed only within the same category."""
    total = 0.0
    for level, weight in weights.items():
        subset = frame[frame[category].eq(level)]
        if subset["label"].nunique() != 2:
            return float("nan")
        total += weight * _auc(subset)
    return float(total)


def _category_distribution(
    frame: pd.DataFrame, category: str, label: int, levels: list[str]
) -> dict[str, float]:
    subset = frame[frame["label"].eq(label)]
    return {level: float(subset[category].eq(level).mean()) for level in levels}


def _composition_point(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
    *,
    category: str,
    analysis: str,
    levels: list[str],
) -> dict[str, float | dict[str, float]]:
    """Compute a composition contrast, estimating its target mix from this sample."""
    p_pos = _category_distribution(primary, category, 1, levels)
    m_pos = _category_distribution(metastatic, category, 1, levels)
    pos_weights = {key: (p_pos[key] + m_pos[key]) / 2 for key in levels}
    pos_total = sum(pos_weights.values())
    if pos_total <= 0:
        raise ValueError("No positive-class mass in composition estimand")
    pos_weights = {key: value / pos_total for key, value in pos_weights.items()}

    negative_weights: dict[str, float] | None = None
    if analysis == "source_type":
        p_neg = _category_distribution(primary, category, 0, levels)
        m_neg = _category_distribution(metastatic, category, 0, levels)
        negative_weights = {key: (p_neg[key] + m_neg[key]) / 2 for key in levels}
        neg_total = sum(negative_weights.values())
        if neg_total <= 0:
            raise ValueError("No negative-class mass in composition estimand")
        negative_weights = {key: value / neg_total for key, value in negative_weights.items()}

    p_observed = _auc(primary)
    m_observed = _auc(metastatic)
    p_standardized = _standardized_auc(primary, category, pos_weights, negative_weights)
    m_standardized = _standardized_auc(metastatic, category, pos_weights, negative_weights)
    observed_difference = m_observed - p_observed
    standardized_difference = m_standardized - p_standardized

    result: dict[str, float | dict[str, float]] = {
        "primary_observed_auroc": p_observed,
        "metastatic_observed_auroc": m_observed,
        "observed_difference": observed_difference,
        "primary_standardized_auroc": p_standardized,
        "metastatic_standardized_auroc": m_standardized,
        "standardized_difference": standardized_difference,
        "descriptive_composition_point_component": (observed_difference - standardized_difference),
        "positive_target_weights": pos_weights,
    }
    if negative_weights is not None:
        result["negative_target_weights"] = negative_weights

        combined = pd.concat([primary, metastatic], ignore_index=True)
        within_weights = {level: float(combined[category].eq(level).mean()) for level in levels}
        within_total = sum(within_weights.values())
        within_weights = {level: value / within_total for level, value in within_weights.items()}
        p_within = _within_category_standardized_auc(primary, category, within_weights)
        m_within = _within_category_standardized_auc(metastatic, category, within_weights)
        result.update(
            {
                "primary_within_source_standardized_auroc": p_within,
                "metastatic_within_source_standardized_auroc": m_within,
                "within_source_standardized_difference": m_within - p_within,
                "within_source_target_weights": within_weights,
            }
        )
    return result


def decomposition_table(scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bootstrap_requested = min(n_bootstrap, SECONDARY_BOOTSTRAPS)
    family = scores[scores["lineage"].eq("family_naive_univ1_5seed")]
    for cohort, frame in family.groupby("cohort", sort=True):
        primary, metastatic, removed = _disjoint_roles(frame)
        for analysis, category, levels in (
            (
                "source_type",
                "source_type",
                sorted(set(primary["source_type"]) | set(metastatic["source_type"])),
            ),
            ("mutant_subtype", "subtype_group", list(SUBTYPE_LEVELS)),
        ):
            if analysis == "mutant_subtype":
                levels = [
                    level
                    for level in levels
                    if primary.loc[primary["label"].eq(1), category].eq(level).any()
                    or metastatic.loc[metastatic["label"].eq(1), category].eq(level).any()
                ]
            point = _composition_point(
                primary,
                metastatic,
                category=category,
                analysis=analysis,
                levels=levels,
            )

            rng = np.random.default_rng(_stable_seed(RNG_SEED, "decomposition", cohort, analysis))
            draws = {
                "observed": [],
                "standardized": [],
                "composition_point_component": [],
                "within_standardized": [],
            }
            invalid = 0
            for _ in range(bootstrap_requested):
                p_draw = _resample_stratified(primary, rng, ["label"])
                m_draw = _resample_stratified(metastatic, rng, ["label"])
                try:
                    estimate = _composition_point(
                        p_draw,
                        m_draw,
                        category=category,
                        analysis=analysis,
                        levels=levels,
                    )
                except (ValueError, ZeroDivisionError):
                    invalid += 1
                    continue
                values = {
                    "observed": estimate["observed_difference"],
                    "standardized": estimate["standardized_difference"],
                    "composition_point_component": estimate[
                        "descriptive_composition_point_component"
                    ],
                }
                if analysis == "source_type":
                    values["within_standardized"] = estimate[
                        "within_source_standardized_difference"
                    ]
                if not all(np.isfinite(float(value)) for value in values.values()):
                    invalid += 1
                    continue
                for name, value in values.items():
                    draws[name].append(float(value))

            required_cells: list[int] = []
            if analysis == "source_type":
                for role_frame in (primary, metastatic):
                    for level in levels:
                        for label in (0, 1):
                            required_cells.append(
                                int(
                                    (
                                        role_frame[category].eq(level)
                                        & role_frame["label"].eq(label)
                                    ).sum()
                                )
                            )
            else:
                for role_frame in (primary, metastatic):
                    for level in levels:
                        required_cells.append(
                            int((role_frame[category].eq(level) & role_frame["label"].eq(1)).sum())
                        )
            strict_support = bool(required_cells) and min(required_cells) >= MIN_CELL_PER_CLASS
            bootstrap_valid_fraction = (bootstrap_requested - invalid) / bootstrap_requested
            row: dict[str, Any] = {
                "cohort": cohort,
                "analysis": analysis,
                "overlap_removed": removed,
                "primary_n": len(primary),
                "metastatic_n": len(metastatic),
                "primary_observed_auroc": point["primary_observed_auroc"],
                "metastatic_observed_auroc": point["metastatic_observed_auroc"],
                "observed_difference": point["observed_difference"],
                "primary_standardized_auroc": point["primary_standardized_auroc"],
                "metastatic_standardized_auroc": point["metastatic_standardized_auroc"],
                "standardized_difference": point["standardized_difference"],
                "descriptive_composition_point_component": point[
                    "descriptive_composition_point_component"
                ],
                "primary_within_source_standardized_auroc": point.get(
                    "primary_within_source_standardized_auroc"
                ),
                "metastatic_within_source_standardized_auroc": point.get(
                    "metastatic_within_source_standardized_auroc"
                ),
                "within_source_standardized_difference": point.get(
                    "within_source_standardized_difference"
                ),
                "within_source_target_weights": (
                    json.dumps(point["within_source_target_weights"], sort_keys=True)
                    if "within_source_target_weights" in point
                    else None
                ),
                "positive_target_weights": json.dumps(
                    point["positive_target_weights"], sort_keys=True
                ),
                "negative_target_weights": (
                    json.dumps(point["negative_target_weights"], sort_keys=True)
                    if "negative_target_weights" in point
                    else "all_wild_type"
                ),
                "minimum_required_cell_n": min(required_cells),
                "strict_cell_support": strict_support,
                "bootstrap_requested": bootstrap_requested,
                "bootstrap_invalid": invalid,
                "bootstrap_valid_fraction": bootstrap_valid_fraction,
                "evidence_state": (
                    "POST_HOC_COMPOSITION_SENSITIVITY"
                    if strict_support and bootstrap_valid_fraction >= 0.95
                    else "DESCRIPTIVE_SPARSE_OR_UNSTABLE_COMPOSITION_CELLS"
                ),
            }
            for name, values in draws.items():
                if not values or bootstrap_valid_fraction < 0.95:
                    row[f"{name}_ci_low"] = None
                    row[f"{name}_ci_high"] = None
                    row[f"{name}_bootstrap_valid"] = len(values)
                    continue
                low, high, valid = _percentile(values)
                row[f"{name}_ci_low"] = low
                row[f"{name}_ci_high"] = high
                row[f"{name}_bootstrap_valid"] = valid
            rows.append(row)
    return pd.DataFrame(rows)


def site_source_standardization(scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    """Compare liver/non-liver ranking at a common source-type composition."""
    rows: list[dict[str, Any]] = []
    bootstrap_requested = min(n_bootstrap, SECONDARY_BOOTSTRAPS)
    metastatic = scores[scores["role"].eq("metastatic")]
    for (lineage, cohort), frame in metastatic.groupby(["lineage", "cohort"], sort=True):
        liver = frame[frame["site_binary"].eq("liver")].copy()
        non_liver = frame[frame["site_binary"].eq("non_liver")].copy()
        levels = sorted(set(liver["source_type"]) | set(non_liver["source_type"]))
        pos_liver = _category_distribution(liver, "source_type", 1, levels)
        pos_non = _category_distribution(non_liver, "source_type", 1, levels)
        neg_liver = _category_distribution(liver, "source_type", 0, levels)
        neg_non = _category_distribution(non_liver, "source_type", 0, levels)
        positive_weights = {level: (pos_liver[level] + pos_non[level]) / 2 for level in levels}
        negative_weights = {level: (neg_liver[level] + neg_non[level]) / 2 for level in levels}
        pos_total = sum(positive_weights.values())
        neg_total = sum(negative_weights.values())
        positive_weights = {level: value / pos_total for level, value in positive_weights.items()}
        negative_weights = {level: value / neg_total for level, value in negative_weights.items()}
        observed = _auc(liver) - _auc(non_liver)
        liver_standardized = _standardized_auc(
            liver, "source_type", positive_weights, negative_weights
        )
        non_standardized = _standardized_auc(
            non_liver, "source_type", positive_weights, negative_weights
        )
        standardized = liver_standardized - non_standardized
        rng = np.random.default_rng(
            _stable_seed(RNG_SEED, "site_source_standardization", lineage, cohort)
        )
        observed_draws: list[float] = []
        standardized_draws: list[float] = []
        invalid_standardized = 0
        for _ in range(bootstrap_requested):
            liver_draw = _resample_stratified(liver, rng, ["label"])
            non_draw = _resample_stratified(non_liver, rng, ["label"])
            observed_draws.append(_auc(liver_draw) - _auc(non_draw))
            draw_pos_liver = _category_distribution(liver_draw, "source_type", 1, levels)
            draw_pos_non = _category_distribution(non_draw, "source_type", 1, levels)
            draw_neg_liver = _category_distribution(liver_draw, "source_type", 0, levels)
            draw_neg_non = _category_distribution(non_draw, "source_type", 0, levels)
            draw_positive = {
                level: (draw_pos_liver[level] + draw_pos_non[level]) / 2 for level in levels
            }
            draw_negative = {
                level: (draw_neg_liver[level] + draw_neg_non[level]) / 2 for level in levels
            }
            draw_pos_total = sum(draw_positive.values())
            draw_neg_total = sum(draw_negative.values())
            if draw_pos_total <= 0 or draw_neg_total <= 0:
                invalid_standardized += 1
                continue
            draw_positive = {
                level: value / draw_pos_total for level, value in draw_positive.items()
            }
            draw_negative = {
                level: value / draw_neg_total for level, value in draw_negative.items()
            }
            draw_standardized = _standardized_auc(
                liver_draw, "source_type", draw_positive, draw_negative
            ) - _standardized_auc(non_draw, "source_type", draw_positive, draw_negative)
            if np.isfinite(draw_standardized):
                standardized_draws.append(float(draw_standardized))
            else:
                invalid_standardized += 1
        observed_low, observed_high, observed_valid = _percentile(observed_draws)
        standardized_valid_fraction = len(standardized_draws) / bootstrap_requested
        if standardized_draws and standardized_valid_fraction >= 0.95:
            std_low, std_high, std_valid = _percentile(standardized_draws)
        else:
            std_low = std_high = None
            std_valid = len(standardized_draws)
        cell_counts = frame.groupby(["site_binary", "source_type", "label"]).size().to_dict()
        strict_support = all(
            cell_counts.get((site, source, label), 0) >= 10
            for site in ("liver", "non_liver")
            for source in levels
            for label in (0, 1)
        )
        rows.append(
            {
                "lineage": lineage,
                "encoder": frame.iloc[0].encoder,
                "cohort": cohort,
                "source_levels": ",".join(levels),
                "observed_liver_minus_non_liver": observed,
                "observed_ci_low": observed_low,
                "observed_ci_high": observed_high,
                "observed_bootstrap_valid": observed_valid,
                "liver_source_standardized_auroc": liver_standardized,
                "non_liver_source_standardized_auroc": non_standardized,
                "source_standardized_liver_minus_non_liver": standardized,
                "standardized_ci_low": std_low,
                "standardized_ci_high": std_high,
                "standardized_bootstrap_valid": std_valid,
                "strict_cell_support": strict_support,
                "evidence_state": (
                    "MINIMUM_CELL_COUNT_SUPPORT"
                    if strict_support
                    else "DESCRIPTIVE_SPARSE_SOURCE_BY_SITE_CELLS"
                ),
                "standardized_bootstrap_invalid": invalid_standardized,
                "standardized_bootstrap_valid_fraction": standardized_valid_fraction,
                "bootstrap_requested": bootstrap_requested,
                "positive_target_weights": json.dumps(positive_weights, sort_keys=True),
                "negative_target_weights": json.dumps(negative_weights, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _weighted_mean_variance(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return float("nan"), float("nan")
    mean = float(np.sum(weights * values) / total)
    variance = float(np.sum(weights * np.square(values - mean)) / total)
    return mean, variance


def _role_design(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], dict[str, Any]]:
    """Construct one fixed design for a class-specific role propensity."""
    combined = pd.concat(
        [primary.assign(_target_role=0), metastatic.assign(_target_role=1)],
        ignore_index=True,
    )
    design_parts: list[pd.DataFrame] = []
    balance_parts: list[pd.DataFrame] = []
    support: dict[str, Any] = {
        "categorical_target_levels_absent_from_primary": {},
        "target_outside_primary_age_range_fraction": 0.0,
    }
    for feature in features:
        if feature == "age":
            values = pd.to_numeric(combined[feature], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            missing = values.isna().astype(float)
            filled = values.fillna(median).astype(float)
            scale = float(filled.std(ddof=0))
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0
            standardized = (filled - float(filled.mean())) / scale
            design_parts.append(pd.DataFrame({"age": standardized, "age_missing": missing}))
            balance_parts.append(pd.DataFrame({"age": standardized, "age_missing": missing}))
            p_age = pd.to_numeric(primary[feature], errors="coerce").dropna()
            m_age = pd.to_numeric(metastatic[feature], errors="coerce")
            if len(p_age) and m_age.notna().any():
                outside = m_age.notna() & (
                    (m_age < float(p_age.min())) | (m_age > float(p_age.max()))
                )
                support["target_outside_primary_age_range_fraction"] = float(
                    outside.sum() / m_age.notna().sum()
                )
            elif m_age.notna().any():
                support["target_outside_primary_age_range_fraction"] = 1.0
        else:
            values = combined[feature].fillna("missing").astype(str).replace("", "missing")
            primary_levels = set(
                primary[feature].fillna("missing").astype(str).replace("", "missing")
            )
            metastatic_levels = set(
                metastatic[feature].fillna("missing").astype(str).replace("", "missing")
            )
            absent = sorted(metastatic_levels - primary_levels)
            if absent:
                support["categorical_target_levels_absent_from_primary"][feature] = absent
            full = pd.get_dummies(values, prefix=feature, dtype=float)
            balance_parts.append(full)
            if full.shape[1] > 1:
                design_parts.append(full.iloc[:, 1:])

    design = pd.concat(design_parts, axis=1) if design_parts else pd.DataFrame(index=combined.index)
    if design.shape[1] == 0:
        design = pd.DataFrame({"intercept_only_marker": np.zeros(len(combined))})
    balance = (
        pd.concat(balance_parts, axis=1) if balance_parts else pd.DataFrame(index=combined.index)
    )
    if not np.isfinite(design.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite role-propensity design")
    return (
        design.to_numpy(dtype=float),
        combined["_target_role"].to_numpy(dtype=int),
        balance,
        list(design.columns),
        support,
    )


def _fit_class_att_weights(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
    features: tuple[str, ...],
    *,
    label: int,
) -> dict[str, Any]:
    primary = primary[primary["label"].eq(label)].reset_index(drop=True)
    metastatic = metastatic[metastatic["label"].eq(label)].reset_index(drop=True)
    if len(primary) < 20 or len(metastatic) < MIN_CELL_PER_CLASS:
        return {
            "estimable": False,
            "reason": "fewer_than_20_primary_or_10_metastatic_patients_in_label_cell",
            "primary_n": len(primary),
            "metastatic_n": len(metastatic),
        }
    design, role, balance, design_columns, support = _role_design(primary, metastatic, features)
    model = LogisticRegression(
        C=10.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=RNG_SEED,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(design, role)
        raw_propensity = model.predict_proba(design)[:, 1]
    except (ValueError, FloatingPointError, ConvergenceWarning) as exc:
        return {
            "estimable": False,
            "reason": f"propensity_fit_failed:{type(exc).__name__}",
            "primary_n": len(primary),
            "metastatic_n": len(metastatic),
        }
    low, high = PROPENSITY_CLIP
    propensity = np.clip(raw_propensity, low, high)
    primary_weights = propensity[: len(primary)] / (1 - propensity[: len(primary)])
    clipped_fraction = float(np.mean((raw_propensity < low) | (raw_propensity > high)))
    weight_sum = float(primary_weights.sum())
    ess = float(weight_sum**2 / np.square(primary_weights).sum())
    ess_fraction = ess / len(primary)
    max_weight_ratio = float(primary_weights.max() / primary_weights.mean())
    max_weight_share = float(primary_weights.max() / weight_sum)
    primary_propensity = raw_propensity[: len(primary)]
    metastatic_propensity = raw_propensity[len(primary) :]
    propensity_outside_fraction = float(
        np.mean(
            (metastatic_propensity < primary_propensity.min())
            | (metastatic_propensity > primary_propensity.max())
        )
    )

    categorical_features = tuple(feature for feature in features if feature != "age")
    if categorical_features:
        primary_profiles = set(
            primary.loc[:, categorical_features]
            .fillna("missing")
            .astype(str)
            .replace("", "missing")
            .itertuples(index=False, name=None)
        )
        metastatic_profiles = list(
            metastatic.loc[:, categorical_features]
            .fillna("missing")
            .astype(str)
            .replace("", "missing")
            .itertuples(index=False, name=None)
        )
        joint_profile_coverage = float(
            np.mean([profile in primary_profiles for profile in metastatic_profiles])
        )
    else:
        joint_profile_coverage = 1.0

    p_balance = balance.iloc[: len(primary)].to_numpy(dtype=float)
    m_balance = balance.iloc[len(primary) :].to_numpy(dtype=float)
    smds: dict[str, float] = {}
    for index, column in enumerate(balance.columns):
        p_mean, p_var = _weighted_mean_variance(p_balance[:, index], primary_weights)
        m_mean, m_var = _weighted_mean_variance(m_balance[:, index], np.ones(len(metastatic)))
        denominator = math.sqrt(max((p_var + m_var) / 2, 0.0))
        difference = abs(p_mean - m_mean)
        smds[str(column)] = (
            0.0 if denominator == 0 and difference < 1e-12 else difference / denominator
        )
    max_abs_smd = max(smds.values(), default=0.0)
    categorical_support = not support["categorical_target_levels_absent_from_primary"]
    finite = bool(
        np.isfinite(primary_weights).all()
        and np.isfinite(raw_propensity).all()
        and np.isfinite(max_abs_smd)
    )
    gates = {
        "finite_fit_and_weights": finite,
        "categorical_target_support": categorical_support,
        "target_age_outside_source_support_le_0_05": support[
            "target_outside_primary_age_range_fraction"
        ]
        <= 0.05,
        "target_propensity_outside_primary_range_le_0_05": (propensity_outside_fraction <= 0.05),
        "clipped_fraction_le_0_10": clipped_fraction <= 0.10,
        "primary_weighted_ess_ge_20": ess >= 20,
        "primary_weighted_ess_fraction_ge_0_25": ess_fraction >= 0.25,
        "maximum_weight_ratio_le_10": max_weight_ratio <= 10,
        "maximum_weight_share_le_0_10": max_weight_share <= 0.10,
        "postweight_max_abs_smd_le_0_20": max_abs_smd <= 0.20,
    }
    return {
        "estimable": all(gates.values()),
        "reason": "all_gates_pass" if all(gates.values()) else "one_or_more_gates_failed",
        "primary_n": len(primary),
        "metastatic_n": len(metastatic),
        "primary_scores": primary["score"].to_numpy(dtype=float),
        "metastatic_scores": metastatic["score"].to_numpy(dtype=float),
        "primary_weights": primary_weights,
        "design_columns": design_columns,
        "diagnostics": {
            **support,
            "propensity_clip": [low, high],
            "propensity_model": "additive_L2_logistic_C_10.0",
            "clipped_fraction": clipped_fraction,
            "target_propensity_outside_primary_range_fraction": (propensity_outside_fraction),
            "joint_categorical_profile_target_coverage": joint_profile_coverage,
            "primary_weighted_ess": ess,
            "primary_weighted_ess_fraction": ess_fraction,
            "maximum_weight_ratio_to_mean": max_weight_ratio,
            "maximum_weight_share": max_weight_share,
            "postweight_max_abs_smd": max_abs_smd,
            "postweight_smd_count_gt_0_10": int(sum(value > 0.10 for value in smds.values())),
            "postweight_smd": smds,
            "gates": gates,
        },
    }


def _weighted_pairwise_auc(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    positive_weights: np.ndarray,
    negative_weights: np.ndarray,
) -> float:
    comparison = (positive_scores[:, None] > negative_scores[None, :]).astype(float)
    comparison += 0.5 * (positive_scores[:, None] == negative_scores[None, :]).astype(float)
    pair_weights = positive_weights[:, None] * negative_weights[None, :]
    return float(np.sum(comparison * pair_weights) / np.sum(pair_weights))


def _case_mix_fit(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
    features: tuple[str, ...],
) -> dict[str, Any]:
    fitted = {
        label: _fit_class_att_weights(primary, metastatic, features, label=label)
        for label in (0, 1)
    }
    point_estimable = all(fitted[label]["estimable"] for label in (0, 1))
    raw_primary = _auc(primary)
    raw_metastatic = _auc(metastatic)
    result: dict[str, Any] = {
        "estimable": point_estimable,
        "primary_observed_auroc": raw_primary,
        "metastatic_observed_auroc": raw_metastatic,
        "observed_difference": raw_metastatic - raw_primary,
        "label_diagnostics": {
            str(label): {
                key: value
                for key, value in fitted[label].items()
                if key
                not in {
                    "primary_scores",
                    "metastatic_scores",
                    "primary_weights",
                }
            }
            for label in (0, 1)
        },
    }
    has_weighted_fit = all("primary_weights" in fitted[label] for label in (0, 1))
    if not has_weighted_fit:
        return result
    weighted_primary = _weighted_pairwise_auc(
        fitted[1]["primary_scores"],
        fitted[0]["primary_scores"],
        fitted[1]["primary_weights"],
        fitted[0]["primary_weights"],
    )
    standardized_difference = raw_metastatic - weighted_primary
    result.update(
        {
            "diagnostic_ungated_primary_att_to_metastatic_auroc": weighted_primary,
            "diagnostic_ungated_att_standardized_difference": standardized_difference,
            "diagnostic_ungated_case_mix_point_component": (
                raw_metastatic - raw_primary - standardized_difference
            ),
        }
    )
    if point_estimable:
        result.update(
            {
                "primary_att_to_metastatic_auroc": weighted_primary,
                "att_standardized_difference": standardized_difference,
                "descriptive_case_mix_point_component": (
                    raw_metastatic - raw_primary - standardized_difference
                ),
            }
        )
    return result


def case_mix_standardization_table(scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    """Class-stratified ATT-to-metastatic standardization with fail-closed gates."""
    rows: list[dict[str, Any]] = []
    family = scores[scores["lineage"].eq("family_naive_univ1_5seed")]
    bootstrap_requested = min(n_bootstrap, SECONDARY_BOOTSTRAPS)
    for cohort, frame in family.groupby("cohort", sort=True):
        primary, metastatic, removed = _disjoint_roles(frame)
        for block, features in CASE_MIX_BLOCKS.items():
            point = _case_mix_fit(primary, metastatic, features)
            rng = np.random.default_rng(_stable_seed(RNG_SEED, "case_mix_att", cohort, block))
            standardized_draws: list[float] = []
            component_draws: list[float] = []
            invalid = 0
            if point["estimable"]:
                for _ in range(bootstrap_requested):
                    p_draw = _resample_stratified(primary, rng, ["label"])
                    m_draw = _resample_stratified(metastatic, rng, ["label"])
                    estimate = _case_mix_fit(p_draw, m_draw, features)
                    if not estimate["estimable"]:
                        invalid += 1
                        continue
                    standardized_draws.append(float(estimate["att_standardized_difference"]))
                    component_draws.append(float(estimate["descriptive_case_mix_point_component"]))
            else:
                invalid = bootstrap_requested
            valid_fraction = (
                len(standardized_draws) / bootstrap_requested if bootstrap_requested else 0.0
            )
            ci_available = bool(point["estimable"] and valid_fraction >= 0.95)
            std_low = std_high = component_low = component_high = None
            if ci_available:
                std_low, std_high, _ = _percentile(standardized_draws)
                component_low, component_high, _ = _percentile(component_draws)
            residual_balance = bool(
                point["estimable"]
                and any(
                    point["label_diagnostics"][str(label)]
                    .get("diagnostics", {})
                    .get("postweight_smd_count_gt_0_10", 0)
                    > 0
                    for label in (0, 1)
                )
            )
            if not point["estimable"]:
                evidence_state = "NOT_ESTIMABLE_NO_COMMON_SUPPORT_OR_BALANCE"
            elif not ci_available:
                evidence_state = "BOOTSTRAP_OVERLAP_UNSTABLE"
            elif residual_balance:
                evidence_state = "ESTIMABLE_POST_HOC_ATT_WITH_RESIDUAL_BALANCE_DIAGNOSTIC"
            else:
                evidence_state = "ESTIMABLE_POST_HOC_CLASS_STRATIFIED_ATT_TO_METASTATIC"
            rows.append(
                {
                    "cohort": cohort,
                    "block": block,
                    "features": ",".join(features),
                    "overlap_removed": removed,
                    "primary_n": len(primary),
                    "metastatic_n": len(metastatic),
                    "primary_observed_auroc": point["primary_observed_auroc"],
                    "metastatic_observed_auroc": point["metastatic_observed_auroc"],
                    "observed_difference": point["observed_difference"],
                    "primary_att_to_metastatic_auroc": point.get("primary_att_to_metastatic_auroc"),
                    "diagnostic_ungated_primary_att_to_metastatic_auroc": point.get(
                        "diagnostic_ungated_primary_att_to_metastatic_auroc"
                    ),
                    "att_standardized_difference": point.get("att_standardized_difference"),
                    "diagnostic_ungated_att_standardized_difference": point.get(
                        "diagnostic_ungated_att_standardized_difference"
                    ),
                    "att_standardized_ci_low": std_low,
                    "att_standardized_ci_high": std_high,
                    "descriptive_case_mix_point_component": point.get(
                        "descriptive_case_mix_point_component"
                    ),
                    "diagnostic_ungated_case_mix_point_component": point.get(
                        "diagnostic_ungated_case_mix_point_component"
                    ),
                    "case_mix_component_ci_low": component_low,
                    "case_mix_component_ci_high": component_high,
                    "bootstrap_requested": bootstrap_requested,
                    "bootstrap_valid": len(standardized_draws),
                    "bootstrap_invalid": invalid,
                    "bootstrap_valid_fraction": valid_fraction,
                    "label_diagnostics": json.dumps(
                        _sanitize_json(point["label_diagnostics"]), sort_keys=True
                    ),
                    "evidence_state": evidence_state,
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_mean_draws(
    values: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    counts = rng.multinomial(len(values), np.full(len(values), 1 / len(values)), size=n_bootstrap)
    return counts @ values / len(values)


def score_structure_table(scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    """Separate class-common logit location shift from class-separation shift."""
    rows: list[dict[str, Any]] = []
    family = scores[scores["lineage"].eq("family_naive_univ1_5seed")]
    for cohort, frame in family.groupby("cohort", sort=True):
        primary, metastatic, removed = _disjoint_roles(frame)
        values: dict[tuple[str, int], np.ndarray] = {}
        for role, role_frame in (("primary", primary), ("metastatic", metastatic)):
            for label in (0, 1):
                values[(role, label)] = role_frame.loc[
                    role_frame["label"].eq(label), "score"
                ].to_numpy(dtype=float)
        shift_wild_type = float(values[("metastatic", 0)].mean() - values[("primary", 0)].mean())
        shift_mutant = float(values[("metastatic", 1)].mean() - values[("primary", 1)].mean())
        midpoint_shift = (shift_mutant + shift_wild_type) / 2
        separation_shift = shift_mutant - shift_wild_type
        rng = np.random.default_rng(_stable_seed(RNG_SEED, "score_structure", cohort))
        mean_draws = {
            key: _bootstrap_mean_draws(value, n_bootstrap, rng) for key, value in values.items()
        }
        wt_draws = mean_draws[("metastatic", 0)] - mean_draws[("primary", 0)]
        mutant_draws = mean_draws[("metastatic", 1)] - mean_draws[("primary", 1)]
        midpoint_draws = (mutant_draws + wt_draws) / 2
        separation_draws = mutant_draws - wt_draws
        row: dict[str, Any] = {
            "lineage": "family_naive_univ1_5seed",
            "cohort": cohort,
            "overlap_removed": removed,
            "primary_n": len(primary),
            "metastatic_n": len(metastatic),
            "wild_type_metastatic_minus_primary_mean_logit": shift_wild_type,
            "mutant_metastatic_minus_primary_mean_logit": shift_mutant,
            "class_midpoint_shift": midpoint_shift,
            "mutant_minus_wild_type_separation_shift": separation_shift,
            "evidence_state": "POST_HOC_CONFORMANT_SCORE_SCALE_DIAGNOSTIC",
            "interpretation": (
                "midpoint is the average class shift; separation is mutant shift minus "
                "wild-type shift; neither is a causal mechanism"
            ),
        }
        for prefix, draws in (
            ("wild_type_shift", wt_draws),
            ("mutant_shift", mutant_draws),
            ("midpoint_shift", midpoint_draws),
            ("separation_shift", separation_draws),
        ):
            low, high, valid = _percentile(draws.tolist())
            row[f"{prefix}_ci_low"] = low
            row[f"{prefix}_ci_high"] = high
            row[f"{prefix}_bootstrap_valid"] = valid
        for role, role_frame in (("primary", primary), ("metastatic", metastatic)):
            for label, label_name in ((0, "wild_type"), (1, "mutant")):
                subset = role_frame.loc[role_frame["label"].eq(label), "score"]
                row[f"{role}_{label_name}_median_logit"] = float(subset.median())
                row[f"{role}_{label_name}_q25_logit"] = float(subset.quantile(0.25))
                row[f"{role}_{label_name}_q75_logit"] = float(subset.quantile(0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def source_alignment_table(scores: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    """Apply primary-derived marginal source offsets to metastatic scores."""
    rows: list[dict[str, Any]] = []
    bootstrap_requested = min(n_bootstrap, SECONDARY_BOOTSTRAPS)
    for (lineage, cohort), frame in scores.groupby(["lineage", "cohort"], sort=True):
        primary, metastatic, removed = _disjoint_roles(frame)
        comparison_status = str(frame.iloc[0].comparison_status)
        if comparison_status.startswith("nonconformant_"):
            rows.append(
                {
                    "lineage": lineage,
                    "encoder": frame.iloc[0].encoder,
                    "cohort": cohort,
                    "comparison_status": comparison_status,
                    "overlap_removed": removed,
                    "source_levels": None,
                    "primary_n": len(primary),
                    "metastatic_n": len(metastatic),
                    "native_metastatic_auroc": _auc(metastatic),
                    "source_aligned_metastatic_auroc": None,
                    "aligned_minus_native": None,
                    "ci_low": None,
                    "ci_high": None,
                    "n_bootstrap_valid": 0,
                    "bootstrap_requested": bootstrap_requested,
                    "method": None,
                    "evidence_state": (
                        "NOT_INTERPRETABLE_NONCOMPARABLE_PRIMARY_OOF_AND_TARGET_REFIT_SCORE_SCALE"
                    ),
                }
            )
            continue
        levels = sorted(set(primary["source_type"]) & set(metastatic["source_type"]))
        primary = primary[primary["source_type"].isin(levels)].copy()
        metastatic = metastatic[metastatic["source_type"].isin(levels)].copy()
        overall_mean = float(primary["score"].mean())
        offsets = primary.groupby("source_type")["score"].mean() - overall_mean
        metastatic["aligned_score"] = metastatic["score"] - metastatic["source_type"].map(offsets)
        native = _auc(metastatic)
        aligned = _auc(metastatic, "aligned_score")
        rng = np.random.default_rng(_stable_seed(RNG_SEED, "alignment", lineage, cohort))
        draws: list[float] = []
        for _ in range(bootstrap_requested):
            p_draw = _resample_stratified(primary, rng, ["source_type"])
            m_draw = _resample_stratified(metastatic, rng, ["label", "source_type"])
            p_mean = float(p_draw["score"].mean())
            draw_offsets = p_draw.groupby("source_type")["score"].mean() - p_mean
            m_draw["aligned_score"] = m_draw["score"] - m_draw["source_type"].map(draw_offsets)
            draws.append(_auc(m_draw, "aligned_score") - _auc(m_draw))
        low, high, valid = _percentile(draws)
        rows.append(
            {
                "lineage": lineage,
                "encoder": frame.iloc[0].encoder,
                "cohort": cohort,
                "comparison_status": comparison_status,
                "overlap_removed": removed,
                "source_levels": ",".join(levels),
                "primary_n": len(primary),
                "metastatic_n": len(metastatic),
                "native_metastatic_auroc": native,
                "source_aligned_metastatic_auroc": aligned,
                "aligned_minus_native": aligned - native,
                "ci_low": low,
                "ci_high": high,
                "n_bootstrap_valid": valid,
                "bootstrap_requested": bootstrap_requested,
                "method": (
                    "primary-derived offsets; metastatic labels unused in offset fit; "
                    "analysis selected and evaluated post-outcome"
                ),
                "evidence_state": "VALID_POST_HOC_PRIMARY_DERIVED_CENTERING_SENSITIVITY",
            }
        )
    return pd.DataFrame(rows)


def paired_role_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (lineage, cohort), frame in scores.groupby(["lineage", "cohort"], sort=True):
        primary = frame[frame["role"].eq("primary")].set_index("patient_id")
        metastatic = frame[frame["role"].eq("metastatic")].set_index("patient_id")
        shared = sorted(set(primary.index) & set(metastatic.index))
        if not shared:
            continue
        for patient_id in shared:
            p = primary.loc[patient_id]
            m = metastatic.loc[patient_id]
            rows.append(
                {
                    "lineage": lineage,
                    "encoder": p.encoder,
                    "cohort": cohort,
                    "patient_id": patient_id,
                    "primary_label": int(p.label),
                    "metastatic_label": int(m.label),
                    "primary_source_type": p.source_type,
                    "metastatic_source_type": m.source_type,
                    "primary_score": float(p.score),
                    "metastatic_score": float(m.score),
                    "metastatic_minus_primary": float(m.score - p.score),
                }
            )
    return pd.DataFrame(rows)


def build_patient_embeddings(
    patient_metadata: pd.DataFrame, slide_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray]:
    archive = np.load(SLIDE_MEANS)
    slide_ids = archive["slide_ids"].astype(str)
    means = archive["means"].astype(np.float32)
    if len(slide_ids) != len(means) or len(set(slide_ids)) != len(slide_ids):
        raise ValueError("Invalid frozen slide-mean archive")
    coordinates = pd.read_parquet(COORDINATES)
    required_coordinate_fields = {
        "slide_id",
        "cohort",
        "encoder",
        "n_patches",
        "umap_1",
        "umap_2",
    }
    if set(coordinates.columns) != required_coordinate_fields:
        raise ValueError("Unexpected frozen coordinate schema")
    if coordinates["slide_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate slide IDs in frozen coordinates")
    if set(coordinates["slide_id"].astype(str)) != set(slide_ids):
        raise ValueError("Frozen coordinates and slide-mean archive have different slides")
    if not coordinates["encoder"].eq("univ1").all():
        raise ValueError("Frozen coordinates include a non-UNI-v1 encoder")
    coordinate_numeric = coordinates[["n_patches", "umap_1", "umap_2"]].to_numpy(dtype=float)
    if not np.isfinite(coordinate_numeric).all() or (coordinates["n_patches"] <= 0).any():
        raise ValueError("Invalid numeric values in frozen coordinates")
    index = {slide_id: idx for idx, slide_id in enumerate(slide_ids)}
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for (cohort, role, patient_id), group in slide_metadata.groupby(
        ["analysis_cohort", "analysis_role", "patient_id"], sort=True
    ):
        missing = [slide for slide in group["slide_id"].astype(str) if slide not in index]
        if missing:
            raise ValueError(f"Frozen slide means missing roster slides: {missing[:5]}")
        vector = np.mean([means[index[str(slide)]] for slide in group["slide_id"]], axis=0)
        rows.append({"cohort": cohort, "role": role, "patient_id": str(patient_id)})
        vectors.append(vector)
    keys = pd.DataFrame(rows)
    keys = keys.merge(
        patient_metadata,
        on=["cohort", "role", "patient_id"],
        validate="one_to_one",
    )
    matrix = np.stack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("Patient embedding matrix contains non-finite values")
    return keys, matrix


def _repeated_embedding_cv(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    repeats: int,
    seed_key: str,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=int)
    counts = np.bincount(target, minlength=2)
    if min(counts) < 4:
        return {
            "n": int(len(target)),
            "n_positive": int(counts[1]),
            "n_negative": int(counts[0]),
            "estimable": False,
        }
    n_splits = min(5, int(min(counts)))
    aucs: list[float] = []
    balanced: list[float] = []
    base_seed = _stable_seed(RNG_SEED, "embedding_cv", seed_key)
    for repeat in range(repeats):
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=(base_seed + repeat) % (2**32 - 1),
        )
        oof = np.full(len(target), np.nan)
        for train, test in splitter.split(matrix, target):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.01,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=RNG_SEED,
                ),
            )
            model.fit(matrix[train], target[train])
            oof[test] = model.predict_proba(matrix[test])[:, 1]
        aucs.append(float(roc_auc_score(target, oof)))
        balanced.append(float(balanced_accuracy_score(target, oof >= 0.5)))
    return {
        "n": int(len(target)),
        "n_positive": int(counts[1]),
        "n_negative": int(counts[0]),
        "estimable": True,
        "cv_folds": n_splits,
        "cv_repeats": repeats,
        "feature_dimensions": int(matrix.shape[1]),
        "preprocessing_scope": "standardization_fit_within_each_training_fold",
        "classifier_scope": "repeated_patient_level_stratified_cross_validation",
        "regularization": "L2_logistic_C_0.01",
        "auroc_median": float(np.median(aucs)),
        "auroc_split_p025": float(np.quantile(aucs, 0.025)),
        "auroc_split_p975": float(np.quantile(aucs, 0.975)),
        "balanced_accuracy_median": float(np.median(balanced)),
        "split_range_not_confidence_interval": True,
    }


def structural_cv_table(keys: pd.DataFrame, matrix: np.ndarray, repeats: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        mask: pd.Series,
        cohort: str,
        task: str,
        stratum: str,
        target: pd.Series,
        *,
        overlap_removed: int = 0,
    ) -> None:
        indices = np.flatnonzero(mask.to_numpy())
        block = _repeated_embedding_cv(
            matrix[indices],
            target.loc[mask].to_numpy(dtype=int),
            repeats=repeats,
            seed_key=f"{cohort}/{task}/{stratum}",
        )
        rows.append(
            {
                "cohort": cohort,
                "task": task,
                "stratum": stratum,
                "role_overlap_removed": overlap_removed,
                "evidence_state": "DIRECT_HASHED_AUDIT_ONLY_UNGOVERNED_SLIDE_MEANS",
                **block,
            }
        )

    for cohort in ("RIH", "SR1482"):
        cohort_mask = keys["cohort"].eq(cohort)
        primary_ids = set(keys.loc[cohort_mask & keys["role"].eq("primary"), "patient_id"])
        metastatic_ids = set(keys.loc[cohort_mask & keys["role"].eq("metastatic"), "patient_id"])
        shared_role_ids = primary_ids & metastatic_ids
        disjoint_role_mask = cohort_mask & ~keys["patient_id"].isin(shared_role_ids)
        for role in ("primary", "metastatic"):
            mask = (
                cohort_mask
                & keys["role"].eq(role)
                & keys["source_type"].isin(["biopsy", "resection"])
            )
            add(
                mask,
                cohort,
                "source_type_biopsy_vs_resection",
                role,
                keys["source_type"].eq("biopsy").astype(int),
            )
        for source in ("all", "biopsy", "resection"):
            mask = disjoint_role_mask.copy()
            if source != "all":
                mask &= keys["source_type"].eq(source)
            add(
                mask,
                cohort,
                "specimen_role_metastatic_vs_primary",
                source,
                keys["role"].eq("metastatic").astype(int),
                overlap_removed=len(shared_role_ids),
            )
        met_mask = cohort_mask & keys["role"].eq("metastatic")
        add(
            met_mask,
            cohort,
            "organ_liver_vs_non_liver",
            "all",
            keys["site_binary"].eq("liver").astype(int),
        )
        for source in ("biopsy", "resection"):
            mask = met_mask & keys["source_type"].eq(source)
            add(
                mask,
                cohort,
                "organ_liver_vs_non_liver",
                source,
                keys["site_binary"].eq("liver").astype(int),
            )
    return pd.DataFrame(rows)


def population_table(metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cohort, role, source), frame in metadata.groupby(
        ["cohort", "role", "source_type"], sort=True
    ):
        rows.append(
            {
                "cohort": cohort,
                "role": role,
                "source_type": source,
                "n": len(frame),
                "n_mutant": int(frame["label"].sum()),
                "liver_n": int(frame["site_binary"].eq("liver").sum()),
                "set_d_n": int(frame["set_d"].sum()),
                "age_mean": float(frame["age"].mean()),
                "sex_known_n": int(frame["sex"].ne("missing").sum()),
            }
        )
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_interval(point: Any, low: Any, high: Any, digits: int = 3) -> str:
    values = (point, low, high)
    unavailable = any(
        value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        for value in values
    )
    if unavailable:
        if point is None or (isinstance(point, (float, np.floating)) and not np.isfinite(point)):
            return "interval not estimable"
        return f"{_fmt(point, digits)} (interval not estimable)"
    return f"{_fmt(point, digits)} [{_fmt(low, digits)}, {_fmt(high, digits)}]"


def _metric_lookup(
    metrics: pd.DataFrame,
    lineage: str,
    cohort: str,
    role: str,
    dimension: str,
    level: str,
) -> pd.Series:
    row = metrics[
        metrics["lineage"].eq(lineage)
        & metrics["cohort"].eq(cohort)
        & metrics["role"].eq(role)
        & metrics["dimension"].eq(dimension)
        & metrics["level"].eq(level)
    ]
    if len(row) != 1:
        raise ValueError(f"No unique metric row: {lineage}/{cohort}/{role}/{dimension}/{level}")
    return row.iloc[0]


def _contrast_lookup(contrasts: pd.DataFrame, lineage: str, cohort: str, name: str) -> pd.Series:
    row = contrasts[
        contrasts["lineage"].eq(lineage)
        & contrasts["cohort"].eq(cohort)
        & contrasts["contrast"].eq(name)
    ]
    if len(row) != 1:
        raise ValueError(f"No unique contrast: {lineage}/{cohort}/{name}")
    return row.iloc[0]


def make_figure(
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    decomposition: pd.DataFrame,
    structural: pd.DataFrame,
    path_png: Path,
    path_pdf: Path,
) -> None:
    family = "family_naive_univ1_5seed"
    colors = {"primary": "#2878b5", "metastatic": "#d95f02"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    x = 0
    ticks = []
    labels = []
    for cohort in ("RIH", "SR1482"):
        for source in ("all", "biopsy", "resection", "unknown"):
            if source == "unknown" and cohort == "RIH":
                continue
            for role, offset in (("primary", -0.12), ("metastatic", 0.12)):
                dimension = "overall" if source == "all" else "source_type"
                level = "all" if source == "all" else source
                row = _metric_lookup(metrics, family, cohort, role, dimension, level)
                ax.errorbar(
                    x + offset,
                    row.auroc,
                    yerr=[[row.auroc - row.auroc_ci_low], [row.auroc_ci_high - row.auroc]],
                    fmt="o",
                    color=colors[role],
                    capsize=2,
                    markersize=5,
                )
            ticks.append(x)
            labels.append(f"{cohort}\n{source}")
            x += 1
        x += 0.35
    ax.axhline(0.5, color="#777777", lw=1, ls="--")
    ax.set_xticks(ticks, labels, rotation=35, ha="right")
    ax.set_ylim(0.25, 1.02)
    ax.set_ylabel("AUROC")
    ax.set_title("A. Family-naive ranking by source type")
    ax.scatter([], [], color=colors["primary"], label="Primary")
    ax.scatter([], [], color=colors["metastatic"], label="Metastatic")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[0, 1]
    contrast_names = [
        "metastatic_biopsy_minus_resection",
        "metastatic_liver_minus_non_liver",
    ]
    lineages = [
        "family_naive_univ1_5seed",
        "tcga_surgen_univ1_5seed",
        "tcga_surgen_virchow2_cls_5seed",
    ]
    y = 0
    ticks = []
    labels = []
    palette = ["#355c7d", "#6c5b7b", "#c06c84"]
    for cohort in ("RIH", "SR1482"):
        for name in contrast_names:
            for lineage, color in zip(lineages, palette, strict=True):
                row = _contrast_lookup(contrasts, lineage, cohort, name)
                ax.errorbar(
                    row.delta_auroc,
                    y,
                    xerr=[[row.delta_auroc - row.ci_low], [row.ci_high - row.delta_auroc]],
                    fmt="o",
                    color=color,
                    capsize=2,
                    markersize=5,
                )
                y += 0.22
            ticks.append(y - 0.22)
            labels.append(
                f"{cohort}: " + ("biopsy−resection" if "biopsy" in name else "liver−non-liver")
            )
            y += 0.28
    ax.axvline(0, color="#777777", lw=1, ls="--")
    ax.set_yticks(ticks, labels)
    ax.set_xlim(-0.75, 0.75)
    ax.set_xlabel("Difference in metastatic AUROC")
    ax.set_title("B. Organ and source-type contrasts")
    for lineage, color in zip(lineages, palette, strict=True):
        label = {
            "family_naive_univ1_5seed": "Family-naive UNI",
            "tcga_surgen_univ1_5seed": "TCGA+SurGen UNI",
            "tcga_surgen_virchow2_cls_5seed": "TCGA+SurGen Virchow2",
        }[lineage]
        ax.scatter([], [], color=color, label=label)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[1, 0]
    labels = []
    observed = []
    standardized = []
    for cohort in ("RIH", "SR1482"):
        for analysis in ("source_type", "mutant_subtype"):
            row = decomposition[
                decomposition["cohort"].eq(cohort) & decomposition["analysis"].eq(analysis)
            ].iloc[0]
            labels.append(f"{cohort}\n{analysis.replace('_', ' ')}")
            observed.append(row.observed_difference)
            standardized.append(row.standardized_difference)
    positions = np.arange(len(labels))
    width = 0.36
    ax.bar(positions - width / 2, observed, width, color="#777777", label="Observed")
    ax.bar(
        positions + width / 2,
        standardized,
        width,
        color="#4c9f70",
        label="Composition-standardized",
    )
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Metastatic minus primary AUROC")
    ax.set_title("C. Observed and composition-standardized point differences")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    selected = structural[
        (
            (structural["task"].eq("source_type_biopsy_vs_resection"))
            & structural["stratum"].eq("metastatic")
        )
        | (
            (structural["task"].eq("specimen_role_metastatic_vs_primary"))
            & structural["stratum"].isin(["biopsy", "resection"])
        )
        | ((structural["task"].eq("organ_liver_vs_non_liver")) & structural["stratum"].eq("all"))
    ].copy()
    selected = selected[selected["estimable"].eq(True)]
    selected["label"] = selected.apply(
        lambda row: (
            f"{row.cohort}: source type"
            if row.task == "source_type_biopsy_vs_resection"
            else (
                f"{row.cohort}: role within {row.stratum}"
                if row.task == "specimen_role_metastatic_vs_primary"
                else f"{row.cohort}: liver"
            )
        ),
        axis=1,
    )
    selected = selected.sort_values(["cohort", "task", "stratum"])
    positions = np.arange(len(selected))
    ax.errorbar(
        selected["auroc_median"],
        positions,
        xerr=[
            selected["auroc_median"] - selected["auroc_split_p025"],
            selected["auroc_split_p975"] - selected["auroc_median"],
        ],
        fmt="o",
        color="#5b4b8a",
        capsize=2,
    )
    ax.axvline(0.5, color="#777777", lw=1, ls="--")
    ax.set_yticks(positions, selected["label"])
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("Cross-fitted averaged-feature classifier AUROC")
    ax.set_title("D. Audit-only averaged UNI-v1 feature decodability")

    fig.suptitle(
        "CRC metastatic-transfer diagnostics using user-supplied source-type labels",
        fontsize=14,
    )
    fig.savefig(path_png, dpi=220)
    fig.savefig(path_pdf)
    plt.close(fig)


def _legacy_build_report(
    validation: dict[str, Any],
    population: pd.DataFrame,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    decomposition: pd.DataFrame,
    site_source: pd.DataFrame,
    alignment: pd.DataFrame,
    structural: pd.DataFrame,
    paired: pd.DataFrame,
) -> str:
    family = "family_naive_univ1_5seed"
    lines = [
        "# CRC FINAL-v6 metastatic-transfer association diagnostics",
        "",
        "## Bottom line",
        "",
        "This is an unused historical formatter retained only for code-lineage review. "
        "It must not be used to generate a scientific report.",
        "",
        "The current report is generated only by `build_report`; this historical formatter "
        "has no publication authority.",
        "",
        "## Evidence status and inputs",
        "",
        f"- `crc_final_v6.csv` has {validation['rows']:,} rows and "
        f"{validation['columns']} columns. All 62 v5 columns are cell-identical; only "
        "`source_type` was appended.",
        "- The primary estimator is the controlling five-seed family-naive UNI-v1 score: "
        "RIH-family-held-out for RIH and whole-SurGen-held-out for SR1482.",
        "- TCGA+SurGen UNI-v1 and Virchow2-CLS five-seed results are correlated sensitivities. "
        "RIH primary/metastatic use the same target refits; the SR1482 comparison is "
        "explicitly nonconformant (source OOF primary versus source-exposed metastatic refit).",
        "- All role gaps remove patients represented in both roles. Bootstrap units are "
        "patients, never seeds, slides, or folds.",
        f"- Provenance boundary: {validation['provenance_boundary']}",
        "",
        "## Population and overlap",
        "",
        "This table shows the full frozen target rosters. Population, source-only, and "
        "site-only rows retain all governed metastatic patients; only cross-role estimands "
        "remove patients represented in both roles.",
        "",
        "| Cohort | Role | Source type | Patients | KRAS mutant | Liver | Set D |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in population.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.role} | {row.source_type} | {row.n} | "
            f"{row.n_mutant} | {row.liver_n} | {row.set_d_n} |"
        )

    lines += [
        "",
        "RIH shows the sharpest composition change: primary tissue is mostly resection, "
        "whereas metastatic tissue is mostly biopsy. SR1482-M also mixes biopsy, "
        "resection, and an explicitly retained unknown group. Liver and source type are "
        "therefore not exchangeable exposures.",
        "",
        "## 1. Metastatic-site association",
        "",
        "| Score view | Cohort | Liver AUROC [95% CI] | Non-liver AUROC [95% CI] | Difference [95% CI] | Source-standardized difference [95% CI] | Support |",
        "|---|---|---|---|---|---|---|",
    ]
    for lineage in (
        family,
        "tcga_surgen_univ1_5seed",
        "tcga_surgen_virchow2_cls_5seed",
    ):
        for cohort in ("RIH", "SR1482"):
            liver = _metric_lookup(metrics, lineage, cohort, "metastatic", "site_binary", "liver")
            non = _metric_lookup(metrics, lineage, cohort, "metastatic", "site_binary", "non_liver")
            delta = _contrast_lookup(contrasts, lineage, cohort, "metastatic_liver_minus_non_liver")
            adjusted = site_source[
                site_source["lineage"].eq(lineage) & site_source["cohort"].eq(cohort)
            ].iloc[0]
            support = (
                "inferential marginal cells"
                if bool(liver.inferential_support and non.inferential_support)
                else "descriptive marginal cells"
            )
            lines.append(
                f"| {lineage} | {cohort} | {_fmt(liver.auroc)} "
                f"[{_fmt(liver.auroc_ci_low)}, {_fmt(liver.auroc_ci_high)}] | "
                f"{_fmt(non.auroc)} [{_fmt(non.auroc_ci_low)}, {_fmt(non.auroc_ci_high)}] | "
                f"{_fmt(delta.delta_auroc)} [{_fmt(delta.ci_low)}, {_fmt(delta.ci_high)}] | "
                f"{_fmt(adjusted.source_standardized_liver_minus_non_liver)} "
                f"[{_fmt(adjusted.standardized_ci_low)}, "
                f"{_fmt(adjusted.standardized_ci_high)}] | {support}; "
                f"crossed cells {str(adjusted.evidence_state).lower()} |"
            )
    lines += [
        "",
        "A causal liver explanation would require a reasonably consistent adverse liver "
        "effect. Instead, the direction is cohort-dependent; SR1482 liver metastases rank "
        "substantially better than its non-liver mixture, while RIH does not reproduce that "
        "pattern. Small organ-by-procedure cells remain descriptive.",
        "",
        "## 2. Source-type association",
        "",
        "| Score view | Cohort | Biopsy AUROC [95% CI] | Resection AUROC [95% CI] | Biopsy minus resection [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for lineage in (
        family,
        "tcga_surgen_univ1_5seed",
        "tcga_surgen_virchow2_cls_5seed",
    ):
        for cohort in ("RIH", "SR1482"):
            biopsy = _metric_lookup(metrics, lineage, cohort, "metastatic", "source_type", "biopsy")
            resection = _metric_lookup(
                metrics, lineage, cohort, "metastatic", "source_type", "resection"
            )
            delta = _contrast_lookup(
                contrasts, lineage, cohort, "metastatic_biopsy_minus_resection"
            )
            lines.append(
                f"| {lineage} | {cohort} | {_fmt(biopsy.auroc)} "
                f"[{_fmt(biopsy.auroc_ci_low)}, {_fmt(biopsy.auroc_ci_high)}] | "
                f"{_fmt(resection.auroc)} [{_fmt(resection.auroc_ci_low)}, "
                f"{_fmt(resection.auroc_ci_high)}] | {_fmt(delta.delta_auroc)} "
                f"[{_fmt(delta.ci_low)}, {_fmt(delta.ci_high)}] |"
            )

    lines += [
        "",
        "The RIH point estimates consistently make biopsy look harder than resection, but "
        "the resection arm has only 18 patients and intervals are wide. SR1482 does not "
        "show a comparable biopsy penalty. Every biopsy-versus-resection contrast is "
        "descriptive because at least one metastatic resection KRAS class has fewer than "
        "10 patients. Pairwise source-cell results in "
        "`pairwise_source_cells.csv` distinguish within-type ranking from cross-type score "
        "offsets.",
        "",
        "### Source-composition decomposition",
        "",
        "| Cohort | Observed metastatic−primary AUROC | Within-source standardized gap | Marginal source-standardized gap | Composition point component | Mutant-subtype-standardized gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cohort in ("RIH", "SR1482"):
        source = decomposition[
            decomposition["cohort"].eq(cohort) & decomposition["analysis"].eq("source_type")
        ].iloc[0]
        subtype = decomposition[
            decomposition["cohort"].eq(cohort) & decomposition["analysis"].eq("mutant_subtype")
        ].iloc[0]
        lines.append(
            f"| {cohort} | {_fmt(source.observed_gap)} | "
            f"{_fmt(source.within_source_standardized_gap)} | "
            f"{_fmt(source.standardized_gap)} | {_fmt(source.composition_contribution)} | "
            f"{_fmt(subtype.standardized_gap)} |"
        )

    lines += [
        "",
        "The within-source estimator averages biopsy-only, resection-only, and—where "
        "present—unknown-only AUROCs at one common patient distribution; it removes all "
        "cross-source score comparisons. The marginal estimator uses a common, "
        "label-conditional source distribution and retains every positive-source × "
        "negative-source concordance cell. Their agreement or disagreement separates "
        "within-procedure ranking from cross-procedure score offsets. The composition point "
        "component is descriptive, not a causal proportion explained.",
        "",
        "### Label-blind source-offset alignment",
        "",
        "| Score view | Cohort | Native metastatic AUROC | Source-aligned AUROC | Change [95% CI] |",
        "|---|---|---:|---:|---|",
    ]
    for row in alignment.itertuples(index=False):
        lines.append(
            f"| {row.lineage} | {row.cohort} | {_fmt(row.native_metastatic_auroc)} | "
            f"{_fmt(row.source_aligned_metastatic_auroc)} | "
            f"{_fmt(row.aligned_minus_native)} [{_fmt(row.ci_low)}, {_fmt(row.ci_high)}] |"
        )
    lines += [
        "",
        "This alignment estimates biopsy/resection score offsets from disjoint primary "
        "patients without using metastatic labels. A failure to restore AUROC means that "
        "a simple source-type intercept shift is insufficient; it does not rule out more "
        "complex acquisition effects.",
        "",
        "## 3. Is molecular or KRAS-subtype case mix the cause?",
        "",
        "The mutant-subtype decomposition standardizes G12D, G12V, G13D, and other-mutant "
        "weights across roles. The source table also reports the disjoint Set-D "
        "(MSS/pMMR and BRAF-wild-type) role contrast. Neither sensitivity identifies a "
        "causal molecular explanation. This does not test "
        "unmeasured treatment-driven evolution, tumor purity, clonality, or assay error.",
        "",
        "| Score view | Cohort | Set-D primary n / AUROC | Set-D metastatic n / AUROC | Metastatic−primary [95% CI] | Support |",
        "|---|---|---|---|---|---|",
    ]
    for lineage in (
        family,
        "tcga_surgen_univ1_5seed",
        "tcga_surgen_virchow2_cls_5seed",
    ):
        for cohort in ("RIH", "SR1482"):
            row = _contrast_lookup(contrasts, lineage, cohort, "metastatic_minus_primary/set_d")
            lines.append(
                f"| {lineage} | {cohort} | {int(row.second_n)} / "
                f"{_fmt(row.second_auroc)} | {int(row.first_n)} / "
                f"{_fmt(row.first_auroc)} | {_fmt(row.delta_auroc)} "
                f"[{_fmt(row.ci_low)}, {_fmt(row.ci_high)}] | "
                f"{'inferential cells' if row.strict_support else 'descriptive sparse cells'} |"
            )
    lines += [
        "",
        "Set D is a complete-case molecular sensitivity; molecular completeness is lower "
        "in metastases, particularly RIH. It therefore cannot rule out effects carried by "
        "the missing molecular records.",
        "",
        "## 4. Is there an overall structural shift?",
        "",
        "Frozen UNI-v1 slide means were averaged within the exact patient rosters. A "
        "label-blind PCA-20 projection was fitted once in each complete task population; a "
        "class-balanced logistic classifier was then evaluated with repeated stratified "
        "cross-validation. This is a transductive separability diagnostic, not external "
        "classifier validation. The 2.5th–97.5th percentiles below are split "
        "sensitivity ranges, not confidence intervals.",
        "",
        "| Cohort | Structural task | Stratum | n | CV AUROC median [split range] |",
        "|---|---|---|---:|---|",
    ]
    for row in structural.itertuples(index=False):
        if not row.estimable:
            lines.append(f"| {row.cohort} | {row.task} | {row.stratum} | {row.n} | NOT_ESTIMABLE |")
        else:
            lines.append(
                f"| {row.cohort} | {row.task} | {row.stratum} | {row.n} | "
                f"{_fmt(row.auroc_median)} [{_fmt(row.auroc_split_p025)}, "
                f"{_fmt(row.auroc_split_p975)}] |"
            )

    lines += [
        "",
        "Strong source-type separability establishes a global specimen/preparation axis. "
        "Role separability that remains within biopsy-only or resection-only strata shows "
        "that the binary procedure label is not sufficient to describe the embedding "
        "difference. It cannot distinguish tumor biology, organ context, case mix, or "
        "unmeasured technical differences. Organ decodability likewise describes tissue "
        "context, not causation.",
        "",
        "## 5. Paired-patient sensitivity",
        "",
    ]
    if paired.empty:
        lines.append("No exact primary/metastatic patient overlaps were available.")
    else:
        summary = paired.groupby(["lineage", "cohort"]).agg(
            n=("patient_id", "size"),
            label_discordant=(
                "primary_label",
                lambda values: int(
                    np.sum(
                        values.to_numpy() != paired.loc[values.index, "metastatic_label"].to_numpy()
                    )
                ),
            ),
            mean_shift=("metastatic_minus_primary", "mean"),
            median_shift=("metastatic_minus_primary", "median"),
        )
        lines += [
            "| Score view | Cohort | Pairs | Label-discordant | Mean logit shift | Median logit shift |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for index, row in summary.iterrows():
            lines.append(
                f"| {index[0]} | {index[1]} | {int(row.n)} | "
                f"{int(row.label_discordant)} | {_fmt(row.mean_shift)} | "
                f"{_fmt(row.median_shift)} |"
            )

    lines += [
        "",
        "## Integrated interpretation",
        "",
        "1. **Metastatic site.** The large SR1482 contrast was not reproduced in RIH.",
        "2. **Source type.** Descriptive contrasts do not establish a replicated effect.",
        "3. **Recorded gene context.** Set-D restriction and KRAS-subtype standardization "
        "are post-outcome sensitivities. Recorded KRAS is the "
        "endpoint; this analysis cannot measure clonal fraction, viability, treatment "
        "selection, or spatial heterogeneity.",
        "4. **The remaining association is unresolved.** Primary and metastatic "
        "embeddings can remain separable even within a procedure stratum, but that "
        "separability is still confounded by organ, case mix, and unmeasured technical "
        "factors. Together with the previously governed failure of metastatic-only training "
        "and target-internal residual adaptation to reproducibly improve ranking, this "
        "does not localize a technical or biological mechanism.",
        "",
        "## Limits",
        "",
        "- This analysis is post hoc and outcome-aware; no p-value is presented as "
        "preregistered confirmation.",
        "- RIH-M has only 18 resections. SR1482-M has 24 patients with unknown source type. "
        "Many organ-by-procedure cells are too small for stable inference.",
        "- `source_type` captures biopsy versus resection, not tumor fraction, necrosis, "
        "treatment, fixation, grade, block age, or tissue area.",
        "- AUROC is pairwise ranking. Composition can change pooled AUROC through "
        "positive-source × negative-source score offsets even when within-type AUROCs are "
        "unchanged.",
        "- The SR1482 two-encoder role comparison is a sensitivity only because its primary "
        "and metastatic scores come from different evaluation modes.",
        "- Results diagnose the frozen models and cohorts; they do not establish a universal "
        "metastatic mechanism or clinical utility.",
        "",
        "## Artifact guide",
        "",
        "- `performance_strata.csv`: all overall, source, organ, and crossed performance cells.",
        "- `contrasts.csv`: paired-bootstrap AUROC contrasts and disjoint role gaps.",
        "- `pairwise_source_cells.csv`: positive-source × negative-source concordance cells.",
        "- `decomposition.csv`: source-type and KRAS-subtype standardization.",
        "- `site_source_standardization.csv`: liver/non-liver contrasts at common source composition.",
        "- `source_alignment.csv`: label-blind primary-derived source-offset sensitivity.",
        "- `structural_cv.csv`: repeated-CV high-dimensional domain separability.",
        "- `paired_role_scores.csv`: deidentified paired-patient score transitions.",
        "- `analysis_table.parquet`: the patient-level score/covariate table.",
        "- `patient_embeddings.parquet`: embedding keys plus 1,024 frozen dimensions.",
        "- `met_transfer_diagnostics.png` and `.pdf`: integrated four-panel figure.",
        "- `receipt.json`: exact input/output identities and execution settings.",
        "",
    ]
    return "\n".join(lines)


def build_report(
    validation: dict[str, Any],
    population: pd.DataFrame,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    decomposition: pd.DataFrame,
    site_source: pd.DataFrame,
    case_mix: pd.DataFrame,
    score_structure: pd.DataFrame,
    alignment: pd.DataFrame,
    structural: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    n_bootstrap: int,
    cv_repeats: int,
) -> str:
    family = "family_naive_univ1_5seed"
    smoke = n_bootstrap < DEFAULT_BOOTSTRAPS or cv_repeats < DEFAULT_CV_REPEATS
    lines = [
        "# CRC FINAL-v6 post-outcome metastatic-transfer association diagnostics",
        "",
    ]
    if smoke:
        lines += [
            "> **SMOKE TEST ONLY — NOT FOR SCIENTIFIC INTERPRETATION.** "
            f"This run used {n_bootstrap:,} bootstrap draws and {cv_repeats} CV repeats; "
            "its intervals and split ranges are pipeline checks.",
            "",
        ]
    lines += [
        "## Bottom line",
        "",
        "Across these post-outcome frozen-score diagnostics, SR1482 showed large "
        "positive liver-versus-non-liver point contrasts, whereas RIH contrasts were "
        "near zero. An audit-only panel found source type highly decodable from averaged "
        "UNI-v1 features, "
        "but biopsy–resection AUROC contrasts and source-composition components were "
        "imprecise and did not establish an effect on transfer. MSS/BRAF-WT restriction "
        "and coarse KRAS-subtype reweighting did not consistently attenuate the observed "
        "point differences. The remaining associations are unresolved and compatible "
        "with technical, tissue-context, clinical, or biological differences.",
        "",
        "This analysis evaluates whether a limited set of measured factors changes "
        "observed ranking point estimates under specified stratifications and "
        "data-dependent standardizations. These are descriptive decompositions, not "
        "causal fractions explained, and the analysis was not retrospectively preregistered.",
        "",
        "## Evidence status and governed context",
        "",
        "- The primary frozen score view uses the five-seed family-naive UNI-v1 ensemble. "
        "That scorer governs E2-MET, but every new FINAL-v6 stratification and decomposition "
        "here is post-outcome exploratory evidence.",
        "- The governed five-seed E2-MET result passed its narrow macro gate: RIH-M AUROC "
        "0.609, SR1482-M 0.587, equal-cohort macro 0.598 [0.505, 0.690]. Thus this report "
        "does not relabel E2-MET as a categorical failure; it investigates modest target "
        "ranking and lower role-comparison point estimates.",
        "- TCGA+SurGen UNI-v1 and Virchow2-CLS are correlated secondary sensitivity views "
        "on the same target patients, not independent replications. SR1482 primary OOF "
        "and metastatic target-refit logits have noncomparable marginal scales.",
        "- RIH primary/metastatic role comparisons remove the eight patients appearing in "
        "both roles from both arms. Patients are the analysis and bootstrap units; seeds, "
        "folds, and slides are not independent replicates.",
        f"- `crc_final_v6` has {validation['rows']:,} rows and {validation['columns']} "
        "columns. Its 62 inherited columns are cell-identical to FINAL-v5; `source_type` "
        "is the only appended field, and CSV/XLSX cells agree.",
        f"- Provenance boundary: {validation['provenance_boundary']}",
        "- The averaged UNI-v1 slide-mean features are directly hashed for this diagnostic "
        "but are not governed sources in FINAL-v12; their feature-decoding panel is audit-only.",
        "- All intervals are post-outcome and unadjusted across multiple correlated views. "
        "No interval is confirmatory.",
        f"- Overall/stratum and direct contrast intervals use {n_bootstrap:,} "
        "patient-bootstrap draws. Data-dependent composition, ATT, and centering "
        f"sensitivities use {min(n_bootstrap, SECONDARY_BOOTSTRAPS):,} refitted draws; "
        f"feature decodability uses {cv_repeats} repeated patient-level CV splits.",
        "",
        "## Population and overlap",
        "",
        "This table shows the full frozen target rosters. Population, source-only, and "
        "site-only rows retain all governed metastatic patients; only cross-role estimands "
        "remove patients represented in both roles.",
        "",
        "| Cohort | Role | Source type | Patients | KRAS mutant | Liver | Set D |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in population.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.role} | {row.source_type} | {row.n} | "
            f"{row.n_mutant} | {row.liver_n} | {row.set_d_n} |"
        )
    lines += [
        "",
        "RIH changes from predominantly resection in primary tissue to predominantly "
        "biopsy in metastatic tissue. SR1482-M mixes biopsy, resection, and an explicitly "
        "retained unknown group. Liver site and source type are imbalanced, and many "
        "crossed source-by-site cells are sparse; marginal contrasts do not isolate one "
        "factor independently of the other.",
        "",
        "### Disjoint primary-versus-metastatic ranking",
        "",
        "| Cohort | Population | n (mutant) | AUROC | Metastatic−primary [95% CI] |",
        "|---|---|---:|---:|---|",
    ]
    for cohort in ("RIH", "SR1482"):
        contrast = _contrast_lookup(contrasts, family, cohort, "metastatic_minus_primary/all")
        lines += [
            f"| {cohort} | Primary | {int(contrast.second_n)} "
            f"({int(contrast.second_n_mutant)}) | {_fmt(contrast.second_auroc)} | — |",
            f"| {cohort} | Metastatic | {int(contrast.first_n)} "
            f"({int(contrast.first_n_mutant)}) | {_fmt(contrast.first_auroc)} | "
            f"{_fmt(contrast.delta_auroc)} [{_fmt(contrast.ci_low)}, "
            f"{_fmt(contrast.ci_high)}] |",
        ]
    lines += [
        "",
        "Both role-comparison intervals include zero. These are observed AUROC point "
        "differences, not established transport gaps.",
        "",
        "## 1. How do ranking estimates vary by metastatic site?",
        "",
        "| Score view | Cohort | Liver AUROC [95% CI] | Non-liver AUROC [95% CI] | Liver−non-liver [95% CI] | Source-standardized point contrast | Crossed-cell state |",
        "|---|---|---|---|---|---|---|",
    ]
    for lineage in (
        family,
        "tcga_surgen_univ1_5seed",
        "tcga_surgen_virchow2_cls_5seed",
    ):
        for cohort in ("RIH", "SR1482"):
            liver = _metric_lookup(metrics, lineage, cohort, "metastatic", "site_binary", "liver")
            non_liver = _metric_lookup(
                metrics, lineage, cohort, "metastatic", "site_binary", "non_liver"
            )
            contrast = _contrast_lookup(
                contrasts, lineage, cohort, "metastatic_liver_minus_non_liver"
            )
            adjusted = site_source[
                site_source["lineage"].eq(lineage) & site_source["cohort"].eq(cohort)
            ].iloc[0]
            lines.append(
                f"| {lineage} | {cohort} | {_fmt(liver.auroc)} "
                f"[{_fmt(liver.auroc_ci_low)}, {_fmt(liver.auroc_ci_high)}] | "
                f"{_fmt(non_liver.auroc)} [{_fmt(non_liver.auroc_ci_low)}, "
                f"{_fmt(non_liver.auroc_ci_high)}] | {_fmt(contrast.delta_auroc)} "
                f"[{_fmt(contrast.ci_low)}, {_fmt(contrast.ci_high)}] | "
                f"{_fmt_interval(adjusted.source_standardized_liver_minus_non_liver, adjusted.standardized_ci_low, adjusted.standardized_ci_high)} | "
                f"{adjusted.evidence_state} |"
            )
    lines += [
        "",
        "Across score views, SR1482 liver-minus-non-liver point contrasts ranged from "
        "+0.209 to +0.444, whereas RIH contrasts ranged from +0.007 to +0.069. The large "
        "SR1482 contrast was therefore not reproduced in RIH. Every source-standardized "
        "site analysis fails the strict crossed-cell count gate; those adjusted numbers "
        "remain descriptive. Sparse cells, post-outcome analysis, and unadjusted "
        "multiplicity preclude a liver-effect claim.",
        "",
        "Fine liver/lung/peritoneum/other cells are retained in `performance_strata.csv`; "
        "they are exploratory and must not be used as independent validations.",
        "",
        "## 2. How do ranking estimates vary by source type?",
        "",
        "| Score view | Cohort | Biopsy AUROC [95% CI] | Resection AUROC [95% CI] | Biopsy−resection [95% CI] | Count state |",
        "|---|---|---|---|---|---|",
    ]
    for lineage in (
        family,
        "tcga_surgen_univ1_5seed",
        "tcga_surgen_virchow2_cls_5seed",
    ):
        for cohort in ("RIH", "SR1482"):
            biopsy = _metric_lookup(metrics, lineage, cohort, "metastatic", "source_type", "biopsy")
            resection = _metric_lookup(
                metrics, lineage, cohort, "metastatic", "source_type", "resection"
            )
            contrast = _contrast_lookup(
                contrasts, lineage, cohort, "metastatic_biopsy_minus_resection"
            )
            state = (
                "minimum cell criterion met"
                if contrast.strict_support
                else "descriptive: sparse role×label cell"
            )
            lines.append(
                f"| {lineage} | {cohort} | {_fmt(biopsy.auroc)} "
                f"[{_fmt(biopsy.auroc_ci_low)}, {_fmt(biopsy.auroc_ci_high)}] | "
                f"{_fmt(resection.auroc)} [{_fmt(resection.auroc_ci_low)}, "
                f"{_fmt(resection.auroc_ci_high)}] | {_fmt(contrast.delta_auroc)} "
                f"[{_fmt(contrast.ci_low)}, {_fmt(contrast.ci_high)}] | {state} |"
            )
    lines += [
        "",
        "RIH biopsy-minus-resection point contrasts were negative across the three "
        "correlated views, but estimates rely on only 18 metastatic resections and are "
        "post-outcome, unadjusted, and nonreplicated. SR1482 showed no comparable "
        "magnitude. These descriptive results do not establish a replicated source-type "
        "effect on KRAS ranking.",
        "",
        "### Source-composition standardizations",
        "",
        "| Cohort | Observed M−P [95% CI] | Within-source standardized M−P [95% CI] | Marginal source-standardized M−P [95% CI] | Descriptive composition component [95% CI] | State |",
        "|---|---|---|---|---|---|",
    ]
    for cohort in ("RIH", "SR1482"):
        row = decomposition[
            decomposition["cohort"].eq(cohort) & decomposition["analysis"].eq("source_type")
        ].iloc[0]
        lines.append(
            f"| {cohort} | {_fmt(row.observed_difference)} "
            f"[{_fmt(row.observed_ci_low)}, {_fmt(row.observed_ci_high)}] | "
            f"{_fmt(row.within_source_standardized_difference)} "
            f"[{_fmt(row.within_standardized_ci_low)}, "
            f"{_fmt(row.within_standardized_ci_high)}] | "
            f"{_fmt(row.standardized_difference)} [{_fmt(row.standardized_ci_low)}, "
            f"{_fmt(row.standardized_ci_high)}] | "
            f"{_fmt(row.descriptive_composition_point_component)} "
            f"[{_fmt(row.composition_point_component_ci_low)}, "
            f"{_fmt(row.composition_point_component_ci_high)}] | {row.evidence_state} |"
        )
    lines += [
        "",
        "The within-source summary excludes cross-source score comparisons; the marginal "
        "summary retains them at common label-conditional source weights. Target weights "
        "are re-estimated inside each patient bootstrap. Their difference describes "
        "sensitivity to this chosen reweighting and is neither a causal amount explained "
        "nor a percentage attribution. RIH's negative point difference became smaller, "
        "but intervals did not establish either a residual role difference or a "
        "source-composition component. SR1482 changed little.",
        "The displayed component is `observed M−P − standardized M−P`; when the observed "
        "difference is negative, a negative component denotes attenuation toward zero, "
        "not worsening.",
        "",
        "### Primary-derived marginal score-centering sensitivity",
        "",
        "| Score view | Cohort | Native metastatic AUROC | Centered AUROC | Change [95% CI] | State |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in alignment.itertuples(index=False):
        lines.append(
            f"| {row.lineage} | {row.cohort} | "
            f"{_fmt(row.native_metastatic_auroc)} | "
            f"{_fmt(row.source_aligned_metastatic_auroc)} | "
            f"{_fmt_interval(row.aligned_minus_native, row.ci_low, row.ci_high)} | "
            f"{row.evidence_state} |"
        )
    lines += [
        "",
        "Offsets were estimated from disjoint primary patients without metastatic labels, "
        "but the operation was selected and evaluated post-outcome. Marginal offsets can "
        "reflect molecular prevalence and case mix as well as preparation. The observed "
        "AUROC changed negligibly under this chosen centering operation; that result does "
        "not address other source-associated corrections. Noncomparable SR1482 OOF/refit "
        "score scales are explicitly not interpreted.",
        "",
        "## 3. How do recorded molecular restrictions and case mix affect point estimates?",
        "",
        "### MSS/pMMR and BRAF-wild-type complete-case restriction",
        "",
        "| Score view | Cohort | Primary n / AUROC | Metastatic n / AUROC | M−P [95% CI] | Count state |",
        "|---|---|---|---|---|---|",
    ]
    for lineage in (
        family,
        "tcga_surgen_univ1_5seed",
        "tcga_surgen_virchow2_cls_5seed",
    ):
        for cohort in ("RIH", "SR1482"):
            row = _contrast_lookup(contrasts, lineage, cohort, "metastatic_minus_primary/set_d")
            state = (
                "minimum count criterion met; post-hoc sensitivity"
                if row.strict_support
                else "descriptive sparse cells"
            )
            lines.append(
                f"| {lineage} | {cohort} | {int(row.second_n)} / "
                f"{_fmt(row.second_auroc)} | {int(row.first_n)} / "
                f"{_fmt(row.first_auroc)} | {_fmt(row.delta_auroc)} "
                f"[{_fmt(row.ci_low)}, {_fmt(row.ci_high)}] | {state} |"
            )
    lines += [
        "",
        "Set D is complete-case evidence; molecular completeness is lower in metastatic "
        "samples, especially RIH. The restriction does not isolate treatment-driven "
        "evolution, clonality, viable tumor fraction, or assay error.",
        "",
        "### Coarse KRAS-mutant subtype reweighting",
        "",
        "| Cohort | Observed M−P [95% CI] | Subtype-standardized M−P [95% CI] | Descriptive component [95% CI] | State |",
        "|---|---|---|---|---|",
    ]
    for cohort in ("RIH", "SR1482"):
        row = decomposition[
            decomposition["cohort"].eq(cohort) & decomposition["analysis"].eq("mutant_subtype")
        ].iloc[0]
        lines.append(
            f"| {cohort} | {_fmt(row.observed_difference)} "
            f"[{_fmt(row.observed_ci_low)}, {_fmt(row.observed_ci_high)}] | "
            f"{_fmt(row.standardized_difference)} [{_fmt(row.standardized_ci_low)}, "
            f"{_fmt(row.standardized_ci_high)}] | "
            f"{_fmt(row.descriptive_composition_point_component)} "
            f"[{_fmt(row.composition_point_component_ci_low)}, "
            f"{_fmt(row.composition_point_component_ci_high)}] | {row.evidence_state} |"
        )
    lines += [
        "",
        "The subtype grouping is G12D, G12V, G13D, and a heterogeneous "
        "other-or-unknown-mutant group. Sparse metastatic subtype cells prevent "
        "subtype-specific biological conclusions.",
        "",
        "### Overlap-gated, class-stratified odds-weighted covariate standardization to the metastatic distribution",
        "",
        "| Cohort | Covariate block | Weighted primary AUROC | Standardized M−P [95% CI] | Descriptive component [95% CI] | State |",
        "|---|---|---:|---|---|---|",
    ]
    for row in case_mix.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {row.block} | "
            f"{_fmt(row.primary_att_to_metastatic_auroc)} | "
            f"{_fmt_interval(row.att_standardized_difference, row.att_standardized_ci_low, row.att_standardized_ci_high)} | "
            f"{_fmt_interval(row.descriptive_case_mix_point_component, row.case_mix_component_ci_low, row.case_mix_component_ci_high)} | "
            f"{row.evidence_state} |"
        )
    reportable_case_mix = case_mix["evidence_state"].str.startswith("ESTIMABLE_")
    if not bool(reportable_case_mix.any()):
        lines += [
            "",
            "No case-mix block produced a reportable interval in this run: all RIH point "
            "fits failed overlap or balance gates, while SR1482 point fits had unstable "
            "bootstrap overlap. Displayed SR1482 points are diagnostic only.",
        ]
    lines += [
        "",
        "Within each KRAS class, an L2-regularized logistic role model produces clipped "
        "ATT-form odds weights for primary patients, with propensities bounded to "
        "[0.02, 0.98]. This is noncausal covariate standardization, "
        "not a treatment-effect estimator. Blocks add MSI/BRAF, then age/sex, then source "
        "type. Fits must pass "
        "categorical support, clipping, effective-sample-size, maximum-weight, age-support, "
        "and post-weight balance gates separately in mutant and wild-type patients. A "
        "non-estimable row is not interpreted as evidence of no confounding. Complete "
        "diagnostics are serialized in `case_mix_standardization.csv`.",
        "",
        "Taken together, the Set-D and subtype sensitivities did not consistently attenuate "
        "the point differences. The overlap-gated ATT rows state whether broader recorded "
        "case-mix adjustment is supportable; incomplete records and wide uncertainty "
        "neither establish nor exclude molecular confounding.",
        "",
        "### Class-conditional frozen-score structure",
        "",
        "| Cohort | WT M−P mean-logit shift [95% CI] | Mutant M−P mean-logit shift [95% CI] | Class-midpoint shift [95% CI] | Mutant−WT separation shift [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for row in score_structure.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | "
            f"{_fmt(row.wild_type_metastatic_minus_primary_mean_logit)} "
            f"[{_fmt(row.wild_type_shift_ci_low)}, {_fmt(row.wild_type_shift_ci_high)}] | "
            f"{_fmt(row.mutant_metastatic_minus_primary_mean_logit)} "
            f"[{_fmt(row.mutant_shift_ci_low)}, {_fmt(row.mutant_shift_ci_high)}] | "
            f"{_fmt(row.class_midpoint_shift)} "
            f"[{_fmt(row.midpoint_shift_ci_low)}, {_fmt(row.midpoint_shift_ci_high)}] | "
            f"{_fmt(row.mutant_minus_wild_type_separation_shift)} "
            f"[{_fmt(row.separation_shift_ci_low)}, "
            f"{_fmt(row.separation_shift_ci_high)}] |"
        )
    lines += [
        "",
        "The midpoint term is the average shift of mutant and wild-type logits; a purely "
        "class-common shift cannot change AUROC. The separation term is mutant shift minus "
        "wild-type shift and can reveal compression or expansion of mean class separation, "
        "but it is not itself a rank statistic or a biological attribution. These summaries "
        "use only the conformant family-naive score scale.",
        "",
        "## 4. Audit-only decodability from averaged UNI-v1 features",
        "",
        "Patient vectors average frozen 1,024-dimensional UNI-v1 slide-mean encoder "
        "features. They are not governed 512-dimensional MIL patient embeddings and are "
        "not pathology-defined morphologic measurements. Standardization and an L2 "
        "logistic classifier are fitted independently inside each patient-level CV fold. "
        "The 2.5th–97.5th percentile ranges summarize repeated split sensitivity, not "
        "confidence intervals.",
        "",
        "| Cohort | Feature-decoding task | Stratum | n | AUROC median [split range] |",
        "|---|---|---|---:|---|",
    ]
    for row in structural.itertuples(index=False):
        if not row.estimable:
            value = "NOT_ESTIMABLE"
        else:
            value = (
                f"{_fmt(row.auroc_median)} [{_fmt(row.auroc_split_p025)}, "
                f"{_fmt(row.auroc_split_p975)}]"
            )
        lines.append(f"| {row.cohort} | {row.task} | {row.stratum} | {row.n} | {value} |")
    lines += [
        "",
        "This audit-only panel found source type highly decodable from these features "
        "within the analyzed cohorts. "
        "Role was also decodable within source-type strata, but role and organ remain "
        "confounded with tissue context, treatment and clinical case mix, tissue amount, "
        "and technical regime. Decodability does not identify a biological, morphologic, "
        "acquisition, or KRAS-specific shift. Because v6 lacks per-row source-label "
        "provenance, this panel cannot independently validate the source labels themselves.",
        "",
        "## 5. Paired-patient sensitivity",
        "",
    ]
    if paired.empty:
        lines.append("No exact primary/metastatic patient overlaps were available.")
    else:
        summary = paired.groupby(["lineage", "cohort"]).agg(
            n=("patient_id", "size"),
            label_discordant=(
                "primary_label",
                lambda values: int(
                    np.sum(
                        values.to_numpy() != paired.loc[values.index, "metastatic_label"].to_numpy()
                    )
                ),
            ),
            mean_shift=("metastatic_minus_primary", "mean"),
            median_shift=("metastatic_minus_primary", "median"),
        )
        lines += [
            "| Score view | Cohort | Pairs | Label-discordant | Mean logit shift | Median logit shift |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for index, row in summary.iterrows():
            lines.append(
                f"| {index[0]} | {index[1]} | {int(row.n)} | "
                f"{int(row.label_discordant)} | {_fmt(row.mean_shift)} | "
                f"{_fmt(row.median_shift)} |"
            )
        family_pairs = paired[paired["lineage"].eq(family)]
        transitions = (
            family_pairs.groupby(["primary_source_type", "metastatic_source_type"]).size().to_dict()
        )
        lines += [
            "",
            "For the family-naive RIH pairs, source transitions were "
            + ", ".join(
                f"{source[0]}→{source[1]}: {count}" for source, count in sorted(transitions.items())
            )
            + ". One pair had discordant recorded KRAS status. These unadjusted shifts "
            "cannot isolate a within-person role or acquisition effect.",
        ]
    lines += [
        "",
        "## Integrated interpretation",
        "",
        "1. **Metastatic site.** Large positive SR1482 liver-versus-non-liver contrasts "
        "were not reproduced in RIH; sparse source-by-site cells preclude an organ-effect claim.",
        "2. **Source type.** An audit-only panel found biopsy/resection highly decodable "
        "from averaged UNI-v1 "
        "features. RIH ranking contrasts were negative but based on only 18 metastatic "
        "resections and were not reproduced at comparable magnitude in SR1482; a replicated "
        "source-type effect was not established.",
        "3. **Recorded molecular composition.** MSS/BRAF-WT restriction and coarse "
        "KRAS-subtype reweighting did not consistently attenuate point differences; "
        "incomplete records and wide uncertainty neither establish nor exclude confounding.",
        "4. **Unresolved association.** Standardized point differences and role decodability "
        "remain compatible with technical, tissue-context, clinical, and biological "
        "differences. E2e and E2f show only that the tested learning and adaptation "
        "procedures did not reproducibly improve ranking at this sample size; they do not "
        "localize a mechanism.",
        "",
        "## Limits",
        "",
        "- `source_type` distinguishes biopsy/resection/unknown but does not measure tumor "
        "fraction, necrosis, viable tissue, treatment, fixation, grade, block age, or area.",
        "- Metastatic site is structurally undefined for primary tumors, and specimen "
        "procedure is partly determined by clinical role. This is not a causal mediation analysis.",
        "- RIH-M has only 18 resections; SR1482-M has 24 unknown-source patients. Fine "
        "organ and crossed source-by-site cells are often sparse.",
        "- AUROC is a pairwise ranking estimand. Reweighting can change it through both "
        "within-source ranking and cross-source score offsets.",
        "- No target-label refit, calibration, feature selection, or model update is used "
        "to create the primary frozen-score comparisons.",
        "- Results characterize these frozen models and cohorts, not a universal metastatic "
        "mechanism or clinical utility.",
        "- The receipt verifies a closed-world bundle, current byte identities, and sealed "
        "parent lineage. Until its hash is pinned by an external release or successor "
        "bundle, it is a self-consistency receipt rather than an external execution attestation.",
        "",
        "## Artifact guide",
        "",
        "- `population.csv`: patient counts by cohort, role, and source type.",
        "- `performance_strata.csv`: overall, source, organ, fine-site, and crossed cells.",
        "- `contrasts.csv`: patient-bootstrap site, source, and disjoint-role contrasts.",
        "- `pairwise_source_cells.csv`: positive-source × negative-source concordance cells.",
        "- `decomposition.csv`: source and coarse KRAS-subtype standardizations.",
        "- `site_source_standardization.csv`: site contrasts at common source composition.",
        "- `case_mix_standardization.csv`: overlap diagnostics and class-stratified ATT rows.",
        "- `score_structure.csv`: class-specific logit location and separation shifts.",
        "- `source_alignment.csv`: primary-derived marginal centering sensitivity.",
        "- `structural_cv.csv`: audit-only averaged-feature decodability summaries.",
        "- `paired_role_scores.csv`: deidentified paired-patient score transitions.",
        "- `analysis_table.parquet`: patient frozen scores with joined covariates.",
        "- `patient_embeddings.parquet`: patient-averaged 1,024-dimensional UNI-v1 "
        "slide-mean encoder features; not MIL patient embeddings.",
        "- `met_transfer_diagnostics.png` and `.pdf`: integrated four-panel figure.",
        "- `results.json`: machine-readable results and evidence states.",
        "- `receipt.json`: fixed input/output identities and verification checks.",
        "",
    ]
    return "\n".join(lines)


def build_result_summary(
    validation: dict[str, Any],
    population: pd.DataFrame,
    metrics: pd.DataFrame,
    contrasts: pd.DataFrame,
    decomposition: pd.DataFrame,
    site_source: pd.DataFrame,
    case_mix: pd.DataFrame,
    score_structure: pd.DataFrame,
    alignment: pd.DataFrame,
    structural: pd.DataFrame,
    paired: pd.DataFrame,
    n_bootstrap: int,
    cv_repeats: int,
) -> dict[str, Any]:
    smoke = n_bootstrap < DEFAULT_BOOTSTRAPS or cv_repeats < DEFAULT_CV_REPEATS
    return {
        "schema_version": "crc-final-v6-met-transfer-diagnostics-1.0",
        "analysis_status": (
            "SMOKE_TEST_ONLY_NOT_SCIENTIFIC_RESULT"
            if smoke
            else "POST_OUTCOME_EXPLORATORY_DIAGNOSTIC"
        ),
        "causal_status": "NONCAUSAL_OBSERVATIONAL_DECOMPOSITION",
        "headline": (
            "SR1482 showed large positive liver-versus-non-liver point contrasts, whereas "
            "RIH contrasts were near zero. An audit-only panel found source type highly "
            "feature-decodable, but "
            "biopsy-resection AUROC contrasts and composition components were imprecise. "
            "Recorded molecular sensitivities did not consistently attenuate role point "
            "differences. The remaining associations are unresolved and cannot be assigned "
            "to technical, tissue-context, clinical, or biological causes."
        ),
        "v6_validation": validation,
        "settings": {
            "n_bootstrap": n_bootstrap,
            "secondary_composition_bootstrap": min(n_bootstrap, SECONDARY_BOOTSTRAPS),
            "bootstrap_unit": "patient",
            "case_mix_propensity_clip": list(PROPENSITY_CLIP),
            "cv_repeats": cv_repeats,
            "random_seed": RNG_SEED,
            "role_overlap_policy": "remove shared patients from both role arms",
            "unknown_source_policy": "retain as a distinct level; binary contrasts are complete-case",
        },
        "population": population.to_dict(orient="records"),
        "performance_strata": metrics.to_dict(orient="records"),
        "contrasts": contrasts.to_dict(orient="records"),
        "decomposition": decomposition.to_dict(orient="records"),
        "site_source_standardization": site_source.to_dict(orient="records"),
        "case_mix_standardization": case_mix.to_dict(orient="records"),
        "score_structure": score_structure.to_dict(orient="records"),
        "source_alignment": alignment.to_dict(orient="records"),
        "structural_cv": structural.to_dict(orient="records"),
        "paired_role_summary": (
            paired.groupby(["lineage", "cohort"])
            .agg(n_pairs=("patient_id", "size"), mean_shift=("metastatic_minus_primary", "mean"))
            .reset_index()
            .to_dict(orient="records")
            if not paired.empty
            else []
        ),
        "interpretation_boundary": [
            "No single measured factor is interpreted causally.",
            "Seed/fold replicates are not inference units.",
            "Small organ-by-source cells are descriptive.",
            "Frozen models are not updated; target labels are used only for post-outcome "
            "evaluation, stratification, and standardization.",
        ],
    }


def input_artifacts() -> dict[str, dict[str, Any]]:
    paths = {
        "analysis_script": Path(__file__).resolve(),
        "crc_final_v5_csv": V5,
        "crc_final_v6_csv": V6,
        "crc_final_v6_xlsx": V6_XLSX,
        "final_v12_receipt": FINAL_V12_RECEIPT,
        "final_v12_source_manifest": FINAL_V12_MANIFEST,
        "final_v12_verifier": FINAL_V12_VERIFIER,
        "loco_contract": LOCO_CONTRACT,
        "loco_inference_seal": LOCO_INFERENCE_SEAL,
        "loco_primary_scores": LOCO_PRIMARY_SCORES,
        "loco_metastatic_scores": LOCO_MET_SCORES,
        "loco_results_receipt": LOCO_RESULTS_RECEIPT,
        "two_encoder_patient_scores": TWO_ENCODER_SCORES,
        "two_encoder_results_receipt": TWO_ENCODER_RESULTS_RECEIPT,
        "univ1_slide_means": SLIDE_MEANS,
        "univ1_coordinates": COORDINATES,
    }
    for (cohort, role), path in ROSTERS.items():
        paths[f"roster_{cohort.lower()}_{role}"] = path
    return {name: _artifact(path) for name, path in sorted(paths.items())}


def run_analysis(
    output_dir: Path,
    *,
    n_bootstrap: int,
    cv_repeats: int,
) -> None:
    output_dir = output_dir.absolute()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")
    if cv_repeats < 2:
        raise ValueError("cv_repeats must be at least 2")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        input_snapshot = input_artifacts()
        parent_validation = validate_parent_score_receipts()
        v6, validation = validate_v6()
        validation["parent_score_receipts"] = parent_validation
        metadata, slide_metadata = build_patient_metadata(v6)
        scores = build_score_table(metadata)
        population = population_table(metadata)
        metrics = performance_strata(scores, n_bootstrap)
        contrasts = contrast_table(scores, n_bootstrap)
        pairwise = pairwise_source_cells(scores)
        decomposition = decomposition_table(scores, n_bootstrap)
        site_source = site_source_standardization(scores, n_bootstrap)
        case_mix = case_mix_standardization_table(scores, n_bootstrap)
        score_structure = score_structure_table(scores, n_bootstrap)
        alignment = source_alignment_table(scores, n_bootstrap)
        paired = paired_role_table(scores)
        embedding_keys, embedding_matrix = build_patient_embeddings(metadata, slide_metadata)
        structural = structural_cv_table(embedding_keys, embedding_matrix, cv_repeats)

        population.to_csv(temporary / "population.csv", index=False)
        metrics.to_csv(temporary / "performance_strata.csv", index=False)
        contrasts.to_csv(temporary / "contrasts.csv", index=False)
        pairwise.to_csv(temporary / "pairwise_source_cells.csv", index=False)
        decomposition.to_csv(temporary / "decomposition.csv", index=False)
        site_source.to_csv(temporary / "site_source_standardization.csv", index=False)
        case_mix.to_csv(temporary / "case_mix_standardization.csv", index=False)
        score_structure.to_csv(temporary / "score_structure.csv", index=False)
        alignment.to_csv(temporary / "source_alignment.csv", index=False)
        structural.to_csv(temporary / "structural_cv.csv", index=False)
        paired.to_csv(temporary / "paired_role_scores.csv", index=False)
        scores.to_parquet(temporary / "analysis_table.parquet", index=False)
        embedding_frame = pd.concat(
            [
                embedding_keys.reset_index(drop=True),
                pd.DataFrame(
                    embedding_matrix,
                    columns=[
                        f"univ1_dim_{index:04d}" for index in range(embedding_matrix.shape[1])
                    ],
                ),
            ],
            axis=1,
        )
        embedding_frame.to_parquet(temporary / "patient_embeddings.parquet", index=False)
        summary = build_result_summary(
            validation,
            population,
            metrics,
            contrasts,
            decomposition,
            site_source,
            case_mix,
            score_structure,
            alignment,
            structural,
            paired,
            n_bootstrap,
            cv_repeats,
        )
        _write_json(temporary / "results.json", summary)
        report = build_report(
            validation,
            population,
            metrics,
            contrasts,
            decomposition,
            site_source,
            case_mix,
            score_structure,
            alignment,
            structural,
            paired,
            n_bootstrap=n_bootstrap,
            cv_repeats=cv_repeats,
        )
        (temporary / "report.md").write_text(report)
        make_figure(
            metrics,
            contrasts,
            decomposition,
            structural,
            temporary / "met_transfer_diagnostics.png",
            temporary / "met_transfer_diagnostics.pdf",
        )

        if input_artifacts() != input_snapshot:
            raise RuntimeError("At least one input changed during analysis; refusing publication")

        outputs = {
            path.name: _artifact(path)
            for path in sorted(temporary.iterdir())
            if path.name != "receipt.json"
        }
        if set(outputs) != EXPECTED_OUTPUTS:
            raise RuntimeError(
                f"Output roster drift: expected={sorted(EXPECTED_OUTPUTS)}, "
                f"observed={sorted(outputs)}"
            )
        for record in outputs.values():
            record["path"] = f"{output_dir}/{Path(record['path']).name}"
        receipt = {
            "schema_version": "crc-final-v6-met-transfer-diagnostics-receipt-1.0",
            "status": (
                "COMPLETED_POST_OUTCOME_EXPLORATORY_DIAGNOSTIC"
                if n_bootstrap >= DEFAULT_BOOTSTRAPS and cv_repeats >= DEFAULT_CV_REPEATS
                else "SMOKE_TEST_ONLY_NOT_SCIENTIFIC_RESULT"
            ),
            "analysis_command": (
                "python tools/crc_final_v6_met_transfer_diagnostics.py run "
                f"--output-dir {output_dir} --n-bootstrap {n_bootstrap} "
                f"--cv-repeats {cv_repeats}"
            ),
            "settings": {
                "n_bootstrap": n_bootstrap,
                "cv_repeats": cv_repeats,
                "random_seed": RNG_SEED,
            },
            "inputs": input_snapshot,
            "outputs": outputs,
            "checks": {
                "v6_is_exact_v5_plus_source_type": "PASS",
                "v6_csv_xlsx_cell_equality": "PASS",
                "exact_target_roster_joins": "PASS",
                "patient_role_metadata_consistency": "PASS",
                "score_label_identity": "PASS",
                "score_artifacts_match_parent_receipts": "PASS",
                "final_v12_parent_chain_authenticated": "PASS",
                "role_overlap_removed_from_contrasts": "PASS",
                "patient_level_bootstrap": "PASS",
                "unknown_source_retained_explicitly": "PASS",
                "post_outcome_noncausal_status_declared": "PASS",
                "frozen_models_not_refit": "PASS",
                "inputs_unchanged_during_analysis": "PASS",
                "source_case_mix_overlap_gates": "PASS",
                "audit_only_embedding_provenance_declared": "PASS",
                "strict_json_no_nonfinite": "PASS",
                "exact_output_roster": "PASS",
            },
        }
        if set(receipt["checks"]) != EXPECTED_CHECKS:
            raise RuntimeError("Internal receipt check roster drift")
        _write_json(temporary / "receipt.json", receipt)
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_identity_record(record: Any, *, context: str) -> None:
    if not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"Invalid identity-record schema for {context}")
    if not isinstance(record["path"], str) or not record["path"]:
        raise ValueError(f"Invalid artifact path for {context}")
    if type(record["size_bytes"]) is not int or record["size_bytes"] < 0:
        raise ValueError(f"Invalid artifact size for {context}")
    if (
        not isinstance(record["sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
    ):
        raise ValueError(f"Invalid artifact digest for {context}")


def verify_published(output_dir: Path) -> None:
    output_dir = output_dir.absolute()
    receipt_path = output_dir / "receipt.json"
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError("Published bundle root must be a real directory")
    entries_before = {path.name for path in output_dir.iterdir()}
    expected_entries = EXPECTED_OUTPUTS | {"receipt.json"}
    if entries_before != expected_entries:
        raise ValueError(
            f"Published bundle entry roster drift: expected={sorted(expected_entries)}, "
            f"observed={sorted(entries_before)}"
        )

    receipt_identity_before = _artifact(receipt_path)
    receipt = _strict_json_load(receipt_path)
    if _artifact(receipt_path) != receipt_identity_before:
        raise RuntimeError("Receipt changed while it was being parsed")
    expected_receipt_keys = {
        "schema_version",
        "status",
        "analysis_command",
        "settings",
        "inputs",
        "outputs",
        "checks",
    }
    if set(receipt) != expected_receipt_keys:
        raise ValueError("Receipt top-level schema drift")
    if receipt["schema_version"] != "crc-final-v6-met-transfer-diagnostics-receipt-1.0":
        raise ValueError("Receipt schema version drift")
    settings = receipt["settings"]
    if not isinstance(settings, dict) or set(settings) != {
        "n_bootstrap",
        "cv_repeats",
        "random_seed",
    }:
        raise ValueError("Receipt settings schema drift")
    if any(
        type(settings[field]) is not int for field in ("n_bootstrap", "cv_repeats", "random_seed")
    ):
        raise ValueError("Receipt integer settings have invalid types")
    if (
        settings["n_bootstrap"] < 100
        or settings["cv_repeats"] < 2
        or settings["random_seed"] != RNG_SEED
    ):
        raise ValueError("Receipt settings violate minimum execution contract")
    final_run = (
        settings["n_bootstrap"] >= DEFAULT_BOOTSTRAPS
        and settings["cv_repeats"] >= DEFAULT_CV_REPEATS
    )
    expected_status = (
        "COMPLETED_POST_OUTCOME_EXPLORATORY_DIAGNOSTIC"
        if final_run
        else "SMOKE_TEST_ONLY_NOT_SCIENTIFIC_RESULT"
    )
    if receipt["status"] != expected_status:
        raise ValueError("Receipt status does not match its execution settings")
    expected_command = (
        "python tools/crc_final_v6_met_transfer_diagnostics.py run "
        f"--output-dir {output_dir} --n-bootstrap {settings['n_bootstrap']} "
        f"--cv-repeats {settings['cv_repeats']}"
    )
    if receipt["analysis_command"] != expected_command:
        raise ValueError("Receipt analysis command drift")

    if not isinstance(receipt["inputs"], dict) or set(receipt["inputs"]) != EXPECTED_INPUTS:
        raise ValueError("Receipt input roster drift")
    current_inputs = input_artifacts()
    if set(current_inputs) != EXPECTED_INPUTS or receipt["inputs"] != current_inputs:
        raise ValueError("Receipt inputs do not match the fixed current input roster")
    for name, record in receipt["inputs"].items():
        _validate_identity_record(record, context=f"input/{name}")

    # Replay scientific parent authentication and v6 extension semantics rather
    # than trusting receipt PASS strings.
    validate_parent_score_receipts()
    validate_v6()

    if not isinstance(receipt["outputs"], dict) or set(receipt["outputs"]) != EXPECTED_OUTPUTS:
        raise ValueError("Receipt output roster drift")
    input_inodes = {
        (Path(record["path"]).stat().st_dev, Path(record["path"]).stat().st_ino)
        for record in receipt["inputs"].values()
    }
    verified_outputs: dict[str, dict[str, Any]] = {}
    for name in sorted(EXPECTED_OUTPUTS):
        record = receipt["outputs"][name]
        _validate_identity_record(record, context=f"output/{name}")
        expected_path = output_dir / name
        if record["path"] != str(expected_path):
            raise ValueError(f"Receipt-controlled output path rejected: {name}")
        if expected_path.name != name or expected_path.parent != output_dir:
            raise ValueError(f"Output escaped bundle root: {name}")
        actual = _artifact(expected_path)
        if actual != record:
            raise ValueError(f"Output identity mismatch: {name}")
        verified_outputs[name] = actual
        output_stat = expected_path.stat()
        if (output_stat.st_dev, output_stat.st_ino) in input_inodes:
            raise ValueError(f"Output aliases an input inode: {name}")

    if not isinstance(receipt["checks"], dict) or set(receipt["checks"]) != EXPECTED_CHECKS:
        raise ValueError("Receipt check roster drift")
    if any(value != "PASS" for value in receipt["checks"].values()):
        raise ValueError("Receipt contains a non-PASS check")

    results_path = output_dir / "results.json"
    results_identity_before = verified_outputs["results.json"]
    results = _strict_json_load(results_path)
    if _artifact(results_path) != results_identity_before:
        raise RuntimeError("Results JSON changed while it was being parsed")
    expected_results_keys = {
        "schema_version",
        "analysis_status",
        "causal_status",
        "headline",
        "v6_validation",
        "settings",
        "population",
        "performance_strata",
        "contrasts",
        "decomposition",
        "site_source_standardization",
        "case_mix_standardization",
        "score_structure",
        "source_alignment",
        "structural_cv",
        "paired_role_summary",
        "interpretation_boundary",
    }
    if set(results) != expected_results_keys:
        raise ValueError("Results JSON top-level schema drift")
    if results["schema_version"] != "crc-final-v6-met-transfer-diagnostics-1.0":
        raise ValueError("Results schema version drift")
    expected_analysis_status = (
        "POST_OUTCOME_EXPLORATORY_DIAGNOSTIC"
        if final_run
        else "SMOKE_TEST_ONLY_NOT_SCIENTIFIC_RESULT"
    )
    if (
        results["analysis_status"] != expected_analysis_status
        or results["causal_status"] != "NONCAUSAL_OBSERVATIONAL_DECOMPOSITION"
    ):
        raise ValueError("Results evidence status drift")
    if any(
        results["settings"].get(key) != settings[key]
        for key in ("n_bootstrap", "cv_repeats", "random_seed")
    ):
        raise ValueError("Results/receipt settings mismatch")

    canonical_receipt = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if receipt_path.read_bytes() != canonical_receipt:
        raise ValueError("Receipt is not in canonical strict-JSON form")
    if input_artifacts() != current_inputs:
        raise RuntimeError("At least one input changed during verification")
    for name, expected_identity in verified_outputs.items():
        if _artifact(output_dir / name) != expected_identity:
            raise RuntimeError(f"Output changed during verification: {name}")
    if _artifact(receipt_path) != receipt_identity_before:
        raise RuntimeError("Receipt changed during verification")
    if {path.name for path in output_dir.iterdir()} != entries_before:
        raise RuntimeError("Bundle directory changed during verification")
    print(f"VERIFIED: {receipt_path} ({expected_status})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAPS)
    run.add_argument("--cv-repeats", type=int, default=DEFAULT_CV_REPEATS)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "run":
        run_analysis(
            args.output_dir,
            n_bootstrap=args.n_bootstrap,
            cv_repeats=args.cv_repeats,
        )
    else:
        verify_published(args.output_dir)


if __name__ == "__main__":
    main()
