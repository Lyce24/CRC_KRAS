#!/usr/bin/env python3
"""E2d-6 - RIH acquisition/processing-regime robustness (0 new fits).

QUESTION. Does RIH's descriptive primary-to-metastatic AUROC point contrast
change across the two acquisition/processing regimes that contain enough
patients for inference? E2b did not establish an overall decrement, so this
post-hoc robustness panel must not assume one exists.

This is a stratified re-analysis of E2a's frozen RIH-held-out three-seed
ensemble.  It never trains, calibrates, adapts, or selects a model.  The score
used for every ranking endpoint is the native patient mean logit.

THE THREE FILE-PROVENANCE CATEGORIES.  The frozen 369-slide RIH inventory maps
deterministically to 265 untouched 0.5016-MPP SVS files (the ``native SVS``
regime),
103 verified repaired TIFF replacements (``repaired-converted technical
regime``), and one 0.13899-MPP SVS (``Versa``).  Those are inventory counts,
not analysis counts.  Only 240 slides have eligible KRAS labels and frozen E2a
scores.  The Versa category contributes one metastatic patient and no primary
patient, so it is descriptive only and is never included in an AUROC, contrast,
or interaction.

THE E2b RULE IS INHERITED VERBATIM.  Eight RIH patients have both roles.  They
remain in standalone primary and metastatic summaries, but all eight are
removed from BOTH sides before either within-regime decrement or the
between-regime interaction is calculated.  The resulting four inferential
arms must be pairwise patient-disjoint.

THE HEADLINE INTERACTION IS

    [AUROC(M) - AUROC(P)]_repaired - [AUROC(M) - AUROC(P)]_native-SVS

and its percentile interval comes from independently resampling the four
disjoint patient arms, outcome-stratified within each arm.  A non-significant
interaction is not evidence of equivalence or robustness.

IMPORTANT INTERPRETATION LIMIT.  ``Repaired-converted`` is a file-processing
lineage, not a proved scanner model.  Platform, native resolution, acquisition
era, staining/storage history, and repair status are inseparable here.  This
panel can establish effect modification by a composite technical regime; it
cannot attribute an effect to a scanner.

DESIGN STATUS. This component was implemented after the core E2a/E2b results.
It is exploratory: the historical design archive describes the intended
within-regime check, while the then-current Experimental Setup still marked it
as unbuilt. Both documents are bound as context; neither is mislabeled as proof
of prospective pre-specification.

Usage:
    python aim2_rih_acquisition_regime.py report --cap 8192
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_loco_transport  # noqa: E402
import aim2_metastatic_transport  # noqa: E402
from oceanpath.aim1 import lineage, paths  # noqa: E402
from oceanpath.eval.external import sigmoid  # noqa: E402

INVENTORY_PATH = paths.MANIFEST_ROOT / "colon_ready_inventory.csv"
RIH_REPAIR_STATE_PATH = Path("/mnt/d/YC.Liu/slides/colon/fix_rih_state.jsonl")
PROTOCOL_PATH = REPO / "reports" / "Experimental_Setup.md"
DESIGN_ARCHIVE_PATH = REPO / "reports" / "Experimental_Setup_DESIGN_ARCHIVE.md"

REGIME_APERIO = "aperio_native"
REGIME_REPAIRED = "repaired_converted_technical_regime"
REGIME_VERSA = "versa_descriptive_only"
MAIN_REGIMES = (REGIME_APERIO, REGIME_REPAIRED)
ALL_REGIMES = (*MAIN_REGIMES, REGIME_VERSA)
REGIME_LABELS = {
    REGIME_APERIO: "native 0.5016-MPP SVS regime",
    REGIME_REPAIRED: "repaired-converted technical regime",
    REGIME_VERSA: "Versa (descriptive only)",
}

DEFAULT_N_BOOTSTRAP = aim2_metastatic_transport.DEFAULT_N_BOOTSTRAP

EXPECTED_INVENTORY_COUNTS = {
    REGIME_APERIO: 265,
    REGIME_REPAIRED: 103,
    REGIME_VERSA: 1,
}
EXPECTED_INVENTORY_MPP_COUNTS = {
    REGIME_APERIO: {"0.501600": 265},
    REGIME_REPAIRED: {"0.189012": 98, "0.378024": 5},
    REGIME_VERSA: {"0.138990": 1},
}
EXPECTED_STANDALONE = {
    ("primary", REGIME_APERIO): {"slides": 107, "patients": 106, "wild_type": 56, "mutant": 50},
    ("primary", REGIME_REPAIRED): {"slides": 48, "patients": 47, "wild_type": 27, "mutant": 20},
    ("primary", REGIME_VERSA): {"slides": 0, "patients": 0, "wild_type": 0, "mutant": 0},
    ("metastatic", REGIME_APERIO): {"slides": 56, "patients": 56, "wild_type": 31, "mutant": 25},
    ("metastatic", REGIME_REPAIRED): {"slides": 28, "patients": 28, "wild_type": 17, "mutant": 11},
    ("metastatic", REGIME_VERSA): {"slides": 1, "patients": 1, "wild_type": 0, "mutant": 1},
}
EXPECTED_CONTRAST = {
    ("primary", REGIME_APERIO): {"patients": 99, "wild_type": 54, "mutant": 45},
    ("primary", REGIME_REPAIRED): {"patients": 46, "wild_type": 26, "mutant": 20},
    ("primary", REGIME_VERSA): {"patients": 0, "wild_type": 0, "mutant": 0},
    ("metastatic", REGIME_APERIO): {"patients": 50, "wild_type": 29, "mutant": 21},
    ("metastatic", REGIME_REPAIRED): {"patients": 26, "wild_type": 15, "mutant": 11},
    ("metastatic", REGIME_VERSA): {"patients": 1, "wild_type": 0, "mutant": 1},
}


def report_path(cap: int) -> Path:
    return lineage.eval_root() / f"e2d6_rih_acquisition_regime_cap{cap}.json"


def patient_table_path(cap: int) -> Path:
    return lineage.component_root("e2d6") / "tables" / f"rih_regime_patients_cap{cap}.parquet"


def patient_table_receipt_path(cap: int) -> Path:
    return patient_table_path(cap).with_suffix(".receipt.json")


def _utc_now() -> str:
    """Python-3.10-compatible timezone-aware UTC timestamp."""

    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_repair_state(path: Path = RIH_REPAIR_STATE_PATH) -> pd.DataFrame:
    """Load the preserved RIH repair ledger and retain its original file key."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid repair JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Repair entry at {path}:{line_number} is not an object")
            rows.append(value)
    frame = pd.DataFrame(rows)
    required = {"file", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Repair ledger lacks columns: {sorted(missing)}")
    frame["slide_id"] = frame["file"].map(lambda value: Path(str(value)).stem)
    if frame["slide_id"].duplicated().any():
        duplicate = frame.loc[frame["slide_id"].duplicated(), "slide_id"].iloc[0]
        raise ValueError(f"Repair ledger contains duplicate slide_id {duplicate!r}")
    if not frame["status"].eq("done").all():
        bad = frame.loc[~frame["status"].eq("done"), ["slide_id", "status"]]
        raise ValueError(f"RIH repair ledger contains unfinished entries: {bad.to_dict('records')}")
    return frame.sort_values("slide_id").reset_index(drop=True)


def _parse_details(value: object, slide_id: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ValueError(f"RIH inventory details are not JSON text for {slide_id}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid RIH inventory details JSON for {slide_id}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"RIH inventory details are not an object for {slide_id}")
    return parsed


def _is_mpp(values: pd.Series, expected: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return pd.Series(
        np.isclose(numeric, expected, rtol=0.0, atol=1e-9),
        index=values.index,
    )


def _mpp_count(frame: pd.DataFrame) -> dict[str, int]:
    return {
        f"{float(value):.6f}": int(count)
        for value, count in frame["mpp"].value_counts().sort_index().items()
    }


def build_slide_regime_map(
    inventory: pd.DataFrame,
    repair_state: pd.DataFrame,
    *,
    enforce_expected: bool = True,
) -> pd.DataFrame:
    """Map every ready RIH inventory slide to one deterministic regime.

    Classification deliberately uses the frozen inventory and repair ledger,
    not mutable file mtimes or a filename-number heuristic.
    """

    required_inventory = {
        "output_id",
        "cohort",
        "status",
        "source_status",
        "wsi",
        "mpp",
        "details",
    }
    missing = required_inventory - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory lacks columns: {sorted(missing)}")
    required_repair = {"slide_id", "file", "status"}
    missing_repair = required_repair - set(repair_state.columns)
    if missing_repair:
        raise ValueError(f"Repair state lacks columns: {sorted(missing_repair)}")

    rih = inventory.loc[inventory["cohort"].eq("RIH")].copy()
    rih["slide_id"] = rih["output_id"].astype(str)
    if rih.empty:
        raise ValueError("RIH inventory is empty")
    if rih["slide_id"].duplicated().any():
        duplicate = rih.loc[rih["slide_id"].duplicated(), "slide_id"].iloc[0]
        raise ValueError(f"RIH inventory contains duplicate slide_id {duplicate!r}")
    if not rih["status"].eq("ready").all():
        bad = rih.loc[~rih["status"].eq("ready"), ["slide_id", "status"]]
        raise ValueError(f"RIH inventory contains non-ready slides: {bad.to_dict('records')}")
    if repair_state["slide_id"].duplicated().any():
        duplicate = repair_state.loc[repair_state["slide_id"].duplicated(), "slide_id"].iloc[0]
        raise ValueError(f"Repair state contains duplicate slide_id {duplicate!r}")
    if not repair_state["status"].eq("done").all():
        raise ValueError("Repair state contains entries not marked done")

    details = [
        _parse_details(value, slide_id)
        for value, slide_id in zip(rih["details"], rih["slide_id"], strict=True)
    ]
    rih["canonical_source"] = [value.get("canonical_source") for value in details]
    rih["repair_state"] = [value.get("repair_state") for value in details]
    rih["suffix"] = rih["wsi"].map(lambda value: Path(str(value)).suffix.lower())
    repair_ids = set(repair_state["slide_id"].astype(str))

    aperio = (
        rih["source_status"].eq("original_standard")
        & rih["canonical_source"].eq("original")
        & rih["repair_state"].eq("not_selected")
        & rih["suffix"].eq(".svs")
        & _is_mpp(rih["mpp"], 0.5016)
        & ~rih["slide_id"].isin(repair_ids)
    )
    repaired_mpp = _is_mpp(rih["mpp"], 0.18901199443536876) | _is_mpp(
        rih["mpp"], 0.3780239888707375
    )
    repaired = (
        rih["source_status"].eq("fixed_verified")
        & rih["canonical_source"].eq("fixed")
        & rih["repair_state"].eq("done")
        & rih["suffix"].eq(".tiff")
        & repaired_mpp
        & rih["slide_id"].isin(repair_ids)
    )
    versa = (
        rih["slide_id"].eq("SL-348")
        & rih["source_status"].eq("original_standard")
        & rih["canonical_source"].eq("original")
        & rih["repair_state"].eq("not_selected")
        & rih["suffix"].eq(".svs")
        & _is_mpp(rih["mpp"], 0.13899)
        & ~rih["slide_id"].isin(repair_ids)
    )
    memberships = aperio.astype(int) + repaired.astype(int) + versa.astype(int)
    if not memberships.eq(1).all():
        bad = rih.loc[memberships.ne(1), ["slide_id", "wsi", "mpp", "source_status"]]
        raise ValueError(
            "RIH slide provenance is unmapped or multiply mapped: "
            f"{bad.to_dict('records')[:10]}"
        )

    rih["technical_regime"] = np.select(
        [aperio, repaired, versa],
        [REGIME_APERIO, REGIME_REPAIRED, REGIME_VERSA],
        default="",
    )
    mapped_repaired = set(rih.loc[rih["technical_regime"].eq(REGIME_REPAIRED), "slide_id"])
    if mapped_repaired != repair_ids:
        raise ValueError(
            "Repair ledger and repaired inventory membership differ: "
            f"inventory_only={sorted(mapped_repaired - repair_ids)[:10]}, "
            f"ledger_only={sorted(repair_ids - mapped_repaired)[:10]}"
        )

    if enforce_expected:
        counts = rih["technical_regime"].value_counts().to_dict()
        if counts != EXPECTED_INVENTORY_COUNTS:
            raise RuntimeError(
                f"RIH inventory regime counts changed: {counts} != {EXPECTED_INVENTORY_COUNTS}"
            )
        observed_mpp = {
            regime: _mpp_count(rih[rih["technical_regime"].eq(regime)])
            for regime in ALL_REGIMES
        }
        if observed_mpp != EXPECTED_INVENTORY_MPP_COUNTS:
            raise RuntimeError(
                "RIH inventory MPP composition changed: "
                f"{observed_mpp} != {EXPECTED_INVENTORY_MPP_COUNTS}"
            )

    columns = [
        "slide_id",
        "technical_regime",
        "wsi",
        "mpp",
        "source_status",
        "canonical_source",
        "repair_state",
        "suffix",
    ]
    return rih[columns].sort_values("slide_id").reset_index(drop=True)


def attach_regime_to_manifest(
    manifest: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    role: str,
) -> pd.DataFrame:
    """Attach one file-provenance regime to every scored manifest slide."""

    required = {"slide_id", "patient_id", "target_label", "specimen_role"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{role} manifest lacks columns: {sorted(missing)}")
    roles = set(manifest["specimen_role"].dropna().astype(str))
    if roles != {role}:
        raise ValueError(f"Expected only role {role!r}, observed {sorted(roles)}")
    if manifest["slide_id"].duplicated().any():
        duplicate = manifest.loc[manifest["slide_id"].duplicated(), "slide_id"].iloc[0]
        raise ValueError(f"{role} manifest contains duplicate slide_id {duplicate!r}")
    if mapping["slide_id"].duplicated().any():
        raise ValueError("Regime mapping contains duplicate slide_id")

    tagged = manifest.merge(mapping, on="slide_id", how="left", validate="one_to_one")
    missing_map = tagged.loc[tagged["technical_regime"].isna(), "slide_id"].astype(str)
    if not missing_map.empty:
        raise ValueError(f"{role} manifest has unmapped slides: {missing_map.tolist()[:10]}")
    label_counts = tagged.groupby("patient_id")["target_label"].nunique()
    if (label_counts != 1).any():
        bad = label_counts[label_counts != 1].index.astype(str).tolist()
        raise ValueError(f"{role} patients have inconsistent within-role labels: {bad[:10]}")
    regime_counts = tagged.groupby("patient_id")["technical_regime"].nunique()
    if (regime_counts != 1).any():
        bad = regime_counts[regime_counts != 1].index.astype(str).tolist()
        raise ValueError(f"{role} patients span technical regimes: {bad[:10]}")
    return tagged.sort_values("slide_id").reset_index(drop=True)


def attach_patient_regime(
    scores: pd.DataFrame,
    tagged_manifest: pd.DataFrame,
    *,
    role: str,
) -> pd.DataFrame:
    """Join slide-derived regime metadata onto one-row-per-patient scores."""

    required_scores = {"patient_id", "label", "mean_logit", "prob_raw"}
    missing = required_scores - set(scores.columns)
    if missing:
        raise ValueError(f"{role} scores lack columns: {sorted(missing)}")
    if scores["patient_id"].duplicated().any():
        duplicate = scores.loc[scores["patient_id"].duplicated(), "patient_id"].iloc[0]
        raise ValueError(f"{role} scores duplicate patient {duplicate!r}")
    if not np.isfinite(scores[["mean_logit", "prob_raw"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{role} scores contain non-finite values")

    patient_meta = (
        tagged_manifest.groupby("patient_id", sort=True)
        .agg(
            target_label=("target_label", "first"),
            technical_regime=("technical_regime", "first"),
            slide_count=("slide_id", "size"),
            slide_ids=("slide_id", lambda values: ";".join(sorted(map(str, values)))),
            native_mpps=("mpp", lambda values: ";".join(f"{float(v):.12g}" for v in sorted(values))),
            source_statuses=(
                "source_status",
                lambda values: ";".join(sorted(set(map(str, values)))),
            ),
        )
        .reset_index()
    )
    score_ids = set(scores["patient_id"].astype(str))
    manifest_ids = set(patient_meta["patient_id"].astype(str))
    if score_ids != manifest_ids:
        raise ValueError(
            f"{role} score/manifest patients differ: "
            f"score_only={sorted(score_ids - manifest_ids)[:10]}, "
            f"manifest_only={sorted(manifest_ids - score_ids)[:10]}"
        )
    out = scores.merge(patient_meta, on="patient_id", validate="one_to_one")
    if not out["label"].astype(int).eq(out["target_label"].astype(int)).all():
        raise ValueError(f"{role} score labels differ from target manifest")
    out["role"] = role
    return out.sort_values("patient_id").reset_index(drop=True)


def _counts(frame: pd.DataFrame, *, include_slides: bool) -> dict[str, int]:
    labels = frame["label"].astype(int) if not frame.empty else pd.Series(dtype=int)
    result = {
        "patients": int(len(frame)),
        "wild_type": int((labels == 0).sum()),
        "mutant": int((labels == 1).sum()),
    }
    if include_slides:
        result["slides"] = int(frame["slide_count"].sum()) if not frame.empty else 0
    return result


def _validate_expected_population(table: pd.DataFrame, dual_ids: set[str]) -> dict[str, Any]:
    if table.duplicated(["role", "patient_id"]).any():
        raise ValueError("Patient table duplicates a role-by-patient unit")
    if set(table["technical_regime"]) != set(ALL_REGIMES):
        raise RuntimeError(
            f"Analyzed RIH regimes changed: {sorted(set(table['technical_regime']))}"
        )
    if len(dual_ids) != 8:
        raise RuntimeError(f"Expected 8 RIH dual-role patients, observed {len(dual_ids)}")

    standalone: dict[str, dict[str, int]] = {}
    for role in ("primary", "metastatic"):
        for regime in ALL_REGIMES:
            observed = _counts(
                table[table["role"].eq(role) & table["technical_regime"].eq(regime)],
                include_slides=True,
            )
            expected = EXPECTED_STANDALONE[(role, regime)]
            if observed != expected:
                raise RuntimeError(
                    f"Standalone {role}/{regime} population changed: "
                    f"{observed} != {expected}"
                )
            standalone[f"{role}/{regime}"] = observed

    contrast = table[~table["patient_id"].isin(dual_ids)].copy()
    dual = table[table["patient_id"].isin(dual_ids)].copy()
    dual_regime_counts = dual.groupby("patient_id")["technical_regime"].nunique()
    dual_label_counts = dual.groupby("patient_id")["label"].nunique()
    for role in ("primary", "metastatic"):
        for regime in ALL_REGIMES:
            observed = _counts(
                contrast[
                    contrast["role"].eq(role)
                    & contrast["technical_regime"].eq(regime)
                ],
                include_slides=False,
            )
            expected = EXPECTED_CONTRAST[(role, regime)]
            if observed != expected:
                raise RuntimeError(
                    f"Contrast {role}/{regime} population changed: "
                    f"{observed} != {expected}"
                )
    return {
        "standalone_counts": standalone,
        "dual_role_patient_count": int(len(dual_ids)),
        "dual_role_patient_ids": sorted(dual_ids),
        "dual_role_cross_regime_patient_ids": sorted(
            dual_regime_counts[dual_regime_counts > 1].index.astype(str)
        ),
        "dual_role_label_discordant_patient_ids": sorted(
            dual_label_counts[dual_label_counts > 1].index.astype(str)
        ),
        "contrast_primary_patients": int(contrast["role"].eq("primary").sum()),
        "contrast_metastatic_patients": int(contrast["role"].eq("metastatic").sum()),
    }


def validate_disjoint_contrast_arms(
    arms: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> None:
    """Require every role-by-regime inferential arm to contain distinct patients."""

    if set(arms) != set(MAIN_REGIMES):
        raise ValueError(f"Expected contrast regimes {MAIN_REGIMES}, got {sorted(arms)}")
    patient_sets: dict[str, set[str]] = {}
    for regime in MAIN_REGIMES:
        primary, metastatic = arms[regime]
        aim2_metastatic_transport._validate_patient_arm(primary, f"{regime} primary arm")
        aim2_metastatic_transport._validate_patient_arm(metastatic, f"{regime} metastatic arm")
        patient_sets[f"{regime}/primary"] = set(primary["patient_id"].astype(str))
        patient_sets[f"{regime}/metastatic"] = set(metastatic["patient_id"].astype(str))
    names = list(patient_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                raise ValueError(
                    f"Contrast arms {left} and {right} overlap on patients: "
                    f"{sorted(overlap)[:10]}"
                )


def regime_interaction_ci(
    arms: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = paths.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Direct four-arm bootstrap of technical-regime effect modification."""

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    validate_disjoint_contrast_arms(arms)

    def decrement(primary: pd.DataFrame, metastatic: pd.DataFrame) -> float:
        return float(aim2_loco_transport._auroc(metastatic) - aim2_loco_transport._auroc(primary))

    point_by_regime = {
        regime: decrement(*arms[regime])
        for regime in MAIN_REGIMES
    }
    point = point_by_regime[REGIME_REPAIRED] - point_by_regime[REGIME_APERIO]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        resampled: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for regime in MAIN_REGIMES:
            primary, metastatic = arms[regime]
            resampled[regime] = (
                aim2_metastatic_transport._resample_patient_arm(primary, rng),
                aim2_metastatic_transport._resample_patient_arm(metastatic, rng),
            )
        draws[index] = decrement(*resampled[REGIME_REPAIRED]) - decrement(
            *resampled[REGIME_APERIO]
        )
    interval = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    if interval[1] < 0:
        inference = "repaired-converted regime has a more negative decrement"
    elif interval[0] > 0:
        inference = "repaired-converted regime has a more positive decrement"
    else:
        inference = "technical-regime interaction not established"
    return {
        "estimand": (
            "[AUROC(M)-AUROC(P)]_repaired-converted - "
            "[AUROC(M)-AUROC(P)]_native-0.5016-MPP-SVS"
        ),
        "per_regime_delta_auroc": point_by_regime,
        "interaction_delta_auroc": float(point),
        "interaction_delta_auroc_ci": interval,
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "bootstrap_fraction_below_zero": float(np.mean(draws < 0)),
        "bootstrap_method": (
            "independent outcome-stratified patient resampling of all four disjoint "
            "role-by-regime arms; inference conditions on the frozen fitted ensemble"
        ),
        "four_arms_pairwise_patient_disjoint": True,
        "ci_excludes_zero": bool(interval[1] < 0 or interval[0] > 0),
        "inference": inference,
        "interpretation_guardrail": (
            "A confidence interval containing zero is absence of established interaction, "
            "not evidence of equivalence, invariance, or robustness."
        ),
    }


def _descriptive_block(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        **_counts(frame, include_slides=True),
        "inferential_status": "descriptive only; no AUROC or confidence interval",
    }
    if not frame.empty:
        result.update(
            {
                "patient_ids": sorted(frame["patient_id"].astype(str).tolist()),
                "mean_logit": float(frame["mean_logit"].mean()),
                "prob_raw": float(frame["prob_raw"].mean()),
            }
        )
    return result


def _metric_block(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    aim2_metastatic_transport._validate_patient_arm(frame, "standalone regime arm")
    return {
        **aim2_loco_transport.block_metrics(
            frame,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
        ),
        "class_counts": _counts(frame, include_slides=True),
        "score_scale": "native patient mean logit; sigmoid only for probability metrics",
    }


def _input_artifacts(cap: int) -> dict[str, Any]:
    return {
        "mapping_sources": {
            "ready_inventory": lineage.artifact_identity(INVENTORY_PATH),
            "rih_repair_state": lineage.artifact_identity(RIH_REPAIR_STATE_PATH),
            "current_protocol_context": lineage.artifact_identity(PROTOCOL_PATH),
            "historical_design_archive": lineage.artifact_identity(
                DESIGN_ARCHIVE_PATH
            ),
            "mapping_code": lineage.artifact_identity(Path(__file__)),
        },
        "target_manifests": {
            role: lineage.artifact_identity(aim2_loco_transport.target_manifest("RIH", role))
            for role in ("primary", "metastatic")
        },
        "fit_receipts": {
            str(seed): lineage.artifact_identity(aim2_loco_transport.fit_summary_path("RIH", seed, cap))
            for seed in aim2_loco_transport.SEEDS
        },
        "score_receipts": {
            role: {
                str(seed): lineage.artifact_identity(
                    aim2_loco_transport.score_receipt_path("RIH", seed, role, cap)
                )
                for seed in aim2_loco_transport.SEEDS
            }
            for role in ("primary", "metastatic")
        },
    }


def _assemble_patient_table(
    cap: int,
    mapping: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    role_tables: list[pd.DataFrame] = []
    tagged_manifests: dict[str, pd.DataFrame] = {}
    expected_seeds = set(aim2_loco_transport.SEEDS)
    for role in ("primary", "metastatic"):
        manifest = pd.read_csv(aim2_loco_transport.target_manifest("RIH", role))
        tagged = attach_regime_to_manifest(manifest, mapping, role=role)
        tagged_manifests[role] = tagged
        ensemble, per_seed = aim2_loco_transport.seed_ensemble("RIH", role, cap)
        if set(per_seed) != expected_seeds:
            raise RuntimeError(
                f"RIH/{role}: incomplete seed ensemble {sorted(per_seed)} != {sorted(expected_seeds)}"
            )
        patient = attach_patient_regime(ensemble, tagged, role=role)
        for seed in aim2_loco_transport.SEEDS:
            seed_table = attach_patient_regime(per_seed[seed], tagged, role=role)
            if not seed_table[["patient_id", "label", "technical_regime"]].equals(
                patient[["patient_id", "label", "technical_regime"]]
            ):
                raise RuntimeError(f"RIH/{role}/seed{seed} population differs from ensemble")
            patient[f"mean_logit_seed{seed}"] = seed_table["mean_logit"].to_numpy()
        role_tables.append(patient)
    table = pd.concat(role_tables, ignore_index=True)
    table = table.sort_values(["role", "patient_id"]).reset_index(drop=True)
    return table, tagged_manifests


def _seed_frame(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = frame.copy()
    out["mean_logit"] = out[f"mean_logit_seed{seed}"].to_numpy(dtype=float)
    out["prob_raw"] = sigmoid(out["mean_logit"].to_numpy())
    return out


def _per_seed_diagnostic(
    table: pd.DataFrame,
    dual_ids: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "inferential_role": (
            "none; training seeds share patients and source data and are only a "
            "computational-stability diagnostic"
        ),
        "seeds": {},
    }
    contrast = table[~table["patient_id"].isin(dual_ids)]
    for seed in aim2_loco_transport.SEEDS:
        seed_result: dict[str, Any] = {}
        for regime in MAIN_REGIMES:
            primary = _seed_frame(
                contrast[
                    contrast["role"].eq("primary")
                    & contrast["technical_regime"].eq(regime)
                ],
                seed,
            )
            metastatic = _seed_frame(
                contrast[
                    contrast["role"].eq("metastatic")
                    & contrast["technical_regime"].eq(regime)
                ],
                seed,
            )
            seed_result[regime] = {
                "primary_auroc": float(aim2_loco_transport._auroc(primary)),
                "metastatic_auroc": float(aim2_loco_transport._auroc(metastatic)),
                "delta_auroc": float(aim2_loco_transport._auroc(metastatic) - aim2_loco_transport._auroc(primary)),
            }
        seed_result["interaction_delta_auroc"] = float(
            seed_result[REGIME_REPAIRED]["delta_auroc"]
            - seed_result[REGIME_APERIO]["delta_auroc"]
        )
        result["seeds"][str(seed)] = seed_result
    return result


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def cmd_report(args: argparse.Namespace) -> None:
    lineage.lineage_name()
    destinations = [
        report_path(args.cap),
        patient_table_path(args.cap),
        patient_table_receipt_path(args.cap),
    ]
    for destination in destinations:
        lineage.ensure_absent(destination)

    inventory = pd.read_csv(INVENTORY_PATH, low_memory=False)
    repair_state = load_repair_state()
    mapping = build_slide_regime_map(inventory, repair_state)
    patient_table, tagged_manifests = _assemble_patient_table(args.cap, mapping)

    primary_ids = set(
        patient_table.loc[patient_table["role"].eq("primary"), "patient_id"].astype(str)
    )
    metastatic_ids = set(
        patient_table.loc[patient_table["role"].eq("metastatic"), "patient_id"].astype(str)
    )
    dual_ids = primary_ids & metastatic_ids
    population_audit = _validate_expected_population(patient_table, dual_ids)

    contrast = patient_table[~patient_table["patient_id"].isin(dual_ids)].copy()
    arms: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    standalone: dict[str, Any] = {}
    within_regime: dict[str, Any] = {}
    for regime in MAIN_REGIMES:
        primary_all = patient_table[
            patient_table["role"].eq("primary")
            & patient_table["technical_regime"].eq(regime)
        ].copy()
        metastatic_all = patient_table[
            patient_table["role"].eq("metastatic")
            & patient_table["technical_regime"].eq(regime)
        ].copy()
        standalone[regime] = {
            "label": REGIME_LABELS[regime],
            "primary": _metric_block(
                primary_all,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
            ),
            "metastatic": _metric_block(
                metastatic_all,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
            ),
        }

        primary = contrast[
            contrast["role"].eq("primary")
            & contrast["technical_regime"].eq(regime)
        ].copy()
        metastatic = contrast[
            contrast["role"].eq("metastatic")
            & contrast["technical_regime"].eq(regime)
        ].copy()
        arms[regime] = (primary, metastatic)
        within_regime[regime] = {
            "label": REGIME_LABELS[regime],
            "definition": (
                "RIH primary versus RIH metastatic after excluding every dual-role "
                "patient from both arms"
            ),
            "primary": _metric_block(
                primary,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
            ),
            "metastatic": _metric_block(
                metastatic,
                n_bootstrap=args.n_bootstrap,
                bootstrap_seed=args.bootstrap_seed,
            ),
            **aim2_metastatic_transport.contrast_ci(
                primary,
                metastatic,
                n_boot=args.n_bootstrap,
                seed=args.bootstrap_seed,
            ),
        }
    validate_disjoint_contrast_arms(arms)

    versa_primary = patient_table[
        patient_table["role"].eq("primary")
        & patient_table["technical_regime"].eq(REGIME_VERSA)
    ]
    versa_metastatic = patient_table[
        patient_table["role"].eq("metastatic")
        & patient_table["technical_regime"].eq(REGIME_VERSA)
    ]
    standalone[REGIME_VERSA] = {
        "label": REGIME_LABELS[REGIME_VERSA],
        "primary": _descriptive_block(versa_primary),
        "metastatic": _descriptive_block(versa_metastatic),
    }

    # Finish every inferential computation before publishing the first
    # immutable artifact.  A bootstrap/runtime failure must leave the new
    # lineage wholly untouched rather than strand a valid-looking partial
    # patient table that cannot be replaced on retry.
    interaction = regime_interaction_ci(
        arms,
        n_bootstrap=args.n_bootstrap,
        seed=args.bootstrap_seed,
    )
    inputs = _input_artifacts(args.cap)
    inventory_counts = mapping["technical_regime"].value_counts().to_dict()
    analytic_slide_counts = {
        f"{role}/{regime}": int(
            len(tagged_manifests[role][tagged_manifests[role]["technical_regime"].eq(regime)])
        )
        for role in ("primary", "metastatic")
        for regime in ALL_REGIMES
    }

    output_columns = [
        "patient_id",
        "role",
        "label",
        "mean_logit",
        "prob_raw",
        "technical_regime",
        "slide_count",
        "slide_ids",
        "native_mpps",
        "source_statuses",
        *(f"mean_logit_seed{seed}" for seed in aim2_loco_transport.SEEDS),
    ]
    output_table = patient_table[output_columns].copy()
    per_seed_diagnostic = _per_seed_diagnostic(patient_table, dual_ids)
    versa_contrast_summary = {
        "status": "descriptive only and excluded from every inferential contrast",
        "primary": _descriptive_block(versa_primary),
        "metastatic": _descriptive_block(versa_metastatic),
    }
    inventory_mpp_counts = {
        regime: _mpp_count(mapping[mapping["technical_regime"].eq(regime)])
        for regime in ALL_REGIMES
    }

    lineage.write_parquet_once(patient_table_path(args.cap), output_table)
    lineage.write_json_once(
        patient_table_receipt_path(args.cap),
        {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "lineage": lineage.lineage_name(),
            "cap": int(args.cap),
            "inputs": inputs,
            "population_audit": population_audit,
            "artifact": lineage.artifact_identity(patient_table_path(args.cap)),
        },
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "lineage": lineage.lineage_name(),
        "cap": int(args.cap),
        "question": (
            "Does RIH's primary-to-metastatic AUROC point contrast differ between the "
            "native 0.5016-MPP SVS and repaired-converted technical regimes?"
        ),
        "source": "frozen E2a RIH-held-out three-seed native-logit ensemble; 0 new fits",
        "seeds_complete": list(aim2_loco_transport.SEEDS),
        "input_artifacts": inputs,
        "mapping_audit": {
            "mapping_coverage": "369/369 ready RIH inventory slides mapped exactly once",
            "inventory_regime_counts": inventory_counts,
            "inventory_mpp_counts": inventory_mpp_counts,
            "eligible_scored_slide_counts": analytic_slide_counts,
            "repair_ledger_membership_matches_repaired_regime": True,
            "category_definition": {
                REGIME_APERIO: (
                    "original_standard, canonical_source=original, repair_state=not_selected, "
                    ".svs, native MPP 0.5016"
                ),
                REGIME_REPAIRED: (
                    "fixed_verified, canonical_source=fixed, repair_state=done, .tiff, "
                    "native MPP 0.189012 or 0.378024, and present in repair ledger"
                ),
                REGIME_VERSA: (
                    "the unique residual original-standard SL-348 .svs at native MPP 0.13899"
                ),
            },
        },
        "population_audit": population_audit,
        "patient_table": {
            "artifact": lineage.artifact_identity(patient_table_path(args.cap)),
            "receipt": lineage.artifact_identity(patient_table_receipt_path(args.cap)),
            "rows": int(len(output_table)),
        },
        "inference": {
            "n_bootstrap": int(args.n_bootstrap),
            "bootstrap_seed": int(args.bootstrap_seed),
            "sampling_unit": "patient",
            "score_scale": "native patient mean logit",
            "conditioning": "all intervals condition on the frozen fitted ensemble",
            "training_seed_role": "computational stability only; not inferential replication",
            "design_status": (
                "exploratory post-core robustness panel; not claimed as prospectively "
                "pre-specified"
            ),
            "multiplicity": "exploratory panel; no multiplicity adjustment",
        },
        "standalone_by_regime": standalone,
        "e2b_aligned_primary_to_metastatic": {
            "dual_role_rule": (
                "all 8 RIH dual-role patients excluded from both contrast roles before "
                "technical-regime stratification"
            ),
            "within_regime": within_regime,
            "versa": versa_contrast_summary,
            "four_arm_interaction": interaction,
        },
        "per_training_seed_diagnostic": per_seed_diagnostic,
        "guardrails": {
            "confirmatory_claim_allowed": False,
            "scanner_claim_allowed": False,
            "causal_specimen_role_claim_allowed": False,
            "equivalence_or_invariance_claim_allowed": False,
            "versa_inferential_claim_allowed": False,
            "technical_regime_interpretation": (
                "Platform, native resolution, acquisition era, staining/storage history, "
                "and file repair are confounded. Repaired-converted is a composite technical "
                "regime, not an identified scanner effect."
            ),
            "negative_interaction_result": (
                "Failure to establish an interaction does not prove the metastatic decrement "
                "is invariant to acquisition or processing."
            ),
        },
    }
    lineage.write_json_once(report_path(args.cap), report)

    print("E2d-6 RIH acquisition/processing-regime robustness")
    for regime in MAIN_REGIMES:
        block = within_regime[regime]
        ci = block["delta_auroc_ci"]
        print(
            f"  {REGIME_LABELS[regime]}: P n={block['primary']['n']}, "
            f"M n={block['metastatic']['n']}, AUROC(M-P) {block['delta_auroc']:+.4f} "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        )
    ci = interaction["interaction_delta_auroc_ci"]
    print(
        f"  interaction (repaired - native SVS): "
        f"{interaction['interaction_delta_auroc']:+.4f} "
        f"[{ci[0]:+.4f}, {ci[1]:+.4f}] -> {interaction['inference']}"
    )
    print("  Versa: one metastatic patient, no primary comparator; descriptive only.")
    print(f"Wrote {report_path(args.cap)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="write the immutable RIH regime report")
    report.add_argument("--cap", type=int, default=8192, choices=aim2_loco_transport.E2A_CAPS)
    report.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    report.add_argument("--bootstrap-seed", type=int, default=paths.BOOTSTRAP_SEED)
    report.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
