#!/usr/bin/env python3
"""Create the append-only, paper-ready final-v3 figure package.

The generator reads only the authoritative sources bound by
``reports/final_v3/report_bundle_receipt.json``.  It refuses to overwrite an
existing figure directory and publishes PDF/SVG vector figures plus 600-dpi
PNG/TIFF exports, source-data tables, legends, and a receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import tempfile
import textwrap
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
FINAL_V3 = REPO / "reports" / "final_v3"
PARENT_RECEIPT = FINAL_V3 / "report_bundle_receipt.json"
SUPERSEDED_OUTPUT = FINAL_V3 / "figures_v2"
DEFAULT_OUTPUT = FINAL_V3 / "figures_v3"
ADDITIONS = REPO / "reports" / "reruns" / "final_v3_additions_20260820"
AIM1_ROOT = ADDITIONS / "aim1_worklist"
AIM2_ROOT = ADDITIONS / "aim2_operational_v3"
AIM3_ROOT = ADDITIONS / "aim3_actionability"
AIM4_ROOT = ADDITIONS / "aim4_compressibility_v2"
CLAIM_MAP = (
    REPO
    / "reports"
    / "reruns"
    / "aim2_aim3_aim4_cap8192_numeric_validated_pathology_pending_20260820"
    / "claim_source_map.json"
)

PALETTE = {
    "ink": "#1B1F23",
    "muted": "#5B6573",
    "grid": "#D9DEE5",
    "light": "#F3F5F7",
    "wsi": "#0072B2",
    "clinical": "#C47A00",
    "fusion": "#7A5195",
    "random": "#777777",
    "supported": "#009E73",
    "metastatic": "#D55E00",
    "unresolved": "#E69F00",
    "sky": "#56B4E9",
    "pink": "#CC79A7",
}

METHOD_STYLE = {
    "random_expected": ("Random expected", PALETTE["random"], "--", "s"),
    "clinical_oof": ("Clinical", PALETTE["clinical"], "-.", "D"),
    "clinical_source_only": ("Clinical", PALETTE["clinical"], "-.", "D"),
    "wsi_declared_seed_median": ("WSI", PALETTE["wsi"], "-", "o"),
    "wsi_heldout_ensemble": ("WSI", PALETTE["wsi"], "-", "o"),
    "fusion_declared_seed_median": ("WSI + clinical", PALETTE["fusion"], ":", "^"),
    "fusion_source_only": ("WSI + clinical", PALETTE["fusion"], ":", "^"),
}

EXPECTED_BOUND_RECEIPT_SUFFIXES = {
    "aim1_worklist": "aim1_worklist/receipt.json",
    "aim2_operational": "aim2_operational_v3/receipt.json",
    "aim3_actionability": "aim3_actionability/receipt.json",
    "aim4_compressibility": "aim4_compressibility_v2/completion_receipt.json",
}
RASTER_DPI = 600


class FigureBuildError(RuntimeError):
    """Fail-closed figure generation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FigureBuildError(f"missing file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if relative_to is not None:
        result["relative_path"] = str(resolved.relative_to(relative_to.resolve()))
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FigureBuildError(f"expected JSON object: {path}")
    return value


def verify_identity(record: dict[str, Any]) -> Path:
    path = Path(str(record["path"]))
    observed = file_identity(path)
    if observed["sha256"] != record["sha256"] or observed["size_bytes"] != record["size_bytes"]:
        raise FigureBuildError(f"identity mismatch: {path}")
    return path


def setup_matplotlib() -> None:
    cache = Path(tempfile.gettempdir()) / "oceanpath_final_v3_figures_mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.6,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.7,
            "axes.labelcolor": PALETTE["ink"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "legend.fontsize": 6.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "lines.linewidth": 1.5,
            "lines.markersize": 4.8,
            "errorbar.capsize": 2.0,
        }
    )


def clean_axis(ax: Any, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, color=PALETTE["grid"], linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def panel_label(ax: Any, label: str, *, x: float = -0.12, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def save_figure(fig: Any, root: Path, stem: str, *, main: bool) -> list[Path]:
    folder = root / ("main" if main else "supplementary")
    folder.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in ("pdf", "svg"):
        path = folder / f"{stem}.{suffix}"
        fig.savefig(path)
        outputs.append(path)
    png = folder / f"{stem}.png"
    fig.savefig(png, dpi=RASTER_DPI, transparent=False)
    flatten_raster_to_rgb(png)
    outputs.append(png)
    tiff = folder / f"{stem}.tiff"
    fig.savefig(
        tiff,
        dpi=RASTER_DPI,
        transparent=False,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    flatten_raster_to_rgb(tiff)
    outputs.append(tiff)
    return outputs


def flatten_raster_to_rgb(path: Path) -> None:
    """Composite a rendered raster onto white and save a submission-safe RGB file."""

    with Image.open(path) as opened:
        rgba = opened.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
    save_kwargs: dict[str, Any] = {"dpi": (RASTER_DPI, RASTER_DPI)}
    if path.suffix.lower() in {".tif", ".tiff"}:
        save_kwargs["compression"] = "tiff_lzw"
    rgb.save(path, **save_kwargs)


def write_source_data(df: pd.DataFrame, root: Path, filename: str) -> Path:
    path = root / "source_data" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def ensure_ci(df: pd.DataFrame, point: str, low: str, high: str) -> None:
    valid = df[[point, low, high]].dropna()
    if not ((valid[low] <= valid[point]) & (valid[point] <= valid[high])).all():
        raise FigureBuildError(f"CI containment failed for {point}/{low}/{high}")


def verify_bound_sources() -> dict[str, Any]:
    bundle = load_json(PARENT_RECEIPT)
    if bundle.get("status") != "PASS":
        raise FigureBuildError("final-v3 parent receipt is not PASS")
    checks = bundle.get("independent_checks", {})
    if checks.get("rehash_mismatches") != 0:
        raise FigureBuildError("parent receipt records rehash mismatches")
    for name, suffix in EXPECTED_BOUND_RECEIPT_SUFFIXES.items():
        record = bundle["bound_receipts"][name]
        path = verify_identity(record)
        if not str(path).endswith(suffix):
            raise FigureBuildError(f"wrong authoritative receipt for {name}: {path}")
    for record in bundle["documents"].values():
        verify_identity(record)
    return bundle


def verify_component_receipt(path: Path, expected_status: set[str]) -> dict[str, Any]:
    payload = load_json(path)
    if str(payload.get("status", "")).upper() not in expected_status:
        raise FigureBuildError(f"component receipt status failure: {path}")
    for section in ("artifacts", "outputs"):
        value = payload.get(section)
        if isinstance(value, list):
            for record in value:
                if isinstance(record, dict) and {"path", "sha256", "size_bytes"} <= set(record):
                    verify_identity(record)
        elif isinstance(value, dict):
            for record in value.values():
                if isinstance(record, dict) and {"path", "sha256", "size_bytes"} <= set(record):
                    verify_identity(record)
    return payload


def authoritative_legacy_sources() -> dict[str, Path]:
    claim_map = load_json(CLAIM_MAP)
    sources: dict[str, Path] = {}
    for name, record in claim_map["sources"].items():
        if isinstance(record, dict) and {"path", "sha256"} <= set(record):
            path = Path(record["path"])
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise FigureBuildError(f"legacy source identity mismatch: {name}")
            sources[name] = path
    return sources


def manual_report_frames(bundle: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Values whose corrected Aim-1 authority is the sealed Results Markdown."""
    results_path = FINAL_V3 / "Results.md"
    if sha256_file(results_path) != bundle["documents"]["Results.md"]["sha256"]:
        raise FigureBuildError("sealed final-v3 Results.md identity changed")
    text = results_path.read_text(encoding="utf-8")
    required_fragments = [
        "| **D, MSS/pMMR and BRAF-WT** | **1,129 (514)** | **0.688**",
        "| AUROC | 0.5387 | 0.6643 [0.6582, 0.6731] | +0.1256",
        "adjusted BRAF main effect is +1.676",
        "| Virchow2-CLS, cap 4,096 | 0.6717 | 0.7103",
    ]
    if not all(fragment in text for fragment in required_fragments):
        raise FigureBuildError("manual Aim-1 plotting frame no longer matches Results.md")

    challenge = pd.DataFrame(
        [
            ("A", "All primary", 1486, 604, 0.664, 0.636, 0.693),
            ("A-complete", "Complete MSI/BRAF labels", 1387, 555, 0.661, 0.630, 0.689),
            ("B", "MSS/pMMR", 1262, 543, 0.674, 0.645, 0.701),
            ("C", "BRAF wild type", 1247, 568, 0.681, 0.651, 0.710),
            ("D", "MSS/pMMR + BRAF wild type", 1129, 514, 0.688, 0.656, 0.717),
        ],
        columns=["set", "population", "n", "n_mutant", "auroc", "ci_low", "ci_high"],
    )
    model_deltas = pd.DataFrame(
        [
            ("WSI − clinical", 0.1256, 0.0856, 0.1644),
            ("Fusion − clinical", 0.1176, 0.0822, 0.1528),
            ("Fusion − WSI", -0.0080, -0.0150, -0.0007),
        ],
        columns=["contrast", "delta_auroc", "ci_low", "ci_high"],
    )
    context = pd.DataFrame(
        [
            ("BRAF mutant", 1.676, 0.617, 2.635),
            ("MSI/dMMR", 0.314, -0.722, 1.788),
            ("BRAF × MSI", -1.396, -3.288, 0.402),
        ],
        columns=["term", "logit_effect", "ci_low", "ci_high"],
    )
    robustness = pd.DataFrame(
        [
            ("UNIv1, cap 8,192", 0.6643, 0.6877, 0.0222, 0.0103, 0.0344),
            ("UNIv1, cap 4,096", 0.6669, 0.6868, 0.0215, 0.0099, 0.0340),
            ("Virchow2-CLS, cap 4,096", 0.6717, 0.7103, 0.0383, 0.0252, 0.0533),
        ],
        columns=["arm", "auroc_A", "auroc_D", "delta_D_minus_A_complete", "ci_low", "ci_high"],
    )
    full_challenge = pd.DataFrame(
        [
            ("A", "All primary", 1486, 604, 0.664, 0.636, 0.693),
            ("A-complete", "Complete labels", 1387, 555, 0.661, 0.630, 0.689),
            ("B", "MSS/pMMR", 1262, 543, 0.674, 0.645, 0.701),
            ("C", "BRAF wild type", 1247, 568, 0.681, 0.651, 0.710),
            ("D", "MSS/pMMR + BRAF WT", 1129, 514, 0.688, 0.656, 0.717),
            ("E", "Colon", 1038, 433, 0.670, 0.637, 0.703),
            ("F", "Rectum", 435, 164, 0.648, 0.594, 0.701),
            ("G", "Stage known", 1270, 508, 0.675, 0.644, 0.705),
            ("H", "Stage IV", 156, 74, 0.692, 0.605, 0.772),
            ("I", "Right/proximal", 319, 147, 0.634, 0.570, 0.693),
            ("J", "Left/distal", 694, 256, 0.659, 0.617, 0.701),
            ("K", "Transverse", 53, 17, 0.638, 0.453, 0.809),
        ],
        columns=["set", "population", "n", "n_mutant", "auroc", "ci_low", "ci_high"],
    )
    for frame in (challenge, model_deltas, context, robustness, full_challenge):
        ensure_ci(frame, frame.columns[-3], frame.columns[-2], frame.columns[-1])
    return {
        "challenge": challenge,
        "model_deltas": model_deltas,
        "context": context,
        "robustness": robustness,
        "full_challenge": full_challenge,
    }


def load_and_validate_data() -> dict[str, Any]:
    bundle = verify_bound_sources()
    verify_component_receipt(AIM1_ROOT / "receipt.json", {"PASS"})
    verify_component_receipt(AIM2_ROOT / "receipt.json", {"PASS"})
    verify_component_receipt(AIM3_ROOT / "receipt.json", {"PASS"})
    verify_component_receipt(AIM4_ROOT / "completion_receipt.json", {"COMPLETE"})
    validation = load_json(AIM1_ROOT / "validation.json")
    if validation.get("overall", {}).get("status") != "PASS":
        raise FigureBuildError("Aim-1 worklist validation is not PASS")

    worklist = pd.read_csv(AIM1_ROOT / "worklist_results.csv")
    comparisons = pd.read_csv(AIM1_ROOT / "comparisons.csv")
    if worklist.shape != (265, 29) or comparisons.shape != (105, 17):
        raise FigureBuildError("unexpected Aim-1 worklist table shape")
    key = ["analysis", "population", "method", "capacity_nominal"]
    if worklist.duplicated(key).any():
        raise FigureBuildError("duplicate Aim-1 worklist keys")
    ensure_ci(worklist, "capture", "capture_ci_low", "capture_ci_high")
    for _, group in worklist.groupby(["analysis", "population", "method"]):
        if len(group) > 1 and (np.diff(group.sort_values("capacity_nominal")["capture"]) < -1e-12).any():
            raise FigureBuildError("non-monotone worklist capture curve")

    agreement = pd.read_csv(AIM2_ROOT / "agreement_metrics.csv")
    census = pd.read_csv(AIM2_ROOT / "multi_slide_census.csv")
    random_draws = pd.read_parquet(AIM2_ROOT / "random_slide_draws.parquet")
    technical = pd.read_csv(AIM2_ROOT / "technical_associations.csv")
    technical_patients = pd.read_parquet(AIM2_ROOT / "technical_patient_metrics.parquet")
    deployment = pd.read_csv(AIM2_ROOT / "deployment_matrix.csv")
    aim2_results = load_json(AIM2_ROOT / "results.json")
    if agreement.shape != (120, 10) or random_draws.shape != (40000, 7) or deployment.shape != (4, 7):
        raise FigureBuildError("unexpected Aim-2 operational table shape")
    if census[["n_patients", "n_slides", "n_one_slide_patients", "n_two_slide_patients", "n_three_slide_patients", "n_multislide_patients"]].sum().tolist() != [1486, 1642, 1331, 154, 1, 155]:
        raise FigureBuildError("multi-slide census mismatch")
    ensure_ci(agreement, "estimate", "bootstrap_ci_low", "bootstrap_ci_high")
    if not np.allclose(
        technical_patients["patch_count_log2_range"],
        technical_patients["tissue_area_log2_range"],
        atol=1e-12,
    ):
        raise FigureBuildError("technical range identity changed")

    actionability = pd.read_csv(AIM3_ROOT / "actionability_ladder.csv")
    if actionability.shape[0] != 8 or set(actionability["direct_clinical_decision_supported"]) != {"No"}:
        raise FigureBuildError("Aim-3 actionability contract mismatch")

    compress = pd.read_csv(AIM4_ROOT / "per_target_metrics.csv")
    predictions = pd.read_parquet(AIM4_ROOT / "patient_predictions.parquet")
    compress_results = load_json(AIM4_ROOT / "results.json")
    model_selection = pd.read_csv(AIM4_ROOT / "model_selection.csv")
    if compress.shape != (168, 6) or predictions.shape[0] != 1486 or model_selection.shape != (80, 6):
        raise FigureBuildError("Aim-4 compressibility table shape mismatch")
    ensure_ci(compress, "point", "ci_low", "ci_high")
    if compress.duplicated(["model", "summary", "metric"]).any():
        raise FigureBuildError("duplicate Aim-4 metric keys")
    selected = model_selection[model_selection["selected"]]
    if selected.groupby(["model", "outer_target"]).size().ne(1).any() or len(selected) != 16:
        raise FigureBuildError("Aim-4 alpha-selection contract mismatch")

    legacy = authoritative_legacy_sources()
    required_legacy = {
        "aim2_e2a",
        "aim2_e2b",
        "aim2_e2c",
        "aim3_fixed",
        "aim3_repeated_report",
        "aim4_v2_report",
        "aim4_v2_prototype_table",
        "aim4_stability_results",
    }
    if not required_legacy <= set(legacy):
        raise FigureBuildError("claim map lacks required figure sources")

    return {
        "bundle": bundle,
        "manual": manual_report_frames(bundle),
        "worklist": worklist,
        "comparisons": comparisons,
        "agreement": agreement,
        "census": census,
        "random_draws": random_draws,
        "technical": technical,
        "technical_patients": technical_patients,
        "deployment": deployment,
        "aim2_results": aim2_results,
        "actionability": actionability,
        "compress": compress,
        "predictions": predictions,
        "compress_results": compress_results,
        "model_selection": model_selection,
        "legacy_paths": legacy,
        "e2a": load_json(legacy["aim2_e2a"]),
        "e2b": load_json(legacy["aim2_e2b"]),
        "e2c": load_json(legacy["aim2_e2c"]),
        "aim3_fixed": load_json(legacy["aim3_fixed"]),
        "aim3_repeated": load_json(legacy["aim3_repeated_report"]),
        "aim4_report": load_json(legacy["aim4_v2_report"]),
        "aim4_prototypes": pd.read_csv(legacy["aim4_v2_prototype_table"]),
        "aim4_stability": load_json(legacy["aim4_stability_results"]),
    }


def source_frames(data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    e2a_rows: list[dict[str, Any]] = []
    for cohort, values in data["e2a"]["targets"].items():
        for population, field in (("Overall", "primary_overall"), ("Set D", "primary_D_subset")):
            item = values[field]
            e2a_rows.append(
                {
                    "cohort": cohort,
                    "population": population,
                    "n": item["n"],
                    "n_mutant": item["n_mutant"],
                    "auroc": item["auroc"],
                    "ci_low": item["auroc_ci"][0],
                    "ci_high": item["auroc_ci"][1],
                }
            )
    external_primary = pd.DataFrame(e2a_rows)

    metastatic_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    for cohort, values in data["e2b"]["targets"].items():
        item = values["metastatic_overall"]
        metastatic_rows.append(
            {
                "cohort": "SR1482" if cohort == "SurGen" else cohort,
                "auroc": item["auroc"],
                "ci_low": item["auroc_ci"][0],
                "ci_high": item["auroc_ci"][1],
                "n": item["n"],
                "n_mutant": item["n_mutant"],
            }
        )
        contrast = values["primary_vs_metastatic"]
        role_rows.append(
            {
                "cohort": cohort,
                "delta": contrast["delta_auroc"],
                "ci_low": contrast["delta_auroc_ci"][0],
                "ci_high": contrast["delta_auroc_ci"][1],
            }
        )
    macro = data["e2b"]["conclusion"]
    metastatic_rows.append(
        {
            "cohort": "Equal-cohort macro",
            "auroc": macro["metastatic_macro_auroc"],
            "ci_low": macro["metastatic_macro_ci"][0],
            "ci_high": macro["metastatic_macro_ci"][1],
            "n": np.nan,
            "n_mutant": np.nan,
        }
    )
    combined = data["e2b"]["combined_decrement"]
    role_rows.append(
        {
            "cohort": "Equal-cohort",
            "delta": combined["delta_auroc"],
            "ci_low": combined["delta_auroc_ci"][0],
            "ci_high": combined["delta_auroc_ci"][1],
        }
    )
    metastatic = pd.DataFrame(metastatic_rows)
    role_deltas = pd.DataFrame(role_rows)

    adapter_rows: list[dict[str, Any]] = []
    for cohort, values in data["e2c"]["cohorts"].items():
        for budget in (2, 4, 8):
            for arm in ("S1", "S2"):
                item = values["contrasts"][f"{arm}_k{budget}_minus_S0"]
                adapter_rows.append(
                    {
                        "cohort": cohort,
                        "support_arm": arm,
                        "labels_per_class": budget,
                        "delta_auroc": item["delta"],
                        "ci_low": item["ci"][0],
                        "ci_high": item["ci"][1],
                    }
                )
    sparse_adapter = pd.DataFrame(adapter_rows)

    rung_map = {
        "codon": "Codon 12 vs other KRAS",
        "g12d_broad": "G12D vs other KRAS",
        "allele1": "G12D vs other G12",
        "allele2": "G12V vs other G12",
        "g12c": "G12C vs other G12",
    }
    aim3_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    fixed = data["aim3_fixed"]
    for key, label in rung_map.items():
        pair = fixed["pairs"][key]
        control_key = pair["control"]
        fine = fixed["tasks"][key]
        control = fixed["tasks"][control_key]
        aim3_rows.extend(
            [
                {
                    "task_key": key,
                    "task": label,
                    "series": "Fine task",
                    "auroc": fine["auroc"],
                    "ci_low": fine["ci95"][0],
                    "ci_high": fine["ci95"][1],
                },
                {
                    "task_key": key,
                    "task": label,
                    "series": "Matched control",
                    "auroc": control["auroc"],
                    "ci_low": control["ci95"][0],
                    "ci_high": control["ci95"][1],
                },
            ]
        )
        family = pair["familywise_ceiling_sensitivity"]
        gate_rows.append(
            {
                "task_key": key,
                "task": label,
                "fine_upper_99": family["fine_99_upper"],
                "fine_gate_pass": family["fine_99_upper"] < 0.60,
                "control_lower_99": family["control_99_lower"],
                "control_gate_pass": family["control_99_lower"] > 0.50,
                "delta_lower_99": family["delta_99_lower"],
                "delta_gate_pass": family["delta_99_lower"] > 0.0,
                "overall_pass": family["pass"],
            }
        )
    aim3_tasks = pd.DataFrame(aim3_rows)
    aim3_gates = pd.DataFrame(gate_rows)

    repeated_rows: list[dict[str, Any]] = []
    for key, label in rung_map.items():
        rung = data["aim3_repeated"]["rungs"][key]
        for draw, verdict in zip((20260823, 20260824, 20260825), rung["draw_verdicts"], strict=True):
            repeated_rows.append(
                {
                    "task_key": key,
                    "task": label,
                    "draw": str(draw),
                    "verdict": verdict,
                    "consensus": rung["consensus_verdict"],
                }
            )
    aim3_repeated = pd.DataFrame(repeated_rows)

    proto = data["aim4_prototypes"].copy()
    if proto.shape != (32, 21) or set(proto["prototype"]) != set(range(32)):
        raise FigureBuildError("Aim-4 prototype inventory mismatch")
    stability_rows = [
        row
        for row in data["aim4_stability"]["downstream_abundance_claim_stability"]["association_effects"]
        if row["anchor_prototype"] in {17, 28} and row["variant"] != "canonical_k32"
    ]
    stability = pd.DataFrame(stability_rows)
    if stability.shape[0] != 18:
        raise FigureBuildError("Aim-4 stability grid mismatch")

    candidate_rows: list[dict[str, Any]] = []
    report = data["aim4_report"]
    for prototype, quantity in ((17, "abundance"), (28, "abundance"), (28, "attn_mass_mean"), (5, "attn_mass_mean")):
        effects = report["readout"][str(prototype)]["specificity_effects"][quantity]
        for population in ("A", "D"):
            item = effects[population]
            candidate_rows.append(
                {
                    "prototype": f"p{prototype}",
                    "quantity": "Attention" if quantity == "attn_mass_mean" else "Abundance",
                    "population": population,
                    "auroc": item["auc"],
                    "ci_low": item["ci_low"],
                    "ci_high": item["ci_high"],
                    "q": item["q"],
                    "significant": item["significant"],
                }
            )
    aim4_candidates = pd.DataFrame(candidate_rows)

    for frame, point in (
        (external_primary, "auroc"),
        (metastatic, "auroc"),
        (role_deltas, "delta"),
        (sparse_adapter, "delta_auroc"),
        (aim3_tasks, "auroc"),
        (aim4_candidates, "auroc"),
    ):
        ensure_ci(frame, point, "ci_low", "ci_high")

    return {
        **data["manual"],
        "external_primary": external_primary,
        "metastatic": metastatic,
        "role_deltas": role_deltas,
        "sparse_adapter": sparse_adapter,
        "aim3_tasks": aim3_tasks,
        "aim3_gates": aim3_gates,
        "aim3_repeated": aim3_repeated,
        "actionability": data["actionability"],
        "worklist": data["worklist"],
        "comparisons": data["comparisons"],
        "agreement": data["agreement"],
        "census": data["census"],
        "random_draws": data["random_draws"],
        "technical": data["technical"],
        "technical_patients": data["technical_patients"],
        "deployment": data["deployment"],
        "aim2_results": data["aim2_results"],
        "aim4_prototypes": proto,
        "aim4_stability_grid": stability,
        "aim4_candidates": aim4_candidates,
        "compress": data["compress"],
        "predictions": data["predictions"],
        "compress_results": data["compress_results"],
    }


def forest_points(
    ax: Any,
    frame: pd.DataFrame,
    *,
    point: str,
    low: str,
    high: str,
    labels: Iterable[str],
    color: str,
    marker: str = "o",
    reference: float | None = None,
    xlim: tuple[float, float] | None = None,
    xlabel: str = "",
    value_format: str = ".3f",
) -> None:
    values = frame.reset_index(drop=True)
    y = np.arange(len(values))[::-1]
    x = values[point].to_numpy(float)
    left = x - values[low].to_numpy(float)
    right = values[high].to_numpy(float) - x
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([left, right]),
        fmt=marker,
        color=color,
        ecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.55,
        markersize=5.8,
        elinewidth=1.2,
        capsize=2.3,
        zorder=3,
    )
    if reference is not None:
        ax.axvline(reference, color=PALETTE["random"], linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(y, list(labels))
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    clean_axis(ax, grid_axis="x")
    span = (xlim[1] - xlim[0]) if xlim else max(float(np.ptp(x)), 0.1)
    for yi, xi in zip(y, x, strict=True):
        ax.text(
            xi + span * 0.018,
            yi,
            f"{xi:{value_format}}",
            va="center",
            ha="left",
            fontsize=6.6,
            color=PALETTE["ink"],
        )


def draw_study_schematic(ax: Any) -> pd.DataFrame:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ax.set_axis_off()
    boxes = [
        (0.02, 0.58, 0.23, 0.30, "Development data", "1,486 patients\n1,642 H&E slides\n604 KRAS-mutant"),
        (0.38, 0.58, 0.23, 0.30, "Frozen WSI model", "4 source cohorts\n5 folds × 3 seeds\npatient-level score"),
        (0.74, 0.58, 0.23, 0.30, "Clinical question", "Can scores order a\nretrospective molecular-\ntesting worklist?"),
    ]
    for x, y, w, h, title, body in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=PALETTE["light"],
            edgecolor=PALETTE["grid"],
            linewidth=0.9,
        )
        ax.add_patch(patch)
        ax.text(x + 0.02, y + h - 0.07, title, fontsize=8.0, fontweight="bold", va="top")
        ax.text(x + 0.02, y + h - 0.14, body, fontsize=6.8, va="top", linespacing=1.35)
    for x0, x1 in ((0.25, 0.38), (0.61, 0.74)):
        ax.add_patch(
            FancyArrowPatch(
                (x0 + 0.01, 0.73),
                (x1 - 0.01, 0.73),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color=PALETTE["muted"],
            )
        )
    aim_labels = [
        (0.02, "Aim 1", "Validity +\nworklist enrichment"),
        (0.27, "Aim 2", "Primary transport +\ndeployment boundary"),
        (0.52, "Aim 3", "Molecular-resolution\nceiling"),
        (0.77, "Aim 4", "Numeric prototypes +\npartial compression"),
    ]
    for x, aim, body in aim_labels:
        ax.text(x, 0.34, aim, color=PALETTE["wsi"], fontsize=7.2, fontweight="bold")
        ax.text(x, 0.28, body, fontsize=6.5, va="top", linespacing=1.25)
    ax.text(
        0.5,
        0.04,
        "Ranking association only  •  Molecular testing remains the reference standard",
        ha="center",
        fontsize=7.2,
        color=PALETTE["muted"],
        fontweight="semibold",
    )
    return pd.DataFrame(
        [
            ("patients", 1486),
            ("slides", 1642),
            ("KRAS_mutant_patients", 604),
            ("source_cohorts", 4),
            ("cross_validation_folds", 5),
            ("model_seeds", 3),
        ],
        columns=["quantity", "value"],
    )


def figure_1_foundation(frames: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2, 5.9))
    ax_a = fig.add_axes((0.06, 0.53, 0.90, 0.37))
    ax_b = fig.add_axes((0.17, 0.10, 0.25, 0.34))
    ax_c = fig.add_axes((0.50, 0.10, 0.18, 0.34))
    ax_d = fig.add_axes((0.78, 0.10, 0.20, 0.34))

    schematic = draw_study_schematic(ax_a)
    panel_label(ax_a, "A", x=-0.02, y=1.02)
    ax_a.set_title("Study logic and evidence boundary", loc="left", pad=2)

    challenge = frames["challenge"].copy()
    compact_population = {
        "A": "All primary",
        "A-complete": "Labels complete",
        "B": "MSS/pMMR",
        "C": "BRAF WT",
        "D": "MSS + BRAF WT",
    }
    labels = [
        f"{'A-c' if r.set == 'A-complete' else r.set}: {compact_population[r.set]}\n"
        f"{r.n:,} / {r.n_mutant} mutant"
        for r in challenge.itertuples()
    ]
    forest_points(
        ax_b,
        challenge,
        point="auroc",
        low="ci_low",
        high="ci_high",
        labels=labels,
        color=PALETTE["wsi"],
        reference=0.5,
        xlim=(0.49, 0.76),
        xlabel="AUROC (median-seed E0)",
    )
    ax_b.set_title("Molecular-context sets", loc="left")
    ax_b.get_yticklabels()[-1].set_fontweight("bold")
    panel_label(ax_b, "B", x=-0.26)

    deltas = frames["model_deltas"].copy()
    forest_points(
        ax_c,
        deltas,
        point="delta_auroc",
        low="ci_low",
        high="ci_high",
        labels=deltas["contrast"].str.replace("clinical", "clinic", regex=False),
        color=PALETTE["fusion"],
        reference=0.0,
        xlim=(-0.035, 0.19),
        xlabel="Paired ΔAUROC",
    )
    ax_c.set_title("Model increments", loc="left")
    panel_label(ax_c, "C", x=-0.32)

    context = frames["context"].copy()
    context_labels = [
        "BRAF mutant\nwithin MSS/pMMR",
        "MSI/dMMR\nwithin BRAF-WT",
        "BRAF × MSI",
    ]
    forest_points(
        ax_d,
        context,
        point="logit_effect",
        low="ci_low",
        high="ci_high",
        labels=context_labels,
        color=PALETTE["metastatic"],
        reference=0.0,
        xlim=(-3.7, 3.2),
        xlabel="Adjusted native-logit effect",
        value_format="+.2f",
    )
    ax_d.set_title("WT-context model", loc="left")
    panel_label(ax_d, "D", x=-0.30)

    fig.suptitle(
        "Figure 1 | Gene-level WSI ranking persists across measured context checks",
        fontsize=10.8,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    sources = [
        write_source_data(schematic, root, "figure_1A_study_counts.csv"),
        write_source_data(challenge, root, "figure_1B_challenge_sets.csv"),
        write_source_data(deltas, root, "figure_1C_model_contrasts.csv"),
        write_source_data(context, root, "figure_1D_adjusted_context.csv"),
    ]
    outputs = save_figure(fig, root, "Figure_1_foundation", main=True)
    plt.close(fig)
    return outputs, sources


def plot_worklist_curve(ax: Any, frame: pd.DataFrame, *, title: str, macro: bool) -> None:
    from matplotlib.ticker import PercentFormatter

    for method in frame["method"].unique():
        item = frame[frame["method"] == method].sort_values("capacity_nominal")
        label, color, linestyle, marker = METHOD_STYLE[method]
        x = item["capacity_nominal"].to_numpy(float)
        y = item["capture"].to_numpy(float)
        ax.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=1.55,
            markersize=4.4,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=label,
            zorder=3,
        )
        if method != "random_expected" and not macro:
            ax.fill_between(
                x,
                item["capture_ci_low"].to_numpy(float),
                item["capture_ci_high"].to_numpy(float),
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=1,
            )
    ax.axvline(0.30, color=PALETTE["grid"], linewidth=0.9, zorder=0)
    ax.set_xlim(0.08, 0.52)
    ax.set_ylim(0.04, 0.74)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Worklist capacity")
    ax.set_ylabel("KRAS-mutant patients captured")
    ax.set_title(title, loc="left")
    clean_axis(ax)
    ax.legend(loc="upper left", ncol=2, handlelength=2.2, columnspacing=0.9)


def figure_2_worklist(frames: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig = plt.figure(figsize=(7.2, 6.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    ax_a, ax_b, ax_c, ax_d = axes

    worklist = frames["worklist"]
    e0_methods = ["random_expected", "clinical_oof", "wsi_declared_seed_median", "fusion_declared_seed_median"]
    e0 = worklist[
        (worklist["analysis"] == "E0_frozen_OOF")
        & (worklist["population"] == "A_all_primary")
        & worklist["method"].isin(e0_methods)
    ].copy()
    external_methods = ["random_expected", "clinical_source_only", "wsi_heldout_ensemble", "fusion_source_only"]
    macro = worklist[
        (worklist["analysis"] == "E2a_heldout_primary")
        & (worklist["population"] == "equal_target_macro")
        & worklist["method"].isin(external_methods)
    ].copy()
    plot_worklist_curve(ax_a, e0, title="Development OOF ranking", macro=False)
    ax_a.text(
        0.98,
        0.05,
        "+14.6 pp vs random\n+12.9 pp vs clinical",
        transform=ax_a.transAxes,
        fontsize=6.7,
        color=PALETTE["wsi"],
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["grid"]},
    )
    panel_label(ax_a, "A")
    plot_worklist_curve(ax_b, macro, title="Held-out targets (equal-target)", macro=True)
    ax_b.text(
        0.5,
        -0.23,
        "Clinical and fusion comparators are post-hoc, source-only",
        transform=ax_b.transAxes,
        ha="center",
        fontsize=6.4,
        color=PALETTE["muted"],
    )
    panel_label(ax_b, "B")

    top30 = worklist[
        (worklist["analysis"] == "E2a_heldout_primary")
        & np.isclose(worklist["capacity_nominal"], 0.30)
        & worklist["population"].isin(["CPTAC", "RIH", "SurGen", "TCGA", "equal_target_macro"])
        & worklist["method"].isin(external_methods)
    ].copy()
    cohorts = ["CPTAC", "RIH", "SurGen", "TCGA", "equal_target_macro"]
    display = {"equal_target_macro": "Equal-target"}
    offsets = dict(zip(external_methods, [-0.24, -0.08, 0.08, 0.24], strict=True))
    ybase = np.arange(len(cohorts))[::-1]
    for method in external_methods:
        label, color, _, marker = METHOD_STYLE[method]
        item = top30[top30["method"] == method].set_index("population").loc[cohorts]
        y = ybase + offsets[method]
        x = item["capture"].to_numpy(float)
        if method == "random_expected":
            ax_c.plot(x, y, marker, color=color, markerfacecolor="white", label=label, linestyle="none")
        else:
            ax_c.errorbar(
                x,
                y,
                xerr=np.vstack(
                    [x - item["capture_ci_low"].to_numpy(float), item["capture_ci_high"].to_numpy(float) - x]
                ),
                fmt=marker,
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                elinewidth=0.9,
                markersize=4.4,
                label=label,
            )
    ax_c.set_yticks(ybase, [display.get(x, x) for x in cohorts])
    ax_c.set_xlim(0.20, 0.67)
    ax_c.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_c.set_xlabel("Mutants captured at top 30%")
    ax_c.set_title("Held-out worklist capture by target", loc="left")
    clean_axis(ax_c, grid_axis="x")
    ax_c.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        columnspacing=0.7,
        handletextpad=0.4,
    )
    panel_label(ax_c, "C")

    external = frames["external_primary"].copy()
    cohorts4 = ["CPTAC", "RIH", "SurGen", "TCGA"]
    ybase = np.arange(4)[::-1]
    for population, offset, color, marker in (
        ("Overall", 0.12, PALETTE["wsi"], "o"),
        ("Set D", -0.12, PALETTE["supported"], "s"),
    ):
        item = external[external["population"] == population].set_index("cohort").loc[cohorts4]
        x = item["auroc"].to_numpy(float)
        ax_d.errorbar(
            x,
            ybase + offset,
            xerr=np.vstack([x - item["ci_low"].to_numpy(float), item["ci_high"].to_numpy(float) - x]),
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=1.0,
            markersize=5.2,
            label=population,
        )
    ax_d.axvline(0.5, color=PALETTE["random"], linestyle="--", linewidth=1.0)
    ax_d.set_yticks(ybase, cohorts4)
    ax_d.set_xlim(0.47, 0.89)
    ax_d.set_xlabel("Held-out AUROC")
    ax_d.set_title("Primary-tumour ranking transports", loc="left")
    clean_axis(ax_d, grid_axis="x")
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    panel_label(ax_d, "D")

    fig.suptitle(
        "Figure 2 | WSI scores enrich retrospective molecular-testing worklists",
        fontsize=10.8,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    sources = [
        write_source_data(e0, root, "figure_2A_development_worklist.csv"),
        write_source_data(macro, root, "figure_2B_equal_target_worklist.csv"),
        write_source_data(top30, root, "figure_2C_heldout_top30.csv"),
        write_source_data(external, root, "figure_2D_external_primary.csv"),
    ]
    outputs = save_figure(fig, root, "Figure_2_worklist_enrichment", main=True)
    plt.close(fig)
    return outputs, sources


def draw_deployment_matrix(ax: Any, deployment: pd.DataFrame) -> pd.DataFrame:
    from matplotlib.patches import Rectangle

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 4.45)
    ax.set_axis_off()
    settings = [
        "Development\nprimary",
        "Held-out\nprimary",
        "Metastatic\ntissue",
        "Exact codon /\nsubstitution",
    ]
    statuses = [
        "Worklist / QA",
        "Prospective\nvalidation",
        "Not supported",
        "Molecular ID\nrequired",
    ]
    evidence = [
        "Modest ranking",
        "Ranking transported",
        "Fixed gate failed",
        "Ranking not established",
    ]
    colors = [PALETTE["wsi"], PALETTE["supported"], PALETTE["metastatic"], PALETTE["unresolved"]]
    ax.text(0.01, 4.16, "Setting", fontweight="bold", fontsize=7.0)
    ax.text(0.48, 4.16, "Recommendation", fontweight="bold", fontsize=7.0)
    for index, (setting, status, _summary, color) in enumerate(
        zip(settings, statuses, evidence, colors, strict=True)
    ):
        y = 3.45 - index
        ax.add_patch(Rectangle((0, y - 0.39), 1, 0.78, facecolor="white", edgecolor=PALETTE["grid"], linewidth=0.7))
        ax.add_patch(Rectangle((0, y - 0.39), 0.018, 0.78, facecolor=color, edgecolor="none"))
        ax.text(0.04, y, setting, va="center", fontsize=6.6, fontweight="semibold")
        ax.text(0.48, y, status, va="center", fontsize=6.4, color=color, fontweight="semibold")
    compact = deployment.copy()
    compact.insert(0, "display_evidence", evidence)
    compact.insert(1, "display_status", statuses)
    return compact


def figure_3_deployment(frames: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2, 7.65), layout="constrained")
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.92, 1.0, 1.18],
        width_ratios=[1.0, 1.15],
        hspace=0.30,
        wspace=0.30,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    ax_e = fig.add_subplot(grid[2, 0])
    ax_f = fig.add_subplot(grid[2, 1])

    metastatic = frames["metastatic"].copy()
    forest_points(
        ax_a,
        metastatic,
        point="auroc",
        low="ci_low",
        high="ci_high",
        labels=metastatic["cohort"],
        color=PALETTE["metastatic"],
        reference=0.5,
        xlim=(0.36, 0.79),
        xlabel="Metastatic AUROC",
    )
    ax_a.set_title("Metastatic ranking gate not met", loc="left")
    panel_label(ax_a, "A", x=-0.23)

    role = frames["role_deltas"].copy()
    display_role = role["cohort"].replace({"SurGen": "SR1482", "Equal-cohort": "Equal-cohort macro"})
    forest_points(
        ax_b,
        role,
        point="delta",
        low="ci_low",
        high="ci_high",
        labels=display_role,
        color=PALETTE["metastatic"],
        reference=0.0,
        xlim=(-0.34, 0.11),
        xlabel="ΔAUROC (metastatic − primary)",
        value_format="+.3f",
    )
    ax_b.set_title("Role contrast unresolved", loc="left")
    panel_label(ax_b, "B", x=-0.23)

    adapter = frames["sparse_adapter"].copy()
    cohort_styles = {
        "RIH": (PALETTE["wsi"], "o"),
        "SurGen": (PALETTE["metastatic"], "s"),
    }
    arm_offsets = {"S1": -0.09, "S2": 0.09}
    for cohort, (color, marker) in cohort_styles.items():
        for arm, offset in arm_offsets.items():
            item = adapter[(adapter["cohort"] == cohort) & (adapter["support_arm"] == arm)].sort_values(
                "labels_per_class"
            )
            x = item["labels_per_class"].to_numpy(float) + offset
            y = item["delta_auroc"].to_numpy(float)
            ax_c.errorbar(
                x,
                y,
                yerr=np.vstack([y - item["ci_low"].to_numpy(float), item["ci_high"].to_numpy(float) - y]),
                fmt=marker,
                color=color,
                markerfacecolor="white" if arm == "S2" else color,
                markeredgecolor=color,
                elinewidth=0.9,
                markersize=4.7,
                label=f"{'SR1482' if cohort == 'SurGen' else cohort}, {arm}",
            )
    ax_c.axhline(0, color=PALETTE["random"], linestyle="--", linewidth=1.0)
    ax_c.set_xticks([2, 4, 8])
    ax_c.set_ylim(-0.095, 0.105)
    ax_c.set_xlabel("Support labels per class")
    ax_c.set_ylabel("ΔAUROC vs frozen S0")
    ax_c.set_title("Sparse residual adaptation", loc="left")
    clean_axis(ax_c)
    ax_c.legend(ncol=2, loc="upper left", columnspacing=0.7, handletextpad=0.3)
    panel_label(ax_c, "C")

    primary_id = frames["aim2_results"]["primary_analysis_id"]
    agreement = frames["agreement"]
    metric_order = ["icc_1_1", "icc_1_k_eff", "pearson_pair_symmetric", "spearman_pair_symmetric"]
    metric_labels = ["ICC(1,1)", "ICC(1,k=2)", "Pearson r", "Spearman ρ"]
    agree = agreement[
        (agreement["analysis_id"] == primary_id)
        & (agreement["KRAS_stratum"] == "all")
        & agreement["metric"].isin(metric_order)
    ].set_index("metric").loc[metric_order].reset_index()
    forest_points(
        ax_d,
        agree,
        point="estimate",
        low="bootstrap_ci_low",
        high="bootstrap_ci_high",
        labels=metric_labels,
        color=PALETTE["wsi"],
        xlim=(0.46, 0.94),
        xlabel="Between-slide agreement",
    )
    ax_d.set_title("Slide agreement (n=144)", loc="left")
    ax_d.set_ylim(-1.10, 3.55)
    ax_d.text(
        0.98,
        -0.78,
        "Median |Δ native logit|  1.409\nIQR  0.552–3.061",
        ha="right",
        va="center",
        fontsize=6.6,
        color=PALETTE["muted"],
    )
    panel_label(ax_d, "D", x=-0.23)

    draws = frames["random_draws"]
    primary_draws = draws[draws["analysis_id"] == primary_id].copy()
    operational = frames["aim2_results"]["operational_slide_selection"][primary_id]
    ax_e.hist(
        primary_draws["auroc"],
        bins=36,
        color=PALETTE["sky"],
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
        density=True,
    )
    refs = [
        ("All-slide mean", operational["all_slide_mean"]["auroc"], PALETTE["ink"], "-"),
        ("Highest", operational["highest_slide_sensitivity"]["auroc"], PALETTE["metastatic"], ":"),
        ("Lowest", operational["lowest_slide_sensitivity"]["auroc"], PALETTE["supported"], "--"),
    ]
    for label, value, color, line in refs:
        ax_e.axvline(value, color=color, linestyle=line, linewidth=1.25, label=f"{label} {value:.3f}")
    ax_e.set_xlabel("AUROC from one random slide per patient")
    ax_e.set_ylabel("Slide-choice density")
    ax_e.set_title("Random-slide ranking sensitivity", loc="left")
    clean_axis(ax_e)
    ax_e.legend(loc="upper left")
    ax_e.text(
        0.98,
        0.05,
        "Top-30% membership flip\n5.11% all patients\n10.67% multi-slide patients",
        transform=ax_e.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": PALETTE["grid"]},
    )
    panel_label(ax_e, "E")

    deployment_display = draw_deployment_matrix(ax_f, frames["deployment"])
    ax_f.set_title("Deployment recommendations", loc="left", pad=2)
    panel_label(ax_f, "F", x=-0.07)

    fig.suptitle(
        "Figure 3 | Primary-tumour transport has clear deployment boundaries",
        fontsize=10.8,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    sources = [
        write_source_data(metastatic, root, "figure_3A_metastatic_transport.csv"),
        write_source_data(role, root, "figure_3B_role_contrasts.csv"),
        write_source_data(adapter, root, "figure_3C_sparse_adapter.csv"),
        write_source_data(agree, root, "figure_3D_slide_agreement.csv"),
        write_source_data(primary_draws, root, "figure_3E_random_slide_draws.csv"),
        write_source_data(deployment_display, root, "figure_3F_deployment_matrix.csv"),
    ]
    outputs = save_figure(fig, root, "Figure_3_deployment_boundaries", main=True)
    plt.close(fig)
    return outputs, sources


def compact_resolution_task(task: str) -> str:
    return {
        "Codon 12 vs other KRAS": "Codon 12 / other KRAS",
        "G12D vs other KRAS": "G12D / other KRAS",
        "G12D vs other G12": "G12D / other G12",
        "G12V vs other G12": "G12V / other G12",
        "G12C vs other G12": "G12C / other G12",
    }.get(task, task)


def draw_gate_matrix(ax: Any, gates: pd.DataFrame) -> None:
    from matplotlib.patches import Rectangle

    tasks = gates["task"].tolist()
    conditions = [
        ("fine_upper_99", "Fine upper\n< 0.60"),
        ("control_lower_99", "Control lower\n> 0.50"),
        ("delta_lower_99", "Δ lower\n> 0"),
    ]
    pass_fields = ["fine_gate_pass", "control_gate_pass", "delta_gate_pass"]
    matrix = gates[[x[0] for x in conditions]].to_numpy(float)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, len(tasks) - 0.5)
    for row in range(len(tasks)):
        for col in range(3):
            passed = bool(gates.iloc[row][pass_fields[col]])
            face = PALETTE["supported"] if passed else PALETTE["unresolved"]
            ax.add_patch(
                Rectangle(
                    (col - 0.47, len(tasks) - 1 - row - 0.43),
                    0.94,
                    0.86,
                    facecolor=face,
                    alpha=0.23,
                    edgecolor="white",
                    linewidth=1.0,
                )
            )
            ax.text(
                col,
                len(tasks) - 1 - row,
                f"{matrix[row, col]:.3f}\n{'PASS' if passed else 'FAIL'}",
                ha="center",
                va="center",
                fontsize=6.2,
                color=PALETTE["ink"],
                fontweight="semibold" if passed else "normal",
            )
    ax.set_xticks(range(3), [x[1] for x in conditions])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(np.arange(len(tasks))[::-1], [compact_resolution_task(task) for task in tasks])
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_repeated_verdicts(ax: Any, repeated: pd.DataFrame) -> None:
    from matplotlib.patches import Rectangle

    task_order = list(dict.fromkeys(repeated["task"].tolist()))
    draw_order = ["20260823", "20260824", "20260825"]
    colors = {
        "CEILING": PALETTE["supported"],
        "INCONCLUSIVE": PALETTE["unresolved"],
        "UNDERPOWERED": PALETTE["random"],
    }
    marks = {"CEILING": "C", "INCONCLUSIVE": "I", "UNDERPOWERED": "U"}
    ax.set_xlim(-0.5, 3.9)
    ax.set_ylim(-0.5, len(task_order) - 0.5)
    for row, task in enumerate(task_order):
        y = len(task_order) - 1 - row
        item = repeated[repeated["task"] == task].set_index("draw").loc[draw_order]
        for col, (_, value) in enumerate(item.iterrows()):
            verdict = value["verdict"]
            ax.add_patch(
                Rectangle(
                    (col - 0.44, y - 0.40),
                    0.88,
                    0.80,
                    facecolor=colors[verdict],
                    alpha=0.25,
                    edgecolor="white",
                )
            )
            ax.text(col, y, marks[verdict], ha="center", va="center", fontsize=7.0, fontweight="bold")
        consensus = str(item.iloc[0]["consensus"])
        consensus_label = "Ceiling" if consensus == "CONSENSUS_CEILING" else "No consensus"
        ax.text(3.15, y, consensus_label, va="center", fontsize=6.0, color=PALETTE["muted"])
    ax.set_xticks(range(3), ["Draw 1", "Draw 2", "Draw 3"])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks(
        np.arange(len(task_order))[::-1],
        [compact_resolution_task(task) for task in task_order],
    )
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.99,
        -0.16,
        "C = ceiling  •  I = inconclusive  •  U = underpowered",
        transform=ax.transAxes,
        ha="right",
        fontsize=5.9,
        color=PALETTE["muted"],
    )


def draw_actionability_ladder(ax: Any, ladder: pd.DataFrame) -> pd.DataFrame:
    from matplotlib.patches import Rectangle

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 8.6)
    ax.set_axis_off()
    x_level, x_evidence, x_decision, x_meaning = 0.01, 0.33, 0.53, 0.70
    ax.text(x_level, 8.25, "Molecular level", fontsize=6.8, fontweight="bold")
    ax.text(x_evidence, 8.25, "Current evidence", fontsize=6.8, fontweight="bold")
    ax.text(x_decision, 8.25, "Direct decision?", fontsize=6.2, fontweight="bold")
    ax.text(x_meaning, 8.25, "Bounded interpretation", fontsize=6.2, fontweight="bold")
    evidence_labels = [
        "AUROC 0.680 (ensemble ref.)",
        "Not evaluated",
        "Consensus ceiling",
        "Consensus ceiling",
        "Consensus ceiling",
        "Inconclusive",
        "No ceiling consensus",
        "Outside evidence base",
    ]
    meaning_labels = [
        "Retrospective prioritization / QA only",
        "KRAS alone cannot establish anti-EGFR eligibility",
        "Direct molecular confirmation required",
        "Direct molecular confirmation required",
        "Direct molecular confirmation required",
        "No detectability or absence claim",
        "Approved exact molecular test required",
        "Not evaluated in metastatic tissue",
    ]
    for index, row in ladder.reset_index(drop=True).iterrows():
        y = 7.45 - index
        color = "white" if index % 2 == 0 else PALETTE["light"]
        ax.add_patch(Rectangle((0, y - 0.43), 1, 0.86, facecolor=color, edgecolor=PALETTE["grid"], linewidth=0.45))
        ax.text(x_level, y, textwrap.fill(str(row["molecular_level"]), 25), va="center", fontsize=6.15)
        evidence_color = PALETTE["supported"] if "Consensus" in evidence_labels[index] else PALETTE["muted"]
        ax.text(x_evidence, y, evidence_labels[index], va="center", fontsize=6.0, color=evidence_color)
        ax.text(x_decision + 0.035, y, "NO", va="center", fontsize=6.2, color=PALETTE["metastatic"], fontweight="bold")
        ax.text(x_meaning, y, textwrap.fill(meaning_labels[index], 39), va="center", fontsize=5.95)
    display = ladder.copy()
    display.insert(len(display.columns), "figure_evidence_label", evidence_labels)
    display.insert(len(display.columns), "figure_interpretation_label", meaning_labels)
    return display


def figure_4_resolution(frames: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2, 8.25))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.18,
        right=0.96,
        bottom=0.08,
        top=0.90,
        height_ratios=[1.02, 1.10, 1.60],
        hspace=0.54,
        wspace=0.52,
    )
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, :])

    tasks = frames["aim3_tasks"].copy()
    task_order = list(dict.fromkeys(tasks["task"].tolist()))
    ybase = np.arange(len(task_order))[::-1]
    for series, offset, color, marker in (
        ("Fine task", 0.12, PALETTE["wsi"], "o"),
        ("Matched control", -0.12, PALETTE["clinical"], "s"),
    ):
        item = tasks[tasks["series"] == series].set_index("task").loc[task_order]
        x = item["auroc"].to_numpy(float)
        ax_a.errorbar(
            x,
            ybase + offset,
            xerr=np.vstack([x - item["ci_low"].to_numpy(float), item["ci_high"].to_numpy(float) - x]),
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=1.0,
            markersize=5.3,
            label=series,
        )
    ax_a.axvline(0.5, color=PALETTE["random"], linestyle="--", linewidth=1.0)
    ax_a.axvline(0.6, color=PALETTE["grid"], linestyle=":", linewidth=1.0)
    ax_a.set_yticks(ybase, [compact_resolution_task(task) for task in task_order])
    ax_a.set_xlim(0.42, 0.75)
    ax_a.set_xlabel("AUROC (two-sided 95% interval)")
    ax_a.set_title("Fine tasks versus matched controls", loc="left")
    clean_axis(ax_a, grid_axis="x")
    ax_a.legend(loc="lower right")
    panel_label(ax_a, "A", x=-0.13)

    gates = frames["aim3_gates"].copy()
    draw_gate_matrix(ax_b, gates)
    ax_b.set_title("One-sided 99% ceiling gates", loc="left", pad=2)
    panel_label(ax_b, "B", x=-0.34, y=1.13)

    repeated = frames["aim3_repeated"].copy()
    draw_repeated_verdicts(ax_c, repeated)
    ax_c.set_title("Repeated matched-control draws", loc="left", pad=2)
    panel_label(ax_c, "C", x=-0.34, y=1.13)

    ladder_display = draw_actionability_ladder(ax_d, frames["actionability"])
    ax_d.set_title("Molecular-resolution ladder: scientific detectability is not clinical actionability", loc="left", pad=2)
    panel_label(ax_d, "D", x=-0.08, y=1.05)

    fig.suptitle(
        "Figure 4 | Gene-level ranking does not establish exact-allele actionability",
        fontsize=10.8,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    sources = [
        write_source_data(tasks, root, "figure_4A_exact_resolution_tasks.csv"),
        write_source_data(gates, root, "figure_4B_one_sided_99_gates.csv"),
        write_source_data(repeated, root, "figure_4C_repeated_control_verdicts.csv"),
        write_source_data(ladder_display, root, "figure_4D_actionability_ladder.csv"),
    ]
    outputs = save_figure(fig, root, "Figure_4_resolution_actionability", main=True)
    plt.close(fig)
    return outputs, sources


def draw_stability_grid(ax: Any, stability: pd.DataFrame) -> pd.DataFrame:
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    variant_order = [f"k{k}_seed{seed}" for k in (24, 32, 40) for seed in (20260819, 20260820, 20260821)]
    row_specs = [(17, "A"), (17, "D"), (28, "A"), (28, "D")]
    values = np.empty((4, 9), dtype=float)
    passes = np.empty((4, 9), dtype=bool)
    indexed = stability.set_index(["anchor_prototype", "variant"])
    for i, (prototype, population) in enumerate(row_specs):
        for j, variant in enumerate(variant_order):
            row = indexed.loc[(prototype, variant)]
            values[i, j] = row[f"auc_{population}"]
            passes[i, j] = bool(row["positive_significant_A_and_D"])
    cmap = LinearSegmentedColormap.from_list("prototype_auc", ["#F2F4F6", "#B8DDED", PALETTE["wsi"]])
    norm = Normalize(vmin=0.50, vmax=0.625)
    ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    for i in range(4):
        for j in range(9):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5.35)
            if not passes[i, j]:
                ax.add_patch(
                    Rectangle(
                        (j - 0.44, i - 0.42),
                        0.88,
                        0.84,
                        fill=False,
                        edgecolor=PALETTE["metastatic"],
                        linewidth=1.3,
                    )
                )
    ax.set_xticks(
        range(9),
        [f"{k}·{str(seed)[-2:]}" for k in (24, 32, 40) for seed in (20260819, 20260820, 20260821)],
        rotation=90,
        va="top",
    )
    ax.tick_params(axis="x", labelsize=5.1, pad=1)
    ax.set_yticks(range(4), ["p17 · A", "p17 · D", "p28 · A", "p28 · D"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.50,
        -0.23,
        "Outline = failed joint A+D variant",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.0,
        color=PALETTE["muted"],
    )
    public_columns = [column for column in stability.columns if column != "montage_id"]
    return stability[public_columns].copy()


def draw_r2_heatmap(ax: Any, compress: pd.DataFrame) -> pd.DataFrame:
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    models = ["clinical", "abundance_only", "abundance_plus_attention", "combined"]
    targets = ["CPTAC", "RIH", "SurGen", "TCGA"]
    labels = ["Clinical", "Abundance", "+ attention", "Combined"]
    frame = compress[(compress["metric"] == "r2") & compress["model"].isin(models) & compress["summary"].isin(targets)]
    pivot = frame.pivot(index="model", columns="summary", values="point").loc[models, targets]
    cmap = LinearSegmentedColormap.from_list("r2_diverging", [PALETTE["metastatic"], "#FAFAFA", PALETTE["wsi"]])
    norm = TwoSlopeNorm(vmin=-0.10, vcenter=0.0, vmax=0.52)
    image = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap=cmap, norm=norm)
    for i in range(len(models)):
        for j in range(len(targets)):
            value = pivot.iloc[i, j]
            color = "white" if value > 0.37 else PALETTE["ink"]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=6.0, color=color)
    ax.set_xticks(range(4), targets)
    ax.xaxis.tick_top()
    ax.set_yticks(range(4), labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Held-out R²", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)
    return frame.copy()


def figure_5_partial_explanation(frames: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2, 8.35))
    grid = fig.add_gridspec(
        3,
        2,
        left=0.12,
        right=0.96,
        bottom=0.08,
        top=0.92,
        height_ratios=[1.0, 1.08, 1.05],
        hspace=0.72,
        wspace=0.58,
    )
    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes

    prototypes = frames["aim4_prototypes"].copy()
    ax_a.scatter(
        prototypes["abundance_auc_A"],
        prototypes["abundance_auc_D"],
        s=19,
        facecolor="white",
        edgecolor=PALETTE["muted"],
        linewidth=0.65,
        alpha=0.9,
    )
    for prototype, color, marker in ((17, PALETTE["supported"], "s"), (28, PALETTE["wsi"], "o")):
        row = prototypes[prototypes["prototype"] == prototype].iloc[0]
        ax_a.scatter(
            row["abundance_auc_A"],
            row["abundance_auc_D"],
            s=48,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        ax_a.annotate(
            f"p{prototype}",
            (row["abundance_auc_A"], row["abundance_auc_D"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=6.7,
            fontweight="bold",
            color=color,
        )
    limits = (0.42, 0.64)
    ax_a.plot(limits, limits, color=PALETTE["grid"], linestyle=":", linewidth=1.0)
    ax_a.axvline(0.5, color=PALETTE["grid"], linewidth=0.8)
    ax_a.axhline(0.5, color=PALETTE["grid"], linewidth=0.8)
    ax_a.set_xlim(*limits)
    ax_a.set_ylim(*limits)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xlabel("Set A abundance AUROC")
    ax_a.set_ylabel("Set D abundance AUROC")
    ax_a.set_title("All 32 numeric prototypes", loc="left")
    ax_a.text(
        0.98,
        0.04,
        "PATHOLOGY IDENTITY PENDING",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color=PALETTE["metastatic"],
        fontweight="bold",
    )
    clean_axis(ax_a)
    panel_label(ax_a, "A", x=-0.18)

    candidates = frames["aim4_candidates"].copy()
    candidate_order = [("p17", "Abundance"), ("p28", "Abundance"), ("p28", "Attention"), ("p5", "Attention")]
    ybase = np.arange(4)[::-1]
    for population, offset, color, marker in (
        ("A", 0.11, PALETTE["wsi"], "o"),
        ("D", -0.11, PALETTE["supported"], "s"),
    ):
        rows = []
        for prototype, quantity in candidate_order:
            rows.append(
                candidates[
                    (candidates["prototype"] == prototype)
                    & (candidates["quantity"] == quantity)
                    & (candidates["population"] == population)
                ].iloc[0]
            )
        item = pd.DataFrame(rows)
        x = item["auroc"].to_numpy(float)
        ax_b.errorbar(
            x,
            ybase + offset,
            xerr=np.vstack([x - item["ci_low"].to_numpy(float), item["ci_high"].to_numpy(float) - x]),
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=1.0,
            markersize=5.0,
            label=f"Set {population}",
        )
    ax_b.axvline(0.5, color=PALETTE["random"], linestyle="--", linewidth=1.0)
    ax_b.set_yticks(ybase, [f"{p} {q.lower()}" for p, q in candidate_order])
    ax_b.set_xlim(0.39, 0.69)
    ax_b.set_xlabel("KRAS association AUROC")
    ax_b.set_title("Selected associations", loc="left", pad=16)
    clean_axis(ax_b, grid_axis="x")
    ax_b.legend(loc="lower right")
    ax_b.text(
        0.0,
        1.01,
        "Attention is model-derived",
        transform=ax_b.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=PALETTE["muted"],
    )
    panel_label(ax_b, "B", x=-0.23)

    stability_public = draw_stability_grid(ax_c, frames["aim4_stability_grid"])
    ax_c.set_title("Vocabulary stability", loc="left")
    panel_label(ax_c, "C", x=-0.18, y=1.15)

    compress = frames["compress"].copy()
    model_order = ["clinical", "abundance_only", "abundance_plus_attention", "combined"]
    model_labels = ["Clinical", "Abundance", "Abundance + attention", "Combined"]
    macro = compress[
        (compress["metric"] == "r2")
        & (compress["summary"] == "equal_target_macro")
        & compress["model"].isin(model_order)
    ].set_index("model").loc[model_order].reset_index()
    forest_points(
        ax_d,
        macro,
        point="point",
        low="ci_low",
        high="ci_high",
        labels=model_labels,
        color=PALETTE["wsi"],
        reference=0.0,
        xlim=(-0.10, 0.40),
        xlabel="Held-out R² (equal-target macro)",
    )
    comparison = frames["compress_results"]["comparisons"]["attention_increment_over_abundance"]["r2"]
    ax_d.text(
        0.98,
        0.98,
        f"Attention ΔR² {comparison['point']:.3f}\n({comparison['ci95'][0]:.3f}–{comparison['ci95'][1]:.3f})",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color=PALETTE["muted"],
    )
    ax_d.set_title("Held-out score compression", loc="left")
    panel_label(ax_d, "D", x=-0.23)

    target_r2 = draw_r2_heatmap(ax_e, compress)
    ax_e.set_title("Target heterogeneity", loc="left", pad=2)
    panel_label(ax_e, "E", x=-0.18, y=1.15)

    residual_models = ["abundance_only", "abundance_plus_attention", "combined"]
    residual_labels = ["Abundance", "+ attention", "Combined"]
    residual = compress[
        (compress["summary"] == "equal_target_macro")
        & compress["model"].isin(residual_models)
        & compress["metric"].isin(["prediction_kras_auroc", "residual_kras_auroc", "native_logit_kras_auroc"])
    ].copy()
    ybase = np.arange(3)[::-1]
    for metric, offset, color, marker, label in (
        ("prediction_kras_auroc", 0.11, PALETTE["supported"], "s", "Reconstructed score"),
        ("residual_kras_auroc", -0.11, PALETTE["metastatic"], "o", "Residual score"),
    ):
        item = residual[residual["metric"] == metric].set_index("model").loc[residual_models]
        x = item["point"].to_numpy(float)
        ax_f.errorbar(
            x,
            ybase + offset,
            xerr=np.vstack([x - item["ci_low"].to_numpy(float), item["ci_high"].to_numpy(float) - x]),
            fmt=marker,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=0.9,
            markersize=4.9,
            label=label,
        )
    native = residual[residual["metric"] == "native_logit_kras_auroc"].iloc[0]
    ax_f.axvspan(native["ci_low"], native["ci_high"], color=PALETTE["wsi"], alpha=0.08, linewidth=0)
    ax_f.axvline(native["point"], color=PALETTE["wsi"], linewidth=1.2, linestyle="--", label="Native score")
    ax_f.axvline(0.5, color=PALETTE["random"], linewidth=0.9, linestyle=":")
    ax_f.set_yticks(ybase, residual_labels)
    ax_f.set_xlim(0.50, 0.78)
    ax_f.set_ylim(-0.45, 2.75)
    ax_f.set_xlabel("KRAS AUROC (equal-target macro)")
    ax_f.set_title("Residual KRAS ranking", loc="left")
    clean_axis(ax_f, grid_axis="x")
    if ax_f.legend_ is not None:
        ax_f.legend_.remove()
    ax_f.text(0.715, 2.56, "━ Native", color=PALETTE["wsi"], fontsize=5.8, va="center")
    ax_f.text(0.715, 2.34, "■ Reconstructed", color=PALETTE["supported"], fontsize=5.8, va="center")
    ax_f.text(0.715, 2.12, "● Residual", color=PALETTE["metastatic"], fontsize=5.8, va="center")
    panel_label(ax_f, "F", x=-0.23)

    fig.suptitle(
        "Figure 5 | Numeric prototype features partially explain the WSI score",
        fontsize=10.45,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    prototype_public = prototypes[
        [
            "prototype",
            "abundance_auc_A",
            "abundance_ci_low_A",
            "abundance_ci_high_A",
            "abundance_q_A",
            "abundance_significant_A",
            "abundance_auc_D",
            "abundance_ci_low_D",
            "abundance_ci_high_D",
            "abundance_q_D",
            "abundance_significant_D",
            "pathology_status",
        ]
    ].copy()
    sources = [
        write_source_data(prototype_public, root, "figure_5A_all_numeric_prototypes.csv"),
        write_source_data(candidates, root, "figure_5B_selected_associations.csv"),
        write_source_data(stability_public, root, "figure_5C_vocabulary_stability.csv"),
        write_source_data(macro, root, "figure_5D_equal_target_compressibility.csv"),
        write_source_data(target_r2, root, "figure_5E_target_specific_r2.csv"),
        write_source_data(residual, root, "figure_5F_reconstructed_residual_auroc.csv"),
    ]
    outputs = save_figure(fig, root, "Figure_5_partial_explanation", main=True)
    plt.close(fig)
    return outputs, sources


def supplementary_1_aim1(frames: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig = plt.figure(figsize=(7.2, 6.35), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.14, 1.0], hspace=0.32, wspace=0.32)
    ax_a = fig.add_subplot(grid[:, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 1])

    challenge = frames["full_challenge"].copy()
    labels = [f"{r.set}: {r.population}\nn={r.n:,}" for r in challenge.itertuples()]
    forest_points(
        ax_a,
        challenge,
        point="auroc",
        low="ci_low",
        high="ci_high",
        labels=labels,
        color=PALETTE["wsi"],
        reference=0.5,
        xlim=(0.42, 0.90),
        xlabel="AUROC (median-seed E0)",
    )
    ax_a.set_title("Complete prespecified challenge-set forest", loc="left")
    ax_a.text(
        0.99,
        0.01,
        "H (stage IV) and small anatomic subsets are exploratory",
        transform=ax_a.transAxes,
        ha="right",
        fontsize=6.0,
        color=PALETTE["muted"],
    )
    panel_label(ax_a, "A", x=-0.26)

    robustness = frames["robustness"].copy()
    ybase = np.arange(len(robustness))[::-1]
    for field, offset, color, marker, label in (
        ("auroc_A", 0.12, PALETTE["wsi"], "o", "Set A"),
        ("auroc_D", -0.12, PALETTE["supported"], "s", "Set D"),
    ):
        ax_b.plot(
            robustness[field],
            ybase + offset,
            marker=marker,
            linestyle="none",
            color=color,
            markeredgecolor="white",
            markersize=5.3,
            label=label,
        )
    ax_b.axvline(0.5, color=PALETTE["random"], linestyle="--", linewidth=0.9)
    ax_b.set_yticks(ybase, [textwrap.fill(x, 24) for x in robustness["arm"]])
    ax_b.set_xlim(0.61, 0.75)
    ax_b.set_xlabel("AUROC")
    ax_b.set_title("Encoder sensitivity", loc="left")
    clean_axis(ax_b, grid_axis="x")
    ax_b.legend(loc="lower right")
    panel_label(ax_b, "B", x=-0.30)

    worklist = frames["worklist"]
    sensitivity_pops = ["A_all_primary", "D_mss_braf_wt", "stage_iv_exploratory"]
    sensitivity_labels = ["Set A", "Set D", "Stage IV (exploratory)"]
    colors = [PALETTE["wsi"], PALETTE["supported"], PALETTE["unresolved"]]
    sensitivity_rows = worklist[
        (worklist["analysis"] == "E0_frozen_OOF")
        & worklist["population"].isin(sensitivity_pops)
        & (worklist["method"] == "wsi_declared_seed_median")
    ].copy()
    for population, label, color in zip(sensitivity_pops, sensitivity_labels, colors, strict=True):
        item = sensitivity_rows[sensitivity_rows["population"] == population].sort_values("capacity_nominal")
        ax_c.plot(
            item["capacity_nominal"],
            item["capture"],
            color=color,
            marker="o",
            markeredgecolor="white",
            markeredgewidth=0.45,
            linewidth=1.4,
            label=label,
        )
        ax_c.fill_between(
            item["capacity_nominal"].to_numpy(float),
            item["capture_ci_low"].to_numpy(float),
            item["capture_ci_high"].to_numpy(float),
            color=color,
            alpha=0.09,
            linewidth=0,
        )
    ax_c.plot([0.1, 0.5], [0.1, 0.5], color=PALETTE["random"], linestyle="--", linewidth=1.0, label="Random")
    ax_c.axvline(0.3, color=PALETTE["grid"], linewidth=0.9)
    ax_c.set_xlim(0.08, 0.52)
    ax_c.set_ylim(0.05, 0.75)
    ax_c.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_c.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_c.set_xlabel("Worklist capacity")
    ax_c.set_ylabel("Mutants captured")
    ax_c.set_title("Worklist sensitivity", loc="left")
    clean_axis(ax_c)
    ax_c.legend(loc="upper left")
    panel_label(ax_c, "C", x=-0.30)

    fig.suptitle(
        "Figure S1 | Aim 1 sensitivity analyses",
        fontsize=10.5,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    sources = [
        write_source_data(challenge, root, "supp_figure_1A_full_challenge_sets.csv"),
        write_source_data(robustness, root, "supp_figure_1B_encoder_cap_sensitivity.csv"),
        write_source_data(sensitivity_rows, root, "supp_figure_1C_worklist_sensitivities.csv"),
    ]
    outputs = save_figure(fig, root, "Supplementary_Figure_1_aim1_sensitivities", main=False)
    plt.close(fig)
    return outputs, sources


def supplementary_2_slides(frames: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig = plt.figure(figsize=(7.2, 6.35), layout="constrained")
    grid = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    primary_id = frames["aim2_results"]["primary_analysis_id"]
    draws = frames["random_draws"]
    primary_draws = draws[draws["analysis_id"] == primary_id].copy()
    operational = frames["aim2_results"]["operational_slide_selection"][primary_id]

    ax_a.hist(primary_draws["auroc"], bins=38, color=PALETTE["sky"], edgecolor="white", linewidth=0.3)
    all_value = operational["all_slide_mean"]["auroc"]
    nested = operational["random_one_slide"]["nested_patient_and_slide_bootstrap_95_ci"]
    choice = operational["random_one_slide"]["slide_choice_95_interval"]
    ax_a.axvline(all_value, color=PALETTE["ink"], linewidth=1.25, label=f"All-slide mean {all_value:.3f}")
    ax_a.axvspan(choice[0], choice[1], color=PALETTE["wsi"], alpha=0.14, label="Slide-choice 95% interval")
    ax_a.set_xlabel("Random-one-slide AUROC")
    ax_a.set_ylabel("Draw count")
    ax_a.set_title("10,000 slide-choice draws", loc="left")
    clean_axis(ax_a)
    ax_a.legend(loc="upper left")
    ax_a.text(
        0.98,
        0.05,
        f"Nested patient + slide interval\n{nested[0]:.3f} to {nested[1]:.3f}",
        transform=ax_a.transAxes,
        ha="right",
        fontsize=6.2,
        color=PALETTE["muted"],
    )
    panel_label(ax_a, "A")

    flip_long = primary_draws[
        ["draw", "priority_flip_fraction_all", "priority_flip_fraction_multislide"]
    ].melt(id_vars="draw", var_name="population", value_name="flip_fraction")
    for field, label, color in (
        ("priority_flip_fraction_all", "All patients", PALETTE["wsi"]),
        ("priority_flip_fraction_multislide", "Multi-slide patients", PALETTE["metastatic"]),
    ):
        ax_b.hist(
            primary_draws[field],
            bins=np.linspace(0, 0.17, 27),
            histtype="step",
            linewidth=1.4,
            color=color,
            density=True,
            label=label,
        )
    ax_b.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_b.set_xlabel("Top-30% queue-membership flip")
    ax_b.set_ylabel("Slide-choice density")
    ax_b.set_title("Queue membership variation", loc="left")
    clean_axis(ax_b)
    ax_b.legend(loc="upper right")
    panel_label(ax_b, "B")

    technical_patients = frames["technical_patients"].copy()
    ax_c.scatter(
        technical_patients["patch_count_log2_range"],
        technical_patients["score_range"],
        s=12,
        facecolor=PALETTE["sky"],
        edgecolor="none",
        alpha=0.55,
        rasterized=True,
    )
    ax_c.set_xlabel("Within-patient log₂ patch-count range")
    ax_c.set_ylabel("Within-patient native-logit range")
    ax_c.set_title("No patch-count association established", loc="left")
    clean_axis(ax_c)
    technical = frames["technical"]
    row = technical[technical["predictor"] == "patch_count_log2_range"].iloc[0]
    ax_c.text(
        0.98,
        0.96,
        f"Spearman ρ={row['spearman_rho']:.3f}\nBH q={row['bh_q_across_two_technical_screens']:.3f}",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=6.4,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["grid"]},
    )
    panel_label(ax_c, "C")

    agreement = frames["agreement"]
    strata = ["all", "KRAS_mutant", "KRAS_wild_type"]
    labels = ["All", "KRAS-mutant", "KRAS wild type"]
    metrics = ["icc_1_1", "pearson_pair_symmetric", "spearman_pair_symmetric"]
    colors = [PALETTE["wsi"], PALETTE["supported"], PALETTE["fusion"]]
    offsets = [-0.18, 0.0, 0.18]
    ybase = np.arange(3)[::-1]
    stratum_rows = agreement[
        (agreement["analysis_id"] == primary_id)
        & agreement["KRAS_stratum"].isin(strata)
        & agreement["metric"].isin(metrics)
    ].copy()
    for metric, color, offset in zip(metrics, colors, offsets, strict=True):
        item = stratum_rows[stratum_rows["metric"] == metric].set_index("KRAS_stratum").loc[strata]
        x = item["estimate"].to_numpy(float)
        ax_d.errorbar(
            x,
            ybase + offset,
            xerr=np.vstack(
                [x - item["bootstrap_ci_low"].to_numpy(float), item["bootstrap_ci_high"].to_numpy(float) - x]
            ),
            fmt={"icc_1_1": "o", "pearson_pair_symmetric": "s", "spearman_pair_symmetric": "^"}[metric],
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            elinewidth=0.9,
            markersize=4.6,
            label={"icc_1_1": "ICC(1,1)", "pearson_pair_symmetric": "Pearson", "spearman_pair_symmetric": "Spearman"}[metric],
        )
    ax_d.set_yticks(ybase, labels)
    ax_d.set_xlim(0.38, 0.94)
    ax_d.set_xlabel("Agreement estimate")
    ax_d.set_title("Agreement by KRAS stratum", loc="left")
    clean_axis(ax_d, grid_axis="x")
    ax_d.legend(loc="lower right")
    panel_label(ax_d, "D")

    fig.suptitle(
        "Figure S2 | Held-out slide-sampling diagnostics",
        fontsize=10.5,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    sources = [
        write_source_data(primary_draws, root, "supp_figure_2A_random_slide_draws.csv"),
        write_source_data(flip_long, root, "supp_figure_2B_queue_flip_draws.csv"),
        write_source_data(technical_patients, root, "supp_figure_2C_technical_patient_metrics.csv"),
        write_source_data(stratum_rows, root, "supp_figure_2D_stratified_agreement.csv"),
    ]
    outputs = save_figure(fig, root, "Supplementary_Figure_2_slide_diagnostics", main=False)
    plt.close(fig)
    return outputs, sources


def supplementary_3_score_scatter(frames: dict[str, Any], root: Path) -> tuple[list[Path], list[Path]]:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.25), layout="constrained", sharex=True, sharey=True)
    predictions = frames["predictions"].copy()
    compress = frames["compress"]
    targets = ["CPTAC", "RIH", "SurGen", "TCGA"]
    for label, ax in zip(targets, axes.flat, strict=True):
        item = predictions[predictions["cohort"] == label]
        ax.scatter(
            item["native_ensemble_logit"],
            item["prediction_combined"],
            s=8,
            facecolor=PALETTE["wsi"],
            edgecolor="none",
            alpha=0.34,
            rasterized=True,
        )
        metrics = compress[
            (compress["model"] == "combined")
            & (compress["summary"] == label)
            & compress["metric"].isin(["r2", "pearson"])
        ].set_index("metric")
        r2 = metrics.loc["r2", "point"]
        pearson = metrics.loc["pearson", "point"]
        ax.text(
            0.03,
            0.96,
            f"n={len(item)}\nR²={r2:.3f}\nr={pearson:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=6.6,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["grid"]},
        )
        ax.set_title(label, loc="left")
        clean_axis(ax)
        panel_label(ax, chr(ord("A") + targets.index(label)), x=-0.13)
    for ax in axes[-1, :]:
        ax.set_xlabel("Frozen native WSI logit")
    for ax in axes[:, 0]:
        ax.set_ylabel("LOCO combined-feature reconstruction")
    fig.suptitle(
        "Figure S3 | Cohort-held-out score reconstruction",
        fontsize=10.5,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    public = predictions[
        ["patient_id", "cohort", "label", "native_ensemble_logit", "prediction_combined", "residual_combined"]
    ].copy()
    sources = [write_source_data(public, root, "supp_figure_3_patient_score_reconstruction.csv")]
    outputs = save_figure(fig, root, "Supplementary_Figure_3_score_reconstruction", main=False)
    plt.close(fig)
    return outputs, sources


FIGURE_LEGENDS = """# Figure legends

All estimates are conditional on the frozen fitted models. AUROC describes ranking, not a
validated diagnostic threshold. Error bars and bands are identified below; model seeds are
algorithmic replicates and are never treated as independent samples.

## Figure 1 | A reproducible gene-level WSI signal persists after measured context checks

**A,** Study structure and the bounded retrospective worklist question. **B,** Frozen E0
median-seed AUROC in all primary tumours and molecular-context restrictions. Points and
endpoints are medians of three seed-specific values; each seed uses a 2,000-draw patient
bootstrap. These brackets are descriptive across-seed endpoint summaries, not inference with
three seeds. **C,** Paired AUROC differences among the cross-fitted limited clinical model,
WSI score and late fusion on the same 1,486 patients; whiskers are paired patient-bootstrap
95% intervals. **D,** Adjusted native-logit effects among 826 KRAS-wild-type patients from an
OLS model including BRAF, MSI, their interaction, cohort and site; whiskers are
patient-bootstrap 95% intervals. BRAF and MSI are regression main effects under the stated
interaction coding, not causal effects. The signal persists after restriction to measured
MSI/BRAF context, but molecular confounding is not claimed to be eliminated.

## Figure 2 | WSI scores enrich retrospective molecular-testing worklists

Patients are ordered by frozen scores at five evaluated capacities (10%, 20%, 30%, 40% and
50%); straight segments connect evaluated points and are not a smoothed clinical curve.
Worklist size is `floor(q × n)` with fractional allocation when a score tie crosses the
boundary. **A,** Development E0 OOF capture using the declared median-over-seed estimand.
Shading is the 10,000-draw ordinary patient-bootstrap 95% interval; random ordering is its
analytic expectation. **B,** Equal-target mean across four held-out primary targets. Endpoint
brackets for this descriptive macro are arithmetic means of target endpoints, not a pooled
patient bootstrap. External clinical and fusion rankings are post-hoc but source-only.
**C,** Target-specific capture at the headline 30% capacity; whiskers are target-level
10,000-draw patient-bootstrap 95% intervals. **D,** Held-out primary AUROC overall and in Set
D, with 10,000-draw patient-bootstrap 95% intervals. Equal-target summaries give each target
equal weight. Enrichment supports retrospective ordering research; it does not estimate tests
avoided, time, cost, net benefit or a validated score cutoff.

## Figure 3 | Primary-tumour transport does not generalize to every deployment setting

**A,** Metastatic AUROC and equal-cohort macro with 10,000-draw patient-bootstrap 95%
intervals. The fixed metastatic transport gate was not met; neither cohort is claimed to rank
below chance. **B,** Disjoint-arm metastatic-minus-primary AUROC contrasts; all eight RIH
dual-role patients are excluded from both arms. The equal-cohort interval crosses zero, so an
overall decrement is unresolved. **C,** Difference from frozen S0 for the tested 513-parameter
residual linear adapter with 2, 4 or 8 support labels per class. Whiskers are two-way
patient-by-support-procedure 10,000-draw 95% intervals. Lack of established improvement is
specific to this adapter and budget. **D,** Between-slide agreement in 144 held-out SR1482
multi-slide patients; whiskers are patient-clustered bootstrap 95% intervals. **E,** AUROC over
10,000 one-random-slide draws. The distribution represents slide-choice variation; the
separate patient-plus-slide nested interval is reported in Supplementary Figure 2. Highest and
lowest slide scores are sensitivity bounds, not deployment policies. Queue flips use the
top-30% floor rule. **F,** Deployment recommendation strip derived from the complete no-fit
deployment matrix; all matrix fields are retained in the accompanying source-data table. This is
between-slide tissue sampling, not repeated-stain or repeated-scan technical reproducibility.

## Figure 4 | Gene-level ranking does not support exact-allele treatment decisions

**A,** Fine molecular-resolution tasks and prevalence-matched wild-type controls. Whiskers are
two-sided 95% intervals from 10,000 partially paired, cohort-stratified patient draws; positive
patients are shared and distinct negative pools are sampled independently. **B,** The three
Bonferroni-controlled one-sided 99% gates used for the fixed ceiling verdict: fine upper bound
below 0.60, control lower bound above 0.50 and lower bound of control-minus-fine above zero.
The verdict is determined by these gates, not panel A's 95% intervals. **C,** Verdicts from
three prespecified matched-control draws using a 20,000-draw stratified partially paired
bootstrap. Codon and both G12D tasks reach consensus ceiling; G12V is inconclusive; G12C has
one inconclusive and two underpowered draws and no ceiling consensus. **D,** Actionability
ladder. The gene-level value is the 0.680 three-seed ensemble reference; every direct clinical
decision field is No. A ceiling is architecture- and population-specific and does not imply
biological absence of morphology.

## Figure 5 | Numeric prototype features explain a partial, target-dependent component of the WSI score

Pathology identity remains pending; only numeric IDs are used. **A,** Set-A versus Set-D
abundance AUROC for all 32 canonical prototypes, with p17 and p28 highlighted. **B,** Selected
abundance and model-derived attention associations. Whiskers are 2,000-draw patient-bootstrap
95% intervals; two-sided Mann–Whitney p-values are BH-adjusted within declared families.
**C,** Label-blind matched-set abundance AUROC across nine vocabulary/initialization variants;
outlined cells identify the single p28 variant that fails the joint A+D gate. p17 passes 9/9
and p28 passes 8/9. Alternative clusters do not inherit a histologic identity. **D,**
Equal-target held-out R-squared for four nested ridge reconstructions of the frozen WSI logit.
**E,** Target-specific held-out R-squared, retaining negative CPTAC values. **F,** Descriptive
KRAS AUROC of reconstructed and residual logits; the native-logit reference is shown once.
Panels D–F use 2,000 within-target, no-refit bootstrap intervals conditional on frozen profiles,
outer fits and selected penalties. Attention is model-derived, so this is partial score
compressibility rather than evidence that the model is human-readable or independently
validated.

## Supplementary Figure 1 | Aim 1 population, encoder, and worklist sensitivities

**A,** Full prespecified E0 challenge-set forest. Brackets have the same median-of-seed
endpoint interpretation as Figure 1B; stage IV and small anatomic subsets are exploratory.
**B,** Set-A and Set-D point estimates for the cap sensitivity and broader encoder reference.
Virchow2 is not a matched one-factor ablation. **C,** E0 WSI worklist capture in Set A, Set D
and the exploratory stage-IV primary subset; bands are 10,000-draw ordinary patient-bootstrap
95% intervals and random capture is the analytic expectation.

## Supplementary Figure 2 | Between-slide sampling diagnostics in held-out SR1482

**A,** Ten thousand random-one-slide AUROCs, with the all-slide mean and slide-choice 95%
interval. The annotated nested interval combines patient and slide-choice uncertainty.
**B,** Distribution of top-30% queue-membership flips over slide-choice draws. **C,**
Within-patient score range versus log2 patch-count range; the exploratory Spearman screen uses
10,000 permutations and BH adjustment across two tissue-amount screens. Patch count and
estimated tissue area are numerically equivalent in these sealed data, so only one is shown.
**D,** ICC, Pearson and Spearman agreement by KRAS stratum with patient-clustered bootstrap
95% intervals. These analyses support averaging slides already available and caution near a
queue cutoff; they do not establish universal slide interchangeability or mandate acquisition
of additional slides.

## Supplementary Figure 3 | Cohort-held-out score reconstruction is target-dependent

Frozen native WSI logits versus combined-feature leave-one-cohort-out reconstructions in each
target. Points are patients and are rasterized in vector exports; panel annotations report
held-out R-squared and Pearson correlation. The response is the WSI logit, not KRAS status.
Candidate selection used the full development data, E0 is patient-OOF rather than
cohort-held-out, and attention features are model-derived. The panel is a post-selection
explanatory analysis, not an independently validated predictive model.
"""


PACKAGE_README = """# final_v3 paper-ready figures (v3, authoritative)

This append-only package visualizes the sealed `final_v3` results without modifying the
experimental setup, results, audit, parent receipt, or any analysis artifact.

This v3 package supersedes the preserved `reports/final_v3/figures_v2` render. The scientific
content is unchanged; v3 resolves two final-size heading collisions found during independent
visual QA. The v2 receipt preserves the preceding correction history.

## Contents

- `main/`: five double-column composite figures in PDF, SVG, 600-dpi PNG and 600-dpi
  LZW-compressed TIFF.
- `supplementary/`: three focused supplementary composites in the same formats.
- `source_data/`: one machine-readable CSV per panel or coherent panel group.
- `Figure_Legends.md`: manuscript-ready captions and claim boundaries.
- `source_data_manifest.json`: panel-to-source mapping and export specification.
- `figure_package_receipt.json`: hashes the parent report receipt, all analysis inputs used,
  generator and tests, and every package artifact except itself.

Vector files retain editable text using embedded TrueType-compatible fonts. Dense patient
scatters are rasterized inside PDF/SVG while labels and axes remain vector. The semantic
palette is color-vision-accessible and uses marker/line/text redundancy.

## Rebuild

From the repository root:

```bash
uv run python tools/final_v3_figures.py
```

The generator fails if `reports/final_v3/figures_v3` already exists. A changed figure package
must be published to a new versioned directory; the sealed package is never overwritten.
"""


def write_package_text(root: Path) -> list[Path]:
    legends = root / "Figure_Legends.md"
    readme = root / "README.md"
    legends.write_text(FIGURE_LEGENDS, encoding="utf-8")
    readme.write_text(PACKAGE_README, encoding="utf-8")
    return [legends, readme]


def source_manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "package_version": "final_v3_figures_v3",
        "supersedes": str((SUPERSEDED_OUTPUT / "figure_package_receipt.json").resolve()),
        "parent_report_bundle": str(PARENT_RECEIPT.resolve()),
        "main_figures": {
            "Figure_1_foundation": [
                "figure_1A_study_counts.csv",
                "figure_1B_challenge_sets.csv",
                "figure_1C_model_contrasts.csv",
                "figure_1D_adjusted_context.csv",
            ],
            "Figure_2_worklist_enrichment": [
                "figure_2A_development_worklist.csv",
                "figure_2B_equal_target_worklist.csv",
                "figure_2C_heldout_top30.csv",
                "figure_2D_external_primary.csv",
            ],
            "Figure_3_deployment_boundaries": [
                "figure_3A_metastatic_transport.csv",
                "figure_3B_role_contrasts.csv",
                "figure_3C_sparse_adapter.csv",
                "figure_3D_slide_agreement.csv",
                "figure_3E_random_slide_draws.csv",
                "figure_3F_deployment_matrix.csv",
            ],
            "Figure_4_resolution_actionability": [
                "figure_4A_exact_resolution_tasks.csv",
                "figure_4B_one_sided_99_gates.csv",
                "figure_4C_repeated_control_verdicts.csv",
                "figure_4D_actionability_ladder.csv",
            ],
            "Figure_5_partial_explanation": [
                "figure_5A_all_numeric_prototypes.csv",
                "figure_5B_selected_associations.csv",
                "figure_5C_vocabulary_stability.csv",
                "figure_5D_equal_target_compressibility.csv",
                "figure_5E_target_specific_r2.csv",
                "figure_5F_reconstructed_residual_auroc.csv",
            ],
        },
        "supplementary_figures": {
            "Supplementary_Figure_1_aim1_sensitivities": [
                "supp_figure_1A_full_challenge_sets.csv",
                "supp_figure_1B_encoder_cap_sensitivity.csv",
                "supp_figure_1C_worklist_sensitivities.csv",
            ],
            "Supplementary_Figure_2_slide_diagnostics": [
                "supp_figure_2A_random_slide_draws.csv",
                "supp_figure_2B_queue_flip_draws.csv",
                "supp_figure_2C_technical_patient_metrics.csv",
                "supp_figure_2D_stratified_agreement.csv",
            ],
            "Supplementary_Figure_3_score_reconstruction": [
                "supp_figure_3_patient_score_reconstruction.csv"
            ],
        },
        "export": {
            "width_in": 7.2,
            "vector_formats": ["pdf", "svg"],
            "raster_formats": ["png", "tiff"],
            "raster_dpi": RASTER_DPI,
            "raster_color_mode": "opaque RGB on white",
            "tiff_compression": "LZW",
            "font_family": "DejaVu Sans",
            "pdf_fonttype": 42,
            "svg_text": "preserved",
        },
        "interpretation_boundary": (
            "Retrospective ranking and explanatory evidence only; no validated diagnostic threshold, "
            "test-avoidance claim, treatment decision, or human-readable pathology identity."
        ),
    }


def write_source_manifest(root: Path) -> Path:
    path = root / "source_data_manifest.json"
    path.write_text(json.dumps(source_manifest_payload(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def relative_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "relative_path": str(resolved.relative_to(root.resolve())),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def figure_input_paths(data: dict[str, Any]) -> list[Path]:
    paths = [
        PARENT_RECEIPT,
        FINAL_V3 / "Experimental_Setup.md",
        FINAL_V3 / "Results.md",
        FINAL_V3 / "Audit.md",
        AIM1_ROOT / "receipt.json",
        AIM1_ROOT / "validation.json",
        AIM1_ROOT / "worklist_results.csv",
        AIM1_ROOT / "comparisons.csv",
        AIM2_ROOT / "receipt.json",
        AIM2_ROOT / "agreement_metrics.csv",
        AIM2_ROOT / "multi_slide_census.csv",
        AIM2_ROOT / "random_slide_draws.parquet",
        AIM2_ROOT / "technical_associations.csv",
        AIM2_ROOT / "technical_patient_metrics.parquet",
        AIM2_ROOT / "deployment_matrix.csv",
        AIM2_ROOT / "results.json",
        AIM3_ROOT / "receipt.json",
        AIM3_ROOT / "actionability_ladder.csv",
        AIM4_ROOT / "completion_receipt.json",
        AIM4_ROOT / "per_target_metrics.csv",
        AIM4_ROOT / "patient_predictions.parquet",
        AIM4_ROOT / "model_selection.csv",
        AIM4_ROOT / "results.json",
        CLAIM_MAP,
    ]
    paths.extend(data["legacy_paths"].values())
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path.resolve())] = path.resolve()
    return [unique[key] for key in sorted(unique)]


def validate_export_files(root: Path, *, raster_dpi: int) -> dict[str, Any]:
    expected_main = {
        "Figure_1_foundation",
        "Figure_2_worklist_enrichment",
        "Figure_3_deployment_boundaries",
        "Figure_4_resolution_actionability",
        "Figure_5_partial_explanation",
    }
    expected_supp = {
        "Supplementary_Figure_1_aim1_sensitivities",
        "Supplementary_Figure_2_slide_diagnostics",
        "Supplementary_Figure_3_score_reconstruction",
    }
    for folder, stems in ((root / "main", expected_main), (root / "supplementary", expected_supp)):
        for stem in stems:
            for suffix in ("pdf", "svg", "png", "tiff"):
                path = folder / f"{stem}.{suffix}"
                if not path.is_file() or path.stat().st_size == 0:
                    raise FigureBuildError(f"missing figure export: {path}")
            if not (folder / f"{stem}.pdf").read_bytes().startswith(b"%PDF-"):
                raise FigureBuildError(f"invalid PDF export: {stem}")
            svg_text = (folder / f"{stem}.svg").read_text(encoding="utf-8")
            if "<svg" not in svg_text or "DejaVu Sans" not in svg_text:
                raise FigureBuildError(f"invalid or non-text SVG export: {stem}")
            for suffix in ("png", "tiff"):
                with Image.open(folder / f"{stem}.{suffix}") as image:
                    if image.mode != "RGB":
                        raise FigureBuildError(f"raster export is not opaque RGB: {stem}.{suffix}")
                    if raster_dpi >= 600 and image.width < 3900:
                        raise FigureBuildError(f"raster width is below paper-ready target: {stem}.{suffix}")
                    dpi = image.info.get("dpi", (0, 0))
                    if raster_dpi >= 600 and min(dpi) < 590:
                        raise FigureBuildError(f"raster DPI metadata failure: {stem}.{suffix} {dpi}")

    manifest = load_json(root / "source_data_manifest.json")
    declared_csv = [
        item
        for group in ("main_figures", "supplementary_figures")
        for files in manifest[group].values()
        for item in files
    ]
    if len(declared_csv) != len(set(declared_csv)) or len(declared_csv) != 32:
        raise FigureBuildError("source-data manifest must declare 32 unique panel tables")
    for filename in declared_csv:
        path = root / "source_data" / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FigureBuildError(f"missing source-data table: {filename}")

    forbidden = re.compile(r"\bM(?:04|05|07|11)\b", flags=re.IGNORECASE)
    public_suffixes = {".csv", ".json", ".md", ".svg", ".txt"}
    violations = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in public_suffixes:
            text = path.read_text(encoding="utf-8", errors="replace")
            if forbidden.search(text):
                violations.append(str(path.relative_to(root)))
    if violations:
        raise FigureBuildError(f"historical montage labels leaked into public outputs: {violations}")
    return {
        "main_composites": len(expected_main),
        "supplementary_composites": len(expected_supp),
        "figure_exports": (len(expected_main) + len(expected_supp)) * 4,
        "source_data_tables": len(declared_csv),
        "raster_dpi": raster_dpi,
        "status": "PASS",
    }


def build_figure_receipt(root: Path, data: dict[str, Any], checks: dict[str, Any], output: Path) -> Path:
    test_path = REPO / "tests" / "test_final_v3_figures.py"
    implementation = {
        "generator": file_identity(Path(__file__).resolve()),
        "focused_tests": file_identity(test_path),
    }
    artifacts = [
        relative_identity(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "figure_package_receipt.json"
    ]
    superseded_receipt = SUPERSEDED_OUTPUT / "figure_package_receipt.json"
    payload = {
        "schema_version": 3,
        "status": "PASS",
        "append_only": True,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "published_root": str(output.resolve()),
        "parent_report_bundle": file_identity(PARENT_RECEIPT),
        "supersedes_figure_package": (
            file_identity(superseded_receipt) if superseded_receipt.is_file() else None
        ),
        "authoritative_inputs": [file_identity(path) for path in figure_input_paths(data)],
        "implementation": implementation,
        "artifacts": artifacts,
        "checks": checks,
        "immutability_note": (
            "This receipt does not hash itself. Any later byte change to a bound input, implementation file, "
            "or package artifact invalidates the figure package and requires a new append-only directory."
        ),
        "claim_boundary": source_manifest_payload()["interpretation_boundary"],
    }
    path = root / "figure_package_receipt.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_figure_receipt(receipt_path: Path) -> dict[str, Any]:
    payload = load_json(receipt_path)
    if payload.get("status") != "PASS" or payload.get("append_only") is not True:
        raise FigureBuildError("figure receipt status failure")
    root = receipt_path.resolve().parent
    parent = payload["parent_report_bundle"]
    observed_parent = file_identity(PARENT_RECEIPT)
    if parent["sha256"] != observed_parent["sha256"] or parent["size_bytes"] != observed_parent["size_bytes"]:
        raise FigureBuildError("figure receipt parent-report mismatch")
    superseded = payload.get("supersedes_figure_package")
    if superseded is not None:
        verify_identity(superseded)
    for record in payload["authoritative_inputs"]:
        verify_identity(record)
    for record in payload["implementation"].values():
        verify_identity(record)
    for record in payload["artifacts"]:
        path = root / record["relative_path"]
        observed = relative_identity(path, root)
        if observed["sha256"] != record["sha256"] or observed["size_bytes"] != record["size_bytes"]:
            raise FigureBuildError(f"figure artifact mismatch: {record['relative_path']}")
    checks = validate_export_files(root, raster_dpi=int(payload["checks"]["raster_dpi"]))
    if checks["status"] != "PASS":
        raise FigureBuildError("figure export validation failure")
    result = dict(payload)
    result["verification"] = {
        "status": "PASS",
        "artifacts_rehashed": len(payload["artifacts"]),
        "inputs_rehashed": len(payload["authoritative_inputs"]),
    }
    return result


def build_package(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output in {REPO.resolve(), FINAL_V3.resolve()}:
        raise FigureBuildError(f"refusing broad output target: {output}")
    if output.exists():
        raise FigureBuildError(f"append-only output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    published = False
    try:
        setup_matplotlib()
        data = load_and_validate_data()
        frames = source_frames(data)
        figure_builders = [
            figure_1_foundation,
            figure_2_worklist,
            figure_3_deployment,
            figure_4_resolution,
            figure_5_partial_explanation,
            supplementary_1_aim1,
            supplementary_2_slides,
            supplementary_3_score_scatter,
        ]
        figure_outputs: list[Path] = []
        source_outputs: list[Path] = []
        for builder in figure_builders:
            exports, sources = builder(frames, staging)
            figure_outputs.extend(exports)
            source_outputs.extend(sources)
        text_outputs = write_package_text(staging)
        manifest = write_source_manifest(staging)
        checks = validate_export_files(staging, raster_dpi=RASTER_DPI)
        checks.update(
            {
                "expected_figure_exports_written": len(figure_outputs),
                "expected_source_tables_written": len(source_outputs),
                "package_text_files_written": len(text_outputs),
                "source_manifest_written": manifest.is_file(),
                "parent_report_receipt_reverified": verify_bound_sources()["status"],
            }
        )
        build_figure_receipt(staging, data, checks, output)
        os.replace(staging, output)
        published = True
        return verify_figure_receipt(output / "figure_package_receipt.json")
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Append-only output directory (default: reports/final_v3/figures_v3).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Rehash authoritative inputs and validate all plotting frames without writing figures.",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Verify an existing figure_package_receipt.json and exit.",
    )
    parser.add_argument(
        "--raster-dpi",
        type=int,
        default=600,
        help="Raster export DPI. Values below 600 are allowed only outside the canonical output for previews.",
    )
    return parser.parse_args()


def main() -> None:
    global RASTER_DPI

    args = parse_args()
    if args.verify is not None:
        result = verify_figure_receipt(args.verify)
        print(json.dumps(result["verification"], indent=2, sort_keys=True))
        return
    if args.validate_only:
        data = load_and_validate_data()
        frames = source_frames(data)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "authoritative_inputs": len(figure_input_paths(data)),
                    "plotting_frames": len(frames),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    RASTER_DPI = int(args.raster_dpi)
    if RASTER_DPI < 100:
        raise FigureBuildError("raster DPI must be at least 100")
    if args.output.resolve() == DEFAULT_OUTPUT.resolve() and RASTER_DPI != 600:
        raise FigureBuildError("canonical paper package requires 600-dpi raster exports")
    result = build_package(args.output)
    print(json.dumps(result["verification"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
