#!/usr/bin/env python3
"""Build a publication-candidate figure from the verified FINAL-v6 diagnostic.

This is a read-only derivation.  It authenticates the completed metastatic-
transport diagnostic bundle, reads only its governed tabular outputs, and
publishes a separate PNG, PDF, and caption/data-source note.  It never writes
inside the verified source bundle and refuses to overwrite a derived output.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

try:  # noqa: E402
    from tools.crc_final_v6_met_transfer_diagnostics import (  # type: ignore[no-redef]
        verify_published as verify_diagnostic_bundle,
    )
except ModuleNotFoundError:  # direct ``python tools/...`` execution
    from crc_final_v6_met_transfer_diagnostics import (  # type: ignore[no-redef]
        verify_published as verify_diagnostic_bundle,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "reports/reruns/crc_final_v6_met_transfer_diagnostics_20260827"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "reports/reruns/crc_final_v6_met_transfer_diagnostics_20260827_paper_figure"
)

FIGURE_PNG = "crc_final_v6_met_transfer_paper_figure.png"
FIGURE_PDF = "crc_final_v6_met_transfer_paper_figure.pdf"
SOURCE_NOTE = "caption_and_data_sources.md"
FINAL_STATUS = "COMPLETED_POST_OUTCOME_EXPLORATORY_DIAGNOSTIC"
FINAL_ANALYSIS_STATUS = "POST_OUTCOME_EXPLORATORY_DIAGNOSTIC"

SOURCE_FILES = (
    "population.csv",
    "contrasts.csv",
    "decomposition.csv",
    "case_mix_standardization.csv",
    "score_structure.csv",
)
COHORTS = ("RIH", "SR1482")
LINEAGES = (
    "family_naive_univ1_5seed",
    "tcga_surgen_univ1_5seed",
    "tcga_surgen_virchow2_cls_5seed",
)
LINEAGE_LABELS = {
    "family_naive_univ1_5seed": "Family-naive UNI-v1",
    "tcga_surgen_univ1_5seed": "TCGA+SurGen UNI-v1",
    "tcga_surgen_virchow2_cls_5seed": "TCGA+SurGen Virchow2",
}
LINEAGE_MARKERS = {
    "family_naive_univ1_5seed": "o",
    "tcga_surgen_univ1_5seed": "s",
    "tcga_surgen_virchow2_cls_5seed": "^",
}

COHORT_COLORS = {"RIH": "#0072B2", "SR1482": "#D55E00"}
SOURCE_COLORS = {"biopsy": "#009E73", "resection": "#E69F00", "unknown": "#8F8F8F"}
SITE_COLORS = {"liver": "#CC79A7", "non_liver": "#56B4E9"}
TEXT_COLOR = "#202020"
GRID_COLOR = "#D7D7D7"


class FigureBuildError(RuntimeError):
    """A fail-closed source, topology, value, or publication error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_symlink_chain(path: Path, *, context: str) -> None:
    absolute = path if path.is_absolute() else path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise FigureBuildError(f"{context} contains a symlink component: {component}")


def _identity(path: Path) -> dict[str, Any]:
    _reject_symlink_chain(path, context="artifact path")
    try:
        before = path.stat()
    except FileNotFoundError as exc:
        raise FigureBuildError(f"missing artifact: {path}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise FigureBuildError(f"artifact is not a regular file: {path}")
    digest = _sha256(path)
    after = path.stat()
    before_key = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_key = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_key != after_key:
        raise FigureBuildError(f"artifact changed while hashing: {path}")
    return {"path": str(path), "size_bytes": after.st_size, "sha256": digest}


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise FigureBuildError(f"duplicate JSON key: {key!r}")
        parsed[key] = value
    return parsed


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise FigureBuildError(f"non-finite JSON constant {value!r} in {path}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FigureBuildError(f"invalid strict JSON in {path}: {exc}") from exc

    def reject_nonfinite(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise FigureBuildError(f"non-finite numeric JSON value in {path}")
        if isinstance(value, Mapping):
            for item in value.values():
                reject_nonfinite(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                reject_nonfinite(item)

    if not isinstance(payload, dict):
        raise FigureBuildError(f"JSON root must be an object: {path}")
    reject_nonfinite(payload)
    return payload


def _one(frame: pd.DataFrame, *, context: str, **filters: object) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        if column not in selected.columns:
            raise FigureBuildError(f"{context}: missing filter column {column!r}")
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise FigureBuildError(
            f"{context}: expected one row for {filters}, observed {len(selected)}"
        )
    return selected.iloc[0]


def _finite_row(row: pd.Series, fields: Sequence[str], *, context: str) -> None:
    for field in fields:
        if field not in row or not math.isfinite(float(row[field])):
            raise FigureBuildError(f"{context}: {field} is missing or non-finite")


def _authenticate_source(source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.absolute()
    _reject_symlink_chain(source_dir, context="verified source bundle")
    if not source_dir.is_dir():
        raise FigureBuildError(f"verified source bundle is absent: {source_dir}")
    with contextlib.redirect_stdout(io.StringIO()):
        verify_diagnostic_bundle(source_dir)
    receipt = _strict_json(source_dir / "receipt.json")
    if receipt.get("status") != FINAL_STATUS:
        raise FigureBuildError(f"source bundle is not a completed final diagnostic: {source_dir}")
    settings = receipt.get("settings")
    if (
        not isinstance(settings, dict)
        or type(settings.get("n_bootstrap")) is not int
        or type(settings.get("cv_repeats")) is not int
        or settings["n_bootstrap"] < 10_000
        or settings["cv_repeats"] < 20
    ):
        raise FigureBuildError("source bundle does not meet the definitive execution settings")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise FigureBuildError("source receipt output map is invalid")
    for name in SOURCE_FILES:
        record = outputs.get(name)
        if not isinstance(record, dict):
            raise FigureBuildError(f"source receipt omits {name}")
        actual = _identity(source_dir / name)
        if actual != record:
            raise FigureBuildError(f"source identity drift: {name}")
    results = _strict_json(source_dir / "results.json")
    if results.get("analysis_status") != FINAL_ANALYSIS_STATUS:
        raise FigureBuildError("source results are not the definitive exploratory diagnostic")
    if results.get("causal_status") != "NONCAUSAL_OBSERVATIONAL_DECOMPOSITION":
        raise FigureBuildError("source causal-status boundary drift")
    return receipt


def _load_tables(source_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {name: pd.read_csv(source_dir / name) for name in SOURCE_FILES}
    if set(tables["population.csv"]["cohort"]) != set(COHORTS):
        raise FigureBuildError("population cohort roster drift")
    if set(tables["score_structure.csv"]["cohort"]) != set(COHORTS):
        raise FigureBuildError("score-structure cohort roster drift")
    return tables


def _validate_panel_topology(tables: Mapping[str, pd.DataFrame]) -> None:
    population = tables["population.csv"]
    contrasts = tables["contrasts.csv"]
    decomposition = tables["decomposition.csv"]
    case_mix = tables["case_mix_standardization.csv"]
    structure = tables["score_structure.csv"]

    for cohort in COHORTS:
        for role in ("primary", "metastatic"):
            selected = population[
                population["cohort"].eq(cohort) & population["role"].eq(role)
            ]
            if selected.empty or selected["n"].sum() <= 0:
                raise FigureBuildError(f"population topology absent for {cohort}/{role}")
        for contrast in (
            "metastatic_minus_primary/all",
            "metastatic_minus_primary/set_d",
        ):
            row = _one(
                contrasts,
                context="role contrast",
                lineage="family_naive_univ1_5seed",
                cohort=cohort,
                contrast=contrast,
            )
            _finite_row(
                row,
                ("first_auroc", "second_auroc", "delta_auroc", "ci_low", "ci_high"),
                context=f"{cohort}/{contrast}",
            )
        for lineage in LINEAGES:
            for contrast in (
                "metastatic_liver_minus_non_liver",
                "metastatic_biopsy_minus_resection",
            ):
                row = _one(
                    contrasts,
                    context="metastatic contrast",
                    lineage=lineage,
                    cohort=cohort,
                    contrast=contrast,
                )
                _finite_row(
                    row,
                    ("delta_auroc", "ci_low", "ci_high"),
                    context=f"{cohort}/{lineage}/{contrast}",
                )
        for analysis in ("source_type", "mutant_subtype"):
            row = _one(
                decomposition,
                context="composition standardization",
                cohort=cohort,
                analysis=analysis,
            )
            _finite_row(
                row,
                ("standardized_difference", "standardized_ci_low", "standardized_ci_high"),
                context=f"{cohort}/{analysis}",
            )
        for block in (
            "molecular",
            "clinical_molecular",
            "joint_source_clinical_molecular",
        ):
            _one(case_mix, context="case-mix standardization", cohort=cohort, block=block)
        row = _one(structure, context="score structure", cohort=cohort)
        _finite_row(
            row,
            (
                "wild_type_metastatic_minus_primary_mean_logit",
                "wild_type_shift_ci_low",
                "wild_type_shift_ci_high",
                "mutant_metastatic_minus_primary_mean_logit",
                "mutant_shift_ci_low",
                "mutant_shift_ci_high",
                "mutant_minus_wild_type_separation_shift",
                "separation_shift_ci_low",
                "separation_shift_ci_high",
            ),
            context=f"{cohort}/score_structure",
        )


def _configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
        }
    )


def _panel_a(ax: plt.Axes, population: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    rows = (
        ("RIH", "primary", "source", 5.4),
        ("RIH", "metastatic", "source", 4.55),
        ("RIH", "metastatic", "site", 3.70),
        ("SR1482", "primary", "source", 2.25),
        ("SR1482", "metastatic", "source", 1.40),
        ("SR1482", "metastatic", "site", 0.55),
    )
    labels: list[str] = []
    y_positions: list[float] = []
    for cohort, role, kind, y in rows:
        selected = population[
            population["cohort"].eq(cohort) & population["role"].eq(role)
        ]
        total = int(selected["n"].sum())
        if kind == "source":
            values = {
                level: int(selected.loc[selected["source_type"].eq(level), "n"].sum())
                for level in ("biopsy", "resection", "unknown")
            }
            colors = SOURCE_COLORS
        else:
            liver = int(selected["liver_n"].sum())
            values = {"liver": liver, "non_liver": total - liver}
            colors = SITE_COLORS
        left = 0.0
        for level, count in values.items():
            if count == 0:
                continue
            width = 100.0 * count / total
            ax.barh(y, width, left=left, height=0.58, color=colors[level], edgecolor="white")
            if width >= 12:
                ax.text(
                    left + width / 2,
                    y,
                    f"{count}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white" if level not in {"resection", "non_liver"} else TEXT_COLOR,
                    fontweight="bold",
                )
            left += width
        ax.text(101.5, y, f"n={total}", va="center", ha="left", fontsize=6.2)
        label_kind = "source" if kind == "source" else "site"
        labels.append(f"{cohort}  {role}\n{label_kind}")
        y_positions.append(y)

    overlap = int(
        _one(
            contrasts,
            context="RIH overlap",
            lineage="family_naive_univ1_5seed",
            cohort="RIH",
            contrast="metastatic_minus_primary/all",
        )["overlap_removed"]
    )
    ax.set_yticks(y_positions, labels)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.05, 6.05)
    ax.set_xticks((0, 50, 100))
    ax.set_xlabel("Patients (%)")
    ax.axhline(3.05, color=GRID_COLOR, lw=0.8)
    ax.set_title("A  Cohort topology and imbalance", loc="left", pad=5)
    ax.text(
        110,
        5.83,
        f"RIH role overlap: {overlap}",
        fontsize=6.1,
        va="bottom",
        ha="right",
    )
    handles = [
        Patch(facecolor=SOURCE_COLORS["biopsy"], label="Biopsy"),
        Patch(facecolor=SOURCE_COLORS["resection"], label="Resection"),
        Patch(facecolor=SOURCE_COLORS["unknown"], label="Unknown source"),
        Patch(facecolor=SITE_COLORS["liver"], label="Liver"),
        Patch(facecolor=SITE_COLORS["non_liver"], label="Non-liver"),
    ]
    ax.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.51, -0.19),
        frameon=False,
        columnspacing=0.9,
        handlelength=1.2,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _role_rows(contrasts: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        cohort: _one(
            contrasts,
            context="observed role contrast",
            lineage="family_naive_univ1_5seed",
            cohort=cohort,
            contrast="metastatic_minus_primary/all",
        )
        for cohort in COHORTS
    }


def _panel_b(spec: Any, figure: plt.Figure, contrasts: pd.DataFrame) -> None:
    nested = spec.subgridspec(1, 2, width_ratios=(1.28, 1.0), wspace=0.25)
    ax_auc = figure.add_subplot(nested[0, 0])
    ax_delta = figure.add_subplot(nested[0, 1], sharey=ax_auc)
    role_rows = _role_rows(contrasts)
    y_map = {"RIH": 1.0, "SR1482": 0.0}
    for cohort in COHORTS:
        row = role_rows[cohort]
        y = y_map[cohort]
        primary = float(row["second_auroc"])
        metastatic = float(row["first_auroc"])
        color = COHORT_COLORS[cohort]
        ax_auc.plot([primary, metastatic], [y, y], color=color, lw=1.2, alpha=0.7)
        ax_auc.scatter(
            [primary], [y], s=28, marker="o", facecolor="white", edgecolor=color, linewidth=1.2, zorder=3
        )
        ax_auc.scatter(
            [metastatic], [y], s=28, marker="o", facecolor=color, edgecolor=color, linewidth=1.0, zorder=3
        )
        ax_auc.text(primary, y + 0.17, f"{primary:.2f}", color=color, ha="center", fontsize=6.3)
        ax_auc.text(
            metastatic, y - 0.19, f"{metastatic:.2f}", color=color, ha="center", fontsize=6.3
        )
        delta = float(row["delta_auroc"])
        low = float(row["ci_low"])
        high = float(row["ci_high"])
        ax_delta.errorbar(
            delta,
            y,
            xerr=[[delta - low], [high - delta]],
            fmt="o",
            color=color,
            capsize=2,
            markersize=4.3,
            lw=1.2,
        )
        ax_delta.text(
            0.98,
            y,
            f"{delta:+.2f}\n[{low:+.2f}, {high:+.2f}]",
            transform=ax_delta.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=5.9,
            color=color,
        )

    labels = []
    for cohort in COHORTS:
        row = role_rows[cohort]
        labels.append(
            f"{cohort}\nP {int(row['second_n'])} / M {int(row['first_n'])}"
        )
    ax_auc.set_yticks([1.0, 0.0], labels)
    ax_auc.set_xlim(0.47, 0.80)
    ax_auc.set_xticks((0.5, 0.6, 0.7, 0.8))
    ax_auc.axvline(0.5, color="#888888", lw=0.8, ls="--")
    ax_auc.set_xlabel("Disjoint-role AUROC (point estimate)")
    ax_auc.set_ylim(-0.5, 1.5)
    ax_auc.set_title("B  Disjoint role ranking", loc="left", pad=5)
    ax_auc.legend(
        handles=[
            Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="#555555", lw=0, label="Primary"),
            Line2D([], [], marker="o", markerfacecolor="#555555", markeredgecolor="#555555", lw=0, label="Metastatic"),
        ],
        frameon=False,
        loc="lower left",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.3,
    )
    ax_delta.set_xlim(-0.36, 0.18)
    ax_delta.set_xticks((-0.3, -0.15, 0.0, 0.15))
    ax_delta.axvline(0, color="#666666", lw=0.8, ls="--")
    ax_delta.set_xlabel("ΔAUROC (M − P), 95% CI")
    ax_delta.tick_params(axis="y", labelleft=False, left=False)
    for ax in (ax_auc, ax_delta):
        ax.grid(axis="x", color=GRID_COLOR, lw=0.5, alpha=0.55)
        ax.spines[["top", "right"]].set_visible(False)


def _panel_c(ax: plt.Axes, contrasts: pd.DataFrame) -> None:
    contrast_rows = (
        ("RIH", "metastatic_liver_minus_non_liver", 3.0, "liver − non-liver"),
        ("RIH", "metastatic_biopsy_minus_resection", 2.0, "biopsy − resection"),
        ("SR1482", "metastatic_liver_minus_non_liver", 1.0, "liver − non-liver"),
        ("SR1482", "metastatic_biopsy_minus_resection", 0.0, "biopsy − resection"),
    )
    offsets = {
        "family_naive_univ1_5seed": 0.20,
        "tcga_surgen_univ1_5seed": 0.0,
        "tcga_surgen_virchow2_cls_5seed": -0.20,
    }
    y_ticks: list[float] = []
    y_labels: list[str] = []
    for cohort, contrast, y, short_label in contrast_rows:
        family = _one(
            contrasts,
            context="metastatic contrast counts",
            lineage="family_naive_univ1_5seed",
            cohort=cohort,
            contrast=contrast,
        )
        support = bool(family["strict_support"])
        if not support:
            ax.axhspan(y - 0.42, y + 0.42, color="#F2F2F2", zorder=0)
        for lineage in LINEAGES:
            row = _one(
                contrasts,
                context="metastatic contrast plot",
                lineage=lineage,
                cohort=cohort,
                contrast=contrast,
            )
            point = float(row["delta_auroc"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            color = COHORT_COLORS[cohort]
            primary_view = lineage == "family_naive_univ1_5seed"
            ax.errorbar(
                point,
                y + offsets[lineage],
                xerr=[[point - low], [high - point]],
                fmt=LINEAGE_MARKERS[lineage],
                markerfacecolor=color if primary_view else "white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                color=color,
                alpha=1.0 if primary_view else 0.82,
                capsize=1.8,
                markersize=4.2,
                lw=1.0,
                zorder=2,
            )
        state = "" if support else "; sparse"
        y_ticks.append(y)
        y_labels.append(
            f"{cohort} · {short_label}\n(n={int(family['first_n'])}/{int(family['second_n'])}{state})"
        )
    ax.set_yticks(y_ticks, y_labels)
    ax.set_xlim(-0.65, 0.72)
    ax.set_xticks((-0.6, -0.3, 0.0, 0.3, 0.6))
    ax.axvline(0, color="#666666", lw=0.8, ls="--")
    ax.grid(axis="x", color=GRID_COLOR, lw=0.5, alpha=0.55)
    ax.set_xlabel("Within-metastatic ΔAUROC, 95% patient-bootstrap CI")
    ax.set_title("C  Metastatic site and source-type contrasts", loc="left", pad=5)
    handles = []
    for lineage in LINEAGES:
        primary_view = lineage == "family_naive_univ1_5seed"
        handles.append(
            Line2D(
                [],
                [],
                marker=LINEAGE_MARKERS[lineage],
                markerfacecolor="#555555" if primary_view else "white",
                markeredgecolor="#555555",
                lw=0,
                label=LINEAGE_LABELS[lineage],
            )
        )
    ax.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.17),
        columnspacing=1.0,
        handletextpad=0.4,
    )
    ax.spines[["top", "right"]].set_visible(False)


def _standardization_rows(
    cohort: str,
    contrasts: pd.DataFrame,
    decomposition: pd.DataFrame,
    case_mix: pd.DataFrame,
) -> list[dict[str, Any]]:
    observed = _one(
        contrasts,
        context="observed standardization reference",
        lineage="family_naive_univ1_5seed",
        cohort=cohort,
        contrast="metastatic_minus_primary/all",
    )
    set_d = _one(
        contrasts,
        context="Set D restriction",
        lineage="family_naive_univ1_5seed",
        cohort=cohort,
        contrast="metastatic_minus_primary/set_d",
    )
    source = _one(
        decomposition,
        context="source standardization",
        cohort=cohort,
        analysis="source_type",
    )
    subtype = _one(
        decomposition,
        context="subtype standardization",
        cohort=cohort,
        analysis="mutant_subtype",
    )
    rows: list[dict[str, Any]] = [
        {
            "label": "Observed",
            "point": float(observed["delta_auroc"]),
            "low": float(observed["ci_low"]),
            "high": float(observed["ci_high"]),
            "kind": "supported",
        },
        {
            "label": "Source-composition\nstandardized",
            "point": float(source["standardized_difference"]),
            "low": float(source["standardized_ci_low"]),
            "high": float(source["standardized_ci_high"]),
            "kind": "descriptive" if not bool(source["strict_cell_support"]) else "supported",
        },
        {
            "label": "KRAS-subtype\nstandardized",
            "point": float(subtype["standardized_difference"]),
            "low": float(subtype["standardized_ci_low"]),
            "high": float(subtype["standardized_ci_high"]),
            "kind": "descriptive" if not bool(subtype["strict_cell_support"]) else "supported",
        },
        {
            "label": "Set D restriction\n(MSS/pMMR + BRAF-WT)",
            "point": float(set_d["delta_auroc"]),
            "low": float(set_d["ci_low"]),
            "high": float(set_d["ci_high"]),
            "kind": "restriction",
        },
    ]
    for block, label in (
        ("molecular", "Molecular\nodds-weighted"),
        ("clinical_molecular", "Clinical + molecular\nodds-weighted"),
        (
            "joint_source_clinical_molecular",
            "Source + clinical + molecular\nodds-weighted",
        ),
    ):
        row = _one(case_mix, context="case-mix plot", cohort=cohort, block=block)
        evidence = str(row["evidence_state"])
        point = row["att_standardized_difference"]
        low = row["att_standardized_ci_low"]
        high = row["att_standardized_ci_high"]
        if evidence.startswith("ESTIMABLE_") and all(pd.notna(value) for value in (point, low, high)):
            rows.append(
                {
                    "label": label,
                    "point": float(point),
                    "low": float(low),
                    "high": float(high),
                    "kind": "weighted",
                }
            )
        else:
            reason = (
                "support/balance"
                if evidence == "NOT_ESTIMABLE_NO_COMMON_SUPPORT_OR_BALANCE"
                else f"{float(row['bootstrap_valid_fraction']):.0%} valid"
            )
            rows.append({"label": label, "kind": "not_estimable", "reason": reason})
    return rows


def _panel_d(
    spec: Any,
    figure: plt.Figure,
    contrasts: pd.DataFrame,
    decomposition: pd.DataFrame,
    case_mix: pd.DataFrame,
) -> None:
    nested = spec.subgridspec(1, 2, wspace=0.22)
    labels = [
        "Observed",
        "Source-composition\nstandardized",
        "KRAS-subtype\nstandardized",
        "Set D restriction\n(MSS/pMMR + BRAF-WT)",
        "Molecular\nodds-weighted",
        "Clinical + molecular\nodds-weighted",
        "Source + clinical + molecular\nodds-weighted",
    ]
    for cohort_index, cohort in enumerate(COHORTS):
        ax = figure.add_subplot(nested[0, cohort_index])
        rows = _standardization_rows(cohort, contrasts, decomposition, case_mix)
        y_positions = np.arange(len(rows) - 1, -1, -1, dtype=float)
        ax.axhspan(-0.45, 2.45, color="#F5F5F5", zorder=0)
        for y, row in zip(y_positions, rows, strict=True):
            kind = str(row["kind"])
            if kind == "not_estimable":
                ax.text(
                    0.195,
                    y,
                    f"NE\n{row['reason']}",
                    ha="right",
                    va="center",
                    fontsize=5.8,
                    color="#666666",
                    bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "#AAAAAA", "lw": 0.6},
                )
                continue
            point = float(row["point"])
            low = float(row["low"])
            high = float(row["high"])
            marker = {"supported": "o", "descriptive": "o", "restriction": "D", "weighted": "s"}[kind]
            face = "white" if kind == "descriptive" else COHORT_COLORS[cohort]
            linestyle = ":" if kind == "descriptive" else "-"
            ax.errorbar(
                point,
                y,
                xerr=[[point - low], [high - point]],
                fmt=marker,
                markerfacecolor=face,
                markeredgecolor=COHORT_COLORS[cohort],
                markeredgewidth=1.0,
                color=COHORT_COLORS[cohort],
                capsize=1.8,
                markersize=4.2,
                lw=1.0,
                ls=linestyle,
            )
        ax.axvline(0, color="#666666", lw=0.8, ls="--")
        ax.axhline(2.5, color="#BDBDBD", lw=0.7)
        ax.set_xlim(-0.42, 0.22)
        ax.set_xticks((-0.4, -0.2, 0.0, 0.2))
        ax.set_ylim(-0.55, 6.55)
        ax.set_yticks(y_positions, labels if cohort_index == 0 else [])
        ax.grid(axis="x", color=GRID_COLOR, lw=0.5, alpha=0.55)
        title = f"D  Noncausal ladder — {cohort}" if cohort_index == 0 else cohort
        ax.set_title(title, color=COHORT_COLORS[cohort], fontweight="bold", pad=4, loc="left")
        ax.set_xlabel("ΔAUROC (M − P)" if cohort_index == 0 else "")
        ax.spines[["top", "right"]].set_visible(False)
        if cohort_index == 1:
            ax.tick_params(axis="y", left=False)


def _panel_e(ax: plt.Axes, structure: pd.DataFrame) -> None:
    fields = (
        (
            "WT mean-logit shift",
            "wild_type_metastatic_minus_primary_mean_logit",
            "wild_type_shift_ci_low",
            "wild_type_shift_ci_high",
            "o",
            False,
        ),
        (
            "Mutant mean-logit shift",
            "mutant_metastatic_minus_primary_mean_logit",
            "mutant_shift_ci_low",
            "mutant_shift_ci_high",
            "^",
            True,
        ),
        (
            "Class-separation shift",
            "mutant_minus_wild_type_separation_shift",
            "separation_shift_ci_low",
            "separation_shift_ci_high",
            "D",
            True,
        ),
    )
    y_map = {
        ("RIH", 0): 5.0,
        ("RIH", 1): 4.0,
        ("RIH", 2): 3.0,
        ("SR1482", 0): 1.55,
        ("SR1482", 1): 0.55,
        ("SR1482", 2): -0.45,
    }
    ticks: list[float] = []
    labels: list[str] = []
    ax.axhspan(2.60, 3.40, color="#F2F2F2", zorder=0)
    ax.axhspan(-0.85, -0.05, color="#F2F2F2", zorder=0)
    for cohort in COHORTS:
        row = _one(structure, context="score structure plot", cohort=cohort)
        for index, (label, point_field, low_field, high_field, marker, filled) in enumerate(fields):
            y = y_map[(cohort, index)]
            point = float(row[point_field])
            low = float(row[low_field])
            high = float(row[high_field])
            color = COHORT_COLORS[cohort]
            ax.errorbar(
                point,
                y,
                xerr=[[point - low], [high - point]],
                fmt=marker,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=1.0,
                color=color,
                capsize=1.8,
                markersize=4.3,
                lw=1.0,
            )
            ticks.append(y)
            short = {
                "WT mean-logit shift": "WT",
                "Mutant mean-logit shift": "Mutant",
                "Class-separation shift": "Separation",
            }[label]
            labels.append(f"{cohort} · {short}")
    ax.set_yticks(ticks, labels)
    ax.set_xlim(-3.2, 3.2)
    ax.set_xticks((-3, -2, -1, 0, 1, 2, 3))
    ax.axvline(0, color="#666666", lw=0.8, ls="--")
    ax.grid(axis="x", color=GRID_COLOR, lw=0.5, alpha=0.55)
    ax.set_xlabel("Logit shift (M − P), 95% CI")
    ax.set_title(r"$\mathbf{E}$  Conformant score-distribution shifts", loc="left", pad=5)
    ax.spines[["top", "right"]].set_visible(False)


def _build_figure(tables: Mapping[str, pd.DataFrame]) -> plt.Figure:
    _configure_style()
    figure = plt.figure(figsize=(10.8, 9.2), constrained_layout=False)
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.935,
        bottom=0.075,
        hspace=0.72,
        wspace=0.72,
    )
    outer = figure.add_gridspec(
        3,
        12,
        height_ratios=(1.06, 0.92, 1.52),
    )
    ax_a = figure.add_subplot(outer[0, :5])
    _panel_a(ax_a, tables["population.csv"], tables["contrasts.csv"])
    _panel_b(outer[0, 5:], figure, tables["contrasts.csv"])
    ax_c = figure.add_subplot(outer[1, :])
    _panel_c(ax_c, tables["contrasts.csv"])
    _panel_d(
        outer[2, :8],
        figure,
        tables["contrasts.csv"],
        tables["decomposition.csv"],
        tables["case_mix_standardization.csv"],
    )
    ax_e = figure.add_subplot(outer[2, 8:])
    _panel_e(ax_e, tables["score_structure.csv"])
    figure.suptitle(
        "Post-outcome decomposition of primary-to-metastatic KRAS ranking differences",
        fontsize=11.2,
        fontweight="bold",
    )
    figure.text(
        0.5,
        -0.005,
        "Post-outcome exploratory sensitivities; patient-bootstrap intervals are unadjusted for multiplicity.",
        ha="center",
        va="top",
        fontsize=6.2,
        color="#555555",
    )
    return figure


def _source_note(
    source_dir: Path,
    receipt: Mapping[str, Any],
    figure_identities: Mapping[str, Mapping[str, Any]],
) -> str:
    settings = receipt["settings"]
    receipt_identity = _identity(source_dir / "receipt.json")
    script_identity = _identity(Path(__file__).resolve())
    role_rows = _role_rows(pd.read_csv(source_dir / "contrasts.csv"))
    source_lines = []
    for name in SOURCE_FILES:
        record = receipt["outputs"][name]
        source_lines.append(
            f"| `{name}` | `{record['sha256']}` | {int(record['size_bytes']):,} |"
        )
    output_lines = []
    for name in (FIGURE_PNG, FIGURE_PDF):
        record = figure_identities[name]
        output_lines.append(
            f"| `{name}` | `{record['sha256']}` | {int(record['size_bytes']):,} |"
        )
    rih = role_rows["RIH"]
    sr = role_rows["SR1482"]
    return "\n".join(
        [
            "# Publication-candidate metastatic-transport figure",
            "",
            "## Caption",
            "",
            "**Post-outcome decomposition of primary-to-metastatic KRAS ranking differences.** "
            "(A) Patient composition by user-supplied biopsy/resection source type and recorded "
            "metastatic liver/non-liver site; unknown source is retained explicitly. "
            "(B) Family-naive UNI-v1 five-seed AUROC point estimates in disjoint primary and "
            "metastatic populations, with the metastatic-minus-primary AUROC difference and 95% "
            "patient-bootstrap interval. (C) Within-metastatic liver-minus-non-liver and "
            "biopsy-minus-resection AUROC contrasts. Open symbols are correlated TCGA+SurGen "
            "score-view sensitivities evaluated on the same patients, not independent "
            "replications; shaded source-type rows fail the prespecified count-support rule. "
            "(D) Separate, noncausal sensitivity estimands for observed ranking, source-composition "
            "standardization, KRAS-mutant-subtype standardization, Set D MSS/pMMR+BRAF-wild-type "
            "restriction, and class-stratified odds-weighted covariate standardization to the "
            "metastatic distribution. Open points are descriptive sparse-cell sensitivities; NE "
            "denotes an unavailable gated estimate or interval because of support, balance, or "
            "bootstrap-overlap failure. Rows are not additive fractions explained. "
            "(E) Within-cohort, conformant family-naive logit shifts for wild-type and mutant "
            "patients and the corresponding class-separation shift. These are score-distribution "
            "diagnostics, not biological or technical mechanisms. RIH role comparisons remove the "
            f"{int(rih['overlap_removed'])} patients shared across roles from both arms "
            f"(primary n={int(rih['second_n'])}, metastatic n={int(rih['first_n'])}); SR1482 has "
            f"no role overlap (primary n={int(sr['second_n'])}, metastatic n={int(sr['first_n'])}). "
            "Patients are the analysis and bootstrap units; seeds, folds, and slides are not "
            "independent replicates. All analyses are post-outcome, noncausal, and unadjusted for "
            "multiple correlated comparisons.",
            "",
            "## Evidence and population rules",
            "",
            "- Panels B, D, and E use disjoint role populations; Panel C uses all metastatic patients.",
            "- Panel D plots only gated `att_standardized_difference` values with reportable intervals; "
            "it never plots `diagnostic_ungated_*` fields.",
            "- Composition target weights and odds weights are re-estimated inside each patient-bootstrap draw.",
            "- The main figure excludes `patient_embeddings.parquet` and `structural_cv.csv`; the "
            "averaged UNI-v1 slide-mean decodability analysis is audit-only and is not promoted here.",
            "- Raw logits are interpreted only within cohort and the conformant family-naive score view.",
            "",
            "## Authenticated definitive source",
            "",
            f"- Bundle: `{source_dir}`",
            f"- Status: `{receipt['status']}`",
            f"- Settings: `{int(settings['n_bootstrap']):,}` primary patient bootstraps; "
            f"`{int(settings['cv_repeats'])}` repeated CV splits; random seed `{int(settings['random_seed'])}`.",
            f"- Source receipt: `{receipt_identity['sha256']}` ({int(receipt_identity['size_bytes']):,} bytes)",
            f"- Figure script: `{script_identity['sha256']}` ({int(script_identity['size_bytes']):,} bytes)",
            "- The definitive source bundle was verified immediately before and after rendering.",
            "",
            "| Source file | SHA-256 | Bytes |",
            "|---|---:|---:|",
            *source_lines,
            "",
            "## Derived image identities",
            "",
            "| Derived file | SHA-256 | Bytes |",
            "|---|---:|---:|",
            *output_lines,
            "",
            "Generation command:",
            "",
            "```bash",
            "python tools/crc_final_v6_met_transfer_paper_figure.py",
            "```",
            "",
        ]
    )


def build(source_dir: Path, output_dir: Path) -> None:
    source_dir = source_dir.absolute()
    output_dir = output_dir.absolute()
    if output_dir == source_dir or source_dir in output_dir.parents:
        raise FigureBuildError("derived output must not be inside the verified source bundle")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite derived output: {output_dir}")
    _reject_symlink_chain(output_dir.parent, context="derived output parent")

    receipt = _authenticate_source(source_dir)
    source_snapshot = {
        name: _identity(source_dir / name) for name in ("receipt.json", *SOURCE_FILES)
    }
    tables = _load_tables(source_dir)
    _validate_panel_topology(tables)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        figure = _build_figure(tables)
        png_path = temporary / FIGURE_PNG
        pdf_path = temporary / FIGURE_PDF
        figure.savefig(
            png_path,
            dpi=450,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "matplotlib", "Title": "CRC metastatic transport diagnostic"},
        )
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            facecolor="white",
            metadata={
                "Title": "CRC metastatic transport diagnostic",
                "Author": "OceanPath analysis",
                "Subject": "Post-outcome noncausal metastatic-transport sensitivity figure",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(figure)
        figure_identities = {
            FIGURE_PNG: _identity(png_path),
            FIGURE_PDF: _identity(pdf_path),
        }
        (temporary / SOURCE_NOTE).write_text(
            _source_note(source_dir, receipt, figure_identities), encoding="utf-8"
        )
        expected_outputs = {FIGURE_PNG, FIGURE_PDF, SOURCE_NOTE}
        if {path.name for path in temporary.iterdir()} != expected_outputs:
            raise FigureBuildError("derived output roster drift")
        for path in temporary.iterdir():
            _identity(path)

        with contextlib.redirect_stdout(io.StringIO()):
            verify_diagnostic_bundle(source_dir)
        if {
            name: _identity(source_dir / name) for name in ("receipt.json", *SOURCE_FILES)
        } != source_snapshot:
            raise FigureBuildError("verified source changed while rendering")
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build(args.source_dir, args.output_dir)
    print(f"CREATED: {args.output_dir.absolute()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
