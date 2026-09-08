from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v7_bundle_receipt as bundle_verifier  # noqa: E402
from tools import final_v7_integration as integration  # noqa: E402

TEST_CONTRACT = integration.StudyContract(
    cohorts={
        "RIH": integration.CohortContract(n=8, n_mutant=4, e2b_native_auroc=1.0),
        "SurGen": integration.CohortContract(n=8, n_mutant=4, e2b_native_auroc=1.0),
    },
    outer_seeds=(101, 102),
    primary_outer_seed=102,
    folds=2,
    model_seeds=(42,),
    support_sizes=(2, 4),
    support_draws=2,
    lambda_grid=integration.EXPECTED_LAMBDA_GRID,
    bootstrap_draws=10,
    bootstrap_seed=7,
    embedding_dimensions=2,
    review_cases=6,
    review_panels_per_case=2,
    review_prior_exposed=3,
    review_p17_counts={"absent": 2, "positive_low": 2, "positive_high": 2},
    review_kras_counts={"mutant": 3, "wild_type": 3},
)


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    _write(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_record(point: float, *, auroc_metric: bool = False) -> dict[str, object]:
    if auroc_metric:
        return {"point": point, "ci95": [0.7, 1.0]}
    return {"point": point, "ci95": [max(0.0, point - 0.01), point + 0.01]}


def _patient_rows(cohort: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(8):
        label = int(index >= 4)
        native = 2.0 if label else -2.0
        rows.append(
            {
                "cohort": cohort,
                "patient_id": f"{cohort}_PATIENT_{index}",
                "label": label,
                "fold": index % 2,
                "eta_native": native,
                "eta_adapted": native,
                "eta_platt": native,
                "eta_native_seed42": native,
                "eta_adapted_seed42": native,
            }
        )
    return rows


def _layout(rows_by_cohort: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    procedures = ("native", "adapted", "platt")
    metrics = ("auroc", "log_loss", "brier")
    per_cohort: dict[str, object] = {}
    calculated: dict[str, dict[str, dict[str, float]]] = {}
    for cohort, rows in rows_by_cohort.items():
        calculated[cohort] = {
            procedure: integration.point_metrics(rows, procedure)
            for procedure in procedures
        }
        procedure_block = {
            procedure: {
                metric: _metric_record(
                    calculated[cohort][procedure][metric],
                    auroc_metric=metric == "auroc",
                )
                for metric in metrics
            }
            for procedure in procedures
        }
        contrast_block: dict[str, object] = {}
        for procedure in ("adapted", "platt"):
            name = f"{procedure}_minus_native"
            contrast_block[name] = {
                metric: {
                    "point": calculated[cohort][procedure][metric]
                    - calculated[cohort]["native"][metric],
                    "ci95": [-0.1, 0.1],
                }
                for metric in metrics
            }
        per_cohort[cohort] = {
            "n": 8,
            "n_mutant": 4,
            "procedures": procedure_block,
            "contrasts": contrast_block,
        }
    macro_procedures: dict[str, object] = {}
    for procedure in procedures:
        macro_procedures[procedure] = {
            metric: _metric_record(
                sum(calculated[cohort][procedure][metric] for cohort in calculated)
                / len(calculated),
                auroc_metric=metric == "auroc",
            )
            for metric in metrics
        }
    macro_contrasts: dict[str, object] = {}
    for procedure in ("adapted", "platt"):
        name = f"{procedure}_minus_native"
        macro_contrasts[name] = {
            metric: {
                "point": sum(
                    calculated[cohort][procedure][metric]
                    - calculated[cohort]["native"][metric]
                    for cohort in calculated
                )
                / len(calculated),
                "ci95": [-0.1, 0.1],
            }
            for metric in metrics
        }
    return {
        "bootstrap_draws_requested": TEST_CONTRACT.bootstrap_draws,
        "bootstrap_draws_valid": TEST_CONTRACT.bootstrap_draws,
        "bootstrap_seed": TEST_CONTRACT.bootstrap_seed,
        "per_cohort": per_cohort,
        "macro": {"procedures": macro_procedures, "contrasts": macro_contrasts},
        "contrast_sign_convention": "procedure minus native",
        "fixed_gate_adapted": {
            "both_cohort_points_above_0p5": True,
            "macro_ci_lower_above_0p5": True,
            "pass": True,
        },
        "incremental_improvement_established": {
            "macro_auroc_delta_ci_lower_above_zero": False,
            "both_cohort_auroc_delta_points_above_zero": False,
            "pass": False,
        },
    }


def _solver_diagnostic() -> dict[str, object]:
    return {
        "status": "finite_optimum",
        "lambda": 0.003,
        "scipy_success": True,
        "scipy_status": 0,
        "gradient_inf_norm": 0.0,
        "objective_at_zero": 0.5,
        "objective_at_fit": 0.4,
        "objective_decrease": 0.1,
        "coefficient_l2": 0.0,
        "bias": 0.0,
        "accepted_by": "scipy_success",
    }


def _inner_selection(n_pool: int, selected: str = "0.003") -> dict[str, object]:
    return {
        "scheme": "leave_one_out",
        "inner_seed": 1,
        "n_pool": n_pool,
        "n_splits": n_pool,
        "losses_by_lambda": {key: [0.1] for key in TEST_CONTRACT.lambda_grid},
        "mean_loss_by_lambda": {key: 0.1 for key in TEST_CONTRACT.lambda_grid},
        "solver_summary_by_lambda": {key: {} for key in TEST_CONTRACT.lambda_grid},
        "selected_lambda": selected,
    }


def _fit_record(
    *,
    phase: str,
    model_kind: str,
    cohort: str,
    outer_seed: int,
    fold: int,
    fit_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    support: int | None = None,
    draw: int | None = None,
    model_seed: int | None = 42,
) -> dict[str, object]:
    fit_ids = [str(row["patient_id"]) for row in fit_rows]
    fit_labels = [int(row["label"]) for row in fit_rows]
    selected = None if model_kind == "platt" else "0.003"
    return {
        "phase": phase,
        "model_kind": model_kind,
        "cohort": cohort,
        "outer_seed": outer_seed,
        "outer_fold": fold,
        "model_seed": None if model_kind == "platt" else model_seed,
        "support_requested": support,
        "support_realized": len(fit_rows) if support is not None else None,
        "draw": draw,
        "support_seed": (9000 + (support or 0) * 100 + (draw or 0) * 10 + fold) if support is not None else None,
        "fit_patient_ids": fit_ids,
        "fit_labels": fit_labels,
        "n_fit": len(fit_rows),
        "n_fit_class0": fit_labels.count(0),
        "n_fit_class1": fit_labels.count(1),
        "test_patient_ids": [str(row["patient_id"]) for row in test_rows],
        "n_test": len(test_rows),
        "selected_lambda": selected,
        "inner_selection": None if model_kind == "platt" else _inner_selection(len(fit_rows)),
        "coefficients": [0.0, 1.0] if model_kind == "platt" else [0.0, 0.0],
        "bias": 0.0,
        "solver_diagnostic": _solver_diagnostic(),
    }


def _build_e2f(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True)
    rows_by_cohort = {cohort: _patient_rows(cohort) for cohort in TEST_CONTRACT.cohorts}
    oof_rows: list[dict[str, object]] = []
    for outer_seed in TEST_CONTRACT.outer_seeds:
        for _cohort, rows in rows_by_cohort.items():
            for row in rows:
                oof_rows.append(
                    {
                        "phase": "full_label",
                        "outer_seed": outer_seed,
                        "is_primary": outer_seed == TEST_CONTRACT.primary_outer_seed,
                        "support_requested": "",
                        "draw": "",
                        **row,
                    }
                )
    for support in TEST_CONTRACT.support_sizes:
        for draw in range(TEST_CONTRACT.support_draws):
            for _cohort, rows in rows_by_cohort.items():
                for row in rows:
                    oof_rows.append(
                        {
                            "phase": "support_curve",
                            "outer_seed": TEST_CONTRACT.primary_outer_seed,
                            "is_primary": False,
                            "support_requested": support,
                            "draw": draw,
                            **{**row, "eta_platt": ""},
                        }
                    )
    oof_path = root / "oof_predictions.csv"
    _write_csv(oof_path, list(oof_rows[0]), oof_rows)

    fits: list[dict[str, object]] = []
    for outer_seed in TEST_CONTRACT.outer_seeds:
        for cohort, rows in rows_by_cohort.items():
            for fold in range(TEST_CONTRACT.folds):
                test_rows = [row for row in rows if row["fold"] == fold]
                train_rows = [row for row in rows if row["fold"] != fold]
                fits.append(
                    _fit_record(
                        phase="full_label",
                        model_kind="residual_adapter",
                        cohort=cohort,
                        outer_seed=outer_seed,
                        fold=fold,
                        fit_rows=train_rows,
                        test_rows=test_rows,
                    )
                )
                fits.append(
                    _fit_record(
                        phase="full_label",
                        model_kind="platt",
                        cohort=cohort,
                        outer_seed=outer_seed,
                        fold=fold,
                        fit_rows=train_rows,
                        test_rows=test_rows,
                    )
                )
    for support in TEST_CONTRACT.support_sizes:
        for draw in range(TEST_CONTRACT.support_draws):
            for cohort, rows in rows_by_cohort.items():
                for fold in range(TEST_CONTRACT.folds):
                    test_rows = [row for row in rows if row["fold"] == fold]
                    train_rows = [row for row in rows if row["fold"] != fold]
                    support_rows = [row for row in train_rows if row["label"] == 0][
                        : support // 2
                    ] + [row for row in train_rows if row["label"] == 1][: support // 2]
                    fits.append(
                        _fit_record(
                            phase="support_curve",
                            model_kind="residual_adapter",
                            cohort=cohort,
                            outer_seed=TEST_CONTRACT.primary_outer_seed,
                            fold=fold,
                            fit_rows=support_rows,
                            test_rows=test_rows,
                            support=support,
                            draw=draw,
                        )
                    )
    fits_path = root / "fits.jsonl"
    _write(
        fits_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in fits),
    )

    layouts = {
        str(seed): _layout(rows_by_cohort) for seed in TEST_CONTRACT.outer_seeds
    }
    support_groups: dict[tuple[int, int, str], list[dict[str, object]]] = {}
    for support in TEST_CONTRACT.support_sizes:
        for draw in range(TEST_CONTRACT.support_draws):
            for cohort, rows in rows_by_cohort.items():
                support_groups[(support, draw, cohort)] = rows
    support_summary = integration._recompute_support_curve(  # noqa: SLF001
        support_groups, TEST_CONTRACT
    )
    full_adapter = [
        row
        for row in fits
        if row["phase"] == "full_label" and row["model_kind"] == "residual_adapter"
    ]
    support_adapter = [row for row in fits if row["phase"] == "support_curve"]
    primary_adapter = [
        row
        for row in full_adapter
        if row["outer_seed"] == TEST_CONTRACT.primary_outer_seed
    ]
    primary = layouts[str(TEST_CONTRACT.primary_outer_seed)]
    artifact_census = {
        "oof_rows_total": len(oof_rows),
        "oof_rows_full_label": TEST_CONTRACT.expected_oof_full,
        "oof_rows_primary_full_label": TEST_CONTRACT.patient_total,
        "oof_rows_support_curve": TEST_CONTRACT.expected_oof_support,
        "fit_records_total": len(fits),
        "fit_records_full_label_adapter": TEST_CONTRACT.expected_full_adapter_fits,
        "fit_records_full_label_platt": TEST_CONTRACT.expected_full_platt_fits,
        "fit_records_support_adapter": TEST_CONTRACT.expected_support_adapter_fits,
        "support_draw_records": len(TEST_CONTRACT.support_sizes)
        * TEST_CONTRACT.support_draws,
        "solver_warning_records": 0,
    }
    solver = {
        "n_finite_calls": 100,
        "n_native_exact_calls": 0,
        "n_scipy_success": 100,
        "n_accepted_small_gradient": 0,
        "max_gradient_inf_norm": 0.0,
        "min_objective_decrease": 0.0,
        "n_unaccepted": 0,
        "scipy_status_counts": {"0": 100},
    }
    results = {
        "schema_version": 1,
        "experiment": "e2f_v3_full_label_residual_adapter",
        "status": "PASS",
        "problems": [],
        "scope_warning": (
            "Target-label internal metastatic cross-fitting; not independent deployment "
            "validation, prospective validation, or clinical-readiness evidence."
        ),
        "design": {
            "residual_parameters": TEST_CONTRACT.embedding_dimensions + 1,
            "lambda_grid": list(TEST_CONTRACT.lambda_grid),
            "primary_outer_seed": TEST_CONTRACT.primary_outer_seed,
            "outer_sensitivity_seeds": list(TEST_CONTRACT.outer_seeds),
            "outer_folds": TEST_CONTRACT.folds,
            "model_seeds": list(TEST_CONTRACT.model_seeds),
            "cohorts_fit_separately": True,
            "support_sizes_exact_total": list(TEST_CONTRACT.support_sizes),
            "support_balance": "exactly half class 0 and half class 1",
            "support_draws": TEST_CONTRACT.support_draws,
            "bootstrap_draws": TEST_CONTRACT.bootstrap_draws,
            "bootstrap_seed": TEST_CONTRACT.bootstrap_seed,
        },
        "input_census": {
            cohort: {
                "n_slides": expected.n,
                "n_patients": expected.n,
                "n_mutant": expected.n_mutant,
                "n_embedding_dimensions": TEST_CONTRACT.embedding_dimensions,
                "model_seeds": list(TEST_CONTRACT.model_seeds),
                "label_discordances": 0,
                "nonfinite_values": 0,
            }
            for cohort, expected in TEST_CONTRACT.cohorts.items()
        },
        "frozen_baseline_checks": {
            cohort: {"observed": 1.0, "expected": expected.e2b_native_auroc, "pass": True}
            for cohort, expected in TEST_CONTRACT.cohorts.items()
        },
        "primary": primary,
        "primary_macro_delta_auroc": primary["macro"]["contrasts"]
        ["adapted_minus_native"]["auroc"],
        "primary_incremental_improvement_established": primary[
            "incremental_improvement_established"
        ],
        "outer_fold_sensitivity": layouts,
        "outer_fold_sensitivity_summary": {
            "n_layouts": len(TEST_CONTRACT.outer_seeds),
            "n_gate_pass": len(TEST_CONTRACT.outer_seeds),
            "macro_adapted_auroc_range": [1.0, 1.0],
            "macro_delta_auroc_range": [0.0, 0.0],
        },
        "label_efficiency_curve": support_summary,
        "lambda_inventory_primary": integration._derive_lambda_inventory(  # noqa: SLF001
            primary_adapter, TEST_CONTRACT
        ),
        "lambda_inventory_full_label_all_layouts": integration._derive_lambda_inventory(  # noqa: SLF001
            full_adapter, TEST_CONTRACT
        ),
        "lambda_inventory_support_curve": integration._derive_lambda_inventory(  # noqa: SLF001
            support_adapter, TEST_CONTRACT
        ),
        "solver": solver,
        "artifact_census": artifact_census,
        "interpretation": "internal adaptation feasibility only",
    }
    results_path = root / "results.json"
    _write_json(results_path, results)
    support_draws = root / "support_draws.jsonl"
    _write(support_draws, "{}\n")
    warnings = root / "solver_warnings.jsonl"
    _write(warnings, "")
    source_input = root.parent / "sealed_input.dat"
    source_code = root.parent / "aim2_v3_fulllabel_residual_adaptation.py"
    _write(source_input, "sealed input\n")
    _write(source_code, "# sealed E2f code\n")
    receipt = {
        "schema_version": 1,
        "experiment": "e2f_v3_full_label_residual_adapter",
        "status": "PASS",
        "problems": [],
        "append_only": {
            "output_root": str(root.parent),
            "root_was_absent_at_start": True,
            "v1_v2_modified": False,
        },
        "generating_code": integration.identity(source_code),
        "inputs": {"sealed_input": integration.identity(source_input)},
        "outputs": {
            "results": integration.identity(results_path),
            "oof_predictions": integration.identity(oof_path),
            "fits": integration.identity(fits_path),
            "support_draws": integration.identity(support_draws),
            "solver_warnings": integration.identity(warnings),
        },
        "census": artifact_census,
        "solver": solver,
        "fixed_gate_primary": primary["fixed_gate_adapted"],
        "outer_fold_sensitivity_summary": results[
            "outer_fold_sensitivity_summary"
        ],
    }
    receipt_path = root / "receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "receipt": receipt_path,
        "results": results_path,
        "oof": oof_path,
        "fits": fits_path,
    }


def _write_handoff(reader: Path) -> None:
    manifest = reader / "HANDOFF_MANIFEST.sha256"
    files = sorted(
        path for path in reader.rglob("*") if path.is_file() and path != manifest
    )
    _write(
        manifest,
        "".join(
            f"{integration.sha256_file(path)}  {path.relative_to(reader).as_posix()}\n"
            for path in files
        ),
    )


def _build_reviews(root: Path, external: Path) -> dict[str, Path]:
    reader = root / "FOR_PATHOLOGIST"
    images = reader / "images"
    keys = root / "KEYS_DO_NOT_DISTRIBUTE"
    images.mkdir(parents=True)
    keys.mkdir(parents=True)
    case_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []
    p17_groups = ["absent", "absent", "positive_low", "positive_low", "positive_high", "positive_high"]
    for index in range(TEST_CONTRACT.review_cases):
        case_id = f"Q{index + 1:05d}"
        patient_id = f"SECRET_PATIENT_{index + 1:03d}"
        slide_id = f"SECRET_SLIDE_{index + 1:03d}"
        source = external / f"source_{index + 1:03d}.svs"
        _write(source, f"synthetic source {index}\n")
        source_identity = integration.identity(source)
        source_records.append(
            {
                **source_identity,
                "openslide_quickhash1": f"quickhash-{index + 1}",
            }
        )
        case_rows.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "slide_id": slide_id,
                "cohort": "RIH" if index < 3 else "SurGen",
                "kras": "mutant" if index % 2 == 0 else "wild_type",
                "p17_group": p17_groups[index],
                "p17_abundance": 0.0 if p17_groups[index] == "absent" else index + 0.5,
                "p28_abundance": index / 10,
            }
        )
        source_rows.append(
            {
                "case_id": case_id,
                "patient_id": patient_id,
                "slide_id": slide_id,
                "source_path": str(source.resolve()),
                "source_size_bytes": source.stat().st_size,
                "source_sha256": integration.sha256_file(source),
                "openslide_quickhash1": f"quickhash-{index + 1}",
                "level0_width_px": 100,
                "level0_height_px": 50,
            }
        )
        image_specs = [("overview", None), ("panel", 1), ("panel", 2)]
        for image_index, (_role, panel_number) in enumerate(image_specs):
            filename = (
                f"{case_id}_overview.jpg"
                if panel_number is None
                else f"{case_id}_panel{panel_number}.jpg"
            )
            path = images / filename
            color_seed = index * 40 + image_index * 11
            image = Image.new(
                "RGB",
                (8, 8),
                (
                    color_seed % 256,
                    (color_seed * 3 + 17) % 256,
                    (color_seed * 7 + 31) % 256,
                ),
            )
            image.putpixel((image_index, index % 8), (255 - index, image_index, index))
            image.save(path, format="JPEG", quality=91)
            image.close()
            digest = integration.sha256_file(path)
            image_rows.append(
                {
                    "file": filename,
                    "case_id": case_id,
                    "image_role": (
                        "numbered_overview"
                        if panel_number is None
                        else "whole_section_panel"
                    ),
                    "panel_number": "" if panel_number is None else panel_number,
                    "width_px": 8,
                    "height_px": 8,
                    "mode": "RGB",
                    "size_bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
            if panel_number is not None:
                x0 = 0 if panel_number == 1 else 50
                x1 = 50 if panel_number == 1 else 100
                panel_rows.append(
                    {
                        "case_id": case_id,
                        "panel_number": panel_number,
                        "image_file": filename,
                        "level0_x0": x0,
                        "level0_y0": 0,
                        "level0_x1": x1,
                        "level0_y1": 50,
                        "actual_output_mpp_x": 2.0,
                        "actual_output_mpp_y": 2.0,
                        "size_bytes": path.stat().st_size,
                        "sha256": digest,
                    }
                )

    _write_csv(
        reader / "scoring_form.csv",
        list(integration.EXPECTED_FORM_COLUMNS),
        [
            {
                column: row["case_id"] if column == "case_id" else ""
                for column in integration.EXPECTED_FORM_COLUMNS
            }
            for row in case_rows
        ],
    )
    _write_csv(
        reader / "reviewer_info.csv",
        list(integration.EXPECTED_REVIEWER_COLUMNS),
        [{column: "" for column in integration.EXPECTED_REVIEWER_COLUMNS}],
    )
    _write(reader / "INSTRUCTIONS.md", "# Blinded read\nReview every opaque case.\n")
    shutil.copyfile(reader / "scoring_form.csv", keys / "scoring_form_TEMPLATE.csv")
    shutil.copyfile(reader / "reviewer_info.csv", keys / "reviewer_info_TEMPLATE.csv")
    _write(root / "ANALYSIS_PLAN.md", "# Frozen analysis plan\nDo not execute before return.\n")
    _write(root / "README_COORDINATOR.md", "# Coordinator\nGenerated but unread.\n")
    _write(keys / "README.md", "# Restricted key\n")
    _write_csv(keys / "case_key.csv", list(case_rows[0]), case_rows)
    _write_csv(
        keys / "selection_audit.csv",
        ["group", "selected"],
        [{"group": "synthetic", "selected": TEST_CONTRACT.review_cases}],
    )
    _write_csv(
        keys / "excluded_previously_exposed_patients.csv",
        ["patient_id"],
        [{"patient_id": f"PRIOR_{index}"} for index in range(3)],
    )
    _write_csv(keys / "image_manifest.csv", list(image_rows[0]), image_rows)
    _write_csv(keys / "panel_manifest.csv", list(panel_rows[0]), panel_rows)
    _write_csv(keys / "source_identity_manifest.csv", list(source_rows[0]), source_rows)
    _write_handoff(reader)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "problems": [],
        "cases": TEST_CONTRACT.review_cases,
        "images": TEST_CONTRACT.review_cases
        * (TEST_CONTRACT.review_panels_per_case + 1),
        "overviews": TEST_CONTRACT.review_cases,
        "panels": TEST_CONTRACT.review_cases
        * TEST_CONTRACT.review_panels_per_case,
        "images_per_case": TEST_CONTRACT.review_panels_per_case + 1,
        "target_mpp": 2.0,
        "prior_exposed_union": TEST_CONTRACT.review_prior_exposed,
        "selected_prior_exposure_overlap": 0,
        "by_cohort": {"RIH": 3, "SurGen": 3},
        "by_kras": dict(TEST_CONTRACT.review_kras_counts),
        "by_p17_group": dict(TEST_CONTRACT.review_p17_counts),
        "panel_output_pixels": 12 * 64,
        "reader_payload_size_bytes": sum(
            path.stat().st_size for path in reader.rglob("*") if path.is_file()
        ),
        "scientific_status": "GENERATED_UNREAD",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
    }
    summary_path = root / "build_summary.json"
    _write_json(summary_path, summary)
    builder = external / "build_reviews_v5.py"
    analyzer = external / "analyze_reviews_v5.py"
    frozen_input = external / "frozen_profiles.parquet"
    _write(builder, "# frozen builder\n")
    _write(analyzer, "# frozen analyzer\n")
    _write(frozen_input, "synthetic input\n")
    output_files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "packet_receipt.json"
    )
    manifest_paths = [root / relative for relative in integration.REVIEW_MANIFEST_PATHS]
    receipt = {
        "schema_version": 2,
        "status": "PASS",
        "problems": [],
        "scientific_status": "GENERATED_UNREAD",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
        "builder": integration.identity(builder),
        "frozen_analyzer": integration.identity(analyzer),
        "frozen_inputs": [integration.identity(frozen_input)],
        "source_wsi_inputs": source_records,
        "outputs": [integration.identity(path) for path in output_files],
        "manifests": [integration.identity(path) for path in manifest_paths],
        "census": {
            "cases": TEST_CONTRACT.review_cases,
            "images": TEST_CONTRACT.review_cases
            * (TEST_CONTRACT.review_panels_per_case + 1),
            "overviews": TEST_CONTRACT.review_cases,
            "panels": TEST_CONTRACT.review_cases
            * TEST_CONTRACT.review_panels_per_case,
            "exact_images_per_case": TEST_CONTRACT.review_panels_per_case + 1,
        },
        "panel_design": (
            "two disjoint level-0 rectangles exactly partition the full main-image canvas"
        ),
        "blinding": (
            "reader-only folder contains opaque random case IDs, fresh RGB JPEGs, "
            "blank forms, and no raw WSI/path/key"
        ),
        "write_policy": "staged atomic append-only build",
    }
    receipt_path = keys / "packet_receipt.json"
    _write_json(receipt_path, receipt)
    return {
        "receipt": receipt_path,
        "summary": summary_path,
        "scoring": reader / "scoring_form.csv",
        "reviewer": reader / "reviewer_info.csv",
        "instructions": reader / "INSTRUCTIONS.md",
        "score_template": keys / "scoring_form_TEMPLATE.csv",
        "reviewer_template": keys / "reviewer_info_TEMPLATE.csv",
    }


def _refresh_e2f_output(paths: integration.IntegrationPaths, name: str) -> None:
    receipt = _load_json(paths.e2f_receipt)
    outputs = receipt["outputs"]
    assert isinstance(outputs, dict)
    output_paths = {
        "results": paths.e2f_results,
        "oof_predictions": paths.e2f_oof,
        "fits": paths.e2f_fits,
        "support_draws": paths.e2f_analysis / "support_draws.jsonl",
        "solver_warnings": paths.e2f_analysis / "solver_warnings.jsonl",
    }
    outputs[name] = integration.identity(output_paths[name])
    _write_json(paths.e2f_receipt, receipt)


def _refresh_review_receipt(paths: integration.IntegrationPaths) -> None:
    receipt = _load_json(paths.review_receipt)
    outputs = sorted(
        path
        for path in paths.reviews_v5.rglob("*")
        if path.is_file() and path.resolve() != paths.review_receipt.resolve()
    )
    receipt["outputs"] = [integration.identity(path) for path in outputs]
    receipt["manifests"] = [
        integration.identity(paths.reviews_v5 / relative)
        for relative in integration.REVIEW_MANIFEST_PATHS
    ]
    _write_json(paths.review_receipt, receipt)


def _fixture(tmp_path: Path) -> tuple[integration.IntegrationPaths, dict[str, Path]]:
    e2f_root = tmp_path / "components" / "e2f_v3" / "analysis"
    review_root = tmp_path / "reviews" / "v5"
    external = tmp_path / "external"
    e2f = _build_e2f(e2f_root)
    review = _build_reviews(review_root, external)
    integration_code = tmp_path / "code" / "final_v7_integration.py"
    integration_test = tmp_path / "code" / "test_final_v7_integration.py"
    _write(integration_code, "# frozen integration code\n")
    _write(integration_test, "# frozen integration tests\n")
    paths = integration.IntegrationPaths(
        e2f_analysis=e2f_root,
        reviews_v5=review_root,
        output_root=tmp_path / "reports" / "reruns" / "final_v7" / "integration",
        integration_code=integration_code,
        integration_test=integration_test,
        contract=TEST_CONTRACT,
    )
    return paths, {**e2f, **review}


def test_production_contract_freezes_exact_study_census() -> None:
    contract = integration.PRODUCTION_CONTRACT

    assert {name: (item.n, item.n_mutant) for name, item in contract.cohorts.items()} == {
        "RIH": (85, 37),
        "SurGen": (74, 30),
    }
    assert contract.patient_total == 159
    assert contract.expected_oof_full == 795
    assert contract.expected_oof_support == 12_720
    assert contract.expected_full_adapter_fits == 150
    assert contract.expected_full_platt_fits == 50
    assert contract.expected_support_adapter_fits == 2_400
    assert contract.expected_fit_total == 2_600
    assert contract.bootstrap_draws == 10_000
    assert contract.outer_seeds == (
        20260817,
        20260818,
        20260819,
        20260820,
        20260821,
    )
    assert contract.primary_outer_seed == 20260821
    assert contract.support_sizes == (8, 16, 32, 48)
    assert contract.support_draws == 20
    assert contract.lambda_grid[-6:] == ("0.3", "0.1", "0.03", "0.01", "0.003", "0.001")
    assert (contract.review_cases, contract.review_panels_per_case) == (60, 6)


def test_read_only_integration_verifies_and_keeps_scientific_states_distinct(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)

    product = integration.build_integration(paths)

    assert product.results["status"] == "PASS"
    assert product.results["component_states"]["e2f_v3"] == {
        "integrity_status": "PASS",
        "scientific_status": "EXECUTED",
    }
    review = product.results["reviews_v5"]
    assert review["scientific_status"] == "GENERATED_UNREAD"
    assert review["scientific_result"] is None
    assert "contributes no reader score" in product.results["claim_boundaries"][
        "reviews_v5"
    ]
    assert product.results["e2f_v3"]["primary_metrics"]["fixed_gate_adapted"][
        "pass"
    ] is True
    assert product.results["e2f_v3"]["primary_metrics"][
        "incremental_improvement_established"
    ]["pass"] is False

    report_contract = product.results["report_contract"]
    assert "created_utc" not in report_contract
    assert report_contract["component_states"] == product.results["component_states"]
    assert report_contract["e2f_v3"]["primary_metrics"] == product.results[
        "e2f_v3"
    ]["primary_metrics"]
    primary_metrics = report_contract["e2f_v3"]["primary_metrics"]
    assert set(primary_metrics) == {
        "primary_outer_seed",
        "contrast_sign_convention",
        "per_cohort",
        "macro",
        "fixed_gate_adapted",
        "incremental_improvement_established",
        "bootstrap_draws_requested",
        "bootstrap_draws_valid",
        "bootstrap_seed",
    }
    assert primary_metrics["primary_outer_seed"] == TEST_CONTRACT.primary_outer_seed
    assert primary_metrics["contrast_sign_convention"] == "procedure minus native"
    assert set(primary_metrics["per_cohort"]) == {"RIH", "SurGen"}
    for cohort_metrics in primary_metrics["per_cohort"].values():
        assert set(cohort_metrics) == {
            "n",
            "n_mutant",
            "native",
            "adapted",
            "platt",
            "adapted_minus_native",
            "platt_minus_native",
        }
    assert set(primary_metrics["macro"]) == {"procedures", "contrasts"}
    assert primary_metrics["bootstrap_draws_requested"] == TEST_CONTRACT.bootstrap_draws
    assert primary_metrics["bootstrap_draws_valid"] == TEST_CONTRACT.bootstrap_draws
    assert primary_metrics["bootstrap_seed"] == TEST_CONTRACT.bootstrap_seed
    assert report_contract["e2f_v3"]["outer_fold_layouts"] == product.results[
        "e2f_v3"
    ]["outer_fold_layouts"]
    assert report_contract["e2f_v3"]["outer_fold_sensitivity_summary"] == {
        "n_layouts": 2,
        "n_gate_pass": 2,
        "macro_adapted_auroc_range": [1.0, 1.0],
        "macro_delta_auroc_range": [0.0, 0.0],
    }
    assert report_contract["e2f_v3"]["label_efficiency_curve"] == product.results[
        "e2f_v3"
    ]["label_efficiency_curve"]
    layouts = report_contract["e2f_v3"]["outer_fold_layouts"]
    assert list(layouts) == ["101", "102"]
    for seed, layout in layouts.items():
        assert layout["outer_seed"] == int(seed)
        assert layout["is_primary"] is (int(seed) == TEST_CONTRACT.primary_outer_seed)
        assert layout["bootstrap_draws_requested"] == TEST_CONTRACT.bootstrap_draws
        assert layout["bootstrap_draws_valid"] == TEST_CONTRACT.bootstrap_draws
        assert layout["fixed_gate_adapted"]["pass"] is True
        assert layout["incremental_improvement_established"]["pass"] is False
    assert report_contract["reviews_v5"] == {
        "integrity_status": "PASS",
        "scientific_status": "GENERATED_UNREAD",
        "analysis_executed": False,
        "unblinding_performed": False,
        "analysis_result": None,
        "census": {
            "cases": 6,
            "images": 18,
            "overviews": 6,
            "panels": 12,
            "exact_images_per_case": 3,
        },
        "sampling_strata": {
            "by_cohort": {"RIH": 3, "SurGen": 3},
            "by_kras": {"mutant": 3, "wild_type": 3},
            "by_p17_group": {
                "absent": 2,
                "positive_high": 2,
                "positive_low": 2,
            },
            "prior_exposed_union": 3,
            "selected_prior_exposure_overlap": 0,
        },
    }
    assert integration.json_bytes(report_contract) == integration.json_bytes(
        integration.build_integration(paths).results["report_contract"]
    )
    assert not paths.output_root.exists()


def test_every_layout_requires_all_requested_bootstrap_draws(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    results = _load_json(material["results"])
    layouts = results["outer_fold_sensitivity"]
    assert isinstance(layouts, dict)
    nonprimary = layouts["101"]
    assert isinstance(nonprimary, dict)
    nonprimary["bootstrap_draws_valid"] = TEST_CONTRACT.bootstrap_draws - 1
    _write_json(material["results"], results)
    _refresh_e2f_output(paths, "results")

    with pytest.raises(
        integration.IntegrationError, match="valid bootstrap draw count mismatch"
    ):
        integration.build_integration(paths)


def test_review_cohort_strata_are_recomputed_from_the_case_key(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    summary = _load_json(material["summary"])
    summary["by_cohort"] = {"RIH": 2, "SurGen": 4}
    _write_json(material["summary"], summary)
    _refresh_review_receipt(paths)

    with pytest.raises(
        integration.IntegrationError,
        match="cohort census differs between build summary and case key",
    ):
        integration.build_integration(paths)


def test_e2f_declared_artifact_tampering_fails_before_interpretation(
    tmp_path: Path,
) -> None:
    paths, material = _fixture(tmp_path)
    material["oof"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(integration.IntegrationError, match="SHA-256 mismatch|size mismatch"):
        integration.build_integration(paths)


def test_oof_census_is_independently_checked_after_receipt_refresh(
    tmp_path: Path,
) -> None:
    paths, material = _fixture(tmp_path)
    lines = material["oof"].read_text(encoding="utf-8").splitlines()
    material["oof"].write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")
    _refresh_e2f_output(paths, "oof_predictions")

    with pytest.raises(integration.IntegrationError, match="OOF row census mismatch"):
        integration.build_integration(paths)


@pytest.mark.parametrize("failure", ["leakage", "support_realization", "lower_boundary"])
def test_fit_leakage_support_and_grid_boundary_are_fail_closed(
    tmp_path: Path,
    failure: str,
) -> None:
    paths, material = _fixture(tmp_path)
    rows = [json.loads(line) for line in material["fits"].read_text().splitlines()]
    if failure == "leakage":
        target = next(row for row in rows if row["phase"] == "support_curve")
        target["fit_patient_ids"][0] = target["test_patient_ids"][0]
        message = "fit/test leakage"
    elif failure == "support_realization":
        target = next(row for row in rows if row["phase"] == "support_curve")
        target["support_realized"] += 1
        message = "exact support realization"
    else:
        target = next(
            row
            for row in rows
            if row["phase"] == "full_label" and row["model_kind"] == "residual_adapter"
        )
        target["selected_lambda"] = TEST_CONTRACT.lambda_grid[-1]
        target["inner_selection"]["selected_lambda"] = TEST_CONTRACT.lambda_grid[-1]
        message = "lower boundary"
    _write(
        material["fits"],
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    _refresh_e2f_output(paths, "fits")

    with pytest.raises(integration.IntegrationError, match=message):
        integration.build_integration(paths)


@pytest.mark.parametrize("failure", ["fixed_gate", "incremental", "e2b"])
def test_gate_incremental_and_e2b_verdicts_are_mechanical(
    tmp_path: Path,
    failure: str,
) -> None:
    paths, material = _fixture(tmp_path)
    results = _load_json(material["results"])
    if failure == "e2b":
        checks = results["frozen_baseline_checks"]
        assert isinstance(checks, dict) and isinstance(checks["RIH"], dict)
        checks["RIH"]["expected"] = 0.5
        message = "frozen E2b expected"
    else:
        layouts = results["outer_fold_sensitivity"]
        assert isinstance(layouts, dict)
        primary = layouts[str(TEST_CONTRACT.primary_outer_seed)]
        assert isinstance(primary, dict)
        if failure == "fixed_gate":
            gate = primary["fixed_gate_adapted"]
            assert isinstance(gate, dict)
            gate["pass"] = False
            message = "fixed gate is not mechanically derived"
        else:
            verdict = primary["incremental_improvement_established"]
            assert isinstance(verdict, dict)
            verdict["pass"] = True
            message = "incremental verdict is not mechanically derived"
        results["primary"] = primary
    _write_json(material["results"], results)
    _refresh_e2f_output(paths, "results")

    with pytest.raises(integration.IntegrationError, match=message):
        integration.build_integration(paths)


def test_reviews_v5_generated_unread_state_is_exact(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    receipt = _load_json(paths.review_receipt)
    receipt["scientific_status"] = "EXECUTED"
    _write_json(paths.review_receipt, receipt)

    with pytest.raises(integration.IntegrationError, match="GENERATED_UNREAD"):
        integration.build_integration(paths)


def test_explicitly_absent_openslide_quickhash_is_allowed_when_identities_match(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    manifest = (
        paths.reviews_v5
        / "KEYS_DO_NOT_DISTRIBUTE"
        / "source_identity_manifest.csv"
    )
    header, rows = integration.read_csv_rows(manifest)
    source_path = Path(rows[0]["source_path"]).resolve()
    rows[0]["openslide_quickhash1"] = ""
    _write_csv(manifest, header, rows)

    receipt = _load_json(paths.review_receipt)
    source_records = receipt["source_wsi_inputs"]
    assert isinstance(source_records, list)
    matching = [
        record
        for record in source_records
        if Path(str(record["path"])).resolve() == source_path
    ]
    assert len(matching) == 1
    matching[0]["openslide_quickhash1"] = ""
    _write_json(paths.review_receipt, receipt)
    _refresh_review_receipt(paths)

    assert integration.build_integration(paths).results["status"] == "PASS"


def test_openslide_quickhash_manifest_receipt_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    receipt = _load_json(paths.review_receipt)
    source_records = receipt["source_wsi_inputs"]
    assert isinstance(source_records, list)
    source_records[0]["openslide_quickhash1"] = "different-quickhash"
    _write_json(paths.review_receipt, receipt)

    with pytest.raises(
        integration.IntegrationError,
        match="OpenSlide quickhash differs from manifest",
    ):
        integration.build_integration(paths)


def test_nonblank_reader_form_is_rejected_even_when_hashes_are_refreshed(
    tmp_path: Path,
) -> None:
    paths, material = _fixture(tmp_path)
    header, rows = integration.read_csv_rows(paths.scoring_form)
    rows[0]["assessable"] = "yes"
    _write_csv(paths.scoring_form, header, rows)
    shutil.copyfile(paths.scoring_form, material["score_template"])
    _write_handoff(paths.reviews_v5 / "FOR_PATHOLOGIST")
    _refresh_review_receipt(paths)

    with pytest.raises(integration.IntegrationError, match="scoring form is not blank"):
        integration.build_integration(paths)


def test_reader_payload_source_identity_leak_is_rejected_with_valid_hashes(
    tmp_path: Path,
) -> None:
    paths, material = _fixture(tmp_path)
    case_header, cases = integration.read_csv_rows(
        paths.reviews_v5 / "KEYS_DO_NOT_DISTRIBUTE" / "case_key.csv"
    )
    assert case_header and cases
    _write(
        material["instructions"],
        f"# Instructions\nForbidden source identity: {cases[0]['patient_id']}\n",
    )
    _write_handoff(paths.reviews_v5 / "FOR_PATHOLOGIST")
    _refresh_review_receipt(paths)

    with pytest.raises(integration.IntegrationError, match="leaks source identity/path"):
        integration.build_integration(paths)


def test_analysis_result_file_is_never_treated_as_generated_unread(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)
    _write_json(paths.reviews_v5 / "results.json", {"tau": 0.9})
    _refresh_review_receipt(paths)

    with pytest.raises(integration.IntegrationError, match="contains a scientific result"):
        integration.build_integration(paths)


def test_atomic_seal_matches_frozen_bundle_verifier_and_is_write_once(
    tmp_path: Path,
) -> None:
    paths, _ = _fixture(tmp_path)

    output = integration.seal_integration(paths)

    assert output == paths.output_root.resolve()
    assert sorted(path.name for path in output.iterdir()) == [
        "receipt.json",
        "results.json",
        "verification.json",
    ]
    receipt = _load_json(output / "receipt.json")
    assert receipt["append_only"] is True
    assert set(receipt["components"]) == {"e2f_v3", "reviews_v5"}
    assert receipt["components"]["reviews_v5"]["analysis_result"] is None

    verifier_paths = bundle_verifier.BundlePaths(
        final_v6=tmp_path / "reports" / "final_v6",
        snapshot_root=tmp_path / "snapshot",
        snapshot_receipt=tmp_path / "snapshot_receipt.json",
        final_v7=tmp_path / "reports" / "final_v7",
        integration_receipt=output / "receipt.json",
        reviews_v5=paths.reviews_v5,
        destination=tmp_path / "reports" / "final_v7" / "report_bundle_receipt.json",
    )
    verified = bundle_verifier.verify_integration_component(verifier_paths)
    assert verified["status"] == "PASS"
    assert verified["scientific_states"]["reviews_v5"][
        "scientific_status"
    ] == "GENERATED_UNREAD"
    with pytest.raises(FileExistsError, match="already exists"):
        integration.seal_integration(paths)
