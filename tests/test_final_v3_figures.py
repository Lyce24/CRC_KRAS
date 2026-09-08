from __future__ import annotations

import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v3_figures as figures


@pytest.fixture(scope="module")
def validated() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    data = figures.load_and_validate_data()
    return data, figures.source_frames(data)


def test_authoritative_receipt_bindings_are_exact_and_verified() -> None:
    assert figures.EXPECTED_BOUND_RECEIPT_SUFFIXES == {
        "aim1_worklist": "aim1_worklist/receipt.json",
        "aim2_operational": "aim2_operational_v3/receipt.json",
        "aim3_actionability": "aim3_actionability/receipt.json",
        "aim4_compressibility": "aim4_compressibility_v2/completion_receipt.json",
    }

    bundle = figures.verify_bound_sources()
    assert bundle["status"] == "PASS"
    assert bundle["independent_checks"]["rehash_mismatches"] == 0
    for component, suffix in figures.EXPECTED_BOUND_RECEIPT_SUFFIXES.items():
        record = bundle["bound_receipts"][component]
        bound_path = figures.verify_identity(record)
        assert bound_path.as_posix().endswith(suffix)


def test_identity_verification_fails_closed_after_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "source.csv"
    artifact.write_text("x,y\n1,2\n", encoding="utf-8")
    identity = figures.file_identity(artifact)

    assert figures.verify_identity(identity) == artifact
    artifact.write_text("x,y\n1,3\n", encoding="utf-8")

    with pytest.raises(figures.FigureBuildError, match="identity mismatch"):
        figures.verify_identity(identity)


def test_validated_source_frame_shapes_and_keys(
    validated: tuple[dict[str, Any], dict[str, pd.DataFrame]],
) -> None:
    _, frames = validated
    expected_shapes = {
        "challenge": (5, 7),
        "model_deltas": (3, 4),
        "external_primary": (8, 7),
        "metastatic": (3, 6),
        "sparse_adapter": (12, 6),
        "aim3_tasks": (10, 6),
        "aim3_gates": (5, 9),
        "aim3_repeated": (15, 5),
        "actionability": (8, 10),
        "worklist": (265, 29),
        "comparisons": (105, 17),
        "agreement": (120, 10),
        "census": (6, 9),
        "random_draws": (40_000, 7),
        "technical": (2, 8),
        "technical_patients": (144, 5),
        "aim4_prototypes": (32, 21),
        "aim4_stability_grid": (18, 22),
        "aim4_candidates": (8, 8),
        "compress": (168, 6),
        "predictions": (1486, 19),
    }
    for name, shape in expected_shapes.items():
        assert frames[name].shape == shape, name

    assert not frames["worklist"].duplicated(
        ["analysis", "population", "method", "capacity_nominal"]
    ).any()
    assert not frames["agreement"].duplicated(
        ["analysis_id", "KRAS_stratum", "metric"]
    ).any()
    assert not frames["compress"].duplicated(["model", "summary", "metric"]).any()
    assert set(frames["actionability"]["direct_clinical_decision_supported"]) == {
        "No"
    }


def test_central_worklist_and_multislide_values_are_bound(
    validated: tuple[dict[str, Any], dict[str, pd.DataFrame]],
) -> None:
    data, frames = validated
    worklist = frames["worklist"]
    headline = worklist[
        (worklist["analysis"] == "E0_frozen_OOF")
        & (worklist["population"] == "A_all_primary")
        & np.isclose(worklist["capacity_nominal"], 0.30)
    ].set_index("method")
    expected_capture = {
        "random_expected": 0.2994616419919246,
        "clinical_oof": 0.3162251655629139,
        "wsi_declared_seed_median": 0.445364238410596,
        "fusion_declared_seed_median": 0.4354304635761589,
    }
    for method, expected in expected_capture.items():
        assert headline.loc[method, "capture"] == pytest.approx(expected, abs=1e-12)

    assert data["aim2_results"]["primary_analysis_id"] == (
        "e2a_surgen_heldout__sr1482_primary"
    )
    primary = frames["agreement"]
    primary = primary[
        (primary["analysis_id"] == data["aim2_results"]["primary_analysis_id"])
        & (primary["KRAS_stratum"] == "all")
        & (primary["inferential_status"] == "inferential")
    ].set_index("metric")
    assert primary.loc["icc_1_1", "estimate"] == pytest.approx(
        0.6927623957203712, abs=1e-12
    )
    assert primary.loc["icc_1_k_eff", "estimate"] == pytest.approx(
        0.8184992736981962, abs=1e-12
    )
    assert primary.loc["median_abs_logit_difference", "estimate"] == pytest.approx(
        1.4091796875, abs=1e-12
    )
    assert primary["n_multislide_patients"].nunique() == 1
    assert primary["n_multislide_patients"].iloc[0] == 144

    census_totals = frames["census"][
        [
            "n_patients",
            "n_slides",
            "n_one_slide_patients",
            "n_two_slide_patients",
            "n_three_slide_patients",
            "n_multislide_patients",
        ]
    ].sum()
    assert census_totals.tolist() == [1486, 1642, 1331, 154, 1, 155]


def test_central_actionability_and_compressibility_values_are_bound(
    validated: tuple[dict[str, Any], dict[str, pd.DataFrame]],
) -> None:
    _, frames = validated
    ladder = frames["actionability"].set_index("molecular_level")
    assert ladder.loc["KRAS gene status, primary CRC", "fine_auroc"] == pytest.approx(
        0.679545659323332, abs=1e-12
    )
    assert ladder.loc["Codon 12 versus other KRAS", "fixed_verdict"] == "CEILING"
    assert ladder.loc["G12C versus other G12", "fixed_verdict"] == "UNDERPOWERED"
    assert ladder.loc[
        "G12C versus other G12", "repeated_control_consensus"
    ] == "NO_CEILING_CONSENSUS"
    assert pd.isna(ladder.loc["Extended RAS status", "fine_auroc"])
    assert pd.isna(ladder.loc["Metastatic exact allele", "fine_auroc"])

    macro_r2 = frames["compress"]
    macro_r2 = macro_r2[
        (macro_r2["summary"] == "equal_target_macro")
        & (macro_r2["metric"] == "r2")
    ].set_index("model")
    expected = {
        "clinical": -0.0119637722227027,
        "abundance_only": 0.0934886526299616,
        "abundance_plus_attention": 0.2994444624560695,
        "combined": 0.3003890581375155,
    }
    for model, point in expected.items():
        assert macro_r2.loc[model, "point"] == pytest.approx(point, abs=1e-12)
    assert macro_r2.loc["clinical", "point"] < 0

    cptac = frames["compress"]
    cptac = cptac[(cptac["summary"] == "CPTAC") & (cptac["metric"] == "r2")]
    assert (cptac["point"] < 0).all()


def test_public_figure_outputs_do_not_transfer_historical_montage_names() -> None:
    if not figures.DEFAULT_OUTPUT.is_dir():
        pytest.skip("figure package has not been published yet")

    public_text_suffixes = {".csv", ".json", ".md", ".svg", ".tex", ".txt"}
    public_files = sorted(
        path
        for path in figures.DEFAULT_OUTPUT.rglob("*")
        if path.is_file() and path.suffix.lower() in public_text_suffixes
    )
    assert public_files, "published figure package has no inspectable public text"
    forbidden = re.compile(r"\bM(?:04|05|07|11)\b", flags=re.IGNORECASE)
    violations: list[str] = []
    for path in public_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        if forbidden.search(text):
            violations.append(str(path.relative_to(figures.DEFAULT_OUTPUT)))
    assert not violations, f"historical montage names leaked into: {violations}"


def _published_receipt_verifier() -> Callable[[Path], Any] | None:
    for name in (
        "verify_figure_receipt",
        "verify_published_figure_receipt",
        "validate_published_receipt",
    ):
        helper = getattr(figures, name, None)
        if callable(helper):
            return helper
    return None


def test_published_figure_receipt_verifies_when_helper_is_available() -> None:
    helper = _published_receipt_verifier()
    if helper is None:
        pytest.skip("figure-receipt verification helper is not implemented yet")
    if not figures.DEFAULT_OUTPUT.is_dir():
        pytest.skip("figure package has not been published yet")

    receipts = sorted(figures.DEFAULT_OUTPUT.glob("*receipt*.json"))
    assert len(receipts) == 1, "published package must expose one top-level receipt"
    observed = helper(receipts[0])
    if isinstance(observed, dict) and "status" in observed:
        assert observed["status"] in {"PASS", "COMPLETE"}
