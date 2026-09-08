from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_loco_transport  # noqa: E402
import aim2_surgen_subcohort_gap  # noqa: E402
import aim2_rih_acquisition_regime  # noqa: E402
from oceanpath.aim1 import lineage  # noqa: E402
from tools import aim2_exploratory_refresh as refresh  # noqa: E402


def test_new_output_root_must_be_explicit_absent_and_separate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(refresh.RefreshError, match="already exists"):
        refresh._new_output_root(existing, source)
    with pytest.raises(refresh.RefreshError, match="inside input"):
        refresh._new_output_root(source / "child", source)
    with pytest.raises(refresh.RefreshError, match="explicit absolute"):
        refresh._new_output_root(Path("relative"), source)

    assert refresh._new_output_root(tmp_path / "new-lineage", source) == (
        tmp_path / "new-lineage"
    )


def test_publish_once_copies_every_byte_and_refuses_a_second_writer(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "nested").mkdir()
    (stage / "a.json").write_text('{"a": 1}\n', encoding="utf-8")
    (stage / "nested" / "b.bin").write_bytes(b"\x00\x01\x02")
    output = tmp_path / "published"

    refresh._publish_once(stage, output)

    assert (output / "a.json").read_bytes() == (stage / "a.json").read_bytes()
    assert (output / "nested" / "b.bin").read_bytes() == b"\x00\x01\x02"
    with pytest.raises(refresh.RefreshError, match="output exists"):
        refresh._publish_once(stage, output)


def test_explicit_context_separates_frozen_reads_staged_writes_and_final_identities(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "frozen-v4"
    (input_root / "e2a").mkdir(parents=True)
    stage = tmp_path / "stage"
    stage.mkdir()
    final = tmp_path / "new-lineage"
    protocol = stage / "protocol.md"
    archive = stage / "archive.md"
    artifact = stage / "eval" / "result.json"
    artifact.parent.mkdir()
    protocol.write_text("protocol", encoding="utf-8")
    archive.write_text("archive", encoding="utf-8")
    artifact.write_text("{}\n", encoding="utf-8")
    old_environment = os.environ.get(lineage.AIM2_LINEAGE_ENV)
    original_e2a_root = aim2_loco_transport.e2a_root
    original_aim2_root = lineage.aim2_root
    original_protocol = aim2_rih_acquisition_regime.PROTOCOL_PATH

    with refresh._explicit_analysis_context(
        input_root=input_root,
        stage_root=stage,
        final_root=final,
        source_lineage="frozen-v4",
        output_lineage="new-lineage",
        protocol_snapshot=protocol,
        archive_snapshot=archive,
    ):
        assert aim2_loco_transport.e2a_root() == input_root / "e2a"
        assert lineage.aim2_root() == stage
        assert lineage.lineage_name() == "new-lineage"
        assert protocol == aim2_rih_acquisition_regime.PROTOCOL_PATH
        assert lineage.artifact_identity(artifact)["path"] == str(
            final / "eval" / "result.json"
        )

    assert aim2_loco_transport.e2a_root is original_e2a_root
    assert lineage.aim2_root is original_aim2_root
    assert original_protocol == aim2_rih_acquisition_regime.PROTOCOL_PATH
    assert os.environ.get(lineage.AIM2_LINEAGE_ENV) == old_environment


def _overlap_block(n_bootstrap: int, *, adequate: bool) -> dict:
    observed = {
        "control_fraction_clipped": 0.01 if adequate else 0.50,
        "control_fraction_outside_empirical_common_support": 0.02,
        "control_ess_fraction": 0.80,
        "control_max_normalized_weight": 2.0,
    }
    thresholds = {
        "max_control_fraction_clipped": aim2_surgen_subcohort_gap.OVERLAP_MAX_CONTROL_CLIPPED,
        "max_control_fraction_outside_empirical_common_support": (
            aim2_surgen_subcohort_gap.OVERLAP_MAX_CONTROL_OUTSIDE_COMMON_SUPPORT
        ),
        "min_control_ess_fraction": aim2_surgen_subcohort_gap.OVERLAP_MIN_CONTROL_ESS_FRACTION,
        "max_control_normalized_weight": aim2_surgen_subcohort_gap.OVERLAP_MAX_CONTROL_WEIGHT,
    }
    checks = {
        "clipping": adequate,
        "common_support": True,
        "effective_sample_size": True,
        "maximum_weight": True,
    }
    return {
        "n_bootstrap": n_bootstrap,
        "auroc_SR386": 0.70,
        "auroc_SR1482_reweighted": 0.62,
        "delta": 0.08,
        "delta_ci": [-0.01, 0.17],
        "overlap_assessment": {
            "observed": observed,
            "thresholds": thresholds,
            "checks": checks,
            "estimable_for_inference": adequate,
            "failed_checks": [] if adequate else ["clipping"],
        },
    }


def test_e2d4_refresh_verifier_recomputes_every_overlap_gate() -> None:
    report = {"schema_version": 2, "cap": 8192}
    for index, (name, _covariates) in enumerate(aim2_surgen_subcohort_gap.standardization_specs()):
        report[f"ipw_{name}"] = _overlap_block(25, adequate=index < 2)

    result = refresh._verify_e2d4(report, 25)

    assert result == {
        "standardizations_checked": 5,
        "overlap_guardrails_verified": True,
    }
    report["ipw_age + site"]["overlap_assessment"]["checks"]["clipping"] = False
    with pytest.raises(refresh.RefreshError, match="overlap checks mismatch"):
        refresh._verify_e2d4(report, 25)


def test_e2d2_refresh_verifier_checks_multislide_byte_sum(tmp_path: Path) -> None:
    source_rows = []
    patient_rows = []
    for index in range(14):
        patient = f"p{index:02d}"
        sizes = [100 + index, 200 + index] if index == 0 else [100 + index]
        mpps = [0.25, 0.50] if index == 0 else [0.50]
        for slide, (size, mpp) in enumerate(zip(sizes, mpps, strict=True)):
            source_rows.append(
                {
                    "patient_uid": patient,
                    "specimen_role": "metastatic",
                    "slide_size_bytes": size,
                    "mpp": mpp,
                    "slide_id": f"{patient}-{slide}",
                }
            )
        patient_rows.append(
            {
                "patient": patient,
                "metadata_slide_count": len(sizes),
                "slide_size_bytes": float(sum(sizes)),
                "mpp": float(pd.Series(mpps).median()),
            }
        )
    source = tmp_path / "labels.csv"
    pd.DataFrame(source_rows).to_csv(source, index=False)
    report = {
        "schema_version": 2,
        "cap": 8192,
        "n": 14,
        "n_mut": 4,
        "metadata_aggregation": {"slide_size_bytes": "sum across metastatic slides"},
        "inputs": {"label_source": refresh._identity(source)},
        "patients": patient_rows,
        "ensemble_auroc": 0.075,
        "per_seed_auroc": {"42": 0.1, "43": 0.1, "44": 0.1},
    }

    assert refresh._verify_e2d2(report)["summed_slide_bytes_verified"] is True
    report["patients"][0]["slide_size_bytes"] = 100.0
    with pytest.raises(refresh.RefreshError, match="summed slide bytes mismatch"):
        refresh._verify_e2d2(report)


def test_source_lineage_requires_matching_root_and_receipt(tmp_path: Path) -> None:
    root = tmp_path / "frozen-v4"
    root.mkdir()
    (root / "lineage_start.json").write_text(
        json.dumps({"lineage": "frozen-v4"}), encoding="utf-8"
    )
    assert refresh._source_lineage(root) == "frozen-v4"

    (root / "lineage_start.json").write_text(
        json.dumps({"lineage": "different"}), encoding="utf-8"
    )
    with pytest.raises(refresh.RefreshError, match="root name"):
        refresh._source_lineage(root)
