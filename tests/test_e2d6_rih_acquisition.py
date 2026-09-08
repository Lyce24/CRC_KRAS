from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import aim2_rih_acquisition_regime


def _details(canonical_source: str, repair_state: str) -> str:
    return json.dumps(
        {
            "canonical_source": canonical_source,
            "repair_state": repair_state,
            "metadata_mpp": 0.5,
        }
    )


def _frozen_inventory_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for index in range(265):
        slide_id = f"APERIO-{index:03d}"
        rows.append(
            {
                "output_id": slide_id,
                "cohort": "RIH",
                "status": "ready",
                "source_status": "original_standard",
                "wsi": f"rih/{slide_id}.svs",
                "mpp": 0.5016,
                "details": _details("original", "not_selected"),
            }
        )
    repair_rows: list[dict[str, object]] = []
    for index in range(103):
        slide_id = f"REPAIRED-{index:03d}"
        rows.append(
            {
                "output_id": slide_id,
                "cohort": "RIH",
                "status": "ready",
                "source_status": "fixed_verified",
                "wsi": f"rih/{slide_id}.tiff",
                "mpp": 0.18901199443536876 if index < 98 else 0.3780239888707375,
                "details": _details("fixed", "done"),
            }
        )
        repair_rows.append(
            {"slide_id": slide_id, "file": f"{slide_id}.svs", "status": "done"}
        )
    rows.append(
        {
            "output_id": "SL-348",
            "cohort": "RIH",
            "status": "ready",
            "source_status": "original_standard",
            "wsi": "rih/SL-348.svs",
            "mpp": 0.13899,
            "details": _details("original", "not_selected"),
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(repair_rows)


def test_regime_mapping_reproduces_all_369_frozen_inventory_categories() -> None:
    inventory, repair = _frozen_inventory_fixture()

    mapping = aim2_rih_acquisition_regime.build_slide_regime_map(inventory, repair)

    assert len(mapping) == 369
    assert mapping["technical_regime"].value_counts().to_dict() == {
        aim2_rih_acquisition_regime.REGIME_APERIO: 265,
        aim2_rih_acquisition_regime.REGIME_REPAIRED: 103,
        aim2_rih_acquisition_regime.REGIME_VERSA: 1,
    }
    repaired_ids = set(mapping.loc[
        mapping["technical_regime"].eq(aim2_rih_acquisition_regime.REGIME_REPAIRED), "slide_id"
    ])
    assert repaired_ids == set(repair["slide_id"])
    assert mapping.set_index("slide_id").loc["SL-348", "technical_regime"] == (
        aim2_rih_acquisition_regime.REGIME_VERSA
    )


def test_regime_mapping_fails_closed_on_unmapped_or_duplicate_slide() -> None:
    inventory, repair = _frozen_inventory_fixture()
    inventory.loc[inventory["output_id"].eq("APERIO-000"), "mpp"] = 0.25

    with pytest.raises(ValueError, match="unmapped or multiply mapped"):
        aim2_rih_acquisition_regime.build_slide_regime_map(inventory, repair)

    inventory, repair = _frozen_inventory_fixture()
    inventory = pd.concat([inventory, inventory.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate slide_id"):
        aim2_rih_acquisition_regime.build_slide_regime_map(inventory, repair)


def test_regime_mapping_requires_exact_repair_ledger_membership() -> None:
    inventory, repair = _frozen_inventory_fixture()
    repair = repair.iloc[:-1].copy()

    with pytest.raises(ValueError, match="unmapped or multiply mapped"):
        aim2_rih_acquisition_regime.build_slide_regime_map(inventory, repair, enforce_expected=False)


def test_manifest_attachment_rejects_patient_spanning_regimes() -> None:
    mapping = pd.DataFrame(
        {
            "slide_id": ["a", "b"],
            "technical_regime": [aim2_rih_acquisition_regime.REGIME_APERIO, aim2_rih_acquisition_regime.REGIME_REPAIRED],
            "wsi": ["rih/a.svs", "rih/b.tiff"],
            "mpp": [0.5016, 0.18901199443536876],
            "source_status": ["original_standard", "fixed_verified"],
            "canonical_source": ["original", "fixed"],
            "repair_state": ["not_selected", "done"],
            "suffix": [".svs", ".tiff"],
        }
    )
    manifest = pd.DataFrame(
        {
            "slide_id": ["a", "b"],
            "patient_id": ["same", "same"],
            "target_label": [1, 1],
            "specimen_role": ["primary", "primary"],
        }
    )

    with pytest.raises(ValueError, match="patients span technical regimes"):
        aim2_rih_acquisition_regime.attach_regime_to_manifest(manifest, mapping, role="primary")


def _arm(
    prefix: str,
    *,
    separation: float,
    seed: int,
    n_per_class: int = 100,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = np.repeat([0, 1], n_per_class)
    score = separation * label + rng.normal(0, 0.8, len(label))
    return pd.DataFrame(
        {
            "patient_id": [f"{prefix}-{index}" for index in range(len(label))],
            "label": label,
            "mean_logit": score,
            "prob_raw": 1 / (1 + np.exp(-score)),
        }
    )


def test_four_arm_interaction_is_direct_deterministic_and_ordered() -> None:
    arms = {
        aim2_rih_acquisition_regime.REGIME_APERIO: (
            _arm("ap", separation=2.0, seed=1),
            _arm("am", separation=0.2, seed=2),
        ),
        aim2_rih_acquisition_regime.REGIME_REPAIRED: (
            _arm("rp", separation=0.3, seed=3),
            _arm("rm", separation=1.8, seed=4),
        ),
    }

    first = aim2_rih_acquisition_regime.regime_interaction_ci(arms, n_bootstrap=400, seed=17)
    second = aim2_rih_acquisition_regime.regime_interaction_ci(arms, n_bootstrap=400, seed=17)

    assert first == second
    expected = (
        first["per_regime_delta_auroc"][aim2_rih_acquisition_regime.REGIME_REPAIRED]
        - first["per_regime_delta_auroc"][aim2_rih_acquisition_regime.REGIME_APERIO]
    )
    assert first["interaction_delta_auroc"] == pytest.approx(expected)
    assert first["interaction_delta_auroc_ci"][0] > 0
    assert first["n_bootstrap"] == 400
    assert first["four_arms_pairwise_patient_disjoint"] is True
    assert "four disjoint" in first["bootstrap_method"]


def test_four_arm_interaction_rejects_any_patient_overlap() -> None:
    aperio_primary = _arm("ap", separation=1.0, seed=10, n_per_class=10)
    repaired_primary = _arm("rp", separation=1.0, seed=11, n_per_class=10)
    repaired_primary.loc[0, "patient_id"] = aperio_primary.loc[0, "patient_id"]
    arms = {
        aim2_rih_acquisition_regime.REGIME_APERIO: (
            aperio_primary,
            _arm("am", separation=0.8, seed=12, n_per_class=10),
        ),
        aim2_rih_acquisition_regime.REGIME_REPAIRED: (
            repaired_primary,
            _arm("rm", separation=0.8, seed=13, n_per_class=10),
        ),
    }

    with pytest.raises(ValueError, match="overlap on patients"):
        aim2_rih_acquisition_regime.regime_interaction_ci(arms, n_bootstrap=10)


def test_timestamp_is_aware_utc_and_python310_compatible() -> None:
    timestamp = aim2_rih_acquisition_regime._utc_now()

    assert timestamp.endswith("+00:00")
    assert "T" in timestamp


def test_report_publishes_nothing_when_interaction_analysis_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = []
    for role in ("primary", "metastatic"):
        for regime, prefix in (
            (aim2_rih_acquisition_regime.REGIME_APERIO, "a"),
            (aim2_rih_acquisition_regime.REGIME_REPAIRED, "r"),
        ):
            for label in (0, 1):
                rows.append(
                    {
                        "patient_id": f"{role}-{prefix}-{label}",
                        "role": role,
                        "label": label,
                        "mean_logit": float(label),
                        "prob_raw": 0.25 + 0.5 * label,
                        "technical_regime": regime,
                        "slide_count": 1,
                        "slide_ids": f"{role}-{prefix}-{label}-slide",
                        "native_mpps": "0.5",
                        "source_statuses": "test",
                        "mean_logit_seed42": float(label),
                        "mean_logit_seed43": float(label),
                        "mean_logit_seed44": float(label),
                    }
                )
    rows.append(
        {
            "patient_id": "versa-one",
            "role": "metastatic",
            "label": 1,
            "mean_logit": 0.1,
            "prob_raw": 0.52,
            "technical_regime": aim2_rih_acquisition_regime.REGIME_VERSA,
            "slide_count": 1,
            "slide_ids": "SL-348",
            "native_mpps": "0.13899",
            "source_statuses": "original_standard",
            "mean_logit_seed42": 0.1,
            "mean_logit_seed43": 0.1,
            "mean_logit_seed44": 0.1,
        }
    )
    table = pd.DataFrame(rows)
    tagged = {
        role: pd.DataFrame(
            {
                "technical_regime": table.loc[table["role"].eq(role), "technical_regime"]
            }
        )
        for role in ("primary", "metastatic")
    }
    mapping = pd.DataFrame(
        {
            "technical_regime": [
                aim2_rih_acquisition_regime.REGIME_APERIO,
                aim2_rih_acquisition_regime.REGIME_REPAIRED,
                aim2_rih_acquisition_regime.REGIME_VERSA,
            ],
            "mpp": [0.5016, 0.18901199443536876, 0.13899],
        }
    )
    writes: list[str] = []
    monkeypatch.setattr(aim2_rih_acquisition_regime, "report_path", lambda cap: tmp_path / "report.json")
    monkeypatch.setattr(aim2_rih_acquisition_regime, "patient_table_path", lambda cap: tmp_path / "table.parquet")
    monkeypatch.setattr(
        aim2_rih_acquisition_regime, "patient_table_receipt_path", lambda cap: tmp_path / "table.receipt.json"
    )
    monkeypatch.setattr(aim2_rih_acquisition_regime.lineage, "lineage_name", lambda: "test-lineage")
    monkeypatch.setattr(aim2_rih_acquisition_regime.pd, "read_csv", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(aim2_rih_acquisition_regime, "load_repair_state", lambda: pd.DataFrame())
    monkeypatch.setattr(aim2_rih_acquisition_regime, "build_slide_regime_map", lambda *args: mapping)
    monkeypatch.setattr(aim2_rih_acquisition_regime, "_assemble_patient_table", lambda *args: (table, tagged))
    monkeypatch.setattr(aim2_rih_acquisition_regime, "_validate_expected_population", lambda *args: {})
    monkeypatch.setattr(
        aim2_rih_acquisition_regime,
        "_metric_block",
        lambda frame, **kwargs: {"n": len(frame), "class_counts": {}},
    )
    monkeypatch.setattr(
        aim2_rih_acquisition_regime.aim2_metastatic_transport,
        "contrast_ci",
        lambda *args, **kwargs: {"delta_auroc": 0.0, "delta_auroc_ci": [-1.0, 1.0]},
    )
    monkeypatch.setattr(aim2_rih_acquisition_regime, "validate_disjoint_contrast_arms", lambda arms: None)
    monkeypatch.setattr(
        aim2_rih_acquisition_regime,
        "regime_interaction_ci",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bootstrap failed")),
    )
    monkeypatch.setattr(
        aim2_rih_acquisition_regime,
        "_input_artifacts",
        lambda cap: (_ for _ in ()).throw(AssertionError("reached publication inputs")),
    )
    monkeypatch.setattr(
        aim2_rih_acquisition_regime.lineage,
        "write_parquet_once",
        lambda *args, **kwargs: writes.append("parquet"),
    )
    monkeypatch.setattr(
        aim2_rih_acquisition_regime.lineage,
        "write_json_once",
        lambda *args, **kwargs: writes.append("json"),
    )

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        aim2_rih_acquisition_regime.cmd_report(Namespace(cap=8192, n_bootstrap=10, bootstrap_seed=1))

    assert writes == []
