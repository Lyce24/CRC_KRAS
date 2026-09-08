#!/usr/bin/env python3
"""Independent source-only audit of the sealed v14 Module-II scoring bundle.

No refit, target path stat, target feature read, or target outcome read occurs.
The receipt supplements, and never changes, the earlier Module-I/report audit.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "reports/reruns/final_v14_additions_20260903"
SOURCE = RUN / "e4m2_source_frozen"
OUTPUT = RUN / "e4m2_source_audit/source_audit_receipt.json"
COLS = [f"prototype_{j:02d}" for j in range(32)]


def sha(path: Path) -> str:
    assert path.is_relative_to(REPO), f"Nonlocal input forbidden in source audit: {path}"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    assert identity(path) == {k: item[k] for k in ("path", "sha256", "size_bytes")}
    return path


def sealed(path: Path) -> dict:
    seal = json.loads(Path(str(path) + ".seal.json").read_text())
    assert sha(path) == seal.get("sha256", seal.get("artifact", {}).get("sha256"))
    return json.loads(path.read_text())


def same(actual, expected, tolerance: float = 1e-12) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    error = float(np.max(np.abs(a - b)))
    assert error <= tolerance, (error, tolerance)
    return error


def audit() -> dict:
    path = SOURCE / "source_scoring_bundle.json"
    bundle = sealed(path)
    assert bundle["status"] == "SOURCE_SCORING_BUNDLE_SEALED"
    assert bundle["name_gate_status"] == "NAME_GATE_FAIL"
    for item in bundle["source_inputs"] + bundle["code"] + [bundle["dry_run"]]:
        verify(item)
    assert hashlib.sha256(json.dumps(bundle["code"], sort_keys=True).encode()).hexdigest() == bundle["code_tree_sha256"]
    for item in bundle["code"]:
        snapshot = Path(item["path"])
        original = REPO / snapshot.relative_to(SOURCE / "frozen_code")
        assert sha(original) == item["sha256"]
    expected_entry = SOURCE / "frozen_code/tools/final_v14_module2.py"
    assert Path(bundle["target_entry_command"][1]) == expected_entry
    assert bundle["target_entry_command"][2:] == ["score-targets", "--output", str(SOURCE)]
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "packages": {p: importlib.metadata.version(p) for p in bundle["runtime"]["packages"]}}
    assert runtime == bundle["runtime"]

    dry = sealed(verify(bundle["dry_run"]))
    replay = sealed(verify(dry["serialized_snapshot_replay"]))
    request = sealed(SOURCE / "source_replay_request.json")
    models = sealed(verify(request["models"]))
    for item in replay["inputs"] + list(request.values()):
        verify(item)
    assert models == bundle["models"] and set(models) == {"ALL32"}
    assert set(request) == {"models", "source_profiles", "expected_logits"}
    assert sorted(replay["inputs"], key=lambda x: x["path"]) == sorted(request.values(), key=lambda x: x["path"])
    assert request["source_profiles"] == bundle["source_profiles"]
    assert request["expected_logits"] == dry["logits"]
    assert dry["status"] == "SOURCE_DRY_RUN_SEALED"
    assert dry["snapshot_replay_status"] == replay["status"] == "SERIALIZED_SOURCE_SNAPSHOT_REPLAY_PASS"
    assert dry["target_feature_paths_read"] == replay["target_feature_paths_read"] == 0
    assert dry["representations"]["ALL32"] == {
        "max_absolute_logit_error": 0.0, "patients": 1239, "status": "PASS", "tolerance": 1e-12}
    assert replay["representations"]["ALL32"] == {"max_absolute_error": 0.0, "tolerance": 1e-12}

    profiles = pd.read_parquet(verify(bundle["source_profiles"])).sort_values("patient_id").reset_index(drop=True)
    labels = pd.read_csv(verify(bundle["source_labels"]), usecols=["patient_id", "target_label"])
    assert labels.groupby("patient_id").target_label.nunique().max() == 1
    labels = labels.drop_duplicates("patient_id").set_index("patient_id")
    assert not profiles.patient_id.duplicated().any() and set(profiles.patient_id) == set(labels.index)
    y = labels.loc[profiles.patient_id, "target_label"].to_numpy(int)
    assert len(y) == 1239 and y.sum() == 501 and set(y) == {0, 1}
    X = profiles[COLS].to_numpy(np.float64)
    assert X.shape == (1239, 32) and np.isfinite(X).all() and (X >= 0).all()
    same(X.sum(axis=1), np.ones(len(X)))
    model = models["ALL32"]
    assert model["status"] == "ESTIMABLE" and model["coordinates"] == list(range(32))
    mean, scale = X.mean(axis=0), X.std(axis=0, ddof=0)
    constant = np.ptp(X, axis=0) == 0
    scale[constant] = 1
    errors = {"source_mean": same(model["scaler_mean"], mean),
              "source_scale": same(model["scaler_scale"], scale)}
    assert model["zero_variance_coordinates"] == np.flatnonzero(constant).tolist() == []
    beta = np.asarray(model["coef"])
    assert beta.shape == (32,) and np.isfinite(beta).all() and np.isfinite(model["intercept"])
    assert (beta[constant] == 0).all()
    logits = pd.read_csv(verify(dry["logits"]), float_precision="round_trip")
    assert list(logits) == ["patient_id", "logit_ALL32"]
    assert logits.patient_id.tolist() == profiles.patient_id.tolist()
    with threadpool_limits(limits=1):
        actual = np.einsum("ij,j->i", (X - mean) / scale, beta) + model["intercept"]
    errors["independent_serialized_source_logit_replay"] = same(actual, logits.logit_ALL32)

    summary_path = RUN / "e4m1_source/source_fit_summary.json"
    summary = sealed(summary_path)
    naming_path = RUN / "e4v_post_reader_xlsx/naming/naming_freeze.json"
    naming = sealed(naming_path)
    assert naming["name_gate_status"] == "NAME_GATE_FAIL"
    folds = summary["representations"]["ALL32"]["outer_folds"]
    assert set(folds) == {str(f) for f in range(5)}
    penalties = []
    for fold in range(5):
        entry = folds[str(fold)]
        checkpoint = sealed(verify(entry["checkpoint"]))
        concept = checkpoint["models"]["concept"]
        assert checkpoint["fold"] == fold and checkpoint["representation"] == "ALL32"
        assert entry["status"] == concept["status"] == "ESTIMABLE"
        assert entry["selected_penalty"] == concept["selected_penalty"]
        penalties.append(entry["selected_penalty"])
    assert model["selected_penalty"] == sorted(penalties)[2] == .01
    assert dt.datetime.fromisoformat(naming["created_utc"]) < dt.datetime.fromisoformat(summary["created_utc"]) < dt.datetime.fromisoformat(dry["created_utc"]) < dt.datetime.fromisoformat(bundle["created_utc"])

    # Verify only manifest metadata for unavailable targets; do not stat/open them.
    parent_path = REPO / "reports/final_v13/source_manifest.json"
    parent = json.loads(parent_path.read_text())
    for cohort, target_identity in bundle["target_rosters"].items():
        matches = [r for r in parent["artifacts"] if r.get("role") == f"label_blind_{cohort}_scoring_roster"]
        assert len(matches) == 1
        assert {k: matches[0][k] for k in ("path", "sha256", "size_bytes")} == target_identity
    assert len(bundle["target_rosters"]) == 5
    assert bundle["input_schema"]["coordinates"] == COLS
    assert not set(bundle["input_schema"]["roster_allowed_columns"]) & {"label", "target_label", "KRAS", "kras", "mutation"}
    assert bundle["output_schema"]["scores"] == ["cohort", "patient_id", "logit_ALL32"]
    assert bundle["output_schema"]["profiles"] == ["cohort", "patient_id", "n_slides", *COLS]
    assert not (SOURCE / "target_score_seal.json").exists()

    analysis_path = REPO / "tools/final_v14_module2_analysis.py"
    # The inherited full-score pin is checked textually, avoiding imported scorer execution.
    analysis_text = analysis_path.read_text()
    full = [r for r in parent["artifacts"] if r.get("id") == "aim1-tcga-surgen-two-encoder-downstream-v3-patient-native-logits"]
    assert len(full) == 1
    for key in ("path", "sha256", "size_bytes"):
        assert str(full[0][key]) in analysis_text
    return {"status": "PASS_SOURCE_FREEZE_TARGETS_PENDING", "component": "independent_module_ii_source_audit",
            "source_bundle": identity(path), "audit_code": identity(Path(__file__)),
            "source_dry_run": identity(Path(bundle["dry_run"]["path"])),
            "serialized_snapshot_replay": identity(Path(dry["serialized_snapshot_replay"]["path"])),
            "source_patients": 1239, "source_mutant": 501, "representations": ["ALL32"],
            "coordinates": 32, "verified_outer_fits": 5, "outer_selected_C": penalties,
            "full_source_selected_C": model["selected_penalty"], "numeric_max_absolute_errors": errors,
            "runtime_and_code_pins_verified": True, "frozen_entry_verified": True,
            "name_gate_status": "NAME_GATE_FAIL", "target_roster_metadata_pins_verified": 5,
            "target_paths_opened_by_audit": 0, "source_refits_performed_by_audit": 0,
            "target_scientific_status": "PENDING_NOT_TESTED",
            "analysis_code_review": {"implementation": identity(analysis_path), "status": "PASS_PRE_TARGET_STATIC_REVIEW",
                "checks": ["fixed full-score pin matches inherited parent manifest", "within-cohort KRAS patient pairing",
                           "independent controlling-cohort draws and fixed one-half macro", "hierarchical ALL32 then eligible NAMED_REF gates",
                           "untruncated ratios; nonpositive denominators undefined; 9500 finite-draw floor",
                           "all label-blind score/profile/OOD seals verified before outcome opening",
                           "RIH eight dual-role patients removed from both arms", "role bootstrap and within-KRAS role permutation",
                           "BH family of all32 coordinates per cohort family", "exact fixed-coefficient logit-contrast decomposition",
                           "performance and interaction bootstrap arrays retained"]},
            "limits": ["Target scorer reviewed statically; no target execution or target results are audited here.",
                       "Historical no-target-access claims are receipt/code evidence, not an independent operating-system access log.",
                       "Target packed-store resume authenticates inherited content hashes by frozen mtime and size guards; archive checks await restoration.",
                       "Existing Module-I report and source-audit artifacts remain unchanged."]}


def main() -> None:
    result = audit()
    if OUTPUT.exists():
        old = sealed(OUTPUT)
        assert {k: v for k, v in old.items() if k != "created_utc"} == result
    else:
        result["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        with Path(str(OUTPUT) + ".seal.json").open("x") as stream:
            json.dump(identity(OUTPUT), stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(OUTPUT), "sha256": sha(OUTPUT),
                      "numeric_max_absolute_errors": result["numeric_max_absolute_errors"]}, indent=2))


if __name__ == "__main__":
    main()
