#!/usr/bin/env python3
"""Fit and seal FINAL-v14 Module-I source models before reporting performance."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from oceanpath.aim1 import baselines  # noqa: E402
from tools import final_v14_excel_handoff as io  # noqa: E402
from tools import final_v14_modeling as modeling  # noqa: E402

ROOT = REPO / "reports/reruns/final_v14_additions_20260903"
PRE = ROOT / "e4v_pre_reader"
OUT = ROOT / "e4m1_source"
NAMING = ROOT / "e4v_post_reader_xlsx/naming/naming_freeze.json"
COLS = [f"prototype_{p:02d}" for p in range(32)]


def seal_json(path: Path, value: dict) -> None:
    io.write_new(path, io.json_bytes(value), mode=0o400)
    record = io.identity(path)
    io.write_new(path.with_name(path.name + ".seal.json"), io.json_bytes({
        "created_utc": io.utc_now(), "sha256": record["sha256"], "artifact": record,
    }), mode=0o400)


def read_sealed(path: Path) -> dict:
    seal = json.loads(path.with_name(path.name + ".seal.json").read_text())
    expected = seal.get("sha256", seal.get("artifact", {}).get("sha256"))
    if io.sha256_file(path) != expected:
        raise ValueError(f"seal mismatch: {path}")
    return json.loads(path.read_text())


def verify_record(record: dict) -> Path:
    path = Path(record["path"])
    if io.sha256_file(path) != record["sha256"]:
        raise ValueError(f"input digest mismatch: {path}")
    return path


def patient_frame(pre: Path) -> pd.DataFrame:
    frame = pd.read_csv(pre / "inputs/tcga_surgen_primary.csv")
    # These patient-level clinical fields must not disagree between slides.
    fields = ["target_label", "subcohort", "k_fold", "age_at_diagnosis", "sex",
              "site_class", "stage_class", "tumor_site_group", "stage_group_major", "msi_dmmr", "braf"]
    for column in fields:
        if (frame.groupby("patient_id")[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"conflicting within-patient metadata: {column}")
    frame = frame.sort_values(["patient_id", "slide_id"]).drop_duplicates("patient_id").reset_index(drop=True)
    frame["label"] = frame["target_label"].astype(int)
    if len(frame) != 1239 or frame["label"].sum() != 501:
        raise ValueError("development population census drift")
    return frame


def aligned_matrix(path: Path, frame: pd.DataFrame) -> np.ndarray:
    data = pd.read_parquet(path).set_index("patient_id")
    if not data.index.is_unique or set(data.index) != set(frame.patient_id):
        raise ValueError("profile patient identity mismatch")
    data = data.loc[frame.patient_id]
    if not np.array_equal(data["fold"].to_numpy(int), frame.k_fold.to_numpy(int)):
        raise ValueError("profile outer-fold mismatch")
    X = data[COLS].to_numpy(float)
    if not np.isfinite(X).all() or np.min(X) < 0 or not np.allclose(X.sum(1), 1, atol=1e-12):
        raise ValueError("invalid abundance profile")
    return X


def clinical_design(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    numeric = ["age_at_diagnosis"]
    categorical = ["sex", "site_class", "stage_class"]
    X, names, levels, medians, scales = baselines._design(train, numeric, categorical)
    V, vnames, _, _, _ = baselines._design(test, numeric, categorical, levels, medians, scales)
    V = baselines._align(V, vnames, names)
    return X, V, {"feature_names": names, "levels": levels, "medians": medians, "scales": scales}


def joint_fit(X: np.ndarray, y: np.ndarray, V: np.ndarray,
              train_frame: pd.DataFrame, test_frame: pd.DataFrame, splits: list) -> dict:
    candidates = []
    encoded = []
    for train, val in splits:
        a, b, encoder = clinical_design(train_frame.iloc[train], train_frame.iloc[val])
        encoded.append((np.column_stack([X[train], a]), np.column_stack([X[val], b]), encoder))
    for C in modeling.PENALTIES:
        losses = []
        failures = []
        for f, ((train, val), (a, b, _)) in enumerate(zip(splits, encoded, strict=True)):
            fitted = modeling.fit_logistic(a, y[train], C)
            if fitted["status"] != "ESTIMABLE":
                failures.append({"fold": f, "reason": fitted["reason"]})
                losses.append(None)
            else:
                score = modeling.predict(fitted, b)
                loss = float(log_loss(y[val], expit(score), labels=[0, 1]))
                if not np.isfinite(loss):
                    failures.append({"fold": f, "reason": "nonfinite validation loss"})
                    losses.append(None)
                else:
                    losses.append(loss)
        candidates.append({"penalty": C, "eligible": not failures, "inner_losses": losses,
                           "mean_loss": None if failures else float(np.mean(losses)), "failures": failures})
    good = [c for c in candidates if c["eligible"]]
    if not good:
        return {"status": "NOT_ESTIMABLE", "reason": "no eligible joint penalty", "candidates": candidates}
    loss = min(c["mean_loss"] for c in good)
    C = min(c["penalty"] for c in good if abs(c["mean_loss"] - loss) <= 1e-12)
    a, b, encoder = clinical_design(train_frame, test_frame)
    fitted = modeling.fit_logistic(np.column_stack([X, a]), y, C)
    fitted.update(candidates=candidates, clinical_encoder=encoder,
                  inner_clinical_encoders=[e[2] for e in encoded], concept_columns=X.shape[1])
    if fitted["status"] == "ESTIMABLE":
        fitted["predictions"] = modeling.predict(fitted, np.column_stack([V, b])).tolist()
    return fitted


def inherited_clinical_fit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    X, V, encoder = clinical_design(train, test)
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, train.label.to_numpy(int))
    if any(issubclass(w.category, ConvergenceWarning) for w in recorded):
        return {"status": "NOT_ESTIMABLE", "reason": "inherited clinical-only nonconvergence"}
    scores = model.decision_function(V)
    if not np.isfinite(scores).all():
        return {"status": "NOT_ESTIMABLE", "reason": "nonfinite inherited clinical scores"}
    return {"status": "ESTIMABLE", "selected_penalty": 1.0, "clinical_encoder": encoder,
            "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
            "predictions": scores.tolist(), "recipe": "inherited e1d unweighted C=1 max_iter=2000"}


def prepare(pre: Path = PRE, out: Path = OUT, naming: Path = NAMING) -> dict:
    names = read_sealed(naming)
    gate_status = names.get("name_gate_status", names["status"])
    if gate_status not in {"NAME_GATE_PASS", "NAME_GATE_FAIL"}:
        raise ValueError("naming gate is not frozen")
    read_sealed(pre / "analysis_dictionary.json")
    prepared = json.loads((pre / "receipts/prepare.json").read_text())
    for record in prepared["inputs"].values():
        verify_record(record)
    frame = patient_frame(pre)
    profile_receipt = json.loads((pre / "receipts/profiles.json").read_text())
    pins = [io.identity(naming), io.identity(pre / "analysis_dictionary.json"),
            io.identity(pre / "inputs/tcga_surgen_primary.csv"),
            io.identity(REPO / "reports/final_v14_PREREGISTRATION.md")]
    for f in range(5):
        record = profile_receipt["artifacts"][f"outer_fold_{f}"]["patient_profiles"]
        pins.append(io.identity(verify_record(record)))
        aligned_matrix(Path(record["path"]), frame)
        for seed in range(42, 47):
            receipt_path = pre / f"teachers/seed{seed}/fold_{f}/receipt.json"
            receipt = json.loads(receipt_path.read_text())
            if receipt["status"] != "PASS" or receipt["maximum_absolute_v13_logit_delta"] > 1e-6:
                raise ValueError(f"blocked teacher seed{seed}/fold{f}")
            pins.extend([io.identity(receipt_path), io.identity(verify_record(receipt["artifacts"]["patient_logits"]))])
    teachers = json.loads((pre / "receipts/teachers.json").read_text())
    pins.append(io.identity(verify_record(teachers["artifacts"]["oof_native_logits"])))
    contract = {
        "status": "SOURCE_FIT_CONTRACT_SEALED", "created_utc": io.utc_now(),
        "patients": len(frame), "representations": ["ALL32"] + (["NAMED_OOF"] if gate_status == "NAME_GATE_PASS" else []),
        "naming_status": gate_status, "input_pins": pins,
        "code": [io.identity(Path(__file__)), io.identity(Path(modeling.__file__)),
                 io.identity(Path(baselines.__file__)), io.identity(REPO / "uv.lock")],
        "supplement_to_analysis_dictionary": {
            "ridge_alpha": list(modeling.PENALTIES), "logistic_C": list(modeling.PENALTIES),
            "inner_splits": "4 shuffled StratifiedKFold on sorted source subcohort x KRAS, random_state=20260819",
            "site_reference": "Appendix", "stage_reference": "I",
            "msi_normalization": {"MSI/dMMR": "MSI-H/dMMR"},
            "clinical_fields": ["age_at_diagnosis", "sex", "site_class", "stage_class"],
            "clinical_encoder": "exact inherited baselines._design and alignment; category/age information fit separately within every inner/outer training split",
            "stage_known_sensitivity": "fixed restriction of all-source OOF predictions using inherited derived-stage 1060-patient roster; no subgroup refit",
            "reason": "Resolve dictionary aliases and omitted grids explicitly before fitting; preserve inherited clinical implementation byte-identical.",
        },
        "model_estimability": "no fallback after selected refit failure; all candidates require every inner fit converged and finite",
        "targets_accessed": False,
    }
    seal_json(out / "source_fit_contract.json", contract)
    return contract


def fit_source(pre: Path = PRE, out: Path = OUT, naming: Path = NAMING) -> dict:
    contract_path = out / "source_fit_contract.json"
    contract = read_sealed(contract_path) if contract_path.exists() else prepare(pre, out, naming)
    for pin in contract["input_pins"] + contract["code"]:
        verify_record(pin)
    names = read_sealed(naming)
    frame = patient_frame(pre)
    full = pd.read_parquet(pre / "teachers/oof_native_logits_recomputed.parquet").set_index("patient_id").loc[frame.patient_id]
    summary = {"status": "SOURCE_MODELS_SEALED", "created_utc": io.utc_now(),
               "contract": io.identity(contract_path), "representations": {}}
    outputs = frame[["patient_id", "subcohort", "k_fold", "label", "msi_dmmr", "braf", "tumor_site_group", "stage_group_major"]].copy()
    outputs["full_logit"] = full.mean_logit_5seed.to_numpy()
    for representation in contract["representations"]:
        rep_summary = {"outer_folds": {}, "source_fidelity_status": "ESTIMABLE"}
        for subset in ["all"]:
            use = np.ones(len(frame), dtype=bool)
            for column in ["concept", "joint", "clinical"] + ([f"ridge_seed{s}" for s in range(42, 47)] if subset == "all" else []):
                outputs[f"{representation}__{subset}__{column}"] = np.nan
            for fold in range(5):
                start = time.monotonic()
                mask_train = use & frame.k_fold.ne(fold).to_numpy()
                mask_test = use & frame.k_fold.eq(fold).to_numpy()
                train_idx, test_idx = np.flatnonzero(mask_train), np.flatnonzero(mask_test)
                a, b = frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)
                profile = aligned_matrix(pre / f"profiles/outer_fold_{fold}/patient_profiles.parquet", frame)
                coordinates = list(range(32)) if representation == "ALL32" else names["NAMED_OOF"][str(fold)]
                X, V = profile[train_idx][:, coordinates], profile[test_idx][:, coordinates]
                strata = a.subcohort.astype(str) + "|" + a.label.astype(str)
                if strata.value_counts().min() < 4:
                    raise ValueError("inner joint-stratum count below four")
                splits = list(StratifiedKFold(4, shuffle=True, random_state=20260819).split(X, strata))
                checkpoint = out / "fits" / representation / subset / f"fold_{fold}.json"
                if checkpoint.exists():
                    result = read_sealed(checkpoint)
                    if result["contract_sha256"] != io.sha256_file(contract_path):
                        raise ValueError("checkpoint contract drift")
                else:
                    result = {"contract_sha256": io.sha256_file(contract_path), "fold": fold,
                              "representation": representation, "subset": subset, "coordinates": coordinates,
                              "train_patients": a.patient_id.tolist(), "test_patients": b.patient_id.tolist(),
                              "inner_splits": [{"train": x.tolist(), "validation": y.tolist()} for x, y in splits],
                              "models": {}}
                    result["models"]["concept"] = modeling.nested_logistic(X, a.label.to_numpy(), V, splits)
                    result["models"]["joint"] = joint_fit(X, a.label.to_numpy(), V, a, b, splits)
                    result["models"]["clinical"] = inherited_clinical_fit(a, b)
                    if subset == "all":
                        for seed in range(42, 47):
                            teacher = pd.read_parquet(pre / f"teachers/seed{seed}/fold_{fold}/patient_logits.parquet").set_index("patient_id")
                            target = teacher.loc[a.patient_id, "logit"].to_numpy(float)
                            result["models"][f"ridge_seed{seed}"] = modeling.nested_ridge(X, target, V, splits)
                    result["elapsed_seconds"] = time.monotonic() - start
                    seal_json(checkpoint, result)
                for model_name, model in result["models"].items():
                    if model["status"] == "ESTIMABLE":
                        outputs.loc[test_idx, f"{representation}__{subset}__{model_name}"] = model["predictions"]
                    if model_name.startswith("ridge") and model["status"] != "ESTIMABLE":
                        rep_summary["source_fidelity_status"] = "SCORE_FIDELITY_NOT_EVALUABLE"
                if subset == "all":
                    concept = result["models"]["concept"]
                    rep_summary["outer_folds"][str(fold)] = {
                        "status": concept["status"], "selected_penalty": concept.get("selected_penalty"),
                        "checkpoint": io.identity(checkpoint), "coordinates": coordinates,
                    }
                print(json.dumps({"representation": representation, "subset": subset,
                                  "fold": fold, "elapsed_seconds": time.monotonic() - start}), flush=True)
        ridge_cols = [f"{representation}__all__ridge_seed{s}" for s in range(42, 47)]
        outputs[f"{representation}__all__ridge_mean"] = outputs[ridge_cols].mean(axis=1, skipna=False)
        summary["representations"][representation] = rep_summary
    prediction_path = out / "oof_predictions.parquet"
    if prediction_path.exists():
        raise ValueError("source prediction seal already exists; verify existing summary instead")
    outputs.to_parquet(prediction_path, index=False)
    prediction_path.chmod(0o400)
    summary["predictions"] = io.identity(prediction_path)
    summary["fit_artifacts"] = io.tree_inventory(out / "fits")
    seal_json(out / "source_fit_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "fit-source"])
    parser.add_argument("--pre-reader-root", type=Path, default=PRE)
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--naming", type=Path, default=NAMING)
    args = parser.parse_args()
    result = (prepare if args.command == "prepare" else fit_source)(args.pre_reader_root, args.output_root, args.naming)
    print(json.dumps({"status": result["status"], "output": str(args.output_root)}, indent=2))


if __name__ == "__main__":
    main()
