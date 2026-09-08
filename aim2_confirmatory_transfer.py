#!/usr/bin/env python3
"""Confirmatory final-v8 Cross-Protocol H&E Transfer (E2-CPHT).

This additive runner implements the model that the frozen final-v8 design
declares primary but that the earlier Orion runner deliberately did not train:
three fresh all-conventional UNIv1 gated-ABMIL refits, each trained for exactly
6,060 optimizer steps on the 1,486 conventional primary patients.

The existing ``aim2_cross_protocol_transfer.py`` lineage is immutable and is
consumed here only as a sealed source of the already-run 15-fold rank
sensitivity and four family-LOCO score sets.  Four sibling-LOCO ensembles from
``aim2_sibling_loco.py`` complete the prespecified eight-scorer matrix.

The command boundary is intentionally strict::

    python aim2_confirmatory_transfer.py preflight
    python aim2_confirmatory_transfer.py seal
    python aim2_confirmatory_transfer.py train
    python aim2_confirmatory_transfer.py score --sibling-root <lineage>/e2ad
    python aim2_confirmatory_transfer.py report --sibling-root <lineage>/e2ad

``seal`` freezes the source OOF Platt map, exact refit configurations, source
manifest/splits, label-blind Orion manifest, and full packed-UNI byte
identities. ``train`` never opens Orion outcomes. ``score`` exports native
logits and 512-dimensional patient embeddings without opening outcomes and
writes an inference seal. Only ``report`` joins the already-frozen Orion KRAS
labels.

Governance caveat
-----------------
Orion outcomes were accessed by the earlier sensitivity analysis before this
artifact-level refit contract existed.  The estimator itself was specified in
the timestamped final-v8 prose beforehand, but this run is not represented as
a pristine pre-outcome artifact seal.  Every receipt records that deviation,
and no Orion result may select a seed, checkpoint, calibrator, normalization,
or ensemble weight.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_cross_protocol_transfer as prior_runner  # noqa: E402
from oceanpath.aim1 import evaluate, lineage  # noqa: E402
from oceanpath.datasets.packed import (  # noqa: E402
    PackedFeatureStore,
    feature_inventory_sha256,
    validate_packed_dir,
)
from oceanpath.eval.core import compute_calibration_intercept_slope  # noqa: E402
from oceanpath.eval.external import sigmoid  # noqa: E402

LABEL_SOURCE = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v5.csv")
SOURCE_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
SOURCE_SPLITS = REPO / "outputs/splits/aim1_1a/aim1_balanced5/splits.parquet"
FEATURE_DIR = prior_runner.FEATURE_DIR
PACK_DIR = prior_runner.PACK_DIR
AIM1_ROOT = prior_runner.AIM1_ROOT
PRIOR_SENSITIVITY_ROOT = prior_runner.DEFAULT_OUTPUT_ROOT
DEFAULT_OUTPUT_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_final_v8_v1_20260822")

SEEDS: tuple[int, ...] = (42, 43, 44)
CAP = 8_192
STEP_BUDGET = 6_060
REFIT_EPOCH_CEILING = 10
EXPECTED_SOURCE_SLIDES = 1_642
EXPECTED_SOURCE_PATIENTS = 1_486
EXPECTED_SOURCE_MUTANT = 604
EXPECTED_ORION_SLIDES = 41
EXPECTED_ORION_PATIENTS = 40
EMBED_DIM = 512
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_817
INFERENCE_DEVICE = "cuda"
INFERENCE_NUM_WORKERS = 4

SIBLING_ARMS: tuple[str, ...] = (
    "sr386",
    "sr1482",
    "tcga_coad",
    "tcga_read",
)
SIBLING_DISPLAY = {
    "sr386": "SR386",
    "sr1482": "SR1482",
    "tcga_coad": "TCGA-COAD",
    "tcga_read": "TCGA-READ",
}
FAMILY_TARGETS = prior_runner.FAMILY_LOCO_TARGETS

OUTCOME_ACCESS_CAVEAT = (
    "Orion outcomes were accessed in the sealed existing-model sensitivity run "
    "before this all-conventional refit artifact contract was written. The raw "
    "all-conventional estimator and exact 6,060-step rule were specified in "
    "final-v8 beforehand, but this is not a pristine pre-outcome artifact-level seal."
)


class ContractError(RuntimeError):
    """A frozen CPHT input, training run, or inference artifact changed."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def component_root(output_root: Path) -> Path:
    return output_root / "cpht"


def input_root(output_root: Path) -> Path:
    return component_root(output_root) / "inputs"


def contract_path(output_root: Path) -> Path:
    return input_root(output_root) / "execution_contract.json"


def source_copy_path(output_root: Path) -> Path:
    return input_root(output_root) / "source_primary.csv"


def orion_manifest_path(output_root: Path) -> Path:
    return input_root(output_root) / "orion_primary.csv"


def split_copy_path(output_root: Path) -> Path:
    return input_root(output_root) / "splits_root/aim1_1a/aim1_balanced5/splits.parquet"


def calibrator_path(output_root: Path) -> Path:
    return component_root(output_root) / "calibrators/cap8192_all_conventional.json"


def config_path(output_root: Path, seed: int) -> Path:
    return input_root(output_root) / f"refit_seed{seed}.yaml"


def run_dir(output_root: Path, seed: int) -> Path:
    return component_root(output_root) / f"train/pb_cap8192/all_conventional/seed{seed}"


def checkpoint_path(output_root: Path, seed: int) -> Path:
    return run_dir(output_root, seed) / "final/refit/model.ckpt"


def fit_summary_path(output_root: Path, seed: int) -> Path:
    return run_dir(output_root, seed) / "fit_summary.json"


def run_config_path(output_root: Path, seed: int) -> Path:
    return run_dir(output_root, seed) / "resolved_config.yaml"


def score_path(output_root: Path, kind: str, seed: int, arm: str | None = None) -> Path:
    if kind == "confirmatory":
        name = f"all_conventional_seed{seed}_orion.parquet"
    elif kind == "sibling" and arm in SIBLING_ARMS:
        name = f"loco_{arm}_seed{seed}_orion.parquet"
    else:
        raise ValueError(f"Unknown score kind/arm: {kind}/{arm}")
    return component_root(output_root) / "scores" / name


def score_receipt_path(path: Path) -> Path:
    return path.with_suffix(".receipt.json")


def patient_feature_path(output_root: Path) -> Path:
    return component_root(output_root) / "inference/orion_patient_features.parquet"


def inference_seal_path(output_root: Path) -> Path:
    return component_root(output_root) / "inference_seal.json"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    return _sha256_bytes(frame.to_csv(index=False, lineterminator="\n").encode())


def _artifact(path: Path) -> dict[str, Any]:
    return lineage.artifact_identity(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"Expected JSON object: {path}")
    return value


def _publish_text(path: Path, value: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != value:
            raise FileExistsError(f"Refusing to replace non-identical artifact: {path}")
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(value)


def _publish_bytes(path: Path, value: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != value:
            raise FileExistsError(f"Refusing to replace non-identical artifact: {path}")
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)


def _publish_json(path: Path, value: Any) -> None:
    _publish_text(path, _canonical_json(value))


def _publish_csv(path: Path, value: pd.DataFrame) -> None:
    _publish_text(path, value.to_csv(index=False, lineterminator="\n"))


def _publish_parquet(path: Path, value: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        value.to_parquet(temporary, index=False)
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _nested(record: dict[str, Any], dotted: str) -> Any:
    value: Any = record
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ContractError(f"Missing governed field {dotted!r}")
        value = value[key]
    return value


def _assert_recipe(record: dict[str, Any], expected: dict[str, Any], context: Path) -> None:
    mismatches = {
        key: {"expected": wanted, "observed": _nested(record, key)}
        for key, wanted in expected.items()
        if _nested(record, key) != wanted
    }
    if mismatches:
        raise ContractError(f"Recipe mismatch in {context}: {mismatches}")


def _source_population(manifest: pd.DataFrame) -> None:
    required = {"slide_id", "patient_id", "target_label", "cohort", "subcohort"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ContractError(f"Source manifest lacks columns: {missing}")
    if len(manifest) != EXPECTED_SOURCE_SLIDES:
        raise ContractError(
            f"Source manifest has {len(manifest)} slides, expected {EXPECTED_SOURCE_SLIDES}"
        )
    patients = manifest.groupby("patient_id", sort=False)["target_label"].agg(
        n="size", label="first", labels="nunique"
    )
    if len(patients) != EXPECTED_SOURCE_PATIENTS or not patients["labels"].eq(1).all():
        raise ContractError("Source patient census/grouped labels violate the contract")
    if int(pd.to_numeric(patients["label"], errors="raise").sum()) != EXPECTED_SOURCE_MUTANT:
        raise ContractError("Source mutant patient census violates the contract")
    if set(manifest["cohort"].astype(str)) != {"TCGA", "SurGen", "RIH", "CPTAC"}:
        raise ContractError("Source manifest is not the four conventional families")


def _prior_contract(prior_root: Path) -> dict[str, Any]:
    receipt = _read_json(prior_root / "receipt.json")
    if receipt.get("status") != "PASS" or receipt.get("confirmatory_cpht_completed") is not False:
        raise ContractError("Existing Orion sensitivity receipt is not the sealed partial run")
    contract = _read_json(prior_root / "inputs/preoutcome_contract.json")
    seal = _read_json(prior_root / "inference_seal.json")
    if seal.get("status") != "sealed_before_outcome_join":
        raise ContractError("Existing Orion sensitivity inference is not pre-outcome sealed")
    if seal.get("contract") != _artifact(prior_root / "inputs/preoutcome_contract.json"):
        raise ContractError("Existing Orion sensitivity seal no longer binds its contract")
    for record in seal.get("score_artifacts") or []:
        for key in ("score", "receipt"):
            identity = record[key]
            if identity != _artifact(Path(identity["path"])):
                raise ContractError(f"Existing sensitivity artifact changed: {identity['path']}")
    return contract


def validate_pack_against_prior(
    prior_contract: dict[str, Any], pack_dir: Path = PACK_DIR, feature_dir: Path = FEATURE_DIR
) -> dict[str, Any]:
    """Revalidate full payload hashes already frozen by the prior Orion seal."""

    frozen = prior_contract["feature_store"]
    expected_paths = {
        "meta": pack_dir / "meta.json",
        "index": pack_dir / "index.parquet",
        "features": pack_dir / "features.bin",
        "coords": pack_dir / "coords.bin",
    }
    for key, path in expected_paths.items():
        if frozen.get(key) != _artifact(path):
            raise ContractError(f"Packed UNI {key} differs from the sealed byte identity")
    live_inventory = feature_inventory_sha256(feature_dir)
    meta = validate_packed_dir(pack_dir, verify_source=live_inventory)
    if (
        meta.n_slides != prior_runner.EXPECTED_PACK_SLIDES
        or meta.feat_dim != prior_runner.EXPECTED_FEATURE_DIM
        or live_inventory != frozen.get("source_inventory_sha256")
    ):
        raise ContractError("Packed UNI shape/source inventory differs from the prior seal")
    if str(pack_dir.resolve()) != frozen.get("path"):
        raise ContractError("Packed UNI root differs from the prior seal")
    return frozen


def build_source_calibrator(
    source_manifest: pd.DataFrame,
    *,
    aim1_root: Path = AIM1_ROOT,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the frozen one-map contract on three-seed mean honest source OOF logits."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    frames: list[pd.DataFrame] = []
    inputs: dict[str, Any] = {}
    for seed in SEEDS:
        path = aim1_root / f"seed{seed}/oof_predictions.parquet"
        patient = evaluate.to_patient_level(pd.read_parquet(path), source_manifest)
        patient = patient.sort_values("patient_id", kind="stable").reset_index(drop=True)
        if len(patient) != EXPECTED_SOURCE_PATIENTS:
            raise ContractError(f"Aim1 seed {seed} OOF patient census is incomplete")
        frames.append(patient)
        inputs[str(seed)] = _artifact(path)
    base = frames[0][["patient_id", "label"]].copy()
    for seed, frame in zip(SEEDS[1:], frames[1:], strict=True):
        if not frame["patient_id"].equals(base["patient_id"]):
            raise ContractError(f"Aim1 seed {seed} covers different OOF patients")
        if not frame["label"].equals(base["label"]):
            raise ContractError(f"Aim1 seed {seed} carries different OOF labels")
    eta = np.mean(np.vstack([frame["mean_logit"].to_numpy() for frame in frames]), axis=0)
    y = base["label"].to_numpy(dtype=int)
    if int(y.sum()) != EXPECTED_SOURCE_MUTANT or not np.isfinite(eta).all():
        raise ContractError("Source OOF ensemble census/logits violate the contract")
    model = LogisticRegression(penalty=None, max_iter=1000)
    model.fit(eta.reshape(-1, 1), y)
    intercept = float(model.intercept_[0])
    slope = float(model.coef_[0, 0])
    if not np.isfinite([intercept, slope]).all() or slope <= 0:
        raise ContractError("Source Platt fit is non-finite or non-monotone")
    table = base.assign(mean_oof_logit=eta)
    record = {
        "schema_version": 1,
        "role": "source-only Platt map for the all-conventional three-seed refit",
        "a": intercept,
        "b": slope,
        "n_source": int(len(y)),
        "n_mutant": int(y.sum()),
        "source_prevalence": float(y.mean()),
        "source_oof_auroc": float(roc_auc_score(y, eta)),
        "source_oof_logit_mean": float(np.mean(eta)),
        "source_oof_logit_sd": float(np.std(eta, ddof=1)),
        "source_oof_inputs": inputs,
        # Compatibility alias used by the downstream E2-MET deployment
        # resolver. These are the sealed full-source Aim-1 OOF inputs; no new
        # source-CV fit is required for the all-conventional refit.
        "source_cv_inputs": inputs,
        "aggregation": "mean slide logit within patient, then mean across seeds 42/43/44",
        "target_labels_used": False,
    }
    return record, table


def _compose_refit_config(output_root: Path, seed: int) -> str:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        "data.aim1_model=1a",
        "data.manifest_stem=source_primary",
        f"data.csv_path={source_copy_path(output_root)}",
        f"platform.splits_root={input_root(output_root) / 'splits_root'}",
        "+data.cohort_column=cohort",
        "encoder=univ1",
        "splits=aim1_balanced",
        "model=abmil",
        "model.embed_dim=512",
        "model.attn_dim=384",
        "model.input_dropout=0.10",
        "model.dropout=0.25",
        "training=aim1",
        "training.lr=0.0001",
        "training.weight_decay=0.00001",
        f"training.seed={seed}",
        f"training.dataset_max_instances={CAP}",
        "training.eval_full_bags=true",
        "training.refit_epoch_rule=median",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"training.refit_max_steps={STEP_BUDGET}",
        f"train_dir={run_dir(output_root, seed)}",
        f"exp_name=e2cpht_all_conventional_c8192_s{seed}",
    ]
    with initialize_config_dir(config_dir=str((REPO / "configs").resolve()), version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)
    with open_dict(cfg):
        cfg.training.max_epochs = REFIT_EPOCH_CEILING
        cfg.training.refit_max_steps = STEP_BUDGET
    return OmegaConf.to_yaml(cfg, resolve=True)


def _config_contract(path: Path, seed: int, output_root: Path) -> None:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"Malformed refit configuration: {path}")
    _assert_recipe(
        value,
        {
            "data.csv_path": str(source_copy_path(output_root)),
            "data.num_classes": 2,
            "encoder.name": "uni_v1",
            "encoder.feature_dim": 1024,
            "model.arch": "abmil",
            "model.embed_dim": EMBED_DIM,
            "model.attn_dim": 384,
            "model.gate": True,
            "model.dropout": 0.25,
            "model.input_dropout": 0.1,
            "training.loss_type": "bce",
            "training.training_class_weighted_loss": False,
            "training.class_weights": None,
            "training.sample_weight_column": None,
            "training.train_sampling_strategy": "patient_natural",
            "training.dataset_max_instances": CAP,
            "training.eval_full_bags": True,
            "training.force_float32": True,
            "training.refit_max_steps": STEP_BUDGET,
            "training.seed": seed,
            "platform.precision": "bf16-mixed",
        },
        path,
    )


def implementation_identities() -> list[dict[str, Any]]:
    sources = (
        Path(__file__).resolve(),
        REPO / "aim2_cross_protocol_transfer.py",
        REPO / "src/oceanpath/datasets/datamodule.py",
        REPO / "src/oceanpath/datasets/packed.py",
        REPO / "src/oceanpath/datasets/sampling.py",
        REPO / "src/oceanpath/contracts/slide_ids.py",
        REPO / "src/oceanpath/models/__init__.py",
        REPO / "src/oceanpath/models/abmil.py",
        REPO / "src/oceanpath/models/base.py",
        REPO / "src/oceanpath/models/components.py",
        REPO / "src/oceanpath/models/wsi_classifier.py",
        REPO / "src/oceanpath/training/lightning.py",
        REPO / "src/oceanpath/workflows/finalize.py",
        REPO / "configs/train.yaml",
        REPO / "configs/data/aim1.yaml",
        REPO / "configs/encoder/univ1.yaml",
        REPO / "configs/model/abmil.yaml",
        REPO / "configs/platform/colon_workstation.yaml",
        REPO / "configs/splits/aim1_balanced.yaml",
        REPO / "configs/training/aim1.yaml",
        REPO / "configs/training/default.yaml",
    )
    return [_artifact(path) for path in sources]


def _validate_implementation(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        if entry != _artifact(Path(entry["path"])):
            raise ContractError(f"Implementation changed after seal: {entry['path']}")


def _analysis_environment() -> dict[str, Any]:
    """Bind the exact numeric runtime and executed post-outcome analysis code."""

    import scipy
    import sklearn

    sources = (
        Path(__file__).resolve(),
        Path(prior_runner.__file__).resolve(),
        REPO / "src/oceanpath/eval/core.py",
        REPO / "src/oceanpath/eval/external.py",
    )
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "implementation_sources": [_artifact(path) for path in sources],
    }


def build_execution_contract(
    output_root: Path,
    *,
    label_source: Path = LABEL_SOURCE,
    source_manifest_path: Path = SOURCE_MANIFEST,
    source_splits: Path = SOURCE_SPLITS,
    prior_root: Path = PRIOR_SENSITIVITY_ROOT,
    feature_dir: Path = FEATURE_DIR,
    pack_dir: Path = PACK_DIR,
    aim1_root: Path = AIM1_ROOT,
) -> dict[str, Any]:
    source = pd.read_csv(source_manifest_path)
    _source_population(source)
    prior = _prior_contract(prior_root)
    pack = validate_pack_against_prior(prior, pack_dir, feature_dir)
    pack_index = pd.read_parquet(pack_dir / "index.parquet")
    master = pd.read_csv(label_source)
    orion = prior_runner.build_label_blind_manifest(master, pack_index)
    calibrator, oof_table = build_source_calibrator(source, aim1_root=aim1_root)

    _publish_bytes(source_copy_path(output_root), source_manifest_path.read_bytes())
    _publish_bytes(split_copy_path(output_root), source_splits.read_bytes())
    _publish_csv(orion_manifest_path(output_root), orion)
    calibrator = {
        **calibrator,
        "created_utc": utc_now(),
        "source_manifest": _artifact(source_copy_path(output_root)),
        "source_oof_table_sha256": _frame_sha256(oof_table),
        "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
    }
    _publish_json(calibrator_path(output_root), calibrator)
    config_identities: dict[str, Any] = {}
    for seed in SEEDS:
        path = config_path(output_root, seed)
        _publish_text(path, _compose_refit_config(output_root, seed))
        _config_contract(path, seed, output_root)
        config_identities[str(seed)] = _artifact(path)

    # Validate the inherited 15-fold source campaign, including selected
    # checkpoint and completion identities, even though only its OOF logits
    # enter the new calibrator.
    aim1_models = prior_runner._resolve_aim1_checkpoints(aim1_root)  # noqa: SLF001
    return {
        "schema_version": 1,
        "experiment": "E2-CPHT confirmatory raw all-conventional transfer",
        "created_utc": utc_now(),
        "status": "execution_contract_sealed_after_prior_orion_outcome_access",
        "textual_design_preceded_outcome_access": True,
        "artifact_level_contract_preceded_outcome_access": False,
        "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
        "target_outcomes_opened_by_seal": False,
        "model_selection_from_orion": False,
        "source": {
            "manifest": _artifact(source_copy_path(output_root)),
            "upstream_manifest": _artifact(source_manifest_path),
            "splits": _artifact(split_copy_path(output_root)),
            "upstream_splits": _artifact(source_splits),
            "n_slides": EXPECTED_SOURCE_SLIDES,
            "n_patients": EXPECTED_SOURCE_PATIENTS,
            "n_mutant": EXPECTED_SOURCE_MUTANT,
            "aim1_models_by_seed": aim1_models,
            "calibrator": _artifact(calibrator_path(output_root)),
        },
        "orion": {
            "manifest": _artifact(orion_manifest_path(output_root)),
            "csv_sha256": _frame_sha256(orion),
            "n_slides": EXPECTED_ORION_SLIDES,
            "n_patients": EXPECTED_ORION_PATIENTS,
            "patient_aggregation": "mean native slide logit; C33 has two equal-weight slides",
        },
        "feature_store": pack,
        "prior_sensitivity": {
            "root": str(prior_root.resolve()),
            "receipt": _artifact(prior_root / "receipt.json"),
            "contract": _artifact(prior_root / "inputs/preoutcome_contract.json"),
            "inference_seal": _artifact(prior_root / "inference_seal.json"),
            "role": "sealed outer15 and four-family score inputs",
        },
        "refits": {
            "seeds": list(SEEDS),
            "optimizer_steps_per_seed": STEP_BUDGET,
            "cap": CAP,
            "configs": config_identities,
            "checkpoints_expected": [str(checkpoint_path(output_root, seed)) for seed in SEEDS],
        },
        "sibling_interface": {
            "component": "e2ad",
            "arms": list(SIBLING_ARMS),
            "runner": "aim2_sibling_loco.py",
            "complete_matrix_required_before_report": True,
        },
        "statistics": {
            "primary_endpoint": "all-conventional three-seed mean native patient-logit AUROC",
            "secondary": [
                "AUPRC",
                "source-calibrated Brier",
                "source-calibrated log loss",
                "calibration intercept",
                "calibration slope",
            ],
            "bootstrap": "KRAS-stratified patient percentile bootstrap",
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "corroboration_gate": "raw patient AUROC lower 95% bound > 0.50",
            "analysis_environment": _analysis_environment(),
        },
        "implementation": implementation_identities(),
    }


def _load_execution_contract(output_root: Path, *, validate_pack: bool = False) -> dict[str, Any]:
    path = contract_path(output_root)
    contract = _read_json(path)
    if contract.get("outcome_access_caveat") != OUTCOME_ACCESS_CAVEAT:
        raise ContractError("Outcome-access caveat is absent or changed")
    _validate_implementation(contract.get("implementation") or [])
    if contract.get("statistics", {}).get("analysis_environment") != _analysis_environment():
        raise ContractError("Analysis implementation/runtime changed after execution seal")
    for key in ("manifest", "splits", "calibrator"):
        identity = contract["source"][key]
        if identity != _artifact(Path(identity["path"])):
            raise ContractError(f"Sealed source {key} changed")
    orion_identity = contract["orion"]["manifest"]
    if orion_identity != _artifact(Path(orion_identity["path"])):
        raise ContractError("Sealed label-blind Orion manifest changed")
    for seed in SEEDS:
        identity = contract["refits"]["configs"][str(seed)]
        if identity != _artifact(Path(identity["path"])):
            raise ContractError(f"Sealed seed {seed} refit config changed")
        _config_contract(Path(identity["path"]), seed, output_root)
    prior = contract["prior_sensitivity"]
    for key in ("receipt", "contract", "inference_seal"):
        identity = prior[key]
        if identity != _artifact(Path(identity["path"])):
            raise ContractError(f"Prior sensitivity {key} changed")
    if validate_pack:
        prior_contract = _read_json(Path(prior["contract"]["path"]))
        if validate_pack_against_prior(prior_contract) != contract["feature_store"]:
            raise ContractError("Live packed UNI identity differs from execution contract")
    return contract


def _run_request(output_root: Path, seed: int, contract: dict[str, Any]) -> dict[str, Any]:
    local_config = run_config_path(output_root, seed)
    if not local_config.is_file():
        raise FileNotFoundError(local_config)
    frozen_config = contract["refits"]["configs"][str(seed)]
    local_identity = _artifact(local_config)
    if (
        local_identity["sha256"] != frozen_config["sha256"]
        or local_identity["size_bytes"] != frozen_config["size_bytes"]
    ):
        raise ContractError(f"Seed {seed} run-local config is not the frozen config bytes")
    return {
        "schema_version": 1,
        "status": "requested",
        "created_utc": utc_now(),
        "experiment": "E2-CPHT raw all-conventional refit",
        "lineage": output_root.name,
        "seed": seed,
        "sampling_seed": seed,
        "cap": CAP,
        "optimizer_step_budget": STEP_BUDGET,
        "run_dir": str(run_dir(output_root, seed).resolve()),
        "execution_contract": _artifact(contract_path(output_root)),
        "resolved_config": local_identity,
        "frozen_input_config": frozen_config,
        "source_manifest": contract["source"]["manifest"],
        "splits": contract["source"]["splits"],
        "feature_store": contract["feature_store"],
        "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
        "target_outcomes_opened": False,
    }


def _completed_refit(output_root: Path, seed: int) -> dict[str, Any]:
    summary_path = fit_summary_path(output_root, seed)
    checkpoint = checkpoint_path(output_root, seed)
    summary = _read_json(summary_path)
    expected = {
        "status": "completed",
        "lineage": output_root.name,
        "seed": seed,
        "sampling_seed": seed,
        "cap": CAP,
        "optimizer_step_budget": STEP_BUDGET,
        "sampler": "patient_natural",
        "loss_weighting": "none",
        "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
        "target_outcomes_opened": False,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ContractError(
                f"Refit summary {key} mismatch for seed {seed}: {summary.get(key)!r}"
            )
    result = summary.get("result") or {}
    result_expected = {
        "actual_optimizer_steps": STEP_BUDGET,
        "refit_max_steps": STEP_BUDGET,
        "batch_size": 1,
        "accumulate_grad_batches": 1,
        "seed": seed,
        "sampling_seed": seed,
        "lr_scheduler": "cosine",
        "lr_scheduler_interval": "step",
        "lr_scheduler_total_steps": STEP_BUDGET,
        "train_sampling_strategy": "patient_natural",
        "sample_weight_column": None,
        "class_weights": None,
        "dataset_max_instances": CAP,
        "max_instances": None,
        "eval_full_bags": True,
    }
    for key, value in result_expected.items():
        if result.get(key) != value:
            raise ContractError(f"Seed {seed} refit result {key} mismatch")
    if summary.get("model") != _artifact(checkpoint):
        raise ContractError(f"Seed {seed} checkpoint changed after fit")
    if summary.get("resolved_config") != _artifact(run_config_path(output_root, seed)):
        raise ContractError(f"Seed {seed} refit config identity mismatch")
    request = _read_json(run_dir(output_root, seed) / "run_request.json")
    if request.get("execution_contract") != _artifact(contract_path(output_root)):
        raise ContractError(f"Seed {seed} run request binds a different contract")
    contract = _load_execution_contract(output_root)
    if request.get("resolved_config") != _artifact(run_config_path(output_root, seed)):
        raise ContractError(f"Seed {seed} run request config identity mismatch")
    if request.get("frozen_input_config") != contract["refits"]["configs"][str(seed)]:
        raise ContractError(f"Seed {seed} run request lost its frozen input config")
    return summary


def _run_logged(command: list[str], log_path: Path) -> int:
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError(f"Refusing to replace log: {log_path}")
    with log_path.open("x", encoding="utf-8", buffering=1) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_stream.write(line)
        return int(process.wait())


def train_one(output_root: Path, seed: int, *, dry_run: bool) -> None:
    contract = _load_execution_contract(output_root)
    directory = run_dir(output_root, seed)
    if directory.exists():
        try:
            _completed_refit(output_root, seed)
        except Exception as exc:
            raise ContractError(
                f"Partial/invalid immutable seed {seed} run exists; use a new lineage"
            ) from exc
        print(f"seed {seed}: completed and hash-valid; skipping")
        return
    if dry_run:
        print(
            _canonical_json(
                {
                    "seed": seed,
                    "optimizer_step_budget": STEP_BUDGET,
                    "run_dir": str(directory.resolve()),
                    "run_local_resolved_config": str(run_config_path(output_root, seed)),
                    "frozen_input_config": contract["refits"]["configs"][str(seed)],
                    "target_outcomes_opened": False,
                }
            ),
            end="",
        )
        return
    directory.mkdir(parents=True, exist_ok=False)
    _publish_bytes(
        run_config_path(output_root, seed),
        config_path(output_root, seed).read_bytes(),
    )
    request = _run_request(output_root, seed, contract)
    _publish_json(directory / "run_request.json", request)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_fit",
        "--output-root",
        str(output_root),
        "--seed",
        str(seed),
    ]
    return_code = _run_logged(command, directory / "stdout_stderr.log")
    if return_code != 0:
        _publish_json(
            directory / "failure.json",
            {
                "status": "failed",
                "finished_utc": utc_now(),
                "returncode": return_code,
                "command": command,
                "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
            },
        )
        raise RuntimeError(f"Seed {seed} refit failed; evidence retained at {directory}")
    _completed_refit(output_root, seed)


def _fit(output_root: Path, seed: int) -> None:
    from omegaconf import OmegaConf

    from oceanpath.workflows.finalize import _run_refit

    contract = _load_execution_contract(output_root)
    directory = run_dir(output_root, seed)
    request = _read_json(directory / "run_request.json")
    if request != _run_request_comparable(request, _run_request(output_root, seed, contract)):
        # ``created_utc`` is intentionally unique to the already-written request.
        raise ContractError(f"Seed {seed} run request differs from the sealed inputs")
    cfg = OmegaConf.load(run_config_path(output_root, seed))
    final_dir = directory / "final"
    final_dir.mkdir(parents=True, exist_ok=False)
    result = _run_refit(cfg, final_dir, [{"best_epoch": REFIT_EPOCH_CEILING}])
    checkpoint = checkpoint_path(output_root, seed)
    _publish_json(
        fit_summary_path(output_root, seed),
        {
            "schema_version": 1,
            "status": "completed",
            "finished_utc": utc_now(),
            "experiment": "E2-CPHT raw all-conventional refit",
            "lineage": output_root.name,
            "seed": seed,
            "sampling_seed": seed,
            "cap": CAP,
            "optimizer_step_budget": STEP_BUDGET,
            "sampler": "patient_natural",
            "loss_weighting": "none",
            "model": _artifact(checkpoint),
            "resolved_config": _artifact(run_config_path(output_root, seed)),
            "run_request": _artifact(directory / "run_request.json"),
            "result": result,
            "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
            "target_outcomes_opened": False,
        },
    )


def _run_request_comparable(existing: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    value = dict(live)
    value["created_utc"] = existing.get("created_utc")
    return value


def resolve_sibling_models(sibling_root: Path) -> dict[str, dict[str, Any]]:
    seal_path = sibling_root / "inputs/manifest_contract.json"
    seal = _read_json(seal_path)
    seal_expected = {
        "schema_version": 1,
        "experiment": "E2a-D sibling-stratum LOCO",
        "cap": CAP,
        "optimizer_step_budget": STEP_BUDGET,
        "seeds": list(SEEDS),
    }
    for key, value in seal_expected.items():
        if seal.get(key) != value:
            raise ContractError(f"Sibling manifest contract {key} mismatch")
    if set(seal.get("arms") or {}) != set(SIBLING_ARMS):
        raise ContractError("Sibling manifest contract has an incomplete arm roster")
    resolved: dict[str, dict[str, Any]] = {}
    for arm in SIBLING_ARMS:
        models: dict[str, Any] = {}
        for seed in SEEDS:
            directory = sibling_root / f"train/pb_cap8192/{arm}/seed{seed}"
            summary_path = directory / "fit_summary.json"
            summary = _read_json(summary_path)
            expected = {
                "status": "completed",
                "arm": arm,
                "seed": seed,
                "cap": CAP,
                "optimizer_step_budget": STEP_BUDGET,
            }
            for key, value in expected.items():
                if summary.get(key) != value:
                    raise ContractError(
                        f"Sibling {arm}/seed{seed} summary {key} mismatch: {summary.get(key)!r}"
                    )
            checkpoint = directory / "final/refit/model.ckpt"
            if summary.get("model") != _artifact(checkpoint):
                raise ContractError(f"Sibling checkpoint identity mismatch: {checkpoint}")
            for key in ("resolved_config", "run_request"):
                identity = summary.get(key) or {}
                if not isinstance(identity, dict) or identity != _artifact(
                    Path(str(identity.get("path", "")))
                ):
                    raise ContractError(f"Sibling {key} identity mismatch: {summary_path}")
            result = summary.get("result") or {}
            if (
                result.get("actual_optimizer_steps") != STEP_BUDGET
                or result.get("refit_max_steps") != STEP_BUDGET
                or result.get("train_sampling_strategy") != "patient_natural"
                or result.get("dataset_max_instances") != CAP
                or result.get("eval_full_bags") is not True
            ):
                raise ContractError(f"Sibling fit execution contract mismatch: {summary_path}")
            models[str(seed)] = {
                "checkpoint": _artifact(checkpoint),
                "fit_summary": _artifact(summary_path),
            }
        cal_path = sibling_root / f"calibrators/cap8192_{arm}.json"
        calibrator = _read_json(cal_path)
        if calibrator.get("target_labels_used_for_fit") is not False:
            raise ContractError(
                f"Sibling calibrator does not declare source-only fitting: {cal_path}"
            )
        if not np.isfinite([float(calibrator["a"]), float(calibrator["b"])]).all():
            raise ContractError(f"Sibling calibrator is non-finite: {cal_path}")
        if float(calibrator["b"]) <= 0:
            raise ContractError(f"Sibling calibrator is non-monotone: {cal_path}")
        receipts = calibrator.get("source_cv_receipts") or {}
        if set(map(str, receipts)) != {str(seed) for seed in SEEDS}:
            raise ContractError(f"Sibling calibrator source-CV roster incomplete: {cal_path}")
        for receipt in receipts.values():
            if (
                isinstance(receipt, dict)
                and "path" in receipt
                and receipt != _artifact(Path(receipt["path"]))
            ):
                raise ContractError(f"Sibling source-CV receipt changed: {receipt['path']}")
        models["calibrator"] = {
            "artifact": _artifact(cal_path),
            "a": float(calibrator["a"]),
            "b": float(calibrator["b"]),
            "n_source": int(calibrator["n_source"]),
        }
        resolved[arm] = models
    resolved["_seal"] = {"artifact": _artifact(seal_path), "value": seal}
    return resolved


def _inference_environment() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("Governed CPHT inference requires CUDA bfloat16")
    return {
        "device": INFERENCE_DEVICE,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "lightning_version": importlib.metadata.version("lightning"),
        "python_version": platform.python_version(),
        "autocast_dtype": "bfloat16",
        "force_float32": True,
        "num_workers": INFERENCE_NUM_WORKERS,
        "implementation": implementation_identities(),
    }


def score_checkpoints_with_embeddings(
    checkpoints: list[tuple[int, Path]],
    manifest: pd.DataFrame,
    *,
    feature_dir: Path = FEATURE_DIR,
    pack_dir: Path = PACK_DIR,
    device: str = INFERENCE_DEVICE,
    num_workers: int = INFERENCE_NUM_WORKERS,
    export_embeddings: bool,
) -> pd.DataFrame:
    """Full-bag packed inference with optional 512-d slide embedding export."""

    import torch
    from torch.utils.data import DataLoader

    from oceanpath.datasets.datamodule import SimpleMILCollator, SlideDataset
    from oceanpath.training.lightning import MILTrainModule

    if device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("Governed CPHT inference requires CUDA bfloat16")
    inventory = feature_inventory_sha256(feature_dir)
    store = PackedFeatureStore(pack_dir, verify_source=inventory)
    slide_ids = manifest["slide_id"].astype(str).tolist()
    dataset = SlideDataset(
        feature_dir=str(feature_dir),
        slide_ids=slide_ids,
        labels=dict.fromkeys(slide_ids, 0),
        max_instances=None,
        is_train=False,
        force_float32=True,
        store=store,
    )
    if set(dataset.slide_ids) != set(slide_ids):
        raise ContractError("Packed inference dataset differs from the Orion manifest")
    loader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": SimpleMILCollator(max_instances=None),
        "pin_memory": True,
    }
    if num_workers:
        loader_kwargs.update({"prefetch_factor": 2, "persistent_workers": True})
    loader = DataLoader(dataset, **loader_kwargs)
    rows: list[dict[str, Any]] = []
    for ordinal, (seed, checkpoint) in enumerate(checkpoints, start=1):
        print(f"  model {ordinal}/{len(checkpoints)} seed{seed}: {checkpoint}", flush=True)
        try:
            module = MILTrainModule.load_from_checkpoint(
                str(checkpoint), map_location=device, weights_only=False
            )
        except TypeError:
            module = MILTrainModule.load_from_checkpoint(str(checkpoint), map_location=device)
        module.eval().to(device)
        with torch.inference_mode():
            for batch in loader:
                features = batch["features"].to(device, non_blocking=True)
                mask = (
                    batch["mask"].to(device, non_blocking=True)
                    if batch.get("mask") is not None
                    else None
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    output = module.model(features, mask=mask)
                logits = output.logits.detach().float().cpu().numpy().ravel()
                embeddings = (
                    output.slide_embedding.detach().float().cpu().numpy()
                    if export_embeddings
                    else None
                )
                for index, slide_id in enumerate(batch["slide_ids"]):
                    row: dict[str, Any] = {
                        "slide_id": str(slide_id),
                        "seed": seed,
                        "fold": 0,
                        "logit": float(logits[index]),
                    }
                    if embeddings is not None:
                        vector = embeddings[index].ravel()
                        if vector.size != EMBED_DIM:
                            raise ContractError(
                                f"Expected {EMBED_DIM}-d slide embedding, got {vector.size}"
                            )
                        row.update({f"e{i}": float(value) for i, value in enumerate(vector)})
                    rows.append(row)
        del module
        torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values(["seed", "slide_id"], kind="stable")
    frame = frame.reset_index(drop=True)
    expected_rows = len(checkpoints) * len(manifest)
    if len(frame) != expected_rows or frame.duplicated(["slide_id", "seed"]).any():
        raise ContractError("CPHT inference row census/uniqueness failed")
    numeric = ["logit"] + ([f"e{i}" for i in range(EMBED_DIM)] if export_embeddings else [])
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ContractError("CPHT inference emitted non-finite values")
    return frame


def aggregate_patient_features(scores: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    expected_columns = {"slide_id", "seed", "logit", *(f"e{i}" for i in range(EMBED_DIM))}
    missing = sorted(expected_columns - set(scores.columns))
    if missing:
        raise ContractError(f"Confirmatory score export lacks columns: {missing[:5]}")
    joined = scores.merge(
        manifest[
            [
                "slide_id",
                "patient_id",
                "exclude_neoadjuvant",
                "exclude_ambiguous_crc15",
            ]
        ],
        on="slide_id",
        validate="many_to_one",
    )
    numeric = ["logit", *(f"e{i}" for i in range(EMBED_DIM))]
    patient = joined.groupby(["seed", "patient_id"], as_index=False, sort=True)[numeric].mean()
    flags = manifest.groupby("patient_id", as_index=False, sort=True)[
        ["exclude_neoadjuvant", "exclude_ambiguous_crc15"]
    ].max()
    patient = patient.merge(flags, on="patient_id", validate="many_to_one")
    counts = joined.groupby(["seed", "patient_id"], sort=True).size().rename("n_slides")
    patient = patient.merge(counts.reset_index(), on=["seed", "patient_id"])
    if len(patient) != len(SEEDS) * EXPECTED_ORION_PATIENTS:
        raise ContractError("Patient feature export does not contain 40 patients per seed")
    if set(patient.loc[patient["n_slides"].eq(2), "patient_id"]) != {"ORION:C33"}:
        raise ContractError("C33 is not the only two-slide patient")
    return patient.sort_values(["seed", "patient_id"], kind="stable").reset_index(drop=True)


def _score_inputs(
    output_root: Path, sibling_root: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], pd.DataFrame]:
    contract = _load_execution_contract(output_root, validate_pack=True)
    siblings = resolve_sibling_models(sibling_root)
    manifest = pd.read_csv(orion_manifest_path(output_root))
    if _frame_sha256(manifest) != contract["orion"]["csv_sha256"]:
        raise ContractError("Orion inference manifest differs from its sealed frame hash")
    for seed in SEEDS:
        _completed_refit(output_root, seed)
    return contract, siblings, manifest


def _score_or_validate_one(
    output_root: Path,
    manifest: pd.DataFrame,
    *,
    kind: str,
    seed: int,
    checkpoint: Path,
    checkpoint_identity: dict[str, Any],
    arm: str | None = None,
    export_embeddings: bool,
) -> None:
    path = score_path(output_root, kind, seed, arm)
    receipt_path = score_receipt_path(path)
    expected_inputs = {
        "checkpoint": checkpoint_identity,
        "manifest": _artifact(orion_manifest_path(output_root)),
        "pack": _load_execution_contract(output_root)["feature_store"],
        "seed": seed,
        "arm": arm,
        "export_embeddings": export_embeddings,
    }
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise ContractError(f"Partial score cache: {path}")
        receipt = _read_json(receipt_path)
        if receipt.get("inputs") != expected_inputs or receipt.get("artifact") != _artifact(path):
            raise ContractError(f"Score cache identity mismatch: {path}")
        return
    frame = score_checkpoints_with_embeddings(
        [(seed, checkpoint)], manifest, export_embeddings=export_embeddings
    )
    _publish_parquet(path, frame)
    _publish_json(
        receipt_path,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "label_blind_score_complete",
            "inputs": expected_inputs,
            "artifact": _artifact(path),
            "n_rows": int(len(frame)),
            "n_slides": int(frame["slide_id"].nunique()),
            "target_outcomes_opened": False,
            "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
        },
    )


def _validate_prior_score_artifacts(contract: dict[str, Any]) -> list[dict[str, Any]]:
    prior_root = Path(contract["prior_sensitivity"]["root"])
    seal = _read_json(prior_root / "inference_seal.json")
    records = seal.get("score_artifacts") or []
    if len(records) != 15:
        raise ContractError("Prior sensitivity score roster is not 15 artifacts")
    for record in records:
        for key in ("score", "receipt"):
            if record[key] != _artifact(Path(record[key]["path"])):
                raise ContractError(f"Prior score artifact changed: {record[key]['path']}")
    return records


def seal_inference(output_root: Path, sibling_root: Path) -> dict[str, Any]:
    destination = inference_seal_path(output_root)
    if destination.is_file():
        return verify_inference_seal(output_root, sibling_root)
    contract, siblings, manifest = _score_inputs(output_root, sibling_root)
    new_records: list[dict[str, Any]] = []
    for seed in SEEDS:
        for kind, arm in [("confirmatory", None), *(("sibling", item) for item in SIBLING_ARMS)]:
            path = score_path(output_root, kind, seed, arm)
            receipt = score_receipt_path(path)
            if not path.is_file() or not receipt.is_file():
                raise ContractError(f"Missing score/receipt before inference seal: {path}")
            new_records.append({"score": _artifact(path), "receipt": _artifact(receipt)})
    confirmatory_scores = pd.concat(
        [pd.read_parquet(score_path(output_root, "confirmatory", seed)) for seed in SEEDS],
        ignore_index=True,
    )
    patient_features = aggregate_patient_features(confirmatory_scores, manifest)
    patient_path = patient_feature_path(output_root)
    _publish_parquet(patient_path, patient_features)
    _publish_json(
        patient_path.with_suffix(".receipt.json"),
        {
            "schema_version": 1,
            "status": "label_blind_patient_features_complete",
            "created_utc": utc_now(),
            "artifact": _artifact(patient_path),
            "score_inputs": [
                _artifact(score_path(output_root, "confirmatory", seed)) for seed in SEEDS
            ],
            "n_rows": int(len(patient_features)),
            "n_patients": int(patient_features["patient_id"].nunique()),
            "embedding_dim": EMBED_DIM,
            "target_outcomes_opened": False,
        },
    )
    value = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "label_blind_inference_sealed_after_prior_outcome_access",
        "execution_contract": _artifact(contract_path(output_root)),
        "sibling_manifest_contract": siblings["_seal"]["artifact"],
        "orion_manifest": _artifact(orion_manifest_path(output_root)),
        "prior_score_artifacts": _validate_prior_score_artifacts(contract),
        "new_score_artifacts": new_records,
        "patient_features": _artifact(patient_path),
        "patient_features_receipt": _artifact(patient_path.with_suffix(".receipt.json")),
        "n_new_score_artifacts": len(new_records),
        "complete_eight_model_matrix": True,
        "confirmatory_cpht_model_present": True,
        "target_outcomes_present": False,
        "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
        "inference_environment": _inference_environment(),
    }
    _publish_json(destination, value)
    return value


def verify_inference_seal(output_root: Path, sibling_root: Path) -> dict[str, Any]:
    seal = _read_json(inference_seal_path(output_root))
    if seal.get("complete_eight_model_matrix") is not True:
        raise ContractError("Inference seal lacks the complete eight-scorer matrix")
    if seal.get("confirmatory_cpht_model_present") is not True:
        raise ContractError("Inference seal lacks the confirmatory CPHT model")
    if seal.get("target_outcomes_present") is not False:
        raise ContractError("Inference seal contains target outcomes")
    if seal.get("inference_environment") != _inference_environment():
        raise ContractError("Inference implementation/runtime changed after inference seal")
    if seal.get("execution_contract") != _artifact(contract_path(output_root)):
        raise ContractError("Inference seal execution contract changed")
    siblings = resolve_sibling_models(sibling_root)
    if seal.get("sibling_manifest_contract") != siblings["_seal"]["artifact"]:
        raise ContractError("Inference seal binds a different sibling contract")
    for record in [
        *(seal.get("prior_score_artifacts") or []),
        *(seal.get("new_score_artifacts") or []),
    ]:
        for key in ("score", "receipt"):
            if record[key] != _artifact(Path(record[key]["path"])):
                raise ContractError(f"Sealed inference artifact changed: {record[key]['path']}")
    for key in ("patient_features", "patient_features_receipt"):
        identity = seal[key]
        if identity != _artifact(Path(identity["path"])):
            raise ContractError(f"Sealed {key} changed")
    return seal


def aggregate_patient_logits(scores: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    required = {"slide_id", "logit"}
    if not required.issubset(scores.columns):
        raise ContractError(f"Score table lacks columns: {sorted(required - set(scores.columns))}")
    slide = scores.groupby("slide_id", as_index=False, sort=False)["logit"].mean()
    joined = slide.merge(manifest[["slide_id", "patient_id"]], on="slide_id", validate="one_to_one")
    patient = joined.groupby("patient_id", as_index=False, sort=True)["logit"].mean()
    if len(patient) != EXPECTED_ORION_PATIENTS:
        raise ContractError("Score aggregation did not yield 40 Orion patients")
    return patient.rename(columns={"logit": "mean_logit"})


def _load_all_scores(
    output_root: Path, sibling_root: Path
) -> tuple[dict[str, pd.DataFrame], dict[str, tuple[float, float] | None], pd.DataFrame]:
    contract = _load_execution_contract(output_root)
    prior = _read_json(Path(contract["prior_sensitivity"]["contract"]["path"]))
    prior_root = Path(contract["prior_sensitivity"]["root"])
    siblings = resolve_sibling_models(sibling_root)
    scores: dict[str, pd.DataFrame] = {
        "aim1_outer15": pd.concat(
            [
                pd.read_parquet(prior_root / f"scores/aim1_seed{seed}_orion.parquet")
                for seed in SEEDS
            ],
            ignore_index=True,
        ),
        "all_conventional": pd.concat(
            [pd.read_parquet(score_path(output_root, "confirmatory", seed)) for seed in SEEDS],
            ignore_index=True,
        ),
    }
    calibrators: dict[str, tuple[float, float] | None] = {"aim1_outer15": None}
    own_cal = _read_json(calibrator_path(output_root))
    calibrators["all_conventional"] = (float(own_cal["a"]), float(own_cal["b"]))
    for target in FAMILY_TARGETS:
        name = f"loco_family_{target.lower()}"
        scores[name] = pd.concat(
            [
                pd.read_parquet(
                    prior_root / f"scores/loco_{target.lower()}_seed{seed}_orion.parquet"
                )
                for seed in SEEDS
            ],
            ignore_index=True,
        )
        cal = prior["family_loco"]["models"][target]["calibrator"]
        calibrators[name] = (float(cal["a"]), float(cal["b"]))
    for arm in SIBLING_ARMS:
        name = f"loco_sibling_{arm}"
        scores[name] = pd.concat(
            [pd.read_parquet(score_path(output_root, "sibling", seed, arm)) for seed in SEEDS],
            ignore_index=True,
        )
        cal = siblings[arm]["calibrator"]
        calibrators[name] = (float(cal["a"]), float(cal["b"]))
    manifest = pd.read_csv(orion_manifest_path(output_root))
    return scores, calibrators, manifest


def analyse_transfer(
    scores: dict[str, pd.DataFrame],
    calibrators: dict[str, tuple[float, float] | None],
    manifest: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    order = [
        "all_conventional",
        "aim1_outer15",
        *(f"loco_family_{target.lower()}" for target in FAMILY_TARGETS),
        *(f"loco_sibling_{arm}" for arm in SIBLING_ARMS),
    ]
    patient_frames: dict[str, pd.DataFrame] = {}
    for name in order:
        patient = aggregate_patient_logits(scores[name], manifest)
        patient = patient.merge(labels, on="patient_id", validate="one_to_one")
        patient = patient.merge(
            manifest.groupby("patient_id", as_index=False)[
                ["exclude_neoadjuvant", "exclude_ambiguous_crc15"]
            ].max(),
            on="patient_id",
            validate="one_to_one",
        )
        patient["scorer"] = name
        patient_frames[name] = patient.sort_values("patient_id", kind="stable").reset_index(
            drop=True
        )
    reference = patient_frames[order[0]]
    for name in order[1:]:
        if not patient_frames[name]["patient_id"].equals(reference["patient_id"]):
            raise ContractError(f"Orion scorers cover different patients: {name}")
        if not patient_frames[name]["label"].equals(reference["label"]):
            raise ContractError(f"Orion scorers carry different labels: {name}")

    population_masks = {
        "all_40": np.ones(len(reference), dtype=bool),
        "exclude_neoadjuvant": ~reference["exclude_neoadjuvant"].to_numpy(dtype=bool),
        "exclude_ambiguous_crc15": ~reference["exclude_ambiguous_crc15"].to_numpy(dtype=bool),
    }
    populations: dict[str, Any] = {}
    for population_name, mask in population_masks.items():
        base = reference.loc[mask].reset_index(drop=True)
        y = base["label"].to_numpy(dtype=int)
        indices = prior_runner.stratified_bootstrap_indices(
            y, n_bootstrap=n_bootstrap, seed=BOOTSTRAP_SEED
        )
        blocks: dict[str, Any] = {}
        samples: dict[str, np.ndarray] = {}
        for name in order:
            patient = patient_frames[name].loc[mask].reset_index(drop=True)
            eta = patient["mean_logit"].to_numpy(dtype=float)
            metric_samples = prior_runner.bootstrap_metric_samples(y, eta, indices)
            auc_values = metric_samples["auroc"]
            ap_values = metric_samples["auprc"]
            block: dict[str, Any] = {
                "n": int(len(y)),
                "n_mutant": int(y.sum()),
                "n_wild_type": int(len(y) - y.sum()),
                "prevalence": float(y.mean()),
                "auroc": float(roc_auc_score(y, eta)),
                "auroc_ci95": prior_runner._interval(auc_values),  # noqa: SLF001
                "auprc": float(average_precision_score(y, eta)),
                "auprc_ci95": prior_runner._interval(ap_values),  # noqa: SLF001
            }
            block["lower_auroc_bound_above_0p5"] = bool(block["auroc_ci95"][0] > 0.5)
            calibrator = calibrators[name]
            if calibrator is None:
                block["probability_metrics"] = "not inferred: rank-only outer15 sensitivity"
            else:
                a, b = calibrator
                probability = sigmoid(a + b * eta)
                draw_probability = probability[indices]
                brier_draws = np.mean((draw_probability - y[indices]) ** 2, axis=1)
                eps = 1e-12
                clipped = np.clip(draw_probability, eps, 1 - eps)
                y_draw = y[indices]
                loss_draws = -np.mean(
                    y_draw * np.log(clipped) + (1 - y_draw) * np.log1p(-clipped), axis=1
                )
                calibration = compute_calibration_intercept_slope(y, probability)
                block["source_calibrated"] = {
                    "brier": float(brier_score_loss(y, probability)),
                    "brier_ci95": prior_runner._interval(brier_draws),  # noqa: SLF001
                    "log_loss": float(log_loss(y, probability)),
                    "log_loss_ci95": prior_runner._interval(loss_draws),  # noqa: SLF001
                    "calibration_intercept": float(calibration["calibration_intercept"]),
                    "calibration_slope": float(calibration["calibration_slope"]),
                    "platt_a": a,
                    "platt_b": b,
                }
            blocks[name] = block
            samples[name] = auc_values
        contrasts: dict[str, Any] = {}
        primary_samples = samples["all_conventional"]
        primary_point = blocks["all_conventional"]["auroc"]
        for name in order[1:]:
            delta = blocks[name]["auroc"] - primary_point
            delta_draw = samples[name] - primary_samples
            contrasts[f"{name}_minus_all_conventional"] = {
                "delta_auroc": float(delta),
                "delta_auroc_ci95": prior_runner._interval(delta_draw),  # noqa: SLF001
                "paired_shared_patient_bootstrap": True,
                "role": "sensitivity; no model selection",
            }
        populations[population_name] = {
            "n": int(len(y)),
            "n_mutant": int(y.sum()),
            "n_wild_type": int(len(y) - y.sum()),
            "metrics": blocks,
            "paired_auroc_vs_confirmatory": contrasts,
        }
    patient_table = pd.concat([patient_frames[name] for name in order], ignore_index=True)
    results = {
        "schema_version": 1,
        "experiment": "E2-CPHT confirmatory raw all-conventional transfer",
        "created_utc": utc_now(),
        "status": "analysis_complete_pending_independent_verification",
        "confirmatory_scorer": "all_conventional",
        "complete_eight_model_matrix": True,
        "outer15_rank_sensitivity_present": True,
        "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
        "model_selection_from_orion": False,
        "population": {"slides": 41, "patients": 40, "mutant": 15, "wild_type": 25},
        "statistics": {
            "bootstrap": "KRAS-stratified patient percentile bootstrap",
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "shared_indices_across_models": True,
        },
        "analysis_environment": _analysis_environment(),
        "populations": populations,
    }
    return patient_table, results


def _compact_results(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, content in results["populations"].items():
        for scorer, metrics in content["metrics"].items():
            cal = metrics.get("source_calibrated") or {}
            rows.append(
                {
                    "population": population,
                    "scorer": scorer,
                    "n": metrics["n"],
                    "n_mutant": metrics["n_mutant"],
                    "auroc": metrics["auroc"],
                    "auroc_ci_low": metrics["auroc_ci95"][0],
                    "auroc_ci_high": metrics["auroc_ci95"][1],
                    "auprc": metrics["auprc"],
                    "brier": cal.get("brier"),
                    "log_loss": cal.get("log_loss"),
                    "calibration_intercept": cal.get("calibration_intercept"),
                    "calibration_slope": cal.get("calibration_slope"),
                }
            )
    return pd.DataFrame(rows)


def cmd_preflight(args: argparse.Namespace) -> None:
    source = pd.read_csv(args.source_manifest)
    _source_population(source)
    prior = _prior_contract(args.prior_root)
    validate_pack_against_prior(prior, args.pack_dir, args.feature_dir)
    calibrator, _ = build_source_calibrator(source, aim1_root=args.aim1_root)
    print(
        f"READY: source {EXPECTED_SOURCE_PATIENTS} patients / {EXPECTED_SOURCE_SLIDES} slides; "
        f"Platt a={calibrator['a']:+.6f}, b={calibrator['b']:.6f}"
    )
    print(f"  new raw refits: {len(SEEDS)} x {STEP_BUDGET} optimizer steps")
    print("  existing sealed sensitivity: outer15 + four family LOCO ensembles")
    print("  complete report additionally requires all four e2ad sibling ensembles")
    print(f"  governance caveat: {OUTCOME_ACCESS_CAVEAT}")


def cmd_seal(args: argparse.Namespace) -> None:
    destination = contract_path(args.output_root)
    if destination.is_file():
        _load_execution_contract(args.output_root, validate_pack=True)
        print(f"hash-valid execution contract already exists: {destination}")
        return
    contract = build_execution_contract(
        args.output_root,
        label_source=args.label_source,
        source_manifest_path=args.source_manifest,
        source_splits=args.source_splits,
        prior_root=args.prior_root,
        feature_dir=args.feature_dir,
        pack_dir=args.pack_dir,
        aim1_root=args.aim1_root,
    )
    _publish_json(destination, contract)
    print(f"sealed execution contract: {destination}")
    print(f"governance caveat: {OUTCOME_ACCESS_CAVEAT}")


def cmd_train(args: argparse.Namespace) -> None:
    _load_execution_contract(args.output_root, validate_pack=True)
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for seed in seeds:
        train_one(args.output_root, seed, dry_run=args.dry_run)


def cmd_fit(args: argparse.Namespace) -> None:
    _fit(args.output_root, args.seed)


def cmd_score(args: argparse.Namespace) -> None:
    contract, siblings, manifest = _score_inputs(args.output_root, args.sibling_root)
    environment = _inference_environment()
    print(f"governed inference: {environment['device_name']} / bf16")
    for seed in SEEDS:
        checkpoint = checkpoint_path(args.output_root, seed)
        _score_or_validate_one(
            args.output_root,
            manifest,
            kind="confirmatory",
            seed=seed,
            checkpoint=checkpoint,
            checkpoint_identity=_artifact(checkpoint),
            export_embeddings=True,
        )
    for arm in SIBLING_ARMS:
        for seed in SEEDS:
            identity = siblings[arm][str(seed)]["checkpoint"]
            checkpoint = Path(identity["path"])
            _score_or_validate_one(
                args.output_root,
                manifest,
                kind="sibling",
                arm=arm,
                seed=seed,
                checkpoint=checkpoint,
                checkpoint_identity=identity,
                export_embeddings=False,
            )
    seal = seal_inference(args.output_root, args.sibling_root)
    print(
        f"sealed label-blind inference: {len(seal['new_score_artifacts'])} new score artifacts; "
        "complete eight-model matrix"
    )
    print(f"patient embeddings: {patient_feature_path(args.output_root)}")


def cmd_report(args: argparse.Namespace) -> None:
    seal = verify_inference_seal(args.output_root, args.sibling_root)
    if seal.get("target_outcomes_present") is not False:
        raise ContractError("Outcome-free inference seal is invalid")
    contract = _load_execution_contract(args.output_root)
    if contract["statistics"]["analysis_environment"] != _analysis_environment():
        raise ContractError("Current analysis environment differs from the sealed contract")
    scores, calibrators, manifest = _load_all_scores(args.output_root, args.sibling_root)
    master = pd.read_csv(args.label_source)
    labels = prior_runner._patient_labels(master)  # noqa: SLF001
    patient, results = analyse_transfer(scores, calibrators, manifest, labels)
    analysis = component_root(args.output_root) / "analysis"
    paths = {
        "patient_scores": analysis / "orion_patient_scores.parquet",
        "results": analysis / "results.json",
        "table": analysis / "results.csv",
        "receipt": analysis / "receipt.json",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Immutable analysis already exists below {analysis}")
    _publish_parquet(paths["patient_scores"], patient)
    _publish_json(paths["results"], results)
    _publish_csv(paths["table"], _compact_results(results))
    _publish_json(
        paths["receipt"],
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "analysis_complete_pending_independent_verification",
            "inference_seal": _artifact(inference_seal_path(args.output_root)),
            "label_source_joined_after_inference_seal": _artifact(args.label_source),
            "outputs": {key: _artifact(path) for key, path in paths.items() if key != "receipt"},
            "analysis_environment": _analysis_environment(),
            "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
        },
    )
    primary = results["populations"]["all_40"]["metrics"]["all_conventional"]
    lo, hi = primary["auroc_ci95"]
    print(f"confirmatory raw CPHT AUROC {primary['auroc']:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"lower-bound-above-0.50 gate: {primary['lower_auroc_bound_above_0p5']}")
    print(f"complete results: {paths['results']}")
    print(f"governance caveat: {OUTCOME_ACCESS_CAVEAT}")


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--source-splits", type=Path, default=SOURCE_SPLITS)
    parser.add_argument("--feature-dir", type=Path, default=FEATURE_DIR)
    parser.add_argument("--pack-dir", type=Path, default=PACK_DIR)
    parser.add_argument("--aim1-root", type=Path, default=AIM1_ROOT)
    parser.add_argument("--prior-root", type=Path, default=PRIOR_SENSITIVITY_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    _add_common_paths(preflight)
    preflight.set_defaults(func=cmd_preflight)
    seal = sub.add_parser("seal")
    _add_common_paths(seal)
    seal.set_defaults(func=cmd_seal)
    train = sub.add_parser("train")
    _add_common_paths(train)
    train.add_argument("--seed", type=int, choices=SEEDS)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=cmd_train)
    fit = sub.add_parser("_fit")
    fit.add_argument("--output-root", type=Path, required=True)
    fit.add_argument("--seed", type=int, choices=SEEDS, required=True)
    fit.set_defaults(func=cmd_fit)
    score = sub.add_parser("score")
    _add_common_paths(score)
    score.add_argument("--sibling-root", type=Path, required=True)
    score.set_defaults(func=cmd_score)
    report = sub.add_parser("report")
    _add_common_paths(report)
    report.add_argument("--sibling-root", type=Path, required=True)
    report.set_defaults(func=cmd_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (ContractError, FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"E2-CPHT CONFIRMATORY FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
