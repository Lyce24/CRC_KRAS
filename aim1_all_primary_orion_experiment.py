#!/usr/bin/env python3
"""Exploratory Aim-1 E0 fit on every conventional primary plus Orion.

This runner is deliberately additive.  It does not edit or write below the
final-v8 lineage.  It creates one new, immutable experiment family containing
three full five-fold E0 runs (seeds 42/43/44), their p75 full-source refits,
source OOF diagnostics/calibration, source-refit scale diagnostics, and
deployment-sensitivity scoring on the canonical RIH-M and SR1482-M manifests.

The source layout preserves every conventional Aim-1 outer/inner assignment.
Orion's frozen CPHT-A balanced outer folds are appended; for outer iteration
``i`` the Orion validation carve-out is CPHT-A fold ``(i + 1) % 5``.  Thus no
Orion patient is redrawn and every Orion validation block remains 3 mutant / 5
wild-type.  Metastatic labels are first opened by ``report`` after three
source-refit and six metastatic target-outcome-blind score artifacts have
been sealed.

Examples::

    python aim1_all_primary_orion_experiment.py plan
    python aim1_all_primary_orion_experiment.py manifest
    python aim1_all_primary_orion_experiment.py preflight
    python aim1_all_primary_orion_experiment.py train
    python aim1_all_primary_orion_experiment.py score
    python aim1_all_primary_orion_experiment.py report
    python aim1_all_primary_orion_experiment.py verify

``training.num_workers=6`` means six DataLoader workers inside one serialized
GPU training process.  It does not launch six competing GPU fits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim1_locked_baseline as e0  # noqa: E402
import aim2_cross_protocol_transfer as cpht  # noqa: E402
import aim2_primary_to_metastatic_transfer as e2met  # noqa: E402
from oceanpath.aim1 import evaluate, lineage  # noqa: E402
from oceanpath.aim1.cli import e1a_challenge as e1a  # noqa: E402
from oceanpath.aim1.technical import mpp_bin  # noqa: E402
from oceanpath.eval.external import sigmoid  # noqa: E402


class ContractError(RuntimeError):
    """An immutable experiment input/output violates its recorded contract."""


SEEDS: tuple[int, ...] = (42, 43, 44)
N_FOLDS = 5
CAP = 8_192
TRAINING_NUM_WORKERS = 6
N_BOOTSTRAP = 10_000
E1A_N_BOOTSTRAP = 2_000
BOOTSTRAP_SEED = 20_260_817
SECTION_THRESHOLD_MM2 = 106.0

EXPECTED_CONVENTIONAL_SLIDES = 1_642
EXPECTED_CONVENTIONAL_PATIENTS = 1_486
EXPECTED_CONVENTIONAL_MUTANTS = 604
EXPECTED_ORION_SLIDES = 41
EXPECTED_ORION_PATIENTS = 40
EXPECTED_ORION_MUTANTS = 15
EXPECTED_SOURCE_SLIDES = 1_683
EXPECTED_SOURCE_PATIENTS = 1_526
EXPECTED_SOURCE_MUTANTS = 619
EXPECTED_SOURCE_MUTANT_SLIDES = 691
EXPECTED_PACK_SLIDES = 2_128
EXPECTED_FEATURE_DIM = 1_024

DATA_NAME = "aim1_e0_all_primary_orion"
SPLIT_NAME = "aim1_balanced5"
TARGETS: tuple[str, ...] = ("rih_m", "sr1482_m")
SCORE_DATASETS: tuple[str, ...] = ("source_refit", *TARGETS)

CONVENTIONAL_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
LABEL_SOURCE = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v5.csv")
CONVENTIONAL_SPLITS = REPO / "outputs/splits/aim1_1a/aim1_balanced5/splits.parquet"
CPHT_A_FOLDS = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_final_v8_complete_v1_20260822/cpht_a_v2/inputs/realized_folds.csv"
)
FINAL_V8_MET_INPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_final_v8_complete_v1_20260822/e2met/inputs/manifests"
)
PRIOR_CPHT_CONTRACT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim2_final_v8_complete_v1_20260822/cpht/inputs/execution_contract.json"
)
PRIOR_CPHT_SEAL = PRIOR_CPHT_CONTRACT.parents[1] / "inference_seal.json"
FEATURE_DIR = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/"
    "20x_256px_0px_overlap_mpp0.5/features_uni_v1"
)
PACK_DIR = FEATURE_DIR.parent / "packed_uni_v1"
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/"
    "aim1_e0_all_primary_orion_exp_v1_20260823"
)
PRODUCTION_RERUN_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns")
FINAL_V8_ROOT = PRODUCTION_RERUN_ROOT / "aim2_final_v8_complete_v1_20260822"
HISTORICAL_E0_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/train/1a_pb_cap8192/univ1"
)
GPU_LOCK_PATH = Path("/tmp/oceanpath_gpu0_exclusive.lock")
OUTCOME_SOURCES: dict[str, Path] = {
    "rih_m": Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_rih_metastatic.csv"),
    "sr1482_m": Path("/mnt/d/YC.Liu/manifests/colon/aim1_e2a_surgen_metastatic.csv"),
}
RIH_DUAL_PATIENTS: frozenset[str] = frozenset(
    {
        "RIH:RIH_001216ba7a08c070",
        "RIH:RIH_1845b46a817ef51c",
        "RIH:RIH_24bda4bdbf9140a5",
        "RIH:RIH_28ed5c0131dcfa60",
        "RIH:RIH_3c68f85359d030c4",
        "RIH:RIH_59b36f4590fc4525",
        "RIH:RIH_9db23d204671f3e8",
        "RIH:RIH_c1bac72156d3e1a8",
    }
)

MATERIAL_IMPLEMENTATION_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve(),
    REPO / "aim1_locked_baseline.py",
    REPO / "aim2_cross_protocol_transfer.py",
    REPO / "aim2_primary_to_metastatic_transfer.py",
    REPO / "scripts/train.py",
    REPO / "src/oceanpath/aim1/evaluate.py",
    REPO / "src/oceanpath/aim1/lineage.py",
    REPO / "src/oceanpath/aim1/paths.py",
    REPO / "src/oceanpath/aim1/registry.py",
    REPO / "src/oceanpath/aim1/technical.py",
    REPO / "src/oceanpath/aim1/cli/e1a_challenge.py",
    REPO / "src/oceanpath/eval/core.py",
    REPO / "src/oceanpath/eval/external.py",
    REPO / "src/oceanpath/workflows/training.py",
    REPO / "src/oceanpath/workflows/finalize.py",
    REPO / "src/oceanpath/config/__init__.py",
    REPO / "src/oceanpath/config/access.py",
    REPO / "src/oceanpath/config/paths.py",
    REPO / "src/oceanpath/contracts/__init__.py",
    REPO / "src/oceanpath/contracts/slide_ids.py",
    REPO / "src/oceanpath/contracts/stages.py",
    REPO / "src/oceanpath/datasets/__init__.py",
    REPO / "src/oceanpath/datasets/datamodule.py",
    REPO / "src/oceanpath/datasets/packed.py",
    REPO / "src/oceanpath/datasets/sampling.py",
    REPO / "src/oceanpath/training/lightning.py",
    REPO / "src/oceanpath/training/callbacks.py",
    REPO / "src/oceanpath/training/folds.py",
    REPO / "src/oceanpath/runtime/context.py",
    REPO / "src/oceanpath/runtime/reporting.py",
    REPO / "src/oceanpath/models/__init__.py",
    REPO / "src/oceanpath/models/abmil.py",
    REPO / "src/oceanpath/models/base.py",
    REPO / "src/oceanpath/models/components.py",
    REPO / "src/oceanpath/models/wsi_classifier.py",
    REPO / "src/oceanpath/splitting/core.py",
    REPO / "src/oceanpath/splitting/__init__.py",
    REPO / "configs/train.yaml",
    REPO / "configs/data/aim1.yaml",
    REPO / "configs/encoder/univ1.yaml",
    REPO / "configs/model/abmil.yaml",
    REPO / "configs/platform/colon_workstation.yaml",
    REPO / "configs/splits/aim1_balanced.yaml",
    REPO / "configs/training/aim1.yaml",
    REPO / "configs/training/default.yaml",
)

REFERENCE_CONFIG_ALLOWED_DIFFERENCES: frozenset[str] = frozenset(
    {
        "data.aim1_model",
        "data.csv_path",
        "data.manifest_stem",
        "data.name",
        "exp_name",
        "experiment.name",
        "platform.splits_root",
        "train_dir",
        "training.num_workers",
        "training.refit_max_steps",
        "wandb.group",
        "wandb.tags.0",
    }
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value), indent=2, sort_keys=True, allow_nan=False, default=str
    ) + "\n"


def _publish_text(path: Path, text: str) -> None:
    """Publish once; accept only a byte-identical existing artifact."""

    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace non-identical artifact: {path}")
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _publish_json(path: Path, value: Any) -> None:
    _publish_text(path, _canonical_json(value))


def _publish_csv(path: Path, frame: pd.DataFrame) -> None:
    _publish_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _publish_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.is_file():
        try:
            pd.testing.assert_frame_equal(
                pd.read_parquet(path), frame, check_exact=True, check_dtype=True
            )
        except AssertionError as exc:
            raise FileExistsError(f"Refusing to replace non-identical artifact: {path}") from exc
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _artifact(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], lineage.artifact_identity(path))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"Expected JSON object: {path}")
    return value


def _verify_artifact(record: Any, context: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
        raise ContractError(f"Malformed artifact identity: {context}")
    path = Path(str(record["path"]))
    if _artifact(path) != record:
        raise ContractError(f"Recorded artifact changed: {context}: {path}")
    return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _guard_output_root(output_root: Path, *, before_contract: bool = False) -> Path:
    """Restrict governed writes to a new additive rerun root or /tmp tests."""

    if not output_root.is_absolute():
        raise ContractError("--output-root must be an absolute path")
    lexical = output_root.absolute()
    resolved = output_root.resolve(strict=False)
    if lexical != resolved:
        raise ContractError(f"Output root must not traverse symlinks: {output_root} -> {resolved}")
    cursor = output_root
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise ContractError(f"Output-root path contains a symlink: {cursor}")
        cursor = cursor.parent

    protected = (
        FINAL_V8_ROOT.resolve(),
        Path("/mnt/d/YC.Liu").resolve(),
        HISTORICAL_E0_ROOT.parent.resolve(),
        FEATURE_DIR.parent.resolve(),
        REPO.resolve(),
    )
    for root in protected:
        if _is_relative_to(resolved, root) or _is_relative_to(root, resolved):
            raise ContractError(f"Output root overlaps protected canonical tree: {root}")

    in_tmp = _is_relative_to(resolved, Path("/tmp").resolve())
    in_reruns = resolved.parent == PRODUCTION_RERUN_ROOT.resolve()
    named_additive = resolved.name.startswith("aim1_e0_all_primary_orion_exp_")
    if not in_tmp and not (in_reruns and named_additive):
        raise ContractError(
            "Governed output must be a direct, newly named "
            f"{PRODUCTION_RERUN_ROOT}/aim1_e0_all_primary_orion_exp_* child (or /tmp for tests)"
        )

    if before_contract and output_root.exists() and not _contract_path(output_root).is_file():
        for entry in output_root.rglob("*"):
            if entry.is_symlink():
                raise ContractError(f"Pre-contract output contains a symlink: {entry}")
            if entry.is_file() and not _is_relative_to(entry.resolve(), _input_root(resolved)):
                raise ContractError(f"Unknown pre-contract artifact: {entry}")
    return resolved


def _validate_canonical_input_arguments(args: argparse.Namespace) -> None:
    """Allow input overrides only for isolated /tmp fixtures, never production."""

    output_root = Path(args.output_root).resolve(strict=False)
    if _is_relative_to(output_root, Path("/tmp").resolve()):
        return
    canonical = {
        "conventional_manifest": CONVENTIONAL_MANIFEST,
        "label_source": LABEL_SOURCE,
        "orion_folds": CPHT_A_FOLDS,
        "conventional_splits": CONVENTIONAL_SPLITS,
        "met_input_root": FINAL_V8_MET_INPUT_ROOT,
        "pack_dir": PACK_DIR,
    }
    mismatch = {
        name: {
            "expected": str(expected.resolve()),
            "observed": str(Path(getattr(args, name)).resolve(strict=False)),
        }
        for name, expected in canonical.items()
        if Path(getattr(args, name)).resolve(strict=False) != expected.resolve()
    }
    if mismatch:
        raise ContractError(
            "Production campaigns require the canonical frozen input paths; "
            f"overrides are test-only under /tmp: {mismatch}"
        )


def _implementation_identities() -> list[dict[str, Any]]:
    return [_artifact(path) for path in MATERIAL_IMPLEMENTATION_PATHS]


def _validate_implementation(records: Sequence[dict[str, Any]]) -> None:
    expected_paths = [str(path.resolve()) for path in MATERIAL_IMPLEMENTATION_PATHS]
    if [str(record.get("path")) for record in records] != expected_paths:
        raise ContractError("Material training implementation roster changed")
    for record in records:
        _verify_artifact(record, "material training implementation")


def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_config(item, name))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(_flatten_config(item, f"{prefix}.{index}"))
    else:
        flat[prefix] = value
    return flat


def _validate_against_historical_e0(value: Mapping[str, Any], seed: int) -> None:
    from omegaconf import OmegaConf

    historical_path = HISTORICAL_E0_ROOT / f"seed{seed}/config.yaml"
    historical = cast(
        Mapping[str, Any], OmegaConf.to_container(OmegaConf.load(historical_path), resolve=True)
    )
    reference = _flatten_config(historical)
    candidate = _flatten_config(value)
    differences = {
        key: {"historical": reference.get(key, "<MISSING>"), "candidate": candidate.get(key, "<MISSING>")}
        for key in sorted(set(reference) | set(candidate))
        if reference.get(key, "<MISSING>") != candidate.get(key, "<MISSING>")
    }
    unexpected = set(differences) - set(REFERENCE_CONFIG_ALLOWED_DIFFERENCES)
    if unexpected:
        raise ContractError(
            f"Seed {seed} config drifts from historical E0 outside the whitelist: "
            f"{ {key: differences[key] for key in sorted(unexpected)} }"
        )
    observed_allowed = set(differences)
    if observed_allowed != set(REFERENCE_CONFIG_ALLOWED_DIFFERENCES):
        missing = sorted(set(REFERENCE_CONFIG_ALLOWED_DIFFERENCES) - observed_allowed)
        extra = sorted(observed_allowed - set(REFERENCE_CONFIG_ALLOWED_DIFFERENCES))
        raise ContractError(
            f"Seed {seed} historical E0 difference roster changed; missing={missing}, extra={extra}"
        )


@contextmanager
def _exclusive_gpu_lock(operation: str) -> Any:
    GPU_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GPU_LOCK_PATH.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ContractError(
                f"GPU0 is already reserved by another OceanPath process; cannot start {operation}"
            ) from exc
        stream.seek(0)
        stream.truncate()
        stream.write(f"pid={os.getpid()} operation={operation} at={_utc_now()}\n")
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _cuda_idle_precheck() -> None:
    command = [
        "nvidia-smi",
        "--id=0",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ContractError(f"CUDA idle precheck failed: {result.stderr.strip()}")
    active = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if active:
        raise ContractError(f"GPU0 is not idle; active compute processes: {active}")


def _input_root(output_root: Path) -> Path:
    return output_root / "inputs"


def _source_path(output_root: Path) -> Path:
    return _input_root(output_root) / "source_primary.csv"


def _split_root(output_root: Path) -> Path:
    return _input_root(output_root) / "splits_root"


def _split_dir(output_root: Path) -> Path:
    return _split_root(output_root) / DATA_NAME / SPLIT_NAME


def _config_path(output_root: Path, seed: int) -> Path:
    return _input_root(output_root) / "configs" / f"seed{seed}.yaml"


def _target_path(output_root: Path, target: str) -> Path:
    return _input_root(output_root) / "targets" / f"{target}.csv"


def _score_manifest_path(output_root: Path, dataset: str) -> Path:
    if dataset == "source_refit":
        return _source_path(output_root)
    if dataset in TARGETS:
        return _target_path(output_root, dataset)
    raise ContractError(f"Unknown score dataset: {dataset}")


def _headline_roster_path(output_root: Path, target: str) -> Path:
    return _input_root(output_root) / "headline_rosters" / f"{target}.csv"


def _contract_path(output_root: Path) -> Path:
    return _input_root(output_root) / "experiment_contract.json"


def _run_dir(output_root: Path, seed: int) -> Path:
    return output_root / "train/pb_cap8192/all_primary_orion" / f"seed{seed}"


def _train_receipt_path(output_root: Path, seed: int) -> Path:
    return output_root / "receipts/training" / f"seed{seed}.json"


def _score_path(output_root: Path, dataset: str, seed: int) -> Path:
    return output_root / "scores" / f"all_primary_orion_{dataset}_seed{seed}.parquet"


def _score_receipt_path(output_root: Path, dataset: str, seed: int) -> Path:
    return _score_path(output_root, dataset, seed).with_suffix(".receipt.json")


def _calibrator_path(output_root: Path) -> Path:
    return output_root / "calibration/source_platt.json"


def _calibrator_table_path(output_root: Path) -> Path:
    return output_root / "calibration/source_oof_patient_logits.parquet"


def _model_seal_path(output_root: Path) -> Path:
    return output_root / "model_seal.json"


def _inference_seal_path(output_root: Path) -> Path:
    return output_root / "inference_seal.json"


def _analysis_path(output_root: Path) -> Path:
    return output_root / "analysis/results.json"


def _filter_orion(master: pd.DataFrame) -> pd.DataFrame:
    required = {
        "cohort",
        "include",
        "available",
        "used_kras",
        "qc_slides",
        "output_id",
        "patient_uid",
        "kras",
    }
    missing = required - set(master)
    if missing:
        raise ContractError(f"Orion label source lacks {sorted(missing)}")
    mask = (
        master["cohort"].eq("Orion")
        & master["include"].astype(str).str.lower().eq("yes")
        & master["available"].astype(str).str.lower().eq("yes")
        & master["used_kras"].astype(str).str.lower().eq("yes")
        & master["qc_slides"].astype(str).str.lower().eq("pass")
    )
    return master.loc[mask].copy()


def _build_orion_rows(
    master: pd.DataFrame,
    pack_index: pd.DataFrame,
    fold_frame: pd.DataFrame,
    *,
    section_threshold: float = SECTION_THRESHOLD_MM2,
) -> pd.DataFrame:
    """Build the 41-slide governed Orion addition without redrawing a fold."""

    rows = _filter_orion(master)
    if len(rows) != EXPECTED_ORION_SLIDES or rows["patient_uid"].nunique() != (
        EXPECTED_ORION_PATIENTS
    ):
        raise ContractError(
            f"Orion census is {len(rows)} slides/{rows['patient_uid'].nunique()} patients; "
            f"expected {EXPECTED_ORION_SLIDES}/{EXPECTED_ORION_PATIENTS}"
        )
    if rows["output_id"].duplicated().any():
        raise ContractError("Orion output_id is not slide-unique")

    required_fold = {"patient_id", "label", "outer_fold"}
    if required_fold - set(fold_frame):
        raise ContractError(f"CPHT-A fold table lacks {sorted(required_fold - set(fold_frame))}")
    folds = fold_frame.copy()
    folds["patient_id"] = folds["patient_id"].astype(str)
    folds["label"] = pd.to_numeric(folds["label"], errors="raise").astype(int)
    folds["outer_fold"] = pd.to_numeric(folds["outer_fold"], errors="raise").astype(int)
    if (
        len(folds) != EXPECTED_ORION_PATIENTS
        or folds["patient_id"].duplicated().any()
        or set(folds["outer_fold"]) != set(range(N_FOLDS))
    ):
        raise ContractError("CPHT-A fold roster is not the frozen 40-patient five-fold layout")
    census = folds.groupby(["outer_fold", "label"]).size().to_dict()
    expected = {(fold, label): (3 if label == 1 else 5) for fold in range(5) for label in (0, 1)}
    if census != expected:
        raise ContractError(f"CPHT-A fold x KRAS census changed: {census}")

    required_pack = {"slide_id", "n_patches"}
    if required_pack - set(pack_index):
        raise ContractError(f"Packed index lacks {sorted(required_pack - set(pack_index))}")
    patch_counts = (
        pack_index[["slide_id", "n_patches"]]
        .assign(slide_id=lambda frame: frame["slide_id"].astype(str))
        .set_index("slide_id")["n_patches"]
    )
    missing_pack = sorted(set(rows["output_id"].astype(str)) - set(patch_counts.index))
    if missing_pack:
        raise ContractError(f"Packed UNI lacks Orion slides: {missing_pack[:5]}")

    labels = rows["kras"].map({"wild_type": 0, "mutant": 1})
    if labels.isna().any():
        raise ContractError("Orion contains a non-binary KRAS status")
    out = pd.DataFrame(
        {
            "slide_id": rows["output_id"].astype(str),
            "patient_id": rows["patient_uid"].astype(str),
            "target_label": labels.astype(int),
            "kras": rows["kras"],
            "kras_subvariant": rows.get("kras_subvariant"),
            "cohort": "Orion",
            "subcohort": rows["subcohort"].fillna("Orion-CRC").astype(str),
            "specimen_role": rows["specimen_role"].astype(str),
            "msi_dmmr": rows.get("msi_dmmr"),
            "braf": rows.get("braf"),
            "nras": rows.get("nras"),
            "ras": rows.get("ras"),
            "tumor_site_group": rows.get("tumor_site_group"),
            "tumor_site_raw": rows.get("tumor_site_raw"),
            "stage_group_major": rows.get("stage_group_major"),
            "stage_group_major_filled": rows.get("stage_group_major_filled"),
            "sidedness": rows.get("sidedness"),
            "age_at_diagnosis": rows.get("age_at_diagnosis"),
            "sex": rows.get("sex"),
            "native_mpp": pd.to_numeric(rows["mpp"], errors="raise"),
        }
    )
    out["patch_count"] = out["slide_id"].map(patch_counts).astype(int)
    out["tissue_area_mm2"] = out["patch_count"] * (256 * 0.0005) ** 2
    out["mpp_bin"] = out["native_mpp"].map(mpp_bin)
    out["tissue_grid_occupancy"] = np.nan
    out["thumbnail_hue_median"] = np.nan
    out["section_size_class"] = np.where(
        out["tissue_area_mm2"] < section_threshold, "small_fragment", "large_section"
    )
    out["color_class"] = "unmeasured"
    out["technical_class"] = "Orion-CRC:" + out["mpp_bin"].astype(str)
    out["umap_island"] = np.nan
    out["site_class"] = out["tumor_site_group"].where(
        out["tumor_site_group"].isin(["Colon", "Rectum", "Appendix"]),
        "Other or unknown",
    )
    out["stage_class"] = np.select(
        [out["stage_group_major"].isin(["I", "II"]), out["stage_group_major"].isin(["III", "IV"])],
        ["I-II", "III-IV"],
        default="unknown",
    )
    n_slides = out.groupby("patient_id")["slide_id"].transform("nunique")
    out["slide_weight"] = 1.0 / n_slides
    out["n_slides_class"] = np.where(n_slides.eq(1), "1", "2+")

    out = out.merge(
        folds, on="patient_id", how="left", validate="many_to_one", suffixes=("", "_fold")
    )
    if out["outer_fold"].isna().any() or not out["target_label"].eq(out["label"]).all():
        raise ContractError("Orion labels/folds disagree with the frozen CPHT-A layout")
    out["k_fold"] = out.pop("outer_fold").astype(int)
    out = out.drop(columns=["label"])
    # Frozen deterministic inner validation: the next balanced CPHT-A block.
    for outer in range(N_FOLDS):
        out[f"val_fold_{outer}"] = out["k_fold"].eq((outer + 1) % N_FOLDS).astype(int)
    return out.sort_values("slide_id", kind="stable").reset_index(drop=True)


def _attach_derived_columns(conventional: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if {"sidedness", "stage_group_major_filled"} <= set(conventional):
        return conventional.copy()
    required = {"patient_uid", "sidedness", "stage_group_major_filled"}
    if required - set(master):
        raise ContractError(f"Label source lacks derived covariates {sorted(required - set(master))}")
    # Match E1a's frozen evaluation-side join: the governed master is ordered
    # deterministically and its first patient row supplies these two derived
    # covariates. Some dual-role patients legitimately have different specimen
    # sites; rejecting those would depart from the already-used E1a rule.
    derived = master[list(required)].drop_duplicates("patient_uid").rename(
        columns={"patient_uid": "patient_id"}
    )
    out = conventional.merge(derived, on="patient_id", how="left", validate="many_to_one")
    return out


def _combine_source(conventional: pd.DataFrame, orion: pd.DataFrame) -> pd.DataFrame:
    """Append Orion while preserving every conventional row/fold byte value."""

    conventional = conventional.copy()
    orion = orion.copy()
    if set(conventional["slide_id"].astype(str)) & set(orion["slide_id"].astype(str)):
        raise ContractError("Conventional and Orion slide IDs overlap")
    if set(conventional["patient_id"].astype(str)) & set(orion["patient_id"].astype(str)):
        raise ContractError("Conventional and Orion patient IDs overlap")
    columns = list(conventional.columns) + [c for c in orion.columns if c not in conventional]
    for column in columns:
        if column not in conventional:
            conventional[column] = np.nan
        if column not in orion:
            orion[column] = np.nan
    source = pd.concat([conventional[columns], orion[columns]], ignore_index=True)
    source["target_label"] = pd.to_numeric(source["target_label"], errors="raise").astype(int)
    source["k_fold"] = pd.to_numeric(source["k_fold"], errors="raise").astype(int)
    for fold in range(N_FOLDS):
        source[f"val_fold_{fold}"] = pd.to_numeric(
            source[f"val_fold_{fold}"], errors="raise"
        ).astype(int)
    if source["slide_id"].duplicated().any():
        raise ContractError("Combined source has duplicate slide IDs")
    labels = source.groupby("patient_id")["target_label"].nunique()
    if not labels.eq(1).all():
        raise ContractError("Combined source has within-patient KRAS disagreement")
    patient_folds = source.groupby("patient_id")["k_fold"].nunique()
    if not patient_folds.eq(1).all():
        offenders = patient_folds[patient_folds.ne(1)].index.astype(str).tolist()[:5]
        raise ContractError(f"Combined source has patients spanning outer folds: {offenders}")
    for fold in range(N_FOLDS):
        patient_flags = source.groupby("patient_id")[f"val_fold_{fold}"].nunique()
        if not patient_flags.eq(1).all():
            offenders = patient_flags[patient_flags.ne(1)].index.astype(str).tolist()[:5]
            raise ContractError(
                f"Combined source has patients spanning val_fold_{fold} membership: {offenders}"
            )
    if set(source["specimen_role"].dropna().astype(str)) != {"primary"}:
        raise ContractError("Combined source is not primary-only")
    patient = source.drop_duplicates("patient_id")
    observed = (
        len(source),
        len(patient),
        int(patient["target_label"].sum()),
        int(source["target_label"].sum()),
    )
    expected = (
        EXPECTED_SOURCE_SLIDES,
        EXPECTED_SOURCE_PATIENTS,
        EXPECTED_SOURCE_MUTANTS,
        EXPECTED_SOURCE_MUTANT_SLIDES,
    )
    if observed != expected:
        raise ContractError(f"Combined source census is {observed}, expected {expected}")
    for fold in range(N_FOLDS):
        if set(source["k_fold"].astype(int)) != set(range(N_FOLDS)):
            raise ContractError("Combined source has an invalid outer fold")
        flags = source[f"val_fold_{fold}"]
        if not flags.isin([0, 1]).all() or bool(flags[source["k_fold"].eq(fold)].any()):
            raise ContractError(f"Combined source has invalid/leaking val_fold_{fold}")
        spanning = source.loc[~source["k_fold"].eq(fold)].groupby("patient_id")[
            f"val_fold_{fold}"
        ].nunique()
        if bool(spanning.gt(1).any()):
            raise ContractError(f"A patient spans the train/val boundary for fold {fold}")
    return source.sort_values(["cohort", "subcohort", "patient_id", "slide_id"], kind="stable").reset_index(
        drop=True
    )


def _load_source_inputs(
    conventional_manifest: Path = CONVENTIONAL_MANIFEST,
    label_source: Path = LABEL_SOURCE,
    orion_folds: Path = CPHT_A_FOLDS,
    pack_dir: Path = PACK_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conventional_raw = pd.read_csv(conventional_manifest, low_memory=False)
    conventional = _attach_derived_columns(
        conventional_raw, pd.read_csv(label_source, low_memory=False)
    )
    patient = conventional.drop_duplicates("patient_id")
    observed = (len(conventional), len(patient), int(patient["target_label"].sum()))
    expected = (
        EXPECTED_CONVENTIONAL_SLIDES,
        EXPECTED_CONVENTIONAL_PATIENTS,
        EXPECTED_CONVENTIONAL_MUTANTS,
    )
    if observed != expected:
        raise ContractError(f"Conventional source census is {observed}, expected {expected}")

    master = pd.read_csv(label_source, low_memory=False)
    pack_index = pd.read_parquet(pack_dir / "index.parquet")
    fold_frame = pd.read_csv(orion_folds)
    orion = _build_orion_rows(master, pack_index, fold_frame)
    source = _combine_source(conventional, orion)

    # The append is not allowed to alter any frozen conventional assignment.
    shared = list(conventional_raw.columns)
    before = conventional_raw.sort_values("slide_id", kind="stable").reset_index(drop=True)
    after = (
        source[source["cohort"].ne("Orion")][shared]
        .sort_values("slide_id", kind="stable")
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(before, after, check_exact=True, check_dtype=False)
    except AssertionError as exc:
        raise ContractError("Appending Orion changed the frozen conventional manifest") from exc
    return source, conventional, orion


def _validate_frozen_conventional_splits(
    conventional: pd.DataFrame, frozen_split_path: Path = CONVENTIONAL_SPLITS
) -> None:
    frozen = pd.read_parquet(frozen_split_path)
    columns = ["slide_id", "k_fold", *(f"val_fold_{fold}" for fold in range(N_FOLDS))]
    if len(frozen) != EXPECTED_CONVENTIONAL_SLIDES or set(columns) - set(frozen):
        raise ContractError("Frozen conventional split artifact is incomplete")
    left = conventional[columns].sort_values("slide_id", kind="stable").reset_index(drop=True)
    right = frozen[columns].sort_values("slide_id", kind="stable").reset_index(drop=True)
    for column in columns[1:]:
        left[column] = pd.to_numeric(left[column], errors="raise").astype(int)
        right[column] = pd.to_numeric(right[column], errors="raise").astype(int)
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)
    except AssertionError as exc:
        raise ContractError("Conventional fold/validation columns differ from frozen splits") from exc


def _load_label_blind_targets(met_input_root: Path = FINAL_V8_MET_INPUT_ROOT) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    forbidden = {"target_label", "label", "kras", "ras", "nras", "braf", "msi_dmmr"}
    expected = {"rih_m": (85, 85), "sr1482_m": (100, 74)}
    for target in TARGETS:
        path = met_input_root / f"{target}.csv"
        frame = pd.read_csv(path, low_memory=False)
        required = {"slide_id", "patient_id", "specimen_role", "subcohort", "liver_class"}
        if required - set(frame):
            raise ContractError(f"Canonical {target} manifest lacks {sorted(required - set(frame))}")
        if forbidden & set(frame):
            raise ContractError(f"Canonical {target} manifest contains outcome columns")
        if frame["slide_id"].duplicated().any() or set(frame["specimen_role"].astype(str)) != {
            "metastatic"
        }:
            raise ContractError(f"Canonical {target} manifest is not unique metastatic slides")
        observed = (len(frame), frame["patient_id"].nunique())
        if observed != expected[target]:
            raise ContractError(f"Canonical {target} census is {observed}, expected {expected[target]}")
        frames[target] = frame.sort_values("slide_id", kind="stable").reset_index(drop=True)
    return frames


def _validate_target_overlap(
    source: pd.DataFrame, targets: Mapping[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Freeze exact direct-patient leakage rules before any fit can start."""

    if set(targets) != set(TARGETS):
        raise ContractError(f"Target roster is {sorted(targets)}, expected {sorted(TARGETS)}")
    source_ids = set(source["patient_id"].astype(str))
    rih = targets["rih_m"]
    sr = targets["sr1482_m"]
    rih_ids = set(rih["patient_id"].astype(str))
    sr_ids = set(sr["patient_id"].astype(str))
    if source_ids & rih_ids != set(RIH_DUAL_PATIENTS):
        raise ContractError(
            "Source/RIH-M overlap is not exactly the frozen eight dual-role patients"
        )
    if source_ids & sr_ids:
        raise ContractError("Source unexpectedly overlaps SR1482-M patient IDs")
    headline = {
        "rih_m": rih[~rih["patient_id"].astype(str).isin(RIH_DUAL_PATIENTS)]
        .sort_values("slide_id", kind="stable")
        .reset_index(drop=True),
        "sr1482_m": sr.sort_values("slide_id", kind="stable").reset_index(drop=True),
    }
    observed = {
        target: (len(frame), int(frame["patient_id"].nunique()))
        for target, frame in headline.items()
    }
    expected = {"rih_m": (77, 77), "sr1482_m": (100, 74)}
    if observed != expected:
        raise ContractError(f"Leakage-free target census is {observed}, expected {expected}")
    return headline


def _validate_pack_roster(
    source: pd.DataFrame,
    targets: Mapping[str, pd.DataFrame],
    pack_dir: Path = PACK_DIR,
) -> dict[str, Any]:
    meta_path = pack_dir / "meta.json"
    index_path = pack_dir / "index.parquet"
    meta = _read_json(meta_path)
    index = pd.read_parquet(index_path)
    expected_meta = {"n_slides": EXPECTED_PACK_SLIDES, "feat_dim": EXPECTED_FEATURE_DIM}
    mismatch = {key: (value, meta.get(key)) for key, value in expected_meta.items() if meta.get(key) != value}
    if mismatch:
        raise ContractError(f"Packed UNI metadata mismatch: {mismatch}")
    packed = set(index["slide_id"].astype(str))
    required = set(source["slide_id"].astype(str))
    required.update(*(set(frame["slide_id"].astype(str)) for frame in targets.values()))
    missing = sorted(required - packed)
    if missing:
        raise ContractError(f"Packed UNI lacks experiment slides: {missing[:10]}")
    # final-v8 already sealed the multi-gigabyte payload hashes.  Bind that
    # immutable contract rather than re-hashing 49 GB during every preflight.
    prior = _read_json(PRIOR_CPHT_CONTRACT)
    frozen_store = prior.get("feature_store") or {}
    if str(pack_dir.resolve()) != frozen_store.get("path"):
        raise ContractError("Packed UNI root differs from the final-v8 sealed store")
    if str(FEATURE_DIR.resolve()) != frozen_store.get("source_dir"):
        raise ContractError("UNI feature source differs from the final-v8 sealed store")
    for key, name in (("meta", "meta.json"), ("index", "index.parquet")):
        if frozen_store.get(key) != _artifact(pack_dir / name):
            raise ContractError(f"Packed UNI {key} differs from the final-v8 sealed store")
    return {
        "path": str(pack_dir.resolve()),
        "meta": _artifact(meta_path),
        "index": _artifact(index_path),
        "n_slides": int(meta["n_slides"]),
        "feature_dim": int(meta["feat_dim"]),
        "feature_dtype": meta.get("feat_dtype"),
        "source_dir": frozen_store["source_dir"],
        "source_inventory_sha256": meta.get("source_inventory_sha256"),
        "prior_payload_contract": _artifact(PRIOR_CPHT_CONTRACT),
        "prior_inference_seal": _artifact(PRIOR_CPHT_SEAL),
    }


def _compose_training_cfg(
    seed: int,
    source_manifest_path: Path,
    splits_root: Path,
    train_dir: Path,
) -> Any:
    """Compose the exact E0 cap8192 recipe, changing only source and workers."""

    from hydra import compose, initialize_config_dir

    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        "data.aim1_model=e0_all_primary_orion",
        f"data.name={DATA_NAME}",
        "data.manifest_stem=source_primary",
        f"data.csv_path={source_manifest_path.resolve()}",
        f"platform.splits_root={splits_root.resolve()}",
        "+data.cohort_column=cohort",
        "encoder=univ1",
        "splits=aim1_balanced",
        f"splits.seed={seed}",
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
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"training.num_workers={TRAINING_NUM_WORKERS}",
        "training.refit_epoch_rule=p75",
        "training.refit_max_steps=null",
        f"train_dir={train_dir.resolve()}",
        f"exp_name=e0_all_primary_orion_c{CAP}_s{seed}",
    ]
    with initialize_config_dir(config_dir=str((REPO / "configs").resolve()), version_base="1.3"):
        return compose(config_name="train", overrides=overrides)


def _nested(record: Mapping[str, Any], dotted: str) -> Any:
    value: Any = record
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ContractError(f"Missing configuration field {dotted!r}")
        value = value[key]
    return value


def _validate_training_config(value_or_path: Mapping[str, Any] | Path, seed: int, output_root: Path) -> None:
    if isinstance(value_or_path, Path):
        value = yaml.safe_load(value_or_path.read_text(encoding="utf-8"))
    else:
        value = value_or_path
    if not isinstance(value, Mapping):
        raise ContractError("Resolved training configuration is not a mapping")
    expected = {
        "data.name": DATA_NAME,
        "data.csv_path": str(_source_path(output_root).resolve()),
        "data.num_classes": 2,
        "encoder.name": "uni_v1",
        "encoder.feature_dim": EXPECTED_FEATURE_DIM,
        "model.name": "abmil",
        "model.arch": "abmil",
        "model.embed_dim": 512,
        "model.attn_dim": 384,
        "model.gate": True,
        "model.dropout": 0.25,
        "model.input_dropout": 0.1,
        "training.lr": 1e-4,
        "training.weight_decay": 1e-5,
        "training.lr_scheduler": "cosine",
        "training.final_lr_fraction": 0.01,
        "training.warmup_epochs": 0,
        "training.max_epochs": 20,
        "training.batch_size": 1,
        "training.accumulate_grad_batches": 1,
        "training.loss_type": "bce",
        "training.training_class_weighted_loss": False,
        "training.validation_loss_weighted": False,
        "training.class_weights": None,
        "training.sample_weight_column": None,
        "training.train_sampling_strategy": "patient_natural",
        "training.dataset_max_instances": CAP,
        "training.eval_full_bags": True,
        "training.num_workers": TRAINING_NUM_WORKERS,
        "training.monitor_metric": "val/patient_auroc",
        "training.monitor_mode": "max",
        "training.early_stopping_patience": 5,
        "training.min_epoch_before_stop": 10,
        "training.refit_epoch_rule": "p75",
        "training.refit_max_steps": None,
        "platform.precision": "bf16-mixed",
        "splits.scheme": "predefined_oof_kfold",
        "splits.name": SPLIT_NAME,
        "splits.n_folds": N_FOLDS,
        "splits.fold_column": "k_fold",
        "training.seed": seed,
        "train_dir": str(_run_dir(output_root, seed).resolve()),
    }
    mismatch = {
        key: {"expected": wanted, "observed": _nested(value, key)}
        for key, wanted in expected.items()
        if _nested(value, key) != wanted
    }
    if mismatch:
        raise ContractError(f"Locked E0 recipe mismatch for seed {seed}: {mismatch}")
    _validate_against_historical_e0(value, seed)


def _ensure_splits(output_root: Path) -> Path:
    from oceanpath.splitting import SplitConfig, generate_splits

    directory = _split_dir(output_root)
    split_path = directory / "splits.parquet"
    spec = SplitConfig(
        scheme="predefined_oof_kfold",
        name=SPLIT_NAME,
        csv_path=str(_source_path(output_root).resolve()),
        output_dir=str(directory.resolve()),
        filename_column="slide_id",
        label_column="target_label",
        group_column="patient_id",
        fold_column="k_fold",
        n_folds=N_FOLDS,
        seed=SEEDS[0],
    )
    generate_splits(spec, force=False)
    if not split_path.is_file():
        raise ContractError(f"Split generator did not publish {split_path}")
    return split_path


def _fold_census(source: pd.DataFrame) -> dict[str, Any]:
    patient = source.drop_duplicates("patient_id")
    out: dict[str, Any] = {}
    for fold in range(N_FOLDS):
        test = patient["k_fold"].eq(fold)
        val = patient[f"val_fold_{fold}"].eq(1)
        train = ~(test | val)
        out[str(fold)] = {
            "train_patients": int(train.sum()),
            "validation_patients": int(val.sum()),
            "test_patients": int(test.sum()),
            "orion_validation": patient.loc[val & patient["cohort"].eq("Orion"), "target_label"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "orion_test": patient.loc[test & patient["cohort"].eq("Orion"), "target_label"]
            .value_counts()
            .sort_index()
            .to_dict(),
        }
    return out


def _build_manifest_contract(
    output_root: Path,
    *,
    conventional_manifest: Path,
    label_source: Path,
    orion_folds: Path,
    conventional_splits: Path,
    met_input_root: Path,
    pack_dir: Path,
) -> dict[str, Any]:
    source = pd.read_csv(_source_path(output_root), low_memory=False)
    targets = {target: pd.read_csv(_target_path(output_root, target)) for target in TARGETS}
    headlines = _validate_target_overlap(source, targets)
    for target, frame in headlines.items():
        published = pd.read_csv(_headline_roster_path(output_root, target))
        try:
            pd.testing.assert_frame_equal(published, frame, check_exact=True, check_dtype=False)
        except AssertionError as exc:
            raise ContractError(f"Published headline roster differs for {target}") from exc
    pack = _validate_pack_roster(source, targets, pack_dir)
    configs = {str(seed): _artifact(_config_path(output_root, seed)) for seed in SEEDS}
    return {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "experiment": "exploratory Aim-1 E0 all primary plus Orion",
        "lineage": output_root.name,
        "analysis_role": "target-primary-exposed deployment sensitivity; not external transport",
        "output_root": str(output_root.resolve()),
        "implementation": _artifact(Path(__file__).resolve()),
        "material_training_implementation": _implementation_identities(),
        "historical_e0_configs": {
            str(seed): _artifact(HISTORICAL_E0_ROOT / f"seed{seed}/config.yaml")
            for seed in SEEDS
        },
        "final_v8_is_read_only": True,
        "source": {
            "manifest": _artifact(_source_path(output_root)),
            "n_slides": EXPECTED_SOURCE_SLIDES,
            "n_patients": EXPECTED_SOURCE_PATIENTS,
            "n_mutant_patients": EXPECTED_SOURCE_MUTANTS,
            "n_mutant_slides": EXPECTED_SOURCE_MUTANT_SLIDES,
            "components": {
                "conventional_manifest": _artifact(conventional_manifest),
                "governed_label_source": _artifact(label_source),
                "frozen_conventional_splits": _artifact(conventional_splits),
                "orion_cpht_a_folds": _artifact(orion_folds),
            },
            "splits": _artifact(_split_dir(output_root) / "splits.parquet"),
            "fold_census": _fold_census(source),
        },
        "recipe": {
            "arm": "E0",
            "encoder": "UNIv1",
            "aggregator": "gated ABMIL",
            "cap": CAP,
            "sampler": "patient_natural",
            "seeds": list(SEEDS),
            "outer_folds": N_FOLDS,
            "fit_count": len(SEEDS) * (N_FOLDS + 1),
            "refit_rule": "p75 of fold stopping epochs; no fixed optimizer-step budget",
            "training_num_workers": TRAINING_NUM_WORKERS,
            "gpu_execution": {
                "device": "cuda:0",
                "host_wide_lock": str(GPU_LOCK_PATH.resolve()),
                "concurrent_training_processes": 1,
                "cuda_idle_precheck": True,
            },
            "configs": configs,
        },
        "targets": {
            target: {
                "manifest": _artifact(_target_path(output_root, target)),
                "canonical_source": _artifact(met_input_root / f"{target}.csv"),
                "contains_target_outcomes": False,
                "n_slides": int(len(frame)),
                "n_patients": int(frame["patient_id"].nunique()),
                "source_patient_overlap": (
                    sorted(RIH_DUAL_PATIENTS) if target == "rih_m" else []
                ),
                "headline_roster": _artifact(_headline_roster_path(output_root, target)),
                "headline_n_slides": int(len(headlines[target])),
                "headline_n_patients": int(headlines[target]["patient_id"].nunique()),
            }
            for target, frame in targets.items()
        },
        "outcome_sources_not_opened_by_train_or_score": {
            target: {
                "artifact": _artifact(path),
                "filter": "subcohort == SR1482" if target == "sr1482_m" else None,
            }
            for target, path in OUTCOME_SOURCES.items()
        },
        "rih_headline_exclusions": sorted(RIH_DUAL_PATIENTS),
        "feature_store": pack,
        "reporting": {
            "source": "three honest seed OOF runs; E0 plus E1a core diagnostics",
            "calibration": (
                "one source Platt map fit to mean three-seed honest OOF patient logits; "
                "target Brier/log-loss/calibration are exploratory deployment-calibration "
                "sensitivities, not externally validated calibration"
            ),
            "refit_scale": (
                "per-seed and three-seed patient native-logit SD ratios: full-source p75 refit "
                "divided by honest OOF"
            ),
            "metastatic": "three-seed mean native refit logits; RIH77 and SR1482-M74 headline",
            "rih85": "explicit contaminated target-primary-exposed sensitivity only",
            "bootstrap": {
                "n": N_BOOTSTRAP,
                "seed": BOOTSTRAP_SEED,
                "method": "fixed-class-count KRAS-stratified patient resampling",
            },
        },
    }


def _load_contract(output_root: Path, *, verify_outcomes: bool = False) -> dict[str, Any]:
    _guard_output_root(output_root)
    contract = _read_json(_contract_path(output_root))
    expected = {
        "schema_version": 1,
        "experiment": "exploratory Aim-1 E0 all primary plus Orion",
        "lineage": output_root.name,
        "final_v8_is_read_only": True,
    }
    mismatch = {key: (wanted, contract.get(key)) for key, wanted in expected.items() if contract.get(key) != wanted}
    if mismatch:
        raise ContractError(f"Experiment contract header mismatch: {mismatch}")
    if contract.get("implementation") != _artifact(Path(__file__).resolve()):
        raise ContractError("Runner implementation changed after manifest seal")
    if contract.get("output_root") != str(output_root.resolve()):
        raise ContractError("Experiment contract was moved to a different output root")
    _validate_implementation(contract.get("material_training_implementation") or [])
    historical = contract.get("historical_e0_configs") or {}
    if set(historical) != {str(seed) for seed in SEEDS}:
        raise ContractError("Historical E0 reference config roster is incomplete")
    for seed in SEEDS:
        path = HISTORICAL_E0_ROOT / f"seed{seed}/config.yaml"
        if historical[str(seed)] != _artifact(path):
            raise ContractError(f"Historical E0 seed {seed} config changed after seal")
    source = contract.get("source") or {}
    expected_source_census = {
        "n_slides": EXPECTED_SOURCE_SLIDES,
        "n_patients": EXPECTED_SOURCE_PATIENTS,
        "n_mutant_patients": EXPECTED_SOURCE_MUTANTS,
        "n_mutant_slides": EXPECTED_SOURCE_MUTANT_SLIDES,
    }
    source_mismatch = {
        key: {"expected": value, "observed": source.get(key)}
        for key, value in expected_source_census.items()
        if source.get(key) != value
    }
    if source_mismatch:
        raise ContractError(f"Source census contract changed: {source_mismatch}")
    _verify_artifact(source.get("manifest"), "source manifest")
    _verify_artifact(source.get("splits"), "combined splits")
    for name, identity in (source.get("components") or {}).items():
        _verify_artifact(identity, f"source component/{name}")
    recipe = contract.get("recipe") or {}
    if recipe.get("training_num_workers") != TRAINING_NUM_WORKERS:
        raise ContractError("Training worker count changed after seal")
    expected_gpu = {
        "device": "cuda:0",
        "host_wide_lock": str(GPU_LOCK_PATH.resolve()),
        "concurrent_training_processes": 1,
        "cuda_idle_precheck": True,
    }
    if recipe.get("gpu_execution") != expected_gpu:
        raise ContractError("GPU serialization/idle-check contract changed after seal")
    for seed in SEEDS:
        path = _verify_artifact((recipe.get("configs") or {}).get(str(seed)), f"config/seed{seed}")
        _validate_training_config(path, seed, output_root)
    for target in TARGETS:
        record = (contract.get("targets") or {}).get(target) or {}
        _verify_artifact(record.get("manifest"), f"target/{target}")
        _verify_artifact(record.get("canonical_source"), f"canonical target/{target}")
        _verify_artifact(record.get("headline_roster"), f"headline target/{target}")
        expected_target = (
            {
                "n_slides": 85,
                "n_patients": 85,
                "headline_n_slides": 77,
                "headline_n_patients": 77,
                "source_patient_overlap": sorted(RIH_DUAL_PATIENTS),
            }
            if target == "rih_m"
            else {
                "n_slides": 100,
                "n_patients": 74,
                "headline_n_slides": 100,
                "headline_n_patients": 74,
                "source_patient_overlap": [],
            }
        )
        target_mismatch = {
            key: {"expected": value, "observed": record.get(key)}
            for key, value in expected_target.items()
            if record.get(key) != value
        }
        if target_mismatch:
            raise ContractError(f"Target contract changed for {target}: {target_mismatch}")
    if verify_outcomes:
        for target, record in (
            contract.get("outcome_sources_not_opened_by_train_or_score") or {}
        ).items():
            _verify_artifact(record.get("artifact"), f"outcome source/{target}")
    feature = contract.get("feature_store") or {}
    if feature.get("path") != str(PACK_DIR.resolve()) or feature.get("source_dir") != str(
        FEATURE_DIR.resolve()
    ):
        raise ContractError("Sealed UNI feature/pack roots changed")
    for key in ("meta", "index", "prior_payload_contract", "prior_inference_seal"):
        _verify_artifact(feature.get(key), f"feature store/{key}")
    if set(contract.get("rih_headline_exclusions") or []) != set(RIH_DUAL_PATIENTS):
        raise ContractError("RIH headline exclusion roster changed")
    source_frame = pd.read_csv(_source_path(output_root), low_memory=False)
    target_frames = {
        target: pd.read_csv(_target_path(output_root, target), low_memory=False)
        for target in TARGETS
    }
    headline_frames = _validate_target_overlap(source_frame, target_frames)
    for target, frame in headline_frames.items():
        published = pd.read_csv(_headline_roster_path(output_root, target), low_memory=False)
        try:
            pd.testing.assert_frame_equal(published, frame, check_exact=True, check_dtype=False)
        except AssertionError as exc:
            raise ContractError(f"Headline roster changed after seal: {target}") from exc
    return contract


def _validate_generated_splits(output_root: Path) -> None:
    source = pd.read_csv(_source_path(output_root), low_memory=False)
    split = pd.read_parquet(_split_dir(output_root) / "splits.parquet")
    if len(split) != len(source) or set(split["slide_id"].astype(str)) != set(
        source["slide_id"].astype(str)
    ):
        raise ContractError("Generated split and source slide rosters differ")
    columns = ["slide_id", "k_fold", *(f"val_fold_{fold}" for fold in range(N_FOLDS))]
    left = source[columns].sort_values("slide_id", kind="stable").reset_index(drop=True)
    right = split[columns].sort_values("slide_id", kind="stable").reset_index(drop=True)
    for column in columns[1:]:
        left[column] = pd.to_numeric(left[column], errors="raise").astype(int)
        right[column] = pd.to_numeric(right[column], errors="raise").astype(int)
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True, check_dtype=False)
    except AssertionError as exc:
        raise ContractError("Split generator did not preserve predefined fold columns") from exc


def _validate_oof_frame(path: Path, source: pd.DataFrame, seed: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"slide_id", "label", "logit", "fold"}
    if required - set(frame):
        raise ContractError(f"Seed {seed} OOF lacks {sorted(required - set(frame))}")
    if frame["slide_id"].duplicated().any():
        raise ContractError(f"Seed {seed} OOF has duplicate slide IDs")
    if set(frame["slide_id"].astype(str)) != set(source["slide_id"].astype(str)) or len(frame) != len(
        source
    ):
        raise ContractError(f"Seed {seed} OOF does not cover all {len(source)} source slides")
    if not np.isfinite(pd.to_numeric(frame["logit"], errors="coerce")).all():
        raise ContractError(f"Seed {seed} OOF has non-finite native logits")
    if set(pd.to_numeric(frame["fold"], errors="raise").astype(int)) != set(range(N_FOLDS)):
        raise ContractError(f"Seed {seed} OOF does not contain exactly folds 0..4")
    truth = source.assign(slide_id=source["slide_id"].astype(str)).set_index("slide_id")[
        "target_label"
    ]
    observed = frame.assign(slide_id=frame["slide_id"].astype(str)).set_index("slide_id")[
        "label"
    ]
    ordered_ids = sorted(truth.index)
    if not np.array_equal(
        observed.loc[ordered_ids].to_numpy(dtype=int), truth.loc[ordered_ids].to_numpy(dtype=int)
    ):
        raise ContractError(f"Seed {seed} OOF labels disagree with source")
    expected_folds = source.assign(slide_id=source["slide_id"].astype(str)).set_index("slide_id")[
        "k_fold"
    ]
    observed_folds = frame.assign(slide_id=frame["slide_id"].astype(str)).set_index("slide_id")[
        "fold"
    ]
    if not np.array_equal(
        observed_folds.loc[ordered_ids].to_numpy(dtype=int),
        expected_folds.loc[ordered_ids].to_numpy(dtype=int),
    ):
        raise ContractError(f"Seed {seed} OOF fold identities disagree with frozen source folds")
    return frame


def _validate_root_oof_against_folds(directory: Path, seed: int) -> None:
    """Authenticate the unhashed root OOF by reconstructing it from hashed folds."""

    root_path = directory / "oof_predictions.parquet"
    root = pd.read_parquet(root_path)
    fold_frames = []
    for fold in range(N_FOLDS):
        path = directory / f"fold_{fold}/preds_test.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        raw_columns = ["slide_id", "label", "prob_1", "logit"]
        if list(frame.columns) != raw_columns:
            raise ContractError(f"Seed {seed} fold {fold} test prediction schema changed")
        frame = frame.copy()
        frame["fold"] = fold
        fold_frames.append(frame)
    reconstructed = pd.concat(fold_frames, ignore_index=True)
    expected_columns = ["slide_id", "label", "prob_1", "logit", "fold"]
    if list(root.columns) != expected_columns or list(reconstructed.columns) != expected_columns:
        raise ContractError(
            f"Seed {seed} OOF schema differs from authenticated fold predictions"
        )
    root = root.sort_values("slide_id", kind="stable").reset_index(drop=True)
    reconstructed = reconstructed.sort_values("slide_id", kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(root, reconstructed, check_exact=True, check_dtype=True)
    except AssertionError as exc:
        raise ContractError(
            f"Seed {seed} root OOF differs from authenticated fold prediction artifacts"
        ) from exc


def _training_record(output_root: Path, seed: int) -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    from oceanpath.workflows.training import (
        training_run_fingerprint,
        validate_training_run_dir,
    )

    directory = _run_dir(output_root, seed)
    required = {
        "completion": directory / "training_completion.json",
        "config": directory / "config.yaml",
        "oof": directory / "oof_predictions.parquet",
        "cv_summary": directory / "cv_summary.json",
        "refit_info": directory / "final/refit/info.json",
        "refit_checkpoint": directory / "final/refit/model.ckpt",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    completion = _read_json(required["completion"])
    if completion.get("status") != "completed" or completion.get("n_folds") != N_FOLDS:
        raise ContractError(f"Seed {seed} training completion is incomplete")
    frozen_cfg = OmegaConf.load(_config_path(output_root, seed))
    expected_fingerprint = training_run_fingerprint(frozen_cfg)
    try:
        validate_training_run_dir(
            directory,
            expected_fingerprint=expected_fingerprint,
            require_test_predictions=True,
        )
    except RuntimeError as exc:
        raise ContractError(f"Seed {seed} native training-run validation failed") from exc
    if completion.get("training_fingerprint") != expected_fingerprint:
        raise ContractError(f"Seed {seed} completion has the wrong semantic fingerprint")
    _validate_training_config(required["config"], seed, output_root)
    frozen_identity = _artifact(_config_path(output_root, seed))
    run_identity = _artifact(required["config"])
    if (run_identity["sha256"], run_identity["size_bytes"]) != (
        frozen_identity["sha256"],
        frozen_identity["size_bytes"],
    ):
        raise ContractError(f"Seed {seed} run config bytes differ from the frozen input config")
    source = pd.read_csv(_source_path(output_root), low_memory=False)
    _validate_oof_frame(required["oof"], source, seed)
    _validate_root_oof_against_folds(directory, seed)
    info = _read_json(required["refit_info"])
    if (
        info.get("strategy") != "refit"
        or info.get("refit_epoch_rule") != "p75"
        or not isinstance(info.get("refit_epochs"), int)
        or int(info["refit_epochs"]) <= 0
        or len(info.get("fold_best_epochs") or []) != N_FOLDS
        or int(info.get("n_train_slides", -1)) != EXPECTED_SOURCE_SLIDES
    ):
        raise ContractError(f"Seed {seed} does not contain the governed p75 refit")
    fold_best_epochs = []
    for fold in range(N_FOLDS):
        metrics = _read_json(directory / f"fold_{fold}/fold_metrics.json")
        best_epoch = metrics.get("best_epoch")
        if not isinstance(best_epoch, (int, float)) or best_epoch <= 0:
            raise ContractError(f"Seed {seed} fold {fold} has no valid best_epoch")
        fold_best_epochs.append(int(best_epoch))
    expected_refit_epochs = int(np.ceil(np.percentile(fold_best_epochs, 75)))
    if [int(value) for value in info["fold_best_epochs"]] != fold_best_epochs or int(
        info["refit_epochs"]
    ) != expected_refit_epochs:
        raise ContractError(
            f"Seed {seed} p75 refit mismatch: folds={fold_best_epochs}, "
            f"expected={expected_refit_epochs}, observed={info.get('refit_epochs')}"
        )
    expected_slide_labels = {
        str(label): int(count)
        for label, count in source["target_label"].astype(int).value_counts().sort_index().items()
    }
    if {str(key): int(value) for key, value in (info.get("label_counts") or {}).items()} != (
        expected_slide_labels
    ):
        raise ContractError(f"Seed {seed} refit label census differs from the source manifest")
    checkpoint = torch.load(
        required["refit_checkpoint"], map_location="cpu", weights_only=False
    )
    expected_steps = expected_refit_epochs * EXPECTED_SOURCE_PATIENTS
    if int(checkpoint.get("global_step", -1)) != expected_steps:
        raise ContractError(
            f"Seed {seed} refit global_step is {checkpoint.get('global_step')}, "
            f"expected {expected_steps}"
        )
    return {
        "seed": seed,
        "fit_count": N_FOLDS + 1,
        "optimizer_step_convention": "patient_natural: one patient per optimizer step",
        "refit_epoch_rule": "p75",
        "training_fingerprint": expected_fingerprint,
        "refit_epochs": expected_refit_epochs,
        "refit_optimizer_steps": expected_steps,
        "checkpoint_global_step": int(checkpoint["global_step"]),
        "fold_best_epochs": fold_best_epochs,
        **{name: _artifact(path) for name, path in required.items()},
    }


def _load_source_seed(output_root: Path, seed: int) -> pd.DataFrame:
    source = pd.read_csv(_source_path(output_root), low_memory=False)
    oof = _validate_oof_frame(_run_dir(output_root, seed) / "oof_predictions.parquet", source, seed)
    patients = evaluate.to_patient_level(oof, source)
    extras = [
        "patient_id",
        "stage_group_major",
        "stage_group_major_filled",
        "sidedness",
    ]
    patient_context = source.drop_duplicates("patient_id")[[column for column in extras if column in source]]
    patients = patients.merge(patient_context, on="patient_id", how="left", validate="one_to_one")
    if len(patients) != EXPECTED_SOURCE_PATIENTS or int(patients["label"].sum()) != (
        EXPECTED_SOURCE_MUTANTS
    ):
        raise ContractError(f"Seed {seed} patient-level OOF census is incomplete")
    return patients.sort_values("patient_id", kind="stable").reset_index(drop=True)


def _aggregate_source_refit_scores(
    scores: pd.DataFrame, source: pd.DataFrame, *, seed: int
) -> pd.DataFrame:
    """Aggregate one full-source refit's native slide logits to patients."""

    required = {"slide_id", "patient_id", "target_label"}
    if required - set(source):
        raise ContractError(
            f"Source refit manifest lacks {sorted(required - set(source))}"
        )
    _validate_score_frame(scores, source, seed=seed, context=f"source_refit/seed{seed}")
    patient_label_counts = source.groupby("patient_id")["target_label"].nunique(dropna=False)
    if not patient_label_counts.eq(1).all():
        raise ContractError("Source refit manifest has inconsistent patient labels")
    merged = source[["slide_id", "patient_id", "target_label"]].merge(
        scores[["slide_id", "logit"]], on="slide_id", how="left", validate="one_to_one"
    )
    patient = (
        merged.sort_values("slide_id", kind="stable")
        .groupby("patient_id", sort=True)
        .agg(
            label=("target_label", "first"),
            mean_logit=("logit", "mean"),
            n_slides=("slide_id", "nunique"),
        )
        .reset_index()
    )
    patient["label"] = pd.to_numeric(patient["label"], errors="raise").astype(int)
    if not np.isfinite(patient["mean_logit"].to_numpy(dtype=float)).all():
        raise ContractError(f"Source refit seed {seed} has non-finite patient logits")
    return patient


def _load_source_refit_seed(output_root: Path, seed: int) -> pd.DataFrame:
    patient = _aggregate_source_refit_scores(
        pd.read_parquet(_score_path(output_root, "source_refit", seed)),
        pd.read_csv(_source_path(output_root), low_memory=False),
        seed=seed,
    )
    observed = (len(patient), int(patient["label"].sum()))
    expected = (EXPECTED_SOURCE_PATIENTS, EXPECTED_SOURCE_MUTANTS)
    if observed != expected:
        raise ContractError(
            f"Source refit seed {seed} patient census is {observed}, expected {expected}"
        )
    return patient


def _refit_oof_logit_sd_diagnostic(
    oof_frames: Mapping[int, pd.DataFrame],
    refit_frames: Mapping[int, pd.DataFrame],
) -> dict[str, Any]:
    """Compare native patient-logit dispersion after 5/5 refit versus honest OOF."""

    def normalized(frame: pd.DataFrame, *, role: str, seed: int) -> pd.DataFrame:
        required = {"patient_id", "label", "mean_logit"}
        if required - set(frame):
            raise ContractError(
                f"{role} seed {seed} lacks {sorted(required - set(frame))}"
            )
        if frame["patient_id"].astype(str).duplicated().any():
            raise ContractError(f"{role} seed {seed} has duplicate patients")
        out = frame[["patient_id", "label", "mean_logit"]].copy()
        out["patient_id"] = out["patient_id"].astype(str)
        out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
        out["mean_logit"] = pd.to_numeric(out["mean_logit"], errors="coerce")
        out = out.sort_values("patient_id", kind="stable").reset_index(drop=True)
        if len(out) < 2 or not np.isfinite(out["mean_logit"].to_numpy(dtype=float)).all():
            raise ContractError(f"{role} seed {seed} has insufficient/non-finite logits")
        return out

    if set(oof_frames) != set(SEEDS) or set(refit_frames) != set(SEEDS):
        raise ContractError(f"Refit/OOF scale diagnostic requires seeds {SEEDS}")
    oof = {seed: normalized(oof_frames[seed], role="OOF", seed=seed) for seed in SEEDS}
    refit = {
        seed: normalized(refit_frames[seed], role="refit", seed=seed) for seed in SEEDS
    }
    base = oof[SEEDS[0]][["patient_id", "label"]]
    for role, frames in (("OOF", oof), ("refit", refit)):
        for seed, frame in frames.items():
            if not frame["patient_id"].equals(base["patient_id"]) or not frame[
                "label"
            ].equals(base["label"]):
                raise ContractError(f"{role} seed {seed} patient roster/labels differ")

    def block(oof_logits: np.ndarray, refit_logits: np.ndarray) -> dict[str, Any]:
        oof_sd = float(np.std(oof_logits, ddof=1))
        refit_sd = float(np.std(refit_logits, ddof=1))
        if not np.isfinite([oof_sd, refit_sd]).all() or oof_sd <= 0:
            raise ContractError("Honest-OOF native-logit SD is non-positive or non-finite")
        return {
            "n_patients": int(len(oof_logits)),
            "honest_oof_logit_sd": oof_sd,
            "full_source_refit_logit_sd": refit_sd,
            "refit_to_oof_sd_ratio": float(refit_sd / oof_sd),
        }

    per_seed = {
        seed: block(
            oof[seed]["mean_logit"].to_numpy(dtype=float),
            refit[seed]["mean_logit"].to_numpy(dtype=float),
        )
        for seed in SEEDS
    }
    mean_oof = np.mean(
        np.vstack([oof[seed]["mean_logit"].to_numpy(dtype=float) for seed in SEEDS]), axis=0
    )
    mean_refit = np.mean(
        np.vstack([refit[seed]["mean_logit"].to_numpy(dtype=float) for seed in SEEDS]), axis=0
    )
    return {
        "unit": "patient; mean native slide logit",
        "sd_definition": "sample standard deviation (ddof=1)",
        "per_seed": per_seed,
        "three_seed_mean_native_logit": block(mean_oof, mean_refit),
        "interpretation": (
            "Required scale diagnostic only. Ratios quantify 5/5-refit versus honest-OOF "
            "logit dispersion; they do not alter raw-logit ranks or the source-fitted Platt map."
        ),
    }


def _ensure_training_receipt(
    output_root: Path, seed: int, record: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    record = dict(record or _training_record(output_root, seed))
    request_path = output_root / "requests/training" / f"seed{seed}.json"
    request = _read_json(request_path)
    if request != _train_request_comparable(request, _train_request(output_root, seed)):
        raise ContractError(f"Seed {seed} completed run has no matching immutable request")
    path = _train_receipt_path(output_root, seed)
    expected = {
        "schema_version": 1,
        "status": "completed",
        "request": _artifact(request_path),
        "artifacts": record,
        "target_outcomes_opened": False,
    }
    if path.is_file():
        receipt = _read_json(path)
        mismatch = {
            key: {"expected": value, "observed": receipt.get(key)}
            for key, value in expected.items()
            if receipt.get(key) != value
        }
        if mismatch or not isinstance(receipt.get("finished_utc"), str):
            raise ContractError(f"Seed {seed} training receipt mismatch: {mismatch}")
    else:
        _publish_json(path, {**expected, "finished_utc": _utc_now()})
    return _artifact(path)


def _fit_source_platt(
    seed_frames: Mapping[int, pd.DataFrame],
    source_oof_inputs: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one source-only Platt map on mean honest OOF native logits."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    if set(seed_frames) != set(SEEDS):
        raise ContractError(f"Source Platt needs seeds {SEEDS}, got {sorted(seed_frames)}")
    frames = {seed: frame.sort_values("patient_id", kind="stable").reset_index(drop=True) for seed, frame in seed_frames.items()}
    base = frames[SEEDS[0]][["patient_id", "label"]].copy()
    for seed in SEEDS[1:]:
        if not frames[seed]["patient_id"].equals(base["patient_id"]) or not frames[seed]["label"].astype(
            int
        ).equals(base["label"].astype(int)):
            raise ContractError(f"Seed {seed} OOF patient roster/labels differ")
    eta = np.mean(
        np.vstack([frames[seed]["mean_logit"].to_numpy(dtype=float) for seed in SEEDS]), axis=0
    )
    y = base["label"].to_numpy(dtype=int)
    if not np.isfinite(eta).all() or set(y) != {0, 1}:
        raise ContractError("Source Platt inputs are non-finite or single-class")
    model = LogisticRegression(penalty=None, max_iter=1_000)
    model.fit(eta.reshape(-1, 1), y)
    a = float(model.intercept_[0])
    b = float(model.coef_[0, 0])
    if not np.isfinite([a, b]).all() or b <= 0:
        raise ContractError("Source Platt map is non-finite or non-monotone")
    table = base.assign(mean_oof_logit=eta)
    record = {
        "schema_version": 1,
        "role": "source-only Platt map for all-primary-plus-Orion refit ensemble",
        "a": a,
        "b": b,
        "n_source": int(len(y)),
        "n_mutant": int(y.sum()),
        "source_prevalence": float(y.mean()),
        "source_oof_auroc": float(roc_auc_score(y, eta)),
        "source_oof_logit_mean": float(np.mean(eta)),
        "source_oof_logit_sd": float(np.std(eta, ddof=1)),
        "source_oof_inputs": dict(source_oof_inputs or {}),
        "aggregation": "mean slide native logit within patient, then mean across seeds 42/43/44",
        "target_labels_used": False,
        "target_labels_used_for_fit": False,
    }
    return record, table


def _ensure_model_seal(output_root: Path) -> dict[str, Any]:
    if _model_seal_path(output_root).is_file():
        return _load_model_seal(output_root)
    contract = _load_contract(output_root)
    records: dict[str, Any] = {}
    for seed in SEEDS:
        record = _training_record(output_root, seed)
        records[str(seed)] = {
            **record,
            "runner_receipt": _ensure_training_receipt(output_root, seed, record),
        }
    frames = {seed: _load_source_seed(output_root, seed) for seed in SEEDS}
    inputs = {str(seed): records[str(seed)]["oof"] for seed in SEEDS}
    calibrator, table = _fit_source_platt(frames, inputs)
    _publish_parquet(_calibrator_table_path(output_root), table)
    calibrator["patient_logit_table"] = _artifact(_calibrator_table_path(output_root))
    calibrator["source_manifest"] = contract["source"]["manifest"]
    _publish_json(_calibrator_path(output_root), calibrator)
    seal = {
        "schema_version": 1,
        "status": "three-seed OOF and p75 refit ensemble complete",
        "contract": _artifact(_contract_path(output_root)),
        "training": records,
        "calibrator": _artifact(_calibrator_path(output_root)),
        "target_labels_used": False,
    }
    _publish_json(_model_seal_path(output_root), seal)
    return _load_model_seal(output_root)


def _load_model_seal(output_root: Path) -> dict[str, Any]:
    _load_contract(output_root)
    seal = _read_json(_model_seal_path(output_root))
    if seal.get("status") != "three-seed OOF and p75 refit ensemble complete":
        raise ContractError("Model seal status is incomplete")
    if seal.get("contract") != _artifact(_contract_path(output_root)):
        raise ContractError("Model seal binds a different experiment contract")
    if set(seal.get("training") or {}) != {str(seed) for seed in SEEDS}:
        raise ContractError("Model seal has an incomplete seed roster")
    for seed in SEEDS:
        record = _training_record(output_root, seed)
        live = {**record, "runner_receipt": _ensure_training_receipt(output_root, seed, record)}
        if seal["training"][str(seed)] != live:
            raise ContractError(f"Seed {seed} training artifacts changed after model seal")
    _verify_artifact(seal.get("calibrator"), "source calibrator")
    calibrator = _read_json(_calibrator_path(output_root))
    if calibrator.get("target_labels_used") is not False or calibrator.get(
        "target_labels_used_for_fit"
    ) is not False:
        raise ContractError("Source calibrator does not declare outcome-free fitting")
    _verify_artifact(calibrator.get("patient_logit_table"), "source Platt patient table")
    for seed, identity in (calibrator.get("source_oof_inputs") or {}).items():
        _verify_artifact(identity, f"source Platt OOF/{seed}")
    return seal


def _inference_environment(num_workers: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ContractError("Governed scoring requires a CUDA GPU with bfloat16 support")
    return {
        "device": "cuda",
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "autocast": True,
        "autocast_dtype": "bfloat16",
        "batch_size": 1,
        "num_workers": int(num_workers),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "lightning_version": importlib.metadata.version("lightning"),
        "implementation": [
            _artifact(Path(__file__).resolve()),
            _artifact(REPO / "aim2_cross_protocol_transfer.py"),
            _artifact(REPO / "src/oceanpath/training/lightning.py"),
            _artifact(REPO / "src/oceanpath/datasets/packed.py"),
            _artifact(REPO / "src/oceanpath/models/abmil.py"),
        ],
    }


def _validate_score_frame(
    scores: pd.DataFrame, manifest: pd.DataFrame, *, seed: int, context: Path | str
) -> None:
    expected_columns = ["slide_id", "seed", "fold", "logit"]
    if list(scores.columns) != expected_columns:
        raise ContractError(f"Score schema mismatch for {context}: {list(scores.columns)}")
    if set(scores["seed"].astype(int)) != {seed} or set(scores["fold"].astype(int)) != {0}:
        raise ContractError(f"Wrong seed/fold in {context}")
    if scores.duplicated(["slide_id", "seed", "fold"]).any():
        raise ContractError(f"Duplicate slide/model rows in {context}")
    if set(scores["slide_id"].astype(str)) != set(manifest["slide_id"].astype(str)) or len(
        scores
    ) != len(manifest):
        raise ContractError(f"Incomplete score-dataset slide roster in {context}")
    if not np.isfinite(pd.to_numeric(scores["logit"], errors="coerce")).all():
        raise ContractError(f"Non-finite native logits in {context}")


def _score_cached(
    output_root: Path,
    dataset: str,
    seed: int,
    *,
    environment: Mapping[str, Any] | None = None,
) -> bool:
    path = _score_path(output_root, dataset, seed)
    receipt_path = _score_receipt_path(output_root, dataset, seed)
    if not path.exists() and not receipt_path.exists():
        return False
    if not path.is_file() or not receipt_path.is_file():
        raise ContractError(f"Partial score cache: {path}")
    receipt = _read_json(receipt_path)
    record = _training_record(output_root, seed)
    expected = {
        "schema_version": 1,
        "preoutcome": True,
        "contains_target_outcomes": False,
        "dataset": dataset,
        "seed": seed,
        "manifest": _artifact(_score_manifest_path(output_root, dataset)),
        "contract": _artifact(_contract_path(output_root)),
        "model_seal": _artifact(_model_seal_path(output_root)),
        "checkpoint": record["refit_checkpoint"],
    }
    mismatch = {key: (wanted, receipt.get(key)) for key, wanted in expected.items() if receipt.get(key) != wanted}
    if environment is not None and receipt.get("inference_environment") != dict(environment):
        mismatch["inference_environment"] = "changed"
    if mismatch or receipt.get("artifact") != _artifact(path):
        raise ContractError(f"Invalid score receipt {receipt_path}: {mismatch}")
    _validate_score_frame(
        pd.read_parquet(path),
        pd.read_csv(_score_manifest_path(output_root, dataset)),
        seed=seed,
        context=path,
    )
    return True


def _score_one(
    output_root: Path,
    dataset: str,
    seed: int,
    *,
    device: str,
    num_workers: int,
    environment: Mapping[str, Any],
) -> None:
    if _score_cached(output_root, dataset, seed, environment=environment):
        print(f"cached and verified: {dataset}/seed{seed}")
        return
    contract = _load_contract(output_root)
    _load_model_seal(output_root)
    manifest = pd.read_csv(_score_manifest_path(output_root, dataset))
    record = _training_record(output_root, seed)
    checkpoint = Path(record["refit_checkpoint"]["path"])
    scores = cpht._score_checkpoints(  # noqa: SLF001
        [(seed, 0, checkpoint)],
        manifest,
        feature_dir=Path(str(contract["feature_store"]["source_dir"])),
        pack_dir=Path(str(contract["feature_store"]["path"])),
        device=device,
        num_workers=num_workers,
    )
    _validate_score_frame(
        scores, manifest, seed=seed, context=_score_path(output_root, dataset, seed)
    )
    _publish_parquet(_score_path(output_root, dataset, seed), scores)
    _publish_json(
        _score_receipt_path(output_root, dataset, seed),
        {
            "schema_version": 1,
            "created_utc": _utc_now(),
            "preoutcome": True,
            "contains_target_outcomes": False,
            "dataset": dataset,
            "seed": seed,
            "manifest": _artifact(_score_manifest_path(output_root, dataset)),
            "contract": _artifact(_contract_path(output_root)),
            "model_seal": _artifact(_model_seal_path(output_root)),
            "checkpoint": record["refit_checkpoint"],
            "inference_environment": dict(environment),
            "artifact": _artifact(_score_path(output_root, dataset, seed)),
            "n_rows": int(len(scores)),
        },
    )


def _seal_inference(output_root: Path) -> dict[str, Any]:
    if _inference_seal_path(output_root).is_file():
        return _load_inference_seal(output_root)
    environments = []
    artifacts = []
    for dataset in SCORE_DATASETS:
        for seed in SEEDS:
            receipt = _read_json(_score_receipt_path(output_root, dataset, seed))
            if not _score_cached(output_root, dataset, seed):
                raise ContractError(f"Cannot seal incomplete inference: {dataset}/seed{seed}")
            environments.append(receipt["inference_environment"])
            artifacts.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "score": _artifact(_score_path(output_root, dataset, seed)),
                    "receipt": _artifact(_score_receipt_path(output_root, dataset, seed)),
                }
            )
    if any(value != environments[0] for value in environments[1:]):
        raise ContractError("Scoring environment changed within the nine-job inference campaign")
    seal = {
        "schema_version": 1,
        "status": "sealed before target-outcome join",
        "contains_target_outcomes": False,
        "contract": _artifact(_contract_path(output_root)),
        "model_seal": _artifact(_model_seal_path(output_root)),
        "calibrator": _artifact(_calibrator_path(output_root)),
        "datasets": list(SCORE_DATASETS),
        "targets": list(TARGETS),
        "seeds": list(SEEDS),
        "score_artifact_count": len(SCORE_DATASETS) * len(SEEDS),
        "inference_environment": environments[0],
        "score_artifacts": artifacts,
    }
    _publish_json(_inference_seal_path(output_root), seal)
    return _load_inference_seal(output_root)


def _load_inference_seal(output_root: Path) -> dict[str, Any]:
    _load_model_seal(output_root)
    seal = _read_json(_inference_seal_path(output_root))
    expected = {
        "schema_version": 1,
        "status": "sealed before target-outcome join",
        "contains_target_outcomes": False,
        "contract": _artifact(_contract_path(output_root)),
        "model_seal": _artifact(_model_seal_path(output_root)),
        "calibrator": _artifact(_calibrator_path(output_root)),
        "datasets": list(SCORE_DATASETS),
        "targets": list(TARGETS),
        "seeds": list(SEEDS),
        "score_artifact_count": len(SCORE_DATASETS) * len(SEEDS),
    }
    mismatch = {key: (wanted, seal.get(key)) for key, wanted in expected.items() if seal.get(key) != wanted}
    if mismatch:
        raise ContractError(f"Inference seal mismatch: {mismatch}")
    records = seal.get("score_artifacts") or []
    jobs = [(dataset, seed) for dataset in SCORE_DATASETS for seed in SEEDS]
    if len(records) != len(jobs):
        raise ContractError("Inference seal has an incomplete score roster")
    for record, (dataset, seed) in zip(records, jobs, strict=True):
        if record != {
            "dataset": dataset,
            "seed": seed,
            "score": _artifact(_score_path(output_root, dataset, seed)),
            "receipt": _artifact(_score_receipt_path(output_root, dataset, seed)),
        }:
            raise ContractError(f"Score changed after inference seal: {dataset}/seed{seed}")
        _score_cached(output_root, dataset, seed, environment=seal["inference_environment"])
    return seal


def _load_outcome(output_root: Path, target: str) -> pd.DataFrame:
    contract = _load_contract(output_root)
    record = contract["outcome_sources_not_opened_by_train_or_score"][target]
    path = _verify_artifact(record["artifact"], f"outcome/{target}")
    frame = pd.read_csv(path, low_memory=False)
    if record.get("filter") == "subcohort == SR1482":
        frame = frame[frame["subcohort"].eq("SR1482")].copy()
    required = {"slide_id", "patient_id", "target_label", "specimen_role"}
    if required - set(frame):
        raise ContractError(f"Outcome source {target} lacks {sorted(required - set(frame))}")
    labels = frame[["patient_id", "target_label"]].drop_duplicates()
    if labels["patient_id"].duplicated().any():
        raise ContractError(f"Outcome source {target} has within-patient KRAS disagreement")
    expected = (85, 37) if target == "rih_m" else (74, 30)
    observed = (len(labels), int(labels["target_label"].sum()))
    if observed != expected:
        raise ContractError(f"Outcome source {target} census is {observed}, expected {expected}")
    return frame


def _headline_target_frame(target: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical leakage-free metastatic headline population."""

    if target == "rih_m":
        return frame[~frame["patient_id"].astype(str).isin(RIH_DUAL_PATIENTS)].copy()
    if target == "sr1482_m":
        return frame.copy()
    raise ValueError(f"Unknown target: {target}")


def _acquisition_domains(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "TCGA": frame[frame["cohort"].eq("TCGA")],
        "SR386": frame[frame["subcohort"].eq("SR386")],
        "SR1482": frame[frame["subcohort"].eq("SR1482")],
        "RIH": frame[frame["cohort"].eq("RIH")],
        "CPTAC": frame[frame["cohort"].eq("CPTAC")],
        "Orion": frame[frame["cohort"].eq("Orion")],
    }


def _mean_native_logit_ensemble(frames: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Align patients and average native logits without a probability round-trip."""

    if set(frames) != set(SEEDS):
        raise ContractError(f"OOF ensemble needs seeds {SEEDS}, got {sorted(frames)}")
    ordered = {
        seed: frame.sort_values("patient_id", kind="stable").reset_index(drop=True)
        for seed, frame in frames.items()
    }
    base = ordered[SEEDS[0]].copy()
    for seed in SEEDS[1:]:
        if not ordered[seed]["patient_id"].equals(base["patient_id"]) or not ordered[seed][
            "label"
        ].astype(int).equals(base["label"].astype(int)):
            raise ContractError(f"Seed {seed} OOF patient roster/labels differ")
    base["mean_logit"] = np.mean(
        np.vstack([ordered[seed]["mean_logit"].to_numpy(dtype=float) for seed in SEEDS]),
        axis=0,
    )
    if not np.isfinite(base["mean_logit"]).all():
        raise ContractError("OOF native-logit ensemble contains non-finite values")
    # Probability is retained only for probability/calibration consumers. Rank
    # metrics below always name mean_logit explicitly, so saturation cannot
    # introduce artificial ties.
    base["prob_raw"] = sigmoid(base["mean_logit"].to_numpy(dtype=float))
    return base


def _source_e0_diagnostics(
    frames: Mapping[int, pd.DataFrame], *, n_bootstrap: int
) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss

    per_seed: dict[int, Any] = {}
    for seed in SEEDS:
        frame = frames[seed]
        calibrated = evaluate.cross_fitted_platt(frame)
        cal = evaluate.calibration_block(
            calibrated["label"].to_numpy(), calibrated["prob_cal"].to_numpy()
        )
        domains: dict[str, Any] = {}
        for name, block in _acquisition_domains(frame).items():
            domains[name] = evaluate.bootstrap_auroc(
                block,
                score_column="mean_logit",
                n_bootstrap=n_bootstrap,
                seed=BOOTSTRAP_SEED,
            )
        per_seed[seed] = {
            "n": int(len(frame)),
            "n_mutant": int(frame["label"].sum()),
            "pooled_oof": evaluate.bootstrap_auroc(
                frame,
                score_column="mean_logit",
                n_bootstrap=n_bootstrap,
                seed=BOOTSTRAP_SEED,
            ),
            "auprc": evaluate.auprc_with_ci(
                frame,
                score_column="mean_logit",
                n_bootstrap=n_bootstrap,
                seed=BOOTSTRAP_SEED,
            ),
            "cross_fitted_calibration": {
                "brier": float(brier_score_loss(calibrated["label"], calibrated["prob_cal"])),
                "calibration_intercept": float(cal["calibration_intercept"]),
                "calibration_slope": float(cal["calibration_slope"]),
            },
            "youden_within_seed": e0.youden_within_seed(frame),
            "per_fold_auroc": [
                evaluate.patient_auroc(
                    frame[frame["k_fold"].eq(fold)], score_column="mean_logit"
                )
                for fold in range(N_FOLDS)
            ],
            "per_cohort_auroc": {
                str(name): evaluate.patient_auroc(block, score_column="mean_logit")
                for name, block in frame.groupby("cohort")
            },
            "per_subcohort_auroc": {
                str(name): evaluate.patient_auroc(block, score_column="mean_logit")
                for name, block in frame.groupby("subcohort")
            },
            "acquisition_domains": domains,
            "equal_six_domain_macro_auroc": float(
                np.mean([value["auroc"] for value in domains.values()])
            ),
        }
    # Mean native logit across the three honest OOF models is a deployment
    # sensitivity.  Seed-median remains the E0 headline convention.
    ensemble = _mean_native_logit_ensemble(frames)
    return {
        "per_seed": per_seed,
        "headline_convention": "median with observed min-max over the three seed OOF estimates",
        "median_seed_auroc": float(
            np.median([per_seed[seed]["pooled_oof"]["auroc"] for seed in SEEDS])
        ),
        "seed_auroc_range": [
            float(min(per_seed[seed]["pooled_oof"]["auroc"] for seed in SEEDS)),
            float(max(per_seed[seed]["pooled_oof"]["auroc"] for seed in SEEDS)),
        ],
        "median_equal_six_domain_macro_auroc": float(
            np.median([per_seed[seed]["equal_six_domain_macro_auroc"] for seed in SEEDS])
        ),
        "three_seed_mean_native_logit_oof_sensitivity": {
            "auroc": evaluate.bootstrap_auroc(
                ensemble,
                score_column="mean_logit",
                n_bootstrap=n_bootstrap,
                seed=BOOTSTRAP_SEED,
            ),
            "auprc": evaluate.auprc_with_ci(
                ensemble,
                score_column="mean_logit",
                n_bootstrap=n_bootstrap,
                seed=BOOTSTRAP_SEED,
            ),
        },
    }


def _source_e1a_diagnostics(
    frames: Mapping[int, pd.DataFrame], *, n_bootstrap: int
) -> dict[str, Any]:
    per_seed: dict[int, Any] = {}
    for seed in SEEDS:
        frame = frames[seed]
        sets: dict[str, Any] = {}
        for key, (title, tier) in e1a.SETS.items():
            mask = e1a.subset_mask(frame, key)
            block = frame[mask]
            delta = evaluate.shared_resample_delta(
                frame,
                mask,
                score_column="mean_logit",
                n_bootstrap=n_bootstrap,
                seed=BOOTSTRAP_SEED,
            )
            ap = evaluate.auprc_with_ci(
                block,
                score_column="mean_logit",
                n_bootstrap=n_bootstrap,
                seed=BOOTSTRAP_SEED,
            )
            sets[key] = {
                "title": title,
                "figure": tier,
                "support": evaluate.support_tier(block["label"].to_numpy()),
                "n": int(len(block)),
                "n_mutant": int(block["label"].sum()),
                "auroc": delta["auroc_subset"],
                "ci_low": delta["subset_ci_low"],
                "ci_high": delta["subset_ci_high"],
                "auprc": ap["auprc"],
                "auprc_ci": [ap["ci_low"], ap["ci_high"]],
                "auprc_baseline": ap["baseline"],
                "delta_vs_A": delta["delta"],
                "delta_ci": [delta["delta_ci_low"], delta["delta_ci_high"]],
            }
        per_seed[seed] = sets
    return {
        "per_seed": per_seed,
        "gate": e0.evaluate_gate(per_seed),
        "delta_sign": "A minus restricted set; positive means discrimination fell in restriction",
        "bootstrap": "shared patient resample within each seed; seed effects remain separate",
    }


def _target_metric_block(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int = BOOTSTRAP_SEED,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    ordered = frame.sort_values("patient_id", kind="stable").reset_index(drop=True)
    generator = rng if rng is not None else np.random.default_rng(seed)
    indices = e2met._shared_stratified_indices(  # noqa: SLF001
        ordered["label"].to_numpy(dtype=int), n_bootstrap, generator
    )
    return e2met._metric_block(ordered, indices)  # noqa: SLF001


def _metastatic_analysis(
    patient_scores: Mapping[str, pd.DataFrame], *, n_bootstrap: int, bootstrap_seed: int
) -> dict[str, Any]:
    expected = {"rih_m": (85, 37), "sr1482_m": (74, 30)}
    headline_expected = {"rih_m": (77, 33), "sr1482_m": (74, 30)}
    metrics: dict[str, Any] = {}
    samples: dict[str, dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(bootstrap_seed)
    for target in TARGETS:
        full = patient_scores[target]
        observed = (len(full), int(full["label"].sum()))
        if observed != expected[target]:
            raise ContractError(f"{target} patient score census is {observed}, expected {expected[target]}")
        headline = _headline_target_frame(target, full)
        observed_headline = (len(headline), int(headline["label"].sum()))
        if observed_headline != headline_expected[target]:
            raise ContractError(
                f"{target} headline census is {observed_headline}, expected {headline_expected[target]}"
            )
        block, draw = _target_metric_block(
            headline, n_bootstrap=n_bootstrap, rng=rng
        )
        block["exposure"] = "target-primary-exposed deployment sensitivity"
        block["probability_metrics_role"] = (
            "exploratory deployment-calibration sensitivity; the Platt map was fit only on "
            "source honest OOF and target calibration is not externally validated"
        )
        block["source_calibrated"]["interpretation"] = block["probability_metrics_role"]
        block["population"] = "RIH77 direct-patient-overlap-excluded" if target == "rih_m" else "SR1482-M74"
        metrics[target] = block
        samples[target] = draw

    macro: dict[str, Any] = {}
    for key, point_path in (
        ("auroc", ("auroc",)),
        ("auprc", ("auprc",)),
        ("brier", ("source_calibrated", "brier")),
        ("log_loss", ("source_calibrated", "log_loss")),
    ):
        draws = np.mean([samples[target][key] for target in TARGETS], axis=0)
        points = []
        for target in TARGETS:
            value: Any = metrics[target]
            for part in point_path:
                value = value[part]
            points.append(float(value))
        macro[key] = {
            "value": float(np.mean(points)),
            "ci95": cpht._interval(draws),  # noqa: SLF001
        }
    macro["calibration_intercept"] = float(
        np.mean([metrics[target]["source_calibrated"]["calibration_intercept"] for target in TARGETS])
    )
    macro["calibration_slope"] = float(
        np.mean([metrics[target]["source_calibrated"]["calibration_slope"] for target in TARGETS])
    )
    macro["probability_metrics_role"] = (
        "exploratory deployment-calibration sensitivity; equal cohort weighted"
    )

    rih85, _ = _target_metric_block(
        patient_scores["rih_m"], n_bootstrap=n_bootstrap, seed=bootstrap_seed
    )
    rih85["warning"] = (
        "contaminated sensitivity: eight patients have primaries in the training source; "
        "never substitute for the RIH77 headline"
    )
    return {
        "headline": metrics,
        "equal_cohort_macro": macro,
        "rih85_contaminated_sensitivity": rih85,
        "probability_and_calibration_claim": (
            "Brier, log-loss, calibration intercept, and calibration slope are exploratory "
            "deployment-calibration sensitivities only. Raw native-logit AUROC/AUPRC are primary."
        ),
        "claim_boundary": (
            "Both targets are target-primary-exposed. These estimates test deployment on metastatic "
            "specimens, not external institutional transport, and no primary-to-metastatic delta is valid."
        ),
        "bootstrap": {
            "n": n_bootstrap,
            "seed": bootstrap_seed,
            "unit": "patient",
            "method": "fixed-class-count KRAS-stratified resampling; equal cohort weights",
        },
    }


def _analyse(output_root: Path, *, n_bootstrap: int, e1a_n_bootstrap: int) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    _load_inference_seal(output_root)
    source_frames = {seed: _load_source_seed(output_root, seed) for seed in SEEDS}
    source_refit_frames = {
        seed: _load_source_refit_seed(output_root, seed) for seed in SEEDS
    }
    calibrator = _read_json(_calibrator_path(output_root))
    patient_scores: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        score_frames = [pd.read_parquet(_score_path(output_root, target, seed)) for seed in SEEDS]
        patient_scores[target] = e2met.aggregate_patient_scores(
            score_frames,
            pd.read_csv(_target_path(output_root, target)),
            _load_outcome(output_root, target),
            calibrator=calibrator,
        )
    results = {
        "schema_version": 1,
        "experiment": "exploratory Aim-1 E0 all primary plus Orion",
        "source_population": {
            "n_slides": EXPECTED_SOURCE_SLIDES,
            "n_patients": EXPECTED_SOURCE_PATIENTS,
            "n_mutant_patients": EXPECTED_SOURCE_MUTANTS,
            "n_mutant_slides": EXPECTED_SOURCE_MUTANT_SLIDES,
            "conventional": {
                "n_slides": EXPECTED_CONVENTIONAL_SLIDES,
                "n_patients": EXPECTED_CONVENTIONAL_PATIENTS,
                "n_mutant_patients": EXPECTED_CONVENTIONAL_MUTANTS,
            },
            "orion": {
                "n_slides": EXPECTED_ORION_SLIDES,
                "n_patients": EXPECTED_ORION_PATIENTS,
                "n_mutant_patients": EXPECTED_ORION_MUTANTS,
            },
        },
        "training": {
            "seeds": list(SEEDS),
            "fits": {str(seed): _training_record(output_root, seed) for seed in SEEDS},
            "total_fit_count": len(SEEDS) * (N_FOLDS + 1),
            "training_num_workers": TRAINING_NUM_WORKERS,
            "refit_rule": "p75 epoch rule; one epoch is 1,526 patient-natural optimizer steps",
        },
        "source_platt": calibrator,
        "refit_vs_honest_oof_logit_scale": _refit_oof_logit_sd_diagnostic(
            source_frames, source_refit_frames
        ),
        "aim1_e0": _source_e0_diagnostics(source_frames, n_bootstrap=e1a_n_bootstrap),
        "aim1_e1a": _source_e1a_diagnostics(source_frames, n_bootstrap=e1a_n_bootstrap),
        "metastatic_transfer": _metastatic_analysis(
            patient_scores, n_bootstrap=n_bootstrap, bootstrap_seed=BOOTSTRAP_SEED
        ),
        "interpretation_boundary": (
            "This deliberately exploratory fit includes Orion and all conventional primaries. "
            "Its source OOF estimates remain honest under the appended folds, but Orion is no longer "
            "an external transfer test and both metastatic targets are primary-domain-exposed. "
            "Metastatic probability/calibration metrics are exploratory deployment-calibration "
            "sensitivities; raw native-logit rank metrics remain primary."
        ),
    }
    return results, patient_scores


def _run_logged(command: list[str], log_path: Path) -> int:
    if log_path.exists() or log_path.is_symlink():
        raise FileExistsError(f"Refusing to replace training log: {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8", buffering=1) as log_stream:
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
        return int(process.wait())


def cmd_plan(args: argparse.Namespace) -> None:
    _guard_output_root(args.output_root, before_contract=True)
    _validate_canonical_input_arguments(args)
    source, conventional, orion = _load_source_inputs(
        args.conventional_manifest, args.label_source, args.orion_folds, args.pack_dir
    )
    _validate_frozen_conventional_splits(conventional, args.conventional_splits)
    targets = _load_label_blind_targets(args.met_input_root)
    _validate_target_overlap(source, targets)
    pack = _validate_pack_roster(source, targets, args.pack_dir)
    print("Exploratory E0 all-primary-plus-Orion campaign")
    print(f"  immutable output root: {args.output_root}")
    print(
        f"  source: {len(source):,} slides / {source.patient_id.nunique():,} patients / "
        f"{source.drop_duplicates('patient_id').target_label.sum():,} mutant patients"
    )
    print(
        f"    conventional (frozen folds): {len(conventional):,} slides / "
        f"{conventional.patient_id.nunique():,} patients"
    )
    print(
        f"    Orion (CPHT-A folds): {len(orion):,} slides / "
        f"{orion.patient_id.nunique():,} patients"
    )
    print(f"  fits: 3 seeds x (5 outer folds + 1 p75 refit) = {len(SEEDS) * 6}")
    print(
        f"  training: UNIv1 gated ABMIL, patient_natural, cap {CAP:,}, "
        f"training.num_workers={TRAINING_NUM_WORKERS}; seeds {SEEDS} serialized by default"
    )
    print(
        f"  tests: RIH-M {targets['rih_m'].patient_id.nunique()} (headline 77) and "
        f"SR1482-M {targets['sr1482_m'].patient_id.nunique()} patients"
    )
    print("  scoring: 3 source-refit scale checks + 6 metastatic target jobs")
    print(f"  packed UNI: {pack['n_slides']:,} slides x {pack['feature_dim']:,}")
    print("  claim: deployment sensitivity only; Orion and target-primary domains enter fitting")


def cmd_manifest(args: argparse.Namespace) -> None:
    _guard_output_root(args.output_root, before_contract=True)
    _validate_canonical_input_arguments(args)
    if _contract_path(args.output_root).is_file():
        _load_contract(args.output_root)
        _validate_generated_splits(args.output_root)
        print(f"Manifest contract already exists and is valid: {_contract_path(args.output_root)}")
        return
    source, conventional, _orion = _load_source_inputs(
        args.conventional_manifest, args.label_source, args.orion_folds, args.pack_dir
    )
    _validate_frozen_conventional_splits(conventional, args.conventional_splits)
    targets = _load_label_blind_targets(args.met_input_root)
    headlines = _validate_target_overlap(source, targets)
    _validate_pack_roster(source, targets, args.pack_dir)
    _publish_csv(_source_path(args.output_root), source)
    for target, frame in targets.items():
        _publish_csv(_target_path(args.output_root, target), frame)
        _publish_csv(_headline_roster_path(args.output_root, target), headlines[target])
    _ensure_splits(args.output_root)
    _validate_generated_splits(args.output_root)

    from omegaconf import OmegaConf

    for seed in SEEDS:
        cfg = _compose_training_cfg(
            seed,
            _source_path(args.output_root),
            _split_root(args.output_root),
            _run_dir(args.output_root, seed),
        )
        _validate_training_config(
            cast(Mapping[str, Any], OmegaConf.to_container(cfg, resolve=True)), seed, args.output_root
        )
        _publish_text(_config_path(args.output_root, seed), OmegaConf.to_yaml(cfg, resolve=True))
    contract = _build_manifest_contract(
        args.output_root,
        conventional_manifest=args.conventional_manifest,
        label_source=args.label_source,
        orion_folds=args.orion_folds,
        conventional_splits=args.conventional_splits,
        met_input_root=args.met_input_root,
        pack_dir=args.pack_dir,
    )
    _publish_json(_contract_path(args.output_root), contract)
    _load_contract(args.output_root)
    print(f"Sealed additive experiment inputs: {_contract_path(args.output_root)}")


def cmd_preflight(args: argparse.Namespace) -> None:
    _guard_output_root(args.output_root, before_contract=True)
    _validate_canonical_input_arguments(args)
    if _contract_path(args.output_root).is_file():
        contract = _load_contract(args.output_root, verify_outcomes=True)
        _validate_generated_splits(args.output_root)
        source = pd.read_csv(_source_path(args.output_root))
        print(f"PASS: sealed contract {contract['lineage']} is hash-valid")
    else:
        source, conventional, _ = _load_source_inputs(
            args.conventional_manifest, args.label_source, args.orion_folds, args.pack_dir
        )
        _validate_frozen_conventional_splits(conventional, args.conventional_splits)
        targets = _load_label_blind_targets(args.met_input_root)
        _validate_target_overlap(source, targets)
        _validate_pack_roster(source, targets, args.pack_dir)
        for seed in SEEDS:
            from omegaconf import OmegaConf

            cfg = _compose_training_cfg(
                seed,
                _source_path(args.output_root),
                _split_root(args.output_root),
                _run_dir(args.output_root, seed),
            )
            _validate_training_config(
                cast(Mapping[str, Any], OmegaConf.to_container(cfg, resolve=True)),
                seed,
                args.output_root,
            )
        print("PASS: live inputs/configs are ready; run `manifest` to seal them")
    print(
        f"  {len(source):,} slides / {source.patient_id.nunique():,} patients; "
        f"training.num_workers={TRAINING_NUM_WORKERS}; 18 total fits"
    )


def _train_request(output_root: Path, seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "requested",
        "created_utc": _utc_now(),
        "seed": seed,
        "sampling_seed": seed,
        "fit_count": N_FOLDS + 1,
        "cap": CAP,
        "training_num_workers": TRAINING_NUM_WORKERS,
        "refit_epoch_rule": "p75",
        "refit_max_steps": None,
        "target_outcomes_opened": False,
        "contract": _artifact(_contract_path(output_root)),
        "config": _artifact(_config_path(output_root, seed)),
        "source_manifest": _artifact(_source_path(output_root)),
        "splits": _artifact(_split_dir(output_root) / "splits.parquet"),
        "run_dir": str(_run_dir(output_root, seed).resolve()),
    }


def _train_request_comparable(existing: Mapping[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    live["created_utc"] = existing.get("created_utc")
    return live


def _validate_partial_training_identity(output_root: Path, seed: int) -> None:
    from omegaconf import OmegaConf

    from oceanpath.workflows.training import training_run_fingerprint

    directory = _run_dir(output_root, seed)
    if not directory.exists():
        return
    if not directory.is_dir() or directory.is_symlink():
        raise ContractError(f"Seed {seed} run path is not a real directory")
    entries = list(directory.iterdir())
    if not entries:
        return
    identity_path = directory / "training_identity.json"
    identity = _read_json(identity_path)
    expected = training_run_fingerprint(OmegaConf.load(_config_path(output_root, seed)))
    if identity.get("fingerprint") != expected:
        raise ContractError(f"Seed {seed} partial run has the wrong training fingerprint")
    run_config = directory / "config.yaml"
    if run_config.is_file():
        frozen = _artifact(_config_path(output_root, seed))
        observed = _artifact(run_config)
        if (observed["sha256"], observed["size_bytes"]) != (
            frozen["sha256"],
            frozen["size_bytes"],
        ):
            raise ContractError(f"Seed {seed} partial run config differs from the frozen config")


def _fit(output_root: Path, seed: int) -> None:
    from omegaconf import OmegaConf

    from oceanpath.workflows.training import run_training

    _load_contract(output_root)
    request_path = output_root / "requests/training" / f"seed{seed}.json"
    request = _read_json(request_path)
    if request != _train_request_comparable(request, _train_request(output_root, seed)):
        raise ContractError(f"Seed {seed} request differs from sealed inputs")
    cfg = OmegaConf.load(_config_path(output_root, seed))
    _validate_training_config(
        cast(Mapping[str, Any], OmegaConf.to_container(cfg, resolve=True)), seed, output_root
    )
    with _exclusive_gpu_lock(f"E0 all-primary-Orion train seed{seed}"):
        _cuda_idle_precheck()
        print(run_training(cfg).to_json())
    record = _training_record(output_root, seed)
    _ensure_training_receipt(output_root, seed, record)


def cmd_train(args: argparse.Namespace) -> None:
    _load_contract(args.output_root)
    _validate_generated_splits(args.output_root)
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    for seed in seeds:
        completion_path = _run_dir(args.output_root, seed) / "training_completion.json"
        if completion_path.is_file():
            record = _training_record(args.output_root, seed)
            _ensure_training_receipt(args.output_root, seed, record)
            print(
                f"seed {seed}: complete ({record['fit_count']} fits, p75 refit "
                f"{record['refit_epochs']} epochs); skipping"
            )
            continue
        _validate_partial_training_identity(args.output_root, seed)
        request_path = args.output_root / "requests/training" / f"seed{seed}.json"
        if request_path.is_file():
            existing = _read_json(request_path)
            if existing != _train_request_comparable(
                existing, _train_request(args.output_root, seed)
            ):
                raise ContractError(f"Seed {seed} existing train request differs")
        else:
            if not args.dry_run:
                _publish_json(request_path, _train_request(args.output_root, seed))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_fit",
            "--output-root",
            str(args.output_root),
            "--seed",
            str(seed),
        ]
        if args.dry_run:
            print(" ".join(command))
            continue
        log_dir = args.output_root / "logs/training" / f"seed{seed}"
        attempts = sorted(log_dir.glob("attempt-*.log")) if log_dir.is_dir() else []
        attempt = len(attempts) + 1
        log_path = log_dir / f"attempt-{attempt:03d}.log"
        code = _run_logged(command, log_path)
        if code != 0:
            _publish_json(
                log_dir / f"attempt-{attempt:03d}.failure.json",
                {
                    "schema_version": 1,
                    "status": "failed",
                    "finished_utc": _utc_now(),
                    "attempt": attempt,
                    "returncode": code,
                    "command": command,
                    "log": _artifact(log_path),
                    "target_outcomes_opened": False,
                },
            )
            raise RuntimeError(
                f"Seed {seed} training attempt {attempt} failed with rc={code}; "
                "immutable evidence retained and a later attempt may safely resume"
            )
        record = _training_record(args.output_root, seed)
        _ensure_training_receipt(args.output_root, seed, record)


def cmd_score(args: argparse.Namespace) -> None:
    _ensure_model_seal(args.output_root)
    if args.device != "cuda":
        raise ContractError("Governed target scoring requires --device cuda")
    if args.num_workers != TRAINING_NUM_WORKERS:
        raise ContractError(
            f"Governed scoring worker count is frozen at {TRAINING_NUM_WORKERS}"
        )
    datasets = [args.target] if args.target is not None else list(SCORE_DATASETS)
    seeds = [args.seed] if args.seed is not None else list(SEEDS)
    with _exclusive_gpu_lock("E0 all-primary-Orion source-refit/metastatic scoring"):
        _cuda_idle_precheck()
        environment = _inference_environment(args.num_workers)
        for dataset in datasets:
            for seed in seeds:
                _score_one(
                    args.output_root,
                    dataset,
                    seed,
                    device=args.device,
                    num_workers=args.num_workers,
                    environment=environment,
                )
    if args.target is None and args.seed is None:
        seal = _seal_inference(args.output_root)
        print(
            "Sealed three source-refit and six label-blind target scores: "
            f"{_inference_seal_path(args.output_root)}"
        )
        print(f"  score artifacts: {seal['score_artifact_count']}")
    else:
        print("Partial score selection complete; run unfiltered `score` to seal all nine artifacts")


def cmd_report(args: argparse.Namespace) -> None:
    _load_inference_seal(args.output_root)
    results, patient_scores = _analyse(
        args.output_root,
        n_bootstrap=args.n_bootstrap,
        e1a_n_bootstrap=args.e1a_n_bootstrap,
    )
    tables: dict[str, Any] = {}
    for target, frame in patient_scores.items():
        path = args.output_root / "analysis" / f"{target}_patient_scores.parquet"
        out = frame.copy()
        out["headline_population"] = out["patient_id"].isin(
            set(_headline_target_frame(target, out)["patient_id"])
        )
        _publish_parquet(path, out)
        tables[target] = _artifact(path)
    _publish_json(_analysis_path(args.output_root), results)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "inference_seal": _artifact(_inference_seal_path(args.output_root)),
        "outcome_sources": {
            target: _load_contract(args.output_root, verify_outcomes=True)[
                "outcome_sources_not_opened_by_train_or_score"
            ][target]["artifact"]
            for target in TARGETS
        },
        "results": _artifact(_analysis_path(args.output_root)),
        "patient_tables": tables,
    }
    _publish_json(args.output_root / "analysis/report_receipt.json", receipt)
    met = results["metastatic_transfer"]
    e0_result = results["aim1_e0"]
    print(
        f"Aim-1 source OOF median seed AUROC: {e0_result['median_seed_auroc']:.4f} "
        f"(range {e0_result['seed_auroc_range'][0]:.4f}-"
        f"{e0_result['seed_auroc_range'][1]:.4f})"
    )
    for target in TARGETS:
        block = met["headline"][target]
        print(
            f"{target}: n={block['n']}, AUROC {block['auroc']:.4f} "
            f"({block['auroc_ci95'][0]:.4f}-{block['auroc_ci95'][1]:.4f}), "
            f"AUPRC {block['auprc']:.4f}"
        )
    macro = met["equal_cohort_macro"]["auroc"]
    print(
        f"Equal-cohort metastatic macro AUROC {macro['value']:.4f} "
        f"({macro['ci95'][0]:.4f}-{macro['ci95'][1]:.4f})"
    )
    print(f"Wrote {_analysis_path(args.output_root)}")


def cmd_verify(args: argparse.Namespace) -> None:
    contract = _load_contract(args.output_root, verify_outcomes=True)
    _validate_generated_splits(args.output_root)
    _load_model_seal(args.output_root)
    _load_inference_seal(args.output_root)
    receipt_path = args.output_root / "analysis/report_receipt.json"
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("inference_seal") != _artifact(
        _inference_seal_path(args.output_root)
    ):
        raise ContractError("Analysis receipt is incomplete or binds a different inference seal")
    _verify_artifact(receipt.get("results"), "analysis results")
    for target, identity in (receipt.get("patient_tables") or {}).items():
        _verify_artifact(identity, f"patient table/{target}")
    results = _read_json(_analysis_path(args.output_root))
    if results.get("source_population", {}).get("n_patients") != EXPECTED_SOURCE_PATIENTS:
        raise ContractError("Reported source census is wrong")
    scale = results.get("refit_vs_honest_oof_logit_scale") or {}
    if set(scale.get("per_seed") or {}) != {str(seed) for seed in SEEDS}:
        raise ContractError("Reported refit/OOF scale diagnostic has an incomplete seed roster")
    scale_blocks = [
        *(scale["per_seed"][str(seed)] for seed in SEEDS),
        scale.get("three_seed_mean_native_logit") or {},
    ]
    if any(
        block.get("n_patients") != EXPECTED_SOURCE_PATIENTS
        or not np.isfinite(float(block.get("refit_to_oof_sd_ratio", np.nan)))
        for block in scale_blocks
    ):
        raise ContractError("Reported refit/OOF scale diagnostic is incomplete or non-finite")

    source_ids = set(pd.read_csv(_source_path(args.output_root))["patient_id"].astype(str))
    rih_ids = set(_load_outcome(args.output_root, "rih_m")["patient_id"].astype(str))
    sr_ids = set(_load_outcome(args.output_root, "sr1482_m")["patient_id"].astype(str))
    if source_ids & rih_ids != set(RIH_DUAL_PATIENTS):
        raise ContractError("RIH direct patient-overlap roster changed")
    if source_ids & sr_ids:
        raise ContractError("SR1482 metastatic patients unexpectedly overlap source patients")
    print(
        f"PASS: {contract['lineage']} is complete and hash-valid: 18 fits, nine scores, "
        "source-only calibration, RIH77/SR1482-M74 report"
    )


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conventional-manifest", type=Path, default=CONVENTIONAL_MANIFEST)
    parser.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    parser.add_argument("--orion-folds", type=Path, default=CPHT_A_FOLDS)
    parser.add_argument("--conventional-splits", type=Path, default=CONVENTIONAL_SPLITS)
    parser.add_argument("--met-input-root", type=Path, default=FINAL_V8_MET_INPUT_ROOT)
    parser.add_argument("--pack-dir", type=Path, default=PACK_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("plan", cmd_plan),
        ("manifest", cmd_manifest),
        ("preflight", cmd_preflight),
    ):
        sub = commands.add_parser(name)
        _add_input_arguments(sub)
        sub.set_defaults(func=function)

    train = commands.add_parser("train")
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--seed", type=int, choices=SEEDS)
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=cmd_train)

    internal = commands.add_parser("_fit")
    internal.add_argument("--output-root", type=Path, required=True)
    internal.add_argument("--seed", type=int, choices=SEEDS, required=True)
    internal.set_defaults(func=lambda args: _fit(args.output_root, args.seed))

    score = commands.add_parser("score")
    score.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    score.add_argument("--target", choices=SCORE_DATASETS)
    score.add_argument("--seed", type=int, choices=SEEDS)
    score.add_argument("--device", default="cuda")
    score.add_argument("--num-workers", type=int, default=TRAINING_NUM_WORKERS)
    score.set_defaults(func=cmd_score)

    report = commands.add_parser("report")
    report.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    report.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    report.add_argument("--e1a-n-bootstrap", type=int, default=E1A_N_BOOTSTRAP)
    report.set_defaults(func=cmd_report)

    verify = commands.add_parser("verify")
    verify.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
