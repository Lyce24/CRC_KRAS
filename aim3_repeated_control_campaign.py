#!/usr/bin/env python3
"""Append-only Aim-3 repeated-control robustness campaign.

This is a standalone correction/sensitivity experiment.  It never writes to
the completed ``outputs/aim1/e3a`` tree.  The 75 fine-task folds are imported
as frozen OOF inputs; only the controls are retrained:

    3 WT draws x 5 controls x 3 model seeds x 5 folds = 225 new folds.

Each WT draw replaces the corresponding fine task's negatives exactly within
``subcohort x frozen outer fold``.  The report uses native logits and a
partial-paired bootstrap: shared positive patients are resampled once for both
arms, while the distinct fine/control negatives are resampled independently.
The primary family-wise rule is a Bonferroni correction over five rung-level
intersection-union tests (one-sided 99% bounds); components and draws are
intersections, so they receive no extra penalty.  A central 99.6667% interval
over all 15 rung-by-draw comparisons is retained only as an ultra-conservative
sensitivity.  A ceiling is reported only when all three draws agree.

Mutating commands require ``--apply`` and a new absolute root below
``outputs/aim1/reruns``.  Every published file is exclusive-create; a partial
campaign is evidence, never something this script silently replaces.

Typical workflow (the root must not already exist at prepare time):

    python aim3_repeated_control_campaign.py preflight --output-root /abs/new/root
    python aim3_repeated_control_campaign.py prepare --output-root /abs/new/root --apply
    python aim3_repeated_control_campaign.py train --output-root /abs/new/root --jobs 6
    python aim3_repeated_control_campaign.py train --output-root /abs/new/root --jobs 6 --apply
    python aim3_repeated_control_campaign.py status --output-root /abs/new/root
    python aim3_repeated_control_campaign.py report --output-root /abs/new/root --apply
    python aim3_repeated_control_campaign.py verify-output --output-root /abs/new/root
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import balance, evaluate, paths  # noqa: E402
from oceanpath.contracts import normalize_slide_id  # noqa: E402
from oceanpath.datasets.packed import validate_packed_dir  # noqa: E402
from oceanpath.splitting import SplitConfig, generate_splits  # noqa: E402
from oceanpath.workflows.training import validate_training_run_dir  # noqa: E402

SCHEMA_VERSION = 2
CAP = 8192
MODEL_SEEDS = (42, 43, 44)
WT_DRAW_SEEDS = (20260823, 20260824, 20260825)
BOOTSTRAP_SEED = 20260826
N_BOOTSTRAP = 20_000
FWER_ALPHA = 0.05
CEILING_BOUND = 0.60
CHANCE = 0.50
FIXED_EPOCH_BUDGET = 12
MAX_JOBS = 6
PACKED_FEATURE_DIM = 1024

PAIRS = (
    ("codon", "ctrl_codon"),
    ("g12d_broad", "ctrl_g12d_broad"),
    ("allele1", "ctrl_allele1"),
    ("allele2", "ctrl_allele2"),
    ("g12c", "ctrl_g12c"),
)
ALLELE_OF = {"allele1": "G12D", "allele2": "G12V", "g12c": "G12C"}
FINE_TASKS = tuple(fine for fine, _ in PAIRS)
CONTROL_FOR = dict(PAIRS)
CONTROLS = tuple(CONTROL_FOR[fine] for fine in FINE_TASKS)
PRIMARY_FAMILY_SIZE = len(FINE_TASKS)
SENSITIVITY_FAMILY_SIZE = len(FINE_TASKS) * len(WT_DRAW_SEEDS)
EXPECTED_FINE_FOLDS = len(FINE_TASKS) * len(MODEL_SEEDS) * paths.N_FOLDS
EXPECTED_CONTROL_CHAINS = len(CONTROLS) * len(WT_DRAW_SEEDS) * len(MODEL_SEEDS)
EXPECTED_CONTROL_FOLDS = EXPECTED_CONTROL_CHAINS * paths.N_FOLDS
DEFAULT_FINE_ROOT = paths.OUTPUT_ROOT / "e3a"

def material_source_files() -> tuple[str, ...]:
    """All repository bytes that can affect preparation, training, or analysis.

    Snapshotting a hand-maintained import subset is unsafe: several material
    imports happen lazily inside the trainer.  Freeze the complete OceanPath
    Python package, every Hydra YAML, both entry points, and the dependency
    declarations instead.  ``verify_prepared`` also compares this inventory to
    the live tree, so adding a new material module after preparation fails
    closed rather than silently escaping the snapshot.
    """

    fixed = ("aim3_repeated_control_campaign.py", "tools/study_train.py", "pyproject.toml", "uv.lock")
    package = tuple(
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "src" / "oceanpath").rglob("*.py"))
    )
    configs = tuple(
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "configs").rglob("*.yaml"))
    )
    values = (*fixed, *package, *configs)
    if len(values) != len(set(values)):
        raise RuntimeError("material source inventory contains duplicate paths")
    return values


SOURCE_FILES = material_source_files()


def _tokens(value: object) -> list[str]:
    return [token.strip() for token in str(value).split(";") if token.strip()]


def is_g12(value: object) -> bool:
    return any(token.startswith("G12") for token in _tokens(value))


def has_allele(value: object, allele: str) -> bool:
    return any(token == allele for token in _tokens(value))


def is_g12d(value: object) -> bool:
    return has_allele(value, "G12D")


def fine_manifest_path(task: str) -> Path:
    return paths.MANIFEST_DIR / f"aim1_e3a_{task}.csv"


def fine_split_dir(task: str) -> Path:
    return REPO / paths.SPLIT_ROOT / f"aim1_e3a_{task}" / paths.SPLIT_NAME


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _file_provenance(path: Path, *, hash_payload: bool) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    record: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if hash_payload:
        record["sha256"] = sha256_file(resolved)
    return record


def _packed_coverage(
    manifests: dict[tuple[str, int], pd.DataFrame], index: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    if "slide_id" not in index or index["slide_id"].isna().any():
        raise RuntimeError("packed index lacks valid slide_id values")
    packed_ids = [normalize_slide_id(value) for value in index["slide_id"].astype(str)]
    if len(packed_ids) != len(set(packed_ids)):
        raise RuntimeError("packed index contains duplicate normalized slide IDs")
    available = set(packed_ids)
    coverage: dict[str, dict[str, Any]] = {}
    for (control, draw_seed), manifest in sorted(manifests.items()):
        selected = {normalize_slide_id(value) for value in manifest["slide_id"].astype(str)}
        missing = sorted(selected - available)
        if missing:
            raise RuntimeError(
                f"{control}/wt{draw_seed}: {len(missing)} selected slides absent from packed store"
            )
        coverage[manifest_name(control, draw_seed)] = {
            "selected_slides": len(selected),
            "missing_slides": 0,
        }
    return coverage


def packed_store_provenance(
    manifests: dict[tuple[str, int], pd.DataFrame], *, hash_payload: bool = True
) -> dict[str, Any]:
    """Validate and bind the packed bytes that the trainer actually reads."""

    pack_dir = paths.PACKED_FEATURE_DIR.resolve(strict=True)
    meta = validate_packed_dir(pack_dir)
    if int(meta.feat_dim) != PACKED_FEATURE_DIM:
        raise RuntimeError(
            f"packed feature dimension is {meta.feat_dim}, expected {PACKED_FEATURE_DIM}"
        )
    if Path(meta.source_dir).resolve(strict=False) != paths.PINNED_FEATURE_DIR.resolve(strict=False):
        raise RuntimeError("packed store source_dir is not the pinned UNI-v1 feature directory")
    names = ["meta.json", "index.parquet", "features.bin"]
    if bool(meta.has_coords):
        names.append("coords.bin")
    files = {
        name: _file_provenance(
            pack_dir / name,
            # Metadata/index are cheap enough to authenticate on every check;
            # multi-GB payloads are fully hashed at preflight and final report.
            hash_payload=hash_payload or name in {"meta.json", "index.parquet"},
        )
        for name in names
    }
    index = pd.read_parquet(pack_dir / "index.parquet")
    coverage = _packed_coverage(manifests, index)
    record: dict[str, Any] = {
        "path": str(pack_dir),
        "meta": {
            "schema_version": int(meta.schema_version),
            "feat_dim": int(meta.feat_dim),
            "feat_dtype": str(meta.feat_dtype),
            "n_slides": int(meta.n_slides),
            "total_patches": int(meta.total_patches),
            "source_dir": str(Path(meta.source_dir).resolve(strict=False)),
            "source_inventory_sha256": str(meta.source_inventory_sha256),
            "has_coords": bool(meta.has_coords),
        },
        "files": files,
        "selected_manifest_coverage": coverage,
    }
    if hash_payload:
        fingerprint_payload = {
            name: {
                "size_bytes": value["size_bytes"],
                "sha256": value["sha256"],
            }
            for name, value in sorted(files.items())
        }
        record["fingerprint_sha256"] = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    return record


def assert_packed_store_matches(
    expected: dict[str, Any],
    manifests: dict[tuple[str, int], pd.DataFrame],
    *,
    full_hash: bool = False,
) -> None:
    observed = packed_store_provenance(manifests, hash_payload=full_hash)
    if observed["path"] != expected.get("path") or observed["meta"] != expected.get("meta"):
        raise RuntimeError("packed store metadata drifted from prepared provenance")
    if observed["selected_manifest_coverage"] != expected.get("selected_manifest_coverage"):
        raise RuntimeError("packed selected-slide coverage drifted from preparation")
    expected_files = expected.get("files")
    if not isinstance(expected_files, dict) or set(observed["files"]) != set(expected_files):
        raise RuntimeError("packed store file inventory drifted from preparation")
    for name, current in observed["files"].items():
        recorded = expected_files[name]
        keys = {"path", "size_bytes", "mtime_ns"}
        if name in {"meta.json", "index.parquet"} or full_hash:
            keys.add("sha256")
        if any(current.get(key) != recorded.get(key) for key in keys):
            raise RuntimeError(f"packed store artifact drifted: {name}")
    if full_hash and observed.get("fingerprint_sha256") != expected.get("fingerprint_sha256"):
        raise RuntimeError("packed store payload fingerprint drifted from preparation")


def _write_bytes_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite Aim-3 campaign artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _write_text_once(path: Path, payload: str) -> None:
    _write_bytes_once(path, payload.encode("utf-8"))


def _write_json_once(path: Path, payload: Any) -> None:
    _write_text_once(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _copy_once(source: Path, destination: Path) -> None:
    _write_bytes_once(destination, source.read_bytes())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_output_root(root: Path, *, must_exist: bool | None = None) -> Path:
    if not root.is_absolute():
        raise ValueError("--output-root must be absolute")
    root = root.resolve(strict=False)
    expected_parent = (paths.OUTPUT_ROOT / "reruns").resolve(strict=False)
    if root.parent != expected_parent or not root.name.startswith("aim3_"):
        raise ValueError(
            f"output root must be one new aim3_* directory directly below {expected_parent}"
        )
    if root == DEFAULT_FINE_ROOT.resolve(strict=False):
        raise ValueError("output root may not overlap the completed E3 tree")
    if must_exist is True and not root.is_dir():
        raise FileNotFoundError(root)
    if must_exist is False and (root.exists() or root.is_symlink()):
        raise FileExistsError(f"new campaign root already exists: {root}")
    return root


def _patient_table(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id", "slide_id", "kras", "kras_subvariant", "cohort",
        "subcohort", "k_fold",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"development manifest lacks columns: {sorted(missing)}")
    constant = ["kras", "kras_subvariant", "cohort", "subcohort", "k_fold"]
    counts = manifest.groupby("patient_id", sort=False)[constant].nunique(dropna=False)
    bad = counts.gt(1).any(axis=1)
    if bad.any():
        raise ValueError(f"patient metadata is inconsistent: {counts.index[bad].tolist()[:5]}")
    patients = manifest.sort_values("slide_id").drop_duplicates("patient_id").copy()
    patients["kras_subvariant"] = patients["kras_subvariant"].fillna("")
    patients["is_g12"] = patients["kras_subvariant"].map(is_g12)
    patients["is_g12d"] = patients["kras_subvariant"].map(is_g12d)
    if not patients["k_fold"].isin(range(paths.N_FOLDS)).all():
        raise ValueError("frozen k_fold must be 0..4")
    return patients


def fine_patient_labels(manifest: pd.DataFrame, task: str) -> pd.DataFrame:
    if task not in FINE_TASKS:
        raise ValueError(task)
    patients = _patient_table(manifest)
    mutant = patients[patients["kras"].eq("mutant")]
    g12 = mutant[mutant["is_g12"]]
    if task == "codon":
        positive, negative = g12, mutant[~mutant["is_g12"]]
    elif task == "g12d_broad":
        positive, negative = mutant[mutant["is_g12d"]], mutant[~mutant["is_g12d"]]
    else:
        allele = ALLELE_OF[task]
        hit = g12["kras_subvariant"].map(lambda value: has_allele(value, allele))
        positive, negative = g12[hit], g12[~hit]
    columns = ["patient_id", "cohort", "subcohort", "k_fold"]
    out = pd.concat(
        [positive[columns].assign(target_label=1), negative[columns].assign(target_label=0)],
        ignore_index=True,
    )
    if not out["patient_id"].is_unique:
        raise ValueError(f"{task}: fine classes overlap")
    return out


def draw_control_patients(
    manifest: pd.DataFrame, fine_task: str, draw_seed: int
) -> pd.DataFrame:
    """Replace fine negatives with WT exactly by subcohort x frozen fold."""
    if draw_seed not in WT_DRAW_SEEDS:
        raise ValueError(f"unregistered WT draw seed: {draw_seed}")
    patients = _patient_table(manifest)
    fine = fine_patient_labels(manifest, fine_task)
    positive = fine[fine["target_label"].eq(1)].copy()
    replaced = fine[fine["target_label"].eq(0)].copy()
    wild_type = patients[patients["kras"].eq("wild_type")].copy()
    rng = np.random.default_rng(draw_seed)
    selected: list[pd.DataFrame] = []
    strata = replaced.groupby(["subcohort", "k_fold"], sort=True).size()
    for (subcohort, fold), requested in strata.items():
        pool = wild_type[
            wild_type["subcohort"].eq(subcohort) & wild_type["k_fold"].eq(fold)
        ].sort_values("patient_id")
        if len(pool) < int(requested):
            raise ValueError(
                f"{fine_task} draw {draw_seed}: {subcohort}/fold{fold} needs "
                f"{requested} WT but only {len(pool)} exist"
            )
        indices = rng.choice(len(pool), size=int(requested), replace=False)
        selected.append(pool.iloc[np.sort(indices)])
    columns = ["patient_id", "cohort", "subcohort", "k_fold"]
    negative = pd.concat(selected, ignore_index=True)[columns].assign(target_label=0)
    out = pd.concat([positive[columns + ["target_label"]], negative], ignore_index=True)
    if not out["patient_id"].is_unique:
        raise ValueError(f"{fine_task} draw {draw_seed}: duplicate/control-overlap patient")
    observed = negative.groupby(["subcohort", "k_fold"], sort=True).size()
    if not observed.equals(strata):
        raise AssertionError(f"{fine_task} draw {draw_seed}: stratum matching failed")
    if len(negative) != len(replaced) or len(positive) != int(fine.target_label.sum()):
        raise AssertionError(f"{fine_task} draw {draw_seed}: class size changed")
    return out.sort_values(["target_label", "subcohort", "k_fold", "patient_id"]).reset_index(
        drop=True
    )


def control_manifest(manifest: pd.DataFrame, fine_task: str, draw_seed: int) -> pd.DataFrame:
    selected = draw_control_patients(manifest, fine_task, draw_seed)
    label = selected[["patient_id", "target_label"]]
    rows = manifest.drop(columns=[c for c in manifest if c.startswith("val_fold_")]).merge(
        label, on="patient_id", how="inner", validate="many_to_one", suffixes=("_old", "")
    )
    if "target_label_old" in rows:
        rows = rows.drop(columns="target_label_old")
    small_class_folds: list[int] = []
    for fold in range(paths.N_FOLDS):
        pool = rows[rows["k_fold"].ne(fold)]
        val = balance.carve_out_validation(
            pool, ["cohort", "target_label"], pool["patient_id"].unique(), paths.ES_VAL_RATIO
        )
        rows[f"val_fold_{fold}"] = (
            rows["patient_id"].isin(val) & rows["k_fold"].ne(fold)
        ).astype(int)
        val_patients = rows[rows[f"val_fold_{fold}"].eq(1)].drop_duplicates("patient_id")
        if int(val_patients["target_label"].sum()) < paths.MIN_ES_VAL_POSITIVES:
            small_class_folds.append(fold)
    rows.attrs["small_class_folds"] = small_class_folds
    _validate_control_manifest(rows, manifest, fine_task, draw_seed)
    return rows.sort_values("slide_id").reset_index(drop=True)


def _validate_control_manifest(
    rows: pd.DataFrame, development: pd.DataFrame, fine_task: str, draw_seed: int
) -> None:
    patients = rows.drop_duplicates("patient_id")
    expected = draw_control_patients(development, fine_task, draw_seed)
    got = patients[["patient_id", "target_label", "subcohort", "k_fold"]].sort_values(
        "patient_id"
    ).reset_index(drop=True)
    want = expected[["patient_id", "target_label", "subcohort", "k_fold"]].sort_values(
        "patient_id"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_dtype=False)
    if rows["slide_id"].duplicated().any():
        raise ValueError("control manifest contains duplicate slides")
    for fold in range(paths.N_FOLDS):
        val = rows[f"val_fold_{fold}"]
        if not val.isin([0, 1]).all() or rows.loc[val.eq(1), "k_fold"].eq(fold).any():
            raise ValueError(f"invalid val_fold_{fold}")


def manifest_name(control: str, draw_seed: int) -> str:
    return f"aim1_aim3rc_{control}_wt{draw_seed}"


def manifest_path(root: Path, control: str, draw_seed: int) -> Path:
    return root / "manifests" / f"{manifest_name(control, draw_seed)}.csv"


def split_dir(root: Path, control: str, draw_seed: int) -> Path:
    return root / "splits" / manifest_name(control, draw_seed) / paths.SPLIT_NAME


def run_dir(root: Path, control: str, draw_seed: int, model_seed: int) -> Path:
    return root / "train" / control / f"wt{draw_seed}" / f"seed{model_seed}"


def frozen_fine_oof(root: Path, fine: str, model_seed: int) -> Path:
    return root / "input_snapshot" / "fine" / fine / f"seed{model_seed}" / "oof_predictions.parquet"


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Tie-correct AUROC from native positive/negative logits."""
    if not len(pos) or not len(neg):
        return float("nan")
    ordered = np.sort(np.asarray(neg, dtype=float))
    positive = np.asarray(pos, dtype=float)
    less = np.searchsorted(ordered, positive, side="left")
    right = np.searchsorted(ordered, positive, side="right")
    return float((less + 0.5 * (right - less)).sum() / (len(pos) * len(neg)))


def _blocks(frame: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    return {
        (str(subcohort), int(fold)): group["mean_logit"].to_numpy(dtype=float)
        for (subcohort, fold), group in frame.groupby(["subcohort", "k_fold"], sort=True)
    }


def _draw_block_matrix(
    blocks: dict[tuple[str, int], np.ndarray], count: int, rng: np.random.Generator
) -> np.ndarray:
    parts = []
    for key in sorted(blocks):
        values = blocks[key]
        indices = rng.integers(0, len(values), size=(count, len(values)))
        parts.append(values[indices])
    return np.concatenate(parts, axis=1)


def partial_paired_bootstrap(
    fine: pd.DataFrame,
    control: pd.DataFrame,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Joint bootstrap pairing shared positives, not distinct negatives.

    Resampling is stratified by subcohort and frozen outer fold.  Within each
    replicate the same sampled positive IDs feed both arms.  Fine and control
    negative indices are independently sampled within the same strata.
    """
    needed = {"patient_id", "label", "mean_logit", "subcohort", "k_fold"}
    if needed - set(fine) or needed - set(control):
        raise ValueError("partial-paired inputs lack required native-logit columns")
    fp = fine[fine["label"].eq(1)].sort_values("patient_id")
    cp = control[control["label"].eq(1)].sort_values("patient_id")
    if fp["patient_id"].tolist() != cp["patient_id"].tolist():
        raise ValueError("fine/control positive patients are not exactly shared")
    if not fp[["subcohort", "k_fold"]].reset_index(drop=True).equals(
        cp[["subcohort", "k_fold"]].reset_index(drop=True)
    ):
        raise ValueError("shared positives disagree on bootstrap strata")
    fn = fine[fine["label"].eq(0)]
    cn = control[control["label"].eq(0)]
    if set(fn["patient_id"]) & set(cn["patient_id"]):
        raise ValueError("fine/control negatives unexpectedly overlap")
    fnb, cnb = _blocks(fn), _blocks(cn)
    if {key: len(value) for key, value in fnb.items()} != {
        key: len(value) for key, value in cnb.items()
    }:
        raise ValueError("negative subcohort/fold counts are not exactly matched")
    # Shared positive IDs are aligned before their common resampling indices.
    fp_values = fp["mean_logit"].to_numpy(dtype=float)
    cp_values = cp["mean_logit"].to_numpy(dtype=float)
    fp_strata = fp[["subcohort", "k_fold"]].astype({"subcohort": str, "k_fold": int})
    pos_groups = {
        key: np.flatnonzero(
            fp_strata["subcohort"].eq(key[0]).to_numpy()
            & fp_strata["k_fold"].eq(key[1]).to_numpy()
        )
        for key in sorted(set(map(tuple, fp_strata.to_numpy())))
    }
    rng = np.random.default_rng(seed)
    fine_values = np.empty(n_bootstrap, dtype=float)
    control_values = np.empty(n_bootstrap, dtype=float)
    cursor = 0
    while cursor < n_bootstrap:
        count = min(chunk_size, n_bootstrap - cursor)
        pos_indices = np.concatenate(
            [
                indices[rng.integers(0, len(indices), size=(count, len(indices)))]
                for _, indices in sorted(pos_groups.items())
            ],
            axis=1,
        )
        fine_neg = _draw_block_matrix(fnb, count, rng)
        control_neg = _draw_block_matrix(cnb, count, rng)
        for row in range(count):
            fine_values[cursor + row] = _auc(fp_values[pos_indices[row]], fine_neg[row])
            control_values[cursor + row] = _auc(
                cp_values[pos_indices[row]], control_neg[row]
            )
        cursor += count
    delta_values = control_values - fine_values
    return {
        "point": {
            "fine": _auc(fp_values, fn["mean_logit"].to_numpy(dtype=float)),
            "control": _auc(cp_values, cn["mean_logit"].to_numpy(dtype=float)),
        },
        "fine_values": fine_values,
        "control_values": control_values,
        "delta_values": delta_values,
        "n_bootstrap": n_bootstrap,
        "design": "shared positives paired; distinct negatives independent; subcohort x k_fold stratified",
    }


def interval(values: np.ndarray, *, alpha: float, family_size: int = 1) -> list[float]:
    tail = alpha / (2 * family_size)
    return [
        float(np.quantile(values, tail, method="linear")),
        float(np.quantile(values, 1.0 - tail, method="linear")),
    ]


def one_sided_bounds(
    values: np.ndarray, *, alpha: float = FWER_ALPHA, family_size: int = PRIMARY_FAMILY_SIZE
) -> dict[str, float]:
    """Bonferroni bounds for the five rung-level intersection-union tests.

    A ceiling claim is an intersection across its three component nulls and
    across all three draws.  No within-rung component/draw penalty is needed:
    under a false rung-level claim at least one required component/draw is null,
    and its one-sided alpha/5 test controls that rung's contribution to FWER.
    """
    tail = alpha / family_size
    return {
        "lower": float(np.quantile(values, tail, method="linear")),
        "upper": float(np.quantile(values, 1.0 - tail, method="linear")),
        "one_sided_confidence": float(1.0 - tail),
    }


def verdict_for(fine_ci: list[float], control_ci: list[float], delta_ci: list[float]) -> str:
    if not np.isfinite(control_ci[0]) or control_ci[0] <= CHANCE:
        return "UNDERPOWERED"
    ceiling = fine_ci[1] < CEILING_BOUND and delta_ci[0] > 0.0
    signal = fine_ci[0] > CHANCE
    if ceiling and signal:
        return "CEILING_WITH_RESIDUAL_SIGNAL"
    if ceiling:
        return "CEILING"
    if signal:
        return "FINE_RESOLUTION_EVIDENCE"
    return "INCONCLUSIVE"


def consensus_verdict(draw_verdicts: Iterable[str]) -> str:
    values = tuple(draw_verdicts)
    if len(values) != len(WT_DRAW_SEEDS):
        raise ValueError("ceiling consensus requires all three predeclared WT draws")
    ceiling = {"CEILING", "CEILING_WITH_RESIDUAL_SIGNAL"}
    if all(value in ceiling for value in values):
        return "CONSENSUS_CEILING"
    if all(value == "UNDERPOWERED" for value in values):
        return "CONSENSUS_UNDERPOWERED"
    return "NO_CEILING_CONSENSUS"


def _validated_training_completion(
    directory: Path, training_identity_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_identity = _read_json(training_identity_path)
    payload = run_identity.get("payload")
    fingerprint = run_identity.get("fingerprint")
    if run_identity.get("schema_version") != 2 or not isinstance(payload, dict):
        raise RuntimeError(f"training identity is malformed: {training_identity_path}")
    recalculated = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]
    if not isinstance(fingerprint, str) or fingerprint != recalculated:
        raise RuntimeError(f"training identity fingerprint is invalid: {training_identity_path}")
    completion = validate_training_run_dir(
        directory,
        expected_fingerprint=fingerprint,
        require_test_predictions=True,
    )
    if completion.get("n_folds") != paths.N_FOLDS or completion.get("skip_finalize") is not False:
        raise RuntimeError(f"training completion recipe is invalid: {directory}")
    fold_records = completion.get("fold_completions")
    expected_paths = [f"fold_{fold}/completion.json" for fold in range(paths.N_FOLDS)]
    if not isinstance(fold_records, list) or [
        str(record.get("path", "")) if isinstance(record, dict) else ""
        for record in fold_records
    ] != expected_paths:
        raise RuntimeError(f"training completion does not bind folds 0..4 exactly: {directory}")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict) or not {
        "cv_summary",
        "best_fold_checkpoint",
    } <= set(artifacts):
        raise RuntimeError(f"training completion lacks required root artifacts: {directory}")
    return completion, run_identity


def _validate_exact_oof_union(directory: Path, oof_path: Path) -> pd.DataFrame:
    """Require OOF to be the byte-value-equivalent union of held-out fold predictions."""

    parts = []
    for fold in range(paths.N_FOLDS):
        part = pd.read_parquet(directory / f"fold_{fold}" / "preds_test.parquet").copy()
        if "fold" in part:
            if not pd.to_numeric(part["fold"], errors="coerce").eq(fold).all():
                raise RuntimeError(f"fold_{fold} test predictions carry a wrong fold column")
        else:
            part["fold"] = fold
        parts.append(part)
    expected = pd.concat(parts, ignore_index=True)
    observed = pd.read_parquet(oof_path)
    if set(observed.columns) != set(expected.columns):
        raise RuntimeError(f"OOF columns differ from held-out fold union: {directory}")
    if "slide_id" not in observed or observed["slide_id"].duplicated().any():
        raise RuntimeError(f"OOF slide IDs are missing or duplicated: {directory}")
    columns = sorted(observed.columns)
    expected = expected.sort_values(["fold", "slide_id"]).reset_index(drop=True)[columns]
    observed = observed.sort_values(["fold", "slide_id"]).reset_index(drop=True)[columns]
    try:
        pd.testing.assert_frame_equal(observed, expected, check_exact=True)
    except AssertionError as error:
        raise RuntimeError(
            f"OOF is not the exact concatenated held-out fold prediction union: {directory}"
        ) from error
    return observed


def _validate_fine_chain(fine_root: Path, task: str, seed: int) -> dict[str, Any]:
    directory = fine_root / "train" / task / f"seed{seed}"
    completion_path = directory / "training_completion.json"
    training_identity_path = directory / "training_identity.json"
    oof_path = directory / "oof_predictions.parquet"
    completion, training_identity = _validated_training_completion(
        directory, training_identity_path
    )
    folds = completion["fold_completions"]
    fold_inventory = []
    for record in folds:
        fold_path = directory / str(record.get("path", ""))
        observed = identity(fold_path)
        if observed["sha256"] != record.get("sha256") or observed["size_bytes"] != record.get(
            "size_bytes"
        ):
            raise RuntimeError(f"fine fold receipt identity changed: {fold_path}")
        fold_inventory.append(observed)
    manifest = fine_manifest_path(task)
    expected_manifest_hash = sha256_file(manifest)
    evidence = training_identity.get("payload", {}).get("input_evidence", {})
    if evidence.get("manifest_sha256") != expected_manifest_hash:
        raise RuntimeError(f"fine manifest no longer matches training identity: {task}/seed{seed}")
    split_integrity = fine_split_dir(task) / ".integrity_hash"
    if evidence.get("split_integrity_sha256") != sha256_file(split_integrity):
        raise RuntimeError(f"fine split no longer matches training identity: {task}/seed{seed}")
    training = training_identity.get("payload", {}).get("material_config", {}).get("training", {})
    expected_recipe = {
        "dataset_max_instances": CAP,
        "eval_full_bags": True,
        "train_sampling_strategy": "patient_natural",
        "seed": seed,
        "sample_weight_column": None,
    }
    for key, expected in expected_recipe.items():
        if training.get(key) != expected:
            raise RuntimeError(f"fine chain recipe drift: {task}/seed{seed}/{key}")
    oof = _validate_exact_oof_union(directory, oof_path)
    fine_manifest = pd.read_csv(manifest)
    if set(oof["slide_id"]) != set(fine_manifest["slide_id"]) or oof["slide_id"].duplicated().any():
        raise RuntimeError(f"fine OOF is not exact slide coverage: {task}/seed{seed}")
    merged = oof.merge(
        fine_manifest[["slide_id", "target_label", "k_fold"]], on="slide_id", validate="one_to_one"
    )
    if not merged["label"].astype(int).eq(merged["target_label"].astype(int)).all():
        raise RuntimeError(f"fine OOF labels disagree with manifest: {task}/seed{seed}")
    if not merged["fold"].astype(int).eq(merged["k_fold"].astype(int)).all():
        raise RuntimeError(f"fine OOF folds disagree with manifest: {task}/seed{seed}")
    if "logit" not in oof or not np.isfinite(pd.to_numeric(oof["logit"])).all():
        raise RuntimeError(f"fine OOF lacks finite native logits: {task}/seed{seed}")
    return {
        "task": task,
        "model_seed": seed,
        "directory": str(directory.resolve()),
        "manifest": identity(manifest),
        "split": identity(fine_split_dir(task) / "splits.parquet"),
        "split_integrity": identity(split_integrity),
        "training_identity": identity(training_identity_path),
        "training_completion": identity(completion_path),
        "oof": identity(oof_path),
        "fold_completions": fold_inventory,
    }


@dataclass
class Preflight:
    development: pd.DataFrame
    manifests: dict[tuple[str, int], pd.DataFrame]
    fine_inventory: list[dict[str, Any]]
    feature_hashes: dict[str, str]
    packed_store: dict[str, Any]


def preflight_inputs(fine_root: Path = DEFAULT_FINE_ROOT) -> Preflight:
    fine_root = fine_root.resolve(strict=True)
    if not fine_root.is_dir():
        raise FileNotFoundError(fine_root)
    development = pd.read_csv(paths.DEV_MANIFEST)
    patient_count = development["patient_id"].nunique()
    if patient_count != paths.EXPECTED_PATIENTS["dev"]["n"]:
        raise RuntimeError(f"development population drifted: {patient_count}")
    manifests: dict[tuple[str, int], pd.DataFrame] = {}
    feature_hashes: dict[str, str] = {}
    available_features = {path.stem for path in paths.PINNED_FEATURE_DIR.glob("*.h5")}
    for fine, control in PAIRS:
        draws = []
        for draw_seed in WT_DRAW_SEEDS:
            frame = control_manifest(development, fine, draw_seed)
            missing = sorted(set(frame["slide_id"].astype(str)) - available_features)
            if missing:
                raise RuntimeError(
                    f"{control}/wt{draw_seed}: {len(missing)} selected feature files missing"
                )
            manifests[(control, draw_seed)] = frame
            draws.append(
                frozenset(
                    frame.drop_duplicates("patient_id")
                    .loc[lambda value: value["target_label"].eq(0), "patient_id"]
                    .astype(str)
                )
            )
        if len(set(draws)) != len(WT_DRAW_SEEDS):
            raise RuntimeError(f"{control}: independent seeds produced duplicate WT draws")
    fine_inventory = [
        _validate_fine_chain(fine_root, task, seed)
        for task in FINE_TASKS
        for seed in MODEL_SEEDS
    ]
    if len(fine_inventory) * paths.N_FOLDS != EXPECTED_FINE_FOLDS:
        raise AssertionError("fine-fit count is not 75")
    # Hash each selected feature inventory exactly as the trainer will.  Use a
    # temporary in-memory CSV surrogate only in prepare; here the equivalent
    # slide/stat inventory is computed directly and stored by campaign name.
    for (control, draw_seed), frame in manifests.items():
        digest = hashlib.sha256()
        for slide_id in sorted(set(frame["slide_id"].astype(str))):
            feature = paths.PINNED_FEATURE_DIR / f"{slide_id}.h5"
            stat = feature.stat()
            digest.update(f"{feature.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        feature_hashes[manifest_name(control, draw_seed)] = digest.hexdigest()
    packed_store = packed_store_provenance(manifests, hash_payload=True)
    return Preflight(development, manifests, fine_inventory, feature_hashes, packed_store)


def _snapshot_sources(root: Path) -> list[dict[str, Any]]:
    records = []
    for relative in SOURCE_FILES:
        live = REPO / relative
        if not live.is_file():
            raise FileNotFoundError(live)
        frozen = root / "source_snapshot" / relative
        _copy_once(live, frozen)
        records.append({"relative_path": relative, "live": identity(live), "frozen": identity(frozen)})
    return records


def _snapshot_fine_inputs(
    root: Path, preflight: Preflight, fine_root: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    _copy_once(paths.DEV_MANIFEST, root / "input_snapshot" / "aim1_dev.csv")
    for task in FINE_TASKS:
        _copy_once(fine_manifest_path(task), root / "input_snapshot" / "fine" / task / "manifest.csv")
        split_source = fine_split_dir(task)
        for name in ("splits.parquet", ".integrity_hash", "split_metadata.json"):
            source = split_source / name
            if source.is_file():
                _copy_once(source, root / "input_snapshot" / "fine" / task / "splits" / name)
        for seed in MODEL_SEEDS:
            source_dir = fine_root / "train" / task / f"seed{seed}"
            destination = root / "input_snapshot" / "fine" / task / f"seed{seed}"
            for name in ("oof_predictions.parquet", "training_identity.json", "training_completion.json"):
                _copy_once(source_dir / name, destination / name)
            for fold in range(paths.N_FOLDS):
                _copy_once(
                    source_dir / f"fold_{fold}" / "completion.json",
                    destination / f"fold_{fold}" / "completion.json",
                )
    # Close the preflight-to-copy race: every frozen object must equal the
    # exact live identity validated before preparation began.
    for record in preflight.fine_inventory:
        task, seed = record["task"], int(record["model_seed"])
        frozen = root / "input_snapshot" / "fine" / task
        comparisons = {
            "manifest": frozen / "manifest.csv",
            "split": frozen / "splits" / "splits.parquet",
            "split_integrity": frozen / "splits" / ".integrity_hash",
            "oof": frozen / f"seed{seed}" / "oof_predictions.parquet",
            "training_identity": frozen / f"seed{seed}" / "training_identity.json",
            "training_completion": frozen / f"seed{seed}" / "training_completion.json",
        }
        for label, frozen_path in comparisons.items():
            if identity(frozen_path)["sha256"] != record[label]["sha256"]:
                raise RuntimeError(
                    f"fine input changed between preflight and snapshot: {task}/seed{seed}/{label}"
                )
        frozen_fold_receipts = sorted(
            (frozen / f"seed{seed}").glob("fold_*/completion.json")
        )
        if len(frozen_fold_receipts) != paths.N_FOLDS:
            raise RuntimeError(f"fine snapshot lacks five fold receipts: {task}/seed{seed}")
        if [identity(path)["sha256"] for path in frozen_fold_receipts] != [
            item["sha256"] for item in record["fold_completions"]
        ]:
            raise RuntimeError(
                f"fine fold receipts changed between preflight and snapshot: {task}/seed{seed}"
            )
    for path in sorted((root / "input_snapshot").rglob("*")):
        if path.is_file():
            records.append(identity(path))
    return records


def _snapshot_packed_inputs(root: Path, preflight: Preflight) -> list[dict[str, Any]]:
    destination = root / "input_snapshot" / "packed_store"
    for name in ("meta.json", "index.parquet"):
        _copy_once(paths.PACKED_FEATURE_DIR / name, destination / name)
    _write_json_once(destination / "provenance.json", preflight.packed_store)
    # Close the provenance-to-copy race cheaply; payload hashes were computed
    # during preflight and size/mtime plus metadata/index hashes must still bind.
    assert_packed_store_matches(preflight.packed_store, preflight.manifests, full_hash=False)
    return [identity(path) for path in sorted(destination.iterdir()) if path.is_file()]


def _prepare_splits(root: Path, preflight: Preflight) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for (control, draw_seed), frame in preflight.manifests.items():
        destination = manifest_path(root, control, draw_seed)
        _write_text_once(destination, frame.to_csv(index=False))
        directory = split_dir(root, control, draw_seed)
        directory.mkdir(parents=True, exist_ok=False)
        generate_splits(
            SplitConfig(
                scheme="predefined_oof_kfold",
                name=paths.SPLIT_NAME,
                csv_path=str(destination),
                output_dir=str(directory),
                filename_column="slide_id",
                label_column="target_label",
                group_column="patient_id",
                fold_column="k_fold",
                n_folds=paths.N_FOLDS,
                seed=paths.PRIMARY_SEED,
            ),
            force=False,
        )
    for path in sorted((root / "manifests").rglob("*")) + sorted((root / "splits").rglob("*")):
        if path.is_file():
            inventory.append(identity(path))
    return inventory


def _prepared_control_manifests(root: Path) -> dict[tuple[str, int], pd.DataFrame]:
    return {
        (control, draw_seed): pd.read_csv(manifest_path(root, control, draw_seed))
        for control in CONTROLS
        for draw_seed in WT_DRAW_SEEDS
    }


def verify_prepared(root: Path, *, verify_live_source: bool = True) -> dict[str, Any]:
    root = validate_output_root(root, must_exist=True)
    start = _read_json(root / "lineage_start.json")
    if start.get("schema_version") != SCHEMA_VERSION or start.get("status") != "prepared":
        raise RuntimeError(f"campaign lineage is not a prepared schema-v{SCHEMA_VERSION} lineage")
    if Path(str(start.get("output_root", ""))).resolve() != root:
        raise RuntimeError("lineage root does not match --output-root")
    if verify_live_source:
        recorded_sources = {
            str(row.get("relative_path")) for row in start.get("source_inventory", [])
        }
        if recorded_sources != set(material_source_files()):
            raise RuntimeError("material source/config/lock inventory drifted from preparation")
    for section in ("source_inventory", "input_inventory", "generated_inventory"):
        rows = start.get(section)
        if not isinstance(rows, list):
            raise RuntimeError(f"lineage_start lacks {section}")
        for row in rows:
            recorded = row["frozen"] if section == "source_inventory" else row
            observed = identity(Path(recorded["path"]))
            if observed != recorded:
                raise RuntimeError(f"prepared identity changed: {recorded['path']}")
            if section == "source_inventory" and verify_live_source:
                live = identity(REPO / row["relative_path"])
                if live["sha256"] != recorded["sha256"]:
                    raise RuntimeError(f"live source drifted from frozen snapshot: {row['relative_path']}")
    packed = start.get("packed_store")
    if not isinstance(packed, dict) or not isinstance(packed.get("fingerprint_sha256"), str):
        raise RuntimeError("lineage_start lacks packed-store provenance")
    prepared_manifests = _prepared_control_manifests(root)
    assert_packed_store_matches(packed, prepared_manifests, full_hash=False)
    return start


def cmd_preflight(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=False)
    result = preflight_inputs(Path(args.fine_root))
    print("PASS — read-only Aim-3 repeated-control preflight")
    print(f"  proposed root: {root}")
    print(f"  fine input: {EXPECTED_FINE_FOLDS}/75 folds complete and immutable-input eligible")
    print(f"  controls: {len(result.manifests)} manifests = 3 draws x 5 controls")
    print(f"  training: {EXPECTED_CONTROL_CHAINS} chains = {EXPECTED_CONTROL_FOLDS} folds")
    print("  WT matching: exact patient counts in every subcohort x frozen outer-fold cell")
    print("  recipe: cap8192, full-bag evaluation, patient-natural training, seeds 42/43/44")
    print("  estimated production: ~3–4 h at --jobs 6, ~15 GB GPU, ~19/36 CPU cores, ~5.8 GB disk")
    print("  no files were written")


def cmd_prepare(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=False)
    preflight = preflight_inputs(Path(args.fine_root))
    if not args.apply:
        print(f"DRY RUN — would exclusively create {root}")
        print("Re-run with --apply after reviewing preflight; no files were written.")
        return
    root.mkdir(parents=True, exist_ok=False)
    source_inventory = _snapshot_sources(root)
    input_inventory = _snapshot_fine_inputs(root, preflight, Path(args.fine_root).resolve())
    input_inventory.extend(_snapshot_packed_inputs(root, preflight))
    generated_inventory = _prepare_splits(root, preflight)
    start = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "output_root": str(root),
        "fine_input_root": str(Path(args.fine_root).resolve()),
        "protocol": {
            "cap": CAP,
            "model_seeds": list(MODEL_SEEDS),
            "wt_draw_seeds": list(WT_DRAW_SEEDS),
            "n_control_chains": EXPECTED_CONTROL_CHAINS,
            "n_control_folds": EXPECTED_CONTROL_FOLDS,
            "fine_folds_reused": EXPECTED_FINE_FOLDS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "primary_fwer": "five rung-level IUTs; Bonferroni alpha=.05/5; one-sided 99% bounds; three-draw consensus",
        },
        "feature_inventory_sha256": preflight.feature_hashes,
        "packed_store": preflight.packed_store,
        "fine_source_inventory": preflight.fine_inventory,
        "source_inventory": source_inventory,
        "input_inventory": input_inventory,
        "generated_inventory": generated_inventory,
    }
    _write_json_once(root / "lineage_start.json", start)
    _write_json_once(
        root / "state" / "events" / "000000_prepared.json",
        {
            "schema_version": SCHEMA_VERSION,
            "event": "prepared",
            "lineage_start": identity(root / "lineage_start.json"),
        },
    )
    verify_prepared(root)
    print(f"PASS — prepared immutable Aim-3 campaign at {root}")


def train_command(
    root: Path,
    control: str,
    draw_seed: int,
    model_seed: int,
    num_workers: int,
    *,
    packed_fingerprint: str,
    attempt_id: str,
) -> list[str]:
    if not packed_fingerprint or not attempt_id or "/" in attempt_id or "\\" in attempt_id:
        raise ValueError("packed fingerprint and a single-component attempt ID are required")
    data_name = manifest_name(control, draw_seed)
    model_id = data_name.removeprefix("aim1_")
    destination = run_dir(root, control, draw_seed, model_seed)
    hydra_dir = (
        root
        / "state"
        / "hydra"
        / control
        / f"wt{draw_seed}"
        / f"seed{model_seed}"
        / attempt_id
    )
    overrides = [
        "platform=colon_workstation",
        "data=aim1",
        f"data.aim1_model={model_id}",
        f"data.name={data_name}",
        f"data.manifest_stem={data_name}",
        f"data.csv_path={manifest_path(root, control, draw_seed)}",
        "+data.cohort_column=cohort",
        "encoder=univ1",
        "splits=aim1_balanced",
        f"+splits.output_dir={split_dir(root, control, draw_seed)}",
        f"splits.seed={model_seed}",
        "model=abmil",
        "model.embed_dim=512",
        "model.attn_dim=384",
        "model.input_dropout=0.10",
        f"model.dropout={paths.DROPOUT}",
        "training=aim1",
        f"training.lr={paths.LR:g}",
        f"training.weight_decay={paths.WEIGHT_DECAY:g}",
        f"training.seed={model_seed}",
        f"training.dataset_max_instances={CAP}",
        "training.eval_full_bags=true",
        f"training.fixed_epoch_budget={FIXED_EPOCH_BUDGET}",
        "training.train_sampling_strategy=patient_natural",
        "training.sample_weight_column=null",
        f"training.packed_dir={paths.PACKED_FEATURE_DIR}",
        "training.verify_packed_source=false",
        f"+training.campaign_packed_store_sha256={packed_fingerprint}",
        f"training.num_workers={num_workers}",
        f"train_dir={destination}",
        f"exp_name=aim3rc_{control}_wt{draw_seed}_seed{model_seed}",
        f"hydra.run.dir={hydra_dir}",
        "hydra.job.chdir=false",
    ]
    return [sys.executable, str(REPO / "tools" / "study_train.py"), "hydra-train", *overrides]


def all_chains() -> list[tuple[str, int, int]]:
    return [
        (control, draw_seed, model_seed)
        for control in CONTROLS
        for draw_seed in WT_DRAW_SEEDS
        for model_seed in MODEL_SEEDS
    ]


def _lock_path(root: Path, name: str) -> Path:
    path = root / "state" / "locks" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _exclusive_lock(path: Path, *, label: str):
    if path.is_symlink():
        raise RuntimeError(f"refusing symlinked {label} lock: {path}")
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "a+") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another process holds the {label} lock: {path}") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    except Exception:
        # fdopen owns and closes the descriptor once constructed.  If fdopen
        # itself failed, close the raw descriptor here.
        with suppress(OSError):
            os.close(descriptor)
        raise


def _campaign_train_lock(root: Path):
    return _exclusive_lock(_lock_path(root, "campaign_train.lock"), label="campaign train")


def _chain_train_lock(root: Path, control: str, draw_seed: int, model_seed: int):
    name = f"{control}_wt{draw_seed}_seed{model_seed}.lock"
    return _exclusive_lock(_lock_path(root, name), label=f"{control}/wt{draw_seed}/seed{model_seed}")


def _validate_control_chain(
    root: Path, control: str, draw_seed: int, model_seed: int
) -> dict[str, Any]:
    lineage = _read_json(root / "lineage_start.json")
    assert_packed_store_matches(
        lineage["packed_store"], _prepared_control_manifests(root), full_hash=False
    )
    directory = run_dir(root, control, draw_seed, model_seed)
    completion_path = directory / "training_completion.json"
    training_identity_path = directory / "training_identity.json"
    oof_path = directory / "oof_predictions.parquet"
    completion, run_identity = _validated_training_completion(directory, training_identity_path)
    fold_records = completion["fold_completions"]
    fold_inventory = []
    for record in fold_records:
        observed = identity(directory / str(record.get("path", "")))
        if observed["sha256"] != record.get("sha256") or observed["size_bytes"] != record.get(
            "size_bytes"
        ):
            raise RuntimeError(f"fold completion changed: {observed['path']}")
        fold_inventory.append(observed)
    payload = run_identity.get("payload", {})
    evidence = payload.get("input_evidence", {})
    expected_manifest = manifest_path(root, control, draw_seed)
    if evidence.get("manifest_sha256") != sha256_file(expected_manifest):
        raise RuntimeError(f"control manifest identity mismatch: {directory}")
    if evidence.get("split_integrity_sha256") != sha256_file(
        split_dir(root, control, draw_seed) / ".integrity_hash"
    ):
        raise RuntimeError(f"control split identity mismatch: {directory}")
    expected_features = _read_json(root / "lineage_start.json")["feature_inventory_sha256"][
        manifest_name(control, draw_seed)
    ]
    if evidence.get("feature_inventory_sha256") != expected_features:
        raise RuntimeError(f"control feature inventory mismatch: {directory}")
    material = payload.get("material_config", {})
    training = material.get("training", {})
    required_recipe = {
        "dataset_max_instances": CAP,
        "eval_full_bags": True,
        "train_sampling_strategy": "patient_natural",
        "seed": model_seed,
        "sample_weight_column": None,
        "fixed_epoch_budget": FIXED_EPOCH_BUDGET,
        "packed_dir": str(paths.PACKED_FEATURE_DIR),
        "verify_packed_source": False,
        "campaign_packed_store_sha256": lineage["packed_store"]["fingerprint_sha256"],
    }
    for key, expected in required_recipe.items():
        if training.get(key) != expected:
            raise RuntimeError(f"{directory}: recipe drift {key}={training.get(key)!r}")
    oof = _validate_exact_oof_union(directory, oof_path)
    manifest = pd.read_csv(expected_manifest)
    merged = oof.merge(
        manifest[["slide_id", "target_label", "k_fold"]], on="slide_id", validate="one_to_one"
    )
    if len(merged) != len(manifest) or set(oof["slide_id"]) != set(manifest["slide_id"]):
        raise RuntimeError(f"control OOF coverage incomplete: {directory}")
    if not merged["label"].astype(int).eq(merged["target_label"].astype(int)).all():
        raise RuntimeError(f"control OOF labels disagree: {directory}")
    if not merged["fold"].astype(int).eq(merged["k_fold"].astype(int)).all():
        raise RuntimeError(f"control OOF folds disagree: {directory}")
    if "logit" not in oof or not np.isfinite(pd.to_numeric(oof["logit"])).all():
        raise RuntimeError(f"control OOF lacks finite native logits: {directory}")
    return {
        "control": control,
        "draw_seed": draw_seed,
        "model_seed": model_seed,
        "training_identity": identity(training_identity_path),
        "training_completion": identity(completion_path),
        "oof": identity(oof_path),
        "fold_completions": fold_inventory,
    }


def _event(root: Path, payload: dict[str, Any]) -> Path:
    name = f"{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}_{uuid.uuid4().hex}.json"
    destination = root / "state" / "events" / name
    _write_json_once(destination, {"schema_version": SCHEMA_VERSION, **payload})
    return destination


def _run_chain(
    root: Path,
    control: str,
    draw_seed: int,
    model_seed: int,
    num_workers: int,
    packed_fingerprint: str,
) -> tuple[str, int, int, int, Path]:
    attempt_id = (
        f"{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}"
        f"_{os.getpid()}_{uuid.uuid4().hex}"
    )
    log_path = (
        root
        / "logs"
        / control
        / f"wt{draw_seed}"
        / f"seed{model_seed}_{attempt_id}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = train_command(
        root,
        control,
        draw_seed,
        model_seed,
        num_workers,
        packed_fingerprint=packed_fingerprint,
        attempt_id=attempt_id,
    )
    with _chain_train_lock(root, control, draw_seed, model_seed):
        lineage = _read_json(root / "lineage_start.json")
        prepared_manifests = _prepared_control_manifests(root)
        assert_packed_store_matches(
            lineage["packed_store"], prepared_manifests, full_hash=False
        )
        _event(
            root,
            {
                "event": "chain_claimed",
                "attempt_id": attempt_id,
                "control": control,
                "draw_seed": draw_seed,
                "model_seed": model_seed,
                "log": str(log_path),
                "hydra_run_dir": next(
                    value.split("=", 1)[1] for value in command if value.startswith("hydra.run.dir=")
                ),
                "command": command,
            },
        )
        with log_path.open("x", encoding="utf-8") as stream:
            code = subprocess.run(
                command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, check=False
            ).returncode
        assert_packed_store_matches(
            lineage["packed_store"], prepared_manifests, full_hash=False
        )
        event: dict[str, Any] = {
            "event": "chain_completed" if code == 0 else "chain_failed",
            "attempt_id": attempt_id,
            "control": control,
            "draw_seed": draw_seed,
            "model_seed": model_seed,
            "exit_code": code,
            "log": identity(log_path),
        }
        if code == 0:
            event["artifacts"] = _validate_control_chain(root, control, draw_seed, model_seed)
        _event(root, event)
    return control, draw_seed, model_seed, code, log_path


def cmd_train(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    if args.jobs < 1 or args.jobs > MAX_JOBS:
        raise ValueError(f"--jobs must be 1..{MAX_JOBS}; six is the validated 24-GB GPU ceiling")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    lock = _campaign_train_lock(root) if args.apply else nullcontext()
    with lock:
        start = verify_prepared(root)
        if (root / "lineage_complete.json").exists():
            raise RuntimeError("campaign is sealed; refusing further training")
        packed_fingerprint = start["packed_store"]["fingerprint_sha256"]
        selected = [
            chain
            for chain in all_chains()
            if (args.control is None or chain[0] == args.control)
            and (args.draw_seed is None or chain[1] == args.draw_seed)
            and (args.model_seed is None or chain[2] == args.model_seed)
        ]
        pending = []
        for chain in selected:
            completion = run_dir(root, *chain) / "training_completion.json"
            if completion.is_file():
                _validate_control_chain(root, *chain)
                print(f"complete — skip {chain[0]} wt{chain[1]} seed{chain[2]}")
            else:
                pending.append(chain)
        print(
            f"{len(pending)} pending chains = {len(pending) * paths.N_FOLDS} folds; "
            f"jobs={args.jobs}"
        )
        if not args.apply:
            for chain in pending:
                attempt_id = f"dryrun_{chain[0]}_wt{chain[1]}_seed{chain[2]}"
                print(
                    shlex.join(
                        train_command(
                            root,
                            *chain,
                            args.num_workers,
                            packed_fingerprint=packed_fingerprint,
                            attempt_id=attempt_id,
                        )
                    )
                )
            print("DRY RUN — no trainer processes were started; pass --apply to execute")
            return
        failures = []
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    _run_chain,
                    root,
                    *chain,
                    args.num_workers,
                    packed_fingerprint,
                ): chain
                for chain in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                control, draw_seed, model_seed, code, log_path = future.result()
                print(
                    f"[{index}/{len(pending)}] {control} wt{draw_seed} seed{model_seed}: "
                    f"{'PASS' if code == 0 else f'FAIL exit={code}'} ({log_path})",
                    flush=True,
                )
                if code:
                    failures.append((control, draw_seed, model_seed, code))
        if failures:
            raise SystemExit(f"failed chains retained for audit: {failures}")
        print("PASS — all selected chains completed")


def _ensemble_from_oof(oof_paths: list[Path], manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    if len(oof_paths) != len(MODEL_SEEDS):
        raise ValueError("three OOF paths are required")
    frames = []
    per_seed: dict[str, float] = {}
    for seed, path in zip(MODEL_SEEDS, oof_paths, strict=True):
        patient = evaluate.to_patient_level(pd.read_parquet(path), manifest).sort_values(
            "patient_id"
        ).reset_index(drop=True)
        frames.append(patient)
        per_seed[str(seed)] = _auc(
            patient.loc[patient["label"].eq(1), "mean_logit"].to_numpy(),
            patient.loc[patient["label"].eq(0), "mean_logit"].to_numpy(),
        )
    reference = frames[0]
    for frame in frames[1:]:
        if reference["patient_id"].tolist() != frame["patient_id"].tolist():
            raise RuntimeError("model seeds cover different OOF patients")
        if not reference["label"].astype(int).equals(frame["label"].astype(int)):
            raise RuntimeError("model seeds disagree on labels")
    out = reference[["patient_id", "label", "subcohort", "k_fold"]].copy()
    out["mean_logit"] = np.mean(
        [frame["mean_logit"].to_numpy(dtype=float) for frame in frames], axis=0
    )
    return out, per_seed


def _fine_ensemble(root: Path, fine: str) -> tuple[pd.DataFrame, dict[str, float]]:
    manifest = pd.read_csv(root / "input_snapshot" / "fine" / fine / "manifest.csv")
    return _ensemble_from_oof(
        [frozen_fine_oof(root, fine, seed) for seed in MODEL_SEEDS], manifest
    )


def _control_ensemble(
    root: Path, control: str, draw_seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    manifest = pd.read_csv(manifest_path(root, control, draw_seed))
    return _ensemble_from_oof(
        [run_dir(root, control, draw_seed, seed) / "oof_predictions.parquet" for seed in MODEL_SEEDS],
        manifest,
    )


def _stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "ci95_two_sided": interval(values, alpha=FWER_ALPHA),
        "primary_fwer_one_sided": one_sided_bounds(values),
        "ultra_conservative_15_comparison_two_sided": interval(
            values, alpha=FWER_ALPHA, family_size=SENSITIVITY_FAMILY_SIZE
        ),
    }


def _build_report(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    chain_inventory = [_validate_control_chain(root, *chain) for chain in all_chains()]
    if len(chain_inventory) != EXPECTED_CONTROL_CHAINS:
        raise AssertionError("report did not validate all 45 control chains")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "output_root": str(root),
        "protocol": {
            "native_score": "mean slide native logit per patient; mean across seeds 42/43/44",
            "bootstrap": "partial-paired patient bootstrap, subcohort x frozen k_fold stratified",
            "shared_component": "positive patient IDs and resampling indices shared across arms",
            "distinct_component": "fine/control negatives independently resampled within matched strata",
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed_base": BOOTSTRAP_SEED,
            "bootstrap_common_random_numbers": (
                "one task-specific bootstrap stream reused across its three draws; "
                "this keeps the frozen fine-arm distribution identical and changes only control data"
            ),
            "primary_fwer": {
                "family": "five rung-level intersection-union ceiling tests",
                "alpha": FWER_ALPHA,
                "per_rung_one_sided_alpha": FWER_ALPHA / PRIMARY_FAMILY_SIZE,
                "bounds": "one-sided 99%",
                "within_rung_penalty": "none: components and three-draw consensus are intersections",
            },
            "sensitivity_only": "central 99.6667% intervals over 15 draw-by-rung comparisons",
            "ceiling_consensus": "all three predeclared WT draws must meet the primary FWER ceiling rule",
        },
        "accounting": {
            "fine_folds_reused": EXPECTED_FINE_FOLDS,
            "control_chains": EXPECTED_CONTROL_CHAINS,
            "control_folds": EXPECTED_CONTROL_FOLDS,
        },
        "rungs": {},
    }
    arrays: dict[str, np.ndarray] = {}
    fine_cache = {fine: _fine_ensemble(root, fine) for fine in FINE_TASKS}
    for task_index, fine in enumerate(FINE_TASKS):
        control = CONTROL_FOR[fine]
        fine_ensemble, fine_per_seed = fine_cache[fine]
        rung: dict[str, Any] = {
            "control": control,
            "fine_per_model_seed_auroc": fine_per_seed,
            "draws": {},
        }
        decisions = []
        for draw_seed in WT_DRAW_SEEDS:
            control_ensemble, control_per_seed = _control_ensemble(root, control, draw_seed)
            bootstrap_seed = int(
                np.random.SeedSequence([BOOTSTRAP_SEED, task_index]).generate_state(1)[0]
            )
            boot = partial_paired_bootstrap(
                fine_ensemble,
                control_ensemble,
                n_bootstrap=N_BOOTSTRAP,
                seed=bootstrap_seed,
            )
            fine_stats = {"auroc": boot["point"]["fine"], **_stats(boot["fine_values"])}
            control_stats = {
                "auroc": boot["point"]["control"],
                **_stats(boot["control_values"]),
            }
            delta_stats = {
                "estimate": boot["point"]["control"] - boot["point"]["fine"],
                **_stats(boot["delta_values"]),
            }
            primary_fine = fine_stats["primary_fwer_one_sided"]
            primary_control = control_stats["primary_fwer_one_sided"]
            primary_delta = delta_stats["primary_fwer_one_sided"]
            verdict = verdict_for(
                [primary_fine["lower"], primary_fine["upper"]],
                [primary_control["lower"], primary_control["upper"]],
                [primary_delta["lower"], primary_delta["upper"]],
            )
            decisions.append(verdict)
            draw_key = str(draw_seed)
            prefix = f"{fine}__wt{draw_seed}"
            arrays[f"{prefix}__fine"] = boot["fine_values"]
            arrays[f"{prefix}__control"] = boot["control_values"]
            arrays[f"{prefix}__delta"] = boot["delta_values"]
            rung["draws"][draw_key] = {
                "bootstrap_seed": bootstrap_seed,
                "control_per_model_seed_auroc": control_per_seed,
                "fine": fine_stats,
                "control": control_stats,
                "delta_control_minus_fine": delta_stats,
                "primary_conditions": {
                    "fine_upper_99_lt_0.60": primary_fine["upper"] < CEILING_BOUND,
                    "control_lower_99_gt_0.50": primary_control["lower"] > CHANCE,
                    "delta_lower_99_gt_0": primary_delta["lower"] > 0.0,
                    "fine_lower_99_gt_0.50": primary_fine["lower"] > CHANCE,
                },
                "primary_verdict": verdict,
            }
        rung["draw_verdicts"] = decisions
        rung["consensus_verdict"] = consensus_verdict(decisions)
        report["rungs"][fine] = rung
    return report, arrays, chain_inventory


def cmd_report(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    start = verify_prepared(root)
    if not args.apply:
        complete = sum(
            (run_dir(root, *chain) / "training_completion.json").is_file()
            for chain in all_chains()
        )
        print(f"DRY RUN — {complete}/{EXPECTED_CONTROL_CHAINS} control chains have completion files")
        print("Report requires and validates all 45 chains; pass --apply only after 45/45.")
        return
    if (root / "lineage_complete.json").exists() or (root / "analysis").exists():
        raise FileExistsError("analysis or completion already exists; refusing to overwrite")
    assert_packed_store_matches(
        start["packed_store"], _prepared_control_manifests(root), full_hash=True
    )
    report, arrays, chain_inventory = _build_report(root)
    # Claim the final directory exclusively only after every statistic has
    # computed.  A publication failure leaves a visible partial lineage and is
    # never repaired by overwriting it; retry under a new campaign root.
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=False, exist_ok=False)
    report_path = analysis_dir / "aim3_repeated_control_report.json"
    _write_json_once(report_path, report)
    bootstrap_path = analysis_dir / "bootstrap_distributions.npz"
    with bootstrap_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "lineage_start": identity(root / "lineage_start.json"),
        "packed_store_fingerprint_sha256": start["packed_store"]["fingerprint_sha256"],
        "control_chains": chain_inventory,
        "report": identity(report_path),
        "bootstrap_distributions": identity(bootstrap_path),
    }
    _write_json_once(analysis_dir / "analysis_audit.json", audit)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "output_root": str(root),
        "artifacts": {
            name: identity(root / "analysis" / name)
            for name in (
                "aim3_repeated_control_report.json",
                "bootstrap_distributions.npz",
                "analysis_audit.json",
            )
        },
    }
    _write_json_once(root / "lineage_complete.json", complete)
    print(json.dumps({fine: block["consensus_verdict"] for fine, block in report["rungs"].items()}, indent=2))
    print(f"PASS — sealed Aim-3 campaign at {root}")


def cmd_verify_output(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    # A sealed campaign remains independently verifiable after the live
    # worktree evolves; training/reporting require live==snapshot, verification
    # authenticates the frozen source identities themselves.
    start = verify_prepared(root, verify_live_source=False)
    assert_packed_store_matches(
        start["packed_store"], _prepared_control_manifests(root), full_hash=True
    )
    complete = _read_json(root / "lineage_complete.json")
    if (
        complete.get("schema_version") != SCHEMA_VERSION
        or complete.get("status") != "completed"
        or Path(complete.get("output_root", "")).resolve() != root
    ):
        raise RuntimeError("lineage completion receipt is invalid")
    expected_artifacts = {
        "aim3_repeated_control_report.json",
        "bootstrap_distributions.npz",
        "analysis_audit.json",
    }
    artifacts = complete.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != expected_artifacts:
        raise RuntimeError("lineage completion artifact inventory is not exact")
    for name, record in artifacts.items():
        if Path(str(record.get("path", ""))).resolve() != root / "analysis" / name:
            raise RuntimeError(f"completed analysis path is invalid: {name}")
        if identity(Path(record["path"])) != record:
            raise RuntimeError(f"completed analysis identity changed: {record['path']}")
    audit = _read_json(root / "analysis" / "analysis_audit.json")
    if (
        audit.get("schema_version") != SCHEMA_VERSION
        or audit.get("status") != "PASS"
        or audit.get("lineage_start") != identity(root / "lineage_start.json")
        or audit.get("report") != identity(root / "analysis" / "aim3_repeated_control_report.json")
        or audit.get("bootstrap_distributions")
        != identity(root / "analysis" / "bootstrap_distributions.npz")
        or audit.get("packed_store_fingerprint_sha256")
        != start["packed_store"]["fingerprint_sha256"]
    ):
        raise RuntimeError("analysis audit receipt is invalid")
    report = _read_json(root / "analysis" / "aim3_repeated_control_report.json")
    distributions = np.load(root / "analysis" / "bootstrap_distributions.npz")
    expected_array_keys = {
        f"{fine}__wt{draw_seed}__{suffix}"
        for fine in FINE_TASKS
        for draw_seed in WT_DRAW_SEEDS
        for suffix in ("fine", "control", "delta")
    }
    if set(distributions.files) != expected_array_keys:
        raise RuntimeError("bootstrap distribution inventory is not exact")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("report schema is invalid")
    if set(report.get("rungs", {})) != set(FINE_TASKS):
        raise RuntimeError("report does not contain exactly five rungs")
    for fine, block in report["rungs"].items():
        draws = block.get("draws", {})
        if set(draws) != {str(seed) for seed in WT_DRAW_SEEDS}:
            raise RuntimeError(f"{fine}: report lacks all three draws")
        fine_draw_arrays = []
        fine_ensemble, expected_fine_per_seed = _fine_ensemble(root, fine)
        for draw_seed in WT_DRAW_SEEDS:
            draw = draws[str(draw_seed)]
            control_ensemble, expected_control_per_seed = _control_ensemble(
                root, CONTROL_FOR[fine], draw_seed
            )
            prefix = f"{fine}__wt{draw_seed}"
            named = {
                "fine": distributions[f"{prefix}__fine"],
                "control": distributions[f"{prefix}__control"],
                "delta_control_minus_fine": distributions[f"{prefix}__delta"],
            }
            if any(len(values) != N_BOOTSTRAP for values in named.values()):
                raise RuntimeError(f"{fine}/wt{draw_seed}: incomplete bootstrap arrays")
            if not np.array_equal(
                named["delta_control_minus_fine"], named["control"] - named["fine"]
            ):
                raise RuntimeError(f"{fine}/wt{draw_seed}: stored delta arithmetic drifted")
            for label, values in named.items():
                expected_stats = _stats(values)
                stored = draw[label]
                for interval_name, expected_interval in expected_stats.items():
                    observed_interval = stored[interval_name]
                    if isinstance(expected_interval, dict):
                        keys = ("lower", "upper", "one_sided_confidence")
                        if not all(
                            np.isclose(observed_interval[key], expected_interval[key]) for key in keys
                        ):
                            raise RuntimeError(
                                f"{fine}/wt{draw_seed}/{label}: primary bounds drifted"
                            )
                    elif not np.allclose(observed_interval, expected_interval):
                        raise RuntimeError(
                            f"{fine}/wt{draw_seed}/{label}: interval drifted"
                        )
            primary_fine = draw["fine"]["primary_fwer_one_sided"]
            primary_control = draw["control"]["primary_fwer_one_sided"]
            primary_delta = draw["delta_control_minus_fine"]["primary_fwer_one_sided"]
            expected_points = {
                "fine": _auc(
                    fine_ensemble.loc[fine_ensemble["label"].eq(1), "mean_logit"].to_numpy(),
                    fine_ensemble.loc[fine_ensemble["label"].eq(0), "mean_logit"].to_numpy(),
                ),
                "control": _auc(
                    control_ensemble.loc[
                        control_ensemble["label"].eq(1), "mean_logit"
                    ].to_numpy(),
                    control_ensemble.loc[
                        control_ensemble["label"].eq(0), "mean_logit"
                    ].to_numpy(),
                ),
            }
            if (
                not np.isclose(draw["fine"]["auroc"], expected_points["fine"])
                or not np.isclose(draw["control"]["auroc"], expected_points["control"])
                or not np.isclose(
                    draw["delta_control_minus_fine"]["estimate"],
                    expected_points["control"] - expected_points["fine"],
                )
                or block.get("fine_per_model_seed_auroc") != expected_fine_per_seed
                or draw.get("control_per_model_seed_auroc") != expected_control_per_seed
            ):
                raise RuntimeError(f"{fine}/wt{draw_seed}: point estimate drifted")
            recalculated = verdict_for(
                [primary_fine["lower"], primary_fine["upper"]],
                [primary_control["lower"], primary_control["upper"]],
                [primary_delta["lower"], primary_delta["upper"]],
            )
            if draw["primary_verdict"] != recalculated:
                raise RuntimeError(f"{fine}/wt{draw_seed}: primary verdict drifted")
            fine_draw_arrays.append(named["fine"])
        if not all(np.array_equal(fine_draw_arrays[0], values) for values in fine_draw_arrays[1:]):
            raise RuntimeError(f"{fine}: frozen fine bootstrap differs across WT draws")
        expected = consensus_verdict(draw["primary_verdict"] for draw in draws.values())
        if block.get("consensus_verdict") != expected:
            raise RuntimeError(f"{fine}: consensus verdict is inconsistent")
    validated_chains = [_validate_control_chain(root, *chain) for chain in all_chains()]
    if audit.get("control_chains") != validated_chains:
        raise RuntimeError("analysis audit control-chain inventory drifted")
    print(
        f"PASS — immutable Aim-3 campaign verified: 75 reused fine folds, "
        f"{EXPECTED_CONTROL_FOLDS} control folds, five FWER-corrected consensus decisions"
    )


def cmd_status(args: argparse.Namespace) -> None:
    root = validate_output_root(Path(args.output_root), must_exist=True)
    start = verify_prepared(root, verify_live_source=False)
    complete, partial, absent = [], [], []
    for chain in all_chains():
        directory = run_dir(root, *chain)
        if (directory / "training_completion.json").is_file():
            complete.append(chain)
        elif directory.exists():
            partial.append(chain)
        else:
            absent.append(chain)
    events = list((root / "state" / "events").glob("*.json"))
    failed_events = sum(_read_json(path).get("event") == "chain_failed" for path in events)
    print(f"lineage: {start['status']} at {root}")
    print(
        f"chains: {len(complete)}/{EXPECTED_CONTROL_CHAINS} complete, "
        f"{len(partial)} partial/resumable, {len(absent)} absent"
    )
    print(f"fold-equivalents complete: {len(complete) * paths.N_FOLDS}/{EXPECTED_CONTROL_FOLDS}")
    print(f"recorded failed attempts: {failed_events}; immutable state events: {len(events)}")
    print(f"sealed: {(root / 'lineage_complete.json').is_file()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("preflight", cmd_preflight), ("prepare", cmd_prepare)):
        command = commands.add_parser(name)
        command.add_argument("--output-root", required=True)
        command.add_argument("--fine-root", default=str(DEFAULT_FINE_ROOT))
        if name == "prepare":
            command.add_argument("--apply", action="store_true")
        command.set_defaults(func=function)
    command = commands.add_parser("train")
    command.add_argument("--output-root", required=True)
    command.add_argument("--control", choices=CONTROLS)
    command.add_argument("--draw-seed", type=int, choices=WT_DRAW_SEEDS)
    command.add_argument("--model-seed", type=int, choices=MODEL_SEEDS)
    command.add_argument("--jobs", type=int, default=MAX_JOBS)
    command.add_argument("--num-workers", type=int, default=4)
    command.add_argument("--apply", action="store_true")
    command.set_defaults(func=cmd_train)
    command = commands.add_parser("report")
    command.add_argument("--output-root", required=True)
    command.add_argument("--apply", action="store_true")
    command.set_defaults(func=cmd_report)
    command = commands.add_parser("verify-output")
    command.add_argument("--output-root", required=True)
    command.set_defaults(func=cmd_verify_output)
    command = commands.add_parser("status")
    command.add_argument("--output-root", required=True)
    command.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
