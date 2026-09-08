#!/usr/bin/env python3
"""Source-only FINAL-v14 offset geometry, sampling and agreement primitives.

``preflight`` checks operational dependencies without selecting any patient or
opening a WSI. Extraction orchestration lives in final_v14_grid_offset_extract.
Every long eligibility check or extraction should run in persistent tmux.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from tools.final_v14_module3 import identity, json_bytes, seal_file, write_json  # noqa: E402

SUBCOHORTS = ("SR1482", "SR386", "TCGA-COAD", "TCGA-READ")
PROTOTYPES = [f"prototype_{p:02d}" for p in range(32)]
SEED = 20260819
PER_SUBCOHORT = 25
DEFAULT_CANONICAL_ROOT = Path("/mnt/wsl/oceanpath-hot/features/colon_stream")
DEFAULT_PRE_READER = REPO / "reports/reruns/final_v14_additions_20260903/e4v_pre_reader"
DEFAULT_UNI = Path("/home/yc_liu/.cache/huggingface/hub/models--MahmoodLab--uni/snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin")
UNI_SHA256 = "56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40"


class OperationalBlock(RuntimeError):
    """Missing archive/runtime is not a scientific insufficient-sample verdict."""


class InsufficientEligiblePatients(ValueError):
    """A complete, authenticated eligibility audit found fewer than 25 in a group."""


def require_canonical_root(root: Path) -> None:
    required = [root, root / "contours_geojson",
                root / "20x_256px_0px_overlap_mpp0.5/patches"]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise OperationalBlock("Canonical archive directories unavailable: " + ", ".join(missing))


def select_source_patients(eligible: pd.DataFrame, *, canonical_root_verified: bool) -> pd.DataFrame:
    """Choose exactly 25 whole-patient eligible IDs per source subcohort.

    The caller must finish a label-free, all-listed-slide eligibility audit.
    One PCG64 stream is consumed in the fixed UTF-8-sorted subcohort order,
    selecting from patient identifiers sorted before sampling. No replacement.
    """
    if not canonical_root_verified:
        raise OperationalBlock("Restore and authenticate the canonical archive before eligibility selection")
    columns = ["patient_id", "subcohort", "eligible"]
    if set(columns) - set(eligible):
        raise ValueError("Eligibility requires patient_id, subcohort and eligible")
    frame = eligible[columns].copy()
    if frame.isna().any().any() or frame.patient_id.duplicated().any():
        raise ValueError("Eligibility must contain unique patients without missing fields")
    if not frame.eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError("Eligibility must be explicitly boolean")
    if set(frame.subcohort) != set(SUBCOHORTS):
        raise ValueError("Eligibility must contain exactly the four registered source subcohorts")
    frame.patient_id = frame.patient_id.astype(str)
    pools = {group: frame.loc[frame.subcohort.eq(group) & frame.eligible].sort_values("patient_id")
             for group in SUBCOHORTS}
    counts = {group: len(pool) for group, pool in pools.items()}
    if any(count < PER_SUBCOHORT for count in counts.values()):
        raise InsufficientEligiblePatients(f"Complete eligible-patient counts: {counts}")
    rng = np.random.default_rng(SEED)
    selected = [pool.iloc[rng.choice(len(pool), PER_SUBCOHORT, replace=False)]
                for pool in pools.values()]
    return pd.concat(selected, ignore_index=True).sort_values("patient_id").reset_index(drop=True)


def shifted_grid_coordinates(attrs: Mapping[str, Any]) -> np.ndarray:
    """Return the diagonal half-stride grid, retaining only complete footprints."""
    required = {"level0_width", "level0_height", "patch_size_level0", "overlap_level0",
                "patch_size", "target_mpp", "level0_mpp"}
    if required - set(attrs):
        raise ValueError(f"Canonical geometry lacks {sorted(required - set(attrs))}")
    integers = {key: int(attrs[key]) for key in ("level0_width", "level0_height", "patch_size_level0", "overlap_level0", "patch_size")}
    if any(float(attrs[key]) != value for key, value in integers.items()):
        raise ValueError("Recorded dimensions and pixel counts must be exact integers")
    width, height = integers["level0_width"], integers["level0_height"]
    footprint, overlap = integers["patch_size_level0"], integers["overlap_level0"]
    mpp = float(attrs["level0_mpp"])
    if width <= 0 or height <= 0 or not np.isfinite(mpp) or mpp <= 0:
        raise ValueError("Invalid slide dimensions or recorded MPP")
    if (integers["patch_size"] != 256 or float(attrs["target_mpp"]) != .5
            or footprint <= 0 or overlap != 0 or footprint != round(256 * .5 / mpp)):
        raise ValueError("Geometry must preserve the canonical exact-MPP 256/.5 nonoverlap footprint")
    stride = footprint - overlap
    origin = stride // 2
    x = np.arange(origin, width - footprint + 1, stride, dtype=np.int64)
    y = np.arange(origin, height - footprint + 1, stride, dtype=np.int64)
    # TRIDENT canonical order is column/x first, row/y second.
    return np.column_stack([np.repeat(x, len(y)), np.tile(y, len(x))])


def shifted_tissue_coordinates(attrs: Mapping[str, Any], mask: Any,
                               *, filter_fn: Callable[..., np.ndarray | None] | None = None) -> np.ndarray:
    """Reuse the canonical unsimplified exact polygon-area tissue filter."""
    if "min_tissue_proportion" not in attrs or float(attrs["min_tissue_proportion"]) != .5:
        raise ValueError("Canonical colon tissue acceptance must be recorded as 0.5")
    if attrs.get("mask_simplification") != "none":
        raise ValueError("Canonical mask geometry must remain unsimplified")
    candidates = shifted_grid_coordinates(attrs)
    if filter_fn is None:
        from oceanpath.extraction.mpp_sampling import _filter_coordinates_by_mask
        filter_fn = _filter_coordinates_by_mask
    result = filter_fn(candidates, mask=mask, footprint=int(attrs["patch_size_level0"]),
                       threshold=float(attrs["min_tissue_proportion"]))
    if result is None:
        raise ValueError("Unsupported canonical polygon mask; no fallback segmentation permitted")
    result = np.asarray(result)
    if result.shape != (len(result), 2) or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("Tissue filter returned invalid integer coordinates")
    allowed = set(map(tuple, candidates.tolist()))
    if len(set(map(tuple, result.tolist()))) != len(result) or any(tuple(row) not in allowed for row in result.tolist()):
        raise ValueError("Tissue filter added or duplicated coordinates outside the shifted grid")
    return result.astype(np.int64, copy=False)


def canonical_quantize(features: np.ndarray) -> np.ndarray:
    """Match the sealed UNI packed-store float16->float32 assignment inputs."""
    array = np.asarray(features)
    if array.ndim != 2 or array.shape[1] != 1024 or not np.isfinite(array).all():
        raise ValueError("Expected finite N x 1024 UNI-v1 features")
    with np.errstate(over="ignore", invalid="ignore"):
        result = array.astype(np.float16).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("UNI features overflow canonical float16 storage")
    return result


def _validate_profiles(frame: pd.DataFrame) -> np.ndarray:
    if set(PROTOTYPES) - set(frame):
        raise ValueError("Every canonical prototype coordinate is required")
    matrix = frame[PROTOTYPES].to_numpy(dtype=float)
    if (not np.isfinite(matrix).all() or (matrix < 0).any()
            or not np.allclose(matrix.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ValueError("Profiles must be finite nonnegative unit abundance masses")
    return matrix


def equal_slide_profiles(slides: pd.DataFrame) -> pd.DataFrame:
    """Average slide abundance equally, never pooling different-sized tile bags."""
    if {"slide_id", "patient_id", "subcohort"} - set(slides) or slides.slide_id.duplicated().any():
        raise ValueError("Unique source slides and patient/subcohort identifiers are required")
    if slides[["slide_id", "patient_id", "subcohort"]].isna().any().any():
        raise ValueError("Missing slide/patient identifiers")
    _validate_profiles(slides)
    if slides.groupby("patient_id").subcohort.nunique().gt(1).any():
        raise ValueError("Patient subcohort changes between slides")
    profiles = slides.groupby(["patient_id", "subcohort"], sort=True)[PROTOTYPES].mean().reset_index()
    count = slides.groupby("patient_id", sort=True).size()
    profiles.insert(2, "n_slides", profiles.patient_id.map(count))
    return profiles.sort_values("patient_id").reset_index(drop=True)


def icc_2_1(original: np.ndarray, offset: np.ndarray) -> dict[str, Any]:
    """Two-way random-effects, absolute-agreement, single-measure ICC(2,1)."""
    a, b = np.asarray(original, dtype=float), np.asarray(offset, dtype=float)
    if a.ndim != 1 or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("ICC requires aligned finite patient vectors")
    result = {"n_patients": len(a), "n_nonzero_original": int(np.count_nonzero(a)),
              "n_nonzero_offset": int(np.count_nonzero(b)), "icc": None}
    if len(a) < 2:
        return result | {"status": "ICC_UNDEFINED_INSUFFICIENT_PATIENTS"}
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return result | {"status": "ICC_UNDEFINED_ZERO_BETWEEN_PATIENT_VARIANCE"}
    matrix = np.column_stack([a, b])
    n, k = matrix.shape
    grand = matrix.mean()
    row_means, column_means = matrix.mean(1), matrix.mean(0)
    ms_patients = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_grids = n * np.sum((column_means - grand) ** 2) / (k - 1)
    residual = matrix - row_means[:, None] - column_means[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denominator = ms_patients + (k - 1) * ms_error + k * (ms_grids - ms_error) / n
    if not np.isfinite(denominator) or denominator == 0:
        return result | {"status": "ICC_UNDEFINED_DENOMINATOR"}
    return result | {"status": "ESTIMABLE", "icc": float((ms_patients - ms_error) / denominator),
                     "ms_patients": float(ms_patients), "ms_grids": float(ms_grids), "ms_error": float(ms_error)}


def paired_profile_summary(original: pd.DataFrame, offset: pd.DataFrame) -> dict[str, Any]:
    """Summarize the exact paired patient roster without selecting favorable cases."""
    for frame in (original, offset):
        if {"patient_id", "subcohort"} - set(frame) or frame.patient_id.duplicated().any():
            raise ValueError("Paired summaries require unique patient identifiers")
    original = original.sort_values("patient_id").reset_index(drop=True)
    offset = offset.sort_values("patient_id").reset_index(drop=True)
    if not original[["patient_id", "subcohort"]].equals(offset[["patient_id", "subcohort"]]):
        raise ValueError("Original and offset profiles must contain exactly the same patient roster")
    a, b = _validate_profiles(original), _validate_profiles(offset)
    cosine = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    patient = original[["patient_id", "subcohort"]].copy()
    patient["cosine_similarity"] = cosine
    return {"n_patients": len(a), "patient_cosine": patient.to_dict("records"),
            "cosine_mean": float(np.mean(cosine)), "cosine_median": float(np.median(cosine)),
            "cosine_min": float(np.min(cosine)),
            "prototype_icc": [{"prototype_id": j, **icc_2_1(a[:, j], b[:, j])} for j in range(32)],
            "role": "descriptive grid-phase sensitivity; cannot select or change any main model or result"}


def execution_contract() -> dict[str, Any]:
    return {"schema_version": 1, "seed": SEED, "subcohort_order": list(SUBCOHORTS),
            "patients_per_subcohort": 25, "patient_count": 100, "random_generator": "PCG64",
            "eligibility": "Label-free; every listed source WSI, canonical HEST mask and required metadata must pass before sampling; no replacement after the draw.",
            "source_grid": "256 output pixels, 0.5 um/pixel; canonical round(256*.5/source_mpp) level0 footprint; zero overlap",
            "offset_grid": "Both origins floor(recorded level0_stride/2), unchanged stride; remove all boundary-crossing footprints before canonical tissue filtering",
            "tissue": "Canonical unsimplified HEST polygons, half-footprint area rule; never regenerate masks",
            "features": "Frozen UNI-v1 eval transforms/eval mode; explicit local checkpoint; float16 then float32 before canonical reference assignment",
            "uni_checkpoint_sha256": UNI_SHA256,
            "profiles": "All32 reference abundance, equal average across each patient's listed slides",
            "agreement": "Per-patient cosine, prototype ICC(2,1); ICC undefined if either grid has zero between-patient variance; show support",
            "operational_block_is_not_scientific_failure": True,
            "inference_role": "Descriptive only; all primary modules and inherited estimates remain unchanged"}


def preflight(canonical_root: Path, pre_reader: Path, uni: Path, output_root: Path | None = None) -> dict[str, Any]:
    """Dependency-only check: no roster draw, raw-slide access, or extraction."""
    missing = []
    try:
        require_canonical_root(canonical_root)
    except OperationalBlock as error:
        missing.append(str(error))
    required_files = [pre_reader / "inputs/tcga_surgen_primary.csv",
                      pre_reader / "profiles/reference/patient_profiles.parquet",
                      pre_reader / "vocabularies/reference/variants/k32_seed20260819/vocabulary.npz", uni]
    missing.extend(str(path) for path in required_files if not path.is_file())
    versions = {}
    for package in ("numpy", "pandas", "torch", "trident", "shapely", "geopandas", "timm", "openslide-python", "h5py"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            missing.append(f"runtime package {package}")
    result = {"schema_version": 1,
              "status": "OPERATIONALLY_BLOCKED" if missing else "DEPENDENCIES_PRESENT_ELIGIBILITY_NOT_CHECKED",
              "scientific_status": "PENDING_NOT_TESTED", "missing_dependencies": missing,
              "canonical_root": str(canonical_root), "runtime_python": sys.executable,
              "runtime_versions": versions, "execution_contract": execution_contract(),
              "runner": identity(Path(__file__)), "patients_sampled": 0, "slides_extracted": 0,
              "eligibility_checked": False, "target_data_accessed": False}
    if output_root is not None:
        output_root = output_root.resolve()
        if not output_root.is_relative_to(REPO / "reports/reruns") or output_root.is_relative_to(DEFAULT_PRE_READER):
            raise ValueError("Write new offset artifacts below reports/reruns, outside the sealed pre-reader")
        digest = hashlib.sha256(json_bytes(result)).hexdigest()
        destination = output_root / "dependency_preflight" / f"{digest}.json"
        write_json(destination, result)
        seal_file(destination)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight"])
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--pre-reader-root", type=Path, default=DEFAULT_PRE_READER)
    parser.add_argument("--uni-checkpoint", type=Path, default=DEFAULT_UNI)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    result = preflight(args.canonical_root, args.pre_reader_root, args.uni_checkpoint, args.output_root)
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "OPERATIONALLY_BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
