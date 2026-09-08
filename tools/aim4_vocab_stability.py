#!/usr/bin/env python3
"""Append-only Aim-4 vocabulary-size and clustering-seed sensitivity.

The completed k=32 atlas remains immutable.  This sensitivity reconstructs the
exact label-blind tile sample from its frozen sampling plan, projects it through
the *frozen k=32 PCA basis*, and varies only:

* vocabulary size: k = 24, 32, 40;
* K-means initialization seed: 20260819, 20260820, 20260821.

Holding the sample and PCA fixed is deliberate: it isolates the two requested
sources of vocabulary instability rather than mixing them with sampling or PCA
instability.  The canonical atlas is reconstructed by assigning the exact
content-hashed sample through its persisted PCA basis and frozen centroids.  Its
indexed cluster masses must remain within strict numerical-reprojection bounds
of the training metadata.  A fresh k=32/seed=20260819 K-means fit remains one of
the nine sensitivity variants; a random seed does not identify a unique local
optimum after replaying persisted PCA coordinates, so that refit is not an
identity control.

M04/prototype 17 and M07/prototype 28 are tracked in two complementary ways:

1. overlap of every sampled tile assigned to the canonical prototype;
2. reassignment of the exact 12 tiles in each completed blinded montage key.

These are computational correspondences, not pathology annotations.  An
alternative cluster must receive a new blinded review before it can inherit a
morphology name from M04 or M07.

The input paths are always explicit.  ``run`` requires ``--apply`` and an
absolute output root that does not exist.  Work is staged in a new sibling
directory and atomically published; failed staging directories are retained as
evidence and are never deleted by this program.

Typical workflow::

    python tools/aim4_vocab_stability.py preflight \
      --canonical-vocab /abs/e3b/vocabulary/vocab_k32.npz \
      --sample-plan /abs/e3b/vocabulary/sample_plan_k32.parquet \
      --feature-root /abs/features_uni_v1 \
      --atlas-report /abs/eval/e3b_atlas_k32.json \
      --review-key /abs/e3b/montages/k32/KEY_do_not_open_before_review.csv \
      --development-manifest /abs/manifests/aim1_dev.csv \
      --corrected-aim4-root /abs/e4_corrected \
      --output-root /abs/reruns/aim4_vocab_stability_v1

    python tools/aim4_vocab_stability.py run [same arguments] --threads 12 --apply
    python tools/aim4_vocab_stability.py verify \
      --output-root /abs/reruns/aim4_vocab_stability_v1 --replay-input
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import datetime as dt
import errno
import hashlib
import io
import json
import multiprocessing as mp
import os
import platform
import struct
import sys
import uuid
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]

# Version 2 separates the model-native float32 inertia diagnostic from the
# deterministic replay identity.  This is intentionally a schema break: v1
# artifacts with the ambiguous ``inertia`` field must not pass this verifier.
SCHEMA_VERSION = 2
K_VALUES = (24, 32, 40)
CLUSTER_SEEDS = (20260819, 20260820, 20260821)
CANONICAL_K = 32
CANONICAL_SEED = 20260819
N_INIT = 10
EXPECTED_COMPONENTS = 64
EXPECTED_NORMALIZE = "l2"
ANCHORS = {17: "M04", 28: "M07"}
COVERAGE_TARGET = 0.80
# Exact typed/block-framed hash produced by ``collect_projected_sample`` for the
# immutable canonical k32 sampling plan and pinned UNIv1 feature corpus.  This
# turns sample reconstruction into an aborting identity check rather than merely
# recording a new hash after the fact.
CANONICAL_SAMPLE_VALUES_SHA256 = (
    "d0ac16fb094b0adc368b325a714e6f4ad08ad10d534ddfe52b70c63d4a4c5e5d"
)
# The original atlas clustered ``PCA.fit_transform(x)`` but persisted only the
# mean/components used later by ``PCA.transform``.  Those mathematically
# equivalent float32 paths move a very small number of boundary assignments.
# These limits allow at most 0.05% total cluster-mass variation and 0.01% in any
# indexed prototype; the observed canonical replay is 0.0130896% and 0.0025172%,
# respectively.  They are identity tolerances, not biological claim thresholds.
CANONICAL_MAX_CLUSTER_MASS_TOTAL_VARIATION = 5e-4
CANONICAL_MAX_SINGLE_CLUSTER_MASS_DELTA = 1e-4
EXPECTED_MONTAGE_TILES = 12
N_BOOTSTRAP = 2_000
FDR_ALPHA = 0.05
ALL_CANONICAL_ANCHORS = tuple(range(CANONICAL_K))

# ``KMeans.inertia_`` is accumulated from float32 residuals by scikit-learn's
# OpenMP implementation.  Its reduction order is a useful fit diagnostic but
# is not a stable replay identity across thread counts.  Seal that native value
# separately from an explicitly reproducible NumPy reduction over the persisted
# float32 sample, assignments, and centroids.
INERTIA_CONTRACT = {
    "model_inertia": "scikit_learn_float32_fit_diagnostic_finite_positive_only",
    "replay_inertia_float64": (
        "numpy_sum_float32_squared_residuals_with_float64_accumulator"
    ),
    "replay_verification": "exact_equality_from_sealed_sample_assignments_centroids",
}

COMPLETION_NAME = "completion_receipt.json"
VERIFICATION_NAME = "verification_receipt.json"
RESULTS_NAME = "results.json"
INPUT_RECEIPT_NAME = "input_receipt.json"
ASSIGNMENTS_NAME = "assignments.npz"
CENTROIDS_NAME = "variant_centroids.npz"
PATIENT_PROFILES_NAME = "development_patient_abundance.parquet"
ASSOCIATIONS_NAME = "association_effects.csv"
CORRESPONDENCE_NAME = "all_anchor_correspondence.csv"


class StabilityError(RuntimeError):
    """A frozen-input, append-only, or numerical contract failed."""


@dataclass(frozen=True)
class Inputs:
    canonical_vocab: Path
    sample_plan: Path
    feature_root: Path
    atlas_report: Path
    review_key: Path
    development_manifest: Path
    corrected_aim4_root: Path
    corrected_numeric_completion: Path
    corrected_numeric_verification: Path
    canonical_patient_profiles: Path
    canonical_specificity: Path

    @property
    def canonical_meta(self) -> Path:
        return self.canonical_vocab.with_suffix(".json")


@dataclass(frozen=True)
class FrozenVocabulary:
    centroids: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    normalize: str
    metadata: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StabilityError(message)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file(), f"expected file input: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_bytes_once(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite Aim-4 sensitivity artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _write_json_once(path: Path, value: Any) -> None:
    _write_bytes_once(path, _json_bytes(value))


def _write_csv_once(path: Path, frame: pd.DataFrame) -> None:
    _write_bytes_once(path, frame.to_csv(index=False).encode("utf-8"))


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    _write_bytes_once(path, buffer.getvalue())


def _write_npz_once(path: Path, arrays: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    _write_bytes_once(path, buffer.getvalue())


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing any destination collision.

    Plain ``os.rename`` may replace an empty destination directory on Linux,
    which is incompatible with the study's no-overwrite contract.  Linux's
    ``renameat2(RENAME_NOREPLACE)`` makes the absence check part of the atomic
    filesystem operation.  Fail closed if the filesystem lacks that primitive.
    """

    at_fdcwd = -100
    rename_noreplace = 1
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    _require(function is not None, "atomic no-replace publication is unavailable")
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(f"refusing to overwrite output root: {destination}")
    raise StabilityError(
        f"atomic no-replace publication failed ({os.strerror(error)}); "
        f"stage retained at {source}"
    )


def _existing_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"{label} must be an explicit absolute path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StabilityError(f"{label} does not resolve: {path}: {exc}") from exc
    _require(resolved.is_file(), f"{label} is not a file: {resolved}")
    return resolved


def _existing_dir(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"{label} must be an explicit absolute path: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise StabilityError(f"{label} does not resolve: {path}: {exc}") from exc
    _require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    return resolved


def resolve_inputs(args: argparse.Namespace) -> Inputs:
    vocab = _existing_file(args.canonical_vocab, "canonical vocabulary")
    _require(vocab.suffix == ".npz", "canonical vocabulary must be an .npz file")
    meta = vocab.with_suffix(".json")
    _require(meta.is_file(), f"canonical vocabulary metadata is missing: {meta}")
    corrected_root = _existing_dir(args.corrected_aim4_root, "corrected Aim-4 root")
    return Inputs(
        canonical_vocab=vocab,
        sample_plan=_existing_file(args.sample_plan, "sample plan"),
        feature_root=_existing_dir(args.feature_root, "feature root"),
        atlas_report=_existing_file(args.atlas_report, "atlas report"),
        review_key=_existing_file(args.review_key, "review key"),
        development_manifest=_existing_file(args.development_manifest, "development manifest"),
        corrected_aim4_root=corrected_root,
        corrected_numeric_completion=_existing_file(
            corrected_root / "numeric_complete.json", "corrected Aim-4 numeric completion"
        ),
        corrected_numeric_verification=_existing_file(
            corrected_root / "numeric_verification.json",
            "corrected Aim-4 numeric verification",
        ),
        canonical_patient_profiles=_existing_file(
            corrected_root / "profiles" / "patient_profiles_k32.parquet",
            "canonical patient profiles",
        ),
        canonical_specificity=_existing_file(
            corrected_root / "analysis" / "specificity_k32.json",
            "canonical specificity report",
        ),
    )


def new_output_root(value: str | Path, inputs: Inputs | None = None) -> Path:
    path = Path(value).expanduser()
    _require(path.is_absolute(), f"output root must be an explicit absolute path: {path}")
    _require(not path.exists() and not path.is_symlink(), f"output root already exists: {path}")
    parent = path.parent.resolve(strict=True)
    output = parent / path.name
    _require(output.name not in {"", ".", ".."}, "invalid output-root name")
    if inputs is not None:
        source = inputs.feature_root
        _require(output != source, f"output root aliases the feature root: {source}")
        _require(source not in output.parents, f"output root is inside the feature root: {source}")
        _require(output not in source.parents, f"feature root is inside the output root: {source}")
        source = inputs.corrected_aim4_root
        _require(output != source, f"output root aliases corrected Aim-4 root: {source}")
        _require(source not in output.parents, f"output root is inside corrected Aim-4 root: {source}")
        _require(output not in source.parents, f"corrected Aim-4 root is inside output root: {source}")
        for source_file in (
            inputs.canonical_vocab,
            inputs.canonical_meta,
            inputs.sample_plan,
            inputs.atlas_report,
            inputs.review_key,
            inputs.development_manifest,
            inputs.corrected_numeric_completion,
            inputs.corrected_numeric_verification,
            inputs.canonical_patient_profiles,
            inputs.canonical_specificity,
        ):
            _require(
                output not in source_file.parents,
                f"input file would be inside output root: {source_file}",
            )
    return output


def load_vocabulary(inputs: Inputs) -> FrozenVocabulary:
    try:
        with np.load(inputs.canonical_vocab, allow_pickle=False) as blob:
            keys = set(blob.files)
            _require(
                keys == {"centroids", "pca_mean", "pca_components"},
                f"unexpected canonical vocabulary arrays: {sorted(keys)}",
            )
            centroids = np.asarray(blob["centroids"], dtype=np.float32)
            mean = np.asarray(blob["pca_mean"], dtype=np.float32)
            components = np.asarray(blob["pca_components"], dtype=np.float32)
        metadata = json.loads(inputs.canonical_meta.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise StabilityError(f"cannot load canonical vocabulary: {exc}") from exc
    _require(isinstance(metadata, dict), "canonical metadata must be a JSON object")
    normalize = str(metadata.get("normalize", ""))
    _require(centroids.shape == (CANONICAL_K, EXPECTED_COMPONENTS), "unexpected centroids shape")
    _require(components.shape[0] == EXPECTED_COMPONENTS, "unexpected PCA rank")
    _require(mean.shape == (components.shape[1],), "PCA mean/components mismatch")
    _require(normalize == EXPECTED_NORMALIZE, f"expected {EXPECTED_NORMALIZE} normalization")
    _require(int(metadata.get("n_prototypes", -1)) == CANONICAL_K, "metadata k is not 32")
    _require(int(metadata.get("n_components", -1)) == EXPECTED_COMPONENTS, "metadata PCA rank mismatch")
    _require(int(metadata.get("seed", -1)) == CANONICAL_SEED, "canonical seed mismatch")
    _require(bool(metadata.get("label_blind")) is True, "vocabulary is not recorded label-blind")
    _require(np.isfinite(centroids).all(), "canonical centroids contain non-finite values")
    _require(np.isfinite(mean).all() and np.isfinite(components).all(), "PCA contains non-finite values")
    return FrozenVocabulary(centroids, mean, components, normalize, metadata)


def _atlas_anchor_map(inputs: Inputs) -> dict[int, str]:
    try:
        atlas = json.loads(inputs.atlas_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StabilityError(f"cannot read atlas report: {exc}") from exc
    _require(isinstance(atlas, dict) and int(atlas.get("k", -1)) == CANONICAL_K, "atlas is not k=32")
    review = atlas.get("pathology_review")
    _require(isinstance(review, dict) and review.get("status") == "complete", "pathology review is not complete")
    raw = review.get("montage_by_prototype")
    _require(isinstance(raw, dict), "atlas lacks montage_by_prototype")
    mapping = {int(key): str(value) for key, value in raw.items()}
    for prototype, montage in ANCHORS.items():
        _require(mapping.get(prototype) == montage, f"atlas anchor mismatch: p{prototype} != {montage}")
    recorded_key_hash = review.get("base_packet", {}).get("key_sha256")
    _require(
        recorded_key_hash == sha256_file(inputs.review_key),
        "review key does not match the completed atlas report",
    )
    return mapping


def load_review_rows(inputs: Inputs, *, strict: bool = True) -> pd.DataFrame:
    try:
        key = pd.read_csv(inputs.review_key)
    except Exception as exc:
        raise StabilityError(f"cannot read review key: {exc}") from exc
    required = {"montage_id", "slot", "prototype", "slide_id", "tile_index"}
    _require(required <= set(key.columns), f"review key lacks columns {sorted(required - set(key.columns))}")
    key = key[key["montage_id"].isin(ANCHORS.values())].copy()
    _require(set(key["montage_id"]) == set(ANCHORS.values()), "review key lacks one or both anchor montages")
    key["prototype"] = pd.to_numeric(key["prototype"], errors="raise").astype(int)
    key["slot"] = pd.to_numeric(key["slot"], errors="raise").astype(int)
    key["tile_index"] = pd.to_numeric(key["tile_index"], errors="raise").astype(int)
    expected_inverse = {montage: prototype for prototype, montage in ANCHORS.items()}
    for montage, block in key.groupby("montage_id"):
        _require(set(block["prototype"]) == {expected_inverse[str(montage)]}, f"{montage} prototype mismatch")
        _require(not block["slot"].duplicated().any(), f"{montage} repeats a slot")
        if strict:
            _require(len(block) == EXPECTED_MONTAGE_TILES, f"{montage} must contain 12 reviewed tiles")
            _require(set(block["slot"]) == set(range(EXPECTED_MONTAGE_TILES)), f"{montage} slots are incomplete")
    return key.sort_values(["montage_id", "slot"], kind="stable").reset_index(drop=True)


def load_development_manifest(inputs: Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the explicit E0 manifest and mechanically reproduce E4 A/D flags."""

    try:
        slides = pd.read_csv(inputs.development_manifest)
    except Exception as exc:
        raise StabilityError(f"cannot read development manifest: {exc}") from exc
    required = {
        "slide_id",
        "patient_id",
        "target_label",
        "kras",
        "msi_dmmr",
        "braf",
        "cohort",
        "subcohort",
        "specimen_role",
    }
    _require(required <= set(slides.columns), f"development manifest lacks {sorted(required - set(slides.columns))}")
    _require(len(slides) > 0 and not slides["slide_id"].duplicated().any(), "development slide IDs are empty or duplicated")
    _require(set(slides["specimen_role"].dropna().astype(str)) == {"primary"}, "development manifest is not primary-only")
    consistency_columns = [
        "target_label",
        "kras",
        "msi_dmmr",
        "braf",
        "cohort",
        "subcohort",
    ]
    inconsistent = [
        column
        for column in consistency_columns
        if int(slides.groupby("patient_id")[column].nunique(dropna=False).max()) != 1
    ]
    _require(not inconsistent, f"development patient metadata varies across slides: {inconsistent}")
    patients = slides.drop_duplicates("patient_id").copy()
    patients["target_label"] = pd.to_numeric(patients["target_label"], errors="raise").astype(int)
    patients["is_mutant"] = patients["kras"].eq("mutant").astype(int)
    _require(
        np.array_equal(patients["target_label"].to_numpy(), patients["is_mutant"].to_numpy()),
        "target_label and KRAS state disagree in development manifest",
    )
    # Exact definitions in e4.dev_patients().
    patients["in_D"] = (
        patients["msi_dmmr"].eq("MSS/pMMR") & patients["braf"].eq("wild_type")
    ).astype(int)
    return slides.reset_index(drop=True), patients.reset_index(drop=True)


def validate_corrected_aim4_numeric_seal(inputs: Inputs) -> dict[str, Any]:
    """Require the immutable, fully replay-verified corrected numeric lineage.

    Pathology review intentionally remains pending at this stage.  This
    sensitivity consumes only the sealed numeric patient profiles and
    specificity analysis; it neither consumes nor transfers pathology labels.
    """

    completion = json.loads(inputs.corrected_numeric_completion.read_text(encoding="utf-8"))
    _require(isinstance(completion, dict), "corrected Aim-4 numeric completion is not an object")
    _require(
        completion.get("schema_version") == 1
        and completion.get("component") == "aim4_corrected_cap8192"
        and completion.get("status") == "numeric_completed_pathology_review_pending",
        "corrected Aim-4 numeric lineage is not sealed with pathology review pending",
    )
    _require(
        completion.get("validation", {}).get("status") == "PASS"
        and completion.get("validation", {}).get("pathology_review")
        == "PENDING_EXPLICIT_COMPLETED_FORMS"
        and completion.get("pathology_status")
        == "fresh corrected blinded packets sealed; completed review pending",
        "corrected Aim-4 numeric completion does not declare a passing numeric validation "
        "with pathology review pending",
    )
    _require(
        Path(str(completion.get("output_root", ""))).resolve(strict=True)
        == inputs.corrected_aim4_root,
        "corrected Aim-4 numeric completion root mismatch",
    )
    artifacts = completion.get("artifacts")
    _require(
        isinstance(artifacts, dict) and bool(artifacts),
        "corrected numeric completion lacks artifacts",
    )
    aggregate = hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _require(
        aggregate == completion.get("aggregate_sha256"),
        "corrected numeric completion aggregate mismatch",
    )
    for relative, path in (
        ("profiles/patient_profiles_k32.parquet", inputs.canonical_patient_profiles),
        ("analysis/specificity_k32.json", inputs.canonical_specificity),
    ):
        record = artifacts.get(relative)
        _require(isinstance(record, dict), f"corrected numeric completion lacks {relative}")
        observed = identity(path)
        _require(
            int(record.get("size_bytes", -1)) == observed["size_bytes"]
            and record.get("sha256") == observed["sha256"],
            f"corrected completed artifact drifted: {relative}",
        )

    verification = json.loads(
        inputs.corrected_numeric_verification.read_text(encoding="utf-8")
    )
    _require(
        isinstance(verification, dict),
        "corrected Aim-4 numeric verification is not an object",
    )
    _require(
        verification.get("schema_version") == 1
        and verification.get("component") == "aim4_corrected_cap8192"
        and verification.get("status") == "PASS"
        and verification.get("receipt_role")
        == "immutable_numeric_verification_addendum",
        "corrected Aim-4 numeric verification is not an immutable PASS addendum",
    )
    _require(
        Path(str(verification.get("output_root", ""))).resolve(strict=True)
        == inputs.corrected_aim4_root,
        "corrected Aim-4 numeric verification root mismatch",
    )
    _require(
        verification.get("checks", {}).get("deterministic_statistical_replay") == "PASS"
        and verification.get("replay", {}).get("status") == "PASS",
        "corrected Aim-4 numeric verification did not perform a passing full statistical replay",
    )
    completion_record = verification.get("numeric_completion", {})
    observed_completion = identity(inputs.corrected_numeric_completion)
    _require(
        completion_record == observed_completion,
        "corrected numeric verification does not exactly bind the supplied numeric completion",
    )
    _require(
        verification.get("numeric_aggregate_sha256") == aggregate,
        "corrected numeric verification does not bind the numeric artifact aggregate",
    )
    return {
        "status": "PASS",
        "pathology_status": "PENDING; NO PATHOLOGY LABELS CONSUMED OR TRANSFERRED",
        "numeric_completion_sha256": observed_completion["sha256"],
        "numeric_verification_sha256": sha256_file(
            inputs.corrected_numeric_verification
        ),
        "numeric_aggregate_sha256": aggregate,
    }


def load_canonical_references(inputs: Inputs) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        profiles = pd.read_parquet(inputs.canonical_patient_profiles)
    except Exception as exc:
        raise StabilityError(f"cannot read canonical patient profiles: {exc}") from exc
    required = {"arm", "patient_id", "prototype", "abundance"}
    _require(required <= set(profiles.columns), f"canonical profiles lack {sorted(required - set(profiles.columns))}")
    profiles = profiles[profiles["arm"].eq("e0")].copy()
    profiles["prototype"] = pd.to_numeric(profiles["prototype"], errors="raise").astype(int)
    _require(set(profiles["prototype"]) == set(ALL_CANONICAL_ANCHORS), "canonical profiles lack one or more k32 prototypes")
    _require(
        not profiles.duplicated(["patient_id", "prototype"]).any(),
        "canonical e0 profiles duplicate patient/prototype rows",
    )
    try:
        specificity = json.loads(inputs.canonical_specificity.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StabilityError(f"cannot read canonical specificity report: {exc}") from exc
    _require(isinstance(specificity, dict) and int(specificity.get("k", -1)) == CANONICAL_K, "canonical specificity is not k32")
    _require(
        specificity.get("component") == "aim4_corrected_cap8192",
        "canonical specificity is not the corrected cap-8192 artifact",
    )
    _require(
        specificity.get("protocol", {}).get("bh_family")
        == "fixed prototypes 0..31, including structural p=1 rows",
        "canonical specificity lacks the corrected fixed-32 BH declaration",
    )
    _require(
        set(map(int, specificity.get("prototypes", {}))) == set(ALL_CANONICAL_ANCHORS),
        "canonical specificity lacks one or more prototypes",
    )
    for prototype in ALL_CANONICAL_ANCHORS:
        for population in ("A", "D"):
            effect = specificity["prototypes"][str(prototype)]["abundance"][population]
            _require(
                {"estimable", "p", "q", "significant"} <= set(effect),
                f"canonical specificity p{prototype}/{population} lacks corrected fields",
            )
            _require(
                np.isfinite([float(effect["p"]), float(effect["q"])]).all(),
                f"canonical specificity p{prototype}/{population} has non-finite p/q",
            )
            if not effect["estimable"]:
                _require(
                    effect.get("auc") is None
                    and float(effect["p"]) == 1.0
                    and not bool(effect["significant"]),
                    f"canonical specificity p{prototype}/{population} structural policy differs",
                )
    return profiles.reset_index(drop=True), specificity


def inspect_inputs(inputs: Inputs, *, strict_montages: bool = True) -> dict[str, Any]:
    """Validate frozen inputs without writing anything."""

    vocab = load_vocabulary(inputs)
    _atlas_anchor_map(inputs)
    corrected_lineage = validate_corrected_aim4_numeric_seal(inputs)
    review_rows = load_review_rows(inputs, strict=strict_montages)
    development_slides, development_patients = load_development_manifest(inputs)
    canonical_profiles, canonical_specificity = load_canonical_references(inputs)
    try:
        plan = pd.read_parquet(inputs.sample_plan)
    except Exception as exc:
        raise StabilityError(f"cannot read sample plan: {exc}") from exc
    required = {"slide_id", "patient_id", "atlas_group", "n_tiles", "n_sample"}
    _require(required <= set(plan.columns), f"sample plan lacks {sorted(required - set(plan.columns))}")
    _require(len(plan) > 0 and not plan["slide_id"].duplicated().any(), "sample plan slide IDs are empty or duplicated")
    for column in ("n_tiles", "n_sample"):
        plan[column] = pd.to_numeric(plan[column], errors="raise").astype(int)
    _require((plan["n_tiles"] > 0).all(), "sample plan contains a non-positive tile count")
    _require((plan["n_sample"] > 0).all(), "sample plan contains a non-positive draw")
    _require((plan["n_sample"] <= plan["n_tiles"]).all(), "sample plan oversamples a slide")
    expected_sample = int(vocab.metadata.get("n_sample_tiles", -1))
    _require(int(plan["n_sample"].sum()) == expected_sample, "sample-plan total differs from vocabulary metadata")
    expected_slides = int(vocab.metadata.get("corpus_slides", -1))
    _require(len(plan) == expected_slides, "sample-plan slide count differs from vocabulary metadata")
    _require(
        int(plan["patient_id"].nunique()) == int(vocab.metadata.get("corpus_patients", -1)),
        "sample-plan patient count differs from vocabulary metadata",
    )
    _require(
        sorted(plan["atlas_group"].astype(str).unique()) == sorted(vocab.metadata.get("groups", [])),
        "sample-plan groups differ from vocabulary metadata",
    )
    _require(
        set(development_slides["slide_id"]) <= set(plan["slide_id"]),
        "one or more E0 development slides are absent from the frozen vocabulary plan",
    )
    _require(
        set(canonical_profiles["patient_id"]) == set(development_patients["patient_id"]),
        "canonical profile and development-manifest patient sets differ",
    )
    populations = canonical_specificity.get("populations", {})
    _require(
        int(populations.get("A", -1)) == int(len(development_patients)),
        "canonical specificity A count differs from development manifest",
    )
    _require(
        int(populations.get("D", -1)) == int(development_patients["in_D"].sum()),
        "canonical specificity D count differs from development manifest",
    )

    review_by_slide = review_rows.groupby("slide_id")["tile_index"].max().to_dict()
    inventory_rows: list[dict[str, Any]] = []
    feature_dim = int(vocab.pca_mean.size)
    largest_bytes = 0
    for row in plan.itertuples(index=False):
        feature_path = inputs.feature_root / f"{row.slide_id}.h5"
        _require(feature_path.is_file(), f"missing feature file: {feature_path}")
        resolved = feature_path.resolve(strict=True)
        stat = resolved.stat()
        largest_bytes = max(largest_bytes, int(stat.st_size))
        try:
            with h5py.File(resolved, "r") as handle:
                _require("features" in handle, f"features dataset missing: {resolved}")
                dataset = handle["features"]
                shape = tuple(int(x) for x in dataset.shape)
                dtype = str(dataset.dtype)
        except OSError as exc:
            raise StabilityError(f"cannot inspect {resolved}: {exc}") from exc
        _require(shape == (int(row.n_tiles), feature_dim), f"feature shape drift: {resolved}: {shape}")
        _require(dtype == "float32", f"feature dtype drift: {resolved}: {dtype}")
        if row.slide_id in review_by_slide:
            _require(int(review_by_slide[row.slide_id]) < shape[0], f"review tile index out of range: {row.slide_id}")
        inventory_rows.append(
            {
                "slide_id": str(row.slide_id),
                "path": str(resolved),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "shape": list(shape),
                "dtype": dtype,
            }
        )

    inventory_payload = _json_bytes(inventory_rows)
    sample_rows = int(plan["n_sample"].sum())
    projected_bytes = sample_rows * EXPECTED_COMPONENTS * np.dtype(np.float32).itemsize
    checked_rows, checked_z, reviewed_feature_hash = collect_review_projection(
        inputs, vocab, strict=strict_montages
    )
    reviewed_labels = assign_nearest(checked_z, vocab.centroids)
    anchor_assignment_check: dict[str, Any] = {}
    for prototype, montage in ANCHORS.items():
        positions = np.flatnonzero(checked_rows["montage_id"].eq(montage).to_numpy())
        observed = reviewed_labels[positions]
        _require(
            np.all(observed == prototype),
            f"{montage} key tiles do not all assign to frozen canonical p{prototype}",
        )
        anchor_assignment_check[montage] = {
            "canonical_prototype": int(prototype),
            "n_tiles": int(len(positions)),
            "n_assigning_to_canonical_prototype": int(np.sum(observed == prototype)),
            "status": "PASS",
        }
    return {
        "status": "PASS",
        "design": {
            "k_values": list(K_VALUES),
            "cluster_seeds": list(CLUSTER_SEEDS),
            "n_init": N_INIT,
            "sample_and_pca": "frozen_from_canonical_k32",
            "kmeans_execution": (
                "independent variants in forked processes; exactly one "
                "BLAS/OpenMP thread per fit"
            ),
            "pathology_label_transfer": "PROHIBITED_WITHOUT_NEW_BLINDED_REVIEW",
            "matched_set_rule": "smallest label-blind alternative-cluster set covering at least 80% of each canonical anchor's sampled tiles",
            "association_populations": "full E0 A and dependency-restricted D",
            "association_family": "exactly 32 canonical-anchor matched sets per variant and population; structural p=1 retained",
            "claim_gate": (
                "p17 and p28 are evaluated separately; each prototype is robust only "
                "if it is estimable, positive, has CI-low>0.5, and has BH-q<0.05 "
                "in A and D for all nine variants"
            ),
        },
        "counts": {
            "sample_plan_slides": int(len(plan)),
            "sample_plan_patients": int(plan["patient_id"].nunique()),
            "sample_plan_groups": int(plan["atlas_group"].nunique()),
            "sampled_tiles": sample_rows,
            "reviewed_anchor_tiles": int(len(review_rows)),
            "variants": len(K_VALUES) * len(CLUSTER_SEEDS),
            "development_slides": int(len(development_slides)),
            "development_patients_A": int(len(development_patients)),
            "development_patients_D": int(development_patients["in_D"].sum()),
            "matched_anchor_tests_per_family": CANONICAL_K,
        },
        "resource_basis": {
            "feature_store_bytes": int(sum(row["size_bytes"] for row in inventory_rows)),
            "largest_feature_file_bytes": largest_bytes,
            "projected_matrix_bytes": int(projected_bytes),
            "recommended_peak_ram_gib": 8,
            "gpu_required": False,
            "production_runtime_estimate": "provision 2-4 h alone at an 18-physical-core cap; nine deterministic single-thread K-means variants run concurrently; confirm from first-run phase logs; do not co-run with GPU training",
        },
        "input_identities": {
            "canonical_vocab": identity(inputs.canonical_vocab),
            "canonical_meta": identity(inputs.canonical_meta),
            "sample_plan": identity(inputs.sample_plan),
            "atlas_report": identity(inputs.atlas_report),
            "review_key": identity(inputs.review_key),
            "development_manifest": identity(inputs.development_manifest),
            "corrected_aim4_numeric_completion": identity(
                inputs.corrected_numeric_completion
            ),
            "corrected_aim4_numeric_verification": identity(
                inputs.corrected_numeric_verification
            ),
            "canonical_patient_profiles": identity(inputs.canonical_patient_profiles),
            "canonical_specificity": identity(inputs.canonical_specificity),
        },
        "corrected_aim4_numeric_seal": corrected_lineage,
        "feature_inventory": inventory_rows,
        "feature_inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
        "reviewed_feature_values_sha256": reviewed_feature_hash,
        "canonical_reviewed_tile_assignment": anchor_assignment_check,
    }


def _project(features: np.ndarray, vocab: FrozenVocabulary) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if vocab.normalize == "l2":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norms, 1e-12)
    return np.asarray((x - vocab.pca_mean) @ vocab.pca_components.T, dtype=np.float32)


def collect_projected_sample(
    inputs: Inputs,
    vocab: FrozenVocabulary,
    *,
    progress: bool = True,
) -> tuple[np.ndarray, str]:
    """Reconstruct the canonical sample and content-hash the exact used values."""

    plan = pd.read_parquet(inputs.sample_plan)
    n_rows = int(plan["n_sample"].sum())
    feature_dim = int(vocab.pca_mean.size)
    digest = hashlib.sha256()
    digest.update(b"aim4-exact-sampled-features-v1\0")
    digest.update(struct.pack("<qq", n_rows, feature_dim))
    rng = np.random.default_rng(CANONICAL_SEED)
    projected: list[np.ndarray] = []
    observed = 0
    for index, row in enumerate(plan.itertuples(index=False), start=1):
        path = inputs.feature_root / f"{row.slide_id}.h5"
        with h5py.File(path, "r") as handle:
            features = np.asarray(handle["features"][:], dtype=np.float32)
        n = int(row.n_sample)
        _require(features.shape[0] == int(row.n_tiles), f"feature count drift during collection: {row.slide_id}")
        picks = rng.choice(features.shape[0], size=n, replace=False)
        selected = np.ascontiguousarray(features[picks, :], dtype="<f4")
        digest.update(struct.pack("<q", n))
        digest.update(selected.tobytes(order="C"))
        projected.append(_project(selected, vocab))
        observed += n
        if progress and index % 200 == 0:
            print(f"  reconstructed {index}/{len(plan)} feature files", flush=True)
    _require(observed == n_rows, "reconstructed sample row count mismatch")
    z = np.concatenate(projected, axis=0)
    _require(z.shape == (n_rows, EXPECTED_COMPONENTS), "projected sample shape mismatch")
    _require(np.isfinite(z).all(), "projected sample contains non-finite values")
    return z, digest.hexdigest()


def collect_review_projection(
    inputs: Inputs,
    vocab: FrozenVocabulary,
    *,
    strict: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    rows = load_review_rows(inputs, strict=strict)
    digest = hashlib.sha256()
    digest.update(b"aim4-reviewed-anchor-features-v1\0")
    blocks: list[np.ndarray] = []
    for row in rows.itertuples(index=False):
        path = inputs.feature_root / f"{row.slide_id}.h5"
        with h5py.File(path, "r") as handle:
            selected = np.ascontiguousarray(handle["features"][int(row.tile_index)], dtype="<f4").reshape(1, -1)
        digest.update(str(row.montage_id).encode() + b"\0")
        digest.update(struct.pack("<qq", int(row.slot), int(row.tile_index)))
        digest.update(selected.tobytes(order="C"))
        blocks.append(_project(selected, vocab))
    z = np.concatenate(blocks, axis=0)
    return rows, z, digest.hexdigest()


def assign_nearest(z: np.ndarray, centroids: np.ndarray, *, batch_size: int = 65_536) -> np.ndarray:
    labels = np.empty(len(z), dtype=np.int16)
    penalties = 0.5 * np.sum(np.asarray(centroids, dtype=np.float32) ** 2, axis=1)
    for start in range(0, len(z), batch_size):
        stop = min(start + batch_size, len(z))
        scores = z[start:stop] @ centroids.T - penalties
        labels[start:stop] = np.argmax(scores, axis=1).astype(np.int16)
    return labels


def contingency(reference: np.ndarray, candidate: np.ndarray, n_ref: int, n_candidate: int) -> np.ndarray:
    ref = np.asarray(reference, dtype=int)
    cand = np.asarray(candidate, dtype=int)
    _require(ref.shape == cand.shape, "assignment arrays differ in length")
    _require((ref >= 0).all() and (ref < n_ref).all(), "reference label outside range")
    _require((cand >= 0).all() and (cand < n_candidate).all(), "candidate label outside range")
    table = np.zeros((n_ref, n_candidate), dtype=np.int64)
    np.add.at(table, (ref, cand), 1)
    return table


def partition_overlap(table: np.ndarray) -> dict[str, float]:
    n = int(table.sum())
    _require(n > 0, "empty contingency table")
    anchor_recovery = float(table.max(axis=1).sum() / n)
    variant_purity = float(table.max(axis=0).sum() / n)
    symmetric = (
        2.0 * anchor_recovery * variant_purity / (anchor_recovery + variant_purity)
        if anchor_recovery + variant_purity > 0
        else 0.0
    )
    return {
        "anchor_recovery": anchor_recovery,
        "variant_purity": variant_purity,
        "symmetric_overlap": float(symmetric),
    }


def matched_accuracy(table: np.ndarray) -> float:
    rows, columns = linear_sum_assignment(-table)
    return float(table[rows, columns].sum() / table.sum())


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


def canonical_anchor_correspondence(
    *,
    anchor: int,
    table: np.ndarray,
    canonical_centroids: np.ndarray,
    variant_centroids: np.ndarray,
) -> dict[str, Any]:
    """Label-blind split/merge-aware mapping for one canonical k32 anchor."""

    counts = table[anchor].astype(int)
    anchor_n = int(counts.sum())
    variant_sizes = table.sum(axis=0).astype(int)
    _require(anchor_n > 0, f"canonical p{anchor} has no sampled tiles")
    candidates = []
    for cluster, intersection in enumerate(counts):
        union = anchor_n + int(variant_sizes[cluster]) - int(intersection)
        jaccard = float(intersection / union) if union else 0.0
        candidates.append((jaccard, int(intersection), -cluster, cluster))
    best = max(candidates)[3]
    intersection = int(counts[best])
    anchor_recall = float(intersection / anchor_n)
    precision = float(intersection / int(variant_sizes[best])) if variant_sizes[best] else 0.0
    union = anchor_n + int(variant_sizes[best]) - intersection
    order = sorted(range(len(counts)), key=lambda cluster: (-int(counts[cluster]), cluster))
    cover: list[int] = []
    cumulative = 0
    for cluster in order:
        if counts[cluster] <= 0:
            continue
        cover.append(int(cluster))
        cumulative += int(counts[cluster])
        if cumulative / anchor_n >= COVERAGE_TARGET:
            break
    strongest_anchor_for_best = int(np.argmax(table[:, best]))
    return {
        "anchor_prototype": int(anchor),
        "anchor_sample_tiles": anchor_n,
        "best_alternative_cluster": int(best),
        "intersection_tiles": intersection,
        "anchor_recall": anchor_recall,
        "alternative_precision": precision,
        "jaccard": float(intersection / union) if union else 0.0,
        "centroid_cosine": _cosine(canonical_centroids[anchor], variant_centroids[best]),
        "reciprocal_best_match": bool(strongest_anchor_for_best == anchor),
        "clusters_to_cover_80pct": cover,
        "n_clusters_to_cover_80pct": len(cover),
        "cover_80pct_observed_fraction": float(cumulative / anchor_n),
        "mapping_basis": "frozen_label_blind_vocabulary_sample_only",
    }


def anchor_mapping(
    *,
    anchor: int,
    montage_id: str,
    table: np.ndarray,
    canonical_centroids: np.ndarray,
    variant_centroids: np.ndarray,
    montage_labels: np.ndarray,
) -> dict[str, Any]:
    correspondence = canonical_anchor_correspondence(
        anchor=anchor,
        table=table,
        canonical_centroids=canonical_centroids,
        variant_centroids=variant_centroids,
    )
    best = int(correspondence["best_alternative_cluster"])
    montage_counts = np.bincount(np.asarray(montage_labels, dtype=int), minlength=table.shape[1])
    montage_top = int(np.argmax(montage_counts))
    return {
        "montage_id": montage_id,
        **correspondence,
        "reviewed_montage_tiles": int(len(montage_labels)),
        "reviewed_tiles_in_best_cluster": int(montage_counts[best]),
        "reviewed_fraction_in_best_cluster": float(montage_counts[best] / len(montage_labels)),
        "reviewed_dominant_cluster": montage_top,
        "reviewed_dominant_fraction": float(montage_counts[montage_top] / len(montage_labels)),
        "reviewed_cluster_distribution": {
            str(cluster): int(count) for cluster, count in enumerate(montage_counts) if count
        },
        "interpretation": "computational_correspondence_only_no_pathology_label_transfer",
    }


def all_anchor_correspondences(
    *,
    canonical_labels: np.ndarray,
    canonical_centroids: np.ndarray,
    variant_labels: dict[str, np.ndarray],
    variant_centroids: dict[str, np.ndarray],
    k_values: tuple[int, ...] = K_VALUES,
    seeds: tuple[int, ...] = CLUSTER_SEEDS,
) -> list[dict[str, Any]]:
    """Freeze all 32 mappings per variant before any KRAS label is consulted."""

    rows: list[dict[str, Any]] = []
    for k in k_values:
        for seed in seeds:
            key = f"k{k}_seed{seed}"
            table = contingency(canonical_labels, variant_labels[key], CANONICAL_K, k)
            for anchor in ALL_CANONICAL_ANCHORS:
                rows.append(
                    {
                        "variant": key,
                        "k": int(k),
                        "seed": int(seed),
                        **canonical_anchor_correspondence(
                            anchor=anchor,
                            table=table,
                            canonical_centroids=canonical_centroids,
                            variant_centroids=variant_centroids[key],
                        ),
                    }
                )
    _require(
        len(rows) == len(k_values) * len(seeds) * CANONICAL_K,
        "all-anchor correspondence family is incomplete",
    )
    return rows


_FORKED_FIT_MATRIX: np.ndarray | None = None


def _fit_variant_worker(task: tuple[int, int, int]) -> tuple[
    str, np.ndarray, np.ndarray, dict[str, Any]
]:
    """Fit one independent variant on a fork-inherited, read-only matrix.

    Each worker is deliberately single-threaded.  Running one K-means fit with
    many OpenMP threads makes floating-point reduction order scheduler-dependent
    (even when labels are unchanged).  Forking the nine independent variants
    gives useful physical-core parallelism while preserving byte-stable fits.
    """

    k, seed, n_init = task
    z = _FORKED_FIT_MATRIX
    if z is None:
        raise RuntimeError("forked K-means worker lacks the projected matrix")
    with threadpool_limits(limits=1):
        model = KMeans(
            n_clusters=int(k),
            n_init=int(n_init),
            random_state=int(seed),
            algorithm="lloyd",
        ).fit(z)
    key = f"k{k}_seed{seed}"
    labels = np.asarray(model.labels_, dtype=np.int16)
    centers = np.asarray(model.cluster_centers_, dtype=np.float32)
    info = {
        "k": int(k),
        "seed": int(seed),
        "model_inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
    }
    return key, labels, centers, info


def deterministic_replay_inertia(
    z: np.ndarray, centers: np.ndarray, labels: np.ndarray
) -> float:
    """Return the exact campaign replay identity for one fitted variant.

    Residual subtraction and squaring deliberately retain the sealed float32
    representation.  Only the reduction accumulator is float64.  Unlike the
    model-native OpenMP float32 reduction, this NumPy expression has a fixed
    traversal order and is independent of the BLAS/OpenMP thread cap.
    """

    matrix = np.asarray(z)
    variant_centers = np.asarray(centers)
    variant_labels = np.asarray(labels)
    _require(matrix.ndim == 2 and len(matrix) > 0, "projected sample is empty")
    _require(matrix.dtype == np.float32, "projected sample is not float32")
    _require(
        variant_centers.ndim == 2
        and variant_centers.dtype == np.float32
        and variant_centers.shape[1] == matrix.shape[1],
        "variant centroids are not compatible float32 coordinates",
    )
    _require(
        variant_labels.shape == (len(matrix),)
        and np.issubdtype(variant_labels.dtype, np.integer),
        "variant labels do not match the projected sample",
    )
    _require(
        bool((variant_labels >= 0).all())
        and bool((variant_labels < len(variant_centers)).all()),
        "variant labels are outside the centroid inventory",
    )
    value = float(
        np.sum(
            (matrix - variant_centers[variant_labels.astype(int)]) ** 2,
            dtype=np.float64,
        )
    )
    _require(
        np.isfinite(value) and value >= 0.0,
        "replay inertia is not finite and nonnegative",
    )
    return value


def fit_variants(
    z: np.ndarray,
    *,
    k_values: tuple[int, ...] = K_VALUES,
    seeds: tuple[int, ...] = CLUSTER_SEEDS,
    n_init: int = N_INIT,
    threads: int = 12,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Fit variants concurrently, with one deterministic thread per fit.

    The projected sample is inherited copy-on-write by Linux ``fork`` workers;
    it is never mutated or serialized nine times.  ``threads`` is an upper
    bound on simultaneous physical cores.  There are nine independent variants,
    so K-means uses at most nine cores while projection/assignment may use the
    full declared cap elsewhere in the campaign.
    """

    _require(int(threads) >= 1, "threads must be positive")
    matrix = np.ascontiguousarray(z, dtype=np.float32)
    _require(matrix.ndim == 2 and len(matrix) > 0, "projected sample is empty")
    _require(np.isfinite(matrix).all(), "projected sample contains non-finite values")
    tasks = [
        (int(k), int(seed), int(n_init))
        for k in k_values
        for seed in seeds
    ]
    _require(bool(tasks), "no K-means variants requested")
    for k, seed, _ in tasks:
        print(
            f"  queued k{k}_seed{seed} (n_init={n_init}, worker_threads=1)",
            flush=True,
        )

    labels: dict[str, np.ndarray] = {}
    centers: dict[str, np.ndarray] = {}
    fit_info: dict[str, dict[str, Any]] = {}
    global _FORKED_FIT_MATRIX
    _require(_FORKED_FIT_MATRIX is None, "nested K-means fit is not supported")
    _FORKED_FIT_MATRIX = matrix
    try:
        worker_count = min(int(threads), len(tasks))
        if worker_count == 1:
            fitted = [_fit_variant_worker(task) for task in tasks]
        else:
            _require(
                "fork" in mp.get_all_start_methods(),
                "deterministic copy-on-write K-means requires Linux fork",
            )
            context = mp.get_context("fork")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
            ) as executor:
                fitted = list(executor.map(_fit_variant_worker, tasks))
    finally:
        _FORKED_FIT_MATRIX = None
    for key, variant_labels, variant_centers, info in fitted:
        model_inertia = float(info["model_inertia"])
        _require(
            np.isfinite(model_inertia) and model_inertia > 0.0,
            f"model inertia is not finite and positive: {key}",
        )
        labels[key] = variant_labels
        centers[key] = variant_centers
        fit_info[key] = {
            **info,
            "replay_inertia_float64": deterministic_replay_inertia(
                matrix, variant_centers, variant_labels
            ),
        }
    return labels, centers, fit_info


def canonical_reconstruction_control(
    *,
    canonical_labels: np.ndarray,
    canonical_metadata: dict[str, Any],
    sampled_feature_values_sha256: str,
    same_seed_refit: dict[str, Any],
    expected_sample_hash: str | None = None,
) -> dict[str, Any]:
    """Validate the frozen atlas without pretending a fresh refit is identity.

    The training implementation used ``PCA.fit_transform`` and stored only the
    fitted mean/components.  Replaying those stored arrays uses the
    ``PCA.transform`` algebra.  The two float32 coordinate paths are
    mathematically equivalent but not byte-identical, and K-means can converge
    to a different local optimum even with the same random seed.  Therefore the
    aborting control is the exact sampled-feature hash plus near-exact indexed
    cluster masses under the *frozen* centroids.  The same-seed refit remains a
    reported member of the predeclared nine-variant sensitivity grid.
    """

    labels = np.asarray(canonical_labels, dtype=int)
    _require(labels.ndim == 1 and len(labels) > 0, "canonical labels are empty")
    _require(
        (labels >= 0).all() and (labels < CANONICAL_K).all(),
        "canonical label outside range",
    )
    expected_hash = (
        CANONICAL_SAMPLE_VALUES_SHA256
        if expected_sample_hash is None
        else str(expected_sample_hash)
    )
    _require(
        len(expected_hash) == 64,
        "canonical sampled-feature reference hash is malformed",
    )
    observed_hash = str(sampled_feature_values_sha256)

    raw_sizes = canonical_metadata.get("cluster_sizes")
    _require(
        isinstance(raw_sizes, list) and len(raw_sizes) == CANONICAL_K,
        "canonical metadata lacks 32 training cluster sizes",
    )
    expected_sizes = np.asarray(raw_sizes, dtype=np.int64)
    _require((expected_sizes > 0).all(), "canonical training cluster is empty")
    _require(
        int(canonical_metadata.get("n_sample_tiles", -1)) == len(labels)
        and int(expected_sizes.sum()) == len(labels),
        "canonical training cluster sizes do not match sampled tile count",
    )
    observed_sizes = np.bincount(labels, minlength=CANONICAL_K).astype(np.int64)
    absolute_differences = np.abs(observed_sizes - expected_sizes)
    total_variation = float(absolute_differences.sum() / (2.0 * len(labels)))
    maximum_mass_delta = float(absolute_differences.max() / len(labels))
    sample_hash_exact = bool(observed_hash == expected_hash)
    mass_pass = bool(
        total_variation <= CANONICAL_MAX_CLUSTER_MASS_TOTAL_VARIATION
        and maximum_mass_delta <= CANONICAL_MAX_SINGLE_CLUSTER_MASS_DELTA
    )

    _require(
        same_seed_refit.get("variant") == f"k{CANONICAL_K}_seed{CANONICAL_SEED}",
        "canonical-seed refit sensitivity row is missing",
    )
    return {
        "name": "exact_sample_and_frozen_centroid_reprojection",
        "assignment_rule": "nearest persisted canonical k32 centroid in persisted PCA transform coordinates",
        "sample_values_sha256_expected": expected_hash,
        "sample_values_sha256_observed": observed_hash,
        "sample_values_sha256_exact": sample_hash_exact,
        "training_cluster_sizes": expected_sizes.tolist(),
        "reprojected_cluster_sizes": observed_sizes.tolist(),
        "absolute_cluster_size_differences": absolute_differences.tolist(),
        "cluster_mass_total_variation": total_variation,
        "maximum_cluster_mass_delta": maximum_mass_delta,
        "maximum_cluster_mass_total_variation_allowed": (
            CANONICAL_MAX_CLUSTER_MASS_TOTAL_VARIATION
        ),
        "maximum_single_cluster_mass_delta_allowed": (
            CANONICAL_MAX_SINGLE_CLUSTER_MASS_DELTA
        ),
        "same_seed_refit": {
            "variant": str(same_seed_refit["variant"]),
            "ari_to_frozen_canonical": float(
                same_seed_refit["ari_to_canonical_k32"]
            ),
            "ami_to_frozen_canonical": float(
                same_seed_refit["ami_to_canonical_k32"]
            ),
            "role": "sensitivity_variant_not_an_identity_control",
            "included_in_all_nine_biological_gate": True,
        },
        "rationale": (
            "The original fit used PCA.fit_transform, whereas persisted-basis replay "
            "uses transform algebra; same-seed K-means is consequently a genuine "
            "local-optimum sensitivity result. Identity is instead enforced by the "
            "exact raw sampled-feature hash and frozen-centroid indexed cluster masses."
        ),
        "pass": bool(sample_hash_exact and mass_pass),
    }


def build_analysis(
    *,
    canonical_labels: np.ndarray,
    canonical_centroids: np.ndarray,
    canonical_metadata: dict[str, Any],
    sampled_feature_values_sha256: str,
    review_rows: pd.DataFrame,
    review_canonical_labels: np.ndarray,
    variant_labels: dict[str, np.ndarray],
    variant_centroids: dict[str, np.ndarray],
    review_variant_labels: dict[str, np.ndarray],
    fit_info: dict[str, dict[str, Any]],
    k_values: tuple[int, ...] = K_VALUES,
    seeds: tuple[int, ...] = CLUSTER_SEEDS,
) -> dict[str, Any]:
    for prototype, montage in ANCHORS.items():
        positions = np.flatnonzero(review_rows["montage_id"].eq(montage).to_numpy())
        _require(len(positions) > 0, f"no reviewed rows for {montage}")
        _require(
            np.all(review_canonical_labels[positions] == prototype),
            f"{montage} reviewed tiles no longer assign to canonical p{prototype}",
        )

    variant_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for k in k_values:
        for seed in seeds:
            key = f"k{k}_seed{seed}"
            labels = variant_labels[key]
            table = contingency(canonical_labels, labels, CANONICAL_K, k)
            sizes = np.bincount(labels.astype(int), minlength=k)
            row = {
                "variant": key,
                **fit_info[key],
                "ari_to_canonical_k32": float(adjusted_rand_score(canonical_labels, labels)),
                "ami_to_canonical_k32": float(adjusted_mutual_info_score(canonical_labels, labels)),
                **partition_overlap(table),
                "one_to_one_matched_accuracy": matched_accuracy(table),
                "min_cluster_size": int(sizes.min()),
                "median_cluster_size": float(np.median(sizes)),
                "max_cluster_size": int(sizes.max()),
            }
            variant_rows.append(row)
            for prototype, montage in ANCHORS.items():
                positions = np.flatnonzero(review_rows["montage_id"].eq(montage).to_numpy())
                mapping_rows.append(
                    {
                        "variant": key,
                        "k": int(k),
                        "seed": int(seed),
                        **anchor_mapping(
                            anchor=prototype,
                            montage_id=montage,
                            table=table,
                            canonical_centroids=canonical_centroids,
                            variant_centroids=variant_centroids[key],
                            montage_labels=review_variant_labels[key][positions],
                        ),
                    }
                )

    seed_pair_rows: list[dict[str, Any]] = []
    for k in k_values:
        for seed_a, seed_b in combinations(seeds, 2):
            key_a = f"k{k}_seed{seed_a}"
            key_b = f"k{k}_seed{seed_b}"
            labels_a = variant_labels[key_a]
            labels_b = variant_labels[key_b]
            table = contingency(labels_a, labels_b, k, k)
            seed_pair_rows.append(
                {
                    "k": int(k),
                    "seed_a": int(seed_a),
                    "seed_b": int(seed_b),
                    "ari": float(adjusted_rand_score(labels_a, labels_b)),
                    "ami": float(adjusted_mutual_info_score(labels_a, labels_b)),
                    "one_to_one_matched_accuracy": matched_accuracy(table),
                }
            )

    same_seed_refit = next(
        row for row in variant_rows if row["k"] == CANONICAL_K and row["seed"] == CANONICAL_SEED
    )
    reconstruction_control = canonical_reconstruction_control(
        canonical_labels=canonical_labels,
        canonical_metadata=canonical_metadata,
        sampled_feature_values_sha256=sampled_feature_values_sha256,
        same_seed_refit=same_seed_refit,
    )
    correspondence_rows = all_anchor_correspondences(
        canonical_labels=canonical_labels,
        canonical_centroids=canonical_centroids,
        variant_labels=variant_labels,
        variant_centroids=variant_centroids,
        k_values=k_values,
        seeds=seeds,
    )
    return {
        "canonical_reconstruction_control": reconstruction_control,
        "variant_metrics": variant_rows,
        "anchor_mappings": mapping_rows,
        "all_anchor_correspondences": correspondence_rows,
        "seed_pair_metrics": seed_pair_rows,
    }


def correspondence_matrices(
    correspondence_rows: list[dict[str, Any]],
    variant_centroids: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Turn the frozen label-blind correspondence plan into 32 x k sums."""

    matrices: dict[str, np.ndarray] = {}
    frame = pd.DataFrame(correspondence_rows)
    _require(not frame.empty, "empty all-anchor correspondence plan")
    for variant, centroids in variant_centroids.items():
        block = frame[frame["variant"].eq(variant)].sort_values("anchor_prototype")
        _require(
            block["anchor_prototype"].tolist() == list(ALL_CANONICAL_ANCHORS),
            f"{variant}: correspondence plan is not exactly 32 canonical anchors",
        )
        matrix = np.zeros((CANONICAL_K, len(centroids)), dtype=np.float64)
        for row in block.itertuples(index=False):
            clusters = list(row.clusters_to_cover_80pct)
            _require(bool(clusters), f"{variant}/p{row.anchor_prototype}: empty correspondence set")
            _require(
                len(clusters) == len(set(clusters))
                and min(clusters) >= 0
                and max(clusters) < len(centroids),
                f"{variant}/p{row.anchor_prototype}: invalid correspondence set",
            )
            matrix[int(row.anchor_prototype), clusters] = 1.0
        matrices[variant] = matrix
    return matrices


def equal_slide_patient_abundance(slides: pd.DataFrame) -> pd.DataFrame:
    """Frozen copy of E4's equal-slide patient aggregation for abundance."""

    required = {"arm", "slide_id", "patient_id", "prototype", "abundance"}
    _require(required <= set(slides), f"slide profiles lack {sorted(required - set(slides))}")
    _require(
        not slides.duplicated(["arm", "slide_id", "prototype"]).any(),
        "slide profiles duplicate arm/slide/prototype rows",
    )
    keys = ["arm", "patient_id", "prototype"]
    values = slides.groupby(keys, as_index=False)[["abundance"]].mean()
    counts = (
        slides.groupby(keys, as_index=False)["slide_id"]
        .count()
        .rename(columns={"slide_id": "n_slides"})
    )
    return values.merge(counts, on=keys, validate="one_to_one")


def development_patient_abundance(
    inputs: Inputs,
    vocab: FrozenVocabulary,
    variant_centroids: dict[str, np.ndarray],
    correspondence_rows: list[dict[str, Any]],
    *,
    progress: bool = True,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Assign full E0 bags and reuse E4's exact equal-slide patient aggregation."""

    slides, patients = load_development_manifest(inputs)
    matrices = correspondence_matrices(correspondence_rows, variant_centroids)
    vocabulary_centers = {"canonical_k32": vocab.centroids, **variant_centroids}
    matrices = {"canonical_k32": np.eye(CANONICAL_K, dtype=np.float64), **matrices}
    slide_records: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in vocabulary_centers
    }
    digest = hashlib.sha256()
    digest.update(b"aim4-full-e0-development-features-v1\0")
    digest.update(struct.pack("<qq", int(len(slides)), int(vocab.pca_mean.size)))
    for index, row in enumerate(slides.itertuples(index=False), start=1):
        path = inputs.feature_root / f"{row.slide_id}.h5"
        with h5py.File(path, "r") as handle:
            features = np.ascontiguousarray(handle["features"][:], dtype="<f4")
        digest.update(str(row.slide_id).encode("utf-8") + b"\0")
        digest.update(struct.pack("<q", int(len(features))))
        digest.update(features.tobytes(order="C"))
        z = _project(features, vocab)
        for variant, centers in vocabulary_centers.items():
            labels = assign_nearest(z, centers)
            counts = np.bincount(labels.astype(int), minlength=len(centers)).astype(np.float64)
            matched = matrices[variant] @ counts / max(float(len(labels)), 1.0)
            records = slide_records[variant]
            records.extend(
                {
                    "arm": "e0",
                    "slide_id": str(row.slide_id),
                    "patient_id": str(row.patient_id),
                    "prototype": int(anchor),
                    "abundance": float(matched[anchor]),
                }
                for anchor in ALL_CANONICAL_ANCHORS
            )
        if progress and index % 100 == 0:
            print(f"  full-E0 abundance {index}/{len(slides)} slides", flush=True)

    patient_blocks: list[pd.DataFrame] = []
    expected_rows = int(len(patients) * CANONICAL_K)
    for variant, records in slide_records.items():
        # Mechanically frozen from E4: equal slide means inside patient x prototype.
        patient = equal_slide_patient_abundance(pd.DataFrame(records))
        _require(len(patient) == expected_rows, f"{variant}: patient-profile row count mismatch")
        _require(
            not patient.duplicated(["patient_id", "prototype"]).any(),
            f"{variant}: duplicate patient/prototype profile",
        )
        patient.insert(0, "variant", variant)
        patient_blocks.append(patient)
    profiles = pd.concat(patient_blocks, ignore_index=True)

    canonical_reference, _ = load_canonical_references(inputs)
    expected = canonical_reference[["patient_id", "prototype", "abundance"]].sort_values(
        ["patient_id", "prototype"], kind="stable"
    ).reset_index(drop=True)
    observed = profiles[profiles["variant"].eq("canonical_k32")][
        ["patient_id", "prototype", "abundance"]
    ].sort_values(["patient_id", "prototype"], kind="stable").reset_index(drop=True)
    _require(
        observed[["patient_id", "prototype"]].equals(expected[["patient_id", "prototype"]]),
        "canonical reconstructed and sealed profile keys differ",
    )
    differences = np.abs(observed["abundance"].to_numpy() - expected["abundance"].to_numpy())
    max_difference = float(differences.max(initial=0.0))
    _require(max_difference <= 1e-12, f"canonical patient abundance does not reproduce: max diff {max_difference}")
    reconstruction = {
        "n_rows_compared": int(len(observed)),
        "maximum_absolute_abundance_difference": max_difference,
        "tolerance": 1e-12,
        "pass": True,
    }
    return profiles, digest.hexdigest(), reconstruction


def auc_effect(values: np.ndarray, positive: np.ndarray) -> float:
    labels = np.asarray(positive).astype(int)
    n_positive = int(labels.sum())
    n_negative = int(len(labels) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = stats.rankdata(np.asarray(values, dtype=float))
    return float(
        (ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def bootstrap_auc_effect(
    values: np.ndarray,
    positive: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    abundance = np.asarray(values, dtype=float)
    labels = np.asarray(positive).astype(int)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(abundance), len(abundance))
        if len(np.unique(labels[indices])) < 2:
            continue
        draws.append(auc_effect(abundance[indices], labels[indices]))
    sampled = np.asarray(draws, dtype=float)
    point = auc_effect(abundance, labels)
    return {
        "auc": point,
        "delta": point - 0.5,
        "ci_low": float(np.percentile(sampled, 2.5)) if sampled.size else float("nan"),
        "ci_high": float(np.percentile(sampled, 97.5)) if sampled.size else float("nan"),
        "n": int(len(abundance)),
        "n_positive": int(labels.sum()),
    }


def mannwhitney_p(values: np.ndarray, positive: np.ndarray) -> float:
    abundance = np.asarray(values, dtype=float)
    labels = np.asarray(positive).astype(bool)
    if labels.sum() < 2 or (~labels).sum() < 2:
        return float("nan")
    return float(stats.mannwhitneyu(abundance[labels], abundance[~labels], alternative="two-sided").pvalue)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Fixed-input BH; the caller must already retain structural p=1 rows."""

    p = np.asarray(pvalues, dtype=float)
    _require(len(p) == CANONICAL_K, "BH family must contain exactly 32 p-values")
    _require(np.isfinite(p).all(), "BH family contains non-finite p-values")
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    return restored


def _validate_effect_input(frame: pd.DataFrame) -> None:
    required = {"patient_id", "prototype", "is_mutant", "abundance"}
    _require(required <= set(frame), f"effect table input lacks {sorted(required - set(frame))}")
    _require(
        not frame.duplicated(["patient_id", "prototype"]).any(),
        "effect table repeats patient/prototype rows",
    )
    prototypes = set(pd.to_numeric(frame["prototype"], errors="raise").astype(int))
    _require(prototypes == set(ALL_CANONICAL_ANCHORS), "effect family is not prototypes 0..31")
    by_prototype = frame.groupby("prototype")["patient_id"].agg(list)
    reference = set(map(str, by_prototype.iloc[0]))
    _require(
        all(set(map(str, values)) == reference for values in by_prototype.iloc[1:]),
        "effect prototypes do not cover identical patients",
    )
    labels = pd.to_numeric(frame["is_mutant"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame["abundance"], errors="coerce").to_numpy(dtype=float)
    _require(
        np.isfinite(labels).all() and np.isin(labels, [0.0, 1.0]).all(),
        "effect labels are not finite and binary",
    )
    _require(np.isfinite(values).all(), "effect abundance contains non-finite values")


def fixed_effect_family(frame: pd.DataFrame, *, n_bootstrap: int) -> pd.DataFrame:
    """Source-frozen corrected E4 family: 32 rows, structural p=1."""

    _require(int(n_bootstrap) >= 1, "n_bootstrap must be positive")
    _validate_effect_input(frame)
    rows: list[dict[str, Any]] = []
    for prototype in ALL_CANONICAL_ANCHORS:
        block = frame[frame["prototype"].eq(prototype)]
        labels = pd.to_numeric(block["is_mutant"], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(block["abundance"], errors="coerce").to_numpy(dtype=float)
        estimable = bool(
            len(block) > 0
            and np.isfinite(labels).all()
            and np.isfinite(values).all()
            and set(np.unique(labels)) == {0.0, 1.0}
        )
        if estimable:
            effect = bootstrap_auc_effect(
                values,
                labels,
                n_bootstrap=int(n_bootstrap),
                seed=CANONICAL_SEED + prototype,
            )
            p_value = mannwhitney_p(values, labels)
            estimable = bool(
                np.isfinite(
                    [effect["auc"], effect["ci_low"], effect["ci_high"], p_value]
                ).all()
            )
        else:
            effect = {
                "auc": float("nan"),
                "delta": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n": int(len(block)),
                "n_positive": int(np.nansum(labels)) if len(labels) else 0,
            }
            p_value = 1.0
        if not estimable:
            p_value = 1.0
            effect = {
                "auc": float("nan"),
                "delta": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n": int(len(block)),
                "n_positive": int(np.nansum(labels)) if len(labels) else 0,
            }
        rows.append(
            {
                **effect,
                "prototype": int(prototype),
                "p": float(p_value),
                "estimable": estimable,
                "structural_p_policy": None if estimable else "p=1_in_fixed_32_family",
            }
        )
    table = pd.DataFrame(rows).sort_values("prototype").reset_index(drop=True)
    table["q"] = benjamini_hochberg(table["p"].to_numpy(dtype=float))
    _require(len(table) == CANONICAL_K, "fixed family does not contain 32 rows")
    table["significant"] = table["estimable"] & (table["q"] < FDR_ALPHA) & (
        (table["ci_low"] > 0.5) | (table["ci_high"] < 0.5)
    )
    return table


def _effect_family(frame: pd.DataFrame, *, n_bootstrap: int) -> pd.DataFrame:
    return fixed_effect_family(frame, n_bootstrap=n_bootstrap)


def association_analysis(
    inputs: Inputs,
    profiles: pd.DataFrame,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, Any]:
    """Run exact E4 A/D effects and the predeclared all-grid claim gate."""

    _, patient_meta = load_development_manifest(inputs)
    metadata = patient_meta[["patient_id", "is_mutant", "in_D"]]
    rows: list[dict[str, Any]] = []
    variants = ["canonical_k32", *(f"k{k}_seed{seed}" for k in K_VALUES for seed in CLUSTER_SEEDS)]
    for variant in variants:
        block = profiles[profiles["variant"].eq(variant)].drop(
            columns=[column for column in ("is_mutant", "in_D") if column in profiles.columns]
        ).merge(metadata, on="patient_id", validate="many_to_one")
        _require(
            block["patient_id"].nunique() == len(patient_meta),
            f"{variant}: A population is incomplete",
        )
        a_table = _effect_family(block, n_bootstrap=n_bootstrap)
        d_block = block[block["in_D"].eq(1)]
        _require(
            d_block["patient_id"].nunique() == int(patient_meta["in_D"].sum()),
            f"{variant}: D population is incomplete",
        )
        d_table = _effect_family(d_block, n_bootstrap=n_bootstrap)
        for anchor in ALL_CANONICAL_ANCHORS:
            a = a_table[a_table["prototype"].eq(anchor)].iloc[0]
            d = d_table[d_table["prototype"].eq(anchor)].iloc[0]
            row: dict[str, Any] = {
                "variant": variant,
                "anchor_prototype": int(anchor),
                "montage_id": ANCHORS.get(anchor),
                "family_size_A": CANONICAL_K,
                "family_size_D": CANONICAL_K,
            }
            for population, effect in (("A", a), ("D", d)):
                row.update(
                    {
                        f"auc_{population}": float(effect["auc"]),
                        f"ci_low_{population}": float(effect["ci_low"]),
                        f"ci_high_{population}": float(effect["ci_high"]),
                        f"p_{population}": float(effect["p"]),
                        f"q_{population}": float(effect["q"]),
                        f"estimable_{population}": bool(effect["estimable"]),
                        f"structural_p_policy_{population}": effect[
                            "structural_p_policy"
                        ],
                        f"significant_{population}": bool(effect["significant"]),
                    }
                )
            row["positive_significant_A_and_D"] = bool(
                row["estimable_A"]
                and row["estimable_D"]
                and row["auc_A"] > 0.5
                and row["ci_low_A"] > 0.5
                and row["q_A"] < FDR_ALPHA
                and row["significant_A"]
                and row["auc_D"] > 0.5
                and row["ci_low_D"] > 0.5
                and row["q_D"] < FDR_ALPHA
                and row["significant_D"]
            )
            rows.append(row)

    _, sealed_specificity = load_canonical_references(inputs)
    canonical_rows = [row for row in rows if row["variant"] == "canonical_k32"]
    maximum_point_difference = 0.0
    maximum_p_difference = 0.0
    maximum_q_difference = 0.0
    structural_rows_checked = 0
    for row in canonical_rows:
        anchor = int(row["anchor_prototype"])
        sealed = sealed_specificity["prototypes"][str(anchor)]["abundance"]
        for population in ("A", "D"):
            sealed_effect = sealed[population]
            _require(
                bool(row[f"estimable_{population}"])
                == bool(sealed_effect["estimable"]),
                f"canonical p{anchor}/{population} estimability differs",
            )
            maximum_p_difference = max(
                maximum_p_difference,
                abs(float(row[f"p_{population}"]) - float(sealed_effect["p"])),
            )
            maximum_q_difference = max(
                maximum_q_difference,
                abs(float(row[f"q_{population}"]) - float(sealed_effect["q"])),
            )
            if row[f"estimable_{population}"]:
                maximum_point_difference = max(
                    maximum_point_difference,
                    abs(
                        float(row[f"auc_{population}"])
                        - float(sealed_effect["auc"])
                    ),
                )
            else:
                structural_rows_checked += 1
                _require(
                    sealed_effect.get("auc") is None
                    and float(sealed_effect["p"]) == 1.0
                    and not bool(sealed_effect["significant"]),
                    f"canonical p{anchor}/{population} structural contract differs",
                )
    tolerance = 1e-12
    _require(
        maximum_point_difference <= tolerance
        and maximum_p_difference <= tolerance
        and maximum_q_difference <= tolerance,
        "canonical fixed-k32 A/D point effects, p-values, or q-values do not reproduce",
    )
    protocol = sealed_specificity.get("protocol", {})
    _require(
        protocol.get("bh_family")
        == "fixed prototypes 0..31, including structural p=1 rows",
        "canonical specificity does not declare the corrected fixed-32 BH family",
    )

    variant_rows = [row for row in rows if row["variant"] != "canonical_k32"]
    claim_stability: dict[str, Any] = {}
    for anchor, montage in ANCHORS.items():
        anchor_rows = [row for row in variant_rows if row["anchor_prototype"] == anchor]
        _require(len(anchor_rows) == len(K_VALUES) * len(CLUSTER_SEEDS), f"p{anchor}: incomplete stability grid")
        passing = [str(row["variant"]) for row in anchor_rows if row["positive_significant_A_and_D"]]
        all_pass = len(passing) == len(anchor_rows)
        claim_stability[str(anchor)] = {
            "montage_id": montage,
            "n_variants": len(anchor_rows),
            "passing_variants": passing,
            "failing_variants": [
                str(row["variant"])
                for row in anchor_rows
                if not row["positive_significant_A_and_D"]
            ],
            "all_variants_pass": all_pass,
            "verdict": (
                "ROBUST_ACROSS_PREDECLARED_K_SEED_GRID"
                if all_pass
                else "CONDITIONAL_K32_NOT_STABLE_ACROSS_FULL_GRID"
            ),
            "claim_limit": "Matched alternative clusters remain pathology-unlabeled pending new blinded review.",
        }
    result = {
        "multiplicity": {
            "method": "Benjamini-Hochberg exactly within each variant x population family",
            "family_size": CANONICAL_K,
            "families": "A and D separately",
            "a_and_d_gate": "intersection-union; both must pass",
            "variant_grid_gate": "intersection-union; all nine variants must pass",
            "mapping_leakage_guard": "all matched sets frozen from label-blind vocabulary-sample overlap before full-E0 association testing",
        },
        "predeclared_gate": {
            "direction": "AUC > 0.5 in A and D",
            "uncertainty": "95% patient-bootstrap lower bound > 0.5 in A and D",
            "multiplicity": "BH q < 0.05 in both m=32 families",
            "grid": "every k=24/32/40 x seed=20260819/20/21 variant",
        },
        "canonical_effect_reconstruction": {
            "maximum_absolute_auc_difference": float(maximum_point_difference),
            "maximum_absolute_p_difference": float(maximum_p_difference),
            "maximum_absolute_q_difference": float(maximum_q_difference),
            "structural_rows_checked": int(structural_rows_checked),
            "family_declaration": protocol["bh_family"],
            "tolerance": tolerance,
            "pass": True,
        },
        "claim_stability": claim_stability,
        "association_effects": rows,
    }
    return _sanitize_json(result)


def material_source_files() -> list[Path]:
    files = [
        Path(__file__).resolve(),
        REPO / "aim4_morphologic_atlas.py",
        REPO / "aim4_morphologic_atlas_base.py",
        REPO / "src" / "oceanpath" / "aim1" / "atlas.py",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
    ]
    missing = [path for path in files if not path.is_file()]
    _require(not missing, f"material source files missing: {missing}")
    return [path.resolve() for path in files]


def snapshot_sources(stage: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in material_source_files():
        relative = source.relative_to(REPO)
        destination = stage / "source_snapshot" / relative
        payload = source.read_bytes()
        _write_bytes_once(destination, payload)
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return rows


def relative_inventory(root: Path, *, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "h5py": h5py.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_campaign(inputs: Inputs, output: Path, *, threads: int, strict: bool = True) -> Path:
    _require(1 <= threads <= 18, "threads must be between 1 and 18 physical cores")
    preflight = inspect_inputs(inputs, strict_montages=strict)
    output = new_output_root(output, inputs)
    stage = output.parent / f".{output.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir(mode=0o755, parents=False, exist_ok=False)
    print(f"staging append-only Aim-4 sensitivity at {stage}", flush=True)
    try:
        source_manifest = snapshot_sources(stage)
        _write_json_once(stage / "source_snapshot" / "manifest.json", source_manifest)
        _write_json_once(stage / "feature_inventory.json", preflight["feature_inventory"])

        vocab = load_vocabulary(inputs)
        # Apply the declared physical-core cap to every BLAS/OpenMP-heavy
        # projection, assignment, and K-means operation, not only K-means.
        # The exact cap is sealed in results.json and reused by full replay.
        with threadpool_limits(limits=threads):
            review_rows, review_z, review_hash = collect_review_projection(
                inputs, vocab, strict=strict
            )
            _require(
                review_hash == preflight["reviewed_feature_values_sha256"],
                "reviewed feature values drifted after preflight",
            )
            z, sample_hash = collect_projected_sample(inputs, vocab)
            _require(
                sample_hash == CANONICAL_SAMPLE_VALUES_SHA256,
                "canonical sampled-feature values differ from the pinned exact hash",
            )
            canonical_labels = assign_nearest(z, vocab.centroids)
            review_canonical = assign_nearest(review_z, vocab.centroids)

            variant_labels, variant_centroids, fit_info = fit_variants(
                z, threads=threads
            )
            review_variant_labels = {
                key: assign_nearest(review_z, centers)
                for key, centers in variant_centroids.items()
            }
        analysis = build_analysis(
            canonical_labels=canonical_labels,
            canonical_centroids=vocab.centroids,
            canonical_metadata=vocab.metadata,
            sampled_feature_values_sha256=sample_hash,
            review_rows=review_rows,
            review_canonical_labels=review_canonical,
            variant_labels=variant_labels,
            variant_centroids=variant_centroids,
            review_variant_labels=review_variant_labels,
            fit_info=fit_info,
        )
        _require(
            analysis["canonical_reconstruction_control"]["pass"],
            "canonical exact-sample/frozen-centroid reconstruction control failed",
        )

        # Freeze all 32 x 9 mappings on the label-blind vocabulary sample before
        # reading any development KRAS label into the association analysis.
        correspondence_frame = pd.DataFrame(analysis["all_anchor_correspondences"])
        correspondence_frame["clusters_to_cover_80pct"] = correspondence_frame[
            "clusters_to_cover_80pct"
        ].map(lambda value: json.dumps(value, separators=(",", ":")))
        _write_csv_once(stage / CORRESPONDENCE_NAME, correspondence_frame)

        with threadpool_limits(limits=threads):
            patient_profiles, development_hash, profile_reconstruction = (
                development_patient_abundance(
                    inputs,
                    vocab,
                    variant_centroids,
                    analysis["all_anchor_correspondences"],
                )
            )
        downstream = association_analysis(inputs, patient_profiles)
        _write_parquet_once(stage / PATIENT_PROFILES_NAME, patient_profiles)
        _write_csv_once(
            stage / ASSOCIATIONS_NAME,
            pd.DataFrame(downstream["association_effects"]),
        )

        input_receipt = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "input_identities": preflight["input_identities"],
            "feature_root": str(inputs.feature_root),
            "corrected_aim4_root": str(inputs.corrected_aim4_root),
            "feature_inventory_sha256": preflight["feature_inventory_sha256"],
            "exact_sampled_feature_values_sha256": sample_hash,
            "exact_reviewed_feature_values_sha256": review_hash,
            "exact_full_e0_feature_values_sha256": development_hash,
            "source_snapshot": source_manifest,
            "environment": _environment(),
        }
        _write_json_once(stage / INPUT_RECEIPT_NAME, input_receipt)

        assignment_arrays = {"canonical_k32": canonical_labels, "review_canonical_k32": review_canonical}
        assignment_arrays.update({f"sample_{key}": value for key, value in variant_labels.items()})
        assignment_arrays.update({f"review_{key}": value for key, value in review_variant_labels.items()})
        _write_npz_once(stage / ASSIGNMENTS_NAME, assignment_arrays)
        center_arrays = {"canonical_k32": vocab.centroids}
        center_arrays.update(variant_centroids)
        _write_npz_once(stage / CENTROIDS_NAME, center_arrays)

        variant_frame = pd.DataFrame(analysis["variant_metrics"])
        mapping_frame = pd.DataFrame(analysis["anchor_mappings"])
        pair_frame = pd.DataFrame(analysis["seed_pair_metrics"])
        for column in ("clusters_to_cover_80pct", "reviewed_cluster_distribution"):
            mapping_frame[column] = mapping_frame[column].map(
                lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"))
            )
        _write_csv_once(stage / "variant_metrics.csv", variant_frame)
        _write_csv_once(stage / "anchor_mappings.csv", mapping_frame)
        _write_csv_once(stage / "seed_pair_metrics.csv", pair_frame)

        results = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "scope": {
                "question": "k=24/32/40 and K-means initialization stability of the frozen Aim-4 vocabulary",
                "sample": "exact canonical k32 sampled tile draw",
                "projection": "exact canonical k32 normalization and PCA basis",
                "varied": ["number_of_clusters", "K-means_initialization_seed"],
                "not_varied": [
                    "feature_encoder",
                    "tile_sample",
                    "PCA_basis",
                    "development_population",
                    "A_and_D_definitions",
                    "slide_to_patient_aggregation",
                    "attention_claims",
                    "transport_claims",
                ],
                "claim_limit": "Alternative-cluster correspondences do not transfer M04/M07 pathology identities; new blinded review is required.",
            },
            "protocol": {
                "k_values": list(K_VALUES),
                "cluster_seeds": list(CLUSTER_SEEDS),
                "canonical_k": CANONICAL_K,
                "canonical_seed": CANONICAL_SEED,
                "n_init": N_INIT,
                "algorithm": "lloyd",
                "kmeans_execution": "forked_independent_variants_single_thread_each",
                "kmeans_workers": min(
                    int(threads), len(K_VALUES) * len(CLUSTER_SEEDS)
                ),
                "projection_assignment_thread_cap": int(threads),
                "inertia_contract": INERTIA_CONTRACT,
                "coverage_target": COVERAGE_TARGET,
                "canonical_reconstruction_control": {
                    "sample_values_sha256": CANONICAL_SAMPLE_VALUES_SHA256,
                    "maximum_cluster_mass_total_variation": (
                        CANONICAL_MAX_CLUSTER_MASS_TOTAL_VARIATION
                    ),
                    "maximum_single_cluster_mass_delta": (
                        CANONICAL_MAX_SINGLE_CLUSTER_MASS_DELTA
                    ),
                    "same_seed_refit_role": (
                        "sensitivity_variant_not_an_identity_control"
                    ),
                },
                "association_bootstrap_replicates": N_BOOTSTRAP,
                "association_fdr_alpha": FDR_ALPHA,
                "association_family_size": CANONICAL_K,
                "anchors": {str(key): value for key, value in ANCHORS.items()},
                "threads": int(threads),
            },
            "counts": preflight["counts"],
            "canonical_patient_profile_reconstruction": profile_reconstruction,
            "downstream_abundance_claim_stability": downstream,
            **analysis,
        }
        _write_json_once(stage / RESULTS_NAME, results)

        completion = {
            "schema_version": SCHEMA_VERSION,
            "status": "COMPLETE",
            "created_at_utc": _utc_now(),
            "output_root": str(output),
            "payload_inventory": relative_inventory(stage, excluded={COMPLETION_NAME}),
            "results_sha256": sha256_file(stage / RESULTS_NAME),
            "input_receipt_sha256": sha256_file(stage / INPUT_RECEIPT_NAME),
        }
        _write_json_once(stage / COMPLETION_NAME, completion)
        _rename_noreplace(stage, output)
    except Exception:
        print(f"FAILED stage retained without deletion: {stage}", file=sys.stderr, flush=True)
        raise
    print(f"published immutable Aim-4 sensitivity: {output}", flush=True)
    return output


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StabilityError(f"cannot read JSON object {path}: {exc}") from exc
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _compare_identity(record: dict[str, Any], label: str) -> None:
    path = Path(str(record.get("path", "")))
    current = identity(path)
    _require(current == record, f"{label} identity drifted: {path}")


def verify_output(output_root: Path, *, replay_input: bool = False) -> dict[str, Any]:
    root = _existing_dir(output_root, "output root")
    completion = _read_json(root / COMPLETION_NAME)
    _require(completion.get("status") == "COMPLETE", "completion status is not COMPLETE")
    _require(Path(str(completion.get("output_root"))) == root, "completion output-root mismatch")
    observed_inventory = relative_inventory(
        root, excluded={COMPLETION_NAME, VERIFICATION_NAME}
    )
    _require(observed_inventory == completion.get("payload_inventory"), "payload inventory/hash mismatch")
    _require(sha256_file(root / RESULTS_NAME) == completion.get("results_sha256"), "results hash mismatch")
    _require(
        sha256_file(root / INPUT_RECEIPT_NAME) == completion.get("input_receipt_sha256"),
        "input receipt hash mismatch",
    )

    input_receipt = _read_json(root / INPUT_RECEIPT_NAME)
    for label, record in input_receipt.get("input_identities", {}).items():
        _compare_identity(record, label)
    source_manifest = input_receipt.get("source_snapshot")
    _require(isinstance(source_manifest, list), "source snapshot manifest missing")
    for record in source_manifest:
        snapshot = root / "source_snapshot" / str(record["path"])
        _require(snapshot.is_file(), f"source snapshot missing: {snapshot}")
        _require(identity(snapshot)["sha256"] == record["sha256"], f"source snapshot hash mismatch: {snapshot}")
    verifier_source = next(
        (
            record
            for record in source_manifest
            if record.get("path") == "tools/aim4_vocab_stability.py"
        ),
        None,
    )
    _require(verifier_source is not None, "source snapshot lacks the verifier implementation")
    _require(
        sha256_file(Path(__file__).resolve()) == verifier_source["sha256"],
        "verifier bytes differ from the sealed source snapshot; run the verifier copy "
        "inside output_root/source_snapshot/tools instead",
    )
    _require(
        sha256_file(root / "feature_inventory.json")
        == input_receipt.get("feature_inventory_sha256"),
        "feature inventory receipt hash mismatch",
    )
    _require(
        input_receipt.get("exact_sampled_feature_values_sha256")
        == CANONICAL_SAMPLE_VALUES_SHA256,
        "sampled feature values do not match the pinned canonical hash",
    )

    records = input_receipt["input_identities"]
    inputs = Inputs(
        canonical_vocab=Path(records["canonical_vocab"]["path"]),
        sample_plan=Path(records["sample_plan"]["path"]),
        feature_root=Path(str(input_receipt.get("feature_root", ""))),
        atlas_report=Path(records["atlas_report"]["path"]),
        review_key=Path(records["review_key"]["path"]),
        development_manifest=Path(records["development_manifest"]["path"]),
        corrected_aim4_root=Path(str(input_receipt.get("corrected_aim4_root", ""))),
        corrected_numeric_completion=Path(
            records["corrected_aim4_numeric_completion"]["path"]
        ),
        corrected_numeric_verification=Path(
            records["corrected_aim4_numeric_verification"]["path"]
        ),
        canonical_patient_profiles=Path(records["canonical_patient_profiles"]["path"]),
        canonical_specificity=Path(records["canonical_specificity"]["path"]),
    )
    _require(inputs.feature_root.is_absolute(), "input receipt lacks an absolute feature root")
    _require(
        inputs.corrected_aim4_root.is_absolute(),
        "input receipt lacks an absolute corrected Aim-4 root",
    )
    validate_corrected_aim4_numeric_seal(inputs)

    results = _read_json(root / RESULTS_NAME)
    _require(results.get("status") == "COMPLETE", "results status is not COMPLETE")
    protocol = results.get("protocol", {})
    _require(tuple(protocol.get("k_values", ())) == K_VALUES, "unexpected k values")
    _require(tuple(protocol.get("cluster_seeds", ())) == CLUSTER_SEEDS, "unexpected cluster seeds")
    _require(int(protocol.get("n_init", -1)) == N_INIT, "unexpected K-means n_init")
    replay_threads = int(protocol.get("threads", -1))
    _require(1 <= replay_threads <= 18, "unexpected physical-core cap")
    _require(
        protocol.get("kmeans_execution")
        == "forked_independent_variants_single_thread_each"
        and int(protocol.get("kmeans_workers", -1))
        == min(replay_threads, len(K_VALUES) * len(CLUSTER_SEEDS))
        and int(protocol.get("projection_assignment_thread_cap", -1))
        == replay_threads,
        "unexpected deterministic parallel-execution contract",
    )
    _require(
        protocol.get("inertia_contract") == INERTIA_CONTRACT,
        "unexpected deterministic inertia contract",
    )
    control_protocol = protocol.get("canonical_reconstruction_control", {})
    _require(
        control_protocol.get("sample_values_sha256")
        == CANONICAL_SAMPLE_VALUES_SHA256
        and control_protocol.get("maximum_cluster_mass_total_variation")
        == CANONICAL_MAX_CLUSTER_MASS_TOTAL_VARIATION
        and control_protocol.get("maximum_single_cluster_mass_delta")
        == CANONICAL_MAX_SINGLE_CLUSTER_MASS_DELTA
        and control_protocol.get("same_seed_refit_role")
        == "sensitivity_variant_not_an_identity_control",
        "unexpected canonical reconstruction-control protocol",
    )
    _require(
        results.get("canonical_reconstruction_control", {}).get("pass") is True,
        "canonical reconstruction control is not PASS",
    )

    with np.load(root / ASSIGNMENTS_NAME, allow_pickle=False) as assignment_blob:
        assignments = {key: np.asarray(assignment_blob[key]) for key in assignment_blob.files}
    with np.load(root / CENTROIDS_NAME, allow_pickle=False) as center_blob:
        centers = {key: np.asarray(center_blob[key]) for key in center_blob.files}
    expected_keys = {f"k{k}_seed{seed}" for k in K_VALUES for seed in CLUSTER_SEEDS}
    _require(set(centers) == {"canonical_k32", *expected_keys}, "centroid inventory mismatch")
    _require(
        set(assignments)
        == {
            "canonical_k32",
            "review_canonical_k32",
            *(f"sample_{key}" for key in expected_keys),
            *(f"review_{key}" for key in expected_keys),
        },
        "assignment inventory mismatch",
    )
    n_sample = int(results["counts"]["sampled_tiles"])
    n_review = int(results["counts"]["reviewed_anchor_tiles"])
    _require(assignments["canonical_k32"].shape == (n_sample,), "canonical assignment length mismatch")
    _require(assignments["review_canonical_k32"].shape == (n_review,), "review assignment length mismatch")
    for k in K_VALUES:
        for seed in CLUSTER_SEEDS:
            key = f"k{k}_seed{seed}"
            _require(centers[key].shape == (k, EXPECTED_COMPONENTS), f"centroid shape mismatch: {key}")
            _require(assignments[f"sample_{key}"].shape == (n_sample,), f"sample labels mismatch: {key}")
            _require(assignments[f"review_{key}"].shape == (n_review,), f"review labels mismatch: {key}")

    review_rows = load_review_rows(inputs)
    variant_metric_rows = {
        str(row["variant"]): row for row in results["variant_metrics"]
    }
    _require(
        set(variant_metric_rows) == expected_keys
        and len(results["variant_metrics"]) == len(expected_keys),
        "variant-metric inventory mismatch",
    )
    fit_info: dict[str, dict[str, Any]] = {}
    for key, row in variant_metric_rows.items():
        model_inertia = float(row["model_inertia"])
        replay_inertia = float(row["replay_inertia_float64"])
        _require(
            np.isfinite(model_inertia)
            and model_inertia > 0.0
            and np.isfinite(replay_inertia)
            and replay_inertia >= 0.0,
            f"inertia diagnostic is outside its numeric contract: {key}",
        )
        _require("inertia" not in row, f"ambiguous legacy inertia field is present: {key}")
        fit_info[key] = {
            "k": int(row["k"]),
            "seed": int(row["seed"]),
            "model_inertia": model_inertia,
            "replay_inertia_float64": replay_inertia,
            "n_iter": int(row["n_iter"]),
        }
    recomputed = build_analysis(
        canonical_labels=assignments["canonical_k32"],
        canonical_centroids=centers["canonical_k32"],
        canonical_metadata=load_vocabulary(inputs).metadata,
        sampled_feature_values_sha256=str(
            input_receipt["exact_sampled_feature_values_sha256"]
        ),
        review_rows=review_rows,
        review_canonical_labels=assignments["review_canonical_k32"],
        variant_labels={key: assignments[f"sample_{key}"] for key in expected_keys},
        variant_centroids={key: centers[key] for key in expected_keys},
        review_variant_labels={key: assignments[f"review_{key}"] for key in expected_keys},
        fit_info=fit_info,
    )
    recorded_analysis = {
        key: results[key]
        for key in (
            "canonical_reconstruction_control",
            "variant_metrics",
            "anchor_mappings",
            "all_anchor_correspondences",
            "seed_pair_metrics",
        )
    }
    _require(
        _json_bytes(recomputed) == _json_bytes(recorded_analysis),
        "stored stability statistics do not recompute from sealed assignments",
    )

    sealed_profiles = pd.read_parquet(root / PATIENT_PROFILES_NAME)
    recomputed_downstream = association_analysis(inputs, sealed_profiles)
    _require(
        _json_bytes(recomputed_downstream)
        == _json_bytes(results["downstream_abundance_claim_stability"]),
        "stored A/D abundance statistics do not recompute from sealed patient profiles",
    )

    if replay_input:
        current_preflight = inspect_inputs(inputs)
        _require(
            current_preflight["feature_inventory_sha256"] == input_receipt["feature_inventory_sha256"],
            "feature inventory drifted",
        )
        vocab = load_vocabulary(inputs)
        with threadpool_limits(limits=replay_threads):
            review_rows, review_z, review_hash = collect_review_projection(
                inputs, vocab
            )
            z, sample_hash = collect_projected_sample(inputs, vocab)
            _require(
                sample_hash == CANONICAL_SAMPLE_VALUES_SHA256,
                "replayed sample differs from the pinned canonical hash",
            )
            _require(sample_hash == input_receipt["exact_sampled_feature_values_sha256"], "sampled feature values drifted")
            _require(review_hash == input_receipt["exact_reviewed_feature_values_sha256"], "reviewed feature values drifted")
            _require(np.array_equal(assign_nearest(z, centers["canonical_k32"]), assignments["canonical_k32"]), "canonical assignments do not replay")
            _require(np.array_equal(assign_nearest(review_z, centers["canonical_k32"]), assignments["review_canonical_k32"]), "review canonical assignments do not replay")
            for k in K_VALUES:
                for seed in CLUSTER_SEEDS:
                    key = f"k{k}_seed{seed}"
                    replayed = assign_nearest(z, centers[key])
                    _require(np.array_equal(replayed, assignments[f"sample_{key}"]), f"sample assignments do not replay: {key}")
                    replayed_review = assign_nearest(review_z, centers[key])
                    _require(np.array_equal(replayed_review, assignments[f"review_{key}"]), f"review assignments do not replay: {key}")
                    inertia = deterministic_replay_inertia(
                        z, centers[key], replayed
                    )
                    recorded = float(
                        variant_metric_rows[key]["replay_inertia_float64"]
                    )
                    _require(
                        inertia == recorded,
                        f"deterministic inertia does not replay exactly: {key}",
                    )
            replayed_profiles, development_hash, profile_reconstruction = (
                development_patient_abundance(
                    inputs,
                    vocab,
                    {key: centers[key] for key in expected_keys},
                    recomputed["all_anchor_correspondences"],
                )
            )
        _require(
            development_hash == input_receipt["exact_full_e0_feature_values_sha256"],
            "full E0 feature values drifted",
        )
        _require(
            profile_reconstruction == results["canonical_patient_profile_reconstruction"],
            "canonical patient-profile replay differs",
        )
        sort_columns = ["variant", "patient_id", "prototype"]
        expected_profiles = sealed_profiles.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        observed_profiles = replayed_profiles.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            observed_profiles,
            expected_profiles,
            check_exact=True,
            check_like=False,
        )

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "verified_at_utc": _utc_now(),
        "output_root": str(root),
        "completion_sha256": sha256_file(root / COMPLETION_NAME),
        "results_sha256": sha256_file(root / RESULTS_NAME),
        "replay_input": bool(replay_input),
        "checks": {
            "exclusive_payload_inventory": "PASS",
            "input_hashes": "PASS",
            "source_snapshot": "PASS",
            "assignment_and_centroid_shapes": "PASS",
            "statistical_recomputation": "PASS",
            "full_m32_A_D_family_recomputation": "PASS",
            "canonical_reconstruction_control": "PASS",
            "exact_feature_and_assignment_replay": "PASS" if replay_input else "NOT_REQUESTED",
            "deterministic_inertia_replay": "PASS" if replay_input else "NOT_REQUESTED",
            "pathology_label_transfer": "PROHIBITED_WITHOUT_NEW_BLINDED_REVIEW",
        },
    }
    return receipt


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--canonical-vocab", required=True)
    parser.add_argument("--sample-plan", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--atlas-report", required=True)
    parser.add_argument("--review-key", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--corrected-aim4-root", required=True)
    parser.add_argument("--output-root", required=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight", help="read-only validation and resource estimate")
    add_input_arguments(preflight)
    run = sub.add_parser("run", help="fit and publish all nine sensitivity vocabularies")
    add_input_arguments(run)
    run.add_argument("--threads", type=int, default=12)
    run.add_argument("--apply", action="store_true", help="required acknowledgement for new output creation")
    verify = sub.add_parser("verify", help="verify a completed append-only output")
    verify.add_argument("--output-root", required=True)
    verify.add_argument("--replay-input", action="store_true", help="reread exact frozen feature values and replay assignments")
    verify.add_argument("--write-receipt", action="store_true", help="append verification_receipt.json")
    verify.add_argument("--apply", action="store_true", help="required with --write-receipt")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "preflight":
            inputs = resolve_inputs(args)
            output = new_output_root(args.output_root, inputs)
            report = inspect_inputs(inputs)
            report["proposed_output_root"] = str(output)
            printable = dict(report)
            printable.pop("feature_inventory")
            print(json.dumps(printable, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.command == "run":
            if not args.apply:
                raise StabilityError("run is mutating; inspect preflight, then pass --apply")
            inputs = resolve_inputs(args)
            run_campaign(inputs, Path(args.output_root), threads=int(args.threads))
            return 0
        if args.command == "verify":
            if args.write_receipt and not args.apply:
                raise StabilityError("--write-receipt is mutating; pass --apply")
            receipt = verify_output(Path(args.output_root), replay_input=bool(args.replay_input))
            if args.write_receipt:
                root_path = _existing_dir(args.output_root, "output root")
                _write_json_once(root_path / VERIFICATION_NAME, receipt)
            print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
            return 0
        raise AssertionError(args.command)
    except (StabilityError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
