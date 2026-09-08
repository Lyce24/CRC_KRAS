#!/usr/bin/env python3
"""Append-only recovery of E2c with a native-logit residual linear adapter.

This component deliberately does not use the canonical E2c analysis result and
does not reconstruct its zero-shot predictor from a float32 classifier head.
It imports the immutable E2c embeddings and their stored native logits from an
explicit upstream lineage, verifies that whole import chain, and fits

    eta(h) = eta_native + h @ delta_w + delta_b

with an L2 penalty on ``(delta_w, delta_b)``.  Thus lambda=infinity is exactly
the deployed/native S0 predictor, while every finite lambda is continuous from
that same baseline.  The encoder, attention module, source head, embeddings,
and native logits remain frozen; no MIL model is fitted or scored here.

The output root is explicit, must be absolute, and must not already exist.
Every artifact below that newly-created root is written with exclusive and
atomic publication.  A failed run is retained as a partial lineage and must be
restarted with another new output root.

Examples:
    python aim2_residual_adaptation.py verify-inputs \
      --input-lineage-root /abs/path/to/aim2_cap8192_v4_20260819 --cap 8192

    python aim2_residual_adaptation.py run \
      --input-lineage-root /abs/path/to/aim2_cap8192_v4_20260819 \
      --output-root /abs/path/to/aim2_cap8192_e2c_offset_v1_20260819 \
      --cap 8192 --reps 100 --n-bootstrap 10000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

# Only pure statistical helpers are reused.  All input path resolution and all
# provenance validation in this recovery component are independent of aim2_head_adaptation_base's
# lineage environment and canonical result path.
import aim2_head_adaptation_base as e2c_stats  # noqa: E402

COHORTS: tuple[tuple[str, str | None], ...] = (("RIH", None), ("SurGen", "SR1482"))
SEEDS: tuple[int, ...] = (42, 43, 44)
BUDGETS: tuple[int, ...] = (2, 4, 8)
N_FOLDS = 5
PRIMARY_SEED = 42
BOOTSTRAP_SEED = 20260817
DEFAULT_REPS = 100
DEFAULT_N_BOOTSTRAP = 10_000
LAMBDA_GRID: tuple[float, ...] = (
    np.inf,
    1e4,
    3e3,
    1e3,
    3e2,
    1e2,
    3e1,
    1e1,
    3.0,
    1.0,
)
EPS = 1e-6
KKT_TOL = 1e-6
OBJECTIVE_TOL = 1e-9


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected file artifact: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _require_absolute(path: Path, *, label: str, must_exist: bool) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an explicit absolute path: {path}")
    resolved = path.resolve(strict=must_exist)
    if must_exist and not resolved.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_file_identity(
    recorded: dict[str, Any],
    *,
    label: str,
    expected_path: Path | None = None,
    actual_path: Path | None = None,
) -> dict[str, Any]:
    required = {"path", "size_bytes", "sha256"}
    if not isinstance(recorded, dict) or not required.issubset(recorded):
        raise ValueError(f"{label}: malformed artifact identity")
    recorded_path = Path(str(recorded["path"])).expanduser()
    if not recorded_path.is_absolute():
        raise ValueError(f"{label}: recorded path is not absolute: {recorded_path}")
    if expected_path is not None and recorded_path.resolve(strict=True) != expected_path.resolve(
        strict=True
    ):
        raise RuntimeError(
            f"{label}: recorded path mismatch: {recorded_path} != {expected_path}"
        )
    check_path = actual_path if actual_path is not None else recorded_path
    actual = _artifact_identity(check_path)
    if int(recorded["size_bytes"]) != actual["size_bytes"]:
        raise RuntimeError(f"{label}: size mismatch for {check_path}")
    if str(recorded["sha256"]) != actual["sha256"]:
        raise RuntimeError(f"{label}: SHA256 mismatch for {check_path}")
    return actual


def _write_bytes_once_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_once_atomic(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, allow_nan=False, default=str) + "\n").encode()
    _write_bytes_once_atomic(path, payload)


def _current_source_payloads() -> dict[str, bytes]:
    return {
        "aim2_residual_adaptation.py": Path(__file__).read_bytes(),
        "aim2_head_adaptation_base.py": (REPO / "aim2_head_adaptation_base.py").read_bytes(),
    }


def _assert_source_bytes_unchanged(expected: dict[str, bytes]) -> None:
    current = _current_source_payloads()
    if set(current) != set(expected):
        raise RuntimeError("Runtime source inventory changed during the E2c recovery run")
    changed = [name for name in current if current[name] != expected[name]]
    if changed:
        raise RuntimeError(
            "Runtime source bytes changed during the E2c recovery run: "
            + ", ".join(sorted(changed))
        )


def _embedding_path(root: Path, target: str, kind: str, cap: int) -> Path:
    return root / "e2c" / "embeddings" / f"cap{cap}_{target.lower()}_{kind}.parquet"


def _head_path(root: Path, target: str, cap: int) -> Path:
    return root / "e2c" / "heads" / f"cap{cap}_{target.lower()}.npz"


def _score_path(root: Path, target: str, kind: str, seed: int, cap: int) -> Path:
    return (
        root
        / "e2a"
        / "scores"
        / f"pb_cap{cap}_{target.lower()}_seed{seed}_{kind}.parquet"
    )


def _receipt_path(artifact: Path) -> Path:
    return artifact.with_suffix(".receipt.json")


def _validate_artifact_receipt(
    receipt_path: Path,
    artifact_path: Path,
    *,
    lineage_name: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    receipt = _read_json(receipt_path)
    if receipt.get("schema_version") != 2:
        raise RuntimeError(f"{label}: receipt schema must be 2")
    receipt_lineage = receipt.get("lineage", receipt.get("inputs", {}).get("lineage"))
    if receipt_lineage != lineage_name:
        raise RuntimeError(
            f"{label}: receipt lineage {receipt_lineage!r} != {lineage_name!r}"
        )
    artifact_identity = _validate_file_identity(
        receipt.get("artifact", {}),
        label=f"{label} artifact",
        expected_path=artifact_path,
    )
    return receipt, _artifact_identity(receipt_path), artifact_identity


@dataclass(frozen=True)
class InputBundle:
    root: Path
    lineage: str
    lineage_start: dict[str, Any]
    lineage_start_identity: dict[str, Any]
    embeddings: dict[tuple[str, str], Path]
    heads: dict[str, Path]
    manifests: dict[tuple[str, str], Path]
    imports: dict[str, Any]
    diagnostics: dict[str, Any]

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "verified_utc": _utc_now(),
            "input_lineage": self.lineage,
            "input_lineage_root": str(self.root),
            "lineage_start": self.lineage_start_identity,
            "imports": self.imports,
            "diagnostics": self.diagnostics,
            "canonical_e2c_result": {
                "path": str(
                    self.root / "e2c" / "analysis" / "e2c_true_kshot_cap8192.json"
                ),
                "consumed": False,
                "note": "The canonical E2c result is not an input to this recovery.",
            },
        }


def _embedding_contract(
    embedding: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    target: str,
    kind: str,
) -> list[str]:
    required_manifest = {"slide_id", "patient_id", "target_label"}
    if not required_manifest.issubset(manifest.columns):
        raise ValueError(f"{target}/{kind}: manifest missing {required_manifest - set(manifest)}")
    if manifest["slide_id"].duplicated().any():
        raise ValueError(f"{target}/{kind}: manifest slide_id is not unique")
    if manifest[["slide_id", "patient_id", "target_label"]].isna().any().any():
        raise ValueError(f"{target}/{kind}: manifest has missing core fields")
    if set(manifest["target_label"].astype(int).unique()) - {0, 1}:
        raise ValueError(f"{target}/{kind}: target_label is not binary")

    ecols = [f"e{i}" for i in range(512)]
    required_embedding = {"slide_id", "seed", "logit", *ecols}
    if not required_embedding.issubset(embedding.columns):
        raise ValueError(
            f"{target}/{kind}: embedding missing {required_embedding - set(embedding.columns)}"
        )
    if embedding[["slide_id", "seed"]].duplicated().any():
        raise ValueError(f"{target}/{kind}: duplicate (slide_id, seed) embedding rows")
    if set(embedding["seed"].astype(int).unique()) != set(SEEDS):
        raise ValueError(f"{target}/{kind}: embedding seeds are not exactly {SEEDS}")
    numeric = embedding[["logit", *ecols]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{target}/{kind}: embedding contains non-finite values")
    expected_slides = set(manifest["slide_id"].astype(str))
    for seed in SEEDS:
        observed = set(embedding.loc[embedding["seed"].eq(seed), "slide_id"].astype(str))
        if observed != expected_slides:
            raise ValueError(f"{target}/{kind}/seed{seed}: slide coverage mismatch")
    return ecols


def _patient_data(
    bundle: InputBundle,
    target: str,
    kind: str,
    *,
    subcohort: str | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray]]:
    emb = pd.read_parquet(bundle.embeddings[(target, kind)])
    manifest = pd.read_csv(bundle.manifests[(target, kind)])
    if subcohort is not None:
        if "subcohort" not in manifest:
            raise ValueError(f"{target}/{kind}: no subcohort column for {subcohort}")
        manifest = manifest.loc[manifest["subcohort"].eq(subcohort)].copy()
        emb = emb.loc[emb["slide_id"].isin(set(manifest["slide_id"]))].copy()
    sid = manifest.set_index("slide_id")
    emb = emb.assign(patient_id=emb["slide_id"].map(sid["patient_id"]))
    if emb["patient_id"].isna().any():
        raise RuntimeError(f"{target}/{kind}: embedding slide lacks a patient mapping")
    patients = (
        manifest.drop_duplicates("patient_id")[["patient_id", "target_label"]]
        .rename(columns={"target_label": "label"})
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    repeated = manifest.groupby("patient_id")["target_label"].nunique()
    if (repeated > 1).any():
        raise ValueError(f"{target}/{kind}: a patient has inconsistent labels")
    ecols = [f"e{i}" for i in range(512)]
    matrices: dict[int, np.ndarray] = {}
    native: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        rows = emb.loc[emb["seed"].eq(seed)]
        h = rows.groupby("patient_id")[ecols].mean()
        eta = rows.groupby("patient_id")["logit"].mean()
        order = patients["patient_id"]
        matrices[seed] = h.loc[order].to_numpy(dtype=np.float64)
        native[seed] = eta.loc[order].to_numpy(dtype=np.float64)
    return patients, matrices, native


def _validate_one_target(
    root: Path,
    *,
    lineage_name: str,
    target: str,
    cap: int,
    frozen_source: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, str], Path], dict[str, Path]]:
    imported: dict[str, Any] = {"embeddings": {}, "native_scores": {}}
    diagnostics: dict[str, Any] = {}
    embeddings: dict[tuple[str, str], Path] = {}
    manifests: dict[str, Path] = {}
    embedding_frames: dict[str, pd.DataFrame] = {}
    embedding_receipts: dict[str, dict[str, Any]] = {}

    for kind in ("primary", "metastatic"):
        artifact = _embedding_path(root, target, kind, cap)
        receipt_path = _receipt_path(artifact)
        receipt, receipt_id, artifact_id = _validate_artifact_receipt(
            receipt_path,
            artifact,
            lineage_name=lineage_name,
            label=f"{target}/{kind} embedding",
        )
        inputs = receipt.get("inputs", {})
        code = inputs.get("code", {})
        frozen_code_id = _validate_file_identity(
            code,
            label=f"{target}/{kind} frozen E2c source",
            actual_path=frozen_source,
        )
        manifest_record = inputs.get("manifest", {})
        manifest_path = Path(str(manifest_record.get("path", "")))
        manifest_id = _validate_file_identity(
            manifest_record,
            label=f"{target}/{kind} manifest",
        )
        manifest = pd.read_csv(manifest_path)
        emb = pd.read_parquet(artifact)
        ecols = _embedding_contract(emb, manifest, target=target, kind=kind)
        if int(receipt.get("n_rows", -1)) != len(emb):
            raise RuntimeError(f"{target}/{kind}: embedding receipt row count mismatch")
        if int(receipt.get("n_slides", -1)) != emb["slide_id"].nunique():
            raise RuntimeError(f"{target}/{kind}: embedding receipt slide count mismatch")
        checkpoints: dict[str, Any] = {}
        for seed in SEEDS:
            checkpoints[str(seed)] = _validate_file_identity(
                inputs.get("checkpoints", {}).get(str(seed), {}),
                label=f"{target}/{kind}/seed{seed} checkpoint",
            )
        imported["embeddings"][kind] = {
            "artifact": artifact_id,
            "receipt": receipt_id,
            "manifest": manifest_id,
            "frozen_source_snapshot": frozen_code_id,
            "feature_store": inputs.get("feature_store"),
            "checkpoints": checkpoints,
        }
        embeddings[(target, kind)] = artifact
        manifests[kind] = manifest_path
        embedding_frames[kind] = emb
        embedding_receipts[kind] = receipt
        imported["native_scores"][kind] = {}

        for seed in SEEDS:
            score = _score_path(root, target, kind, seed, cap)
            score_receipt_path = _receipt_path(score)
            score_receipt, score_receipt_id, score_id = _validate_artifact_receipt(
                score_receipt_path,
                score,
                lineage_name=lineage_name,
                label=f"{target}/{kind}/seed{seed} native score",
            )
            score_inputs = score_receipt.get("inputs", {})
            expected_fields = {
                "target": target,
                "kind": kind,
                "seed": seed,
                "cap": cap,
            }
            for field, expected in expected_fields.items():
                if score_inputs.get(field) != expected:
                    raise RuntimeError(
                        f"{target}/{kind}/seed{seed}: score receipt {field} mismatch"
                    )
            _validate_file_identity(
                score_inputs.get("checkpoint", {}),
                label=f"{target}/{kind}/seed{seed} score checkpoint",
            )
            if score_inputs.get("checkpoint") != receipt["inputs"]["checkpoints"][str(seed)]:
                raise RuntimeError(f"{target}/{kind}/seed{seed}: checkpoint contracts differ")
            if score_inputs.get("manifest") != receipt["inputs"]["manifest"]:
                raise RuntimeError(f"{target}/{kind}/seed{seed}: manifest contracts differ")
            if score_inputs.get("feature_store") != receipt["inputs"].get("feature_store"):
                raise RuntimeError(f"{target}/{kind}/seed{seed}: feature contracts differ")
            scores = pd.read_parquet(score)
            required_score = {"slide_id", "seed", "logit"}
            if not required_score.issubset(scores):
                raise ValueError(f"{target}/{kind}/seed{seed}: malformed score artifact")
            if scores["slide_id"].duplicated().any() or not scores["seed"].eq(seed).all():
                raise ValueError(f"{target}/{kind}/seed{seed}: invalid score keys")
            from_embedding = (
                emb.loc[emb["seed"].eq(seed), ["slide_id", "logit"]]
                .sort_values("slide_id")
                .reset_index(drop=True)
            )
            from_score = scores[["slide_id", "logit"]].sort_values("slide_id").reset_index(drop=True)
            if not from_embedding["slide_id"].equals(from_score["slide_id"]):
                raise RuntimeError(f"{target}/{kind}/seed{seed}: native score slide mismatch")
            if not np.array_equal(
                from_embedding["logit"].to_numpy(), from_score["logit"].to_numpy()
            ):
                raise RuntimeError(
                    f"{target}/{kind}/seed{seed}: embedded native logits are not bit-exact scores"
                )
            imported["native_scores"][kind][str(seed)] = {
                "artifact": score_id,
                "receipt": score_receipt_id,
                "native_logit_reproduction": "bit-exact",
            }

    head = _head_path(root, target, cap)
    head_receipt_path = _receipt_path(head)
    head_receipt, head_receipt_id, head_id = _validate_artifact_receipt(
        head_receipt_path,
        head,
        lineage_name=lineage_name,
        label=f"{target} source head",
    )
    head_inputs = head_receipt.get("inputs", {})
    head_source_id = _validate_file_identity(
        head_inputs.get("code", {}),
        label=f"{target} frozen head source",
        actual_path=frozen_source,
    )
    head_checkpoints: dict[str, Any] = {}
    completion_receipts: dict[str, Any] = {}
    for seed in SEEDS:
        checkpoint = _validate_file_identity(
            head_inputs.get("checkpoints", {}).get(str(seed), {}),
            label=f"{target}/seed{seed} head checkpoint",
        )
        expected = embedding_receipts["primary"]["inputs"]["checkpoints"][str(seed)]
        if head_inputs["checkpoints"][str(seed)] != expected:
            raise RuntimeError(f"{target}/seed{seed}: embedding/head checkpoint contracts differ")
        head_checkpoints[str(seed)] = checkpoint
        completion_receipts[str(seed)] = _validate_file_identity(
            head_inputs.get("completion_receipts", {}).get(str(seed), {}),
            label=f"{target}/seed{seed} completion receipt",
        )
    with np.load(head) as archive:
        if set(archive.files) != {"w", "b", "seeds"}:
            raise ValueError(f"{target}: unexpected source-head arrays {archive.files}")
        w = archive["w"].astype(np.float64)
        b = archive["b"].astype(np.float64)
        head_seeds = tuple(int(x) for x in archive["seeds"])
    if w.shape != (len(SEEDS), 512) or b.shape != (len(SEEDS),) or head_seeds != SEEDS:
        raise ValueError(f"{target}: source-head dimensional/seed contract mismatch")
    imported["source_head"] = {
        "artifact": head_id,
        "receipt": head_receipt_id,
        "frozen_source_snapshot": head_source_id,
        "checkpoints": head_checkpoints,
        "completion_receipts": completion_receipts,
        "role": "validated frozen upstream artifact; not used to reconstruct S0",
    }

    diagnostics["native_score_max_abs_error"] = 0.0
    diagnostics["by_kind"] = {}
    for kind in ("primary", "metastatic"):
        emb = embedding_frames[kind]
        ecols = [f"e{i}" for i in range(512)]
        manifest = pd.read_csv(manifests[kind])
        sid_to_patient = manifest.set_index("slide_id")["patient_id"]
        drift_by_seed: dict[str, Any] = {}
        for i, seed in enumerate(SEEDS):
            rows = emb.loc[emb["seed"].eq(seed)].copy()
            H = rows[ecols].to_numpy(dtype=np.float64)
            reconstructed = H @ w[i] + b[i]
            native = rows["logit"].to_numpy(dtype=np.float64)
            rows = rows.assign(patient_id=rows["slide_id"].map(sid_to_patient))
            patient_mean_H = rows.groupby("patient_id")[ecols].mean().to_numpy(dtype=np.float64)
            mean_reconstructed = (
                pd.Series(reconstructed, index=rows["patient_id"])
                .groupby(level=0)
                .mean()
                .sort_index()
                .to_numpy()
            )
            patient_mean_H = (
                rows.groupby("patient_id")[ecols]
                .mean()
                .sort_index()
                .to_numpy(dtype=np.float64)
            )
            distributed = patient_mean_H @ w[i] + b[i]
            drift_by_seed[str(seed)] = {
                "native_vs_fp64_head_max_abs_logit": float(
                    np.max(np.abs(native - reconstructed))
                ),
                "fp64_distributivity_max_abs_logit": float(
                    np.max(np.abs(mean_reconstructed - distributed))
                ),
            }
            if drift_by_seed[str(seed)]["fp64_distributivity_max_abs_logit"] > 1e-12:
                raise RuntimeError(f"{target}/{kind}/seed{seed}: fp64 distributivity failed")
        diagnostics["by_kind"][kind] = drift_by_seed
    diagnostics["interpretation"] = (
        "Native-vs-fp64 head drift is an informational BF16 deployment diagnostic, not a "
        "failure. The hard reproduction gate above compares stored native logits bit-exactly."
    )
    return imported, diagnostics, embeddings, manifests


def preflight_inputs(input_root: Path, cap: int) -> InputBundle:
    root = _require_absolute(input_root, label="--input-lineage-root", must_exist=True)
    lineage_start_path = root / "lineage_start.json"
    lineage_start = _read_json(lineage_start_path)
    if lineage_start.get("schema_version") != 1 or lineage_start.get("status") != "started":
        raise RuntimeError("Input lineage_start.json is not the expected immutable started receipt")
    lineage_name = str(lineage_start.get("lineage", ""))
    if not lineage_name:
        raise RuntimeError("Input lineage receipt has no lineage name")
    recorded_root = Path(str(lineage_start.get("lineage_root", "")))
    if not recorded_root.is_absolute() or recorded_root.resolve(strict=True) != root:
        raise RuntimeError("--input-lineage-root does not match lineage_start.json")
    frozen_source = root / "source_snapshot" / "aim2_head_adaptation_base.py"
    frozen_source_id = _artifact_identity(frozen_source)

    imports: dict[str, Any] = {
        "lineage_start": _artifact_identity(lineage_start_path),
        "frozen_e2c_source": frozen_source_id,
        "targets": {},
    }
    diagnostics: dict[str, Any] = {"targets": {}}
    embeddings: dict[tuple[str, str], Path] = {}
    manifests: dict[tuple[str, str], Path] = {}
    heads: dict[str, Path] = {}
    for target, _subcohort in COHORTS:
        target_imports, target_diagnostics, target_embeddings, target_manifests = (
            _validate_one_target(
                root,
                lineage_name=lineage_name,
                target=target,
                cap=cap,
                frozen_source=frozen_source,
            )
        )
        imports["targets"][target] = target_imports
        diagnostics["targets"][target] = target_diagnostics
        embeddings.update(target_embeddings)
        manifests.update({(target, kind): path for kind, path in target_manifests.items()})
        heads[target] = _head_path(root, target, cap)

    e2b_path = root / "eval" / f"e2b_metastatic_cap{cap}.json"
    e2b = _read_json(e2b_path)
    if e2b.get("lineage") != lineage_name or e2b.get("cap") != cap:
        raise RuntimeError("E2b artifact lineage/cap mismatch")
    imports["e2b_result"] = _artifact_identity(e2b_path)
    diagnostics["e2b_native_baseline_reproduction"] = {}
    diagnostics["e2b_input_contract"] = {}
    for target, _subcohort in COHORTS:
        recorded_inputs = e2b.get("targets", {}).get(target, {}).get("input_artifacts", {})
        target_imports = imports["targets"][target]
        for kind in ("primary", "metastatic"):
            receipt_key = f"{kind}_score_receipts"
            manifest_key = f"{kind}_manifest"
            for seed in SEEDS:
                current = target_imports["native_scores"][kind][str(seed)]["receipt"]
                if recorded_inputs.get(receipt_key, {}).get(str(seed)) != current:
                    raise RuntimeError(
                        f"{target}/{kind}/seed{seed}: E2b score-receipt identity mismatch"
                    )
            current_manifest = target_imports["embeddings"][kind]["manifest"]
            if recorded_inputs.get(manifest_key) != current_manifest:
                raise RuntimeError(f"{target}/{kind}: E2b manifest identity mismatch")
        diagnostics["e2b_input_contract"][target] = "PASS"

    bundle = InputBundle(
        root=root,
        lineage=lineage_name,
        lineage_start=lineage_start,
        lineage_start_identity=_artifact_identity(lineage_start_path),
        embeddings=embeddings,
        heads=heads,
        manifests=manifests,
        imports=imports,
        diagnostics=diagnostics,
    )
    for target, _subcohort in COHORTS:
        patients, _H, native = _patient_data(bundle, target, "metastatic")
        y = patients["label"].to_numpy(dtype=int)
        eta = np.mean(np.vstack([native[seed] for seed in SEEDS]), axis=0)
        auc = e2c_stats._auroc_logits(y, eta)  # noqa: SLF001
        reported = e2b.get("targets", {}).get(target, {}).get("metastatic_overall", {})
        if int(reported.get("n", -1)) != len(y):
            raise RuntimeError(f"{target}: E2b metastatic patient count mismatch")
        if abs(float(reported.get("auroc", np.nan)) - auc) > 1e-12:
            raise RuntimeError(f"{target}: native S0 does not reproduce E2b AUROC")
        diagnostics["e2b_native_baseline_reproduction"][target] = {
            "n_patients": int(len(y)),
            "n_mutant": int(y.sum()),
            "native_ensemble_auroc": auc,
            "e2b_reported_auroc": float(reported["auroc"]),
            "absolute_error": abs(float(reported["auroc"]) - auc),
            "status": "PASS",
        }
    return bundle


def fit_native_offset(
    H: np.ndarray,
    y: np.ndarray,
    eta_native: np.ndarray,
    lam: float,
    *,
    iters: int = 60,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Fit the exact convex native-logit residual objective in support span."""
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    eta_native = np.asarray(eta_native, dtype=np.float64)
    if H.ndim != 2 or y.shape != (len(H),) or eta_native.shape != (len(H),):
        raise ValueError("H, y and eta_native have incompatible shapes")
    if not np.isfinite(H).all() or not np.isfinite(y).all() or not np.isfinite(eta_native).all():
        raise ValueError("offset fit inputs must be finite")
    if not np.isfinite(lam):
        return np.zeros(H.shape[1], dtype=np.float64), 0.0
    if lam <= 0:
        raise ValueError("finite lambda must be positive")

    n = len(y)
    gram = H @ H.T
    alpha = np.zeros(n, dtype=np.float64)
    delta_b = 0.0
    eye = np.eye(n + 1)

    def objective(av: np.ndarray, bv: float) -> float:
        z = eta_native + gram @ av + bv
        data_loss = np.mean(np.logaddexp(0.0, z) - y * z)
        penalty = lam / 2 * (av @ gram @ av + bv**2)
        return float(data_loss + penalty)

    current = objective(alpha, delta_b)
    for _ in range(iters):
        z = eta_native + gram @ alpha + delta_b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        curvature = p * (1 - p) + 1e-12
        grad_alpha = gram @ ((p - y) / n + lam * alpha)
        grad_b = float(np.mean(p - y) + lam * delta_b)
        h_aa = gram @ ((curvature / n)[:, None] * gram) + lam * gram
        h_ab = gram @ (curvature / n)
        h_bb = float(np.mean(curvature) + lam)
        hessian = np.block(
            [[h_aa, h_ab[:, None]], [h_ab[None, :], np.array([[h_bb]])]]
        )
        gradient = np.concatenate([grad_alpha, [grad_b]])
        try:
            step = np.linalg.solve(hessian + 1e-10 * eye, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian + 1e-10 * eye, gradient, rcond=None)[0]
        scale = 1.0
        directional = float(gradient @ step)
        for _ in range(40):
            candidate_a = alpha - scale * step[:n]
            candidate_b = delta_b - scale * float(step[n])
            candidate = objective(candidate_a, candidate_b)
            if candidate <= current - 1e-4 * scale * directional:
                break
            scale *= 0.5
        else:
            break
        alpha, delta_b, current = candidate_a, candidate_b, candidate
        if scale * np.max(np.abs(step)) < tol:
            break
    return H.T @ alpha, float(delta_b)


def native_offset_fit_diagnostics(
    H: np.ndarray,
    y: np.ndarray,
    eta_native: np.ndarray,
    delta_w: np.ndarray,
    delta_b: float,
    lam: float,
) -> dict[str, Any]:
    """Objective/KKT evidence for one fitted residual adapter."""
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    eta_native = np.asarray(eta_native, dtype=np.float64)
    delta_w = np.asarray(delta_w, dtype=np.float64)
    weight_norm = float(np.linalg.norm(delta_w))
    if not np.isfinite(lam):
        if not np.array_equal(delta_w, np.zeros_like(delta_w)) or delta_b != 0.0:
            raise RuntimeError("lambda=infinity did not return an exact zero residual")
        return {
            "status": "native_exact",
            "delta_weight_l2": 0.0,
            "delta_bias": 0.0,
            "prediction_rule": "eta_native (no residual arithmetic)",
        }
    baseline_objective = float(np.mean(np.logaddexp(0.0, eta_native) - y * eta_native))
    z = eta_native + H @ delta_w + delta_b
    fitted_objective = float(
        np.mean(np.logaddexp(0.0, z) - y * z)
        + lam / 2 * (np.dot(delta_w, delta_w) + delta_b**2)
    )
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    gradient_w = H.T @ (p - y) / len(y) + lam * delta_w
    gradient_b = float(np.mean(p - y) + lam * delta_b)
    kkt_inf = float(max(np.max(np.abs(gradient_w)), abs(gradient_b)))
    objective_decrease = baseline_objective - fitted_objective
    if objective_decrease < -OBJECTIVE_TOL:
        raise RuntimeError(f"Residual fit increased its objective by {-objective_decrease:.3e}")
    if kkt_inf > KKT_TOL:
        raise RuntimeError(f"Residual fit KKT residual is too large: {kkt_inf:.3e}")
    return {
        "status": "finite_optimum",
        "delta_weight_l2": weight_norm,
        "delta_bias": float(delta_b),
        "objective_at_zero_delta": baseline_objective,
        "objective_at_fit": fitted_objective,
        "objective_decrease": objective_decrease,
        "kkt_gradient_inf_norm": kkt_inf,
        "prediction_rule": "eta_native + H @ delta_w + delta_b",
    }


def loo_loss_native_offset(
    H: np.ndarray,
    y: np.ndarray,
    eta_native: np.ndarray,
    lam: float,
) -> float | None:
    """LOO loss using native offsets for both kept and held-out patients."""
    H = np.asarray(H, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    eta_native = np.asarray(eta_native, dtype=np.float64)
    losses: list[float] = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        if len(np.unique(y[keep])) < 2:
            continue
        delta_w, delta_b = fit_native_offset(
            H[keep], y[keep], eta_native[keep], lam
        )
        z = float(eta_native[i] + H[i] @ delta_w + delta_b)
        p = float(np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), EPS, 1 - EPS))
        losses.append(float(-(y[i] * np.log(p) + (1 - y[i]) * np.log(1 - p))))
    return float(np.mean(losses)) if losses else None


def select_lambda_native_offset(
    H: np.ndarray,
    y: np.ndarray,
    eta_native: np.ndarray,
) -> float:
    """Select lambda inside support only; ties retain the earlier stronger shrinkage."""
    best = LAMBDA_GRID[0]
    best_loss = np.inf
    for lam in LAMBDA_GRID:
        loss = loo_loss_native_offset(H, y, eta_native, lam)
        if loss is not None and loss < best_loss:
            best, best_loss = lam, loss
    return best


def _json_lambda(value: float) -> float | str:
    return float(value) if np.isfinite(value) else "infinity"


def run_cohort(
    bundle: InputBundle,
    target: str,
    subcohort: str | None,
    *,
    reps: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    if reps < 2:
        raise ValueError("--reps must be at least 2")
    met_pat, met_emb, met_native = _patient_data(bundle, target, "metastatic")
    pri_pat, pri_emb, pri_native = _patient_data(
        bundle, target, "primary", subcohort=subcohort
    )
    dual = set(met_pat["patient_id"]) & set(pri_pat["patient_id"])
    y_met = met_pat["label"].to_numpy(dtype=float)
    y_pri = pri_pat["label"].to_numpy(dtype=float)
    folds = e2c_stats.stratified_folds(met_pat, PRIMARY_SEED)
    eta0 = np.mean(np.vstack([met_native[seed] for seed in SEEDS]), axis=0)
    out: dict[str, Any] = {
        "target": target,
        "primary_pool": subcohort or target,
        "n_metastatic": int(len(met_pat)),
        "n_metastatic_mut": int(y_met.sum()),
        "n_primary_pool": int(len(pri_pat)),
        "n_primary_mut": int(y_pri.sum()),
        "n_dual_role": len(dual),
        "reps": reps,
        "folds": N_FOLDS,
        "S0": e2c_stats.metrics(y_met, eta0),
        "S0_inference": e2c_stats.procedure_summary(
            y_met,
            eta0,
            n_boot=n_bootstrap,
            random_seed=e2c_stats._stable_seed(BOOTSTRAP_SEED, target, "S0"),  # noqa: SLF001
        ),
        "arms": {},
        "prediction_draws": {"S0": [eta0.tolist()]},
        "support_draws": {},
        "labels": y_met.astype(int).tolist(),
        "fold_of_patient": folds.tolist(),
        "patient_ids": met_pat["patient_id"].astype(str).tolist(),
    }
    for arm in ("S1", "S2"):
        for k in BUDGETS:
            predictions: list[np.ndarray] = []
            all_lambdas: list[float] = []
            all_fit_diagnostics: list[dict[str, Any]] = []
            support_repetitions: list[dict[str, Any]] = []
            for repetition in range(reps):
                eta = np.full(len(y_met), np.nan)
                repetition_audit: dict[str, Any] = {
                    "repetition": repetition,
                    "folds": [],
                }
                for fold in range(N_FOLDS):
                    test = np.flatnonzero(folds == fold)
                    test_patient_ids = met_pat.iloc[test]["patient_id"].astype(str).tolist()
                    if arm == "S2":
                        pool_idx = np.flatnonzero(folds != fold)
                        pool = met_pat.iloc[pool_idx]
                        source_H, source_y, source_native = met_emb, y_met, met_native
                    else:
                        blocked = set(test_patient_ids) & dual
                        pool_idx = np.flatnonzero(
                            ~pri_pat["patient_id"].astype(str).isin(blocked).to_numpy()
                        )
                        pool = pri_pat.iloc[pool_idx]
                        source_H, source_y, source_native = pri_emb, y_pri, pri_native
                    support_seed = e2c_stats._stable_seed(  # noqa: SLF001
                        BOOTSTRAP_SEED,
                        "support",
                        target,
                        arm,
                        k,
                        repetition,
                        fold,
                    )
                    pick = e2c_stats.draw_support(
                        pool.reset_index(drop=True),
                        k,
                        np.random.default_rng(support_seed),
                    )
                    if pick is None:
                        raise RuntimeError(
                            f"{target} {arm} k={k} rep={repetition} fold={fold}: "
                            "support cannot supply exact k/class"
                        )
                    support = pool_idx[pick]
                    support_ids = pool.iloc[pick]["patient_id"].astype(str).tolist()
                    support_labels = source_y[support].astype(int).tolist()
                    if (
                        len(support_ids) != 2 * k
                        or len(set(support_ids)) != 2 * k
                        or support_labels.count(0) != k
                        or support_labels.count(1) != k
                    ):
                        raise RuntimeError("Exact balanced support contract failed")
                    if set(test_patient_ids) & set(support_ids):
                        raise RuntimeError("Support/test patient leakage detected")

                    seed_predictions: list[np.ndarray] = []
                    fold_lambdas: dict[str, float | str] = {}
                    delta_norms: dict[str, float] = {}
                    delta_biases: dict[str, float] = {}
                    solver_diagnostics: dict[str, Any] = {}
                    prediction_sources: dict[str, str] = {}
                    for seed in SEEDS:
                        support_H = source_H[seed][support]
                        support_native = source_native[seed][support]
                        lam = select_lambda_native_offset(
                            support_H, source_y[support], support_native
                        )
                        delta_w, delta_b = fit_native_offset(
                            support_H,
                            source_y[support],
                            support_native,
                            lam,
                        )
                        fit_diagnostic = native_offset_fit_diagnostics(
                            support_H,
                            source_y[support],
                            support_native,
                            delta_w,
                            delta_b,
                            lam,
                        )
                        if np.isfinite(lam):
                            prediction = (
                                met_native[seed][test]
                                + met_emb[seed][test] @ delta_w
                                + delta_b
                            )
                            prediction_sources[str(seed)] = "native_plus_residual"
                        else:
                            # The exact native branch is intentional: no fp64
                            # reconstruction and no arithmetic perturbation.
                            prediction = met_native[seed][test].copy()
                            prediction_sources[str(seed)] = "native_exact"
                        seed_predictions.append(prediction)
                        all_lambdas.append(lam)
                        all_fit_diagnostics.append(fit_diagnostic)
                        fold_lambdas[str(seed)] = _json_lambda(lam)
                        delta_norms[str(seed)] = float(np.linalg.norm(delta_w))
                        delta_biases[str(seed)] = float(delta_b)
                        solver_diagnostics[str(seed)] = fit_diagnostic
                    eta[test] = np.mean(np.vstack(seed_predictions), axis=0)
                    repetition_audit["folds"].append(
                        {
                            "fold": fold,
                            "support_seed": support_seed,
                            "support_patient_ids": support_ids,
                            "support_labels": support_labels,
                            "test_patient_ids": test_patient_ids,
                            "lambda_by_model_seed": fold_lambdas,
                            "delta_weight_l2_by_model_seed": delta_norms,
                            "delta_bias_by_model_seed": delta_biases,
                            "prediction_source_by_model_seed": prediction_sources,
                            "solver_diagnostics_by_model_seed": solver_diagnostics,
                        }
                    )
                if not np.isfinite(eta).all():
                    raise RuntimeError(
                        f"{target} {arm} k={k} repetition={repetition}: incomplete predictions"
                    )
                predictions.append(eta)
                support_repetitions.append(repetition_audit)
            matrix = np.vstack(predictions)
            inference = e2c_stats.procedure_summary(
                y_met,
                matrix,
                n_boot=n_bootstrap,
                random_seed=e2c_stats._stable_seed(  # noqa: SLF001
                    BOOTSTRAP_SEED, target, arm, k, "summary"
                ),
            )
            finite_lambdas = [value for value in all_lambdas if np.isfinite(value)]
            finite_diagnostics = [
                item for item in all_fit_diagnostics if item["status"] == "finite_optimum"
            ]
            declined_diagnostics = [
                item for item in all_fit_diagnostics if item["status"] == "native_exact"
            ]
            key = f"{arm}_k{k}"
            out["arms"][key] = {
                "arm": arm,
                "k_per_class": k,
                "n_support": 2 * k,
                "reps_completed": reps,
                "support_failures": 0,
                "n_lambda_decisions": len(all_lambdas),
                "lambda_frac_declined": float(
                    np.mean(~np.isfinite(np.asarray(all_lambdas)))
                ),
                "lambda_median_finite": float(np.median(finite_lambdas))
                if finite_lambdas
                else None,
                "solver_audit": {
                    "status": "PASS",
                    "finite_fit_count": len(finite_diagnostics),
                    "declined_exact_native_count": len(declined_diagnostics),
                    "max_finite_kkt_gradient_inf_norm": max(
                        (item["kkt_gradient_inf_norm"] for item in finite_diagnostics),
                        default=0.0,
                    ),
                    "min_finite_objective_decrease": min(
                        (item["objective_decrease"] for item in finite_diagnostics),
                        default=None,
                    ),
                    "max_declined_delta_weight_l2": max(
                        (item["delta_weight_l2"] for item in declined_diagnostics),
                        default=0.0,
                    ),
                    "max_declined_abs_delta_bias": max(
                        (abs(item["delta_bias"]) for item in declined_diagnostics),
                        default=0.0,
                    ),
                },
                "performance": inference,
            }
            out["prediction_draws"][key] = matrix.tolist()
            out["support_draws"][key] = support_repetitions
    return out


def _add_contrasts(block: dict[str, Any], *, target: str, n_bootstrap: int) -> None:
    y = np.asarray(block["labels"], dtype=int)
    draws = block["prediction_draws"]
    block["contrasts"] = {}
    for k in BUDGETS:
        s1, s2 = f"S1_k{k}", f"S2_k{k}"
        block["contrasts"][f"S2_minus_S1_k{k}"] = e2c_stats.procedure_contrast(
            y,
            np.asarray(draws[s2]),
            np.asarray(draws[s1]),
            n_boot=n_bootstrap,
            random_seed=e2c_stats._stable_seed(  # noqa: SLF001
                BOOTSTRAP_SEED, target, "S2_minus_S1", k
            ),
            support_resampling="independent",
        )
        for arm in (s1, s2):
            block["contrasts"][f"{arm}_minus_S0"] = e2c_stats.procedure_contrast(
                y,
                np.asarray(draws[arm]),
                np.asarray(draws["S0"]),
                n_boot=n_bootstrap,
                random_seed=e2c_stats._stable_seed(  # noqa: SLF001
                    BOOTSTRAP_SEED, target, f"{arm}_minus_S0"
                ),
                support_resampling="independent",
            )


def build_report(
    bundle: InputBundle,
    *,
    cap: int,
    reps: int,
    n_bootstrap: int,
    code: dict[str, Any],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "component": "e2c_native_logit_residual_offset",
        "input_lineage": bundle.lineage,
        "input_lineage_root": str(bundle.root),
        "cap": cap,
        "reps": reps,
        "n_bootstrap": n_bootstrap,
        "code": code,
        "design": {
            "model": "eta_native + H @ delta_w + delta_b",
            "adapted": "512 residual weights and one residual intercept per frozen model seed",
            "frozen": "encoder, ABMIL attention, source head, embeddings, and native logits",
            "penalty": "lambda/2 * (||delta_w||^2 + delta_b^2)",
            "lambda_selection": "leave-one-out inside support only; infinity is first/tie-preferred",
            "lambda_infinity": "delta_w=0 and delta_b=0; predictions are bit-exact native S0",
            "budgets_per_class": list(BUDGETS),
            "folds": N_FOLDS,
            "headline": "mean of per-repetition metrics; predictions never averaged across repetitions",
            "uncertainty": "two-way patient x support-procedure bootstrap",
            "n_bootstrap": n_bootstrap,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "fold_seed": PRIMARY_SEED,
            "support_seed_scheme": (
                "SHA256(bootstrap_seed,support,target,arm,k,repetition,fold)"
            ),
            "recovery_reason": (
                "The deployed frozen forward used BF16. A finite adapted fp64 head plus a "
                "separate native S0 branch was discontinuous at lambda=infinity. Residual "
                "adaptation anchors every finite lambda to the same stored native predictor."
            ),
            "solver_acceptance": {
                "max_kkt_gradient_inf_norm": KKT_TOL,
                "minimum_objective_decrease": -OBJECTIVE_TOL,
            },
        },
        "cohorts": {},
    }
    for target, subcohort in COHORTS:
        print(f"\n=== {target}: native-offset E2c (S1 pool={subcohort or target}-P)")
        block = run_cohort(
            bundle,
            target,
            subcohort,
            reps=reps,
            n_bootstrap=n_bootstrap,
        )
        _add_contrasts(block, target=target, n_bootstrap=n_bootstrap)
        report["cohorts"][target] = block
        print(f"    S0 AUROC {block['S0']['auroc']:.6f}")
        for name, arm in block["arms"].items():
            print(
                f"    {name}: expected AUROC {arm['performance']['metrics']['auroc']:.6f}; "
                f"declined {arm['lambda_frac_declined']:.1%}"
            )
    return report


def audit_report(report: dict[str, Any], bundle: InputBundle) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    if report.get("input_lineage") != bundle.lineage:
        raise RuntimeError("Output report input lineage mismatch")
    if int(report.get("n_bootstrap", -1)) < 1:
        raise RuntimeError("Output report does not record a positive bootstrap count")
    checks["input_lineage"] = "PASS"
    checks["n_bootstrap"] = int(report["n_bootstrap"])
    for target, _subcohort in COHORTS:
        block = report["cohorts"][target]
        patients, _emb, native = _patient_data(bundle, target, "metastatic")
        expected_s0 = np.mean(np.vstack([native[seed] for seed in SEEDS]), axis=0)
        actual_s0 = np.asarray(block["prediction_draws"]["S0"], dtype=float)
        if actual_s0.shape != (1, len(patients)) or not np.array_equal(
            actual_s0[0], expected_s0
        ):
            raise RuntimeError(f"{target}: S0 is not bit-exact native ensemble")
        y = np.asarray(block["labels"], dtype=int)
        if not np.array_equal(y, patients["label"].to_numpy(dtype=int)):
            raise RuntimeError(f"{target}: patient labels/order changed")
        e2b_reproduction = bundle.diagnostics["e2b_native_baseline_reproduction"][target]
        if e2b_reproduction.get("status") != "PASS" or e2b_reproduction.get(
            "absolute_error"
        ) != 0.0:
            raise RuntimeError(f"{target}: S0 does not reproduce E2b exactly")
        if block["S0"]["auroc"] != e2b_reproduction["e2b_reported_auroc"]:
            raise RuntimeError(f"{target}: reported S0 AUROC differs from E2b")
        target_checks: dict[str, Any] = {
            "S0_native_exact": "PASS",
            "S0_equals_E2b": "PASS",
            "arms": {},
        }
        fold_assignment = np.asarray(block["fold_of_patient"], dtype=int)
        patient_ids = np.asarray(block["patient_ids"], dtype=str)
        for arm in ("S1", "S2"):
            for k in BUDGETS:
                key = f"{arm}_k{k}"
                matrix = np.asarray(block["prediction_draws"][key], dtype=float)
                if matrix.shape != (report["reps"], len(y)) or not np.isfinite(matrix).all():
                    raise RuntimeError(f"{target}/{key}: incomplete prediction matrix")
                supports = block["support_draws"][key]
                if len(supports) != report["reps"]:
                    raise RuntimeError(f"{target}/{key}: incomplete support repetitions")
                lambda_count = 0
                finite_diagnostics: list[dict[str, Any]] = []
                declined_diagnostics: list[dict[str, Any]] = []
                for repetition, draw in enumerate(supports):
                    if draw.get("repetition") != repetition or len(draw.get("folds", [])) != N_FOLDS:
                        raise RuntimeError(f"{target}/{key}: malformed repetition audit")
                    seen_tests: list[str] = []
                    for fold in draw["folds"]:
                        support_ids = list(map(str, fold["support_patient_ids"]))
                        support_labels = list(map(int, fold["support_labels"]))
                        test_ids = list(map(str, fold["test_patient_ids"]))
                        if (
                            len(support_ids) != len(set(support_ids))
                            or len(support_ids) != 2 * k
                            or support_labels.count(0) != k
                            or support_labels.count(1) != k
                        ):
                            raise RuntimeError(f"{target}/{key}: support budget audit failed")
                        if set(support_ids) & set(test_ids):
                            raise RuntimeError(f"{target}/{key}: support/test leakage")
                        expected_test = set(patient_ids[fold_assignment == int(fold["fold"])])
                        if set(test_ids) != expected_test:
                            raise RuntimeError(f"{target}/{key}: test fold identities changed")
                        seen_tests.extend(test_ids)
                        lambdas = fold["lambda_by_model_seed"]
                        if set(lambdas) != set(map(str, SEEDS)):
                            raise RuntimeError(f"{target}/{key}: incomplete lambdas")
                        solver_by_seed = fold.get("solver_diagnostics_by_model_seed", {})
                        source_by_seed = fold.get("prediction_source_by_model_seed", {})
                        if set(solver_by_seed) != set(lambdas) or set(source_by_seed) != set(
                            lambdas
                        ):
                            raise RuntimeError(f"{target}/{key}: incomplete solver evidence")
                        for seed, value in lambdas.items():
                            numeric = np.inf if value == "infinity" else float(value)
                            if not any(
                                (not np.isfinite(grid) and not np.isfinite(numeric))
                                or numeric == grid
                                for grid in LAMBDA_GRID
                            ):
                                raise RuntimeError(f"{target}/{key}: lambda outside grid")
                            diagnostic = solver_by_seed[seed]
                            delta_norm = float(
                                fold["delta_weight_l2_by_model_seed"][seed]
                            )
                            delta_bias = float(fold["delta_bias_by_model_seed"][seed])
                            if not np.isfinite(numeric):
                                if (
                                    diagnostic.get("status") != "native_exact"
                                    or source_by_seed[seed] != "native_exact"
                                    or delta_norm != 0.0
                                    or delta_bias != 0.0
                                    or diagnostic.get("delta_weight_l2") != 0.0
                                    or diagnostic.get("delta_bias") != 0.0
                                ):
                                    raise RuntimeError(
                                        f"{target}/{key}/seed{seed}: infinity continuity failed"
                                    )
                                declined_diagnostics.append(diagnostic)
                            else:
                                if (
                                    diagnostic.get("status") != "finite_optimum"
                                    or source_by_seed[seed] != "native_plus_residual"
                                    or float(diagnostic.get("kkt_gradient_inf_norm", np.inf))
                                    > KKT_TOL
                                    or float(diagnostic.get("objective_decrease", -np.inf))
                                    < -OBJECTIVE_TOL
                                ):
                                    raise RuntimeError(
                                        f"{target}/{key}/seed{seed}: finite solver audit failed"
                                    )
                                finite_diagnostics.append(diagnostic)
                        lambda_count += len(lambdas)
                    if sorted(seen_tests) != sorted(patient_ids.tolist()):
                        raise RuntimeError(f"{target}/{key}: patients not tested exactly once")
                expected_lambda_count = report["reps"] * N_FOLDS * len(SEEDS)
                if lambda_count != expected_lambda_count:
                    raise RuntimeError(f"{target}/{key}: lambda count mismatch")
                solver_audit = block["arms"][key].get("solver_audit", {})
                expected_solver_audit = {
                    "status": "PASS",
                    "finite_fit_count": len(finite_diagnostics),
                    "declined_exact_native_count": len(declined_diagnostics),
                    "max_finite_kkt_gradient_inf_norm": max(
                        (item["kkt_gradient_inf_norm"] for item in finite_diagnostics),
                        default=0.0,
                    ),
                    "min_finite_objective_decrease": min(
                        (item["objective_decrease"] for item in finite_diagnostics),
                        default=None,
                    ),
                    "max_declined_delta_weight_l2": max(
                        (item["delta_weight_l2"] for item in declined_diagnostics),
                        default=0.0,
                    ),
                    "max_declined_abs_delta_bias": max(
                        (abs(item["delta_bias"]) for item in declined_diagnostics),
                        default=0.0,
                    ),
                }
                if solver_audit != expected_solver_audit:
                    raise RuntimeError(f"{target}/{key}: solver aggregate mismatch")
                recomputed = np.mean(
                    [e2c_stats.metrics(y, row)["auroc"] for row in matrix]
                )
                stored = block["arms"][key]["performance"]["metrics"]["auroc"]
                if abs(recomputed - stored) > 1e-15:
                    raise RuntimeError(f"{target}/{key}: stored expected AUROC mismatch")
                target_checks["arms"][key] = {
                    "predictions_complete": "PASS",
                    "supports_exact_and_leak_free": "PASS",
                    "lambda_decisions_complete": lambda_count,
                    "lambda_infinity_continuity": "PASS",
                    "finite_solver_kkt_and_objective": "PASS",
                    "solver_audit": expected_solver_audit,
                    "expected_auroc_recomputed": recomputed,
                }
        checks[target] = target_checks
    return {
        "schema_version": 1,
        "audited_utc": _utc_now(),
        "status": "PASS",
        "checks": checks,
    }


def _result_path(output_root: Path, cap: int) -> Path:
    return output_root / "analysis" / f"e2c_native_logit_offset_cap{cap}.json"


def cmd_verify_inputs(args: argparse.Namespace) -> None:
    bundle = preflight_inputs(Path(args.input_lineage_root), args.cap)
    summary = {
        "status": "PASS",
        "input_lineage": bundle.lineage,
        "input_lineage_root": str(bundle.root),
        "cap": args.cap,
        "diagnostics": bundle.diagnostics,
        "imported_file_count": 1
        + sum(
            2 + 2 * len(SEEDS) + 2 + len(SEEDS)
            for _target, _subcohort in COHORTS
        ),
        "canonical_e2c_result_consumed": False,
    }
    print(json.dumps(summary, indent=2, allow_nan=False))


def cmd_run(args: argparse.Namespace) -> None:
    input_root = _require_absolute(
        Path(args.input_lineage_root), label="--input-lineage-root", must_exist=True
    )
    raw_output_root = Path(args.output_root)
    # The overwrite/overlap gates precede even input preflight and computation.
    if raw_output_root.exists() or raw_output_root.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output root: {raw_output_root}")
    output_root = _require_absolute(
        raw_output_root, label="--output-root", must_exist=False
    )
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        raise ValueError("Input and output roots must not overlap")
    if not output_root.parent.is_dir():
        raise FileNotFoundError(f"Output parent must already exist: {output_root.parent}")
    if args.reps < 2:
        raise ValueError("--reps must be at least 2")
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")

    bundle = preflight_inputs(input_root, args.cap)
    source_payloads = _current_source_payloads()
    output_root.mkdir(mode=0o755, exist_ok=False)
    snapshot_identities: dict[str, Any] = {}
    for name, payload in source_payloads.items():
        destination = output_root / "source_snapshot" / name
        _write_bytes_once_atomic(destination, payload)
        snapshot_identities[name] = _artifact_identity(destination)
    start = {
        "schema_version": 1,
        "status": "started",
        "created_utc": _utc_now(),
        "component": "e2c_native_logit_residual_offset",
        "output_root": str(output_root),
        "input_lineage": bundle.lineage,
        "input_lineage_root": str(bundle.root),
        "cap": args.cap,
        "reps": args.reps,
        "n_bootstrap": args.n_bootstrap,
        "source_snapshot": snapshot_identities,
    }
    _write_json_once_atomic(output_root / "lineage_start.json", start)
    import_receipt = bundle.receipt_payload()
    _write_json_once_atomic(output_root / "receipts" / "upstream_imports.json", import_receipt)

    report = build_report(
        bundle,
        cap=args.cap,
        reps=args.reps,
        n_bootstrap=args.n_bootstrap,
        code=snapshot_identities,
    )
    # Fail before publishing the result if either this component or its imported
    # statistical-helper source was edited during the campaign.
    _assert_source_bytes_unchanged(source_payloads)
    result = _result_path(output_root, args.cap)
    _write_json_once_atomic(result, report)

    # Validate inputs a second time after the long calculation.  Receipt times
    # differ, so compare the immutable identities and deterministic diagnostics.
    post = preflight_inputs(input_root, args.cap)
    if post.imports != bundle.imports or post.diagnostics != bundle.diagnostics:
        raise RuntimeError("An upstream input changed during the E2c recovery run")
    _assert_source_bytes_unchanged(source_payloads)
    audit = audit_report(report, bundle)
    audit["result"] = _artifact_identity(result)
    audit["upstream_unchanged_during_run"] = "PASS"
    _write_json_once_atomic(output_root / "receipts" / "analysis_audit.json", audit)
    complete = {
        "schema_version": 1,
        "status": "completed",
        "completed_utc": _utc_now(),
        "component": "e2c_native_logit_residual_offset",
        "input_lineage": bundle.lineage,
        "artifacts": {
            "lineage_start": _artifact_identity(output_root / "lineage_start.json"),
            "upstream_imports": _artifact_identity(
                output_root / "receipts" / "upstream_imports.json"
            ),
            "analysis": _artifact_identity(result),
            "analysis_audit": _artifact_identity(
                output_root / "receipts" / "analysis_audit.json"
            ),
            "source_snapshot": snapshot_identities,
        },
    }
    _write_json_once_atomic(output_root / "lineage_complete.json", complete)
    print(f"\nPASS — wrote immutable downstream lineage: {output_root}")
    print_report(report)


def cmd_verify_output(args: argparse.Namespace) -> None:
    input_root = _require_absolute(
        Path(args.input_lineage_root), label="--input-lineage-root", must_exist=True
    )
    output_root = _require_absolute(Path(args.output_root), label="--output-root", must_exist=True)
    bundle = preflight_inputs(input_root, args.cap)
    complete = _read_json(output_root / "lineage_complete.json")
    if complete.get("status") != "completed" or complete.get("input_lineage") != bundle.lineage:
        raise RuntimeError("Output completion receipt does not match the input lineage")
    expected_output_paths = {
        "lineage_start": output_root / "lineage_start.json",
        "upstream_imports": output_root / "receipts" / "upstream_imports.json",
        "analysis": _result_path(output_root, args.cap),
        "analysis_audit": output_root / "receipts" / "analysis_audit.json",
    }
    if set(complete.get("artifacts", {})) != {*expected_output_paths, "source_snapshot"}:
        raise RuntimeError("Completion receipt has an unexpected artifact inventory")
    for label, recorded in complete.get("artifacts", {}).items():
        if label == "source_snapshot":
            for source_name, source_record in recorded.items():
                _validate_file_identity(
                    source_record,
                    label=f"output source {source_name}",
                    expected_path=output_root / "source_snapshot" / source_name,
                )
        else:
            _validate_file_identity(
                recorded,
                label=f"output {label}",
                expected_path=expected_output_paths[label],
            )
    import_receipt = _read_json(output_root / "receipts" / "upstream_imports.json")
    if import_receipt.get("imports") != bundle.imports:
        raise RuntimeError("Current upstream identities differ from the import receipt")
    result = _read_json(_result_path(output_root, args.cap))
    fresh_audit = audit_report(result, bundle)
    recorded_audit = _read_json(output_root / "receipts" / "analysis_audit.json")
    _validate_file_identity(
        recorded_audit.get("result", {}),
        label="recorded audit result",
        expected_path=_result_path(output_root, args.cap),
    )
    if recorded_audit.get("status") != "PASS" or fresh_audit["checks"] != recorded_audit.get(
        "checks"
    ):
        raise RuntimeError("Fresh output audit differs from the recorded audit")
    print(
        json.dumps(
            {
                "status": "PASS",
                "input_lineage": bundle.lineage,
                "output_root": str(output_root),
                "result": _artifact_identity(_result_path(output_root, args.cap)),
            },
            indent=2,
        )
    )


def print_report(report: dict[str, Any]) -> None:
    print(
        f"\nE2c native-logit residual offset · cap {report['cap']} · "
        f"{report['reps']} repetitions"
    )
    for target, block in report["cohorts"].items():
        print(f"\n{target}: S0 AUROC {block['S0']['auroc']:.4f}")
        for arm in ("S1", "S2"):
            for k in BUDGETS:
                key = f"{arm}_k{k}"
                item = block["arms"][key]
                perf = item["performance"]
                contrast = block["contrasts"][f"{key}_minus_S0"]
                print(
                    f"  {key}: expected AUROC {perf['metrics']['auroc']:.4f} "
                    f"[{perf['expected_auroc_ci'][0]:.4f}, "
                    f"{perf['expected_auroc_ci'][1]:.4f}], "
                    f"delta vs S0 {contrast['delta']:+.4f} "
                    f"[{contrast['ci'][0]:+.4f}, {contrast['ci'][1]:+.4f}], "
                    f"declined {item['lambda_frac_declined']:.1%}"
                )
        print("  Primary endpoint (S2 - S1):")
        for k in BUDGETS:
            contrast = block["contrasts"][f"S2_minus_S1_k{k}"]
            print(
                f"    k={k}/class: {contrast['delta']:+.4f} "
                f"[{contrast['ci'][0]:+.4f}, {contrast['ci'][1]:+.4f}]"
            )


def cmd_report(args: argparse.Namespace) -> None:
    output_root = _require_absolute(Path(args.output_root), label="--output-root", must_exist=True)
    print_report(_read_json(_result_path(output_root, args.cap)))


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-lineage-root", required=True)
    parser.add_argument("--cap", type=int, required=True, choices=(8192,))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("verify-inputs", help="read-only recursive input audit")
    _add_input(command)
    command.set_defaults(func=cmd_verify_inputs)
    command = commands.add_parser("run", help="create a new immutable downstream lineage")
    _add_input(command)
    command.add_argument("--output-root", required=True)
    command.add_argument("--reps", type=int, default=DEFAULT_REPS)
    command.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    command.set_defaults(func=cmd_run)
    command = commands.add_parser("verify-output", help="read-only audit of a completed recovery")
    _add_input(command)
    command.add_argument("--output-root", required=True)
    command.set_defaults(func=cmd_verify_output)
    command = commands.add_parser("report", help="print a completed recovery result")
    command.add_argument("--output-root", required=True)
    command.add_argument("--cap", type=int, required=True, choices=(8192,))
    command.set_defaults(func=cmd_report)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
