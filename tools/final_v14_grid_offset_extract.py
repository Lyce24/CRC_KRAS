"""Checkpointed source-only FINAL-v14 diagonal grid-offset extraction.

Run the live prepare/extract stages in persistent tmux with the base checkout
extraction runtime. Missing canonical storage is an operational block and
never licenses drawing a replacement roster or using older mask copies.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import importlib.metadata
import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from tools import final_v14_grid_offset as grid  # noqa: E402
from tools import final_v14_module2 as common  # noqa: E402

BASE_PYTHON = Path("/home/yc_liu/projects/OceanPath/.venv/bin/python")
DEFAULT_OUTPUT = common.RUN / "grid_offset/locked_extraction"
READY = Path("/mnt/d/YC.Liu/manifests/colon/colon_ready_inventory.csv")
READY_SHA = "cf95d757cabd677ac03c612f59e7be1b3724875bd233acd2988735646fea79b7"
WSI_ROOT = Path("/mnt/d/YC.Liu/slides/colon")
PROFILE = "20x_256px_0px_overlap_mpp0.5"
SOURCE_ROSTER_SHA = "a2e82b2172edc7643ccc3b643e4d0f597839ce3efe311fb421836fbfd125a2e4"
PACK_META_SHA = "44f1f0c80f4b8740c5fc82caafa96a8ead372c561fe53cda896d83972ff3c80b"
HEST_CHECKPOINT = Path("/home/yc_liu/.cache/trident/deeplabv3_seg_v4.ckpt")
HEST_SHA = "4ddf8be82384544ae9ef9af7ad0cc1e8190c781b945eea9826c778f93d08fd5a"
HISTORICAL_RECEIPT_INVENTORY = common.RUN / "grid_offset/historical_receipt_schema_20260905/receipt_inventory.json"
KNOWN_RECEIPT_IMPLEMENTATIONS = {
    "11169bd9db126013246a80e74535020ee4f1cbe40eb8dd460ac2ba24da93f96d",
    "9553c18c4c6e46bf9326f10b033462e5fb7c6023e42a043981164699b54931fd",
    "a9063a40c31940dc4fa511158ef0f5bc1f2cdbf30a53ce22003b41dd47eb4899",
    "c186b18b06c755fead1c3f6ea9a7e8a9a2ce167e6b2a8c04525599d79fd482de",
}


def atomic_write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise common.ContractError(f"Refusing to replace sealed/prepared artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise common.ContractError(f"Concurrent artifact publication drift: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_write_once(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def seal_json(path: Path) -> None:
    write_json(Path(str(path) + ".seal.json"), common.identity(path))


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [normalize_json(v) for v in value]
    if isinstance(value, np.generic):
        return normalize_json(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def runtime(*, require_base: bool = True) -> dict:
    if require_base and Path(sys.prefix).resolve() != BASE_PYTHON.parent.parent.resolve():
        raise grid.OperationalBlock(f"Use the verified extraction runtime: {BASE_PYTHON}")
    packages = ("numpy", "pandas", "torch", "torchvision", "trident", "timm", "geopandas", "shapely", "h5py", "openslide-python")
    try:
        versions = {p: importlib.metadata.version(p) for p in packages}
    except importlib.metadata.PackageNotFoundError as exc:
        raise grid.OperationalBlock(f"Extraction dependency unavailable: {exc}") from exc
    return {"python_executable": sys.executable, "python_version": sys.version, "packages": versions}


def bound_paths(row: dict, root: Path) -> dict:
    slide = str(row["slide_id"])
    if Path(slide).name != slide or any(c in slide for c in ("/", "\\")):
        raise common.ContractError("Unsafe slide identifier")
    return {"mask": root / "contours_geojson" / f"{slide}.geojson",
            "coordinates": root / PROFILE / "patches" / f"{slide}_patches.h5",
            **{f"receipt_{stage}": root / "_stream_receipts" / slide / f"{stage}.json"
               for stage in ("seg", "uni_v1_coords", "uni_v1_feat")}}


def raw_identity(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path.resolve()), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def verify_raw(item: dict) -> Path:
    path = Path(item["path"])
    if raw_identity(path) != item:
        raise common.ContractError(f"Source WSI identity changed: {path}")
    return path


def verify_receipts(paths: dict, row: dict, raw: dict) -> list[dict]:
    result = []
    for stage in ("seg", "uni_v1_coords", "uni_v1_feat"):
        path = paths[f"receipt_{stage}"]
        data = json.loads(path.read_text())
        source = data.get("source", {})
        if (data.get("stage") != stage or source.get("output_id") != row["slide_id"]
                or int(source.get("size", -1)) != raw["size_bytes"]
                or int(source.get("mtime_ns", -1)) != raw["mtime_ns"]
                or not np.isclose(float(source.get("mpp", np.nan)), float(row["mpp"]), rtol=0, atol=1e-6)):
            raise common.ContractError(f"Canonical stream receipt/source mismatch: {path}")
        if data.get("checkpoint_hashes", {}).get("uni_v1") != grid.UNI_SHA256:
            raise common.ContractError(f"Canonical UNI checkpoint mismatch: {path}")
        if (data.get("checkpoint_hashes", {}).get("hest") != HEST_SHA
                or data.get("implementation_hash") not in KNOWN_RECEIPT_IMPLEMENTATIONS):
            raise grid.OperationalBlock(f"Canonical receipt provenance schema needs inspection before eligibility selection: {path}")
        if stage == "seg":
            policy = data.get("segmentation_policy", {})
            if policy.get("segmenter", "hest") != "hest":
                raise common.ContractError(f"Canonical segmenter contradicts the pinned HEST lineage: {path}")
        result.append(common.identity(path))
    return result


def eligibility_failure(row: dict, error: Exception, canonical_root: Path) -> dict:
    """Individual missing/invalid inputs fail eligibility; outages never do."""
    grid.require_canonical_root(canonical_root)
    if not WSI_ROOT.is_dir() or not READY.is_file():
        raise grid.OperationalBlock("Raw WSI root or frozen ready inventory became unavailable; no eligibility verdict") from error
    meta = canonical_root / PROFILE / "packed_uni_v1/meta.json"
    if not meta.is_file() or common.digest(meta) != PACK_META_SHA:
        raise grid.OperationalBlock("Canonical archive became unavailable during eligibility audit") from error
    if isinstance(error, OSError) and error.errno in {errno.EIO, errno.EACCES, errno.EPERM, errno.ESTALE, errno.ENODEV, errno.ENXIO}:
        raise grid.OperationalBlock(f"Systemic source-input I/O failure; no roster selected: {error}") from error
    return {"slide_id": row["slide_id"], "patient_id": row["patient_id"], "subcohort": row["subcohort"],
            "eligible": False, "reason": f"{type(error).__name__}: {error}"}


def audit_source_slide(row: dict, canonical_root: Path) -> dict:
    """Read-only original-input preflight; never inspect offset features or outcomes."""
    import geopandas as gpd
    import openslide

    from oceanpath.extraction.mpp_sampling import validate_exact_mpp_coordinate_file
    paths = bound_paths(row, canonical_root)
    relative = Path(row["wsi"])
    if relative.is_absolute() or ".." in relative.parts:
        raise common.ContractError("WSI inventory path escapes source root")
    wsi_path = (WSI_ROOT / relative).resolve(strict=True)
    if not wsi_path.is_relative_to(WSI_ROOT.resolve()):
        raise common.ContractError("WSI symlink escapes source root")
    original = raw_identity(wsi_path)
    if original["size_bytes"] != int(row["size_bytes"]) or original["mtime_ns"] != int(row["mtime_ns"]):
        raise common.ContractError("WSI changed since the frozen ready inventory")
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0 or Path(str(path) + ".lock").exists():
            raise common.ContractError(f"Canonical source input unavailable/locked: {path}")
    attributes = normalize_json(validate_exact_mpp_coordinate_file(paths["coordinates"], target_mpp=.5,
                            source_mpp=float(row["mpp"]), patch_size=256, overlap=0, min_tissue_proportion=.5))
    if attributes.get("coordinate_units") != "level0_pixels" or int(attributes.get("overlap_level0", -1)) != 0:
        raise common.ContractError("Canonical coordinates do not declare level0 nonoverlap units")
    with h5py.File(paths["coordinates"], "r") as f:
        coords = np.asarray(f["coords"], dtype=np.int64)
    stride = int(attributes["patch_size_level0"])
    if not len(coords) or (coords < 0).any() or (coords % stride).any() or len(np.unique(coords, axis=0)) != len(coords):
        raise common.ContractError("Canonical coordinates are not a nonempty unique origin-zero grid")
    if len(coords) != int(row["patch_count"]):
        raise common.ContractError("Canonical/source patch count drift")
    try:
        with openslide.OpenSlide(str(wsi_path)) as slide:
            if slide.dimensions != (int(attributes["level0_width"]), int(attributes["level0_height"])):
                raise common.ContractError("WSI dimensions differ from recorded exact-MPP geometry")
    except openslide.OpenSlideError as exc:
        raise common.ContractError(f"Original WSI failed its header preflight: {exc}") from exc
    mask = gpd.read_file(paths["mask"])
    if mask.empty or mask.geometry.is_empty.all():
        raise common.ContractError("Canonical HEST polygons are empty")
    receipts = verify_receipts(paths, row, original)
    verify_raw(original)
    return {"slide_id": row["slide_id"], "patient_id": row["patient_id"], "subcohort": row["subcohort"],
            "eligible": True, "mpp": float(row["mpp"]), "wsi": original,
            "mask": common.identity(paths["mask"]), "coordinates": common.identity(paths["coordinates"]),
            "canonical_attributes": attributes, "stream_receipts": receipts}


def implementation_files() -> list[dict]:
    import trident
    files = [Path(__file__), Path(grid.__file__), Path(common.__file__), REPO / "tools/final_v14_modeling_v2.py",
             REPO / "tools/final_v14_module3.py", REPO / "uv.lock", REPO / "pyproject.toml",
             REPO / "src/oceanpath/__init__.py", REPO / "src/oceanpath/aim1/__init__.py",
             REPO / "src/oceanpath/aim1/v14_concepts.py"]
    files.extend(p for p in (BASE_PYTHON.parent.parent.parent / "uv.lock", BASE_PYTHON.parent.parent.parent / "pyproject.toml") if p.is_file())
    files.extend(sorted((REPO / "src/oceanpath/extraction").glob("*.py")))
    files.extend(sorted(Path(trident.__file__).parent.rglob("*.py")))
    return [common.identity(p) for p in files]


def prepare(pre_reader: Path, canonical_root: Path, uni_checkpoint: Path, output: Path) -> dict:
    """Audit all source patients, then seal the seeded 100-patient/all-slide roster."""
    contract_path = output / "extraction_contract.json"
    if contract_path.exists():
        return verify_contract(output, recover_seal=True)
    grid.require_canonical_root(canonical_root)
    if not WSI_ROOT.is_dir() or not READY.is_file():
        raise grid.OperationalBlock("Restore the raw WSI root and frozen ready inventory before eligibility auditing")
    meta = canonical_root / PROFILE / "packed_uni_v1/meta.json"
    if not meta.is_file() or common.digest(meta) != PACK_META_SHA:
        raise grid.OperationalBlock("Canonical archive pack metadata is absent or unauthenticated")
    environment = runtime()
    if common.digest(uni_checkpoint) != grid.UNI_SHA256:
        raise common.ContractError("Frozen local UNI checkpoint digest mismatch")
    if common.digest(HEST_CHECKPOINT) != HEST_SHA:
        raise common.ContractError("Local HEST checkpoint does not authenticate the recorded canonical mask lineage")
    common.read_sealed(HISTORICAL_RECEIPT_INVENTORY)
    if common.digest(READY) != READY_SHA:
        raise common.ContractError("Frozen ready inventory digest mismatch")
    source_path = pre_reader / "inputs/source_roster_label_blind.csv"
    if common.digest(source_path) != SOURCE_ROSTER_SHA:
        raise common.ContractError("Frozen source label-blind roster digest mismatch")
    prepare_receipt = json.loads((pre_reader / "receipts/prepare.json").read_text())
    common._verify_bound(prepare_receipt, source_path)
    source = pd.read_csv(source_path, dtype={"patient_id": str, "slide_id": str, "subcohort": str})
    if len(source) != 1389 or source.patient_id.nunique() != 1239 or source.slide_id.duplicated().any():
        raise common.ContractError("Source-only grid roster census drift")
    if source.groupby("patient_id").subcohort.nunique().max() != 1 or set(source.subcohort) != set(grid.SUBCOHORTS):
        raise common.ContractError("Source patient/subcohort relationship drift")
    inventory = pd.read_csv(READY, usecols=["output_id", "wsi", "mpp", "status", "size_bytes", "mtime_ns"])
    inventory = inventory[inventory.output_id.isin(source.slide_id)]
    if inventory.output_id.duplicated().any() or len(inventory) != len(source) or not inventory.status.str.lower().eq("ready").all():
        raise common.ContractError("Ready inventory does not resolve all listed source slides")
    source = source.merge(inventory.rename(columns={"output_id": "slide_id"}), on="slide_id", validate="one_to_one")
    audit_path = output / "all_source_eligibility_audit.json"
    if audit_path.exists():
        saved = (common.read_sealed(audit_path) if Path(str(audit_path) + ".seal.json").exists()
                 else json.loads(audit_path.read_text()))
        common.verify_identity(saved["source_roster"])
        common.verify_identity(saved["ready_inventory"])
        audit = saved["slides"]
        for row in audit:
            if row["eligible"]:
                verify_slide_inputs(row)
    else:
        audit = []
        for i, row in enumerate(source.sort_values(["subcohort", "patient_id", "slide_id"]).to_dict("records")):
            try:
                result = audit_source_slide(row, canonical_root)
            except (OSError, ValueError, common.ContractError) as exc:
                result = eligibility_failure(row, exc, canonical_root)
            audit.append(result)
            if (i + 1) % 100 == 0:
                print(json.dumps({"stage": "ORIGINAL_INPUT_ELIGIBILITY", "slides_checked": i + 1}), flush=True)
        grid.require_canonical_root(canonical_root)
        if common.digest(meta) != PACK_META_SHA:
            raise grid.OperationalBlock("Canonical archive changed during complete eligibility audit")
        write_json(audit_path, {"status": "ALL_SOURCE_INPUTS_PREFLIGHTED", "created_utc": common.now(),
                                      "source_roster": common.identity(source_path), "ready_inventory": common.identity(READY),
                                      "slides": audit, "outcome_columns_read": 0})
        seal_json(audit_path)
    if len(audit) != len(source) or {r["slide_id"] for r in audit} != set(source.slide_id):
        raise common.ContractError("Eligibility audit does not cover every original listed slide")
    source_relation = source.set_index("slide_id")
    if any((r["patient_id"], r["subcohort"]) != tuple(source_relation.loc[r["slide_id"], ["patient_id", "subcohort"]]) for r in audit):
        raise common.ContractError("Eligibility audit changed a source slide/patient/subcohort relationship")
    seal_json(audit_path)
    eligibility = pd.DataFrame([{k: r[k] for k in ("patient_id", "subcohort", "eligible")} for r in audit])
    eligibility = eligibility.groupby(["patient_id", "subcohort"], sort=True).eligible.all().reset_index()
    try:
        selected = grid.select_source_patients(eligibility, canonical_root_verified=True)
    except grid.InsufficientEligiblePatients as exc:
        result = {"status": "GRID_OFFSET_NOT_EVALUABLE", "reason": str(exc), "eligibility_audit": common.identity(audit_path),
                  "eligible_counts": eligibility[eligibility.eligible].groupby("subcohort").size().to_dict(),
                  "patient_roster_drawn": False, "slide_extraction_run": False}
        write_json(output / "results.json", result)
        seal_json(output / "results.json")
        return result
    ids = set(selected.patient_id)
    roster = [row for row in audit if row["patient_id"] in ids]
    if any(not r["eligible"] for r in roster) or {r["slide_id"] for r in roster} != set(source[source.patient_id.isin(ids)].slide_id):
        raise common.ContractError("Selected roster dropped or substituted a listed slide")
    roster_path = output / "selected_patient_slide_roster.json"
    write_json(roster_path, {"status": "GRID_OFFSET_100_PATIENT_ROSTER_SEALED", "seed": grid.SEED,
                                   "patients": selected.to_dict("records"), "slides": roster,
                                   "eligibility_audit": common.identity(audit_path), "substitutions_allowed": False})
    seal_json(roster_path)
    vocabulary = pre_reader / "vocabularies/reference/variants/k32_seed20260819/vocabulary.npz"
    profiles = pre_reader / "profiles/reference/patient_profiles.parquet"
    quantiles = pre_reader / "profiles/reference_distance_quantiles.csv"
    common._verify_bound(json.loads((pre_reader / "receipts/vocabularies.json").read_text()), vocabulary)
    for path in (profiles, quantiles):
        common._verify_bound(json.loads((pre_reader / "receipts/profiles.json").read_text()), path)
    contract = {"status": "GRID_OFFSET_EXTRACTION_CONTRACT_SEALED", "created_utc": common.now(),
                "roster": common.identity(roster_path), "uni_checkpoint": common.identity(uni_checkpoint),
                "canonical_mask_hest_checkpoint": common.identity(HEST_CHECKPOINT),
                "historical_receipt_schema_inventory": common.identity(HISTORICAL_RECEIPT_INVENTORY),
                "historical_mask_provenance": "Canonical exact mask bytes and stream receipt lineage checked against local HEST checkpoint; historical implementation source is not independently recovered from its receipt hash",
                "canonical_root": str(canonical_root), "reference_vocabulary": common.identity(vocabulary),
                "original_profiles": common.identity(profiles), "source_distance_quantiles": common.identity(quantiles),
                "implementation": implementation_files(), "runtime": environment, "method": grid.execution_contract(),
                "extraction": {"device": "cuda:0", "batch_size": 64, "workers": 4, "eval_mode": True,
                               "augmentation": False, "autocast": "encoder.precision (UNI-v1 float16)",
                               "eval_transforms": ["Resize(224)", "CenterCrop(224)", "ToTensor", "Normalize(mean=.485,.456,.406;std=.229,.224,.225)"],
                               "patch_read": "Canonical native level0 footprint; unchanged pinned TRIDENT PIL.Image.resize path to256 before frozen UNI evaluation transforms",
                               "feature_checkpoint_dtype": "float32", "assignment_input": "float32(float16(features))",
                               "patient_substitution": False, "empty_selected_slide": "fail entire sensitivity without substitution"},
                "entry_commands": {s: [str(BASE_PYTHON), str(Path(__file__).resolve()), s, "--output", str(output.resolve())]
                                   for s in ("extract", "summarize")}}
    write_json(contract_path, contract)
    seal_json(contract_path)
    return contract


def verify_contract(output: Path, *, live_inputs: bool = True, recover_seal: bool = False) -> dict:
    contract_path = output / "extraction_contract.json"
    missing_seal = not Path(str(contract_path) + ".seal.json").exists()
    contract = (json.loads(contract_path.read_text()) if recover_seal and missing_seal
                else common.read_sealed(contract_path))
    if contract.get("status") != "GRID_OFFSET_EXTRACTION_CONTRACT_SEALED":
        raise common.ContractError("Unexpected grid extraction contract status")
    for record in contract["implementation"] + [contract[k] for k in ("roster", "uni_checkpoint", "canonical_mask_hest_checkpoint", "historical_receipt_schema_inventory", "reference_vocabulary", "original_profiles", "source_distance_quantiles")]:
        common.verify_identity(record)
    if runtime() != contract["runtime"]:
        raise common.ContractError("Frozen grid extraction runtime changed")
    roster = common.read_sealed(Path(contract["roster"]["path"]))
    common.verify_identity(roster["eligibility_audit"])
    patients = pd.DataFrame(roster["patients"])
    if len(patients) != 100 or patients.patient_id.duplicated().any() or patients.groupby("subcohort").size().to_dict() != dict.fromkeys(grid.SUBCOHORTS, 25):
        raise common.ContractError("Locked 100-patient roster drift")
    if live_inputs:
        grid.require_canonical_root(Path(contract["canonical_root"]))
        for row in roster["slides"]:
            verify_slide_inputs(row)
    if recover_seal and missing_seal:
        seal_json(contract_path)
    return contract


def verify_slide_inputs(row: dict) -> None:
    verify_raw(row["wsi"])
    for item in [row["mask"], row["coordinates"], *row["stream_receipts"]]:
        common.verify_identity(item)


def validate_feature_checkpoint(path: Path, coordinates: np.ndarray) -> None:
    with h5py.File(path, "r") as f:
        if ("features" not in f or "coords" not in f or f["features"].shape != (len(coordinates), 1024)
                or f["features"].dtype != np.dtype("float32")
                or not np.array_equal(f["coords"][:], coordinates)
                or str(f["features"].attrs.get("encoder")) != "uni_v1"):
            raise common.ContractError("Grid feature checkpoint geometry/encoder drift")
        for start in range(0, len(coordinates), 8192):
            if not np.isfinite(f["features"][start:start + 8192]).all():
                raise common.ContractError("Nonfinite grid features")


def _recover_or_publish_stage(path: Path, expected: dict) -> dict:
    """Recover a completed immutable data artifact after a receipt-write interruption."""
    receipt_path = Path(str(path) + ".receipt.json")
    record = {"artifact": common.identity(path), **expected}
    if receipt_path.exists():
        seal_path = Path(str(receipt_path) + ".seal.json")
        previous = common.read_sealed(receipt_path) if seal_path.exists() else json.loads(receipt_path.read_text())
        if previous != record:
            raise common.ContractError(f"Stage checkpoint binding drift: {path}")
        seal_json(receipt_path)
        return previous
    write_json(receipt_path, record)
    seal_json(receipt_path)
    return record


def process_slide(row: dict, model: Any, vocabulary: Any, q99: np.ndarray, contract_hash: str, output: Path) -> dict:
    """Each stage persists separately; completed slides survive tmux/process loss."""
    import geopandas as gpd
    from trident import load_wsi

    from oceanpath.extraction.mpp_sampling import ExactMppWSIAdapter, _write_coordinate_h5
    verify_slide_inputs(row)
    directory = output / "slides" / row["slide_id"]
    final = directory / "receipt.json"
    if final.exists() and Path(str(final) + ".seal.json").exists():
        receipt = common.read_sealed(final)
        if receipt["contract_sha256"] != contract_hash:
            raise common.ContractError("Grid slide contract drift")
        for item in receipt["artifacts"]:
            common.verify_identity(item)
        return receipt
    directory.mkdir(parents=True, exist_ok=True)
    coordinates = grid.shifted_tissue_coordinates(row["canonical_attributes"], gpd.read_file(row["mask"]["path"]))
    if not len(coordinates):
        raise common.ContractError(f"Selected slide has no accepted offset tile; no replacement: {row['slide_id']}")
    coords_path = directory / "offset_coordinates.h5"
    attrs = {**row["canonical_attributes"], "savetodir": str(directory), "name": row["slide_id"],
             "grid_offset_x_level0": int(row["canonical_attributes"]["patch_size_level0"]) // 2,
             "grid_offset_y_level0": int(row["canonical_attributes"]["patch_size_level0"]) // 2,
             "grid_offset_contract_sha256": contract_hash, "canonical_coordinates_sha256": row["coordinates"]["sha256"]}
    if coords_path.exists():
        with h5py.File(coords_path, "r") as f:
            if not np.array_equal(f["coords"][:], coordinates) or f["coords"].attrs.get("grid_offset_contract_sha256") != contract_hash:
                raise common.ContractError("Offset coordinate checkpoint drift")
    else:
        _write_coordinate_h5(coords_path, coordinates=coordinates, attributes=attrs)
    _recover_or_publish_stage(coords_path, {"contract_sha256": contract_hash, "stage": "OFFSET_COORDINATES"})
    feature_path = directory / "features" / f"{row['slide_id']}.h5"
    feature_start = directory / "feature_start.json"
    start_record = {"contract_sha256": contract_hash, "coordinates": common.identity(coords_path), "uni_sha256": grid.UNI_SHA256}
    if feature_start.exists():
        existing = (common.read_sealed(feature_start) if Path(str(feature_start) + ".seal.json").exists()
                    else json.loads(feature_start.read_text()))
        if existing != start_record:
            raise common.ContractError("Feature extraction start binding drift")
        seal_json(feature_start)
    else:
        if feature_path.exists():
            raise common.ContractError("An existing feature file has no sealed extraction-start provenance")
        write_json(feature_start, start_record)
        seal_json(feature_start)
    if not feature_path.exists():
        wsi = load_wsi(row["wsi"]["path"], reader_type="openslide", name=row["slide_id"] + Path(row["wsi"]["path"]).suffix,
                       mpp=row["mpp"], max_workers=4)
        adapter = ExactMppWSIAdapter(wsi, target_mpp=.5)
        try:
            adapter.extract_patch_features(model, str(coords_path), str(feature_path.parent), device="cuda:0", saveas="h5", batch_limit=64)
        finally:
            wsi.release()
    validate_feature_checkpoint(feature_path, coordinates)
    feature_stage = _recover_or_publish_stage(feature_path, {"contract_sha256": contract_hash, "stage": "OFFSET_UNI_FEATURES", "coordinates": common.identity(coords_path)})
    assignment_path = directory / "reference_assignments.npz"
    if assignment_path.exists():
        with np.load(assignment_path, allow_pickle=False) as f:
            labels, distances = f["labels"], f["distances"]
            if (str(f["feature_sha256"].item()) != feature_stage["artifact"]["sha256"]
                    or str(f["contract_sha256"].item()) != contract_hash):
                raise common.ContractError("Assignment feature binding drift")
    else:
        with h5py.File(feature_path, "r") as f:
            features = grid.canonical_quantize(f["features"][:])
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            labels, distances = vocabulary.assign_with_distances(features)
        stream = io.BytesIO()
        np.savez_compressed(stream, labels=labels.astype(np.int16), distances=distances.astype(np.float32),
                            feature_sha256=np.asarray(feature_stage["artifact"]["sha256"]),
                            contract_sha256=np.asarray(contract_hash))
        atomic_write_once(assignment_path, stream.getvalue())
    if len(labels) != len(coordinates) or len(distances) != len(coordinates):
        raise common.ContractError("Offset assignments do not cover every tile")
    abundance, diagnostics = common.slide_diagnostics(labels, distances, q99)
    for cell in diagnostics:
        for key in ("median_distance", "fraction_beyond_source_q99"):
            if not np.isfinite(cell[key]):
                cell[key] = None
    _recover_or_publish_stage(assignment_path, {"contract_sha256": contract_hash, "stage": "OFFSET_REFERENCE_ASSIGNMENTS", "features": feature_stage["artifact"]})
    verify_slide_inputs(row)
    receipt = {"status": "GRID_OFFSET_SLIDE_COMPLETE", "contract_sha256": contract_hash,
               "slide_id": row["slide_id"], "patient_id": row["patient_id"], "subcohort": row["subcohort"],
               "tile_count": len(coordinates), "abundance": abundance.tolist(), "distance_support": diagnostics,
               "artifacts": [common.identity(p) for p in (coords_path, feature_path, assignment_path)]}
    write_json(final, receipt)
    seal_json(final)
    return receipt


def extract(output: Path) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/v14-mpl")
    contract = verify_contract(output)
    import torch
    if not torch.cuda.is_available():
        raise grid.OperationalBlock("CUDA is not accessible in this process. Launch the base-runtime tmux job with the approved WSL GPU access; sandbox detection is not a scientific grid verdict.")
    from trident.patch_encoder_models.load import encoder_factory

    from oceanpath.aim1.v14_concepts import V14Vocabulary
    roster = common.read_sealed(Path(contract["roster"]["path"]))
    model = encoder_factory("uni_v1", weights_path=contract["uni_checkpoint"]["path"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    transforms = list(model.eval_transforms.transforms)
    if (model.enc_name != "uni_v1" or model.precision != torch.float16
            or [type(t).__name__ for t in transforms] != ["Resize", "CenterCrop", "ToTensor", "Normalize"]
            or transforms[0].size != 224 or tuple(transforms[1].size) != (224, 224)
            or tuple(transforms[3].mean) != (.485, .456, .406)
            or tuple(transforms[3].std) != (.229, .224, .225)
            or any(module.training for module in model.modules())):
        raise common.ContractError("Frozen UNI precision/evaluation preprocessing drift")
    model.to("cuda:0")
    with np.load(contract["reference_vocabulary"]["path"], allow_pickle=False) as f:
        vocabulary = V14Vocabulary(np.asarray(f["centroids"], dtype=np.float32),
                                   np.asarray(f["pca_mean"], dtype=np.float32),
                                   np.asarray(f["pca_components"], dtype=np.float32))
    q = pd.read_csv(contract["source_distance_quantiles"]["path"]).sort_values("prototype_id")
    q99 = q.q99_distance.to_numpy(float)
    if len(q99) != 32 or not np.isfinite(q99).all():
        raise common.ContractError("Canonical source all-tile distance quantiles invalid")
    contract_hash = common.digest(output / "extraction_contract.json")
    receipts = []
    for position, row in enumerate(roster["slides"], start=1):
        process_slide(row, model, vocabulary, q99, contract_hash, output)
        receipts.append(common.identity(output / "slides" / row["slide_id"] / "receipt.json"))
        print(json.dumps({"stage": "GRID_OFFSET_EXTRACTION", "slide": row["slide_id"], "completed": position, "total": len(roster["slides"])}), flush=True)
    result = {"status": "GRID_OFFSET_ALL_LISTED_SLIDES_COMPLETE", "contract": common.identity(output / "extraction_contract.json"),
              "patient_count": 100, "slide_count": len(receipts), "slide_receipts": receipts}
    write_json(output / "extraction_complete.json", result)
    seal_json(output / "extraction_complete.json")
    return result


def summarize(output: Path) -> dict:
    contract = verify_contract(output, live_inputs=False)
    completion = common.read_sealed(output / "extraction_complete.json")
    roster = common.read_sealed(Path(contract["roster"]["path"]))
    expected = {r["slide_id"] for r in roster["slides"]}
    relationship = {r["slide_id"]: (r["patient_id"], r["subcohort"]) for r in roster["slides"]}
    slide_rows, cells = [], []
    for item in completion["slide_receipts"]:
        receipt = common.read_sealed(common.verify_identity(item))
        if receipt["contract_sha256"] != common.digest(output / "extraction_contract.json"):
            raise common.ContractError("Slide completed under different offset contract")
        if relationship.get(receipt["slide_id"]) != (receipt["patient_id"], receipt["subcohort"]):
            raise common.ContractError("A completed slide changed its locked patient/subcohort assignment")
        for artifact in receipt["artifacts"]:
            common.verify_identity(artifact)
        slide_rows.append({"slide_id": receipt["slide_id"], "patient_id": receipt["patient_id"], "subcohort": receipt["subcohort"],
                           "n_tiles": receipt["tile_count"], **dict(zip(grid.PROTOTYPES, receipt["abundance"], strict=True))})
        cells.extend({"patient_id": receipt["patient_id"], "slide_id": receipt["slide_id"], "cohort": receipt["subcohort"], **r} for r in receipt["distance_support"])
    slides = pd.DataFrame(slide_rows)
    if set(slides.slide_id) != expected or slides.slide_id.duplicated().any():
        raise common.ContractError("Cannot summarize an incomplete/replaced selected-slide roster")
    offset = grid.equal_slide_profiles(slides)
    original = pd.read_parquet(contract["original_profiles"]["path"])
    original = original[original.patient_id.isin(offset.patient_id)].copy()
    if len(offset) != 100 or len(original) != 100:
        raise common.ContractError("Cannot summarize a reduced/replaced patient roster")
    if not np.array_equal(original.set_index("patient_id").loc[offset.patient_id, "n_slides"].to_numpy(), offset.n_slides.to_numpy()):
        raise common.ContractError("Original/offset profiles do not include exactly the same listed slides")
    result = grid.paired_profile_summary(original, offset)
    dummy_slides = slides.rename(columns={"subcohort": "cohort"})
    _, patient_distances = common.aggregate_slides(dummy_slides, pd.DataFrame(cells))
    artifacts = []
    for name, frame in (("slide_abundances.csv", slides), ("offset_patient_profiles.csv", offset),
                        ("offset_patient_distance_support.csv", patient_distances),
                        ("patient_cosine.csv", pd.DataFrame(result["patient_cosine"])),
                        ("prototype_icc.csv", pd.DataFrame(result["prototype_icc"]))):
        path = output / name
        atomic_write_once(path, frame.to_csv(index=False, float_format="%.17g").encode())
        artifacts.append(common.identity(path))
    result.update({"status": "GRID_OFFSET_SENSITIVITY_COMPLETE", "contract": common.identity(output / "extraction_contract.json"),
                   "completion": common.identity(output / "extraction_complete.json"), "artifacts": artifacts})
    write_json(output / "results.json", result)
    seal_json(output / "results.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "extract", "summarize"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--canonical-root", type=Path, default=grid.DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--pre-reader", type=Path, default=grid.DEFAULT_PRE_READER)
    parser.add_argument("--uni-checkpoint", type=Path, default=grid.DEFAULT_UNI)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(common.RUN.resolve()) or output.is_relative_to(grid.DEFAULT_PRE_READER.resolve()):
        raise common.ContractError("Offset extraction writes only to the governed new v14 rerun")
    output.mkdir(parents=True, exist_ok=True)
    # An OS lock protects concurrent writers and automatically releases after crashes.
    with (output / "execution.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise grid.OperationalBlock("Another grid-offset coordinator is already running") from exc
        if args.stage == "prepare":
            result = prepare(args.pre_reader, args.canonical_root, args.uni_checkpoint, output)
        else:
            result = (extract if args.stage == "extract" else summarize)(output)
    print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
