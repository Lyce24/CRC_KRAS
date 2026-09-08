"""Encode CPTAC-COAD with UNI/CONCH, publish, then rebuild both packs.

The run is deliberately isolated from the shared colon feature tree until
both encoder outputs have passed structural validation. Publishing uses
same-filesystem temporary files and atomic renames; pack rebuilding already
uses a validated staging directory and atomic directory replacement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import h5py
import pandas as pd

from oceanpath.datasets.packed import feature_inventory_sha256, validate_packed_dir

REPO = Path(__file__).resolve().parents[1]
AUTHORITATIVE_MPP = Path("/mnt/d/YC.Liu/manifests/colon/colon_ready_wsi_mpp.csv")
SLIDE_ROOT = Path("/mnt/d/YC.Liu/slides/colon")
ISOLATED_ROOT = Path("/mnt/wsl/oceanpath-hot/features/cptac_priority")
CACHE_ROOT = Path("/mnt/wsl/oceanpath-hot/scratch/cptac_priority/wsi_cache")
SHARED_ROOT = Path("/mnt/wsl/oceanpath-hot/features/colon_stream")
RUN_DIR = REPO / "outputs" / "cptac_priority"
SELECTION_CSV = RUN_DIR / "cptac_ready_wsi_mpp.csv"
STATE_PATH = RUN_DIR / "state.json"
EXTRACTION_PYTHON = Path("/home/yc_liu/projects/OceanPath-colon/.venv/bin/python")

UNI_CHECKPOINT = Path(
    "/home/yc_liu/.cache/huggingface/hub/models--MahmoodLab--uni/"
    "snapshots/b55a5ec6cade1a39edfe6534189a9b8ca7a022f0/pytorch_model.bin"
)
CONCH_CHECKPOINT = Path(
    "/home/yc_liu/.cache/huggingface/hub/models--MahmoodLab--conchv1_5/"
    "snapshots/3e5766a5d1500d53c73c03005e24c30c1f27be13/"
    "pytorch_model_vision.bin"
)

ENCODERS = {
    "uni_v1": {
        "config": "univ1",
        "checkpoint": UNI_CHECKPOINT,
        "patch_size": 256,
        "feature_dim": 1024,
        "batch_size": 256,
    },
    "conch_v15": {
        "config": "conch_v15",
        "checkpoint": CONCH_CHECKPOINT,
        "patch_size": 512,
        "feature_dim": 768,
        "batch_size": 128,
    },
}


def _write_state(stage: str, **details: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **details,
    }
    temporary = STATE_PATH.with_name(f".{STATE_PATH.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, STATE_PATH)
    print(f"STATE {stage}: {json.dumps(details, sort_keys=True)}", flush=True)


def _build_selection() -> list[str]:
    if not AUTHORITATIVE_MPP.is_file():
        raise FileNotFoundError(AUTHORITATIVE_MPP)
    rows: list[dict[str, str]] = []
    with AUTHORITATIVE_MPP.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["wsi", "mpp"]:
            raise ValueError(
                f"Expected authoritative columns ['wsi', 'mpp'], got {reader.fieldnames}"
            )
        for row in reader:
            wsi = row["wsi"].strip()
            if not wsi.startswith("CPTAC_COAD/"):
                continue
            mpp = float(row["mpp"])
            if not 0.1 <= mpp <= 1.0:
                raise ValueError(f"Invalid source MPP for {wsi}: {mpp}")
            if not (SLIDE_ROOT / wsi).is_file():
                raise FileNotFoundError(SLIDE_ROOT / wsi)
            rows.append({"wsi": wsi, "mpp": format(mpp, ".17g")})

    if len(rows) != 98:
        raise RuntimeError(f"Expected 98 CPTAC rows, found {len(rows)}")
    slide_ids = [Path(row["wsi"]).stem for row in rows]
    if len(set(slide_ids)) != 98:
        raise RuntimeError("CPTAC slide IDs are not unique after removing extensions")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    temporary = SELECTION_CSV.with_name(f".{SELECTION_CSV.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["wsi", "mpp"])
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, SELECTION_CSV)
    return slide_ids


def _coords_subdir(patch_size: int) -> str:
    return f"20x_{patch_size}px_0px_overlap_mpp0.5"


def _feature_dir(root: Path, encoder: str) -> Path:
    patch_size = int(ENCODERS[encoder]["patch_size"])
    return root / _coords_subdir(patch_size) / f"features_{encoder}"


def _pack_dir(encoder: str) -> Path:
    return _feature_dir(SHARED_ROOT, encoder).with_name(f"packed_{encoder}")


def _run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=REPO,
        check=True,
        env={
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(REPO / "src"),
        },
    )


def _extract(encoder: str, tasks: str) -> None:
    settings = ENCODERS[encoder]
    _run(
        [
            str(EXTRACTION_PYTHON),
            "scripts/extract_features.py",
            "platform=colon_workstation",
            "data=colon",
            f"encoder={settings['config']}",
            "extraction=colon_native",
            f"data.custom_list_of_wsis={SELECTION_CSV}",
            f"data.feature_job_dir={ISOLATED_ROOT}",
            f"extraction.wsi_cache={CACHE_ROOT}",
            f"encoder.checkpoint_path={settings['checkpoint']}",
            "extraction.target_mpp=0.5",
            "extraction.seg_batch_size=64",
            f"extraction.feat_batch_size={settings['batch_size']}",
            "extraction.max_workers=10",
            "extraction.cache_batch_size=8",
            f"tasks={tasks}",
        ]
    )


def _validate_features(encoder: str, slide_ids: list[str]) -> None:
    feature_dir = _feature_dir(ISOLATED_ROOT, encoder)
    expected = set(slide_ids)
    actual = {path.stem for path in feature_dir.glob("*.h5")}
    if actual != expected:
        raise RuntimeError(
            f"{encoder} isolated feature inventory mismatch: "
            f"missing={sorted(expected - actual)[:5]}, unexpected={sorted(actual - expected)[:5]}"
        )
    feature_dim = int(ENCODERS[encoder]["feature_dim"])
    for slide_id in slide_ids:
        path = feature_dir / f"{slide_id}.h5"
        with h5py.File(path, "r") as handle:
            if "features" not in handle or "coords" not in handle:
                raise RuntimeError(f"{path} lacks features or coords")
            features = handle["features"]
            coords = handle["coords"]
            if features.ndim != 2 or features.shape[1] != feature_dim:
                raise RuntimeError(f"Unexpected feature shape in {path}: {features.shape}")
            if features.shape[0] == 0 or coords.shape[0] != features.shape[0]:
                raise RuntimeError(
                    f"Invalid patch/coordinate counts in {path}: {features.shape}, {coords.shape}"
                )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _publish_encoder(encoder: str, slide_ids: list[str]) -> None:
    source_dir = _feature_dir(ISOLATED_ROOT, encoder)
    destination_dir = _feature_dir(SHARED_ROOT, encoder)
    destination_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.stem for path in destination_dir.glob("*.h5")}
    unrelated = existing - set(slide_ids)
    if len(unrelated) != 1989:
        raise RuntimeError(
            f"Expected 1,989 pre-CPTAC {encoder} features, found {len(unrelated)}; refusing publish"
        )

    for slide_id in slide_ids:
        source = source_dir / f"{slide_id}.h5"
        destination = destination_dir / source.name
        source_hash = _sha256(source)
        if destination.exists():
            if _sha256(destination) != source_hash:
                raise RuntimeError(f"Refusing to replace non-identical existing file: {destination}")
            continue
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            shutil.copyfile(source, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            if _sha256(temporary) != source_hash:
                raise RuntimeError(f"Hash mismatch while publishing {source}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    final = {path.stem for path in destination_dir.glob("*.h5")}
    if len(final) != 2087 or not set(slide_ids).issubset(final):
        raise RuntimeError(f"Published {encoder} inventory is incomplete: {len(final)} files")


def _rebuild_pack(encoder: str, slide_ids: list[str]) -> None:
    settings = ENCODERS[encoder]
    _run(
        [
            str(EXTRACTION_PYTHON),
            "scripts/pack_features.py",
            "platform=colon_workstation",
            "data=colon",
            f"encoder={settings['config']}",
            "extraction=colon_native",
            "pack.overwrite=true",
        ]
    )
    source_dir = _feature_dir(SHARED_ROOT, encoder)
    inventory_hash = feature_inventory_sha256(source_dir)
    meta = validate_packed_dir(_pack_dir(encoder), verify_source=inventory_hash)
    if meta.n_slides != 2087 or meta.feat_dim != int(settings["feature_dim"]):
        raise RuntimeError(f"Invalid {encoder} pack metadata: {meta}")
    index = pd.read_parquet(_pack_dir(encoder) / "index.parquet", columns=["slide_id"])
    packed_ids = set(index["slide_id"].astype(str))
    missing = set(slide_ids) - packed_ids
    if missing:
        raise RuntimeError(f"{encoder} pack lacks CPTAC IDs: {sorted(missing)[:5]}")


def main() -> None:
    _write_state("preflight")
    for required in (EXTRACTION_PYTHON, UNI_CHECKPOINT, CONCH_CHECKPOINT):
        if not required.is_file():
            raise FileNotFoundError(required)
    slide_ids = _build_selection()
    _write_state("extract_uni", n_cptac=len(slide_ids), target_mpp=0.5)
    _extract("uni_v1", "[seg,coords,feat]")
    _validate_features("uni_v1", slide_ids)

    _write_state("extract_conch", n_cptac=len(slide_ids), target_mpp=0.5)
    _extract("conch_v15", "[coords,feat]")
    _validate_features("conch_v15", slide_ids)

    _write_state("publish", n_cptac=len(slide_ids))
    _publish_encoder("uni_v1", slide_ids)
    _publish_encoder("conch_v15", slide_ids)

    _write_state("pack_uni", expected_slides=2087)
    _rebuild_pack("uni_v1", slide_ids)
    _write_state("pack_conch", expected_slides=2087)
    _rebuild_pack("conch_v15", slide_ids)
    _write_state("complete", n_cptac=len(slide_ids), packed_slides=2087)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write_state("failed", error=f"{type(exc).__name__}: {exc}")
        raise
