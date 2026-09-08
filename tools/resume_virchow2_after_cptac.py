"""Resume the production Virchow2 watcher after the CPTAC priority job.

The handoff fails closed: CPTAC UNI/CONCH outputs and packs must validate,
the authoritative MPP inventory must include the 98 CPTAC slides, and the
stopped queue must reconcile to 2,087 slides before the watcher is started.
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

import pandas as pd

from oceanpath.datasets.packed import feature_inventory_sha256, validate_packed_dir

DEVELOPMENT_REPO = Path("/home/yc_liu/projects/OceanPath-colon-development")
PRODUCTION_REPO = Path("/home/yc_liu/projects/OceanPath-colon")
PRODUCTION_PYTHON = PRODUCTION_REPO / ".venv/bin/python"
CPTAC_UNIT = "oceanpath-cptac-priority.service"
ENCODER_UNIT = "oceanpath-colon-encoder.service"

RUN_DIR = DEVELOPMENT_REPO / "outputs/cptac_priority"
CPTAC_STATE = RUN_DIR / "state.json"
HANDOFF_STATE = RUN_DIR / "virchow_handoff.json"
MPP_SHEET = Path("/mnt/d/YC.Liu/manifests/colon/colon_ready_wsi_mpp.csv")
SLIDE_ROOT = Path("/mnt/d/YC.Liu/slides/colon")
FEATURE_ROOT = Path("/mnt/wsl/oceanpath-hot/features/colon_stream")
QUEUE_DB = PRODUCTION_REPO / "outputs/colon_stream/queue.sqlite"

ENCODER_LAYOUTS = {
    "uni_v1": ("20x_256px_0px_overlap_mpp0.5", 1024),
    "conch_v15": ("20x_512px_0px_overlap_mpp0.5", 768),
}


def _write_state(stage: str, **details: object) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **details,
    }
    temporary = HANDOFF_STATE.with_name(f".{HANDOFF_STATE.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, HANDOFF_STATE)
    print(f"STATE {stage}: {json.dumps(details, sort_keys=True)}", flush=True)


def _run(command: list[str], *, cwd: Path = DEVELOPMENT_REPO) -> subprocess.CompletedProcess[str]:
    print("RUN " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"},
    )


def _unit_properties(unit: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            unit,
            "--property=ActiveState,SubState,Result,ExecMainStatus",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read valid state from {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def _wait_for_cptac() -> None:
    while True:
        state = _read_json(CPTAC_STATE)
        stage = str(state.get("stage", "unknown"))
        unit = _unit_properties(CPTAC_UNIT)
        active = unit.get("ActiveState", "unknown")
        _write_state("waiting_for_cptac", cptac_stage=stage, cptac_active_state=active)

        if stage == "failed" or active == "failed":
            raise RuntimeError(f"CPTAC job failed: state={state}, unit={unit}")
        if stage == "complete" and active == "inactive":
            if unit.get("Result") not in (None, "", "success"):
                raise RuntimeError(f"CPTAC service did not exit successfully: {unit}")
            return
        if active == "inactive" and stage != "complete":
            raise RuntimeError(f"CPTAC service stopped before completion: state={state}, unit={unit}")
        time.sleep(20)


def _read_mpp_inventory() -> tuple[list[dict[str, str]], set[str]]:
    with MPP_SHEET.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["wsi", "mpp"]:
            raise RuntimeError(f"Unexpected MPP sheet columns: {reader.fieldnames}")
        rows = list(reader)
    if len(rows) != 2087 or len({row["wsi"] for row in rows}) != 2087:
        raise RuntimeError(f"Expected 2,087 unique MPP rows, found {len(rows)}")
    cptac = {row["wsi"] for row in rows if row["wsi"].startswith("CPTAC_COAD/")}
    if len(cptac) != 98:
        raise RuntimeError(f"Expected 98 CPTAC MPP rows, found {len(cptac)}")
    for row in rows:
        mpp = float(row["mpp"])
        if not 0.1 <= mpp <= 1.0 or not (SLIDE_ROOT / row["wsi"]).is_file():
            raise RuntimeError(f"Invalid MPP inventory row: {row}")
    return rows, {Path(wsi).stem for wsi in cptac}


def _validate_cptac_products(cptac_ids: set[str]) -> None:
    for encoder, (coords_dir, feature_dim) in ENCODER_LAYOUTS.items():
        feature_dir = FEATURE_ROOT / coords_dir / f"features_{encoder}"
        pack_dir = FEATURE_ROOT / coords_dir / f"packed_{encoder}"
        feature_ids = {path.stem for path in feature_dir.glob("*.h5")}
        if len(feature_ids) != 2087 or not cptac_ids.issubset(feature_ids):
            raise RuntimeError(f"Incomplete {encoder} feature inventory: {len(feature_ids)}")
        inventory_hash = feature_inventory_sha256(feature_dir)
        meta = validate_packed_dir(pack_dir, verify_source=inventory_hash)
        if meta.n_slides != 2087 or meta.feat_dim != feature_dim:
            raise RuntimeError(f"Invalid {encoder} pack metadata: {meta}")
        index = pd.read_parquet(pack_dir / "index.parquet", columns=["slide_id"])
        packed_ids = set(index["slide_id"].astype(str))
        if len(packed_ids) != 2087 or not cptac_ids.issubset(packed_ids):
            raise RuntimeError(f"Incomplete {encoder} packed index: {len(packed_ids)}")


def _refresh_queue() -> dict[str, int]:
    completed = _run(
        [str(PRODUCTION_PYTHON), "scripts/watch_colon_encoding.py", "inventory"],
        cwd=PRODUCTION_REPO,
    )
    print(completed.stdout, end="", flush=True)
    with sqlite3.connect(QUEUE_DB) as connection:
        summary = {
            str(status): int(count)
            for status, count in connection.execute(
                "SELECT status, COUNT(*) FROM stream_jobs GROUP BY status"
            )
        }
        total = int(connection.execute("SELECT COUNT(*) FROM stream_jobs").fetchone()[0])
        cptac = int(
            connection.execute(
                "SELECT COUNT(*) FROM stream_jobs WHERE source_relpath LIKE 'CPTAC_COAD/%'"
            ).fetchone()[0]
        )
        processing = int(
            connection.execute(
                "SELECT COUNT(*) FROM stream_jobs WHERE status = 'processing'"
            ).fetchone()[0]
        )
    if total != 2087 or cptac != 98 or processing != 0:
        raise RuntimeError(
            f"Unsafe reconciled queue: total={total}, CPTAC={cptac}, processing={processing}"
        )
    return {**summary, "total": total, "cptac": cptac}


def _start_virchow() -> dict[str, object]:
    _run(["systemctl", "--user", "start", ENCODER_UNIT])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        completed = _run(
            [str(PRODUCTION_PYTHON), "scripts/watch_colon_encoding.py", "status"],
            cwd=PRODUCTION_REPO,
        )
        status = json.loads(completed.stdout)
        if status.get("running") and status.get("queue", {}).get("total") == 2087:
            return status
        properties = _unit_properties(ENCODER_UNIT)
        if properties.get("ActiveState") == "failed":
            raise RuntimeError(f"Virchow watcher service failed during startup: {properties}")
        time.sleep(3)
    raise RuntimeError("Virchow watcher did not acquire its process lock within 90 seconds")


def main() -> None:
    _write_state("waiting_for_cptac")
    _wait_for_cptac()
    _write_state("validating_cptac")
    _, cptac_ids = _read_mpp_inventory()
    _validate_cptac_products(cptac_ids)

    _write_state("refreshing_virchow_queue", expected_total=2087, expected_cptac=98)
    queue = _refresh_queue()
    _write_state("starting_virchow", queue=queue)
    status = _start_virchow()
    _write_state("complete", watcher_pid=status.get("pid"), queue=status.get("queue"))


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        _write_state("failed", error=f"{type(exc).__name__}: {exc}")
        raise
