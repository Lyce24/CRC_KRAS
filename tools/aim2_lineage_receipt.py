#!/usr/bin/env python3
"""Seal an immutable Aim-2 rerun and prove the legacy artifacts were untouched."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from oceanpath.aim1 import lineage, paths  # noqa: E402


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, check=False, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _files_below(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []


def _legacy_files() -> list[Path]:
    files: list[Path] = []
    for root in (paths.OUTPUT_ROOT / "e2a", paths.OUTPUT_ROOT / "e2c"):
        files.extend(_files_below(root))
    files.extend(sorted((paths.OUTPUT_ROOT / "eval").glob("e2*")))
    return sorted(set(path.resolve() for path in files if path.is_file()))


def _code_files() -> list[Path]:
    """Exact local source/config bytes used by the corrected Aim-2 campaign."""

    files: list[Path] = [
        *(
            REPO / f"e2{name}.py"
            for name in ("a", "b", "c", "d1", "d2", "d3", "d4", "d5", "d6")
        ),
        REPO / "tools" / "study_train.py",
        REPO / "tools" / "e2_audit.py",
        REPO / "tools" / "aim2_lineage_receipt.py",
        REPO / "pyproject.toml",
        REPO / "uv.lock",
    ]
    files.extend((REPO / "src" / "oceanpath").rglob("*.py"))
    files.extend((REPO / "configs").rglob("*.yaml"))
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required Aim-2 source files: {missing}")
    return sorted(set(path.resolve() for path in files))


def _snapshot_code(root: Path, files: list[Path]) -> list[dict]:
    snapshot_root = root / "source_snapshot"
    for source in files:
        destination = snapshot_root / source.relative_to(REPO.resolve())
        lineage.write_bytes_once(destination, source.read_bytes())
    return _inventory(_files_below(snapshot_root), relative_to=root)


def _inventory(files: list[Path], *, relative_to: Path) -> list[dict]:
    out = []
    for path in files:
        stat = path.stat()
        out.append(
            {
                "path": str(path.relative_to(relative_to.resolve())),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": lineage.sha256_file(path),
            }
        )
    return out


def cmd_start() -> None:
    root = lineage.aim2_root()
    if root.exists():
        raise FileExistsError(
            f"Final Aim-2 lineage must be wholly new at start: {root}"
        )
    legacy = _inventory(_legacy_files(), relative_to=paths.OUTPUT_ROOT)
    code_files = _code_files()
    live_code = _inventory(code_files, relative_to=REPO)
    snapshot_code = _snapshot_code(root, code_files)
    payload = {
        "schema_version": 1,
        "status": "started",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lineage": lineage.lineage_name(),
        "lineage_root": str(root.resolve()),
        "source_cv_input_root": str(lineage.source_cv_input_root()),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain=v1"),
        "legacy_inventory": legacy,
        "legacy_file_count": len(legacy),
        "legacy_total_bytes": int(sum(row["size_bytes"] for row in legacy)),
        "live_code_inventory": live_code,
        "source_snapshot_inventory": snapshot_code,
        "source_file_count": len(live_code),
        "promise": "legacy Aim-2 files are read-only inputs and must remain byte-identical",
    }
    lineage.write_json_once(root / "lineage_start.json", payload)
    print(
        f"Started {root}: sealed {len(legacy)} legacy files "
        f"({payload['legacy_total_bytes']:,} bytes)"
    )


def cmd_finalize() -> None:
    root = lineage.aim2_root()
    start_path = root / "lineage_start.json"
    if not start_path.is_file():
        raise FileNotFoundError(f"Missing lineage start receipt: {start_path}")
    destination = root / "lineage_final.json"
    lineage.ensure_absent(destination)
    started = json.loads(start_path.read_text())
    if started.get("lineage") != lineage.lineage_name():
        raise RuntimeError("Lineage environment does not match start receipt")
    current_legacy = _inventory(_legacy_files(), relative_to=paths.OUTPUT_ROOT)
    unchanged = current_legacy == started.get("legacy_inventory")
    if not unchanged:
        raise RuntimeError(
            "Legacy Aim-2 inventory changed during rerun; refusing to seal lineage"
        )
    current_code = _inventory(_code_files(), relative_to=REPO)
    if current_code != started.get("live_code_inventory"):
        raise RuntimeError(
            "Aim-2 source/config bytes changed after lineage start; refusing to seal lineage"
        )
    current_snapshot = _inventory(
        _files_below(root / "source_snapshot"), relative_to=root
    )
    if current_snapshot != started.get("source_snapshot_inventory"):
        raise RuntimeError(
            "Immutable Aim-2 source snapshot changed; refusing to seal lineage"
        )
    new_files = [
        path
        for path in _files_below(root)
        if path.resolve() != destination.resolve()
    ]
    new_inventory = _inventory(new_files, relative_to=root)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lineage": lineage.lineage_name(),
        "legacy_unchanged": True,
        "legacy_file_count": len(current_legacy),
        "source_unchanged": True,
        "source_file_count": len(current_code),
        "new_inventory": new_inventory,
        "new_file_count": len(new_inventory),
        "new_total_bytes": int(sum(row["size_bytes"] for row in new_inventory)),
        "git_head": _git("rev-parse", "HEAD"),
        "git_status_porcelain": _git("status", "--porcelain=v1"),
    }
    lineage.write_json_once(destination, payload)
    print(
        f"Sealed {root}: {len(new_inventory)} new files; "
        f"all {len(current_legacy)} legacy files remain byte-identical"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "finalize"))
    args = parser.parse_args()
    if args.command == "start":
        cmd_start()
    else:
        cmd_finalize()


if __name__ == "__main__":
    main()
