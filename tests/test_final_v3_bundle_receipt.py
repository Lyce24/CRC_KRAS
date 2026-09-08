from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v3_bundle_receipt as bundle  # noqa: E402


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[bundle.BundlePaths, dict[str, Path]]:
    final_v2 = tmp_path / "reports" / "final_v2"
    final_v3 = tmp_path / "reports" / "final_v3"
    snapshot_root = tmp_path / "reports" / "snapshots" / "final_v2_snapshot"
    snapshot_receipt = tmp_path / "reports" / "snapshots" / "final_v2_snapshot.receipt.json"
    additions = tmp_path / "reports" / "reruns" / "additions"

    for filename in bundle.REPORT_DOCUMENTS:
        _write(final_v2 / filename, f"# final v2 {filename}\n")
    parent = {
        "schema_version": 1,
        "status": "PASS",
        "append_only": True,
        "bundle_root": str(final_v2.resolve()),
        "documents": {
            filename: bundle.identity(final_v2 / filename)
            for filename in bundle.REPORT_DOCUMENTS
        },
    }
    _write_json(final_v2 / "report_bundle_receipt.json", parent)
    final_v3.mkdir(parents=True)
    (final_v3 / "parent_final_v2_receipt.json").write_bytes(
        (final_v2 / "report_bundle_receipt.json").read_bytes()
    )
    for filename in bundle.REPORT_DOCUMENTS:
        _write(final_v3 / filename, f"# final v3 {filename}\ncomplete\n")

    snapshot_root.mkdir(parents=True)
    for filename in bundle.SNAPSHOT_DOCUMENTS:
        (snapshot_root / filename).write_bytes((final_v2 / filename).read_bytes())
    _write_json(
        snapshot_receipt,
        {
            "schema_version": 1,
            "status": "PASS",
            "append_only": True,
            "source_root": str(final_v2.resolve()),
            "snapshot_root": str(snapshot_root.resolve()),
            # This receipt schema intentionally omits a path in each record.
            "documents": {
                filename: {
                    "sha256": bundle.sha256_file(snapshot_root / filename),
                    "size_bytes": (snapshot_root / filename).stat().st_size,
                }
                for filename in bundle.SNAPSHOT_DOCUMENTS
            },
        },
    )

    shared_input = tmp_path / "upstream" / "source.json"
    _write(shared_input, '{"source": true}\n')
    components: dict[str, Path] = {}

    # Aim 1 schema: artifact list that points to a nested input receipt.
    root = additions / "aim1_worklist"
    result = root / "results.json"
    nested = root / "input_receipt.json"
    _write(result, '{"capture": 0.45}\n')
    _write_json(nested, {"inputs": [bundle.identity(shared_input)]})
    receipt = root / "receipt.json"
    _write_json(
        receipt,
        {
            "status": "PASS",
            "append_only": True,
            "artifacts": [bundle.identity(result), bundle.identity(nested)],
        },
    )
    components["aim1_worklist"] = receipt

    # Aim 2 schema: nested lists under semantic files/input/output keys.
    root = additions / "aim2_operational"
    result = root / "agreement.csv"
    _write(result, "icc,0.69\n")
    receipt = root / "receipt.json"
    _write_json(
        receipt,
        {
            "status": "PASS",
            "files": {
                "inputs": [bundle.identity(shared_input)],
                "outputs": [bundle.identity(result)],
            },
        },
    )
    components["aim2_operational"] = receipt

    # Aim 3 schema: keyed input/output dictionaries.
    root = additions / "aim3_actionability"
    result = root / "ladder.csv"
    _write(result, "level,status\ngene,QA only\n")
    receipt = root / "receipt.json"
    _write_json(
        receipt,
        {
            "status": "PASS",
            "append_only": True,
            "inputs": {"fixed": bundle.identity(shared_input)},
            "outputs": {"ladder.csv": bundle.identity(result)},
        },
    )
    components["aim3_actionability"] = receipt

    # Aim 4 schema: keyed relative artifact identity without a path field.
    root = additions / "aim4_compressibility_v2"
    result = root / "compressibility.json"
    _write(result, '{"r2": 0.39}\n')
    receipt = root / "completion_receipt.json"
    _write_json(
        receipt,
        {
            "status": "COMPLETE",
            "append_only": True,
            "upstream_inputs": [bundle.identity(shared_input)],
            "artifacts": {
                "compressibility.json": {
                    "sha256": bundle.sha256_file(result),
                    "size_bytes": result.stat().st_size,
                }
            },
        },
    )
    components["aim4_compressibility"] = receipt

    paths = bundle.BundlePaths(
        final_v2=final_v2,
        snapshot_root=snapshot_root,
        snapshot_receipt=snapshot_receipt,
        final_v3=final_v3,
        component_receipts=components,
        destination=final_v3 / "report_bundle_receipt.json",
    )
    material = {
        "shared_input": shared_input,
        "snapshot_results": snapshot_root / "Results.md",
        "parent_copy": final_v3 / "parent_final_v2_receipt.json",
        "aim2_result": additions / "aim2_operational" / "agreement.csv",
    }
    return paths, material


def test_heterogeneous_component_receipts_and_snapshot_verify(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)

    observed = bundle.verify_bundle(paths)

    assert observed["status"] == "PASS"
    assert set(observed["components"]) == set(bundle.COMPONENT_NAMES)
    assert observed["independent_checks"]["component_receipt_recursive_rehash"] == "4/4 PASS"
    assert observed["parent_final_v2_and_snapshot"]["parent_copy_byte_identical"] is True
    assert observed["components"]["aim1_worklist"]["unique_declared_files"] == 3
    assert observed["components"]["aim4_compressibility"]["component_local_files"] == 1
    assert observed["components"]["aim4_compressibility"]["declared_component_status"] == (
        "COMPLETE"
    )


def test_nested_declared_input_tampering_fails_closed(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    material["shared_input"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(bundle.BundleVerificationError, match="mismatch"):
        bundle.verify_bundle(paths)


def test_component_output_tampering_fails_closed(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    material["aim2_result"].write_text("icc,0.99\n", encoding="utf-8")

    with pytest.raises(bundle.BundleVerificationError, match="mismatch"):
        bundle.verify_bundle(paths)


def test_snapshot_tampering_fails_closed(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    material["snapshot_results"].write_text("changed snapshot\n", encoding="utf-8")

    with pytest.raises(bundle.BundleVerificationError, match="snapshot identity mismatch"):
        bundle.verify_bundle(paths)


def test_parent_receipt_copy_must_be_byte_identical(tmp_path: Path) -> None:
    paths, material = _fixture(tmp_path)
    material["parent_copy"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(bundle.BundleVerificationError, match="not byte-identical"):
        bundle.verify_bundle(paths)


def test_atomic_seal_is_write_once(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    payload = bundle.verify_bundle(paths)

    bundle.write_json_once_atomic(paths.destination, payload)

    assert json.loads(paths.destination.read_text())["status"] == "PASS"
    with pytest.raises(FileExistsError, match="overwrite"):
        bundle.write_json_once_atomic(paths.destination, payload)


def test_destination_must_be_directly_inside_final_v3(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    invalid = bundle.BundlePaths(
        **{
            **paths.__dict__,
            "destination": tmp_path / "elsewhere" / "receipt.json",
        }
    )

    with pytest.raises(bundle.BundleVerificationError, match="directly under"):
        bundle.verify_bundle(invalid)


def test_component_status_must_pass(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    receipt = paths.component_receipts["aim4_compressibility"]
    value = json.loads(receipt.read_text())
    value["status"] = "PENDING"
    _write_json(receipt, value)

    with pytest.raises(bundle.BundleVerificationError, match="not PASS/COMPLETE"):
        bundle.verify_bundle(paths)


def test_superseded_receipt_is_rehashed_but_not_recursed(tmp_path: Path) -> None:
    paths, _ = _fixture(tmp_path)
    aim2_receipt = paths.component_receipts["aim2_operational"]
    superseded = aim2_receipt.parent / "superseded_receipt.json"
    _write_json(
        superseded,
        {
            "status": "PASS",
            "outputs": [
                {
                    "path": str(aim2_receipt.parent / ".deleted-staging" / "result.csv"),
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                }
            ],
        },
    )
    value = json.loads(aim2_receipt.read_text())
    value["supersedes"] = {"artifact": bundle.identity(superseded)}
    _write_json(aim2_receipt, value)

    observed = bundle.verify_bundle(paths)

    component = observed["components"]["aim2_operational"]
    assert component["status"] == "PASS"
    assert component["unique_declared_files"] == 3
    assert str(superseded.resolve()) not in component["recursively_verified_receipts"]
