from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v4_bundle_receipt as bundle  # noqa: E402


def test_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert (
        bundle.sha256(path)
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_component_artifact_tamper_is_detected(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "results.json"
    artifact.write_text("{}\n")
    receipt = {
        "status": "PASS",
        "artifacts": [{"path": str(artifact), "sha256": bundle.sha256(artifact)}],
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))

    monkeypatch.setattr(bundle, "COMPONENT_RECEIPTS", {"only": receipt_path})
    monkeypatch.setattr(bundle, "E2E_RECEIPTS", {})
    monkeypatch.setattr(bundle, "SNAPSHOT_FILES", ())
    monkeypatch.setattr(bundle, "MARKDOWN", ())
    parent = tmp_path / "parent.json"
    parent.write_text("{}")
    monkeypatch.setattr(bundle, "FINAL_V3", tmp_path)
    monkeypatch.setattr(bundle, "FINAL_V4", tmp_path)
    monkeypatch.setattr(bundle, "SNAPSHOT", tmp_path)
    snapshot_receipt = tmp_path / "snap.json"
    snapshot_receipt.write_text(json.dumps({"identities": {}}))
    monkeypatch.setattr(bundle, "SNAPSHOT_RECEIPT", snapshot_receipt)
    (tmp_path / "parent_final_v3_receipt.json").write_text("{}")
    (tmp_path / "report_bundle_receipt.json").write_text("{}")

    problems: list[str] = []
    bundle.verify(problems)
    assert problems == []

    artifact.write_text('{"tampered": true}\n')
    problems = []
    bundle.verify(problems)
    assert any("hash mismatch" in p for p in problems)


def test_non_pass_component_status_is_detected(tmp_path: Path, monkeypatch) -> None:
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"status": "FAIL", "artifacts": []}))
    monkeypatch.setattr(bundle, "COMPONENT_RECEIPTS", {"only": receipt_path})
    monkeypatch.setattr(bundle, "E2E_RECEIPTS", {})
    monkeypatch.setattr(bundle, "SNAPSHOT_FILES", ())
    monkeypatch.setattr(bundle, "MARKDOWN", ())
    monkeypatch.setattr(bundle, "FINAL_V3", tmp_path)
    monkeypatch.setattr(bundle, "FINAL_V4", tmp_path)
    monkeypatch.setattr(bundle, "SNAPSHOT", tmp_path)
    snapshot_receipt = tmp_path / "snap.json"
    snapshot_receipt.write_text(json.dumps({"identities": {}}))
    monkeypatch.setattr(bundle, "SNAPSHOT_RECEIPT", snapshot_receipt)
    (tmp_path / "parent_final_v3_receipt.json").write_text("{}")
    (tmp_path / "report_bundle_receipt.json").write_text("{}")

    problems: list[str] = []
    bundle.verify(problems)
    assert any("!= PASS" in p for p in problems)


@pytest.mark.skipif(
    not bundle.SNAPSHOT.exists() or not bundle.E2E_ROOT.exists(),
    reason="real final-v4 layout not present",
)
def test_real_final_v4_layout_verifies() -> None:
    problems: list[str] = []
    bundle.verify(problems)
    assert problems == []
