from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import final_v5_bundle_receipt as bundle  # noqa: E402


def _minimal_layout(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(bundle, "SNAPSHOT_FILES", ())
    monkeypatch.setattr(bundle, "MARKDOWN", ())
    monkeypatch.setattr(bundle, "FINAL_V4", tmp_path)
    monkeypatch.setattr(bundle, "FINAL_V5", tmp_path)
    monkeypatch.setattr(bundle, "SNAPSHOT", tmp_path)
    snapshot_receipt = tmp_path / "snap.json"
    snapshot_receipt.write_text(json.dumps({"identities": {}}))
    monkeypatch.setattr(bundle, "SNAPSHOT_RECEIPT", snapshot_receipt)
    (tmp_path / "parent_final_v4_receipt.json").write_text("{}")
    (tmp_path / "report_bundle_receipt.json").write_text("{}")
    return tmp_path


def _component(tmp_path: Path, status: str = "PASS") -> Path:
    artifact = tmp_path / "results.json"
    artifact.write_text("{}\n")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "status": status,
                "artifacts": [{"path": str(artifact), "sha256": bundle.sha256(artifact)}],
            }
        )
    )
    return receipt_path


def test_clean_layout_verifies(tmp_path: Path, monkeypatch) -> None:
    _minimal_layout(tmp_path, monkeypatch)
    receipt_path = _component(tmp_path)
    monkeypatch.setattr(bundle, "COMPONENT_RECEIPTS", {"only": receipt_path})
    replay = tmp_path / "replay_receipt.json"
    results = tmp_path / "aim1_results.json"
    results.write_text("{}\n")
    replay.write_text(
        json.dumps({"status": "PASS", "replayed_against": bundle.sha256(results)})
    )
    monkeypatch.setattr(bundle, "AIM1_REPLAY_RECEIPT", replay)
    monkeypatch.setattr(bundle, "ADDITIONS", tmp_path)
    (tmp_path / "aim1_sealed_replay").mkdir()
    (tmp_path / "aim1_sealed_replay" / "results.json").write_text("{}\n")
    problems: list[str] = []
    bundle.verify(problems)
    assert problems == []


def test_artifact_tamper_is_detected(tmp_path: Path, monkeypatch) -> None:
    _minimal_layout(tmp_path, monkeypatch)
    receipt_path = _component(tmp_path)
    monkeypatch.setattr(bundle, "COMPONENT_RECEIPTS", {"only": receipt_path})
    monkeypatch.setattr(bundle, "AIM1_REPLAY_RECEIPT", tmp_path / "missing.json")
    (tmp_path / "results.json").write_text('{"tampered": true}\n')
    problems: list[str] = []
    bundle.verify(problems)
    assert any("hash mismatch" in p for p in problems)
    assert any("replay_receipt.json missing" in p for p in problems)


def test_failed_replay_is_detected(tmp_path: Path, monkeypatch) -> None:
    _minimal_layout(tmp_path, monkeypatch)
    monkeypatch.setattr(bundle, "COMPONENT_RECEIPTS", {})
    replay = tmp_path / "replay_receipt.json"
    replay.write_text(json.dumps({"status": "FAIL", "replayed_against": "x"}))
    monkeypatch.setattr(bundle, "AIM1_REPLAY_RECEIPT", replay)
    monkeypatch.setattr(bundle, "ADDITIONS", tmp_path)
    (tmp_path / "aim1_sealed_replay").mkdir()
    (tmp_path / "aim1_sealed_replay" / "results.json").write_text("{}\n")
    problems: list[str] = []
    bundle.verify(problems)
    assert any("replay status" in p for p in problems)


@pytest.mark.skipif(
    not (bundle.ADDITIONS / "aim1_sealed_replay" / "results.json").exists()
    or not bundle.FINAL_V5.exists(),
    reason="real final-v5 layout not present yet",
)
def test_real_final_v5_layout_verifies() -> None:
    problems: list[str] = []
    bundle.verify(problems)
    assert problems == []
