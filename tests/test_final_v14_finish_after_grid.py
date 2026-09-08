import importlib.util
import json
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "tools/final_v14_finish_after_grid.py"
SPEC = importlib.util.spec_from_file_location("finish_grid", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_predecessor_must_finish_successfully(tmp_path):
    path = tmp_path / "summary.json"
    assert not MODULE.predecessor_succeeded(path)
    for value in [{"status": "JOB_FINISHED", "exit_code": 1}, {"status": "RUNNING", "exit_code": 0}, {}]:
        path.write_text(json.dumps(value))
        with pytest.raises(RuntimeError, match="Grid job failed"):
            MODULE.predecessor_succeeded(path)
    path.write_text(json.dumps({"status": "JOB_FINISHED", "exit_code": 0}))
    assert MODULE.predecessor_succeeded(path)


def test_frozen_code_tampering_stops_resume(tmp_path):
    path = tmp_path / "code.py"
    path.write_text("original")
    pin = MODULE.identity(path)
    MODULE.verify(pin)
    path.write_text("modified")
    with pytest.raises(RuntimeError, match="changed"):
        MODULE.verify(pin)


def test_immutable_receipt_replay_and_tamper(tmp_path):
    path = tmp_path / "receipt.json"
    MODULE.publish_once(path, {"status": "ready"})
    MODULE.publish_once(Path(str(path) + ".seal.json"), MODULE.identity(path))
    assert MODULE.read_sealed(path) == {"status": "ready"}
    MODULE.publish_once(path, {"status": "ready"})
    with pytest.raises(RuntimeError, match="Preserve"):
        MODULE.publish_once(path, {"status": "different"})
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="changed"):
        MODULE.read_sealed(path)


def isolated_handoff(tmp_path, monkeypatch):
    handoff, report = tmp_path / "handoff", tmp_path / "report"
    report.mkdir()
    monkeypatch.setattr(MODULE, "HANDOFF", handoff)
    monkeypatch.setattr(MODULE, "REPORT", report)
    monkeypatch.setattr(MODULE, "RESOURCE", tmp_path / "resource.json")
    path = handoff / "handoff.json"
    MODULE.publish_once(path, {"implementation_and_inputs": []})
    MODULE.publish_once(Path(str(path) + ".seal.json"), MODULE.identity(path))
    commands = []
    monkeypatch.setattr(MODULE.subprocess, "run", lambda args, **kwargs: commands.append(args))
    return handoff, report, commands


def test_failed_grid_never_invokes_audit_report_or_seal(tmp_path, monkeypatch):
    handoff, report, commands = isolated_handoff(tmp_path, monkeypatch)
    MODULE.RESOURCE.write_text(json.dumps({"status": "JOB_FINISHED", "exit_code": 1}))
    with pytest.raises(RuntimeError, match="Grid job failed"):
        MODULE.run()
    assert commands == []
    assert list(report.iterdir()) == []
    assert not (handoff / "completion.json").exists()


def test_completed_bundle_resume_only_invokes_verification(tmp_path, monkeypatch):
    handoff, report, commands = isolated_handoff(tmp_path, monkeypatch)
    receipt = report / "final_bundle_receipt.json"
    MODULE.publish_once(receipt, {"status": "COMPLETED"})
    MODULE.publish_once(Path(str(receipt) + ".seal.json"), MODULE.identity(receipt))
    result = MODULE.run()
    assert len(commands) == 1 and commands[0][-1] == "verify"
    assert result["status"] == "FINAL_V14_COMPLETION_VERIFIED"
    assert MODULE.read_sealed(handoff / "completion.json") == result
