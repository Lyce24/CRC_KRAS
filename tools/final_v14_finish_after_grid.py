"""Resume final Aim-4 audit/report packaging after the existing grid job succeeds.

This operational coordinator never starts or changes a scientific calculation.
Its sealed handoff fixes the reviewed audit, reporting and packaging code before
the final grid numbers exist. Failed or missing stages cannot become completion.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/reruns/final_v14_additions_20260903"
HANDOFF = RUN / "final_completion_20260905"
GRID = RUN / "grid_offset/locked_extraction"
REPORT = ROOT / "reports/final_v14"
BASE = Path("/home/yc_liu/projects/OceanPath/.venv/bin/python")
PYTHON = ROOT / ".venv/bin/python"
RESOURCE = RUN / "grid_offset/execution_handoff_20260905/resources_run1/resource_summary.json"


def identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify(pin: dict) -> None:
    current = identity(Path(pin["path"]))
    if current != pin:
        raise RuntimeError(f"Completion handoff input changed: {pin['path']}")


def read_sealed(path: Path) -> dict:
    pin = json.loads(Path(str(path) + ".seal.json").read_text())
    pin = pin.get("artifact", pin)
    if Path(pin["path"]).resolve() != path.resolve():
        raise RuntimeError(f"Seal path mismatch: {path}")
    verify(pin)
    return json.loads(path.read_text())


def publish_once(path: Path, data: dict) -> None:
    value = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise RuntimeError(f"Preserve existing completion artifact: {path}")
    else:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != value:
                    raise RuntimeError(f"Concurrent completion artifact changed: {path}") from None
        finally:
            temporary.unlink()


def predecessor_succeeded(path: Path) -> bool:
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    if result.get("status") != "JOB_FINISHED" or result.get("exit_code") != 0:
        raise RuntimeError("Grid job failed; preserve its logs and resolve the operational failure")
    return True


def freeze() -> dict:
    files = [Path(__file__), ROOT / "tools/audit_final_v14_grid_results.py",
             ROOT / "tools/final_v14_grid_benchmark.py", ROOT / "tools/final_v14_complete_report.py",
             ROOT / "tools/final_v14_bundle_receipt.py",
             ROOT / "tests/test_final_v14_finish_after_grid.py",
             ROOT / "tests/test_final_v14_complete_report.py",
             ROOT / "tests/test_final_v14_bundle_receipt.py", HANDOFF / "HANDOFF.md",
             GRID / "extraction_contract.json", GRID / "selected_patient_slide_roster.json",
             GRID / "independent_audit/prerequisites.json"]
    data = {"status": "FINAL_COMPLETION_HANDOFF_SEALED", "implementation_and_inputs": [identity(p) for p in files],
            "predecessor_resource_summary": str(RESOURCE),
            "stages": ["require_grid_success", "independent_grid_audit", "final_grid_benchmark",
                       "render_complete_report", "audit_complete_bundle", "seal_complete_bundle", "verify_complete_bundle"],
            "authorization": "User authorized completion of remaining Aim 4 analysis; coordinator reviewed the concrete pipeline, numerical guards and report templates before freezing this handoff.",
            "scientific_changes": "None. No new feature extraction, resampling, fitting, roster substitution or parameter choice occurs here."}
    path = HANDOFF / "handoff.json"
    publish_once(path, data)
    publish_once(Path(str(path) + ".seal.json"), identity(path))
    return data


def run() -> dict:
    contract = read_sealed(HANDOFF / "handoff.json")
    for pin in contract["implementation_and_inputs"]:
        verify(pin)
    env = dict(os.environ, MPLCONFIGDIR="/tmp/v14-mpl", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def command(stage: str, args: list[str | Path]) -> None:
        for pin in contract["implementation_and_inputs"]:
            verify(pin)
        print(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "stage": stage,
                          "command": [str(x) for x in args]}), flush=True)
        subprocess.run([str(x) for x in args], cwd=ROOT, env=env, check=True)

    verifier = ROOT / "tools/final_v14_bundle_receipt.py"
    receipt = REPORT / "final_bundle_receipt.json"
    if receipt.exists():
        # Also permits interrupted publication to recover a missing last seal.
        if not Path(str(receipt) + ".seal.json").exists():
            command("resume_final_seal", [PYTHON, verifier, "seal", "--authorize-final"])
        command("verify_existing_bundle", [PYTHON, verifier, "verify"])
    else:
        while not predecessor_succeeded(RESOURCE):
            print(json.dumps({"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "stage": "WAITING_FOR_EXISTING_GRID_JOB"}), flush=True)
            time.sleep(30)
        result = read_sealed(GRID / "results.json")
        if result.get("status") != "GRID_OFFSET_SENSITIVITY_COMPLETE":
            raise RuntimeError("This sealed 100-patient execution must finish the actual grid sensitivity")
        audit = GRID / "independent_audit/results.json"
        if audit.exists():
            existing = read_sealed(audit)
            verify(existing["implementation"])
            if existing.get("status") != "GRID_OFFSET_INDEPENDENT_AUDIT_PASS" or existing["results"] != identity(GRID / "results.json"):
                raise RuntimeError("Existing grid audit does not cover the actual result")
        else:
            command("independent_grid_audit", [BASE, ROOT / "tools/audit_final_v14_grid_results.py", "audit"])
        command("final_grid_benchmark", [PYTHON, ROOT / "tools/final_v14_grid_benchmark.py", "final"])
        # Never regenerate timestamp-bearing report files after partial sealing.
        if not (REPORT / "source_manifest.json").exists():
            command("render_complete_report", [PYTHON, ROOT / "tools/final_v14_complete_report.py"])
        command("audit_complete_bundle", [PYTHON, verifier, "audit"])
        command("seal_complete_bundle", [PYTHON, verifier, "seal", "--authorize-final"])
        command("verify_complete_bundle", [PYTHON, verifier, "verify"])
    result = {"status": "FINAL_V14_COMPLETION_VERIFIED", "handoff": identity(HANDOFF / "handoff.json"),
              "final_bundle_receipt": identity(receipt), "report": str(REPORT / "Results.md")}
    publish_once(HANDOFF / "completion.json", result)
    publish_once(HANDOFF / "completion.json.seal.json", identity(HANDOFF / "completion.json"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "run"])
    args = parser.parse_args()
    HANDOFF.mkdir(parents=True, exist_ok=True)
    with (HANDOFF / "execution.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = freeze() if args.stage == "freeze" else run()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
