"""Record sampled resource use around an unchanged, persistent analysis script."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import psutil


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def gpu_used() -> int | None:
    result = subprocess.run(
        ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    return int(result.stdout.strip().splitlines()[0]) * 1024**2 if result.returncode == 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started, start_clock = utc(), time.monotonic()
    baseline = gpu_used() if args.gpu else None
    child = subprocess.Popen(["bash", str(args.script.resolve())])
    process = psutil.Process(child.pid)
    maximum_rss, maximum_gpu = 0, baseline
    samples = args.output / "resource_samples.jsonl"
    with samples.open("x") as handle:
        while True:
            rss, hwm = 0, 0
            try:
                children = [process, *process.children(recursive=True)]
            except psutil.NoSuchProcess:
                children = []
            for item in children:
                try:
                    rss += item.memory_info().rss
                    for line in Path(f"/proc/{item.pid}/status").read_text().splitlines():
                        if line.startswith("VmHWM:"):
                            hwm = max(hwm, int(line.split()[1]) * 1024)
                except (psutil.NoSuchProcess, ProcessLookupError, FileNotFoundError):
                    pass
            used = gpu_used() if args.gpu else None
            maximum_rss = max(maximum_rss, rss)
            if used is not None:
                maximum_gpu = max(maximum_gpu or 0, used)
            record = {"utc": utc(), "elapsed_seconds": time.monotonic() - start_clock,
                      "root_pid": child.pid, "process_tree_rss_bytes": rss,
                      "largest_process_vm_hwm_bytes": hwm, "device_used_memory_bytes": used}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if child.poll() is not None:
                break
            time.sleep(2)
    result = {"status": "JOB_FINISHED", "exit_code": child.returncode,
              "started_utc": started, "finished_utc": utc(),
              "elapsed_seconds": time.monotonic() - start_clock, "script": str(args.script.resolve()),
              "process_tree_rss_sampled_peak_bytes": maximum_rss,
              "device_used_memory_baseline_bytes": baseline,
              "device_used_memory_sampled_peak_bytes": maximum_gpu,
              "child_rusage_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
              "nominal_sample_interval_seconds": 2,
              "measurement_limits": "RSS is summed over the live process tree. GPU is total device usage, including baseline/other processes. Sampled peaks can miss transients; rusage is the OS-reported child maximum, not a sum of simultaneous workers."}
    temporary = args.output / "resource_summary.pending.json"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, args.output / "resource_summary.json")
    print(json.dumps(result), flush=True)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
