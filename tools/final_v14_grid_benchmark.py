"""Operational first/final grid resource receipts, separate from frozen science."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_final_v14_grid_results import DEFAULT, ROOT, identity, publish, sealed, verify  # noqa: E402

HANDOFF = DEFAULT.parent / "execution_handoff_20260905"
RESOURCES = HANDOFF / "resources_run1"


def timestamp(path: Path) -> dict:
    nanoseconds = path.stat().st_mtime_ns
    return {"mtime_ns": nanoseconds, "utc": datetime.fromtimestamp(nanoseconds/1e9,timezone.utc).isoformat()}


def numeric_time(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def source_tile_inventory(roster: dict) -> tuple[list[dict],dict]:
    rows, totals = [], {}
    for row in roster["slides"]:
        path = verify(row["coordinates"])
        with h5py.File(path,"r") as handle:
            count = len(handle["coords"])
        rows.append({"slide_id":row["slide_id"],"patient_id":row["patient_id"],"subcohort":row["subcohort"],
                     "canonical_tiles":count,"canonical_coordinates":row["coordinates"]})
        totals[row["subcohort"]] = totals.get(row["subcohort"],0)+count
    return rows, totals


def first(output: Path, resources: Path) -> dict:
    path = output / "benchmarks/first_slide.json"
    if path.exists():
        return sealed(path)
    contract = sealed(output / "extraction_contract.json")
    roster = sealed(verify(contract["roster"]))
    row = roster["slides"][0]
    directory = output / "slides" / row["slide_id"]
    slide_path = directory / "receipt.json"
    receipt = sealed(slide_path)
    assert receipt["status"] == "GRID_OFFSET_SLIDE_COMPLETE"
    assert receipt["slide_id"] == row["slide_id"] and receipt["contract_sha256"] == identity(output / "extraction_contract.json")["sha256"]
    start_path = directory / "feature_start.json"
    start = sealed(start_path)
    for item in receipt["artifacts"] + [start["coordinates"]]:
        verify(item)
    source_rows, totals = source_tile_inventory(roster)
    original_total = sum(totals.values())
    original_first = next(r["canonical_tiles"] for r in source_rows if r["slide_id"] == row["slide_id"])
    start_time, finish_time = timestamp(start_path), timestamp(slide_path)
    elapsed = (finish_time["mtime_ns"]-start_time["mtime_ns"])/1e9
    assert elapsed > 0
    tiles = receipt["tile_count"]
    throughput = tiles/elapsed
    # Freeze only complete flushed lines through the immutable first-slide receipt.
    prefix, parsed = [], []
    for line in (resources / "resource_samples.jsonl").read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            continue
        value = json.loads(line)
        if numeric_time(value["utc"]) <= finish_time["mtime_ns"]/1e9:
            prefix.append(line)
            parsed.append(value)
    assert parsed
    prefix_path = output / "benchmarks/resource_samples_through_first_slide.jsonl"
    prefix_path.parent.mkdir(parents=True,exist_ok=True)
    payload = b"".join(prefix)
    if prefix_path.exists():
        assert prefix_path.read_bytes() == payload
    else:
        prefix_path.write_bytes(payload)
    extraction_samples = [sample for sample in parsed if numeric_time(sample["utc"]) >= start_time["mtime_ns"]/1e9]
    gpu = [r["device_used_memory_bytes"] for r in parsed if r["device_used_memory_bytes"] is not None]
    baseline = parsed[0]["device_used_memory_bytes"]
    first_ratio = tiles/original_first
    estimated_offset_tiles = original_total*first_ratio
    result = {
        "status":"GRID_OFFSET_FIRST_SLIDE_BENCHMARK_SEALED","created_utc":datetime.now(timezone.utc).isoformat(),
        "implementation":[identity(Path(__file__)),identity(ROOT / "tools/audit_final_v14_grid_results.py"),identity(ROOT / "tools/final_v14_job_resource_monitor.py")],
        "contract":identity(output / "extraction_contract.json"),"roster":contract["roster"],
        "first_slide":{"slide_id":row["slide_id"],"feature_start":identity(start_path),"receipt":identity(slide_path),
                       "artifacts":receipt["artifacts"],"feature_start_filesystem_time":start_time,"receipt_filesystem_time":finish_time,
                       "offset_tiles":tiles,"canonical_tiles":original_first,"elapsed_seconds":elapsed,"tiles_per_second":throughput},
        "timing_method":"Difference of immutable feature_start.json and completed receipt.json filesystem modification timestamps; includes extraction, saving, quantization, assignment, and receipt work. Not an instrumented encoder-only timer.",
        "canonical_roster_inventory":{"slides":len(source_rows),"patients":len(roster["patients"]),"tiles":original_total,"tiles_by_subcohort":totals,"rows":source_rows},
        "resource_prefix":identity(prefix_path),
        "resources":{"nominal_sample_interval_seconds":2,"samples":len(parsed),"samples_during_first_slide":len(extraction_samples),
                     "first_sample_utc":parsed[0]["utc"],"last_sample_utc":parsed[-1]["utc"],
                     "process_tree_rss_sampled_peak_bytes_so_far":max(r["process_tree_rss_bytes"] for r in parsed),
                     "first_slide_window_sampled_peak_rss_bytes":max([r["process_tree_rss_bytes"] for r in extraction_samples],default=None),
                     "global_gpu_memory_first_sample_bytes":baseline,"global_gpu_memory_sampled_peak_bytes_so_far":max(gpu) if gpu else None,
                     "global_gpu_peak_above_first_sample_bytes":max(gpu)-baseline if gpu and baseline is not None else None},
        "coarse_projection_at_first_slide":{"offset_equals_original_tile_count_assumption_total_seconds":original_total/throughput,
                     "offset_equals_original_tile_count_assumption_remaining_seconds":max(0,original_total-tiles)/throughput,
                     "first_slide_offset_to_original_count_ratio":first_ratio,
                     "first_slide_ratio_extrapolated_offset_tiles":estimated_offset_tiles,
                     "first_slide_ratio_extrapolated_total_seconds":estimated_offset_tiles/throughput,
                     "first_slide_ratio_extrapolated_remaining_seconds":max(0,estimated_offset_tiles-tiles)/throughput},
        "limits":"A single source slide gives a coarse projection only; tissue geometry, offset tile counts, WSI I/O, cohort, caching and workers vary. Projections are reconstructed from the first checkpoint after publication, not a contemporaneous timer forecast. RSS sums the process tree and may double-count shared pages; GPU is global device usage including other processes; 2-second sampling misses transients. The live resource JSONL is not pinned; only an immutable prefix copy is bound."
    }
    publish(path,result)
    return result


def final(output: Path, resources: Path, log: Path) -> dict:
    path = output / "benchmarks/final.json"
    if path.exists():
        return sealed(path)
    first_result = first(output,resources)
    result = sealed(output / "results.json")
    completion = sealed(output / "extraction_complete.json")
    assert result["status"] == "GRID_OFFSET_SENSITIVITY_COMPLETE"
    assert completion["status"] == "GRID_OFFSET_ALL_LISTED_SLIDES_COMPLETE"
    monitor = json.loads((resources / "resource_summary.json").read_text())
    assert monitor["status"] == "JOB_FINISHED" and monitor["exit_code"] == 0
    rosters = sealed(output / "selected_patient_slide_roster.json")
    expected = {r["slide_id"] for r in rosters["slides"]}
    completed, tiles, records = set(),0,[]
    for pin in completion["slide_receipts"]:
        receipt = sealed(verify(pin))
        assert receipt["slide_id"] not in completed and receipt["slide_id"] in expected
        assert receipt["status"] == "GRID_OFFSET_SLIDE_COMPLETE"
        completed.add(receipt["slide_id"])
        tiles += receipt["tile_count"]
        records.append({"receipt":pin,"mtime":timestamp(Path(pin["path"])),"tile_count":receipt["tile_count"]})
    assert completed == expected and len(completed) == 110
    start_ns = first_result["first_slide"]["feature_start_filesystem_time"]["mtime_ns"]
    last_ns = max(r["mtime"]["mtime_ns"] for r in records)
    elapsed = (last_ns-start_ns)/1e9
    final_result = {"status":"GRID_OFFSET_FINAL_BENCHMARK_SEALED","created_utc":datetime.now(timezone.utc).isoformat(),
                    "implementation":identity(Path(__file__)),"first_benchmark":identity(output / "benchmarks/first_slide.json"),
                    "results":identity(output / "results.json"),"completion":identity(output / "extraction_complete.json"),
                    "monitor_summary":identity(resources / "resource_summary.json"),"resource_samples":identity(resources / "resource_samples.jsonl"),
                    "full_log":identity(log),"resources":monitor,"slide_receipts":records,"patients":100,"slides":len(completed),
                    "offset_tiles":tiles,"canonical_tiles":first_result["canonical_roster_inventory"]["tiles"],
                    "all_slide_extraction_window_seconds_from_filesystem_timestamps":elapsed,
                    "all_slide_extraction_window_tiles_per_second":tiles/elapsed,
                    "timing_limits":"Entire wrapper timer includes the initial all-source eligibility preflight. Extraction window uses first feature_start through last completed slide receipt filesystem times, including each slide's I/O and scoring. The immutable finished monitor provides sampled peaks and OS child max RSS; GPU values are global, not per-job attribution."}
    publish(path,final_result)
    return final_result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["first","final"])
    parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--resources",type=Path,default=RESOURCES)
    parser.add_argument("--log",type=Path,default=HANDOFF / "grid_offset.log")
    args=parser.parse_args()
    value=first(args.output,args.resources) if args.stage=="first" else final(args.output,args.resources,args.log)
    print(json.dumps({"status":value["status"],"output":str(args.output / "benchmarks")},indent=2))
