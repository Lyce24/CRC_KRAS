"""Independent, read-only audit of a sealed grid-offset roster and final outputs.

Preparation audits the complete eligibility denominator and seeded roster.
Final auditing needs completed extraction and never launches an encoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports/reruns/final_v14_additions_20260903"
DEFAULT = RUN / "grid_offset/locked_extraction"
COLS = [f"prototype_{j:02}" for j in range(32)]


def identity(path: Path) -> dict:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(chunk)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def verify(item: dict) -> Path:
    path = Path(item["path"])
    actual = identity(path)
    assert all(actual[key] == item[key] for key in ["size_bytes", "sha256"]), path
    return path


def sealed(path: Path) -> dict:
    seal = json.loads(Path(str(path)+".seal.json").read_text())
    assert Path(seal.get("artifact", seal)["path"]).resolve() == path.resolve()
    verify(seal.get("artifact", seal))
    return json.loads(path.read_text())


def publish(path: Path, value: dict) -> None:
    assert not path.exists(), f"Preserve an already published audit: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    Path(str(path)+".seal.json").write_text(json.dumps(identity(path), indent=2)+"\n")


def close(actual, expected, atol=1e-12) -> float:
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    assert a.shape == b.shape
    assert np.allclose(a, b, rtol=0, atol=atol, equal_nan=True)
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0


def independent_icc(a: np.ndarray, b: np.ndarray) -> dict:
    """Independent ANOVA sums-of-squares form of absolute-agreement ICC(2,1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2:
        return {"status": "ICC_UNDEFINED_INSUFFICIENT_PATIENTS", "icc": None}
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"status": "ICC_UNDEFINED_ZERO_BETWEEN_PATIENT_VARIANCE", "icc": None}
    matrix = np.column_stack([a, b])
    centered = matrix-matrix.mean()
    ss_total = np.einsum("ij,ij->", centered, centered)
    ss_patient = 2 * np.square(centered.mean(axis=1)).sum()
    ss_grid = n * np.square(centered.mean(axis=0)).sum()
    ms_patient = ss_patient/(n-1)
    ms_grid = ss_grid
    ms_error = (ss_total-ss_patient-ss_grid)/(n-1)
    denominator = ms_patient+ms_error+2*(ms_grid-ms_error)/n
    if denominator == 0 or not np.isfinite(denominator):
        return {"status": "ICC_UNDEFINED_DENOMINATOR", "icc": None}
    return {"status": "ESTIMABLE", "icc": float((ms_patient-ms_error)/denominator)}


def prepare(output: Path) -> dict:
    audit_path = output / "independent_audit/prerequisites.json"
    if audit_path.exists():
        audit = sealed(audit_path)
        for key in ["implementation", "contract", "roster", "eligibility_audit"]:
            verify(audit[key])
        return audit
    contract_path = output / "extraction_contract.json"
    contract = sealed(contract_path)
    assert contract["status"] == "GRID_OFFSET_EXTRACTION_CONTRACT_SEALED"
    for item in contract["implementation"] + [contract[key] for key in ["roster", "uni_checkpoint", "canonical_mask_hest_checkpoint", "historical_receipt_schema_inventory", "reference_vocabulary", "original_profiles", "source_distance_quantiles"]]:
        verify(item)
    roster = sealed(Path(contract["roster"]["path"]))
    audit = sealed(verify(roster["eligibility_audit"]))
    assert audit["status"] == "ALL_SOURCE_INPUTS_PREFLIGHTED" and audit["outcome_columns_read"] == 0
    source = pd.read_csv(verify(audit["source_roster"]))
    verify(audit["ready_inventory"])
    audited = pd.DataFrame(audit["slides"])
    assert len(source) == len(audited) == 1389 and source.patient_id.nunique() == 1239
    keys = ["slide_id", "patient_id", "subcohort"]
    assert not audited.slide_id.duplicated().any()
    assert source[keys].sort_values("slide_id").reset_index(drop=True).equals(audited[keys].sort_values("slide_id").reset_index(drop=True))
    assert audited.eligible.map(lambda x: isinstance(x, bool)).all()
    eligibility = audited.groupby(["patient_id", "subcohort"], sort=True).eligible.all().reset_index()
    method = contract["method"]
    assert method["seed"] == roster["seed"] == 20260819
    assert method["patient_count"] == 100 and method["patients_per_subcohort"] == 25
    expected_groups = sorted(source.subcohort.unique().tolist())
    assert method["subcohort_order"] == expected_groups
    rng = np.random.Generator(np.random.PCG64(20260819))
    selected, counts = [], {}
    for subcohort in expected_groups:
        pool = eligibility.loc[eligibility.subcohort.eq(subcohort) & eligibility.eligible].sort_values("patient_id")
        counts[subcohort] = len(pool)
        assert len(pool) >= 25
        selected.extend(pool.iloc[rng.choice(len(pool), 25, replace=False)].patient_id.tolist())
    patients = pd.DataFrame(roster["patients"])
    assert len(patients) == len(set(selected)) == 100 and sorted(patients.patient_id) == sorted(selected)
    assert patients.groupby("subcohort").size().to_dict() == dict.fromkeys(expected_groups,25)
    selected_slides = pd.DataFrame(roster["slides"])
    expected = source[source.patient_id.isin(selected)][keys].sort_values("slide_id").reset_index(drop=True)
    assert selected_slides[keys].sort_values("slide_id").reset_index(drop=True).equals(expected)
    assert roster["substitutions_allowed"] is False and selected_slides.eligible.all()
    source_by_slide = {row["slide_id"]: row for row in audit["slides"]}
    for row in roster["slides"]:
        assert row == source_by_slide[row["slide_id"]]
        raw = Path(row["wsi"]["path"])
        assert raw.stat().st_size == row["wsi"]["size_bytes"] and raw.stat().st_mtime_ns == row["wsi"]["mtime_ns"]
        for pin in [row["mask"], row["coordinates"], *row["stream_receipts"]]:
            verify(pin)
    result = {"status": "GRID_OFFSET_INDEPENDENT_AUDIT_PRECONDITIONS_PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
              "implementation": identity(Path(__file__)), "contract": identity(contract_path),
              "roster": contract["roster"], "eligibility_audit": roster["eligibility_audit"],
              "all_source_slides": 1389, "all_source_patients": 1239, "eligible_patient_counts": counts,
              "selected_patients": 100, "selected_slides": len(selected_slides),
              "seeded_selection_reproduced": True, "selected_patient_all_listed_slides_verified": True,
              "final_audit_plan": "Every stage and slide artifact pin, all assignment counts/distances/support, exact equal-slide profile and ICC replay; deterministic up-to-128-tile feature-assignment and accepted/rejected polygon-area samples per selected slide. No encoder rerun."}
    publish(audit_path, result)
    return result


def sample_geometry(row: dict, coords: np.ndarray) -> dict:
    """Validate the entire grid lattice plus independent mask-area samples."""
    import geopandas as gpd
    from shapely import area, box, intersection, make_valid, union_all

    attrs = row["canonical_attributes"]
    f = int(attrs["patch_size_level0"])
    origin = f//2
    width, height = int(attrs["level0_width"]), int(attrs["level0_height"])
    assert f == round(128/float(attrs["level0_mpp"]))
    assert len(coords) == len(np.unique(coords,axis=0)) and len(coords) > 0
    assert ((coords-origin) % f == 0).all() and (coords >= origin).all()
    assert (coords[:,0]+f <= width).all() and (coords[:,1]+f <= height).all()
    assert list(map(tuple,coords)) == sorted(map(tuple,coords))
    all_x = np.arange(origin,width-f+1,f)
    all_y = np.arange(origin,height-f+1,f)
    accepted = set(map(tuple,coords))
    accepted_sample = coords[np.unique(np.linspace(0,len(coords)-1,min(128,len(coords)),dtype=int))]
    candidates = np.column_stack([np.repeat(all_x,len(all_y)),np.tile(all_y,len(all_x))])
    rejected = np.asarray([p for p in candidates if tuple(p) not in accepted],dtype=np.int64).reshape(-1,2)
    rejected_sample = rejected[np.unique(np.linspace(0,len(rejected)-1,min(128,len(rejected)),dtype=int))] if len(rejected) else rejected
    geometries = gpd.read_file(verify(row["mask"])).geometry
    union = make_valid(union_all([g for g in geometries if g is not None and not g.is_empty]))
    for points, should_accept in [(accepted_sample,True),(rejected_sample,False)]:
        if len(points):
            squares = box(points[:,0],points[:,1],points[:,0]+f,points[:,1]+f)
            measured = area(intersection(squares,union))
            assert ((measured >= .5*f*f) == should_accept).all()
    return {"accepted_mask_samples":len(accepted_sample),"rejected_mask_samples":len(rejected_sample),"all_lattice_coordinates_checked":len(coords)}


def audit(output: Path) -> dict:
    prerequisites = prepare(output)
    result_path = output / "results.json"
    result = sealed(result_path)
    assert result["status"] == "GRID_OFFSET_SENSITIVITY_COMPLETE"
    contract = sealed(verify(result["contract"]))
    assert result["contract"] == prerequisites["contract"]
    roster = sealed(verify(contract["roster"]))
    completion = sealed(verify(result["completion"]))
    assert completion["status"] == "GRID_OFFSET_ALL_LISTED_SLIDES_COMPLETE"
    assert completion["patient_count"] == 100 and completion["slide_count"] == prerequisites["selected_slides"]
    for pin in result["artifacts"]:
        verify(pin)
    vocab = np.load(verify(contract["reference_vocabulary"]))
    q99 = pd.read_csv(verify(contract["source_distance_quantiles"])).sort_values("prototype_id").q99_distance.to_numpy(float)
    expected = {row["slide_id"]: row for row in roster["slides"]}
    completed, details = {}, []
    for pin in completion["slide_receipts"]:
        receipt = sealed(verify(pin))
        slide_id = receipt["slide_id"]
        assert slide_id not in completed and slide_id in expected
        row = expected[slide_id]
        assert (receipt["patient_id"],receipt["subcohort"]) == (row["patient_id"],row["subcohort"])
        assert receipt["contract_sha256"] == result["contract"]["sha256"]
        artifacts = {Path(pin["path"]).name: pin for pin in receipt["artifacts"]}
        for pin in artifacts.values():
            verify(pin)
            stage = sealed(Path(pin["path"]+".receipt.json"))
            assert stage["artifact"] == pin and stage["contract_sha256"] == result["contract"]["sha256"]
        coords_pin = artifacts["offset_coordinates.h5"]
        feat_pin = artifacts[slide_id+".h5"]
        assignment_pin = artifacts["reference_assignments.npz"]
        start = sealed(Path(coords_pin["path"]).parent / "feature_start.json")
        assert start["coordinates"] == coords_pin and start["uni_sha256"] == contract["uni_checkpoint"]["sha256"]
        assert start["contract_sha256"] == result["contract"]["sha256"]
        with h5py.File(coords_pin["path"],"r") as file:
            coords = file["coords"][:]
            attrs = dict(file["coords"].attrs)
        assert attrs["canonical_coordinates_sha256"] == row["coordinates"]["sha256"]
        assert attrs["grid_offset_contract_sha256"] == result["contract"]["sha256"]
        details_row = {"slide_id":slide_id, **sample_geometry(row,coords)}
        assignments = np.load(assignment_pin["path"])
        labels, distances = assignments["labels"], assignments["distances"]
        assert assignments["feature_sha256"].item() == feat_pin["sha256"]
        assert assignments["contract_sha256"].item() == result["contract"]["sha256"]
        assert len(labels) == len(distances) == len(coords) == receipt["tile_count"]
        assert np.isin(labels,np.arange(32)).all() and np.isfinite(distances).all() and (distances >= 0).all()
        counts = np.bincount(labels,minlength=32)
        close(counts/len(labels),receipt["abundance"],0)
        for j,cell in enumerate(receipt["distance_support"]):
            assert cell["prototype_id"] == j and cell["assigned_tiles"] == counts[j]
            assert cell["nonempty_slides"] == int(counts[j] > 0)
            if counts[j]:
                values = distances[labels == j].astype(float)
                close(np.median(values),cell["median_distance"],0)
                close(np.mean(values > q99[j]),cell["fraction_beyond_source_q99"],0)
            else:
                assert cell["median_distance"] is None and cell["fraction_beyond_source_q99"] is None
        selected = np.unique(np.linspace(0,len(coords)-1,min(128,len(coords)),dtype=int))
        with h5py.File(feat_pin["path"],"r") as file:
            assert file["features"].shape == (len(coords),1024) and file["features"].dtype == np.dtype("float32")
            assert str(file["features"].attrs["encoder"]) == "uni_v1"
            assert np.array_equal(file["coords"][:],coords)
            x = file["features"][selected].astype(np.float16).astype(np.float32)
        projected = (x/np.linalg.norm(x,axis=1)[:,None]-vocab["pca_mean"]) @ vocab["pca_components"].T
        independent = cdist(projected.astype(float),vocab["centroids"].astype(float))
        nearest = independent.argmin(axis=1)
        assert np.array_equal(nearest,labels[selected])
        details_row["assignment_samples"] = len(selected)
        details_row["distance_sample_max_absolute_error"] = close(independent[np.arange(len(selected)),nearest],distances[selected],2e-6)
        details.append(details_row)
        completed[slide_id] = receipt
    assert set(completed) == set(expected)
    profiles = pd.read_csv(output / "offset_patient_profiles.csv",float_precision="round_trip").sort_values("patient_id")
    original = pd.read_parquet(verify(contract["original_profiles"])).set_index("patient_id").loc[profiles.patient_id].reset_index()
    assert len(profiles) == 100 and set(profiles.patient_id) == {row["patient_id"] for row in expected.values()}
    assert original.patient_id.tolist() == profiles.patient_id.tolist() and original.subcohort.tolist() == profiles.subcohort.tolist()
    assert np.array_equal(original.n_slides,profiles.n_slides)
    for patient in profiles.itertuples(index=False):
        slides = [r for r in completed.values() if r["patient_id"] == patient.patient_id]
        assert len(slides) == patient.n_slides
        close(np.mean([r["abundance"] for r in slides],axis=0),[getattr(patient,c) for c in COLS])
    a,b = original[COLS].to_numpy(),profiles[COLS].to_numpy()
    cosine = 1-np.diag(cdist(a,b,metric="cosine"))
    reported_cosine = pd.read_csv(output / "patient_cosine.csv",float_precision="round_trip").sort_values("patient_id")
    assert reported_cosine.patient_id.tolist() == profiles.patient_id.tolist()
    cosine_error = close(cosine,reported_cosine.cosine_similarity)
    close([cosine.mean(),np.median(cosine),cosine.min()],[result[k] for k in ["cosine_mean","cosine_median","cosine_min"]])
    iccs = pd.read_csv(output / "prototype_icc.csv",float_precision="round_trip").sort_values("prototype_id")
    assert iccs.prototype_id.tolist() == list(range(32))
    icc_errors = []
    for row in iccs.itertuples(index=False):
        j = row.prototype_id
        calculated = independent_icc(a[:,j],b[:,j])
        assert calculated["status"] == row.status and row.n_patients == 100
        assert row.n_nonzero_original == np.count_nonzero(a[:,j]) and row.n_nonzero_offset == np.count_nonzero(b[:,j])
        if calculated["icc"] is None:
            assert pd.isna(row.icc)
        else:
            icc_errors.append(close(calculated["icc"],row.icc,1e-10))
    support = pd.read_csv(output / "offset_patient_distance_support.csv",float_precision="round_trip").set_index(["patient_id","prototype_id"])
    for patient in profiles.itertuples(index=False):
        slides = [r for r in completed.values() if r["patient_id"] == patient.patient_id]
        for j in range(32):
            all_cells = [s["distance_support"][j] for s in slides]
            cells = [c for c in all_cells if c["assigned_tiles"]]
            row = support.loc[(patient.patient_id,j)]
            close([sum(c["assigned_tiles"] for c in all_cells),len(cells),
                   np.mean([c["median_distance"] for c in cells]) if cells else np.nan,
                   np.mean([c["fraction_beyond_source_q99"] for c in cells]) if cells else np.nan],
                  row[["assigned_tiles","nonempty_slides","median_distance","fraction_beyond_source_q99"]])
    report = {"status":"GRID_OFFSET_INDEPENDENT_AUDIT_PASS","created_utc":datetime.now(timezone.utc).isoformat(),
              "implementation":identity(Path(__file__)),"prerequisites":identity(output / "independent_audit/prerequisites.json"),
              "results":identity(result_path),"contract":result["contract"],"completion":result["completion"],
              "patients":100,"slides":len(details),"tile_count":sum(r["tile_count"] for r in completed.values()),
              "cosine_max_absolute_error":cosine_error,"icc_max_absolute_error":max(icc_errors,default=0),
              "slide_checks":details,"all_profiles_and_support_reconstructed":True,
              "icc_method":"Independent ANOVA sums-of-squares ICC(2,1), including undefined-zero-variance and support rules",
              "limits":"All lattice coordinates, saved assignments, distance/support cells, checkpoints and profiles checked; per-slide feature-assignment and mask-area recomputation use deterministic samples up to128 accepted and128 rejected locations. Encoder pixels/features are not rerun."}
    publish(output / "independent_audit/results.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["prepare","audit"])
    parser.add_argument("--output",type=Path,default=DEFAULT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = (prepare if args.stage == "prepare" else audit)(args.output.resolve())
    print(json.dumps({"status":result["status"],"output":str(args.output)},indent=2))
