#!/usr/bin/env python3
"""Verify every headline number quoted in Results.md against its frozen artifact.

Results.md is written by hand from experiment output. This re-reads the JSON each
table was built from and asserts the document still matches it, so a stale number
cannot survive a rerun unnoticed. Aim 2 invariants live in `tools/e2_audit.py`;
this file checks the VALUES, that one checks the DESIGN.

Usage:
    python tools/results_verify.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from oceanpath.aim1 import paths  # noqa: E402

R = paths.EVAL_ROOT


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
FAIL: list[str] = []


def chk(label: str, got: float | None, doc: float, tol: float = 6e-4) -> None:
    good = got is not None and abs(got - doc) <= tol
    print(f"  [{'PASS' if good else 'FAIL'}] {label:56s} "
          f"artifact {'—' if got is None else f'{got:9.4f}'}   doc {doc:9.4f}")
    if not good:
        FAIL.append(label)


def chk_str(label: str, got: str | None, doc: str) -> None:
    good = got == doc
    print(f"  [{'PASS' if good else 'FAIL'}] {label:56s} "
          f"artifact {str(got):>9s}   doc {doc:>9s}")
    if not good:
        FAIL.append(label)


def chk_bool(label: str, got: bool, doc: bool = True) -> None:
    good = bool(got) is doc
    print(f"  [{'PASS' if good else 'FAIL'}] {label:56s} "
          f"artifact {str(bool(got)):>9s}   doc {str(doc):>9s}")
    if not good:
        FAIL.append(label)


def main() -> None:
    print(f"\n{'=' * 96}\nAIM 1 — Table 1.1 · E0 locked baseline (pb_cap8192)\n{'=' * 96}")
    d = load(R / "e0_e1a_seed_report.json")
    b = d["pb_cap8192"]
    ps, seeds = b["per_seed"], sorted(b["per_seed"])
    for s, doc in (("42", 0.6731), ("43", 0.6582), ("44", 0.6643)):
        chk(f"seed {s} AUROC", ps[s]["auroc"], doc)
    chk("median AUROC", float(np.median([ps[s]["auroc"] for s in seeds])), 0.6643)
    chk("seed 42 AUPRC", ps["42"]["auprc"], 0.5906)
    chk("seed 42 Brier", ps["42"]["brier"], 0.2210)
    chk("seed 42 calibration slope", ps["42"]["calibration_slope"], 0.9377)
    chk("patient-bootstrap SE", b["uncertainty"]["se_patient_bootstrap"], 0.0138)
    chk("seed SD", b["uncertainty"]["sd_across_seeds"], 0.0075)
    chk("3-seed ensemble AUROC", b["ensemble"]["auroc"], 0.6795)
    chk("ensemble CI low", b["ensemble"]["ci_low"], 0.6526)
    chk("ensemble CI high", b["ensemble"]["ci_high"], 0.7066)
    for c, doc in (("CPTAC", 0.5892), ("RIH", 0.6306), ("SurGen", 0.6569), ("TCGA", 0.6922)):
        chk(f"per-cohort {c} (median of seeds)",
            float(np.median([ps[s]["per_cohort"][c] for s in seeds])), doc)
    F = np.array([ps[s]["per_fold"] for s in seeds])
    for i, doc in enumerate((0.6403, 0.6938, 0.6735, 0.7217, 0.6265)):
        chk(f"per-fold f{i} (median of seeds)", float(np.median(F[:, i])), doc)

    print(f"\n{'=' * 96}\nAIM 2 — Tables 2.1–2.6\n{'=' * 96}")
    t = load(R / "e2a_transport_pb_cap8192.json")
    for k, doc in (("CPTAC", 0.7308), ("RIH", 0.7522), ("SurGen", 0.6899), ("TCGA", 0.6778)):
        chk(f"E2a {k} AUROC", t["targets"][k]["primary_overall"]["auroc"], doc)
    chk("E2a four-target macro", t["macro"]["four_target_macro_auroc"], 0.7111)
    chk("E2a SR1482 stratum", t["targets"]["SurGen"]["primary_strata"]["SR1482"]["auroc"], 0.6271)
    chk("E2a SR386 stratum", t["targets"]["SurGen"]["primary_strata"]["SR386"]["auroc"], 0.7369)
    for k, doc in (("CPTAC", 0.7145), ("RIH", 0.7823), ("SurGen", 0.7397), ("TCGA", 0.6943)):
        chk(f"E2a {k} set-D AUROC", t["targets"][k]["primary_D_subset"]["auroc"], doc)
    for k, doc in (("CPTAC", 0.5963), ("RIH", 0.6218), ("SurGen", 0.6269), ("TCGA", 0.6408)):
        chk(f"E2a {k} calibrated log loss",
            t["targets"][k]["source_calibrated"]["log_loss_source_calibrated"], doc)
    diag = {k: v["failure_mode"] for k, v in t["diagnosis"].items()}
    all_cal = set(diag.values()) == {"calibration"}
    print(f"  [{'PASS' if all_cal else 'FAIL'}] all four targets diagnose as calibration"
          f"{'':21s} {diag}")
    if not all_cal:
        FAIL.append("E2a diagnosis")

    e = load(R / "e2b_metastatic_cap8192.json")
    for k, doc in (("RIH", -0.1075), ("SurGen", -0.0498)):
        chk(f"E2b {k} delta AUROC", e["targets"][k]["primary_vs_metastatic"]["delta_auroc"], doc)
        chk(f"E2b {k} metastatic AUROC", e["targets"][k]["metastatic_overall"]["auroc"],
            0.6278 if k == "RIH" else 0.5773)
    chk("E2b metastatic macro", e["conclusion"]["metastatic_macro_auroc"], 0.6025)
    chk("E2b macro CI low", e["conclusion"]["metastatic_macro_ci"][0], 0.5037)
    negs = [ms[s] - ps_[s] < 0 for k, v in e["targets"].items()
            for ps_, ms in [(v["primary_per_seed_auroc"], v["metastatic_per_seed_auroc"])]
            for s in ps_]
    print(f"  [{'PASS' if all(negs) else 'FAIL'}] all 6 seed-level E2b deltas negative"
          f"{'':25s} {sum(negs)}/6")
    if not all(negs):
        FAIL.append("E2b seed concordance")

    c = load(paths.OUTPUT_ROOT / "e2c" / "e2c_head_adaptation_cap8192.json")
    for k, doc in (("RIH", 0.6278), ("SurGen", 0.5773)):
        chk(f"E2c {k} S0 == E2b metastatic", c["cohorts"][k]["S0"]["auroc"], doc)
    chk("E2c SurGen best arm (S2 k=8)", c["cohorts"]["SurGen"]["arms"]["S2_k8"]["mean_auroc"],
        0.6152)
    worst = max(abs(v["delta"]) for k in c["cohorts"]
                for kk, v in c["cohorts"][k]["contrasts"].items() if "S2_minus_S1" in kk)
    print(f"  [{'PASS' if worst < 0.015 else 'FAIL'}] every S2−S1 contrast inside ±0.015"
          f"{'':22s} max |Δ| {worst:.4f}")
    if worst >= 0.015:
        FAIL.append("E2c endpoint magnitude")

    s1 = load(R / "e2d1_metastatic_sites_cap8192.json")
    chk("E2d-1 SurGen liver AUROC",
        s1["cohorts"]["SurGen"]["sites"]["liver"]["auroc"], 0.8026)
    chk("E2d-1 SurGen peritoneum AUROC",
        s1["cohorts"]["SurGen"]["sites"]["peritoneum"]["auroc"], 0.1250)
    chk("E2d-1 SurGen liver−non-liver",
        s1["cohorts"]["SurGen"]["liver_vs_non_liver"]["delta"], 0.4341)
    chk("E2d-1 RIH liver−non-liver",
        s1["cohorts"]["RIH"]["liver_vs_non_liver"]["delta"], 0.0098)

    s2 = load(R / "e2d2_peritoneal_audit_cap8192.json")
    for s, doc in (("42", 0.2500), ("43", 0.1000), ("44", 0.1250)):
        chk(f"E2d-2 seed {s} peritoneum AUROC", s2["per_seed_auroc"][s], doc)
    chk("E2d-2 LOO min", s2["loo"]["min"], 0.0333)
    chk("E2d-2 LOO max", s2["loo"]["max"], 0.1667)
    chk("E2d-2 BRAF-WT-only AUROC", s2["braf_stratified"]["braf_wt_only"]["auroc"], 0.2222)

    s3 = load(R / "e2d3_setd_contrast_cap8192.json")
    for k, doc in (("RIH", -0.1844), ("SurGen", -0.0681)):
        chk(f"E2d-3 {k} set-D delta", s3["cohorts"][k]["set_d"]["delta_auroc"], doc)
    lo = s3["cohorts"]["RIH"]["set_d"]["delta_auroc_ci"][1]
    print(f"  [{'PASS' if lo < 0 else 'FAIL'}] E2d-3 RIH set-D CI excludes zero"
          f"{'':29s} upper {lo:+.4f}")
    if lo >= 0:
        FAIL.append("E2d-3 RIH CI")

    g = load(R / "e2a_surgen_gap_cap8192.json")
    chk("E2d SurGen gap, crude", g["delta_crude"]["delta"], 0.1098)
    chk("E2d SurGen gap, set D", g["D_subset"]["delta"]["delta"], 0.0713)

    print(f"\n{'=' * 96}\nAIM 3 — Tables 3.1 / 3.2 · E3 molecular-resolution ladder\n{'=' * 96}")
    e3 = load(R / "e3a_resolution.json")

    # Table 3.1 - every rung and every matched control, as documented
    LADDER = {
        "gene":            (0.6795, 0.6526, 0.7066, 0.6582, 0.6731),
        "codon":           (0.5266, 0.4750, 0.5783, 0.5142, 0.5294),
        "g12d_broad":      (0.4864, 0.4368, 0.5340, 0.4791, 0.4915),
        "allele1":         (0.5253, 0.4717, 0.5800, 0.5084, 0.5343),
        "allele2":         (0.5491, 0.4902, 0.6087, 0.5301, 0.5886),
        "g12c":            (0.5539, 0.4680, 0.6390, 0.5022, 0.5764),
        "ctrl_codon":      (0.6580, 0.6115, 0.7043, 0.6351, 0.6646),
        "ctrl_g12d_broad": (0.6533, 0.6040, 0.7011, 0.6375, 0.6399),
        "ctrl_allele1":    (0.6439, 0.5895, 0.6958, 0.6177, 0.6478),
        "ctrl_allele2":    (0.6336, 0.5746, 0.6907, 0.5990, 0.6352),
        "ctrl_g12c":       (0.5547, 0.4653, 0.6440, 0.5182, 0.5607),
    }
    for task, (auroc, lo, hi, smin, smax) in LADDER.items():
        e = e3["tasks"][task]["ensemble"]
        chk(f"E3 {task} ensemble AUROC", e["auroc"], auroc)
        chk(f"E3 {task} CI low", e["ci"][0], lo)
        chk(f"E3 {task} CI high", e["ci"][1], hi)
        sr = e3["tasks"][task]["seed_range"]
        chk(f"E3 {task} seed min", sr[0], smin)
        chk(f"E3 {task} seed max", sr[1], smax)

    # every fine rung is 3-seed; a 2-seed ensemble silently changes the headline
    for task in LADDER:
        n = e3["tasks"][task]["n_seeds"]
        good = n == 3
        print(f"  [{'PASS' if good else 'FAIL'}] E3 {task} is a 3-seed ensemble"
              f"{'':{max(0, 32 - len(task))}s} artifact {n} seeds   doc 3 seeds")
        if not good:
            FAIL.append(f"E3 {task} n_seeds")

    # Table 3.2 - the direct contrast and the verdict it produces
    DELTA = {
        "codon":      (+0.1315, +0.0612, +0.1988, 1.000, "CEILING"),
        "g12d_broad": (+0.1669, +0.0943, +0.2350, 1.000, "CEILING"),
        "allele1":    (+0.1187, +0.0425, +0.1961, 0.999, "CEILING"),
        "allele2":    (+0.0845, -0.0003, +0.1685, 0.974, "INCONCLUSIVE"),
        "g12c":       (+0.0008, -0.1236, +0.1223, 0.499, "UNDERPOWERED"),
    }
    for rung, (delta, lo, hi, pgt, verdict) in DELTA.items():
        v = e3["verdict"][rung]
        chk(f"E3 {rung} delta ctrl-fine", v["delta"]["delta"], delta)
        chk(f"E3 {rung} delta CI low", v["delta"]["ci"][0], lo)
        chk(f"E3 {rung} delta CI high", v["delta"]["ci"][1], hi)
        chk(f"E3 {rung} P(delta > 0)", v["delta"]["p_delta_gt_0"], pgt)
        chk_str(f"E3 {rung} verdict", v["verdict"], verdict)

    # the two claims the Aim-3 verdict rests on, asserted directly
    ug = e3["tasks"]["ctrl_g12c"]["ensemble"]["ci"][0]
    print(f"  [{'PASS' if ug < 0.50 else 'FAIL'}] E3 ctrl_g12c lower CI below 0.50 (underpowered)"
          f"{'':11s} lower {ug:+.4f}")
    if ug >= 0.50:
        FAIL.append("E3 ctrl_g12c underpowered")
    a2lo = e3["verdict"]["allele2"]["delta"]["ci"][0]
    print(f"  [{'PASS' if a2lo <= 0 else 'FAIL'}] E3 allele2 delta CI touches zero (inconclusive)"
          f"{'':11s} lower {a2lo:+.4f}")
    if a2lo > 0:
        FAIL.append("E3 allele2 boundary")

    g12c = e3["g12c_descriptive"]
    chk("E3 G12C descriptive mean P(KRAS)", g12c["mean_score"], 0.5337)
    chk("E3 G12C descriptive, all others", g12c["mean_other"], 0.3572)
    chk("E3 G12C descriptive, wild-type", g12c["mean_wt"], 0.2623)

    print(f"\n{'=' * 96}\nAIM 4 — Tables 4.1–4.3 · E4 morphology atlas\n{'=' * 96}")
    e4 = load(R / "e3b_atlas_k32.json")
    m04 = e4["readout"]["17"]
    m05 = e4["readout"]["9"]
    m07 = e4["readout"]["28"]
    m11 = e4["readout"]["5"]

    chk_str("E4 M04 outcome", m04["group"],
            "dependency_robust_transport_underpowered")
    chk_str("E4 M07 outcome", m07["group"],
            "dependency_robust_transport_inconclusive")
    chk_str("E4 M11 outcome", m11["group"], "not_kras_associated")
    chk_str("E4 M05 outcome", m05["group"], "shortcut_technical")
    for montage, row, documented in (
        ("M04", m04, {
            "auc_A": 0.5605, "auc_D": 0.5939, "attention_A": 0.5040,
            "context_A": 0.6724,
        }),
        ("M07", m07, {
            "auc_A": 0.5968, "auc_D": 0.6012, "attention_A": 0.6172,
            "context_A": 0.4974,
        }),
    ):
        effects = row["specificity_effects"]
        chk(f"E4 {montage} abundance A AUC", effects["abundance"]["A"]["auc"],
            documented["auc_A"])
        chk(f"E4 {montage} abundance D AUC", effects["abundance"]["D"]["auc"],
            documented["auc_D"])
        chk(f"E4 {montage} attention A AUC",
            effects["attn_mass_mean"]["A"]["auc"], documented["attention_A"])
        chk(f"E4 {montage} context-in-WT AUC",
            effects["abundance"]["context_in_wt"]["auc"], documented["context_A"])

    for field, documented in (
        ("auc", 0.487701), ("ci_low", 0.458108),
        ("ci_high", 0.515602), ("q", 0.694580),
    ):
        chk(f"E4 M11 abundance A {field}",
            m11["specificity_effects"]["abundance"]["A"][field], documented)
    for field, documented in (
        ("auc", 0.487729), ("ci_low", 0.453688),
        ("ci_high", 0.520084), ("q", 0.711144),
    ):
        chk(f"E4 M11 abundance D {field}",
            m11["specificity_effects"]["abundance"]["D"][field], documented)
    for arm, documented in (
        ("A", (0.454433, 0.426846, 0.482753, 0.022616)),
        ("D", (0.448746, 0.415396, 0.480580, 0.018893)),
    ):
        effect = m11["specificity_effects"]["attn_mass_mean"][arm]
        chk(f"E4 M11 attention {arm} AUC", effect["auc"], documented[0])
        chk(f"E4 M11 attention {arm} CI low", effect["ci_low"], documented[1])
        chk(f"E4 M11 attention {arm} CI high", effect["ci_high"], documented[2])
        chk(f"E4 M11 attention {arm} q", effect["q"], documented[3])
        chk_bool(f"E4 M11 attention {arm} significant", effect["significant"])
    chk_str("E4 M11 attention direction",
            m11["attention_association"]["direction"], "higher in wild-type")
    chk("E4 M11 attention cohort directions agreeing",
        m11["attention_cohort_reproducibility"]["agreeing"], 4, tol=0)
    chk("E4 M11 attention cohorts evaluated",
        m11["attention_cohort_reproducibility"]["n_cohorts"], 4, tol=0)
    chk_str("E4 M11 attention cohort direction",
            m11["attention_cohort_reproducibility"]["direction"], "higher_in_wt")
    chk_bool("E4 M11 abundance A is null",
             not m11["specificity_effects"]["abundance"]["A"]["significant"])
    chk_bool("E4 M11 abundance D is null",
             not m11["specificity_effects"]["abundance"]["D"]["significant"])
    chk("E4 M05 abundance A AUC",
        m05["specificity_effects"]["abundance"]["A"]["auc"], 0.529206)
    chk("E4 M05 abundance A q",
        m05["specificity_effects"]["abundance"]["A"]["q"], 0.069221)
    chk_bool("E4 M05 carries a source flag", bool(m05["shortcut_flags"]))

    for prototype, label, direction in (
        (20, "internal 20", "higher in wild-type"),
        (26, "internal 26", "higher in mutants"),
        (28, "M07", "higher in mutants"),
    ):
        chk_str(f"E4 {label} attention direction",
                e4["readout"][str(prototype)]["attention_association"]["direction"],
                direction)

    for montage, row, documented in (
        ("M04", m04, {"RIH": (0.1393, 0.0037, 0.2733, 0.4700),
                    "SR1482": (-0.0848, -0.2174, 0.0529, 0.9638)}),
        ("M07", m07, {"RIH": (-0.0412, -0.1706, 0.1009, 0.9821),
                    "SR1482": (-0.0073, -0.1305, 0.1140, 0.9638)}),
    ):
        for cohort, (delta, low, high, q) in documented.items():
            effect = row["conservation"]["by_cohort"][cohort]
            chk(f"E4 {montage} {cohort} direct delta",
                effect["delta_auc_metastatic_minus_primary"], delta)
            chk(f"E4 {montage} {cohort} delta CI low", effect["delta_ci_low"], low)
            chk(f"E4 {montage} {cohort} delta CI high", effect["delta_ci_high"], high)
            chk(f"E4 {montage} {cohort} delta q", effect["delta_q"], q)

    review = e4["pathology_review"]
    chk("E4 pathology schema version", review["schema_version"], 2, tol=0)
    chk_str("E4 pathology review status", review["status"], "complete")
    chk("E4 unique reviewed montage count", review["n_reviewed"], 11, tol=0)
    chk("E4 base reviewed montage count", review["base_packet"]["n_reviewed"], 10, tol=0)
    chk("E4 targeted follow-up assessment count",
        review["followup"]["n_assessments"], 4, tol=0)
    chk_str("E4 follow-up extraction status", review["followup"]["extraction_status"],
            "curated_from_completed_blinded_followup")
    annotations = review["annotations"]
    montage_map = review["montage_by_prototype"]
    for prototype, montage in ((17, "M04"), (9, "M05"), (28, "M07"), (5, "M11")):
        chk_str(f"E4 {montage} key mapping", montage_map[str(prototype)], montage)
        chk_str(f"E4 {montage} readout mapping",
                e4["readout"][str(prototype)]["montage_id"], montage)

    chk_str("E4 M04 canonical pathology", annotations["17"]["canonical_description"],
            "Extracellular mucin pools, variably acellular or containing floating "
            "tumour-cell clusters, consistent with focal mucinous differentiation.")
    chk_str("E4 M04 dirty necrosis", annotations["17"]["dirty_necrosis"], "absent")
    chk_str("E4 M05 focal dirty necrosis", annotations["9"]["dirty_necrosis"],
            "focal in tiles 2-4 and 3-3")
    chk_str("E4 M07 space interpretation", annotations["28"]["space_interpretation"],
            "All spaces bordering tumour are true glandular lumina and an intrinsic "
            "architectural feature, not tissue edge or processing artefact.")
    chk_str("E4 M07 artifact concern", annotations["28"]["artifact_concern"], "none")
    chk_str("E4 M11 tissue compartment", annotations["5"]["tissue_compartment"],
            "predominantly normal or non-tumour colonic epithelium")
    chk_str("E4 M11 artifact concern", annotations["5"]["artifact_concern"], "none")
    chk_bool("E4 M11 review discrepancies retained",
             bool(annotations["5"]["quality_flags"]))

    review_form = REPO / "reviews" / "k32" / "review_form.csv"
    structured_review = REPO / "reviews" / "k32" / "completed_review_structured.json"
    raw_followup = REPO / "reviews" / "k32" / "completed_review.md"
    review_key = paths.OUTPUT_ROOT / "e3b" / "montages" / "k32" / "KEY_do_not_open_before_review.csv"
    addendum_key = paths.OUTPUT_ROOT / "e3b" / "montages" / "k32" / review["followup"]["addendum_key"]
    chk_str("E4 review-form checksum", review["form_sha256"],
            hashlib.sha256(review_form.read_bytes()).hexdigest())
    chk_str("E4 base unblinding-key checksum", review["key_sha256"],
            hashlib.sha256(review_key.read_bytes()).hexdigest())
    chk_str("E4 raw follow-up checksum", review["followup"]["source_sha256"],
            hashlib.sha256(raw_followup.read_bytes()).hexdigest())
    chk_str("E4 structured review checksum", review["followup"]["structured_sha256"],
            hashlib.sha256(structured_review.read_bytes()).hexdigest())
    chk_str("E4 addendum-key checksum", review["followup"]["addendum_key_sha256"],
            hashlib.sha256(addendum_key.read_bytes()).hexdigest())
    for montage, documented in (
        ("M04", "d706f5cecfb327f422dfd2ee66d618965bb896dfbf7b35b313f1c8b924979711"),
        ("M05", "0e8005a1a0ab4df2c878db6e131624dbc70d6cbee7f220ff967c1c78f1735725"),
        ("M07", "afe3af8fe64887ddc88b08376749b14bbd52c13ee5f3091042523157179d0448"),
        ("M11", "02c1f33726e6fa0e8ad76b717621f94c51636bf00396ac16c08cc4364667f6c0"),
    ):
        chk_str(f"E4 {montage} montage checksum",
                review["followup"]["montage_sha256"][montage], documented)
    chk_bool("E4 follow-up quality flags retained",
             bool(review["followup"]["quality_flags"]))
    no_conserved = not e4["groups"]["dependency_robust_conserved"]
    no_changed = not e4["groups"]["dependency_robust_changed"]
    print(f"  [{'PASS' if no_conserved else 'FAIL'}] E4 no prototype meets strict conservation")
    if not no_conserved:
        FAIL.append("E4 conserved group")
    print(f"  [{'PASS' if no_changed else 'FAIL'}] E4 no prototype passes direct change test")
    if not no_changed:
        FAIL.append("E4 changed group")

    print(f"\n{'=' * 96}")
    if FAIL:
        print(f"{len(FAIL)} MISMATCH(ES):")
        for f in FAIL:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL DOCUMENTED NUMBERS MATCH THEIR ARTIFACTS")


if __name__ == "__main__":
    main()
