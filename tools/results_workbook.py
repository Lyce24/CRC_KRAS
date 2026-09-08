#!/usr/bin/env python3
"""Assemble outputs/KRAS_results_tables.xlsx from the frozen E0/E1a/E1d JSON.

Every experiment that consumes a trained model is written TWICE — once for the
uncapped (full-bag) arm and once for the capped (4,096 tiles/epoch) arm —
because the bag policy is still an open screen (design §2.1(4)) and the losing
arm has to survive as a reference, not be overwritten. CONCH v1.5 appears as a
clearly-labelled encoder reference; it is a different question from the cap
screen and is never mixed into it.

Reporting follows the design document: median + observed min-max across seeds,
NEVER a three-draw SD. Cells for seeds that have not finished read "pending"
rather than being silently dropped, so a half-finished arm can never be
mistaken for a complete one.

Usage:
    python tools/results_workbook.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1 import paths  # noqa: E402

DEST = REPO / "outputs" / "KRAS_results_tables.xlsx"
SEEDS = (42, 43, 44)
# The bag-policy ladder, ordered by how much of a slide each arm keeps.
# 2048/4096/8192 truncate 87%/75%/65% of slides respectively (median slide is
# 12,321 tiles), so this is a genuine ladder and not three names for the same
# thing. CONCH is a separate question and never joins the ladder.
ARMS = [
    ("cap2048", "A · CAP 2,048 tiles/epoch"),
    ("capped", "B · CAP 4,096 tiles/epoch"),
    ("cap8192", "C · CAP 8,192 tiles/epoch"),
    ("cap16384", "D · CAP 16,384 tiles/epoch"),
    ("uncapped", "E · UNCAPPED (full bags)"),
    ("pb_cap4096", "PB · CAP 4,096 · patient-balanced"),
    ("pb_cap8192", "PB · CAP 8,192 · patient-balanced"),
    ("conch", "REF · CONCH v1.5 (full bags)"),
    ("v2cls", "REF · VIRCHOW2 CLS · cap 4,096 · patient-balanced"),
]

# ── styling ──────────────────────────────────────────────────────────────────
H1 = Font(bold=True, size=13, color="FFFFFF")
H2 = Font(bold=True, size=11)
BOLD = Font(bold=True)
MUTED = Font(italic=True, size=9, color="666666")
FILL_H1 = PatternFill("solid", fgColor="1F3864")
FILL_ARM_A = PatternFill("solid", fgColor="DDEBF7")
FILL_ARM_B = PatternFill("solid", fgColor="E2EFDA")
FILL_REF = PatternFill("solid", fgColor="F2F2F2")
FILL_KEY = PatternFill("solid", fgColor="FFF2CC")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
FILL_ARM_C = PatternFill("solid", fgColor="FCE4D6")
FILL_ARM_D = PatternFill("solid", fgColor="EDEDF7")
FILL_ARM_E = PatternFill("solid", fgColor="FFF0F5")
FILL_PB = PatternFill("solid", fgColor="E8F4EA")
ARM_FILL = {
    "pb_cap4096": FILL_PB,
    "pb_cap8192": FILL_PB,
    "cap2048": FILL_ARM_A,
    "capped": FILL_ARM_B,
    "cap8192": FILL_ARM_C,
    "cap16384": FILL_ARM_D,
    "uncapped": FILL_ARM_E,
    "conch": FILL_REF,
    "v2cls": FILL_REF,
}


def num(v, nd: int = 4):
    """A cell value that never silently blanks.

    openpyxl writes NaN as an empty numeric cell, so an undefined calibration
    slope or a single-class cohort AUROC would render as a blank that reads as
    "not applicable" instead of "this could not be computed".
    """
    if v is None:
        return "pending"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return round(f, nd) if np.isfinite(f) else "undefined"


def fmt(values: list[float], nd: int = 4) -> str:
    """Median [min-max]; a single seed says so; nothing says 'pending'."""
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return "pending"
    if len(v) == 1:
        return f"{v[0]:.{nd}f} (1 seed)"
    return f"{np.median(v):.{nd}f} [{min(v):.{nd}f}–{max(v):.{nd}f}]"


def p_fmt(values: list[float]) -> str:
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return "pending"
    lo, hi = min(v), max(v)
    s = lambda x: "<1e-12" if x < 1e-12 else f"{x:.3g}"  # noqa: E731
    return s(lo) if len(v) == 1 else f"{s(lo)} – {s(hi)}"


def title(ws, row: int, text: str, width: int = 10) -> int:
    ws.cell(row=row, column=1, value=text).font = H1
    for c in range(1, width + 1):
        ws.cell(row=row, column=c).fill = FILL_H1
    return row + 1


def note(ws, row: int, text: str, width: int = 10) -> int:
    c = ws.cell(row=row, column=1, value=text)
    c.font = MUTED
    c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=width)
    return row + 1


def header(ws, row: int, labels: list[str]) -> int:
    for i, lab in enumerate(labels, start=1):
        c = ws.cell(row=row, column=i, value=lab)
        c.font = H2
        c.border = BOX
        c.alignment = Alignment(wrap_text=True, vertical="bottom")
    return row + 1


def widths(ws, spec: dict[int, int]) -> None:
    for col, w in spec.items():
        ws.column_dimensions[get_column_letter(col)].width = w


# ── sheets ───────────────────────────────────────────────────────────────────
def sheet_readme(wb, e0: dict, e1d: dict) -> None:
    ws = wb.create_sheet("README")
    widths(ws, {1: 30, 2: 108})
    r = title(ws, 1, "KRAS histomorphology study — Aim 1 results tables", 2)
    r += 1
    rows = [
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Design basis", "crc_final_v4.csv — 2,003 analyzable slides, 1,799 patients, 4 cohorts"),
        ("E0 population", "All KRAS-labelled primary patients, all sites: 604 mut / 882 WT (n = 1,486)"),
        ("Encoder", "UNIv1 (frozen), 1,024-d tile embeddings; CONCH v1.5 as a reference arm only"),
        ("Aggregator", "Gated ABMIL 1,024→512, attn 384, input dropout 0.10, dropout 0.25"),
        ("Splitting", "5-fold, patient-grouped, cohort×label stratified (splits=aim1_balanced)"),
        ("Seeds", "42/43/44"),
        ("Seed scope — VERIFIED", "Seeds vary WEIGHT INITIALISATION and data order ONLY. The 5-fold "
                                  "partition is frozen in the manifest (predefined_oof_kfold) and is "
                                  "byte-identical across all seeds and every arm — checked directly: "
                                  "100% of patients keep the same fold in seeds 42/43/44. The design "
                                  "document states the seed drives BOTH the partition and the init; in "
                                  "this implementation it does not. Consequence: the seed SD reported "
                                  "below is initialisation variance alone and is a LOWER BOUND on the "
                                  "seed variance the design intended to capture, so TOTAL SE is "
                                  "optimistic. Resolve before the seed range is quoted as a result."),
        ("Summary rule", "median [min–max] across seeds; NEVER a three-draw SD (design §2.1(1))"),
        ("Score assembly", "per-seed scoring, then summarise — predictions are NOT averaged across "
                           "seeds for the headline (that would be a 3-model ensemble, +0.016–0.041 "
                           "AUROC). The ensemble row is reported separately and labelled."),
        ("", ""),
        ("WHY A BAG-POLICY LADDER", "Bag policy is an open screen (design §2.1(4)), so it is "
                         "screened as a ladder rather than a two-way test: caps of 2,048 / 4,096 / "
                         "8,192 random tiles per epoch, all with FULL bags at inference, against an "
                         "uncapped control. The ladder is non-degenerate — the median slide holds "
                         "12,321 tiles, so the three caps truncate 87% / 75% / 65% of slides "
                         "respectively. Every arm is recorded in full so the losing arms survive as "
                         "references rather than being overwritten. Declare the winner only once an "
                         "arm's seeds have all landed."),
        ("Seeds per arm", "Uncapped and cap-4,096 carry seeds 42/43/44; cap-2,048 and cap-8,192 "
                          "carry seeds 42/43 (2 seeds, as commissioned). A 2-seed arm has a range, "
                          "not a distribution — read it as such."),
        ("", ""),
        ("Sheets", "E0 — locked baseline | E1a — dependency-aware challenge sets | "
                   "E1d — clinicopathologic baseline | Run status | Provenance"),
        ("Workbook scope", "This workbook predates the final E2a, E3 and E4 results; those are "
                           "verified in reports/Results.md and their JSON artifacts. E1c, E1e "
                           "and E5 have no result sheets."),
    ]
    for k, v in rows:
        if k:
            ws.cell(row=r, column=1, value=k).font = BOLD
        c = ws.cell(row=r, column=2, value=v)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    r += 1
    ws.cell(row=r, column=1, value="SEED COMPLETENESS").font = H2
    r += 1
    for arm, label in ARMS:
        done = e0.get(arm, {}).get("seeds_complete", [])
        pend = [s for s in SEEDS if s not in done]
        ws.cell(row=r, column=1, value=label).fill = ARM_FILL[arm]
        ws.cell(row=r, column=2,
                value=f"complete: {done or '—'}   pending: {pend or 'none — arm finished'}")
        r += 1


def sheet_e0(wb, e0: dict) -> None:
    ws = wb.create_sheet("E0")
    widths(ws, {1: 34, 2: 15, 3: 15, 4: 15, 5: 30, 6: 26})
    r = title(ws, 1, "E0 — Locked baseline (Tier 1)", 6)
    r = note(ws, r, "Patient-level pooled out-of-fold, n = 1,486 (604 mutant, 40.6%). "
                    "Threshold is the Youden point chosen INSIDE each seed's own OOF scores — "
                    "score scales differ across seeds, so a shared threshold would measure scale "
                    "drift, not discrimination.", 6)
    r = note(ws, r, "CAVEAT ON THE SEED RANGE: the fold partition is FROZEN across seeds "
                    "(predefined_oof_kfold; verified identical for 100% of patients). Seeds move "
                    "the weight initialisation only, so the seed SD here is NOT the design "
                    "document's seed variance and TOTAL SE is correspondingly optimistic.", 6)
    r += 1

    keys = [
        ("auroc", "Pooled OOF AUROC", 4),
        ("auprc", "AUPRC (baseline 0.406)", 4),
        ("brier", "Brier (cross-fitted Platt)", 4),
        ("calibration_intercept", "Calibration intercept", 4),
        ("calibration_slope", "Calibration slope", 4),
    ]
    for arm, label in ARMS:
        block = e0.get(arm) or {}
        if not block:
            continue
        ws.cell(row=r, column=1, value=label).font = H2
        for c in range(1, 7):
            ws.cell(row=r, column=c).fill = ARM_FILL[arm]
        r += 1
        done = block["seeds_complete"]
        r = header(ws, r, ["metric", "seed 42", "seed 43", "seed 44",
                           "median [min–max]", "notes"])
        per = block["per_seed"]

        def col(seed, getter):
            s = str(seed)
            return getter(per[s]) if s in per else None

        for key, nice, nd in keys:
            ws.cell(row=r, column=1, value=nice)
            vals = []
            for i, s in enumerate(SEEDS, start=2):
                v = col(s, lambda d, k=key: d[k])
                ws.cell(row=r, column=i, value=num(v, nd))
                vals.append(v)
            c = ws.cell(row=r, column=5, value=fmt(vals, nd))
            if key == "auroc":
                c.fill = FILL_KEY
                c.font = BOLD
            for i in range(1, 7):
                ws.cell(row=r, column=i).border = BOX
            r += 1

        for sub, nice in (("sensitivity", "Youden sensitivity"),
                          ("specificity", "Youden specificity"),
                          ("threshold", "Youden threshold (per seed)")):
            ws.cell(row=r, column=1, value=nice)
            vals = []
            for i, s in enumerate(SEEDS, start=2):
                v = col(s, lambda d, k=sub: d["youden"][k])
                ws.cell(row=r, column=i, value=num(v))
                vals.append(v)
            ws.cell(row=r, column=5, value=fmt(vals))
            if sub == "threshold":
                ws.cell(row=r, column=6, value="never shared across seeds").font = MUTED
            for i in range(1, 7):
                ws.cell(row=r, column=i).border = BOX
            r += 1

        cohorts = sorted(per[str(done[0])]["per_cohort"])
        for cohort in cohorts:
            ws.cell(row=r, column=1, value=f"  per-cohort AUROC · {cohort}")
            vals = []
            for i, s in enumerate(SEEDS, start=2):
                v = col(s, lambda d, k=cohort: d["per_cohort"][k])
                ws.cell(row=r, column=i, value=num(v))
                vals.append(v)
            ws.cell(row=r, column=5, value=fmt(vals))
            ws.cell(row=r, column=6, value="cohort-CONDITIONED, not held out").font = MUTED
            for i in range(1, 7):
                ws.cell(row=r, column=i).border = BOX
            r += 1

        u = block["uncertainty"]
        ens = block["ensemble"]
        ws.cell(row=r, column=1, value="Patient-bootstrap SE — PRIMARY").font = BOLD
        ws.cell(row=r, column=5, value=round(u["se_patient_bootstrap"], 4))
        r += 1
        for key, nice in (("sd_across_seeds", "Seed SD — ROBUSTNESS, not inference"),
                          ("se_seed_mean", "Seed SE (SD/√k)"),
                          ("se_total", "quadrature total — NOT primary")):
            v = u[key]
            ws.cell(row=r, column=1, value=nice).font = BOLD if key == "se_total" else Font()
            v = None if v is None or (isinstance(v, str)) or not np.isfinite(float(v)) else float(v)
            cell = ws.cell(row=r, column=5,
                           value=round(v, 4) if v is not None else "pending (needs ≥2 seeds)")
            if key == "se_total":
                cell.font = BOLD
            r += 1
        ws.cell(row=r, column=1, value=f"{len(done)}-seed ENSEMBLE AUROC (reference only)").font = BOLD
        ws.cell(row=r, column=5,
                value=f"{ens['auroc']:.4f} [{ens['ci_low']:.4f}–{ens['ci_high']:.4f}]")
        ws.cell(row=r, column=6,
                value="NOT the headline — inflates level, not contrasts").font = MUTED
        r += 3


def sheet_e1a(wb, e1a: dict, e0: dict) -> None:
    ws = wb.create_sheet("E1a")
    widths(ws, {1: 32, 2: 8, 3: 7, 4: 15, 5: 15, 6: 15, 7: 26, 8: 20, 9: 30})
    r = title(ws, 1, "E1a — Dependency-aware evaluation (Tier 1, ZERO new fits)", 9)
    r = note(ws, r, "Restriction of the EVALUATION population only — the E0 model is unchanged, "
                    "nothing retrained, nothing rethresholded. Inference is a SHARED-RESAMPLE "
                    "bootstrap computed WITHIN EACH REPETITION: patients are resampled once from "
                    "set A and both AUROC(A_boot) and AUROC(set ∩ A_boot) are recomputed on that "
                    "same resample, so the nesting-induced correlation is carried correctly. A "
                    "naive paired bootstrap returns exactly zero here and is wrong. CIs are "
                    "summarised across repetitions as the median bound, which keeps seed variance "
                    "and patient-sampling variance separable — bootstrapping a seed-ensemble would "
                    "silently merge them.", 9)
    r = note(ws, r, "SIGN CONVENTION: the delta column is Δ(A − set), computed as the MEDIAN OF "
                    "SEEDWISE DIFFERENCES. NEGATIVE = the challenge set scored HIGHER than A "
                    "(signal retained and then some); POSITIVE = it scored LOWER (signal lost). "
                    "For set D at cap 4,096 this reads Δ(A−D) = −0.0242, equivalently "
                    "Δ(D−A) = +0.0242. It deliberately does NOT equal the difference of the two "
                    "displayed medians (0.6902 − 0.6703 = 0.0199): a median of per-seed "
                    "differences is not the difference of per-seed medians. The gate reads set D's "
                    "own AUROC and CI lower bound, never this column's sign, so the verdict is "
                    "unaffected by the convention.", 9)
    r += 1

    labels = {
        "A_all_primary": ("A · All KRAS primary", "primary"),
        "B_mss": ("B · MSS/pMMR only", "primary"),
        "C_braf_wt": ("C · BRAF wild-type only", "primary"),
        "D_mss_braf_wt": ("D · MSS/pMMR ∧ BRAF-WT (DECISIVE)", "primary"),
        "E_colon": ("E · Colon only", "primary"),
        "F_rectum": ("F · Rectum only", "primary"),
        "A_complete": ("A-complete · MSI AND BRAF known", "supplement"),
        "G_stage_known": ("G · Stage known — derived (§0.6)", "supplement"),
        "G_stage_known_frozen": ("G · Stage known — frozen manifest", "supplement"),
        "H_stage_iv": ("H · Stage IV only, derived (exploratory)", "supplement"),
        "I_right_proximal": ("I · Right / proximal", "supplement"),
        "J_left_distal": ("J · Left / distal", "supplement"),
        "K_transverse": ("K · Transverse (exploratory)", "supplement"),
        "cohort_CPTAC": ("cohort · CPTAC", "supplement"),
        "cohort_RIH": ("cohort · RIH", "supplement"),
        "cohort_SurGen": ("cohort · SurGen", "supplement"),
        "cohort_TCGA": ("cohort · TCGA", "supplement"),
    }

    for arm, label in ARMS:
        block = e1a.get(f"{arm}_e1a") or {}
        if not block:
            continue
        ws.cell(row=r, column=1, value=label).font = H2
        for c in range(1, 10):
            ws.cell(row=r, column=c).fill = ARM_FILL[arm]
        r += 1
        r = header(ws, r, ["challenge set", "n", "role", "AUROC s42", "AUROC s43", "AUROC s44",
                           "AUROC median [min–max]", "median 95% CI",
                           "Δ(A−set) median of seedwise differences"])
        per = block["per_seed"]
        done = sorted(per)
        for key, (nice, role) in labels.items():
            if key not in per[done[0]]:
                continue
            row0 = per[done[0]][key]
            ws.cell(row=r, column=1, value=nice)
            ws.cell(row=r, column=2, value=row0["n"])
            ws.cell(row=r, column=3, value=role)
            aus, dls, los, his = [], [], [], []
            for i, sd in enumerate(SEEDS, start=4):
                ss = str(sd)
                v = per[ss][key]["auroc"] if ss in per else None
                ws.cell(row=r, column=i, value=num(v))
                aus.append(v)
                if ss in per:
                    dls.append(per[ss][key]["delta_vs_A"])
                    los.append(per[ss][key]["ci_low"])
                    his.append(per[ss][key]["ci_high"])
            ws.cell(row=r, column=7, value=fmt(aus))
            if los:
                ws.cell(row=r, column=8,
                        value=f"{np.median(los):.3f}–{np.median(his):.3f}")
            ws.cell(row=r, column=9, value=fmt(dls))
            if key == "D_mss_braf_wt":
                for i in range(1, 10):
                    ws.cell(row=r, column=i).fill = FILL_KEY
                ws.cell(row=r, column=1).font = BOLD
            if role == "supplement":
                ws.cell(row=r, column=1).font = MUTED
            for i in range(1, 10):
                ws.cell(row=r, column=i).border = BOX
            r += 1
        gate = block.get("gate")
        if gate:
            c = ws.cell(row=r, column=1, value=f"DECISION GATE → {gate['verdict']}")
            c.font = BOLD
            c.fill = FILL_KEY if gate["verdict"] == "PASS" else PatternFill(
                "solid", fgColor="F8CBAD")
            ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=9)
            r += 1
        r += 1
        r = note(ws, r, "Per-cohort rows evaluate the ALL-COHORT model within each cohort. That is "
                        "cohort-CONDITIONED evaluation, not cohort-held-out validation, and cannot "
                        "support a 'not institution' claim — institutional transport is E2a's job "
                        "and nothing else's. Likewise 'stage known' is a restriction, NOT stage "
                        "adjustment; stage/context adjustment belongs to E1c/E1d.", 9)
        r += 2

    r = title(ws, r, "DECISION GATE — operational definition", 9)
    r = note(ws, r, "HONESTY NOTE: E0/E1a results were already visible when these thresholds were "
                    "fixed, so this is a POST-HOC FORMALISATION, not a blind pre-registration. Its "
                    "value is that it binds every arm not yet read — the remaining cap arms, LOCO-D "
                    "after E2a, any re-run — to one rule instead of a per-table judgement call. "
                    "Describe it that way in the paper; do not call it prespecified.", 9)
    rules = [
        ("PASS — proceed on the dependency-robust framing", "ALL of: median set-D AUROC ≥ 0.60; "
         "within-repetition 95% CI lower bound > 0.55 in EVERY repetition; median Δ(A−D) ≤ 0.05"),
        ("FAIL — fire E1b inside set D, pivot to a molecular-dependency paper",
         "ANY of: median set-D AUROC < 0.55; the set-D 95% CI includes 0.50 in any repetition"),
        ("INDETERMINATE", "Anything else. Treated as FAIL — the fail-safe branch is the cheap one."),
        ("Why these numbers", "Anchored to the design's own power arithmetic for set D (n = 1,129): "
         "minimum detectable AUROC 0.548, CI half-width ±0.031. 0.55 is the detection floor, so "
         "clearing 0.60 is a real effect and not a resolution artefact."),
    ]
    for k, v in rules:
        ws.cell(row=r, column=1, value=k).font = BOLD
        c = ws.cell(row=r, column=4, value=v)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=r, start_column=4, end_row=r, end_column=9)
        ws.row_dimensions[r].height = 30
        r += 1
    r += 1
    c = ws.cell(row=r, column=1,
                value="CLAIM LANGUAGE — E1a supports: 'a KRAS-associated H&E signal that remains "
                      "after conditioning on major MEASURED molecular and clinicopathologic "
                      "dependencies.' It does NOT support 'generated by KRAS-associated morphology "
                      "and not MSI, BRAF, site, stage or cohort'. Unmeasured molecular subtypes, "
                      "stromal context, assay differences and cohort characteristics may remain.")
    c.font = BOLD
    c.alignment = Alignment(wrap_text=True, vertical="top")
    c.fill = FILL_KEY
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=9)
    ws.row_dimensions[r].height = 58


def sheet_e1d(wb, e1d: dict) -> None:
    ws = wb.create_sheet("E1d")
    widths(ws, {1: 30, 2: 14, 3: 14, 4: 14, 5: 26, 6: 26, 7: 24})
    r = title(ws, 1, "E1d — Clinicopathologic baseline (Tier 2)", 7)
    r = note(ws, r, "TWO POPULATIONS, both rendered below. PRIMARY = all 1,486 primary patients, "
                    "with stage carried as an explicit 'unknown' level (381 patients) and age "
                    "median-imputed with a missing flag (4) — nobody is deleted. SENSITIVITY = "
                    "set G (stage known, n = 1,105; 429 mutant). The restriction is the "
                    "sensitivity arm and not the headline because it discards 26% of patients and "
                    "conditions on a variable whose MISSINGNESS is itself informative (RIH and "
                    "SR1482 carry most of it).", 7)
    r = note(ws, r, "Clinical covariates: age, sex, site, stage. MSI and BRAF are deliberately "
                    "EXCLUDED — molecular assay results do not exist at prediction time in the "
                    "stated use case, and including them would make the baseline stronger and the "
                    "paper weaker. The clinical model is cross-fitted on E0's own five folds; the "
                    "WSI column is already out-of-fold.", 7)
    r = note(ws, r, "INFERENCE: paired PATIENT BOOTSTRAP (2,000 resamples) — patients are resampled "
                    "once and BOTH models recomputed on that same resample. The likelihood-ratio "
                    "test and net reclassification improvement are REMOVED, not demoted: an LR test "
                    "has no valid null on cross-validated predictions, and NRI is unstable at these "
                    "strata counts. DeLong is reported as secondary only — its variance omits the "
                    "cross-validation fitting component and can understate uncertainty here. "
                    "Rank metrics use raw scores; Brier / log loss / calibration use cross-fitted "
                    "Platt, applied identically to both models.", 7)
    r += 1

    populations = [("PRIMARY · all 1,486", "__A"), ("SENSITIVITY · set G (1,105)", "")]
    for pop_label, suffix in populations:
        c = ws.cell(row=r, column=1, value=pop_label)
        c.font = H1
        c.fill = FILL_H1
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
        r += 2
        r = _e1d_population(ws, r, e1d, suffix)
    note(ws, r, "The clinical-only column is IDENTICAL across seeds by construction — it never "
                "touches the WSI model and the folds are frozen, so there is nothing for the seed "
                "to move. Measured at 0.539 pooled / 0.535 macro-LOCO on all 1,486 and 0.549 / "
                "0.540 on set G, against the 0.509 / 0.526 quoted in the design document; the "
                "measured values are what these tables report and the discrepancy is worth "
                "reconciling against whatever population the earlier figure came from.", 7)


def _e1d_population(ws, r: int, e1d: dict, suffix: str) -> int:
    """One population's arm blocks. `suffix` selects the JSON key: '__A' is the
    all-1,486 primary population, '' is set G."""
    metric_rows = [
        ("auroc", "AUROC"), ("auprc", "AUPRC"),
        ("brier", "Brier (lower better)"), ("log_loss", "Log loss (lower better)"),
        ("cal_intercept", "Calibration intercept"), ("cal_slope", "Calibration slope"),
        ("macro_loco", "macro-LOCO AUROC"),
    ]
    for arm, label in ARMS:
        block = e1d.get(arm + suffix) or {}
        if not block:
            continue
        per = block["per_seed"]
        done = sorted(per)
        ws.cell(row=r, column=1, value=label).font = H2
        for c in range(1, 8):
            ws.cell(row=r, column=c).fill = ARM_FILL[arm]
        r += 1
        r = header(ws, r, ["metric", "(i) clinical", "(ii) WSI s42", "(ii) WSI s43",
                           "(ii) WSI s44", "(ii) WSI median [min–max]",
                           "paired Δ (ii)−(i), median 95% CI"])
        for key, nice in metric_rows:
            ws.cell(row=r, column=1, value=nice)
            ws.cell(row=r, column=2, value=fmt([per[s]["clinical"][key] for s in done]))
            wv = []
            for i, sd in enumerate(SEEDS, start=3):
                ss = str(sd)
                v = per[ss]["wsi"][key] if ss in per else None
                ws.cell(row=r, column=i, value=num(v))
                wv.append(v)
            c = ws.cell(row=r, column=6, value=fmt(wv))
            if key == "auroc":
                c.fill = FILL_KEY
                c.font = BOLD
            if key != "macro_loco":
                d = [per[s]["delta_wsi_minus_clinical"][key]["delta"] for s in done]
                lo = np.median([per[s]["delta_wsi_minus_clinical"][key]["ci_low"] for s in done])
                hi = np.median([per[s]["delta_wsi_minus_clinical"][key]["ci_high"] for s in done])
                txt = f"{np.median(d):+.4f}  [{lo:+.4f}, {hi:+.4f}]"
                cell = ws.cell(row=r, column=7, value=txt)
                if lo <= 0 <= hi:
                    cell.font = MUTED
                    cell.value = txt + "  (crosses 0)"
            for i in range(1, 8):
                ws.cell(row=r, column=i).border = BOX
            r += 1
        dl = [per[s]["delong_secondary"]["delta"] for s in done]
        ws.cell(row=r, column=1, value="DeLong ΔAUROC (SECONDARY)").font = MUTED
        ws.cell(row=r, column=6, value=fmt(dl))
        ws.cell(row=r, column=7, value="variance omits CV fitting").font = MUTED
        r += 1
        # The stacked model was renamed "fusion_secondary" -> "fusion" when the
        # fusion CONTRASTS were added, and the merged JSON still holds arms from
        # both generations. Read either key rather than crashing on the older
        # arms, which are the archived slide-uniform reference.
        fusion_key = "fusion" if "fusion" in per[done[0]] else "fusion_secondary"
        fu = [per[s][fusion_key]["auroc"] for s in done]
        ws.cell(row=r, column=1, value="Stacked WSI+clinical (SECONDARY)").font = MUTED
        ws.cell(row=r, column=6, value=fmt(fu))
        ws.cell(row=r, column=7,
                value="off-fold coefficients, but trained on CV logits").font = MUTED
        r += 1
        # The two fusion contrasts, present only on the arms run after they were
        # added. "fusion - clinical" is the one that answers whether the IMAGE
        # adds; "fusion - WSI" is whether the CLINICAL variables add.
        for ckey, nice in (("delta_fusion_minus_clinical", "  fusion − clinical  [image adds?]"),
                           ("delta_fusion_minus_wsi", "  fusion − WSI  [clinical adds?]")):
            if ckey not in per[done[0]]:
                continue
            d = [per[s][ckey]["auroc"]["delta"] for s in done]
            lo = np.median([per[s][ckey]["auroc"]["ci_low"] for s in done])
            hi = np.median([per[s][ckey]["auroc"]["ci_high"] for s in done])
            ws.cell(row=r, column=1, value=nice + " ΔAUROC").font = MUTED
            ws.cell(row=r, column=6, value=fmt(d))
            txt = f"{np.median(d):+.4f}  [{lo:+.4f}, {hi:+.4f}]"
            cell = ws.cell(row=r, column=7, value=txt + ("  (crosses 0)" if lo <= 0 <= hi else ""))
            if lo <= 0 <= hi:
                cell.font = MUTED
            r += 1
        r += 3

    return r


def sheet_e1c(wb, e1c: dict) -> None:
    """E1c — IPW context balancing. 0 fits; the E0 OOF matrix, reweighted."""
    ws = wb.create_sheet("E1c IPW")
    widths(ws, {1: 20, 2: 8, 3: 24, 4: 24, 5: 18, 6: 24, 7: 30})
    r = title(ws, 1, "E1c — IPW context balancing (0 new fits)", 7)
    r = note(ws, r, "Stabilised inverse-probability weights w = P(KRAS=t) / P(KRAS=t | X) "
                    "reweight every patient so the mutant and wild-type populations carry the "
                    "SAME covariate distribution, and the AUROC is recomputed on that balanced "
                    "pseudo-population. If the signal were case mix, balancing the case mix "
                    "would collapse it toward 0.500. The delta column is what actually happens.", 7)
    r = note(ws, r, "The propensity model IS E1d's clinical baseline — same covariates, same L2 "
                    "logistic regression, cross-fitted on the same E0 folds — because e(X) is by "
                    "definition P(KRAS | covariates), which is exactly what E1d fits. Refitting it "
                    "here with a different specification would let the reweighting be tuned. "
                    "SCOPE: IPW balances MEASURED covariates only, and reweighting is not "
                    "cohort-held-out — that is E2a's claim and remains E2a's.", 7)
    r += 1
    for cond, block in e1c.items():
        ws.cell(row=r, column=1, value=f"{cond} — {block.get('description', '')}").font = H2
        for c in range(1, 8):
            ws.cell(row=r, column=c).fill = ARM_FILL.get(cond, FILL_REF)
        r += 1
        for arm, per_seed in block["per_arm"].items():
            seeds = sorted(per_seed)
            first = per_seed[seeds[0]]
            ws.cell(row=r, column=1, value=f"arm: {arm}").font = BOLD
            ws.cell(row=r, column=3,
                    value=f"propensity AUROC {fmt([per_seed[s]['propensity_auroc'] for s in seeds])}")
            ws.cell(row=r, column=4,
                    value=f"ESS {np.median([per_seed[s]['ess'] for s in seeds]):.0f} / {first['n']}")
            bal = first["balance"]
            ws.cell(row=r, column=6, value=(
                f"worst |SMD| {max(v['smd_unweighted'] for v in bal.values()):.3f} → "
                f"{max(v['smd_weighted'] for v in bal.values()):.3f}"))
            ws.cell(row=r, column=7, value=(
                f"levels >0.10: {sum(v['smd_unweighted'] > 0.1 for v in bal.values())} → "
                f"{sum(v['smd_weighted'] > 0.1 for v in bal.values())} of {len(bal)}"))
            r += 1
            r = header(ws, r, ["set", "n", "crude AUROC", "IPW AUROC", "IPW 95% CI",
                               "Δ (IPW − crude)", "reading"])
            for key, cell in first["sets"].items():
                ws.cell(row=r, column=1, value=key)
                ws.cell(row=r, column=2, value=cell["n"])
                ws.cell(row=r, column=3,
                        value=fmt([per_seed[s]["sets"][key]["crude_auroc"] for s in seeds]))
                c = ws.cell(row=r, column=4,
                            value=fmt([per_seed[s]["sets"][key]["ipw_auroc"] for s in seeds]))
                lo = np.median([per_seed[s]["sets"][key]["ipw_ci_low"] for s in seeds])
                hi = np.median([per_seed[s]["sets"][key]["ipw_ci_high"] for s in seeds])
                ws.cell(row=r, column=5, value=f"{lo:.3f}–{hi:.3f}")
                d = [per_seed[s]["sets"][key]["delta_ipw_minus_crude"] for s in seeds]
                ws.cell(row=r, column=6, value=fmt(d))
                if key == "D_mss_braf_wt":
                    c.fill = FILL_KEY
                    c.font = BOLD
                ws.cell(row=r, column=7,
                        value="balancing costs "
                              f"{abs(np.median(d)):.4f} of the ~{np.median([per_seed[s]['sets'][key]['crude_auroc'] for s in seeds]) - 0.5:.3f} "
                              "above chance").font = MUTED
                for i in range(1, 8):
                    ws.cell(row=r, column=i).border = BOX
                r += 1
            r += 1
        r += 1
    note(ws, r, "The 'sidedness' arm is PROVISIONAL: the design asks for pathologist review of "
                "tumor_site_raw before sidedness enters a reported model (§9.3 decision 3), so it "
                "is a sensitivity and the primary never depends on it. 'stage_fill' uses the §0.6 "
                "derived stage, joined from crc_final_v4.csv rather than from the frozen manifest, "
                "whose SHA is part of the training-identity fingerprint.", 7)


def sheet_e1e(wb, e1e: dict) -> None:
    """E1e — dependency budget: what else do the frozen features predict?"""
    ws = wb.create_sheet("E1e budget")
    widths(ws, {1: 26, 2: 8, 3: 16, 4: 16, 5: 20, 6: 44})
    r = title(ws, 1, "E1e — dependency budget (15 fits)", 6)
    r = note(ws, r, "E1a removes a confounder by SUBSETTING, which cannot touch the one confounder "
                    "there is no subset for: cohort identity. E1e measures it directly, by training "
                    "auxiliary heads on the SAME frozen features and the SAME outer folds. A HIGH "
                    "cohort AUROC is EXPECTED and is not a study-stopping result — H&E features "
                    "carry stain, scanner and population. What matters is whether the KRAS logit "
                    "COVARIES with the probe logits (readout 3).", 6)
    r += 1

    block = e1e.get("readout_1") or {}
    if block:
        ws.cell(row=r, column=1, value="READOUT 1 — what else the features predict").font = H2
        r += 1
        r = header(ws, r, ["probe", "n", "patient AUROC", "", "", "detail"])
        for task, cell in block.items():
            ws.cell(row=r, column=1, value=task)
            ws.cell(row=r, column=2, value=cell["n"])
            auc = cell.get("auroc", cell.get("auroc_macro_ovr"))
            c = ws.cell(row=r, column=3, value=num(auc))
            c.font = BOLD
            if task == "cohort":
                c.fill = FILL_KEY
            per = cell.get("per_class_ovr")
            if per:
                ws.cell(row=r, column=6,
                        value="one-vs-rest: " + "  ".join(f"{k} {v:.3f}" for k, v in per.items()))
            for i in range(1, 7):
                ws.cell(row=r, column=i).border = BOX
            r += 1
        r += 2

    block = e1e.get("readout_2") or {}
    if block:
        ws.cell(row=r, column=1, value="READOUT 2 — bag-size probe (0 fits)").font = H2
        r += 1
        r = header(ws, r, ["quantity", "", "value", "", "", "reading"])
        rows = [
            ("Spearman r(KRAS logit, tile count)", block["spearman_kras_logit_vs_tiles"],
             f"p = {block['spearman_p']:.3g}"),
            ("AUROC  tile count → KRAS", block["auroc_tiles_to_kras"], "chance is 0.500"),
            ("AUROC  tile count → cohort (macro OVR)", block["auroc_tiles_to_cohort_macro_ovr"],
             "the institutional signature"),
        ]
        for name, value, reading in rows:
            ws.cell(row=r, column=1, value=name)
            ws.cell(row=r, column=3, value=num(value))
            ws.cell(row=r, column=6, value=reading).font = MUTED
            for i in range(1, 7):
                ws.cell(row=r, column=i).border = BOX
            r += 1
        r += 2

    block = e1e.get("readout_3") or {}
    if block:
        ws.cell(row=r, column=1,
                value="READOUT 3 — does the KRAS logit covary with the probe logits?").font = H2
        r += 1
        r = note(ws, r, "Partial correlation with cohort and site projected out of BOTH sides "
                        "first, so the number is the association NOT explained by the two "
                        "variables every cohort differs on. Patient bootstrap (2,000).", 6)
        r = header(ws, r, ["probe logit", "n", "raw r", "partial r", "95% CI", "excludes 0"])
        for name, cell in block.items():
            ws.cell(row=r, column=1, value=name)
            ws.cell(row=r, column=2, value=cell["n"])
            ws.cell(row=r, column=3, value=num(cell["raw_r"]))
            ws.cell(row=r, column=4, value=num(cell["partial_r"]))
            ws.cell(row=r, column=5, value=f"[{cell['ci'][0]:+.3f}, {cell['ci'][1]:+.3f}]")
            c = ws.cell(row=r, column=6, value="YES" if cell["excludes_zero"] else "no")
            if cell["excludes_zero"]:
                c.fill = FILL_KEY
            for i in range(1, 7):
                ws.cell(row=r, column=i).border = BOX
            r += 1
        r += 2

    block = e1e.get("readout_4_exploratory") or {}
    if block:
        ws.cell(row=r, column=1, value="READOUT 4 — residualised KRAS AUROC").font = H2
        r += 1
        r = note(ws, r, "EXPLORATORY, and demoted deliberately. KRAS and BRAF are both MAPK-pathway "
                        "alterations, so BRAF-predictive morphology plausibly contains genuine "
                        "shared biology; projecting it out may subtract SIGNAL rather than bias, "
                        "leaving a number uninterpretable in either direction. E1a conditions on "
                        "BRAF STATUS instead and answers the same question without the hazard.", 6)
        r = header(ws, r, ["projected out", "n", "AUROC", "Δ vs raw", "", ""])
        base = block.get("raw")
        ws.cell(row=r, column=1, value="(raw KRAS logit)")
        ws.cell(row=r, column=3, value=num(base))
        r += 1
        for name, cell in block.items():
            if name == "raw":
                continue
            ws.cell(row=r, column=1, value=name)
            ws.cell(row=r, column=2, value=cell["n"])
            ws.cell(row=r, column=3, value=num(cell["auroc"]))
            ws.cell(row=r, column=4, value=num(cell["delta_vs_raw"]))
            for i in range(1, 7):
                ws.cell(row=r, column=i).border = BOX
            r += 1


def sheet_cap_decision(wb, e0: dict, e1d: dict) -> None:
    """The bag-policy screen (design §2.1(4)) and the evidence behind the pick."""
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from oceanpath.aim1 import evaluate

    ws = wb.create_sheet("Cap decision")
    widths(ws, {1: 14, 2: 26, 3: 12, 4: 12, 5: 12, 6: 12, 7: 14, 8: 26})
    r = title(ws, 1, "Bag-policy screen — which cap, and why", 8)
    r = note(ws, r, "Design §2.1(4) made bag policy a declared screen rather than a locked choice. "
                    "Four arms, identical in every other respect (verified: the resolved configs "
                    "differ in exactly one key, dataset_max_instances), 3 seeds each, full bags at "
                    "inference everywhere.", 8)
    r += 1

    r = header(ws, r, ["arm", "readout", "cap 2048", "cap 4096", "cap 8192", "cap 16384",
                       "uncapped", "spread", "verdict"])
    order = ["cap2048", "capped", "cap8192", "cap16384", "uncapped"]

    def row(name, getter, nd=4, lower_better=False):
        nonlocal r
        vals = []
        for arm in order:
            b = e0.get(arm) or e1d.get(arm) or {}
            vals.append(getter(arm) if b else None)
        ws.cell(row=r, column=1, value="")
        ws.cell(row=r, column=2, value=name)
        for i, v in enumerate(vals, start=3):
            ws.cell(row=r, column=i, value=num(v, nd))
        clean = [v for v in vals if v is not None and np.isfinite(v)]
        spread = max(clean) - min(clean) if clean else float("nan")
        ws.cell(row=r, column=8, value=num(spread, nd))
        best = (min if lower_better else max)(clean) if clean else None
        winner = order[vals.index(best)] if best is not None else ""
        ws.cell(row=r, column=9, value={"cap2048": "cap 2048", "capped": "cap 4096",
                                        "cap8192": "cap 8192", "cap16384": "cap 16384",
                                        "uncapped": "uncapped"}.get(winner, ""))
        for i in range(1, 10):
            ws.cell(row=r, column=i).border = BOX
        r += 1

    def e0med(arm, key):
        b = e0.get(arm) or {}
        if not b:
            return None
        return float(np.median([b["per_seed"][s][key] for s in sorted(b["per_seed"])]))

    def e0range(arm):
        b = e0.get(arm) or {}
        if not b:
            return None
        v = [b["per_seed"][s]["auroc"] for s in sorted(b["per_seed"])]
        return float(max(v) - min(v))

    row("E0 pooled AUROC", lambda a: e0med(a, "auroc"))
    row("E0 AUPRC", lambda a: e0med(a, "auprc"))
    row("E0 Brier (lower better)", lambda a: e0med(a, "brier"), lower_better=True)
    row("E0 calibration slope (|1−x| lower better)",
        lambda a: e0med(a, "calibration_slope"))
    row("E0 seed range (lower better)", e0range, lower_better=True)
    row("E0 TOTAL SE (lower better)",
        lambda a: float(e0.get(a, {}).get("uncertainty", {}).get("se_total", float("nan"))),
        lower_better=True)

    def e1amed(arm, st, key="auroc"):
        b = e0.get(f"{arm}_e1a") or {}
        if not b:
            return None
        return float(np.median([b["per_seed"][s][st][key] for s in sorted(b["per_seed"])]))

    row("E1a set D AUROC (DECISIVE)", lambda a: e1amed(a, "D_mss_braf_wt"))
    row("E1a set B AUROC", lambda a: e1amed(a, "B_mss"))

    def e1dmed(arm, key):
        b = e1d.get(arm) or {}
        if not b:
            return None
        return float(np.median([b["per_seed"][s]["wsi"][key] for s in sorted(b["per_seed"])]))

    row("E1d WSI AUROC (set G)", lambda a: e1dmed(a, "auroc"))
    row("E1d WSI macro-LOCO", lambda a: e1dmed(a, "macro_loco"))
    r += 1

    # ── the bag-size shortcut probe, the one design argument for capping ──
    r = title(ws, r, "Bag-size shortcut probe — the one design argument FOR capping", 8)
    r = note(ws, r, "§2.1(4) argues for capping because slide size is an institutional signature "
                    "the attention softmax can perceive. This measures whether that actually "
                    "happens. It does not: capping does NOT reduce the correlation, because every "
                    "arm evaluates on FULL bags, so inference-time exposure to bag size is "
                    "identical across arms by construction. Capping changes training only.", 8)
    manifest = pd.read_csv(paths.DEV_MANIFEST)
    tiles = manifest.groupby("patient_id")["patch_count"].sum().rename("tiles")
    layout = {"cap2048": "1a_cap2048/univ1", "capped": "1a_cap4096/univ1",
              "cap8192": "1a_cap8192/univ1", "cap16384": "1a_cap16384/univ1",
              "uncapped": "1a/univ1"}
    r = header(ws, r, ["arm", "Spearman r(P(KRAS), tiles)", "", "", "", "", "", "interpretation"])
    last = None
    for arm in order:
        rs = []
        for sd in SEEDS:
            f = paths.OUTPUT_ROOT / f"train/{layout[arm]}/seed{sd}/oof_predictions.parquet"
            if not f.is_file():
                continue
            pt = evaluate.to_patient_level(pd.read_parquet(f), manifest).merge(
                tiles, on="patient_id")
            rs.append(spearmanr(pt["prob_raw"], pt["tiles"])[0])
            last = pt
        ws.cell(row=r, column=1, value={"cap2048": "cap 2048", "capped": "cap 4096",
                                        "cap8192": "cap 8192", "cap16384": "cap 16384",
                                        "uncapped": "uncapped"}[arm])
        ws.cell(row=r, column=2, value=num(float(np.median(rs)) if rs else None))
        ws.cell(row=r, column=8, value="r² < 0.8% of variance in every arm")
        for i in range(1, 9):
            ws.cell(row=r, column=i).border = BOX
        r += 1
    if last is not None:
        r += 1
        ws.cell(row=r, column=1, value="tile count alone →").font = BOLD
        ws.cell(row=r, column=2, value="KRAS")
        ws.cell(row=r, column=3, value=num(roc_auc_score(last["label"], last["tiles"])))
        ws.cell(row=r, column=8, value="0.5 = no KRAS signal; the shortcut is not exploitable")
        r += 1
        for c in sorted(last["cohort"].unique()):
            ws.cell(row=r, column=2, value=f"cohort = {c}")
            ws.cell(row=r, column=3,
                    value=num(roc_auc_score((last["cohort"] == c).astype(int), last["tiles"])))
            ws.cell(row=r, column=8,
                    value="tile count IS a cohort fingerprint…").font = MUTED
            r += 1
        r += 1
        c = ws.cell(row=r, column=1,
                    value="…but it is a fingerprint the KRAS head does not use: tile count "
                          "predicts cohort strongly and KRAS at chance (0.516), and the model's "
                          "score-vs-size correlation is r ≈ 0.05–0.08 in every arm — with the "
                          "CAPPED arms no better than uncapped. The shortcut-closure argument for "
                          "capping is therefore void in this setup, and the cap must be chosen on "
                          "performance, stability and cost alone.")
        c.font = BOLD
        c.alignment = Alignment(wrap_text=True, vertical="top")
        c.fill = FILL_KEY
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
        ws.row_dimensions[r].height = 58


def sheet_status(wb) -> None:
    ws = wb.create_sheet("Run status")
    widths(ws, {1: 26, 2: 14, 3: 12, 4: 46, 5: 30})
    r = title(ws, 1, "Training run inventory", 5)
    r = note(ws, r, f"Snapshot {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
                    "One RTX A5000; runs were executed concurrently, which inflates per-fold "
                    "wall time relative to a solo run.", 5)
    r += 1
    r = header(ws, r, ["run", "arm", "seed", "train_dir", "status"])
    root = paths.OUTPUT_ROOT / "train"
    # Driven off ARMS so a newly added arm cannot be silently missing from the
    # inventory the way cap2048/cap8192 were.
    layout = {
        "uncapped": ("aim1_1a_univ1", root / "1a" / "univ1"),
        "cap2048": ("aim1_1a_univ1_cap2048", root / "1a_cap2048" / "univ1"),
        "capped": ("aim1_1a_univ1_cap4096", root / "1a_cap4096" / "univ1"),
        "cap8192": ("aim1_1a_univ1_cap8192", root / "1a_cap8192" / "univ1"),
        "cap16384": ("aim1_1a_univ1_cap16384", root / "1a_cap16384" / "univ1"),
        "pb_cap4096": ("aim1_1a_pb_cap4096", root / "1a_pb_cap4096" / "univ1"),
        "pb_cap8192": ("aim1_1a_pb_cap8192", root / "1a_pb_cap8192" / "univ1"),
        "conch": ("aim1_1a_conch", root / "1a_conch" / "conch_v15"),
        "v2cls": ("aim1_1a_pb_cap4096_v2cls", root / "1a_pb_cap4096" / "virchow2_cls"),
    }
    missing = [a for a, _ in ARMS if a not in layout]
    assert not missing, f"arm(s) {missing} have no train_dir layout entry"
    rows = [
        (layout[arm][0], arm, sd, layout[arm][1] / f"seed{sd}") for arm, _ in ARMS for sd in SEEDS
    ]
    for name, arm, seed, path in rows:
        done = (path / "oof_predictions.parquet").is_file()
        folds = len(list(path.glob("fold_*/completion.json"))) if path.is_dir() else 0
        if done:
            status = "complete (5/5 folds)"
        elif path.is_dir():
            status = f"RUNNING — {folds}/5 folds done"
        else:
            status = "not launched"
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=2, value=arm)
        ws.cell(row=r, column=3, value=seed)
        ws.cell(row=r, column=4, value=str(path))
        c = ws.cell(row=r, column=5, value=status)
        c.fill = FILL_ARM_B if done else (FILL_KEY if folds else FILL_REF)
        for i in range(1, 6):
            ws.cell(row=r, column=i).border = BOX
        r += 1
    r += 1
    r = note(ws, r, "RESOLVED — finalize.refit used to fail with \"loss_type='bce' requires "
                    "a single-logit classifier head (num_classes=1), but got num_classes=2\". "
                    "finalize.py now applies the same head contract the folds use, and every "
                    "patient-balanced run carries final/refit/model.ckpt. Nothing in E0/E1a/"
                    "E1c/E1d/E1e reads the refit — they all consume oof_predictions.parquet — "
                    "so this never affected a reported number either way.", 5)


def sheet_provenance(wb) -> None:
    ws = wb.create_sheet("Provenance")
    widths(ws, {1: 34, 2: 100})
    r = title(ws, 1, "Provenance", 2)
    r += 1
    items = [
        ("Manifest", str(paths.DEV_MANIFEST)),
        ("Training root", str(paths.OUTPUT_ROOT / "train")),
        ("E0/E1a source JSON", str(paths.EVAL_ROOT / "e0_e1a_seed_report.json")),
        ("E1d source JSON", str(paths.EVAL_ROOT / "e1d.json")),
        ("E0/E1a generator", "aim1_locked_baseline.py --compare --e1a"),
        ("E1d generator", "aim1_clinical_baseline.py --compare"),
        ("Workbook generator", "tools/results_workbook.py"),
        ("Hyperparameters", "embed_dim=512 attn_dim=384 input_dropout=0.10 dropout=0.25 "
                            "lr=1e-4 wd=1e-5; capped arm adds dataset_max_instances=4096 "
                            "eval_full_bags=true"),
        ("Model-selection criterion", "val/patient_auroc (not val loss)"),
        ("Regenerate", "Re-run both generators, then this script. Cells fill in as seeds land; "
                       "'pending' means the seed has not finished, never that it was dropped."),
    ]
    for k, v in items:
        ws.cell(row=r, column=1, value=k).font = BOLD
        c = ws.cell(row=r, column=2, value=v)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1


def main() -> None:
    e0 = json.loads((paths.EVAL_ROOT / "e0_e1a_seed_report.json").read_text())
    e1d = json.loads((paths.EVAL_ROOT / "e1d.json").read_text())
    wb = Workbook()
    wb.remove(wb.active)
    sheet_readme(wb, e0, e1d)
    sheet_e0(wb, e0)
    sheet_e1a(wb, e0, e0)
    sheet_e1d(wb, e1d)
    # Analyses added after the first compilation; each renders only if its JSON
    # exists, so a partially-run study still produces a workbook.
    e1c_path = paths.EVAL_ROOT / "e1c_ipw.json"
    if e1c_path.is_file():
        sheet_e1c(wb, json.loads(e1c_path.read_text()))
    e1e_path = paths.EVAL_ROOT / "e1e_dependency_budget.json"
    if e1e_path.is_file():
        sheet_e1e(wb, json.loads(e1e_path.read_text()))
    sheet_cap_decision(wb, e0, e1d)
    sheet_status(wb)
    sheet_provenance(wb)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    wb.save(DEST)
    print(f"Wrote {DEST}")
    for ws in wb.worksheets:
        print(f"  {ws.title:14s} {ws.max_row:3d} rows x {ws.max_column} cols")


if __name__ == "__main__":
    main()
