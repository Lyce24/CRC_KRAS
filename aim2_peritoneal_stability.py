#!/usr/bin/env python3
"""E2d-2 - stability audit of the SurGen peritoneal inversion (0 new fits).

WHAT THIS IS FOR. The corrected cap-8192 E2d-1 run found SurGen peritoneal
metastases scoring AUROC 0.0750
(n = 14; 4 mutant, 10 wild-type) - not merely uninformative but inverted. Before
that number is allowed to mean anything it has to survive four questions, none of
which need a new model:

    A  Is the inversion present in every seed, or an artefact of the ensemble?
    B  Which class is shifting - are mutants scored too low, WT too high, or both?
    C  Is one or two patients driving it? Leave-one-patient-out.
    D  Is there an obvious compositional explanation among the 14?

NOTHING IS FITTED, ADAPTED OR RETRAINED HERE, and deliberately so. This panel
was selected after seeing E2d-1 and is therefore a hypothesis-generating
stability/composition audit, not a confirmatory test. With 4 mutant
and 10 wild-type patients there is no honest way to build a peritoneum-specific
classifier, an organ-specific adapter, or a separate MIL model: any such object
would be fitted and evaluated on essentially the same 14 people. This file only
re-reads frozen predictions.

Usage:
    python aim2_peritoneal_stability.py report --cap 8192
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_loco_transport  # noqa: E402
import aim2_metastatic_site as e2d  # noqa: E402
from oceanpath.aim1 import lineage, paths  # noqa: E402

TARGET = "SurGen"
SITE = "peritoneum"


def material_input_identities(
    cap: int,
    *,
    upstream_eval_root: Path | None = None,
) -> dict:
    """Exact identities for every frozen/model or live metadata input used.

    ``upstream_eval_root`` is explicit for append-only downstream refreshes:
    their reports belong to a new output root while E2d-1 remains a frozen
    input from the completed core lineage.
    """

    source_eval = upstream_eval_root or lineage.eval_root()
    return {
        "code": lineage.artifact_identity(Path(__file__)),
        "label_source": lineage.artifact_identity(paths.LABEL_SOURCE),
        "upstream_e2d1": lineage.artifact_identity(
            source_eval / f"e2d1_metastatic_sites_cap{cap}.json"
        ),
        "target_manifests": {
            role: lineage.artifact_identity(aim2_loco_transport.target_manifest(TARGET, role))
            for role in ("primary", "metastatic")
        },
        "fit_receipts": {
            str(seed): lineage.artifact_identity(
                aim2_loco_transport.fit_summary_path(TARGET, seed, cap)
            )
            for seed in aim2_loco_transport.SEEDS
        },
        "score_receipts": {
            role: {
                str(seed): lineage.artifact_identity(
                    aim2_loco_transport.score_receipt_path(TARGET, seed, role, cap)
                )
                for seed in aim2_loco_transport.SEEDS
            }
            for role in ("primary", "metastatic")
        },
    }


def _patient_metadata(source: pd.DataFrame, patient_ids: set[str]) -> pd.DataFrame:
    """Collapse slide metadata with explicit, auditable patient-level rules.

    The old audit used ``drop_duplicates(patient_uid)``, which made continuous
    slide properties depend on CSV row order for multi-slide patients.  Role-
    specific rows are now selected first; patient-constant clinical fields are
    checked, MPP is summarized by the median, and total slide bytes are summed.
    """

    rows = source[source["patient_uid"].astype(str).isin(patient_ids)].copy()
    if "specimen_role" in rows.columns:
        rows = rows[rows["specimen_role"].eq("metastatic")].copy()
    if rows.empty:
        return pd.DataFrame(index=pd.Index([], name="patient_uid"))

    constant = [
        column
        for column in (
            "tumor_site_raw",
            "msi_dmmr",
            "braf",
            "kras_subvariant",
            "subcohort",
            "sex",
            "age_at_diagnosis",
        )
        if column in rows.columns
    ]
    records: list[dict] = []
    for patient_id, block in rows.groupby("patient_uid", sort=True):
        record: dict = {"patient_uid": str(patient_id)}
        for column in constant:
            values = block[column].dropna().unique()
            if len(values) > 1:
                raise RuntimeError(
                    f"{column} is not patient-constant for {patient_id}: "
                    f"{values.tolist()}"
                )
            record[column] = values[0] if len(values) else None
        if "mpp" in block:
            values = pd.to_numeric(block["mpp"], errors="coerce")
            record["mpp"] = float(values.median()) if values.notna().any() else None
        if "slide_size_bytes" in block:
            values = pd.to_numeric(block["slide_size_bytes"], errors="coerce")
            record["slide_size_bytes"] = (
                float(values.sum(min_count=1)) if values.notna().any() else None
            )
        if "image_format" in block:
            formats = sorted(set(block["image_format"].dropna().astype(str)))
            record["image_format"] = "+".join(formats) if formats else None
        record["metadata_slide_count"] = int(len(block))
        records.append(record)
    return pd.DataFrame(records).set_index("patient_uid")


def auroc(y: np.ndarray, eta: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, eta))


def cmd_report(args: argparse.Namespace) -> None:
    cap = args.cap
    output_eval_root = Path(
        getattr(args, "output_eval_root", None) or lineage.eval_root()
    )
    input_eval_root = Path(
        getattr(args, "input_eval_root", None) or lineage.eval_root()
    )
    final_dest = output_eval_root / f"e2d2_peritoneal_audit_cap{cap}.json"
    lineage.ensure_absent(final_dest)
    inputs_before = material_input_identities(
        cap,
        upstream_eval_root=input_eval_root,
    )
    tables = e2d.site_table(cap)
    df = tables[TARGET]
    per = df[df["site"].eq(SITE)].sort_values("patient_id").reset_index(drop=True)
    rest = df[~df["site"].eq(SITE)]
    y = per["label"].to_numpy().astype(int)
    eta = per["mean_logit"].to_numpy()
    upstream = input_eval_root / f"e2d1_metastatic_sites_cap{cap}.json"
    out: dict = {
        "schema_version": 2,
        "cap": cap,
        "lineage": lineage.lineage_name(),
        "n": int(len(per)),
        "n_mut": int(y.sum()),
        "ensemble_auroc": auroc(y, eta),
        "upstream_e2d1": lineage.artifact_identity(upstream),
        "inputs": inputs_before,
        "metadata_aggregation": {
            "unit": "patient within metastatic specimen role",
            "patient_constant_fields": (
                "fail closed when non-null values disagree across slides"
            ),
            "mpp": "median across metastatic slides",
            "slide_size_bytes": "sum across metastatic slides",
            "image_format": "sorted unique formats joined with '+'",
            "metadata_slide_count": "number of metastatic slide rows",
        },
        "seed_results_inferential_role": (
            "none — computational stability diagnostic on the same patients"
        ),
    }

    print(f"\n{'=' * 104}\nE2d-2 · SurGen peritoneal inversion — stability audit"
          f"\n{'=' * 104}")
    print(f"  n = {len(per)}  ({int(y.sum())} KRAS-mutant / {int(len(y) - y.sum())} "
          f"wild-type)   ensemble AUROC {out['ensemble_auroc']:.4f}")

    # ── A. per-seed ──────────────────────────────────────────────────────────
    print(f"\n{'-' * 104}\nA · Is the inversion present in every seed?\n{'-' * 104}")
    ids = list(per["patient_id"])
    seed_auc, seed_eta = {}, {}
    for s in aim2_loco_transport.SEEDS:
        pat = aim2_loco_transport.patients_for(TARGET, s, "metastatic", cap)
        pat = pat[pat["patient_id"].isin(ids)].sort_values("patient_id")
        seed_eta[s] = pat["mean_logit"].to_numpy()
        assert list(pat["patient_id"]) == ids
        seed_auc[s] = auroc(pat["label"].to_numpy().astype(int), seed_eta[s])
    out["per_seed_auroc"] = seed_auc
    print(f"  {'model':10s} {'peritoneum AUROC':>18s}")
    for s, v in seed_auc.items():
        print(f"  seed {s:<5d} {v:18.4f}")
    print(f"  {'ensemble':10s} {out['ensemble_auroc']:18.4f}")
    vals = np.array(list(seed_auc.values()))
    out["per_seed_range"] = float(vals.max() - vals.min())
    print(f"\n  range across seeds {vals.max() - vals.min():.4f}   "
          f"all three below 0.5: {bool((vals < 0.5).all())}")
    print("  Seeds share the training data and differ only in initialisation and")
    print("  data order, so this measures COMPUTATIONAL stability, not sampling")
    print("  variability. Agreement here cannot rescue a 14-patient sample.")

    # ── B. which class is shifting ───────────────────────────────────────────
    print(f"\n{'-' * 104}\nB · Which class is shifting?\n{'-' * 104}")
    from scipy.stats import mannwhitneyu

    ref_sets = {"other SurGen-M sites": rest,
                "SurGen SR1482-P primaries": None}
    ens_p, _ = aim2_loco_transport.seed_ensemble(TARGET, "primary", cap)
    ref_sets["SurGen SR1482-P primaries"] = ens_p[
        ens_p["subcohort"].eq(aim2_loco_transport.SURGEN_MET_SUBCOHORT)]
    print(f"  {'group':30s} {'class':5s} {'n':>4s} {'median logit':>13s} "
          f"{'IQR':>18s}")
    stats: dict = {}
    for name, frame in [("peritoneum", per), *ref_sets.items()]:
        for cls, lab in ((1, "mut"), (0, "WT")):
            v = frame.loc[frame["label"].eq(cls), "mean_logit"].to_numpy()
            if not len(v):
                continue
            q1, q3 = np.percentile(v, [25, 75])
            stats[(name, lab)] = v
            print(f"  {name:30s} {lab:5s} {len(v):4d} {np.median(v):13.3f} "
                  f"[{q1:7.3f},{q3:7.3f}]")
    out["class_shift"] = {}
    for lab in ("mut", "WT"):
        a = stats[("peritoneum", lab)]
        for ref in ("other SurGen-M sites", "SurGen SR1482-P primaries"):
            b = stats[(ref, lab)]
            u = mannwhitneyu(a, b, alternative="two-sided")
            out["class_shift"][f"{lab}_vs_{ref}"] = {
                "median_diff": float(np.median(a) - np.median(b)),
                "p": float(u.pvalue)}
            print(f"    {lab} peritoneum − {ref:28s} median diff "
                  f"{np.median(a) - np.median(b):+7.3f}   Mann-Whitney p = {u.pvalue:.3f}")

    # ── C. leave-one-patient-out ─────────────────────────────────────────────
    print(f"\n{'-' * 104}\nC · Is one or two patients driving it?\n{'-' * 104}")
    loo = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        loo.append((auroc(y[keep], eta[keep]), i))
    vals_loo = np.array([v for v, _ in loo])
    out["loo"] = {"min": float(np.nanmin(vals_loo)), "median": float(np.nanmedian(vals_loo)),
                  "max": float(np.nanmax(vals_loo))}
    print(f"  leave-one-patient-out AUROC over {len(y)} removals:")
    print(f"    min {np.nanmin(vals_loo):.4f}   median {np.nanmedian(vals_loo):.4f}   "
          f"max {np.nanmax(vals_loo):.4f}")
    worst = max(loo, key=lambda t: t[0])
    print(f"    largest single-patient effect: removing {per.loc[worst[1], 'patient_id']} "
          f"({'mut' if y[worst[1]] else 'WT'}) -> {worst[0]:.4f}")
    print(f"    stays below 0.5 for every removal: {bool((vals_loo < 0.5).all())}")
    # drop the single most influential MUTANT, the scarcest class
    mut_idx = np.flatnonzero(y == 1)
    m_loo = [(auroc(y[np.arange(len(y)) != i], eta[np.arange(len(y)) != i]), i)
             for i in mut_idx]
    out["loo_mutants"] = {str(per.loc[i, "patient_id"]): float(v) for v, i in m_loo}
    print(f"    removing each of the {len(mut_idx)} mutants in turn: "
          + ", ".join(f"{v:.3f}" for v, _ in m_loo))
    dest = final_dest

    # ── D. composition of the 14 ─────────────────────────────────────────────
    print(f"\n{'-' * 104}\nD · Composition audit of the 14 patients\n{'-' * 104}")
    src = pd.read_csv(paths.LABEL_SOURCE, low_memory=False)
    cols = ["patient_uid", "tumor_site_raw", "msi_dmmr", "braf", "kras_subvariant",
            "subcohort", "sex", "age_at_diagnosis", "mpp", "image_format",
            "slide_size_bytes"]
    have = [c for c in cols if c in src.columns]
    meta = _patient_metadata(src[have + (["specimen_role"] if "specimen_role" in src else [])], set(per["patient_id"]))
    expected_peritoneal_ids = set(per["patient_id"].astype(str))
    if set(meta.index.astype(str)) != expected_peritoneal_ids:
        raise RuntimeError(
            "Peritoneal metadata coverage differs from the scored population: "
            f"missing={sorted(expected_peritoneal_ids - set(meta.index.astype(str)))[:5]}, "
            f"unexpected={sorted(set(meta.index.astype(str)) - expected_peritoneal_ids)[:5]}"
        )
    print(f"  {'patient':22s} {'KRAS':4s} {'logit':>7s} {'s42':>6s} {'s43':>6s} "
          f"{'s44':>6s} {'MSI':>9s} {'BRAF':>10s} {'variant':>8s} {'mpp':>5s} site")
    rows = []
    for i, r in per.iterrows():
        pid = r["patient_id"]
        m = meta.loc[pid] if pid in meta.index else None
        rec = {"patient": pid, "kras": int(r["label"]), "logit": float(r["mean_logit"]),
               **{f"seed{s}": float(seed_eta[s][i]) for s in aim2_loco_transport.SEEDS}}
        if m is not None:
            rec.update({c: (None if pd.isna(m[c]) else m[c]) for c in have if c != "patient_uid"})
            rec["metadata_slide_count"] = int(m["metadata_slide_count"])
        rows.append(rec)
        print(f"  {str(pid)[:22]:22s} {'MUT' if r['label'] else 'WT':4s} "
              f"{r['mean_logit']:7.2f} "
              + " ".join(f"{seed_eta[s][i]:6.2f}" for s in aim2_loco_transport.SEEDS)
              + f" {str(rec.get('msi_dmmr','?'))[:9]:>9s} {str(rec.get('braf','?'))[:10]:>10s}"
              f" {str(rec.get('kras_subvariant') or '-')[:8]:>8s}"
              f" {rec.get('mpp', float('nan')):5.3f} "
              + str(rec.get("tumor_site_raw", ""))[:34])
    out["patients"] = rows

    # compositional comparison against the rest of SurGen-M
    print(f"\n  composition vs the other {len(rest)} SurGen-M patients:")
    rest_meta_indexed = _patient_metadata(src, set(rest["patient_id"]))
    expected_rest_ids = set(rest["patient_id"].astype(str))
    if set(rest_meta_indexed.index.astype(str)) != expected_rest_ids:
        raise RuntimeError(
            "Non-peritoneal metadata coverage differs from the scored population: "
            f"missing={sorted(expected_rest_ids - set(rest_meta_indexed.index.astype(str)))[:5]}, "
            f"unexpected={sorted(set(rest_meta_indexed.index.astype(str)) - expected_rest_ids)[:5]}"
        )
    rest_meta = rest_meta_indexed.reset_index()
    peri_meta = meta
    out["composition"] = {}
    for col in ("msi_dmmr", "braf", "sex", "subcohort", "image_format"):
        if col not in src.columns:
            continue
        a = peri_meta[col].astype(str).value_counts(normalize=True)
        b = rest_meta[col].astype(str).value_counts(normalize=True)
        keys = sorted(set(a.index) | set(b.index))
        line = ", ".join(f"{k}: {100*a.get(k,0):.0f}% vs {100*b.get(k,0):.0f}%" for k in keys)
        out["composition"][col] = {k: [float(a.get(k, 0)), float(b.get(k, 0))] for k in keys}
        print(f"    {col:14s} {line}")
    for col in ("age_at_diagnosis", "mpp", "slide_size_bytes"):
        if col not in src.columns:
            continue
        a, b = peri_meta[col].astype(float), rest_meta[col].astype(float)
        out["composition"][col] = {"peritoneum_median": float(a.median()),
                                   "rest_median": float(b.median())}
        print(f"    {col:14s} median {a.median():.4g} vs {b.median():.4g}")

    # ── E. does Aim 1's BRAF confound explain it? ────────────────────────────
    # Aim 1 established that BRAF-mutant, KRAS-wild-type tumours are scored
    # KRAS-like (Why-D analysis D: BRAF +1.678 [+0.619, +2.641], the only term
    # surviving adjustment). The composition audit above shows the peritoneal
    # group is 7x enriched for BRAF-mutant. If that enrichment is the whole
    # story, the inversion should disappear among BRAF-wild-type patients.
    print(f"\n{'-' * 104}\nE · Is this Aim 1's BRAF confound, concentrated by site?"
          f"\n{'-' * 104}")
    braf = meta["braf"].reindex(per["patient_id"]).to_numpy()
    per_b = per.assign(braf=braf)
    for status in ("mutant", "wild_type", "unknown"):
        sub = per_b[per_b["braf"].eq(status)]
        if not len(sub):
            continue
        ym = sub["label"].to_numpy().astype(int)
        print(f"  BRAF {status:10s} n={len(sub):2d} ({int(ym.sum())} mut / "
              f"{int(len(ym) - ym.sum())} WT)   median logit "
              f"mut {np.median(sub.loc[sub.label.eq(1), 'mean_logit']) if ym.sum() else float('nan'):7.2f}"
              f"   WT {np.median(sub.loc[sub.label.eq(0), 'mean_logit']) if (ym == 0).any() else float('nan'):7.2f}")
    dsub = per_b[per_b["braf"].eq("wild_type")]
    yd, ed = dsub["label"].to_numpy().astype(int), dsub["mean_logit"].to_numpy()
    out["braf_stratified"] = {
        "n_braf_mutant": int((per_b["braf"] == "mutant").sum()),
        "braf_mutant_all_wild_type_kras": bool(
            (per_b.loc[per_b["braf"].eq("mutant"), "label"] == 0).all()),
        "braf_wt_only": {"n": int(len(dsub)), "n_mut": int(yd.sum()),
                         "auroc": auroc(yd, ed)},
    }
    print(f"\n  All {out['braf_stratified']['n_braf_mutant']} BRAF-mutant peritoneal "
          f"patients are KRAS-wild-type: "
          f"{out['braf_stratified']['braf_mutant_all_wild_type_kras']}")
    print(f"  Restricted to BRAF-wild-type (Aim 1's set-D logic): n={len(dsub)} "
          f"({int(yd.sum())} mut / {int(len(yd) - yd.sum())} WT)  "
          f"AUROC {auroc(yd, ed):.4f}")
    print("  If the enrichment were the whole story this would return to ~0.5.")

    inputs_after = material_input_identities(
        cap,
        upstream_eval_root=input_eval_root,
    )
    if inputs_after != inputs_before:
        raise RuntimeError("E2d2 material inputs changed during analysis")
    lineage.write_json_once(dest, out)
    print(f"\nWrote {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("report")
    x.add_argument("--cap", type=int, required=True, choices=aim2_loco_transport.E2A_CAPS)
    x.set_defaults(func=cmd_report)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
