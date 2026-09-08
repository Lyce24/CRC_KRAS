"""Build reports/final_v15_aim4_combined.

Descriptive combination of the sealed FINAL-v14 Aim 4 bundle (source-only k=32
vocabulary, reader-v2 categorical read) with the completed k32 blinded read of
the legacy target-inclusive k=32 vocabulary (rich free-text rubric plus targeted
follow-up), joined through a full 32-by-32 legacy-to-v14 prototype
correspondence computed from back-projected centroids.

Nothing here fits a model, changes a sealed estimate, gate, or interval, or
selects a feature. Every derived table is descriptive. Legacy numerical results
are not imported; only legacy descriptions, keys and geometry are used.

Usage:
  .venv/bin/python tools/build_final_v15_aim4_combined.py            # tables, figures, draft receipt
  .venv/bin/python tools/build_final_v15_aim4_combined.py --finalize # re-hash outputs incl. documents
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports/final_v15_aim4_combined"
TAB = OUT / "tables"
FIG = OUT / "figures"
RUN = REPO / "reports/reruns/final_v14_additions_20260903"
V14T = REPO / "reports/final_v14/tables"
LEGACY = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim4_cap8192_corrected_v2_20260820")
K32 = REPO / "reviews/k32"
V14M = REPO / "reviews/v14_xlsx/FOR_PATHOLOGIST/montages"

INPUTS = {
    "legacy_vocab": LEGACY / "vocabulary/vocab_k32.npz",
    "legacy_vocab_json": LEGACY / "vocabulary/vocab_k32.json",
    "legacy_key_base": LEGACY / "review_bundles/k32/base/unblinding_key_DO_NOT_SHARE.csv",
    "legacy_key_addendum": LEGACY / "review_bundles/k32/attention_addendum/unblinding_key_DO_NOT_SHARE.csv",
    "k32_form": K32 / "review_form.csv",
    "k32_structured": K32 / "completed_review_structured.json",
    "k32_completed_md": K32 / "completed_review.md",
    "ref_vocab": RUN / "e4v_pre_reader/vocabularies/reference/variants/k32_seed20260819/vocabulary.npz",
    "ref_pca": RUN / "e4v_pre_reader/vocabularies/reference/pca_basis.npz",
    "ref_profiles": RUN / "e4v_pre_reader/profiles/reference/patient_profiles.parquet",
    "source_manifest": RUN / "module_iii/inputs/source_manifest.csv",
    "sealed_correspondence": RUN / "e4v_correspondence_replay/runs/add086ae82ef5820b15847827e7eb8d9d25d7c3740cc69c38b449ab8e7ed1274/results.json",
    "v14_all32": V14T / "All32_evidence.csv",
    "v14_mentor": V14T / "Mentor_annotations.csv",
    "v14_reader_revision": V14T / "Reader_revision.csv",
    "v14_duplicates": V14T / "Duplicate_presentations.csv",
    "v14_m2_rih": V14T / "M2_RIH_interactions.csv",
    "v14_m2_sr1482": V14T / "M2_SR1482_interactions.csv",
    "v14_m2_dist": V14T / "M2_prototype_distributions.csv",
    "v14_m2_perf": V14T / "M2_performance.csv",
    "v14_m3": V14T / "M3_task_summary.csv",
    "v14_m1": V14T / "M1_all_estimands.csv",
}

SHORT = {
    "malignant gland-forming epithelium/gland–lumen": "malignant gland-forming",
    "malignant solid or poorly differentiated epithelium": "malignant solid/poorly differentiated",
    "extracellular mucin/mucinous pattern": "extracellular mucin",
    "normal or benign colonic epithelium": "benign colonic epithelium",
    "desmoplastic": "desmoplastic",
    "benign fibrosis": "benign fibrosis",
    "smooth muscle": "smooth muscle",
    "lymphoid/inflammatory tissue": "lymphoid/inflammatory",
    "necrosis/debris": "necrosis/debris",
    "adipose": "adipose",
    "blood/vessel": "blood/vessel",
    "mixed/other interpretable": "mixed/other",
}

# Analyst-curated descriptive agreement between the two blinded reads. Same
# single reader in both sessions; this is not interobserver reliability.
CURATED_AGREEMENT = {
    "M01": ("agree", "Both reads: smooth muscle."),
    "M02": ("agree", "k32: non-tumour loose connective tissue with red-cell and smooth-muscle tiles; v14: benign fibrosis with fibrin and cautery."),
    "M03": ("weak map", "Best v14 match below cosine 0.80; both reads describe gland-forming tumour."),
    "M04": ("agree", "Both reads: extracellular mucin; k32 adds floating tumour clusters and focal signet-ring morphology."),
    "M05": ("agree", "Both reads: gland-forming tumour; k32 adds moderate differentiation with luminal material confined to glands."),
    "M06": ("partial", "k32: tumour-stroma interface with focal desmoplasia; v14: benign fibrosis with inflammatory cells and one desmoplastic tile."),
    "M07": ("agree, richer", "Both reads: gland-forming tumour; k32 adds well differentiated, prominent true lumina, focal papillary architecture, limited intracellular mucin, defining character classed as geometry."),
    "M08": ("agree", "k32: well-differentiated tumour with narrow slit-like spaces; v14: malignant gland-forming."),
    "M09": ("agree", "Both reads: dirty necrosis / necrosis and debris."),
    "M10": ("ambiguous map", "Hungarian and best-cosine matches differ (v14 0 vs 28); k32: tumour with intermixed immune and stromal cells; v14 0 is a mixed fibrosis/lymphoid/tumour cluster."),
    "A01": ("agree", "k32 (M11 in the follow-up): benign goblet-rich colonic glands with two indeterminate tiles; v14: benign colonic epithelium with two tiles that appear malignant or dysplastic."),
}

V14_MONTAGE_CODE = {}  # filled from Mentor_annotations


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- correspondence
def correspondence() -> tuple[pd.DataFrame, dict]:
    lz = np.load(INPUTS["legacy_vocab"])
    rv = np.load(INPUTS["ref_vocab"])
    rb = np.load(INPUTS["ref_pca"])
    lc = lz["centroids"] @ lz["pca_components"] + lz["pca_mean"]
    rc = rv["centroids"] @ rb["pca_components"] + rb["pca_mean"]
    A = lc / np.linalg.norm(lc, axis=1, keepdims=True)
    B = rc / np.linalg.norm(rc, axis=1, keepdims=True)
    C = A @ B.T
    rows_i, cols_j = linear_sum_assignment(-C)
    hung = dict(zip(rows_i.tolist(), cols_j.tolist()))
    rows = []
    for i in range(32):
        order = np.argsort(-C[i])
        rows.append({
            "legacy_prototype": i,
            "v14_hungarian": hung[i],
            "cosine_hungarian": float(C[i, hung[i]]),
            "v14_best": int(order[0]),
            "cosine_best": float(C[i, order[0]]),
            "v14_second_best": int(order[1]),
            "cosine_second_best": float(C[i, order[1]]),
            "hungarian_equals_best": bool(hung[i] == order[0]),
            "geometry_pass_0_80": bool(C[i, hung[i]] >= 0.80),
        })
    df = pd.DataFrame(rows)
    sealed = json.loads(INPUTS["sealed_correspondence"].read_text())
    anchors = {}
    for name in ("p17", "p28"):
        i = int(name[1:])
        s = sealed["axes"][name]["canonical_geometry"]
        ours = df.loc[i]
        ok = (int(ours.v14_hungarian) == int(s["prototype_id"])) and abs(float(ours.cosine_hungarian) - float(s["cosine"])) < 1e-6
        anchors[name] = {"sealed_prototype": int(s["prototype_id"]), "sealed_cosine": float(s["cosine"]),
                         "recomputed_prototype": int(ours.v14_hungarian), "recomputed_cosine": float(ours.cosine_hungarian), "match": bool(ok)}
        assert ok, f"anchor {name} does not reproduce the sealed replay: {anchors[name]}"
    return df, anchors


# ---------------------------------------------------------------- reader concordance
def reader_concordance(corr: pd.DataFrame) -> pd.DataFrame:
    kb = pd.read_csv(INPUTS["legacy_key_base"])
    ka = pd.read_csv(INPUTS["legacy_key_addendum"])
    key = pd.concat([kb, ka])
    mont = key.groupby("montage_id").agg(
        legacy_prototype=("prototype", lambda s: int(sorted(set(s))[0])),
        n_tiles=("slot", "size"),
        selection_stratum=("selection_stratum", lambda s: ",".join(sorted(set(s)))),
        cohorts=("cohort", lambda s: ", ".join(f"{k} {v}" for k, v in sorted(s.value_counts().items()))),
        roles=("role", lambda s: ", ".join(f"{k} {v}" for k, v in sorted(s.value_counts().items()))),
        tile_kinds=("kind", lambda s: ", ".join(f"{k} {v}" for k, v in sorted(s.value_counts().items()))),
    ).reset_index()
    for _, r in key.groupby("montage_id"):
        assert r["prototype"].nunique() == 1
    form = pd.read_csv(INPUTS["k32_form"]).fillna("")
    fields = ["architecture", "mucin", "dirty_necrosis", "differentiation", "desmoplasia_stroma",
              "budding_invasion", "immune_infiltration", "normal_organ_tissue", "artifact", "free_text"]
    base_desc = {}
    for _, r in form.iterrows():
        parts = [f"{f}: {str(r[f]).strip()}" for f in fields if str(r[f]).strip()]
        base_desc[r["montage_id"]] = "; ".join(parts)
    struct = json.loads(INPUTS["k32_structured"].read_text())
    canon = {m: v["canonical_description"] for m, v in struct["montages"].items()}
    canon["A01"] = canon.pop("M11")  # addendum montage A01 was presented as M11 in the follow-up form
    mentor = pd.read_csv(INPUTS["v14_mentor"]).fillna("")
    mentor_by = {int(r["Concept"]): r for _, r in mentor.iterrows()}
    rows = []
    for _, m in mont.iterrows():
        lp = int(m.legacy_prototype)
        c = corr.loc[lp]
        v = mentor_by[int(c.v14_hungarian)]
        vb = mentor_by[int(c.v14_best)]
        agree, why = CURATED_AGREEMENT[m.montage_id]
        rows.append({
            "k32_montage": m.montage_id + (" (M11 in follow-up)" if m.montage_id == "A01" else ""),
            "legacy_prototype": lp,
            "v14_prototype": int(c.v14_hungarian),
            "cosine": round(float(c.cosine_hungarian), 3),
            "v14_best_if_different": "" if c.hungarian_equals_best else f"{int(c.v14_best)} ({c.cosine_best:.3f})",
            "k32_selection_stratum": m.selection_stratum,
            "k32_tile_cohorts": m.cohorts,
            "k32_tile_roles": m.roles,
            "k32_tile_kinds": m.tile_kinds,
            "k32_base_form_read": base_desc.get(m.montage_id, ""),
            "k32_followup_canonical_description": canon.get(m.montage_id, ""),
            "v14_blinded_code": v["Controlling code"],
            "v14_primary_category": v["Primary category"],
            "v14_secondary": "; ".join(x for x in [v["Secondary 1"], v["Secondary 2"]] if x),
            "v14_comment": v["Mentor comment"],
            "v14_confidence": v["Confidence"],
            "v14_artifact_flag": v["Artifact flag"],
            "descriptive_agreement": agree,
            "agreement_rationale": why,
        })
    df = pd.DataFrame(rows)
    order = ["M07", "M04", "A01", "M01", "M02", "M08", "M09", "M05", "M06", "M10", "M03"]
    df["_o"] = df["k32_montage"].str.slice(0, 3).map({m: i for i, m in enumerate(order)})
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


# ---------------------------------------------------------------- reference-vocabulary descriptives
def reference_descriptives() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prof = pd.read_parquet(INPUTS["ref_profiles"])
    ab = [f"prototype_{i:02d}" for i in range(32)]
    assert len(prof) == 1239
    man = pd.read_csv(INPUTS["source_manifest"])
    pm = man.groupby("patient_id").agg(kras=("kras", "first"), sub=("kras_subvariant", "first")).reset_index()
    d = prof.merge(pm, on="patient_id", how="left")
    assert d["kras"].notna().all()
    # subcohort composition and family dominance
    comp = d.groupby("subcohort")[ab].mean().T * 100
    comp.index = range(32)
    comp = comp[["SR1482", "SR386", "TCGA-COAD", "TCGA-READ"]]
    tot = comp.sum(axis=1)
    fam = pd.DataFrame({"SurGen": comp["SR1482"] + comp["SR386"], "TCGA": comp["TCGA-COAD"] + comp["TCGA-READ"]})
    fam_share = fam.div(tot, axis=0)
    sub_share = comp.div(tot, axis=0)
    cls = []
    for i in range(32):
        f = fam_share.loc[i].idxmax(); fs = fam_share.loc[i].max()
        s = sub_share.loc[i].idxmax(); ss = sub_share.loc[i].max()
        if fs >= 0.85:
            cls.append(f"{s} dominant" if ss >= 0.85 else f"{f} dominant")
        else:
            cls.append("shared")
    comp_out = comp.round(2).copy()
    comp_out.insert(0, "prototype", range(32))
    comp_out["dominant_family_share"] = fam_share.max(axis=1).round(3).values
    comp_out["site_class"] = cls
    # KRAS by subcohort
    rows = []
    for i, c in enumerate(ab):
        g = d.groupby(["subcohort", "kras"])[c].mean().unstack() * 100
        row = {"prototype": i}
        for s in ["SR1482", "SR386", "TCGA-COAD", "TCGA-READ"]:
            row[f"{s}_WT"] = round(float(g.loc[s, "wild_type"]), 3)
            row[f"{s}_mutant"] = round(float(g.loc[s, "mutant"]), 3)
        rows.append(row)
    kras_sub = pd.DataFrame(rows)
    # allele groups
    def grp(r):
        if r["kras"] != "mutant":
            return "WT"
        s = str(r["sub"])
        if s in ("G12D", "G12V", "G12C"):
            return s
        if s.startswith("G12"):
            return "other G12"
        if s.startswith("G13"):
            return "G13"
        return "other mutant"
    d["allele_group"] = d.apply(grp, axis=1)
    order = ["WT", "G12D", "G12V", "G12C", "other G12", "G13", "other mutant"]
    al = d.groupby("allele_group")[ab].mean().reindex(order).T * 100
    al.index = range(32)
    n = d["allele_group"].value_counts().reindex(order)
    al.columns = [f"{c} (n={int(n[c])})" for c in al.columns]
    al = al.round(2)
    al.insert(0, "prototype", range(32))
    return comp_out, kras_sub, al, d[["patient_id", "subcohort", "kras", "allele_group"]]


# ---------------------------------------------------------------- named panel
def named_panel(comp: pd.DataFrame, conc: pd.DataFrame) -> pd.DataFrame:
    a = pd.read_csv(INPUTS["v14_all32"])
    k32_by_v14 = {}
    for _, r in conc.iterrows():
        if r["descriptive_agreement"] in ("weak map", "ambiguous map"):
            continue
        k32_by_v14[int(r["v14_prototype"])] = r["k32_followup_canonical_description"] or r["k32_base_form_read"]
    def fmt(e, lo, hi):
        return "" if pd.isna(e) else f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    rows = []
    for _, r in a.iterrows():
        i = int(r["prototype_id"])
        rows.append({
            "prototype": i,
            "v14_label": SHORT.get(r["primary_category"], r["primary_category"]),
            "v14_secondary": "; ".join(SHORT.get(x, x) for x in [r["secondary_category_1"], r["secondary_category_2"]] if isinstance(x, str) and x),
            "confidence": r["confidence_1_to_5"],
            "artifact_flag": r["artifact_uninterpretable"],
            "site_class": comp.loc[i, "site_class"],
            "mapped_25": r["attribution_status"] == "ATTRIBUTABLE_REF",
            "kras_effect_sd": fmt(r["abundance__kras__standardized_effect"], r["abundance__kras__standardized_effect__ci95_lower"], r["abundance__kras__standardized_effect__ci95_upper"]),
            "kras_q": r["abundance__kras__q_value_bh"],
            "score_effect_sd": fmt(r["abundance__oof_score__standardized_effect"], r["abundance__oof_score__standardized_effect__ci95_lower"], r["abundance__oof_score__standardized_effect__ci95_upper"]),
            "score_q": r["abundance__oof_score__q_value_bh"],
            "attention_score_effect_sd": fmt(r["attention__oof_score__standardized_effect"], r["attention__oof_score__standardized_effect__ci95_lower"], r["attention__oof_score__standardized_effect__ci95_upper"]),
            "attention_score_q": r["attention__oof_score__q_value_bh"],
            "attention_kras_effect_sd": fmt(r["attention__kras__standardized_effect"], r["attention__kras__standardized_effect__ci95_lower"], r["attention__kras__standardized_effect__ci95_upper"]),
            "attention_kras_q": r["attention__kras__q_value_bh"],
            "msi_q": r["abundance__msi_dmmr__q_value_bh"],
            "braf_q": r["abundance__braf__q_value_bh"],
            "subcohort_q": r["abundance__source_subcohort__q_value_bh"],
            "site_q": r["abundance__tumor_site_group__q_value_bh"],
            "logistic_sign_pos_neg": "" if pd.isna(r["logistic__positive"]) else f"{int(r['logistic__positive'])}/{int(r['logistic__negative'])}",
            "k32_description_via_correspondence": k32_by_v14.get(i, ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tissue-role and target composition
def tissue_role_named(panel: pd.DataFrame) -> pd.DataFrame:
    lab = dict(zip(panel.prototype, panel.v14_label))
    out = []
    for fam, key in (("RIH", "v14_m2_rih"), ("SR1482", "v14_m2_sr1482")):
        d = pd.read_csv(INPUTS[key])
        for _, r in d.iterrows():
            i = int(r.prototype_id)
            out.append({
                "family": fam, "prototype": i, "v14_label": lab[i],
                "primary_WT_mean": r.primary_WT_mean, "metastatic_WT_mean": r.metastatic_WT_mean,
                "role_shift_WT": r.role_shift_WT, "role_shift_WT_ci95": f"[{r.role_shift_WT_ci95_low:+.3f}, {r.role_shift_WT_ci95_high:+.3f}]",
                "primary_mutant_mean": r.primary_mutant_mean, "metastatic_mutant_mean": r.metastatic_mutant_mean,
                "role_shift_mutant": r.role_shift_mutant, "role_shift_mutant_ci95": f"[{r.role_shift_mutant_ci95_low:+.3f}, {r.role_shift_mutant_ci95_high:+.3f}]",
                "interaction": r.interaction, "interaction_bh_q": r.bh_q,
            })
    return pd.DataFrame(out)


def target_composition(panel: pd.DataFrame) -> pd.DataFrame:
    lab = dict(zip(panel.prototype, panel.v14_label))
    d = pd.read_csv(INPUTS["v14_m2_dist"])
    d["prototype"] = d["prototype"].str.replace("prototype_", "").astype(int)
    p = d.pivot_table(index="prototype", columns=["cohort", "label"], values="mean") * 100
    p.columns = [f"{c}_{'mutant' if l == 1 else 'WT'}" for c, l in p.columns]
    p = p.round(2)
    p.insert(0, "v14_label", [lab[i] for i in p.index])
    p.insert(0, "prototype", p.index)
    return p.reset_index(drop=True)


# ---------------------------------------------------------------- figures
def figures(conc: pd.DataFrame) -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default(size=30); small = ImageFont.load_default(size=24)
    code = dict(zip(conc["v14_prototype"], conc["v14_blinded_code"]))
    lp = dict(zip(conc["k32_montage"].str.slice(0, 3), conc["legacy_prototype"]))
    vp = dict(zip(conc["k32_montage"].str.slice(0, 3), conc["v14_prototype"]))
    cs = dict(zip(conc["k32_montage"].str.slice(0, 3), conc["cosine"]))
    def kpath(m):
        return (LEGACY / "review_bundles/k32/attention_addendum/packet/montages/A01.jpg") if m == "A01" else (LEGACY / f"review_bundles/k32/base/packet/montages/{m}.jpg")
    def build(pairs, title, path):
        W, H = 1024, 768; s = 0.5; w, h = int(W * s), int(H * s); pad = 20; head = 70; cap = 44
        img = Image.new("RGB", (2 * w + 3 * pad, head + len(pairs) * (h + cap + pad) + pad), "white")
        dr = ImageDraw.Draw(img); dr.text((pad, 18), title, fill="black", font=font)
        y = head
        for m in pairs:
            a = Image.open(kpath(m)).convert("RGB").resize((w, h)); b = Image.open(V14M / f"{code[vp[m]]}.jpg").convert("RGB").resize((w, h))
            img.paste(a, (pad, y)); img.paste(b, (2 * pad + w, y))
            dr.text((pad, y + h + 6), f"k32 {m}: legacy prototype {lp[m]}", fill="black", font=small)
            dr.text((2 * pad + w, y + h + 6), f"v14 concept {vp[m]} ({code[vp[m]]}), cosine {cs[m]:.3f}", fill="black", font=small)
            y += h + cap + pad
        img.save(path); return path
    FIG.mkdir(parents=True, exist_ok=True)
    p1 = build(["M07", "M04", "A01"], "Two vocabularies, same patterns: 22, 17, 23", FIG / "fig_two_vocabularies_key_patterns.png")
    p2 = build(["M01", "M02", "M08", "M09", "M05"], "Concordant pairs: muscle, fibrosis, tumour, necrosis", FIG / "fig_two_vocabularies_concordant_pairs.png")
    p3 = build(["M06", "M10", "M03"], "Partial, ambiguous and weak correspondences", FIG / "fig_two_vocabularies_partial_pairs.png")
    return [p1, p2, p3]


# ---------------------------------------------------------------- main
def main(finalize: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True); TAB.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    corr, anchors = correspondence()
    conc = reader_concordance(corr)
    comp, kras_sub, allele, _ = reference_descriptives()
    panel = named_panel(comp, conc)
    roles = tissue_role_named(panel)
    tcomp = target_composition(panel)
    corr.to_csv(TAB / "legacy_to_v14_correspondence.csv", index=False)
    conc.to_csv(TAB / "k32_v14_reader_concordance.csv", index=False)
    comp.to_csv(TAB / "reference_vocabulary_subcohort_composition.csv", index=False)
    kras_sub.to_csv(TAB / "reference_vocabulary_kras_by_subcohort.csv", index=False)
    allele.to_csv(TAB / "reference_vocabulary_by_allele_group.csv", index=False)
    panel.to_csv(TAB / "named_concept_panel.csv", index=False)
    roles.to_csv(TAB / "tissue_role_shifts_named.csv", index=False)
    tcomp.to_csv(TAB / "target_cohort_composition_named.csv", index=False)
    figs = figures(conc)
    key = {
        "anchors": anchors,
        "site_class_counts": comp["site_class"].value_counts().to_dict(),
        "shared_prototypes": comp.loc[comp.site_class == "shared", "prototype"].tolist(),
        "concordance_counts": conc["descriptive_agreement"].value_counts().to_dict(),
        "pairs_cos_ge_0_95": int((conc["cosine"] >= 0.95).sum()),
        "legacy_vocab": json.loads(INPUTS["legacy_vocab_json"].read_text()),
    }
    (TAB / "key_numbers.json").write_text(json.dumps(key, indent=1, default=str))
    receipt = {
        "schema_version": 1, "bundle": "FINAL-v15 Aim 4 combined (v14 + k32)",
        "status": "FINAL" if finalize else "DRAFT_TABLES_ONLY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Descriptive combination; no fit, gate, interval or sealed estimate changed; legacy numerical results not imported.",
        "correspondence_method": "Both k=32 centroid sets back-projected from their own 64-component PCA to the 1024-d UNI-v1 space; cosine on L2-normalized vectors; joint one-to-one Hungarian maximum-cosine assignment; best-cosine match retained alongside.",
        "anchor_verification": anchors,
        "inputs": {k: identity(p) for k, p in INPUTS.items()},
        "outputs": {str(p.relative_to(OUT)): identity(p) for p in sorted(list(TAB.glob("*")) + list(FIG.glob("*")))},
        "documents": {p.name: identity(p) for p in sorted(OUT.glob("*.md"))} if finalize else {},
        "builder": identity(Path(__file__)),
    }
    (OUT / "Bundle_receipt.json").write_text(json.dumps(receipt, indent=1))
    # console renderings for document authoring
    print("ANCHORS", json.dumps(anchors))
    print("KEY", json.dumps({k: v for k, v in key.items() if k != "legacy_vocab"}))
    print("\n## concordance\n" + md_table(conc[["k32_montage", "legacy_prototype", "v14_prototype", "cosine", "v14_best_if_different", "k32_followup_canonical_description", "k32_base_form_read", "v14_primary_category", "v14_comment", "descriptive_agreement"]]))
    print("\n## composition\n" + md_table(comp, "{:.2f}"))
    print("\n## kras by subcohort (17,19,22,23)\n" + md_table(kras_sub[kras_sub.prototype.isin([17, 19, 22, 23])], "{:.2f}"))
    print("\n## allele\n" + md_table(allele[allele.prototype.isin([17, 19, 22, 23])], "{:.2f}"))
    print("\n## panel (mapped 25)\n" + md_table(panel[panel.mapped_25][["prototype", "v14_label", "site_class", "kras_effect_sd", "kras_q", "score_effect_sd", "score_q", "attention_score_effect_sd", "attention_score_q", "attention_kras_effect_sd", "attention_kras_q", "msi_q", "braf_q"]], "{:.3g}"))
    print("\n## roles SR1482 (|shift|>=0.02)\n" + md_table(roles[(roles.family == "SR1482") & ((roles.role_shift_WT.abs() >= 0.02) | (roles.role_shift_mutant.abs() >= 0.02) | roles.prototype.isin([17, 22, 19]))][["prototype", "v14_label", "primary_WT_mean", "metastatic_WT_mean", "role_shift_WT", "role_shift_WT_ci95", "primary_mutant_mean", "metastatic_mutant_mean", "role_shift_mutant", "role_shift_mutant_ci95", "interaction_bh_q"]], "{:.3f}"))
    print("\n## roles RIH (|shift|>=0.02)\n" + md_table(roles[(roles.family == "RIH") & ((roles.role_shift_WT.abs() >= 0.02) | (roles.role_shift_mutant.abs() >= 0.02) | roles.prototype.isin([17, 22, 19]))][["prototype", "v14_label", "primary_WT_mean", "metastatic_WT_mean", "role_shift_WT", "role_shift_WT_ci95", "primary_mutant_mean", "metastatic_mutant_mean", "role_shift_mutant", "role_shift_mutant_ci95", "interaction_bh_q"]], "{:.3f}"))
    print("\n## target composition (17,22,23,11)\n" + md_table(tcomp[tcomp.prototype.isin([11, 17, 22, 23])], "{:.2f}"))
    print("\nFIGURES", [str(p) for p in figs])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--finalize", action="store_true")
    main(ap.parse_args().finalize)
