#!/usr/bin/env python3
"""Publish the finalized reader-v2 annotation successor without refitting models.

The original report, receipts, code and scientific analyses are retained. Current
reader metadata is replaced from a sealed, post-analysis, revised-rubric intake.
The manifest remaps historical report paths to a byte-identical snapshot; the
existing final_v14_bundle_receipt.py verifier also verifies the successor.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import io
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import final_v14_bundle_receipt as bundle

REPORT = REPO / "reports/final_v14"
SNAPSHOT = REPO / "reports/snapshots/final_v14_pre_reader_v2_20260905"
SNAPSHOT_ID = SNAPSHOT.with_name(SNAPSHOT.name + ".identity.json")
READER = REPO / "reports/reruns/final_v14_additions_20260903/e4v_post_reader_xlsx_v2"
STAGING = REPO / "reports/_staging/final_v14_review_v2"
OLD_WIDE = REPO / "reports/final_v14_review_interpretation/All32_evidence.csv"
CONTROL_FILES = {"source_manifest.json", "Audit.md", "final_bundle_receipt.json", "final_bundle_receipt.json.seal.json"}


def identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def read(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, values: list[dict]) -> None:
    if not values:
        raise ValueError(f"Refusing empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def snapshot_mapping() -> dict[str, dict]:
    result = {}
    for pin in read(SNAPSHOT_ID)["files"]:
        bundle.verify_identity(Path(pin["path"]), pin)
        result[pin["original_path"]] = pin
    return result


def load_reader():
    from tools import final_v14_reader_v2 as intake
    intake.verify()
    names = read(READER / "naming/naming_freeze.json")
    literal = read(READER / "literal_intake/literal_response_values.json")
    return names, literal


def revision_summary(names: dict, literal: dict) -> dict:
    rs = literal["rows"]
    by_field = {key: sum(r.get(key) not in (None, "") for r in rs) for key in (
        "primary_category", "confidence_1_to_5", "artifact_uninterpretable", "review_status", "reviewer_id", "review_date", "blinding_attestation")}
    repeat = names["repeatability"]
    return {
        "revision": 2, "status": "FINALIZED_RETROSPECTIVE_REVISED_RUBRIC",
        "workbook_path": str(READER / "raw_return/completed_review_form_v2.xlsx"),
        "workbook_sha256": identity(READER / "raw_return/completed_review_form_v2.xlsx")["sha256"],
        "presentations": len(rs), "primary_category_count": by_field["primary_category"],
        "confidence_count": by_field["confidence_1_to_5"], "artifact_response_count": by_field["artifact_uninterpretable"],
        "controlling_count": len(names["controlling_reads"]),
        "controlling_eligible_count": len(names["descriptive_eligible_ref"]),
        "eligible_ref_descriptive": names["descriptive_eligible_ref"],
        "repeatability_numerator": repeat["numerator"], "repeatability_denominator": repeat["denominator"],
        "repeatability_exact": repeat["exact"], "repeatability_partial": repeat["partial"],
        "repeatability_ci95": repeat["ci95_clopper_pearson"],
        "confidence_counts": dict(Counter(str(r["confidence_1_to_5"]) for r in rs)),
        "artifact_flag_count": sum(r["artifact_uninterpretable"] == "yes" for r in rs),
        "metadata_recorded_counts": {k: by_field[k] for k in ("review_status", "reviewer_id", "review_date", "blinding_attestation")},
        "historical_name_gate": "NAME_GATE_FAIL",
        "response_only_gate_status": names["response_only_gate_status"],
        "interpretation_scope": "Finalized single-reader revised-rubric annotation after completed analyses; no retrospective upgrade of prospective naming or feature selection.",
        "numerical_models_and_inference_changed": False,
    }


def evidence_table(names: dict, literal: dict) -> list[dict]:
    """Replace only reader fields; statistical strings are never parsed/rounded."""
    old = rows(OLD_WIDE)
    current = {r["prototype_id"]: r for r in names["controlling_reads"]}
    raw = {r["blinded_code"]: r for r in literal["rows"]}
    pairs = {r["prototype_id"]: r for r in names["repeatability"]["pairs"]}
    for row in old:
        pid = int(row["prototype_id"])
        controlling = current[pid]
        response = controlling["response"]
        row["historical_eligible_for_named_ref"] = row["eligible_for_named_ref"]
        row.update(response)
        row["eligible_for_named_ref"] = controlling["eligible_for_named_ref"]
        row["name_gate_status"] = "NAME_GATE_FAIL"
        row["reader_revision"] = 2
        row["review_scope"] = "FINALIZED_RETROSPECTIVE_REVISED_RUBRIC"
        row["response_eligibility_descriptive"] = pid in names["descriptive_eligible_ref"]
        row["revised_named_model_authorized"] = False
        row["original_montage"] = f"../../../reviews/v14_xlsx/FOR_PATHOLOGIST/montages/{response['blinded_code']}.jpg"
        row["original_montage_uri_base"] = "reports/final_v14/tables/"
        pair = pairs.get(pid)
        row["duplicate_category_concordance_descriptive"] = pair["categorical_agreement_descriptive_only"] if pair else None
        duplicate = raw[pair["duplicate_code"]] if pair else None
        for field in response:
            row["duplicate_" + field] = duplicate.get(field) if duplicate else None
    return old


def verify_authoritative_statistics() -> None:
    """Replay the original coordinate table from its sealed context inputs."""
    from tools import final_v14_review_interpretation as legacy
    manifest = read(OLD_WIDE.parent / "input_identity_manifest.json")
    pin = next(p for p in manifest["artifacts"] if p["path"] == str(OLD_WIDE))
    bundle.verify_identity(OLD_WIDE, pin)
    original_mentor = legacy.MENTOR
    try:
        legacy.MENTOR = SNAPSHOT / "tables/Mentor_annotations.csv"
        data, _ = legacy.load_inputs()
    finally:
        legacy.MENTOR = original_mentor
    expected = list(csv.DictReader(io.StringIO(legacy.csv_text(legacy.assemble(data)["All32_evidence"]))))
    for a, b in zip(expected, rows(OLD_WIDE), strict=True):
        for key in a:
            if key.startswith(("abundance__", "attention__", "ridge__", "logistic__")) and a[key] != b[key]:
                raise ValueError(f"Original coordinate table differs from sealed inputs: p{a['prototype_id']}/{key}")


def md_table(values: list[dict], columns: list[tuple[str, str]]) -> str:
    def safe(value):
        return str("" if value is None else value).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join(["| " + " | ".join(title for _, title in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"] +
                     ["| " + " | ".join(safe(row[key]) for key, _ in columns) + " |" for row in values])


def reader_text(summary: dict) -> str:
    ci = summary["repeatability_ci95"]
    return f"""## Reader assessment and montage atlas

The finalized `completed_review_form_v2.xlsx` supersedes the original form for
current annotations (SHA-256 `{summary['workbook_sha256']}`). All 40 presentations
now have a primary category, confidence, and artifact rating: 34 confidence
ratings are 5/5 and six are 4/5. One montage, the controlling read of concept 11,
is flagged artifact/uninterpretable. Under the revised rubric, 31/32 controlling
reads meet the response-based eligibility requirements; concept 11 is excluded.
The eight resampled-presentation pairs show 7/8 exact-or-partial agreements
(six exact, one partial; 95% Clopper–Pearson CI {100*ci[0]:.1f}–{100*ci[1]:.1f}%).
Concept 0 is the discordant pair. These assess joint prototype-presentation/name
repeatability, not same-image or multireader reliability.

The revised rubric explicitly separates **desmoplastic** from **benign fibrosis**.
Concepts 1 and 2 are labeled desmoplastic; concepts 4, 7, 10, 11, 24, 27, 30, and
31 are labeled benign fibrosis. The literal categories and all qualifications
are retained in the [updated annotation table](tables/Mentor_annotations.csv)
and [searchable montage atlas](mentor_atlas.html). Concept 0's primary category
is malignant solid/poorly differentiated epithelium, while its comment begins
“Most are benign fibrosis”; both are preserved without analyst adjudication.
Categories for the KRAS-associated concepts 22, 17, and 19 remain gland-forming
epithelium, extracellular mucin/mucinous pattern, and smooth muscle.

This finalized revision arrived **after the statistical analyses**, and the
ontology changed. Review status, reviewer ID, date, and blinding attestation
remain unrecorded in all 40 rows. Its improved response-based quality is reported
as revised descriptive evidence; it cannot retroactively establish the original
prospective naming gate or a new blinded session. The historical **NAME_GATE_FAIL**
and all ALL32 models, associations, target evaluations, and resolution verdicts
remain unchanged. No name-selected predictor is fitted. No missing response is
imputed. The raw original form, original gate, original report and receipts, and
all original montage bytes are preserved separately.

![Finalized review: complete ratings and resampled-presentation agreement](figures/aim4_reader_v2.png)

[Full review interpretation](Review_Interpretation.md) ·
[All 32 annotation/statistic rows](tables/All32_evidence.csv) ·
[All duplicate presentations](tables/Duplicate_presentations.csv) ·
[Versioned reader intake and provenance](../reruns/final_v14_additions_20260903/e4v_post_reader_xlsx_v2/receipt.json)

"""


def render_atlas(output: Path, names: dict, summary: dict) -> None:
    old = (SNAPSHOT / "mentor_atlas.html").read_text()
    prefix = old[:old.index("<header>")]
    cards = []
    for item in names["controlling_reads"]:
        r = item["response"]
        category, comment = r["primary_category"], r["free_text_description"] or ""
        code, pid = r["blinded_code"], item["prototype_id"]
        secondary = "; ".join(r[k] for k in ("secondary_category_1", "secondary_category_2") if r.get(k))
        img = f"../../reviews/v14_xlsx/FOR_PATHOLOGIST/montages/{code}.jpg"
        cards.append(f'<article data-search="{html.escape(category + " " + comment, quote=True)}"><h2>Concept {pid:02d} <small>{code}</small></h2>'
                     f'<a href="{img}"><img loading="lazy" src="{img}" alt="Original montage for concept {pid:02d}"></a>'
                     f'<h3>{html.escape(category)}</h3><p>{html.escape(comment)}</p>'
                     f'<p class="secondary">Secondary: {html.escape(secondary or "None recorded")}<br>Confidence: {r["confidence_1_to_5"]}/5 · Artifact/uninterpretable: {r["artifact_uninterpretable"]}</p></article>')
    header = '<header><h1>Finalized v2 mentor-annotated concept atlas</h1><p>32 source concepts · Original montages · Literal revised categories, comments, and ratings</p>'
    header += '<p>All 40 presentations have category, confidence and artifact ratings. Revised response eligibility: 31/32 controlling concepts; resampled-presentation agreement: 7/8 (6 exact, 1 partial). Desmoplastic and benign fibrosis are distinct rubric choices.</p>'
    header += '<p>The finalized revised rubric was received after the statistical analyses. Reviewer/date/attestation fields remain blank. These are descriptive single-reader annotations; original prospective naming status and all numerical models are unchanged.</p>'
    header += '<label for="search">Find a category or phrase</label><br><input id="search" type="search" placeholder="e.g. benign fibrosis, desmoplastic, mucin"><p id="count">32 concepts</p></header><main>'
    script = "</main><script>document.querySelector('#search').addEventListener('input',e=>{let n=0;for(const card of document.querySelectorAll('article')){card.hidden=!card.dataset.search.toLowerCase().includes(e.target.value.toLowerCase());if(!card.hidden)n++}document.querySelector('#count').textContent=n+' concepts'});</script></html>"
    (output / "mentor_atlas.html").write_text(prefix + header + "\n".join(cards) + script)


def review_figure(output: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0), gridspec_kw={"width_ratios": [1, 1.2]})
    axes[0].barh([1, 0], [6, 34], color=["#78aa9b", "#16826d"])
    axes[0].set(yticks=[1, 0], yticklabels=["Confidence 4/5", "Confidence 5/5"], xlim=(0, 40), xlabel="Presentations (40 total)")
    for y, n in ((1, 6), (0, 34)):
        axes[0].text(n+.5, y, str(n), va="center")
    axes[0].set_title("A  Complete scientific ratings", loc="left", weight="bold")
    axes[1].barh([2, 1, 0], [6, 1, 1], color=["#16826d", "#78aa9b", "#a28b70"])
    axes[1].set(yticks=[2, 1, 0], yticklabels=["Exact", "Partial", "Different"], xlim=(0, 8), xlabel="Resampled-presentation pairs (8 total)")
    for y, n in ((2, 6), (1, 1), (0, 1)):
        axes[1].text(n+.12, y, str(n), va="center")
    axes[1].set_title("B  Agreement: 7/8 [47.3%, 99.7%]", loc="left", weight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .045, "31/32 controlling reads meet revised response criteria; concept 11 is artifact-flagged.", fontsize=10)
    fig.text(.02, -.015, "Finalized revised rubric after analysis; descriptive single-reader evidence; historical gate unchanged.", fontsize=9)
    fig.tight_layout(rect=[0, .14, 1, 1])
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output / "figures" / f"aim4_reader_v2.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def replace_sheet(book, title: str, table: list[dict]) -> None:
    if title in book:
        del book[title]
    sheet = book.create_sheet(title[:31])
    sheet.append(list(table[0]))
    for row in table:
        values = [json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for v in row.values()]
        sheet.append(values)
        for cell, value in zip(sheet[sheet.max_row], values):
            if isinstance(value, str):
                cell.data_type = "s"
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="244D42")
    for column in sheet.columns:
        sheet.column_dimensions[column[0].column_letter].width = min(60, max(14, max(len(str(c.value or "")) for c in column[:35]) + 2))
        for cell in column:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def build(output: Path = STAGING) -> dict:
    snapshot_mapping()
    names, literal = load_reader()
    summary = revision_summary(names, literal)
    if summary["controlling_eligible_count"] != 31 or summary["repeatability_numerator"] != 7:
        raise ValueError("Finalized review differs from the independently audited revision")
    output.mkdir(parents=True, exist_ok=True)
    for path in SNAPSHOT.rglob("*"):
        if path.is_file() and path.name not in CONTROL_FILES:
            target = output / path.relative_to(SNAPSHOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    mentor = rows(READER / "tables/Mentor_annotations.csv")
    write_csv(output / "tables/Mentor_annotations.csv", mentor)
    wide = evidence_table(names, literal)
    write_csv(output / "tables/All32_evidence.csv", wide)
    duplicates = []
    raw = {r["blinded_code"]: r for r in literal["rows"]}
    for pair in names["repeatability"]["pairs"]:
        a, b = raw[pair["controlling_code"]], raw[pair["duplicate_code"]]
        duplicates.append({**pair, "controlling_primary_literal": a["primary_category"], "duplicate_primary_literal": b["primary_category"],
                           "controlling_comment_literal": a["free_text_description"], "duplicate_comment_literal": b["free_text_description"]})
    write_csv(output / "tables/Duplicate_presentations.csv", duplicates)
    write_csv(output / "tables/Finalized_review_rows.csv", literal["rows"])
    status_rows = [{"item": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value} for key, value in summary.items()]
    write_csv(output / "tables/Reader_revision.csv", status_rows)
    write_json(output / "reader_revision.json", summary)
    render_atlas(output, names, summary)
    review_figure(output, summary)
    original = (SNAPSHOT / "Results.md").read_text()
    start = original.index("## Reader assessment and montage atlas")
    end = original.index("All pooled and per-source-subcohort fidelity", start)
    original = original[:start] + reader_text(summary) + original[end:]
    original = original.replace("**All in-scope analyses have final scientific outputs. This rendered report awaits the separate final audit and bundle seal.**", "**Completed FINAL-v14 with the finalized reader-v2 annotation successor. All numerical analyses are retained; the current bundle receipt records the revision audit.**")
    original = original.replace("because NAME_GATE_FAIL independently applies", "because the original prospective NAME_GATE_FAIL remains fixed; the later revised-form annotations are descriptive")
    original += "\n\n### Finalized reader-v2 supplements\n\n" + "\n".join(
        f"- [{label}](tables/{name})" for label, name in (
            ("Reader revision and scope", "Reader_revision.csv"),
            ("All 40 finalized responses", "Finalized_review_rows.csv"),
            ("All 32 annotation/statistic joins", "All32_evidence.csv"),
            ("All eight resampled-presentation pairs", "Duplicate_presentations.csv"))) + "\n"
    (output / "Results.md").write_text(original)
    setup = (SNAPSHOT / "Experimental_Setup.md").read_text()
    a = setup.index("The accepted single mentor return")
    b = setup.index("\n\n| Module", a)
    setup = setup[:a] + "The finalized v2 review supplies the current literal annotations under a revised rubric that separates desmoplastic from benign fibrosis. All 40 category/confidence/artifact fields are complete; 31/32 controlling reads meet revised response criteria and 7/8 duplicate pairs agree (six exact, one partial; 95% CI 47.3–99.7%). It was received after analysis, with review status/reviewer/date/attestation still unrecorded. The original prospective NAME_GATE_FAIL and all ALL32 inference remain fixed; no revised named predictor or new blinding claim is introduced. Original and revised workbooks, montages, and the pre-revision report are preserved. The original 60-case/420-image whole-section packet remains **GENERATED-UNREAD**." + setup[b:]
    (output / "Experimental_Setup.md").write_text(setup)
    deviations = (SNAPSHOT / "Deviations.md").read_text()
    (output / "Deviations.md").write_text("# Finalized reader-v2 revision\n\nThe user finalized the revised workbook after completion of the numerical campaign. This supersedes the original annotations for reporting, not the frozen prospective gate or predictors. Two helper columns and the revised rubric are preserved literally; helper HYPERLINK formulas are not executed. The newly explicit desmoplastic/benign-fibrosis distinction is a reader-supplied category change, not analyst imputation. Missing session metadata and the category/comment discrepancy at concept 0 remain disclosed.\n\nThe historical report and all original source identities are retained in [the pre-revision snapshot](../snapshots/final_v14_pre_reader_v2_20260905.identity.json). The history below describes the original campaign and is not a claim that v2 ratings are missing.\n\n" + deviations)
    interpretation = "# FINAL-v14 finalized review-v2 interpretation\n\n" + reader_text(summary).replace("## Reader assessment and montage atlas\n", "")
    interpretation += "\n## All controlling annotations\n\n" + md_table(mentor, [("Concept", "Concept"), ("Primary category", "Literal primary"), ("Confidence", "Confidence"), ("Artifact flag", "Artifact"), ("Mentor comment", "Literal comment")])
    interpretation += "\n\n## All resampled-presentation pairs\n\n" + md_table(duplicates, [("prototype_id", "Concept"), ("controlling_primary_literal", "Controlling primary"), ("duplicate_primary_literal", "Resampled primary"), ("agreement", "Agreement")])
    interpretation += "\n\nThe revised labels characterize selected centroid-near examples, not cluster purity or a causal explanation. Full source discrimination (0.601), reconstruction R² (0.195), three KRAS-associated coordinates (22, 17, 19), 25 mapped association coordinates, transport/resolution verdicts, and all uncertainty are unchanged. [The complete 32-coordinate evidence table](tables/All32_evidence.csv) retains every original statistical value with the current reader metadata.\n"
    (output / "Review_Interpretation.md").write_text(interpretation)
    book = load_workbook(SNAPSHOT / "Aim4_Results.xlsx")
    for title, table in (("Mentor annotations", mentor), ("Reader revision", status_rows), ("Duplicate presentations", duplicates), ("Finalized review rows", literal["rows"]), ("All32 evidence", wide)):
        replace_sheet(book, title, table)
    scope = [{"item": "analysis_complete", "value": True}, {"item": "reader_revision", "value": 2}, {"item": "historical_name_gate", "value": "NAME_GATE_FAIL"}, {"item": "reader_scope", "value": summary["status"]}, {"item": "new_model_or_test", "value": False}, {"item": "whole-section packet", "value": "GENERATED-UNREAD"}]
    write_csv(output / "tables/Status_and_scope.csv", scope)
    replace_sheet(book, "Status and scope", scope)
    book.save(output / "Aim4_Results.xlsx")
    status = read(SNAPSHOT / "STATUS.json")
    status.update({"schema_version": 2, "status": "COMPLETED_READER_V2_REPORT", "completed_bundle": True, "pending": [], "reader_revision": summary,
                   "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "code": identity(Path(__file__)),
                   "parent_final_receipt": identity(SNAPSHOT / "final_bundle_receipt.json"), "accepted_reader_receipt": identity(READER / "receipt.json"),
                   "scope": summary["interpretation_scope"], "scientific_values_unchanged": True})
    status["inputs"] += [identity(READER / "receipt.json"), identity(READER / "naming/naming_freeze.json")]
    original_bindings = {item["csv"]: item["sheet"] for item in status["table_bindings"]}
    new_sheets = {"All32_evidence.csv": "All32 evidence", "Duplicate_presentations.csv": "Duplicate presentations",
                  "Finalized_review_rows.csv": "Finalized review rows", "Reader_revision.csv": "Reader revision"}
    status["table_bindings"] = [{"csv": str(p.relative_to(output)),
                                 "sheet": original_bindings.get(str(p.relative_to(output)), new_sheets.get(p.name)),
                                 "markdown": "Results.md", "markdown_role": "summary_or_linked_complete_table"}
                                for p in sorted((output / "tables").glob("*.csv"))]
    status["report_artifacts"] = [{**identity(f), "path": str(f.relative_to(output))} for f in sorted(output.rglob("*")) if f.is_file() and f.name not in CONTROL_FILES | {"STATUS.json"}]
    write_json(output / "STATUS.json", status)
    checks = verify_outputs(output)
    return {"status": "READY_FOR_READER_V2_PUBLICATION", "output": str(output), "checks": checks, "reader_revision": summary}


def verify_outputs(output: Path) -> dict:
    """Check every untouched numerical table and the reader joins separately."""
    names, literal = load_reader()
    verify_authoritative_statistics()
    changed = {"Mentor_annotations.csv", "Status_and_scope.csv"}
    untouched = []
    for old in (SNAPSHOT / "tables").glob("*.csv"):
        if old.name not in changed:
            if old.read_bytes() != (output / "tables" / old.name).read_bytes():
                raise ValueError(f"Numerical table changed: {old.name}")
            untouched.append(old.name)
    expected = evidence_table(names, literal)
    actual = rows(output / "tables/All32_evidence.csv")
    for a, b in zip(expected, actual, strict=True):
        for key, value in a.items():
            if b[key] != ("" if value is None else str(value)):
                raise ValueError(f"Reader/statistic mismatch: p{a['prototype_id']}/{key}")
    if rows(output / "tables/Mentor_annotations.csv") != rows(READER / "tables/Mentor_annotations.csv"):
        raise ValueError("Mentor table does not match sealed finalized reader")
    oldbook, newbook = load_workbook(SNAPSHOT / "Aim4_Results.xlsx", read_only=True), load_workbook(output / "Aim4_Results.xlsx", read_only=True)
    for sheet in oldbook:
        if sheet.title not in {"Mentor annotations", "Status and scope"} and list(sheet.values) != list(newbook[sheet.title].values):
            raise ValueError(f"Scientific workbook values changed: {sheet.title}")
    for item in read(output / "STATUS.json")["table_bindings"]:
        if item["sheet"] not in newbook:
            raise ValueError(f"CSV has no workbook sheet: {item['csv']}")
    for row in actual:
        if not (REPORT / "tables" / row["original_montage"]).resolve().is_file():
            raise ValueError("Broken original montage reference")
    for field in ("Primary category", "Mentor comment"):
        for row in rows(output / "tables/Mentor_annotations.csv"):
            if html.escape(row[field]) not in (output / "mentor_atlas.html").read_text():
                raise ValueError(f"Atlas missing literal {field}")
    for textfile in ("Results.md", "Experimental_Setup.md", "Review_Interpretation.md", "mentor_atlas.html"):
        text = (output / textfile).read_text()
        if "Only 1/32" in text or "ratings were largely absent" in text:
            raise ValueError(f"Stale reader-v1 statement in current prose: {textfile}")
    return {"numerical_csv_files_byte_identical": len(untouched), "all_original_numerical_workbook_values_identical": True,
            "all_32_statistical_fields_preserved": True, "literal_reader_joins": "PASS", "atlas_literal_content": "PASS"}


def publish(output: Path = STAGING) -> dict:
    checks = verify_outputs(output)
    mapping = snapshot_mapping()
    old_manifest = read(SNAPSHOT / "source_manifest.json")
    artifacts = {}
    for original in old_manifest["artifacts"]:
        pin = dict(original)
        if pin["path"] in mapping:
            pin["historical_path"] = pin["path"]
            pin["path"] = mapping[pin["path"]]["path"]
        artifacts[(pin["path"], pin["sha256"])] = pin
    # Check current output identities before any replacement; the historical
    # snapshot must exactly match the old report or the current successor.
    current = read(REPORT / "final_bundle_receipt.json")
    if current.get("reader_revision") != 2:
        for original, pin in mapping.items():
            bundle.verify_identity(Path(original), pin)
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in CONTROL_FILES:
            target = REPORT / path.relative_to(output)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    included = [SNAPSHOT_ID, Path(__file__), REPO / "tests/test_final_v14_review_v2_report.py",
                OLD_WIDE, OLD_WIDE.parent / "input_identity_manifest.json",
                REPO / "tools/final_v14_review_interpretation.py",
                REPO / "tools/final_v14_reader_v2.py", REPO / "tests/test_final_v14_reader_v2.py"]
    included += [p for p in SNAPSHOT.rglob("*") if p.is_file()]
    included += [p for p in READER.rglob("*") if p.is_file()]
    included += [p for p in REPORT.rglob("*") if p.is_file() and p.name not in CONTROL_FILES]
    for path in included:
        pin = {**identity(path), "verification": "SHA256_REPLAYED_NOW"}
        artifacts[(pin["path"], pin["sha256"])] = pin
    manifest = {"schema_version": 2, "status": "PASS_FINAL_CAMPAIGN_AUDIT", "reader_revision": 2,
                     "parent_manifest": identity(SNAPSHOT / "source_manifest.json"), "snapshot_identity": identity(SNAPSHOT_ID),
                     "historical_report_path_mapping": list(mapping.values()), "reader_update_checks": checks,
                     "reader_revision_summary": read(REPORT / "reader_revision.json"),
                     "inherited_campaign_audit": {k: v for k, v in old_manifest.items() if k != "artifacts"},
                     "artifacts": sorted(artifacts.values(), key=lambda x: (x["path"], x["sha256"]))}
    write_json(REPORT / "source_manifest.json", manifest)
    audit = "# FINAL-v14 finalized reader-v2 audit\n\nStatus: **PASS**. The finalized revised-rubric workbook supplies current annotations; all original numerical analyses and prospective gates are retained.\n\n"
    audit += f"Verified {checks['numerical_csv_files_byte_identical']} original numerical CSV files byte for byte, every unchanged numerical workbook cell, and all 32 annotation/statistic joins. The atlas reproduces literal revised categories and comments. The source manifest binds the original campaign artifacts through explicit historical report-to-snapshot mappings and the sealed reader-v2 intake.\n\n"
    audit += "Revised response criteria: 31/32 controlling reads; duplicate agreement 7/8 (six exact, one partial; 95% CI 47.3–99.7%). These are post-analysis descriptive results. Missing reviewer/date/attestation fields and the changed rubric prevent a retrospective claim of a new preregistered blinded session. No model or coordinate-association test was recomputed.\n\n"
    audit += "The pre-revision report, receipt and original code remain byte-identical in the archived snapshot. Inherited packed-input guards retain their original verification scope.\n"
    (REPORT / "Audit.md").write_text(audit)
    receipt = {"schema_version": 2, "status": "COMPLETED", "completed_bundle": True, "reader_revision": 2,
               "manifest": identity(REPORT / "source_manifest.json"), "audit": identity(REPORT / "Audit.md"),
               "report_status": identity(REPORT / "STATUS.json"), "verifier": identity(Path(__file__)),
               "parent_final_receipt": identity(SNAPSHOT / "final_bundle_receipt.json"), "reader_receipt": identity(READER / "receipt.json"),
               "completion_rule": "Finalized reader-v2 intake verified; original numeric CSV/workbook and statistical fields preserved; historical artifacts mapped to their byte-identical snapshot."}
    write_json(REPORT / "final_bundle_receipt.json", receipt)
    write_json(REPORT / "final_bundle_receipt.json.seal.json", {"status": "SEALED", "artifact": identity(REPORT / "final_bundle_receipt.json")})
    return verify()


def verify() -> dict:
    result = bundle.verify_bundle(REPORT)
    checks = verify_outputs(REPORT)
    snapshot_mapping()
    return {"status": "PASS", "reader_revision": 2, "bundle": result, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["build", "publish", "verify"])
    parser.add_argument("--output", type=Path, default=STAGING)
    args = parser.parse_args()
    result = build(args.output) if args.stage == "build" else publish(args.output) if args.stage == "publish" else verify()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
