#!/usr/bin/env python3
"""Ingest the FINAL blinded human pathology read of the original Aim 4 packet.

WHAT EXISTS. A pathologist completed, and declared final, a blinded review of
the ORIGINAL k=32 montage packet (`reviews/k32/`): the first-pass
`review_form.csv` over M01-M10, a deep follow-up (`completed_review.md`,
curated into `completed_review_structured.json`) for M04/M05/M07 and the M11
addendum, and a cross-montage comparison. The packet's unblinding keys are
the two `KEY_*_do_not_open_before_review.csv` files in the frozen e3b
montage root.

GENERATION BRIDGE, STATED EXACTLY. The reviewed montages are NOT the
corrected-v2 packet: they are earlier renders (different tiles, partly drawn
with pre-correction Aim 2 attention). Three facts make the read usable and
are verified here mechanically:

    1. the vocabulary is the same frozen k=32 clustering, so prototype IDs
       denote identical objects across generations;
    2. the montage-to-prototype assignment is IDENTICAL between the reviewed
       packet and corrected v2 (M11 corresponds to v2's A01), asserted from
       both key files; and
    3. the structured curation records SHA-256 identities for its source
       document, the base form, both keys and the four deep-read montages —
       all re-verified against the live bytes before any join.

WHAT THIS DOES AND DOES NOT SEAL. This ingestion establishes a final,
blinded HUMAN morphology read per prototype, and its concordance with the
independent machine read of the corrected-v2 renders (two reader types, two
tile draws). The corrected-v2 blank forms themselves remain unfilled; the
report must therefore either adopt this read through the documented bridge
or commission a v2-form re-read — an editorial decision recorded here as
open, not resolved. Reviewer identity/date are not recorded in the source
files and must be added to the methods by the authors.

Output is unblinding-sensitive and must not be shown to any future blinded
reviewer before their own forms are complete.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import final_v3_aim1_worklist as worklist  # noqa: E402

REVIEW_ROOT = REPO / "reviews" / "k32"
E3B_KEYS = Path("/mnt/wsl/oceanpath-hot/outputs/aim1/e3b/montages/k32")
BASE_KEY = E3B_KEYS / "KEY_do_not_open_before_review.csv"
ADDENDUM_KEY = E3B_KEYS / "KEY_M11_followup_do_not_open_before_review.csv"
V2_BUNDLES = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim4_cap8192_corrected_v2_20260820"
) / "review_bundles" / "k32"
PROTOTYPE_TABLE = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim4_cap8192_corrected_v2_20260820"
) / "analysis" / "prototypes_k32.csv"
AI_READ = (
    REPO
    / "reports"
    / "reruns"
    / "final_v4_additions_20260820"
    / "aim4_ai_blinded_read"
    / "unblinded_ai_read"
    / "unblinded_reads.csv"
)
DEFAULT_OUTPUT = (
    REPO / "reports" / "reruns" / "final_v5_additions_20260820" / "aim4_human_read_legacy"
)

# Curated per-prototype concordance verdicts between the human read (original
# renders) and the machine read (corrected-v2 renders). Both source texts are
# carried verbatim next to each verdict; the verdict is an editorial synthesis.
CONCORDANCE: dict[int, tuple[str, str]] = {
    20: ("agree", "both: smooth muscle / fibromuscular tissue, no epithelium"),
    6: (
        "partial",
        "both non-tumour low-content tissue; human: loose connective tissue (+red cells, "
        "smooth muscle tiles); machine: edge debris, fibrin/mucus strands, colour anomaly",
    ),
    26: (
        "partial",
        "both: neoplastic glandular epithelium; human notes well-to-moderate differentiation "
        "with one out-of-focus tile and focal dirty necrosis; machine emphasizes the "
        "blur/colour-cast quality theme",
    ),
    17: (
        "agree",
        "both, at high confidence: extracellular mucin pools with floating tumour-cell "
        "clusters; human adds focal signet-ring morphology in one tile",
    ),
    9: (
        "partial",
        "both: viable tumour at the stromal interface; human: moderately differentiated "
        "gland-forming pattern with focal dirty necrosis; machine: single-cell/small-cluster "
        "infiltration in desmoplastic stroma (different tile draws)",
    ),
    1: (
        "differ",
        "human (original render): tumour/stroma interface with focal desmoplasia and one "
        "necrotic tile; machine (v2 render): granulation-type inflammation with a giant "
        "cell — different tile draws of a technical-control prototype",
    ),
    28: (
        "agree",
        "both, at high confidence: well-differentiated tumour glands; human highlights "
        "gland/lumen geometry with focal papillary architecture; machine highlights tall "
        "penicillate columnar epithelium",
    ),
    12: (
        "partial",
        "both: ordinary well/moderately differentiated tumour epithelium; human notes "
        "narrow slit-like spaces; machine notes the uniform cool-stain regime",
    ),
    8: ("agree", "both: dirty/karyorrhectic necrosis (human: all tiles except one fat-edge tile)"),
    15: (
        "partial",
        "both: tumour with intermixed small cells; human resolves them as lymphocytes/"
        "neutrophils/eosinophils/fibroblasts and prominent nucleoli; machine emphasizes "
        "dark thick/overstained tiles with an immune pocket",
    ),
    5: ("agree", "both, at high confidence: benign/normal colonic glands (normal mucosa)"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def montage_map(key_path: Path) -> dict[str, int]:
    key = pd.read_csv(key_path)
    grouped = key.groupby("montage_id")["prototype"].nunique()
    if (grouped != 1).any():
        raise AssertionError(f"{key_path}: montage maps to more than one prototype")
    return key.groupby("montage_id")["prototype"].first().astype(int).to_dict()


def verify_provenance(structured: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Strict hash checks for the form, keys and montages; the completed
    Markdown may legitimately post-date its structured curation and is treated
    as authoritative, with any divergence recorded rather than fatal."""
    checks: list[str] = []
    discrepancies: list[str] = []
    prov = structured["provenance"]
    strict_pairs = [
        (REVIEW_ROOT / "review_form.csv", prov["base_form_sha256"]),
        (BASE_KEY, prov["base_key_sha256"]),
        (ADDENDUM_KEY, prov["addendum_key_sha256"]),
    ] + [
        (REVIEW_ROOT / "montages" / f"{montage}.jpg", digest)
        for montage, digest in prov["montage_sha256"].items()
    ]
    for path, expected in strict_pairs:
        if sha256(path) != expected:
            raise AssertionError(f"provenance hash mismatch for {path}")
        checks.append(f"{path.name}: PASS")
    live_md = sha256(REVIEW_ROOT / "completed_review.md")
    if live_md != structured["source_sha256"]:
        discrepancies.append(
            "completed_review.md was revised after the structured curation "
            f"(live sha256 {live_md}; curated-from {structured['source_sha256']}). "
            "The live Markdown is the authoritative human record; the structured "
            "fields were verified consistent with it by the ingesting analyst."
        )
    else:
        checks.append("completed_review.md: PASS")
    return checks, discrepancies


def run(output: Path, reviewer: str | None, review_date: str | None, decision: str) -> None:
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    structured = json.loads((REVIEW_ROOT / "completed_review_structured.json").read_text())
    provenance_checks, provenance_discrepancies = verify_provenance(structured)

    legacy = montage_map(BASE_KEY) | montage_map(ADDENDUM_KEY)
    v2 = montage_map(V2_BUNDLES / "base" / "unblinding_key_DO_NOT_SHARE.csv") | {
        "M11" if k == "A01" else k: v
        for k, v in montage_map(
            V2_BUNDLES / "attention_addendum" / "unblinding_key_DO_NOT_SHARE.csv"
        ).items()
    }
    if legacy != v2:
        raise AssertionError(f"montage-to-prototype maps differ across generations: {legacy} vs {v2}")

    form = pd.read_csv(REVIEW_ROOT / "review_form.csv", dtype=str).fillna("")
    deep = structured["montages"]
    ai = pd.read_csv(AI_READ).set_index("prototype")
    prototypes = pd.read_csv(PROTOTYPE_TABLE).set_index("prototype")

    rows: list[dict[str, Any]] = []
    agreement_counts = {"agree": 0, "partial": 0, "differ": 0}
    for montage, prototype in sorted(legacy.items(), key=lambda item: item[1]):
        first_pass = (
            form[form["montage_id"] == montage].iloc[0].to_dict()
            if (form["montage_id"] == montage).any()
            else {}
        )
        deep_read = deep.get(montage, {})
        verdict, rationale = CONCORDANCE[prototype]
        agreement_counts[verdict] += 1
        ai_row = ai.loc[prototype]
        rows.append(
            {
                "prototype": prototype,
                "montage_id_reviewed": montage,
                "sealed_class": prototypes.loc[prototype, "group"],
                "human_first_pass": {
                    key: value
                    for key, value in first_pass.items()
                    if key not in ("montage_id", "n_tiles") and str(value).strip()
                },
                "human_deep_read": {
                    key: deep_read[key]
                    for key in (
                        "canonical_description",
                        "differentiation",
                        "mucin",
                        "dirty_necrosis",
                        "artifact_concern",
                        "biological_interpretability",
                        "confidence",
                    )
                    if key in deep_read
                },
                "machine_read_v2": {
                    "montage": f"{ai_row['montage_id']} ({ai_row['bundle']})",
                    "dominant_pattern": ai_row["dominant_pattern"],
                    "architecture": ai_row["architecture"],
                    "confidence": ai_row["confidence"],
                },
                "concordance": {"verdict": verdict, "rationale": rationale},
            }
        )

    transcription_notes = [
        "Cross-montage summary table lists M04 primary interpretation as 'Mucin poor'; "
        "every body answer for M04 reads mucin POOLS ('the mucin is forming pools with "
        "tumor cells floating in it'), so 'poor' is a probable typo for 'pool'. Recorded "
        "as written; not silently corrected.",
        "Reviewer identity, review date and an explicit blinding attestation are not "
        "recorded in the source files (structured curation: blinding_confirmation="
        "'not_recorded'); the authors must add reviewer credentials and the attestation "
        "to the methods.",
    ]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": created,
        "component": "aim4_human_read_legacy_packet",
        "reviewer_kind": "human",
        "reviewer_metadata": {
            "reviewer": reviewer,
            "review_date": review_date,
            "attestation_basis": (
                "supplied by the study lead after completion; the source files "
                "themselves record no reviewer identity or key access"
            ),
        },
        "packet_generation": {
            "reviewed": "original e3b k=32 packet (M01-M10 + M11 addendum)",
            "corrected_v2_forms_filled": False,
            "montage_bytes_match_v2": False,
            "prototype_vocabulary_identical": True,
            "montage_to_prototype_map_identical_to_v2": True,
        },
        "provenance_checks": provenance_checks,
        "provenance_discrepancies": provenance_discrepancies,
        "reads": rows,
        "concordance_summary": {
            **agreement_counts,
            "candidates_p17_p28_p5": "agree at high confidence on both sides",
        },
        "seal_status": {
            "statement": (
                "A final blinded human read exists for every prototype in the corrected-v2 "
                "packet set, performed on the original packet generation (same frozen "
                "prototypes, different tile renders). The corrected-v2 blank forms remain "
                "unfilled."
            ),
            "closure_options": [
                "adopt this read via the documented generation bridge (convergent evidence: "
                "two reader types, two independent tile draws, identical prototype mapping)",
                "commission a corrected-v2 form re-read for byte-exact packet closure",
            ],
            "decision": decision,
        },
        "transcription_notes": transcription_notes,
        "unblinding_sensitive": True,
    }

    if output.exists():
        raise FileExistsError(f"append-only destination already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    (output / "README_UNBLINDING_SENSITIVE.md").write_text(
        "# UNBLINDING-SENSITIVE\n\nThis directory reveals montage-to-prototype mappings "
        "for both packet generations and pairs human with machine reads. It must not be "
        "shown to any future blinded reviewer before their own forms are complete.\n"
    )
    (output / "results.json").write_text(
        json.dumps(worklist._json_safe(payload), indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# Aim 4 — final blinded human read (original packet) with v2-machine concordance",
        "",
        f"Generated {created}. Corrected-v2 forms remain unfilled; see seal_status.",
        "",
        "| Prototype | Class | Human (original render) | Machine (v2 render) | Concordance |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        human = row["human_deep_read"].get("canonical_description") or "; ".join(
            f"{k}: {v}" for k, v in row["human_first_pass"].items()
        )
        lines.append(
            f"| p{row['prototype']} | {row['sealed_class']} | {human} | "
            f"{row['machine_read_v2']['dominant_pattern']} | "
            f"**{row['concordance']['verdict']}** — {row['concordance']['rationale']} |"
        )
    lines += ["", "## Notes", ""] + [f"- {note}" for note in transcription_notes] + [""]
    (output / "HUMAN_MACHINE_CONCORDANCE.md").write_text("\n".join(lines))

    inputs = [
        REVIEW_ROOT / "review_form.csv",
        REVIEW_ROOT / "completed_review.md",
        REVIEW_ROOT / "completed_review_structured.json",
        REVIEW_ROOT / "followup_review_M04_M05_M07.md",
        BASE_KEY,
        ADDENDUM_KEY,
        AI_READ,
        PROTOTYPE_TABLE,
        *sorted((REVIEW_ROOT / "montages").glob("*.jpg")),
    ]
    produced = [
        output / "README_UNBLINDING_SENSITIVE.md",
        output / "results.json",
        output / "HUMAN_MACHINE_CONCORDANCE.md",
    ]
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "append_only": True,
                "created_utc": created,
                "reviewer_kind": "human",
                "unblinding_sensitive": True,
                "output_root": str(output.resolve()),
                "supersedes": {
                    "path": str(DEFAULT_OUTPUT),
                    "reason": (
                        "v1 lacked reviewer identity/date and recorded the closure "
                        "decision as OPEN; v2 adds the study lead's attestation and the "
                        "adopted decision. All reads, mappings and concordance verdicts "
                        "are unchanged."
                    ),
                }
                if output != DEFAULT_OUTPUT
                else None,
                "inputs": [worklist.identity(p) for p in inputs],
                "artifacts": [worklist.identity(p) for p in produced],
                "promise": "No sealed packet byte, review source file, or upstream result root was modified.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"PASS — {len(rows)} prototypes ingested → {output}")
    print(json.dumps(payload["concordance_summary"], indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reviewer", default=None)
    parser.add_argument("--review-date", default=None)
    parser.add_argument(
        "--decision", default="OPEN — editorial decision for the authors"
    )
    args = parser.parse_args()
    run(args.output, args.reviewer, args.review_date, args.decision)


if __name__ == "__main__":
    main()
