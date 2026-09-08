#!/usr/bin/env python3
"""Mechanism weld: p17/p28 abundance across MAPK pathway-context cells.

THE PREDICTION. Four sealed observations exist in different report sections:
(a) BRAF-mutant wild-types score KRAS-like (Why-D); (b) NRAS-mutant
wild-types do too (final-v4 composite); (c) p17 is the vocabulary-robust
KRAS-associated abundance prototype (Aim 4); (d) two independent blinded
reads describe p17 as extracellular mucin pools. Both KRAS-mutant and
BRAF-mutant (serrated-pathway) colorectal cancers are classically
mucin-associated, so if the model scores MAPK-activated tumours high because
it sees a mucin-associated phenotype, then p17 abundance itself must be
elevated in NRAS-mutant and BRAF-mutant KRAS-wild-type patients relative to
pathway-quiet wild-types. This tool tests exactly that, read-only.

DESIGN. The frozen corrected-v2 e0-arm patient profiles (1,486 patients x 32
prototypes) are joined with the frozen kras/nras/braf labels. Cells among
KRAS-wild-type patients: NRAS-mutant (64), BRAF-mutant/NRAS-WT (176),
pathway-quiet (618); KRAS-mutant (604) is the reference cell. Primary
endpoints are p17 and p28 ABUNDANCE per cell: mean with a 2,000-draw patient
bootstrap (seed 20260819, the Aim 4 convention), two-sided Mann-Whitney
versus the quiet cell (its U/n1n2 is the cell-versus-quiet AUROC), and BH
across the six primary cell contrasts. Attention mass for p17/p28/p5 is a
secondary, model-derived tier. Before any new number, the sealed canonical
p17/p28 A-population abundance AUROCs are reproduced from the same inputs
and asserted to 1e-6.

BOUNDARY. Association across measured labels only: no causal mucin-MAPK
claim, no whole-tumour subtype claim, and prototype identities remain those
of the sealed numeric atlas (the morphology names live with the pathology
reads, not here).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from tools import final_v3_aim1_worklist as worklist  # noqa: E402

AIM4_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim4_cap8192_corrected_v2_20260820"
)
PROFILES = AIM4_ROOT / "profiles" / "patient_profiles_k32.parquet"
PROTOTYPE_TABLE = AIM4_ROOT / "analysis" / "prototypes_k32.csv"
DEV_MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/aim1_dev.csv")
DEFAULT_OUTPUT = (
    REPO / "reports" / "reruns" / "final_v5_additions_20260820" / "aim4_context_weld"
)
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20260819
PRIMARY_PROTOTYPES = (17, 28)
ATTENTION_PROTOTYPES = (17, 28, 5)
CELLS = ("kras_mutant", "nras_mutant", "braf_mutant_nras_wt", "pathway_quiet")


def build_frame() -> pd.DataFrame:
    profiles = pd.read_parquet(PROFILES)
    e0 = profiles[profiles["arm"] == "e0"]
    if e0["patient_id"].nunique() != 1486:
        raise AssertionError(f"e0 arm has {e0['patient_id'].nunique()} patients, expected 1486")
    manifest = pd.read_csv(DEV_MANIFEST, low_memory=False).drop_duplicates("patient_id")
    labels = manifest[["patient_id", "kras", "nras", "braf"]].copy()
    for column in ("kras", "nras", "braf"):
        labels[column] = labels[column].fillna("unknown").astype(str)
    kras_w = labels["kras"] == "wild_type"
    labels["cell"] = np.where(
        ~kras_w,
        "kras_mutant",
        np.select(
            [
                labels["nras"] == "mutant",
                (labels["braf"] == "mutant") & (labels["nras"] == "wild_type"),
                (labels["nras"] == "wild_type") & (labels["braf"] == "wild_type"),
            ],
            ["nras_mutant", "braf_mutant_nras_wt", "pathway_quiet"],
            default="incomplete_labels",
        ),
    )
    wide = e0.pivot_table(
        index="patient_id",
        columns="prototype",
        values=["abundance", "attn_mass_mean"],
        aggfunc="first",
    )
    wide.columns = [f"{quantity}_p{prototype}" for quantity, prototype in wide.columns]
    frame = labels.merge(wide.reset_index(), on="patient_id", validate="one_to_one")
    frame["kras_label"] = (frame["kras"] == "mutant").astype(int)
    return frame


def anchor_check(frame: pd.DataFrame) -> dict[str, float]:
    """Reproduce the sealed A-population abundance AUROCs for p17/p28."""
    sealed = pd.read_csv(PROTOTYPE_TABLE).set_index("prototype")["abundance_auc_A"]
    out = {}
    for prototype in PRIMARY_PROTOTYPES:
        observed = float(
            roc_auc_score(frame["kras_label"], frame[f"abundance_p{prototype}"])
        )
        expected = float(sealed.loc[prototype])
        if abs(observed - expected) > 1e-6:
            raise AssertionError(
                f"p{prototype} anchor failed: {observed:.6f} != {expected:.6f}"
            )
        out[f"p{prototype}"] = observed
    return out


def cell_block(
    frame: pd.DataFrame, column: str, rng_seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    quiet = frame.loc[frame["cell"] == "pathway_quiet", column].to_numpy()
    rng = np.random.default_rng(rng_seed)
    cells: dict[str, Any] = {}
    contrasts: list[dict[str, Any]] = []
    for cell in CELLS:
        values = frame.loc[frame["cell"] == cell, column].to_numpy()
        boots = [
            float(np.mean(values[rng.integers(0, len(values), len(values))]))
            for _ in range(BOOTSTRAP_DRAWS)
        ]
        entry: dict[str, Any] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "mean_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "median": float(np.median(values)),
        }
        if cell != "pathway_quiet":
            stat = mannwhitneyu(values, quiet, alternative="two-sided")
            contrast = {
                "quantity": column,
                "cell": cell,
                "mean_difference_vs_quiet": float(np.mean(values) - np.mean(quiet)),
                "auroc_cell_vs_quiet": float(stat.statistic / (len(values) * len(quiet))),
                "p_two_sided": float(stat.pvalue),
            }
            entry["vs_pathway_quiet"] = contrast
            contrasts.append(contrast)
        cells[cell] = entry
    return cells, contrasts


def bh_adjust(contrasts: list[dict[str, Any]]) -> None:
    """BH within the primary family, in place."""
    order = np.argsort([c["p_two_sided"] for c in contrasts])
    m = len(contrasts)
    adjusted = np.empty(m)
    previous = 1.0
    for rank_position in range(m - 1, -1, -1):
        index = order[rank_position]
        value = contrasts[index]["p_two_sided"] * m / (rank_position + 1)
        previous = min(previous, value)
        adjusted[index] = previous
    for contrast, q in zip(contrasts, adjusted, strict=True):
        contrast["q_bh"] = float(min(q, 1.0))


def run() -> dict[str, Any]:
    frame = build_frame()
    census = frame["cell"].value_counts().to_dict()
    if census.get("nras_mutant") != 64 or census.get("braf_mutant_nras_wt") != 176:
        raise AssertionError(f"cell census drift: {census}")
    anchors = anchor_check(frame)

    primary_contrasts: list[dict[str, Any]] = []
    abundance: dict[str, Any] = {}
    for prototype in PRIMARY_PROTOTYPES:
        cells, contrasts = cell_block(
            frame, f"abundance_p{prototype}", BOOTSTRAP_SEED + prototype
        )
        abundance[f"p{prototype}"] = cells
        primary_contrasts.extend(contrasts)
    bh_adjust(primary_contrasts)

    attention: dict[str, Any] = {}
    secondary_contrasts: list[dict[str, Any]] = []
    for prototype in ATTENTION_PROTOTYPES:
        cells, contrasts = cell_block(
            frame, f"attn_mass_mean_p{prototype}", BOOTSTRAP_SEED + 100 + prototype
        )
        attention[f"p{prototype}"] = cells
        secondary_contrasts.extend(contrasts)
    bh_adjust(secondary_contrasts)

    prediction = {
        "statement": (
            "if the model reads a mucin-associated MAPK phenotype, p17 abundance is "
            "elevated in NRAS-mutant and BRAF-mutant KRAS-wild-type cells versus "
            "pathway-quiet wild-types"
        ),
        "p17_nras_supported": bool(
            next(
                c
                for c in primary_contrasts
                if c["quantity"] == "abundance_p17" and c["cell"] == "nras_mutant"
            )["mean_difference_vs_quiet"]
            > 0
        ),
        "p17_braf_supported": bool(
            next(
                c
                for c in primary_contrasts
                if c["quantity"] == "abundance_p17" and c["cell"] == "braf_mutant_nras_wt"
            )["mean_difference_vs_quiet"]
            > 0
        ),
    }

    return {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "component": "aim4_context_weld",
        "census": census,
        "sealed_anchor_auroc_A": anchors,
        "abundance_by_cell": abundance,
        "attention_by_cell_secondary": attention,
        "primary_contrast_family_bh": primary_contrasts,
        "secondary_contrast_family_bh": secondary_contrasts,
        "prediction_readout": prediction,
        "conventions": {
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed_base": BOOTSTRAP_SEED,
            "primary_family": "p17/p28 abundance x {nras_mutant, braf_mutant_nras_wt, kras_mutant} vs quiet (6 tests)",
            "secondary_family": "p17/p28/p5 attention mass, same contrasts (9 tests)",
        },
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Final-v5 addition — p17/p28 abundance across MAPK pathway-context cells",
        "",
        f"Generated {payload['created_utc']}. Read-only; sealed anchors reproduced.",
        "",
        "| Quantity | Cell | n | Mean [95% CI] | Delta vs quiet | AUROC vs quiet | q (BH) |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for family, source in (
        ("abundance", payload["abundance_by_cell"]),
        ("attention", payload["attention_by_cell_secondary"]),
    ):
        for prototype, cells in source.items():
            for cell, entry in cells.items():
                contrast = entry.get("vs_pathway_quiet")
                lines.append(
                    f"| {family} {prototype} | {cell} | {entry['n']} | "
                    f"{entry['mean']:.5f} [{entry['mean_ci'][0]:.5f}, {entry['mean_ci'][1]:.5f}] | "
                    + (
                        f"{contrast['mean_difference_vs_quiet']:+.5f} | "
                        f"{contrast['auroc_cell_vs_quiet']:.3f} | {contrast.get('q_bh', float('nan')):.2e} |"
                        if contrast
                        else "— | — | — |"
                    )
                )
    lines += [
        "",
        "Prediction readout: "
        + json.dumps(payload["prediction_readout"], indent=None),
        "",
        "Boundary: measured-label association only; no causal or subtype claim;",
        "prototype names remain with the pathology reads.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run()
    output: Path = args.output
    if output.exists():
        raise FileExistsError(f"append-only destination already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    inputs = [PROFILES, PROTOTYPE_TABLE, DEV_MANIFEST]
    (output / "input_receipt.json").write_text(
        json.dumps(
            {
                "created_utc": payload["created_utc"],
                "inputs": [worklist.identity(p) for p in inputs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (output / "results.json").write_text(
        json.dumps(worklist._json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    (output / "DESIGN_AND_RESULTS.md").write_text(render(payload))
    produced = [output / "input_receipt.json", output / "results.json", output / "DESIGN_AND_RESULTS.md"]
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "append_only": True,
                "created_utc": payload["created_utc"],
                "output_root": str(output.resolve()),
                "artifacts": [worklist.identity(p) for p in produced],
                "promise": "No upstream result root or prior report bundle was modified.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("PASS —", output)
    for contrast in payload["primary_contrast_family_bh"]:
        print(
            f"  {contrast['quantity']} {contrast['cell']}: delta "
            f"{contrast['mean_difference_vs_quiet']:+.5f} AUROC {contrast['auroc_cell_vs_quiet']:.3f} "
            f"q={contrast['q_bh']:.2e}"
        )


if __name__ == "__main__":
    main()
