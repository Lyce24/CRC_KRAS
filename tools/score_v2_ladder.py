#!/usr/bin/env python3
"""Score + report the v2 ladder: DEV OOF and the four external test sets.

For every (task, encoder) v2 run: pooled OOF patient AUROC (+CI) from the
run's outer folds, then the 5-fold mean-logit ensemble scored on rih_all
(patient x role units), rih_primary, rih_metastatic, sr1482_metastatic.
Writes /mnt/wsl/oceanpath-hot/outputs/eval/kras_v2/v2_report.{json,md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.kras import study  # noqa: E402

TRAIN_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/train/kras/v2")
EVAL_ROOT = Path("/mnt/wsl/oceanpath-hot/outputs/eval/kras_v2")

FEATURE_DIRS = {
    "univ1": study.PINNED_FEATURE_DIR,
    "conch_v15": study.FEATURE_ROOT / "20x_512px_0px_overlap_mpp0.5" / "features_conch_v15",
}
TASKS = ["b1_v2", "e4_v2", "e3_v2", "e2_v2", "e5_v2", "p1_v2"]
TITLES = {
    "b1_v2": "gene: mutant vs WT",
    "e4_v2": "exon: exon-2 vs non",
    "e3_v2": "codon: G12* vs non",
    "e2_v2": "functional: G12D/V",
    "e5_v2": "functional-wide: G12D/V/G13D",
    "p1_v2": "allele: G12D",
}
EXTERNAL_SETS = ["rih_all", "rih_primary", "rih_metastatic", "sr1482_metastatic"]


def fold_checkpoints(run_dir: Path) -> list[tuple[int, int, Path]]:
    out = []
    for i in range(5):
        metrics = json.loads((run_dir / f"fold_{i}" / "fold_metrics.json").read_text())
        out.append((42, i, Path(metrics["best_checkpoint"])))
    return out


def main() -> None:
    results: dict = {}
    for task in TASKS:
        manifest_dev = pd.read_csv(study.MANIFEST_ROOT / f"crc_kras_dev_{task}.csv")
        for enc in ("univ1", "conch_v15"):
            run_dir = TRAIN_ROOT / f"{task}_{enc}_seed42"
            if not study.run_is_complete(run_dir):
                print(f"== {task}/{enc}: run incomplete — skipped")
                continue
            oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
            pat = study.to_patient_logits(oof.drop(columns=["fold"]), manifest_dev)
            entry = {"dev_oof": study.bootstrap_patient_auroc(pat), "external": {}}

            for set_name in EXTERNAL_SETS:
                manifest_csv = study.MANIFEST_ROOT / f"crc_kras_{task}_{set_name}.csv"
                if not manifest_csv.is_file():
                    continue
                out_dir = EVAL_ROOT / task / enc / set_name
                slide_path = out_dir / "slide_predictions.parquet"
                manifest = pd.read_csv(manifest_csv)
                if slide_path.is_file():
                    preds = pd.read_parquet(slide_path)
                else:
                    scores = study.score_manifest_with_checkpoints(
                        fold_checkpoints(run_dir),
                        manifest_csv,
                        FEATURE_DIRS[enc],
                        num_classes=2,
                        num_workers=4,
                    )
                    preds = study.ensemble_slide_predictions(scores, manifest)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    scores.to_parquet(out_dir / "model_scores.parquet", index=False)
                    preds.to_parquet(slide_path, index=False)
                pat_ext = study.to_patient_logits(preds, manifest)
                entry["external"][set_name] = study.bootstrap_patient_auroc(pat_ext)
                print(f"{task}/{enc}/{set_name}: "
                      f"{entry['external'][set_name]['auroc']:.3f}")
            results.setdefault(task, {})[enc] = entry
            print(f"== {task}/{enc}: DEV OOF {entry['dev_oof']['auroc']:.3f}")

    lines = ["# V2 ladder — DEV OOF + external validation", "",
             "| Task | Encoder | DEV OOF (95% CI) | RIH-all | RIH-P | RIH-M | SR1482-M |",
             "|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for enc in ("univ1", "conch_v15"):
            if enc not in results.get(task, {}):
                continue
            e = results[task][enc]
            dev = e["dev_oof"]
            cells = []
            for set_name in EXTERNAL_SETS:
                x = e["external"].get(set_name)
                cells.append(f"{x['auroc']:.3f}" if x else "—")
            lines.append(
                f"| {TITLES[task]} | {enc} | {dev['auroc']:.3f} "
                f"({dev['ci_low']:.3f}–{dev['ci_high']:.3f}) | " + " | ".join(cells) + " |")
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    (EVAL_ROOT / "v2_report.md").write_text("\n".join(lines) + "\n")
    (EVAL_ROOT / "v2_report.json").write_text(json.dumps(results, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
