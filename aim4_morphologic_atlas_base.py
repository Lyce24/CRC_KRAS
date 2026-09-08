#!/usr/bin/env python3
"""E4 - AIM 4, KRAS morphologic phenotype and transport atlas (0 new fits).

NAMING. The aims split 2026-08-19: this experiment is AIM 4 / E4, its own aim
rather than an explanation panel under the resolution ladder (now AIM 3 / E3,
`aim3_resolution_ladder.py`). The file name, `outputs/aim1/e3b/` and the
`e3b_*_k32.json` eval artefacts still say "e3b" and are NOT being migrated. See
`Experimental_Setup.md` header for the naming map.

QUESTION. What morphology supports the gene-level KRAS signal, does it survive
measured-context restriction, what is its primary-to-metastatic transport state,
and is any association source-confounded?

ZERO NEW FITS, AND THAT IS THE DESIGN. Every model E3b reads is already frozen:
E0's fold models for the development arm, E2a's full-source refits for the
transport arm. No MIL model, no MSI head, no BRAF head, no adaptation model and
no "explanation model" is trained. The only new objects are an unsupervised
prototype vocabulary and the statistics computed over it.

THIS SUPERSEDES THE DESIGN ARCHIVE'S E4. That prototype atlas was blocked on MSI
and BRAF heads from E1e. E3b drops that dependency: molecular specificity is
established by RESTRICTION TO SET D (MSS/pMMR and BRAF-wild-type) - the same
instrument Aim 1 already validated - rather than by training two more
classifiers whose own dependencies would then need auditing. Set D is where
Aim 1 and Aim 4 join.

FIVE STEPS.

  1 vocabulary   Sample tiles evenly by patient and arm, cluster frozen UNIv1
                 embeddings into ~24-40 prototypes, FREEZE the centroids. The
                 clustering sees no KRAS, MSI, BRAF, cohort or specimen-role
                 label. One vocabulary serves every arm; refitting per arm would
                 make "prototype 7" a different object in each panel.
  2 assign       Per patient x prototype, two quantities that are never
                 collapsed: ABUNDANCE (is this morphology physically more common
                 in mutant tumours?) and ATTENTION MASS (is the frozen KRAS
                 classifier relying on it?).
  3 specificity  KRAS-mutant vs wild-type in population A (all 1,486) and again
                 in population D (1,129), plus prototype behaviour among
                 KRAS-WT patients across MSI x BRAF context. A prototype that
                 loses its association in D is restriction-sensitive; the
                 context-in-WT test distinguishes measured context morphology
                 from unexplained attenuation.
  4 transport    The same frozen vocabulary in RIH-P, RIH-M, SR1482-P and
                 SR1482-M, scored by the frozen LOCO refits, with E2d's organ
                 and acquisition metadata used to ask whether an "important"
                 prototype is really liver parenchyma or a scanner batch.
  5 montages     Blinded review packets written to `reviews/k<K>/`: medoid and
                 high-attention tiles spanning cohorts and specimen roles, the
                 montage set chosen BY THE READOUT - every prototype the final
                 table makes a claim about, plus every shortcut suspect - so no
                 claim reaches the manuscript undescribed. The unblinding key is
                 written OUTSIDE the packet.

THE FINAL READOUT combines independent axes rather than forcing every outcome
into a binary transport verdict: molecular specificity; conserved, directly
changed, inconclusive, underpowered or heterogeneous transport; and source /
technical flags. This combination lives in `report`, which can see both the
specificity and transport analyses.

The shortcut check is evaluated FIRST and dominates, on the same logic as E3's
underpowered check: a prototype whose tissue comes overwhelmingly from one
cohort, one scanner setting or one metastatic organ cannot be reported as
morphology however strong its association, because association and source are
not separable here. Failure to reach significance in a metastatic arm is not
evidence of change: a change requires a direct primary-versus-metastatic AUC
contrast after multiplicity correction. At n = 74-85 per metastatic arm,
"changed", "inconclusive" and "there was never a primary effect to test" are
different claims.

TWO STRUCTURAL RULES INHERITED FROM E2b, NOT NEGOTIABLE HERE:

  RIH paired patients   the 8 dual-role patients are RETAINED in the standalone
                        RIH-P and RIH-M panels and EXCLUDED from the
                        primary-vs-metastatic contrast, where they would make
                        the arms non-independent.
  SurGen subcohort      SR1482 is the only SurGen subcohort with metastases, so
                        the contrast is SR1482-P vs SR1482-M, never pooled
                        SR386+SR1482 on the primary side.

ATTENTION ENTITLEMENT. A slide's attention must come from the model that
actually scored it: E0 fold models out-of-fold for the development arm, the
frozen LOCO refit for each transport arm. Mixing the two would explain a
prediction that was never made.

CLAIM LIMITS. Attention is a region-prioritisation signal, not a causal
explanation. Abundance and attention mass are separate claims. A prototype that
survives set D is dependency-robust with respect to the MEASURED dependencies
only.

Usage:
    python aim4_morphologic_atlas_base.py plan
    python aim4_morphologic_atlas_base.py attention [--arm e0] [--seed 42]
    python aim4_morphologic_atlas_base.py vocabulary [--k 32] [--tiles 400000]
    python aim4_morphologic_atlas_base.py assign [--arm e0]
    python aim4_morphologic_atlas_base.py specificity
    python aim4_morphologic_atlas_base.py transport
    python aim4_morphologic_atlas_base.py montages [--top 8]
    python aim4_morphologic_atlas_base.py montage-addendum --prototype 5 --montage-id M11
    python aim4_morphologic_atlas_base.py report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_loco_transport  # noqa: E402
from oceanpath.aim1 import atlas, attention, paths  # noqa: E402

SEEDS: tuple[int, ...] = (42, 43, 44)
E3B_CAP = 8192                       # the study cap; E3b reads models trained at it
E0_RUN = paths.OUTPUT_ROOT / f"train/1a_pb_cap{E3B_CAP}/univ1"
E3B_ROOT = paths.OUTPUT_ROOT / "e3b"
ATTN_DIR = E3B_ROOT / "attention"
VOCAB_DIR = E3B_ROOT / "vocabulary"
PROFILE_DIR = E3B_ROOT / "profiles"
MONTAGE_DIR = E3B_ROOT / "montages"
# The pathologist packet is a deliverable, not an intermediate, so it is written
# into the repository rather than the scratch output root. The unblinding KEY is
# deliberately NOT written here - it stays under MONTAGE_DIR, so handing someone
# `reviews/` cannot accidentally hand them the answers.
REVIEW_ROOT = REPO / "reviews"

FDR_ALPHA = 0.05
# A prototype drawing this much of its tissue from one cohort, one metastatic
# organ or one acquisition batch is flagged: its KRAS association, however
# strong, cannot be separated from that source.
CONCENTRATION_FLAG = 0.60
# The same idea on an axis whose level count varies. A flat share of 0.60 means
# "2.4x an even split" only when there are four levels, as there are cohorts; on
# TCGA's TWO mpp bins an even split is already 0.50, so a 0.60 rule would flag
# essentially every prototype and call the study's own candidates artefacts.
# Acquisition axes are therefore judged on ENRICHMENT over an even split, set to
# the value the cohort rule implies: 0.60 x 4 levels.
ACQUISITION_ENRICHMENT = CONCENTRATION_FLAG * 4
# Only `mpp_bin` is allowed to SUPPRESS a claim. `technical_class` is recorded
# beside it and never flagged, for two reasons that are properties of the
# variable rather than of the result: it is a composite of cohort x mpp x
# section that re-encodes the cohort axis already screened above, and at 4-6
# levels per cohort it flags 30 of 32 prototypes - a screen that fires on
# almost everything cannot discriminate, and would suppress the study's own
# candidates without evidence that anything technical distinguishes them.
ACQUISITION_FLAG_COLUMNS = ("mpp_bin",)


@dataclass(frozen=True)
class Arm:
    """One evaluated population and the model entitled to score it."""

    name: str
    role: str                    # primary | metastatic
    mode: str                    # oof | refit
    target: str | None           # E2a held-out cohort, for refit arms
    subcohort: str | None = None  # restrict the manifest to one subcohort
    note: str = ""


ARMS: dict[str, Arm] = {
    "e0": Arm(
        "e0", "primary", "oof", None,
        note="all 1,486 primary KRAS-labelled patients, E0 out-of-fold",
    ),
    "rih_primary": Arm(
        "rih_primary", "primary", "refit", "RIH",
        note="RIH-P scored by the RIH-held-out refit",
    ),
    "rih_metastatic": Arm(
        "rih_metastatic", "metastatic", "refit", "RIH",
        note="RIH-M scored by the SAME RIH-held-out refit",
    ),
    "sr1482_primary": Arm(
        "sr1482_primary", "primary", "refit", "SurGen", subcohort="SR1482",
        note="SR1482-P only - never pooled with SR386",
    ),
    "sr1482_metastatic": Arm(
        "sr1482_metastatic", "metastatic", "refit", "SurGen",
        note="SR1482-M, the only SurGen subcohort with metastases",
    ),
}
TRANSPORT_PAIRS = [("RIH", "rih_primary", "rih_metastatic"),
                   ("SR1482", "sr1482_primary", "sr1482_metastatic")]


# ── populations ──────────────────────────────────────────────────────────────
def arm_manifest(arm: Arm) -> pd.DataFrame:
    """The slide manifest for one arm, with the subcohort restriction applied."""
    if arm.mode == "oof":
        frame = pd.read_csv(paths.DEV_MANIFEST)
    else:
        assert arm.target is not None
        frame = pd.read_csv(aim2_loco_transport.target_manifest(arm.target, arm.role))
    if arm.subcohort:
        frame = frame[frame["subcohort"].eq(arm.subcohort)].copy()
    roles = set(frame["specimen_role"].dropna().unique())
    if roles != {arm.role}:
        raise SystemExit(f"{arm.name}: manifest holds roles {sorted(roles)}, expected {arm.role}")
    return frame.reset_index(drop=True)


def dev_patients() -> pd.DataFrame:
    """Patient-level development metadata, including set-D membership."""
    frame = pd.read_csv(paths.DEV_MANIFEST).drop_duplicates("patient_id").copy()
    frame["is_mutant"] = frame["kras"].eq("mutant").astype(int)
    frame["in_D"] = (frame["msi_dmmr"].eq("MSS/pMMR") & frame["braf"].eq("wild_type")).astype(int)
    frame["label_complete"] = (
        frame["msi_dmmr"].isin(["MSS/pMMR", "MSI/dMMR"]) & frame["braf"].isin(
            ["wild_type", "mutant"])
    ).astype(int)
    # "Context-positive" = the molecular neighbourhood D removes.
    frame["context_positive"] = (
        frame["msi_dmmr"].eq("MSI/dMMR") | frame["braf"].eq("mutant")
    ).astype(int)
    frame["msi_braf_cell"] = (
        frame["msi_dmmr"].astype(str) + " x BRAF-" + frame["braf"].astype(str)
    )
    return frame


def rih_dual_role_patients() -> set[str]:
    """The 8 RIH patients contributing both a primary and a metastasis."""
    p = set(arm_manifest(ARMS["rih_primary"])["patient_id"])
    m = set(arm_manifest(ARMS["rih_metastatic"])["patient_id"])
    return p & m


# ── checkpoints ──────────────────────────────────────────────────────────────
def e0_fold_checkpoints(seed: int) -> list[tuple[int, int, Path]]:
    """(seed, fold, checkpoint) for E0's five fold models.

    Read from each fold's own fold_metrics.json rather than globbed off disk:
    a fold directory can hold several `best-epoch=*` files from earlier runs,
    and the run's own record is the only thing that says which one was deployed.
    """
    out: list[tuple[int, int, Path]] = []
    for fold in range(paths.N_FOLDS):
        metrics = E0_RUN / f"seed{seed}" / f"fold_{fold}" / "fold_metrics.json"
        if not metrics.is_file():
            raise SystemExit(f"missing {metrics} - E0 seed{seed} fold{fold} not trained")
        ckpt = Path(json.loads(metrics.read_text())["best_checkpoint"])
        if not ckpt.is_file():
            raise SystemExit(f"E0 seed{seed} fold{fold}: checkpoint {ckpt} is gone")
        out.append((seed, fold, ckpt))
    return out


def e0_fold_of_slide(seed: int) -> pd.DataFrame:
    """slide_id -> outer fold, taken from the OOF predictions of that same run.

    Using the run's own OOF table rather than the splits parquet guarantees the
    fold map matches the checkpoints being loaded; a splits file could have been
    regenerated since.
    """
    p = E0_RUN / f"seed{seed}" / "oof_predictions.parquet"
    if not p.is_file():
        raise SystemExit(f"missing {p}")
    return pd.read_parquet(p)[["slide_id", "fold"]]


def refit_checkpoint(target: str, seed: int) -> Path:
    return aim2_loco_transport.model_ckpt(target, seed, E3B_CAP)


# ── attention ────────────────────────────────────────────────────────────────
def attention_path(arm: str, seed: int) -> Path:
    return ATTN_DIR / f"{arm}_seed{seed}.h5"


def _attention_todo(destination: Path, slide_ids: list[str]) -> list[str]:
    """Slides not yet in ``destination``.

    Export is resumable at slide granularity: the 4,926 E0 forward passes take
    long enough that an interrupted run must not start over, and a completed
    file must not be silently re-created empty.
    """
    if not destination.is_file():
        return list(slide_ids)
    with h5py.File(destination, "r") as handle:
        have = set(handle.keys())
    return [s for s in slide_ids if s not in have]


def cmd_attention(args: argparse.Namespace) -> None:
    """Export per-tile attention from the model entitled to score each slide."""
    names = [args.arm] if args.arm else list(ARMS)
    seeds = [args.seed] if args.seed else list(SEEDS)
    for name in names:
        arm = ARMS[name]
        manifest = arm_manifest(arm)
        if args.limit:
            manifest = manifest.head(args.limit)
        for seed in seeds:
            dest = attention_path(name, seed)
            todo = _attention_todo(dest, manifest["slide_id"].tolist())
            if not todo and not args.force:
                print(f"  {name} seed{seed}: all {len(manifest)} slides present - skipping")
                continue
            print(f"== {name} seed{seed}: {len(todo)} of {len(manifest)} slides -> {dest}")
            if args.dry_run:
                continue
            if arm.mode == "oof":
                summary = attention.export_attention(
                    checkpoints=e0_fold_checkpoints(seed),
                    manifest=manifest,
                    splits=e0_fold_of_slide(seed),
                    destination=dest,
                    top_fraction=args.top_fraction,
                )
            else:
                assert arm.target is not None
                summary = attention.export_attention_refit(
                    checkpoint=refit_checkpoint(arm.target, seed),
                    slide_ids=manifest["slide_id"].tolist(),
                    destination=dest,
                    top_fraction=args.top_fraction,
                    seed=seed,
                )
            print(f"   wrote {summary['n_slides']} slides ({summary['mode']})")


# ── vocabulary ───────────────────────────────────────────────────────────────
def vocab_path(k: int) -> Path:
    return VOCAB_DIR / f"vocab_k{k}.npz"


def corpus_slides() -> pd.DataFrame:
    """Every slide any arm evaluates, once, with its tile count and atlas group.

    The vocabulary is fitted over the UNION of arms, metastases included: a
    vocabulary built on primaries alone would have no prototype for normal liver
    parenchyma, and E3b would then be unable to flag the single most obvious
    metastatic shortcut.
    """
    rows: list[pd.DataFrame] = []
    for arm in ARMS.values():
        frame = arm_manifest(arm)[["slide_id", "patient_id", "cohort", "subcohort"]].copy()
        frame["role"] = arm.role
        rows.append(frame)
    slides = pd.concat(rows, ignore_index=True).drop_duplicates("slide_id").reset_index(drop=True)
    slides["atlas_group"] = slides["subcohort"].astype(str) + "|" + slides["role"]
    counts = []
    for slide_id in slides["slide_id"]:
        with h5py.File(paths.PINNED_FEATURE_DIR / f"{slide_id}.h5", "r") as handle:
            counts.append(int(handle["features"].shape[0]))
    slides["n_tiles"] = counts
    return slides


def cmd_vocabulary(args: argparse.Namespace) -> None:
    dest = vocab_path(args.k)
    if dest.is_file() and not args.force:
        raise SystemExit(f"{dest} exists - pass --force to refit (this invalidates every profile)")
    print("Building the label-blind prototype vocabulary.")
    slides = corpus_slides()
    print(f"  corpus: {len(slides)} slides, {slides['patient_id'].nunique()} patients, "
          f"{slides['n_tiles'].sum() / 1e6:.1f}M tiles across "
          f"{slides['atlas_group'].nunique()} groups")
    plan = atlas.sample_plan(slides, total_tiles=args.tiles, group_column="atlas_group")
    by_group = plan.groupby("atlas_group")["n_sample"].sum()
    for group, n in by_group.items():
        print(f"    {group:24s} {int(n):7d} tiles from "
              f"{int(plan[plan.atlas_group.eq(group)].patient_id.nunique()):4d} patients")
    print(f"  sampling {int(plan['n_sample'].sum())} tiles ...")
    if args.dry_run:
        return
    sample = atlas.collect_sample(plan, paths.PINNED_FEATURE_DIR, seed=args.seed)
    print(f"  fitting PCA({args.components}) + KMeans({args.k}) on {sample.shape} ...")
    vocab = atlas.fit_vocabulary(
        sample, n_prototypes=args.k, n_components=args.components, seed=args.seed,
        normalize=args.normalize,
        config={
            "corpus_slides": int(len(slides)),
            "corpus_patients": int(slides["patient_id"].nunique()),
            "groups": sorted(slides["atlas_group"].unique().tolist()),
            "tiles_requested": int(args.tiles),
            "label_blind": True,
        },
    )
    vocab.save(dest)
    plan.to_parquet(VOCAB_DIR / f"sample_plan_k{args.k}.parquet", index=False)
    print(f"  explained variance {vocab.config['explained_variance_ratio']:.3f}; "
          f"cluster sizes {vocab.config['cluster_sizes']}")
    print(f"Wrote {dest}")


# ── assignment ───────────────────────────────────────────────────────────────
def profile_path(k: int, level: str) -> Path:
    return PROFILE_DIR / f"{level}_profiles_k{k}.parquet"


def candidates_path(k: int) -> Path:
    return PROFILE_DIR / f"tile_candidates_k{k}.parquet"


def cmd_assign(args: argparse.Namespace) -> None:
    """Assign every evaluated tile to a prototype; emit abundance + attention mass."""
    vocab = atlas.Vocabulary.load(vocab_path(args.k))
    names = [args.arm] if args.arm else list(ARMS)
    seeds = (
        [int(x) for x in args.seeds.split(",")] if args.seeds else list(SEEDS)
    )
    unknown = set(seeds) - set(SEEDS)
    if unknown:
        raise SystemExit(f"unknown seeds {sorted(unknown)}; the study seeds are {list(SEEDS)}")
    slide_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for name in names:
        arm = ARMS[name]
        manifest = arm_manifest(arm)
        if args.limit:
            manifest = manifest.head(args.limit)
        handles: dict[int, h5py.File] = {}
        for seed in seeds:
            p = attention_path(name, seed)
            if not p.is_file():
                continue
            try:
                handles[seed] = h5py.File(p, "r")
            except OSError as exc:
                # HDF5 takes a whole-file lock, so an export still writing this
                # seed cannot be read. Silently dropping the seed would change
                # the ensemble without saying so.
                for handle in handles.values():
                    handle.close()
                raise SystemExit(
                    f"{p} is locked ({exc}). An `attention` export is probably still "
                    f"writing it - wait for it to finish, or restrict this run with "
                    f"--seeds {','.join(str(x) for x in seeds if x != seed)}."
                ) from exc
        if not handles:
            raise SystemExit(f"{name}: no attention exported - run `attention --arm {name}` first")
        # INTERSECTION, not union: a slide present in seed 42 but missing from
        # seed 43 would get a two-seed mean while its neighbours got three, and
        # nothing downstream would say so.
        covered = set.intersection(*(set(h.keys()) for h in handles.values()))
        uncovered = sorted(set(manifest["slide_id"]) - covered)
        if uncovered and not args.allow_missing_attention:
            for handle in handles.values():
                handle.close()
            raise SystemExit(
                f"{name}: {len(uncovered)} slide(s) lack attention for at least one of "
                f"seeds {sorted(handles)} (e.g. {uncovered[:3]}), so their attention mass "
                f"would be an inconsistent or NaN average. Run `attention --arm {name}` to "
                "completion, or pass --allow-missing-attention to accept it."
            )
        print(f"== {name}: {len(manifest)} slides, attention seeds {sorted(handles)}"
              + (f", {len(uncovered)} WITHOUT attention" if uncovered else ""))
        try:
            for index, row in enumerate(manifest.itertuples(index=False), start=1):
                with h5py.File(paths.PINNED_FEATURE_DIR / f"{row.slide_id}.h5", "r") as fh:
                    features = fh["features"][:].astype(np.float32)
                    coords = fh["coords"][:]
                labels = vocab.assign(features)
                weights: dict[int, np.ndarray] = {}
                for seed, handle in handles.items():
                    if row.slide_id not in handle:
                        continue
                    weights[seed] = handle[row.slide_id]["attention"][:]
                profile = atlas.slide_profile(labels, weights, vocab.n_prototypes)
                mass_cols = [c for c in profile if c.startswith("attn_mass_seed")]
                mass_mean = (
                    np.mean([profile[c] for c in mass_cols], axis=0)
                    if mass_cols
                    else np.full(vocab.n_prototypes, np.nan)
                )
                for k in range(vocab.n_prototypes):
                    record = {
                        "arm": name, "role": arm.role, "slide_id": row.slide_id,
                        "patient_id": row.patient_id,
                        "cohort": getattr(row, "cohort", None),
                        "subcohort": getattr(row, "subcohort", None),
                        "label": int(row.target_label),
                        "prototype": k,
                        "n_tiles_slide": int(len(labels)),
                        "n_tiles_prototype": int(profile["n_tiles"][k]),
                        "abundance": float(profile["abundance"][k]),
                        "attn_mass_mean": float(mass_mean[k]),
                        "n_attention_seeds": len(mass_cols),
                    }
                    for c in mass_cols:
                        record[c] = float(profile[c][k])
                    for extra in ("liver_class", "met_site_class", "technical_class", "mpp_bin"):
                        if hasattr(row, extra):
                            record[extra] = getattr(row, extra)
                    slide_rows.append(record)
                candidate_rows.extend(
                    _tile_candidates(name, arm, row, vocab, features, coords, labels, weights)
                )
                if index % 100 == 0:
                    print(f"    {index}/{len(manifest)} slides", flush=True)
        finally:
            for handle in handles.values():
                handle.close()

    slides = pd.DataFrame(slide_rows)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    _merge_write(slides, profile_path(args.k, "slide"), keys=["arm", "slide_id", "prototype"])
    candidates = pd.DataFrame(candidate_rows)
    _merge_write(candidates, candidates_path(args.k), keys=["arm", "slide_id", "prototype", "kind"])

    full = pd.read_parquet(profile_path(args.k, "slide"))
    patients = _to_patient_level(full)
    patients.to_parquet(profile_path(args.k, "patient"), index=False)
    print(f"\nWrote {profile_path(args.k, 'slide')} ({len(full)} rows)")
    print(f"Wrote {profile_path(args.k, 'patient')} ({len(patients)} rows)")
    print(f"Wrote {candidates_path(args.k)}")


def _tile_candidates(
    name: str,
    arm: Arm,
    row: Any,
    vocab: atlas.Vocabulary,
    features: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
    weights: dict[int, np.ndarray],
) -> list[dict[str, Any]]:
    """One medoid and one top-attention tile per prototype per slide.

    Storing every tile would be 20M rows; storing the best candidate per
    (slide, prototype) keeps the montage step's choice diverse ACROSS slides by
    construction, which is what the blinded review needs - a montage drawn from
    one slide's neighbouring tiles would show one field of view six times.
    """
    z = vocab.project(features)
    mean_attention = (
        np.mean(list(weights.values()), axis=0) if weights else np.zeros(len(labels))
    )
    out: list[dict[str, Any]] = []
    for k in np.unique(labels):
        idx = np.flatnonzero(labels == k)
        if idx.size == 0:
            continue
        distance = np.linalg.norm(z[idx] - vocab.centroids[k], axis=1)
        picks = {"medoid": idx[int(np.argmin(distance))],
                 "top_attention": idx[int(np.argmax(mean_attention[idx]))]}
        for kind, tile in picks.items():
            out.append({
                "arm": name, "role": arm.role, "slide_id": row.slide_id,
                "patient_id": row.patient_id, "cohort": getattr(row, "cohort", None),
                "subcohort": getattr(row, "subcohort", None),
                "label": int(row.target_label), "prototype": int(k), "kind": kind,
                "tile_index": int(tile),
                "x": int(coords[tile][0]), "y": int(coords[tile][1]),
                "distance_to_centroid": float(np.linalg.norm(z[tile] - vocab.centroids[k])),
                "attention": float(mean_attention[tile]),
            })
    return out


def _merge_write(new: pd.DataFrame, path: Path, keys: list[str]) -> None:
    """Upsert into an existing parquet so per-arm runs accumulate.

    Assignment is expensive enough that it must be resumable arm by arm; a plain
    overwrite would silently drop the arms computed in an earlier invocation.
    """
    if new.empty:
        return
    if path.is_file():
        old = pd.read_parquet(path)
        old = old[~old["arm"].isin(set(new["arm"].unique()))]
        new = pd.concat([old, new], ignore_index=True)
    new = new.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)


def _to_patient_level(slides: pd.DataFrame) -> pd.DataFrame:
    """Average slide-level quantities EQUALLY within patient x arm x prototype.

    Equal weighting, not tile-count weighting: it matches the patient-level
    inference scheme the rest of the study uses, so a two-block patient cannot
    outvote a one-block patient in the atlas any more than they can in the AUROC.
    """
    value_cols = [
        c for c in slides.columns
        if c.startswith("attn_mass") or c in ("abundance", "n_tiles_prototype", "n_tiles_slide")
    ]
    context = [c for c in ("role", "cohort", "subcohort", "label", "liver_class",
                           "met_site_class", "technical_class", "mpp_bin",
                           "n_attention_seeds")
               if c in slides.columns]
    grouped = slides.groupby(["arm", "patient_id", "prototype"], as_index=False)
    values = grouped[value_cols].mean()
    first = grouped[context].first() if context else None
    counts = (
        slides.groupby(["arm", "patient_id", "prototype"], as_index=False)["slide_id"]
        .count().rename(columns={"slide_id": "n_slides"})
    )
    out = values
    if first is not None:
        out = out.merge(first, on=["arm", "patient_id", "prototype"], validate="one_to_one")
    return out.merge(counts, on=["arm", "patient_id", "prototype"], validate="one_to_one")


# ── step 3: molecular specificity ────────────────────────────────────────────
QUANTITIES = ("abundance", "attn_mass_mean")


def _effect_table(
    frame: pd.DataFrame, positive_column: str, quantity: str, n_bootstrap: int, seed: int
) -> pd.DataFrame:
    """Per-prototype AUC effect of ``positive_column`` on ``quantity``, BH-adjusted."""
    rows = []
    for prototype, block in frame.groupby("prototype"):
        stats = atlas.bootstrap_auc_effect(
            block[quantity].to_numpy(), block[positive_column].to_numpy(),
            n_bootstrap=n_bootstrap, seed=seed + int(prototype),
        )
        stats["prototype"] = int(prototype)
        stats["p"] = atlas.mannwhitney_p(
            block[quantity].to_numpy(), block[positive_column].to_numpy()
        )
        rows.append(stats)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q"] = atlas.benjamini_hochberg(out["p"].to_numpy())
    out["significant"] = (out["q"] < FDR_ALPHA) & (
        (out["ci_low"] > 0.5) | (out["ci_high"] < 0.5)
    )
    return out.sort_values("prototype").reset_index(drop=True)


def _auc_difference(
    primary_values: np.ndarray,
    primary_labels: np.ndarray,
    metastatic_values: np.ndarray,
    metastatic_labels: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Independent-patient bootstrap of metastatic minus primary AUC.

    Failure to reach significance in the metastatic arm is not evidence that
    an effect changed. This direct contrast supplies the missing interaction
    test. Its normal-approximation p-value is BH-adjusted across prototypes by
    the caller, and a change requires both q < 0.05 and a bootstrap interval
    excluding zero, mirroring the study's single-arm significance rule.
    """
    from scipy import stats

    p_values = np.asarray(primary_values, dtype=float)
    p_labels = np.asarray(primary_labels).astype(int)
    m_values = np.asarray(metastatic_values, dtype=float)
    m_labels = np.asarray(metastatic_labels).astype(int)
    point = atlas.auc_effect(m_values, m_labels) - atlas.auc_effect(p_values, p_labels)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        p_index = rng.integers(0, len(p_values), len(p_values))
        m_index = rng.integers(0, len(m_values), len(m_values))
        if len(np.unique(p_labels[p_index])) < 2 or len(np.unique(m_labels[m_index])) < 2:
            continue
        draws.append(
            atlas.auc_effect(m_values[m_index], m_labels[m_index])
            - atlas.auc_effect(p_values[p_index], p_labels[p_index])
        )
    sampled = np.asarray(draws, dtype=float)
    se = float(sampled.std(ddof=1)) if len(sampled) > 1 else float("nan")
    p_value = (
        float(2 * stats.norm.sf(abs(point / se)))
        if np.isfinite(se) and se > 0 else float("nan")
    )
    return {
        "delta_auc_metastatic_minus_primary": float(point),
        "delta_ci_low": (
            float(np.percentile(sampled, 2.5)) if sampled.size else float("nan")
        ),
        "delta_ci_high": (
            float(np.percentile(sampled, 97.5)) if sampled.size else float("nan")
        ),
        "delta_se": se,
        "delta_p": p_value,
        "n_bootstrap_valid": int(sampled.size),
    }


def _auc_difference_table(
    primary: pd.DataFrame,
    metastatic: pd.DataFrame,
    quantity: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Direct primary-versus-metastatic AUC contrasts with multiplicity control."""
    rows: list[dict[str, Any]] = []
    shared = sorted(set(primary["prototype"]) & set(metastatic["prototype"]))
    for prototype in shared:
        p_block = primary[primary["prototype"].eq(prototype)]
        m_block = metastatic[metastatic["prototype"].eq(prototype)]
        row = _auc_difference(
            p_block[quantity].to_numpy(), p_block["label"].to_numpy(),
            m_block[quantity].to_numpy(), m_block["label"].to_numpy(),
            n_bootstrap=n_bootstrap, seed=seed + int(prototype),
        )
        row["prototype"] = int(prototype)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["delta_q"] = atlas.benjamini_hochberg(out["delta_p"].to_numpy())
    out["changed"] = (out["delta_q"] < FDR_ALPHA) & (
        (out["delta_ci_low"] > 0) | (out["delta_ci_high"] < 0)
    )
    return out.sort_values("prototype").reset_index(drop=True)


def cmd_specificity(args: argparse.Namespace) -> None:
    patients = pd.read_parquet(profile_path(args.k, "patient"))
    e0 = patients[patients["arm"].eq("e0")].copy()
    if e0.empty:
        raise SystemExit("no e0 profiles - run `assign --arm e0` first")
    meta = dev_patients()[
        ["patient_id", "is_mutant", "in_D", "label_complete", "context_positive",
         "msi_braf_cell", "cohort"]
    ]
    e0 = e0.drop(columns=[c for c in ("cohort",) if c in e0.columns]).merge(
        meta, on="patient_id", validate="many_to_one"
    )

    report: dict[str, Any] = {"k": args.k, "populations": {}, "prototypes": {}}
    panels: dict[str, dict[str, pd.DataFrame]] = {}
    for quantity in QUANTITIES:
        panels[quantity] = {
            "A": _effect_table(e0, "is_mutant", quantity, args.n_bootstrap, args.seed),
            "D": _effect_table(
                e0[e0["in_D"].eq(1)], "is_mutant", quantity, args.n_bootstrap, args.seed
            ),
            "context_in_wt": _effect_table(
                e0[e0["is_mutant"].eq(0) & e0["label_complete"].eq(1)],
                "context_positive", quantity, args.n_bootstrap, args.seed,
            ),
        }
    report["populations"] = {
        "A": int(e0["patient_id"].nunique()),
        "D": int(e0.loc[e0["in_D"].eq(1), "patient_id"].nunique()),
        "wt_label_complete": int(
            e0.loc[e0["is_mutant"].eq(0) & e0["label_complete"].eq(1), "patient_id"].nunique()
        ),
    }

    # Cohort concentration: is this morphology specific to one institution?
    # Measured as mean abundance per cohort, NOT tile counts - see
    # atlas.concentration for why a tile-count screen just re-measures cohort
    # size.
    cohort_share = e0.groupby(["prototype", "cohort"])["abundance"].mean()

    # Acquisition concentration: is this morphology really a scanner setting?
    #
    # MEASURED WITHIN COHORT, and that conditioning is the whole point. Scanned
    # resolution is almost perfectly confounded with institution here - CPTAC,
    # SurGen and TCGA are all ~0.25 um/px - so an unconditioned screen flags a
    # prototype as "mpp~0.25 dominated" for the sole reason that it is
    # SurGen-dominated, and re-reports the cohort axis under a second name. Only
    # a cohort that actually spans several acquisition levels can separate the
    # scanner from the institution that owns it: RIH covers three mpp bins and
    # TCGA two, and both span several technical classes. A prototype
    # concentrated in one level INSIDE such a cohort is evidence of an
    # acquisition artefact; one concentrated across cohorts is not.
    acquisition_share: dict[str, dict[str, pd.Series]] = {}
    for column in ("mpp_bin", "technical_class"):
        if column not in e0.columns or not e0[column].notna().any():
            continue
        per_cohort_levels = e0.drop_duplicates("patient_id").groupby("cohort")[column].nunique()
        informative = [c for c, n in per_cohort_levels.items() if n > 1]
        if not informative:
            continue
        acquisition_share[column] = {
            str(cohort): e0[e0["cohort"].eq(cohort)]
            .groupby(["prototype", column])["abundance"].mean()
            for cohort in informative
        }

    # Cohort reproducibility: the same KRAS contrast run inside each cohort.
    # An association that only exists once the cohorts are pooled is a case-mix
    # effect wearing a KRAS label, and pooled A cannot tell the two apart.
    per_cohort: dict[str, dict[str, pd.DataFrame]] = {}
    for quantity in QUANTITIES:
        per_cohort[quantity] = {
            str(cohort): _effect_table(
                block, "is_mutant", quantity, args.n_bootstrap, args.seed
            )
            for cohort, block in e0.groupby("cohort")
            if block.drop_duplicates("patient_id")["is_mutant"].nunique() > 1
        }

    for prototype in sorted(e0["prototype"].unique()):
        block: dict[str, Any] = {"prototype": int(prototype)}
        for quantity, tables in panels.items():
            block[quantity] = {
                name: _row_dict(table, prototype) for name, table in tables.items()
            }
        block["cohort_concentration"] = atlas.concentration(cohort_share.loc[int(prototype)])
        acquisition: dict[str, Any] = {}
        for column, by_cohort in acquisition_share.items():
            for cohort, share in by_cohort.items():
                if int(prototype) not in share.index.get_level_values(0):
                    continue
                levels = share.loc[int(prototype)]
                stats = atlas.concentration(levels)
                stats["n_levels"] = int(len(levels))
                stats["enrichment"] = (
                    float(stats["top_share"] * len(levels))
                    if np.isfinite(stats["top_share"]) else float("nan")
                )
                acquisition[f"{column}@{cohort}"] = stats
        block["acquisition_concentration"] = acquisition
        block["by_cohort"] = {
            quantity: {
                cohort: _row_dict(table, int(prototype))
                for cohort, table in tables.items()
            }
            for quantity, tables in per_cohort.items()
        }
        block["cohort_reproducibility"] = _reproducibility(block["by_cohort"])
        block["msi_braf_cells"] = _context_cells(e0, int(prototype))
        block["classification"] = _classify_prototype(block)
        report["prototypes"][str(int(prototype))] = block

    print(f"\n{'=' * 108}\nE4 STEP 3 - MOLECULAR SPECIFICITY  (A n={report['populations']['A']}, "
          f"D n={report['populations']['D']})\n{'=' * 108}")
    for quantity in QUANTITIES:
        print(f"\n  {quantity.upper()}: KRAS-mutant vs wild-type, AUC effect (0.5 = no effect)")
        print(f"  {'proto':>5s} {'A AUC [CI]':>26s} {'q':>8s} {'D AUC [CI]':>26s} {'q':>8s} "
              f"{'top cohort':>18s}")
        for prototype in sorted(e0["prototype"].unique()):
            b = report["prototypes"][str(int(prototype))]
            a_s, d_s = b[quantity]["A"], b[quantity]["D"]
            conc = b["cohort_concentration"]
            top = "{} {:.2f}".format(conc["top_key"], conc["top_share"])
            print(f"  {int(prototype):5d} {_fmt_effect(a_s):>26s} {_fmt_q(a_s):>8s} "
                  f"{_fmt_effect(d_s):>26s} {_fmt_q(d_s):>8s} {top:>18s}")

    print("\n  CLASSIFICATION  (repro = cohorts agreeing on the direction of the "
          "abundance effect)")
    for prototype in sorted(e0["prototype"].unique()):
        b = report["prototypes"][str(int(prototype))]
        repro = b["cohort_reproducibility"].get("abundance", {})
        agree = f"{repro.get('agreeing', 0)}/{repro.get('n_cohorts', 0)}"
        print(f"    proto {int(prototype):3d}  {b['classification']['bucket']:38s} "
              f"repro {agree:>5s}  {'; '.join(b['classification']['flags']) or '-'}")

    dest = paths.EVAL_ROOT / f"e3b_specificity_k{args.k}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {dest}")


def _reproducibility(by_cohort: dict[str, dict[str, dict]]) -> dict[str, Any]:
    """How many cohorts agree on the direction of a prototype's KRAS effect.

    Direction, not significance: at 94-737 patients per cohort, per-cohort CIs
    are wide enough that requiring significance in every cohort would reject
    every prototype. Sign agreement is the honest reproducibility claim at
    these sizes, and the count is reported rather than thresholded into a
    verdict.
    """
    out: dict[str, Any] = {}
    for quantity, cohorts in by_cohort.items():
        signs = [
            np.sign(stats["auc"] - 0.5)
            for stats in cohorts.values()
            if stats and np.isfinite(stats.get("auc", np.nan))
        ]
        if not signs:
            out[quantity] = {"n_cohorts": 0, "agreeing": 0, "direction": None}
            continue
        positive = int(sum(1 for x in signs if x > 0))
        negative = int(sum(1 for x in signs if x < 0))
        out[quantity] = {
            "n_cohorts": len(signs),
            "agreeing": max(positive, negative),
            "direction": "higher_in_mutant" if positive >= negative else "higher_in_wt",
        }
    return out


def _context_cells(e0: pd.DataFrame, prototype: int) -> dict[str, Any]:
    """Prototype behaviour among KRAS-WT patients, by MSI x BRAF cell.

    The same four cells as Why-D Analysis C, and for the same reason: MSI and
    BRAF are correlated, so a prototype that looks "KRAS-associated" in A may
    simply be the morphology of one of these cells. Reported as descriptive
    means - the inferential statement is the A-vs-D contrast, not a four-way
    comparison at n = 40-615.
    """
    block = e0[e0["prototype"].eq(prototype) & e0["is_mutant"].eq(0)
               & e0["label_complete"].eq(1)]
    out: dict[str, Any] = {}
    for cell, rows in block.groupby("msi_braf_cell"):
        out[str(cell)] = {
            "n": int(len(rows)),
            **{q: float(rows[q].mean()) for q in QUANTITIES},
        }
    return out


def _row_dict(table: pd.DataFrame, prototype: int) -> dict[str, Any]:
    if table.empty:
        return {}
    hit = table[table["prototype"].eq(int(prototype))]
    return {} if hit.empty else {k: _plain(v) for k, v in hit.iloc[0].to_dict().items()}


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fmt_effect(stats: dict[str, Any]) -> str:
    if not stats or not np.isfinite(stats.get("auc", np.nan)):
        return "-"
    return f"{stats['auc']:.3f} [{stats['ci_low']:.3f},{stats['ci_high']:.3f}]"


def _fmt_q(stats: dict[str, Any]) -> str:
    q = stats.get("q", np.nan) if stats else np.nan
    return "-" if not np.isfinite(q) else f"{q:.3g}" + ("*" if stats.get("significant") else "")


def _classify_prototype(block: dict[str, Any]) -> dict[str, Any]:
    """The four buckets of §5, applied to abundance; attention is reported beside.

    Abundance carries the classification because it is a property of the TISSUE.
    Attention is a property of the model, so a prototype the model leans on
    without a tissue-level difference is a separate, weaker statement and is
    recorded as a flag rather than promoted to a bucket.
    """
    a = block["abundance"].get("A", {})
    d = block["abundance"].get("D", {})
    ctx = block["abundance"].get("context_in_wt", {})
    attn_a = block["attn_mass_mean"].get("A", {})
    conc = block["cohort_concentration"]

    sig_a = bool(a.get("significant"))
    sig_d = bool(d.get("significant"))
    same_sign = (
        np.sign(a.get("auc", np.nan) - 0.5) == np.sign(d.get("auc", np.nan) - 0.5)
        if a and d else False
    )
    sig_ctx = bool(ctx.get("significant"))
    dominated = np.isfinite(conc.get("top_share", np.nan)) and (
        conc["top_share"] >= CONCENTRATION_FLAG
    )
    acq = block.get("acquisition_concentration", {})
    acq_hits = [
        (column, c) for column, c in acq.items()
        if column.split("@")[0] in ACQUISITION_FLAG_COLUMNS
        and np.isfinite(c.get("enrichment", np.nan))
        and c["enrichment"] >= ACQUISITION_ENRICHMENT
    ]

    if sig_a and sig_d and same_sign:
        bucket = "kras_associated_dependency_robust"
    elif sig_a and sig_ctx:
        bucket = "shared_msi_braf_context_morphology"
    elif sig_a:
        bucket = "kras_associated_attenuates_in_D"
    elif dominated:
        bucket = "cohort_associated"
    else:
        bucket = "not_kras_associated"

    flags = []
    if dominated:
        flags.append(f"cohort_dominated({conc['top_key']} {conc['top_share']:.2f})")
    for column, c in acq_hits:
        flags.append(
            f"acquisition_dominated({column}={c['top_key']} "
            f"{c['top_share']:.2f}, {c['enrichment']:.1f}x)"
        )
    if bool(attn_a.get("significant")):
        direction = "mutants" if attn_a.get("auc", 0.5) > 0.5 else "wild_type"
        flags.append(f"attention_differs_by_kras(higher_in_{direction})")
    if sig_a and not sig_d:
        flags.append("association_lost_in_D")
    if sig_ctx:
        flags.append("context_associated_among_WT")
    return {"bucket": bucket, "flags": flags}


# ── step 4: primary -> metastatic conservation ───────────────────────────────
def cmd_transport(args: argparse.Namespace) -> None:
    patients = pd.read_parquet(profile_path(args.k, "patient"))
    dual = rih_dual_role_patients()
    report: dict[str, Any] = {
        "k": args.k,
        "rih_dual_role_excluded_from_contrast": sorted(dual),
        "arms": {}, "conservation": {},
    }

    for name in ("rih_primary", "rih_metastatic", "sr1482_primary", "sr1482_metastatic"):
        block = patients[patients["arm"].eq(name)]
        if block.empty:
            print(f"  {name}: no profiles - run `assign --arm {name}` first")
            continue
        report["arms"][name] = {
            "n_patients": int(block["patient_id"].nunique()),
            "n_mutant": int(block.drop_duplicates("patient_id")["label"].sum()),
            "effects": {
                q: _effect_table(block, "label", q, args.n_bootstrap, args.seed)
                .to_dict("records")
                for q in QUANTITIES
            },
        }

    for pair_index, (cohort, primary_arm, met_arm) in enumerate(TRANSPORT_PAIRS):
        if primary_arm not in report["arms"] or met_arm not in report["arms"]:
            continue
        # The paired-patient rule: retained in the standalone panels above,
        # removed here, where a patient in both arms would make the contrast
        # non-independent.
        p_side = patients[patients["arm"].eq(primary_arm) & ~patients["patient_id"].isin(dual)]
        m_side = patients[patients["arm"].eq(met_arm) & ~patients["patient_id"].isin(dual)]
        conservation = {}
        for quantity in QUANTITIES:
            pe = _effect_table(p_side, "label", quantity, args.n_bootstrap, args.seed)
            me = _effect_table(m_side, "label", quantity, args.n_bootstrap, args.seed)
            merged = pe.merge(me, on="prototype", suffixes=("_primary", "_metastatic"))
            differences = _auc_difference_table(
                p_side, m_side, quantity, args.n_bootstrap,
                args.seed + (pair_index + 1) * 10_000,
            )
            merged = merged.merge(differences, on="prototype", validate="one_to_one")
            merged["same_direction"] = (
                np.sign(merged["auc_primary"] - 0.5) == np.sign(merged["auc_metastatic"] - 0.5)
            )
            merged["conserved"] = (
                merged["significant_primary"] & merged["significant_metastatic"]
                & merged["same_direction"]
            )
            conservation[quantity] = merged.to_dict("records")
        report["conservation"][cohort] = {
            "n_primary": int(p_side["patient_id"].nunique()),
            "n_metastatic": int(m_side["patient_id"].nunique()),
            "by_quantity": conservation,
        }

    # Shortcut screen: is a prototype's metastatic tissue really one organ?
    # Mean abundance per organ class, for the same size-free reason as the
    # cohort screen: liver is most of the metastatic material, so a tile-count
    # screen would flag every prototype as liver-dominated.
    met = patients[patients["role"].eq("metastatic")]
    if not met.empty and "met_site_class" in met.columns:
        organ = met.groupby(["prototype", "met_site_class"])["abundance"].mean()
        report["organ_concentration"] = {
            str(int(p)): atlas.concentration(organ.loc[int(p)])
            for p in sorted(met["prototype"].unique())
        }

    print(f"\n{'=' * 108}\nE4 STEP 4 - PRIMARY -> METASTATIC CONSERVATION\n{'=' * 108}")
    for cohort, block in report["conservation"].items():
        print(f"\n  {cohort}: primary n={block['n_primary']} vs metastatic "
              f"n={block['n_metastatic']} (dual-role patients excluded)")
        for quantity, records in block["by_quantity"].items():
            conserved = [r["prototype"] for r in records if r["conserved"]]
            changed = [r["prototype"] for r in records if r["changed"]]
            flipped = [r["prototype"] for r in records
                       if r["significant_primary"] and not r["same_direction"]]
            print(f"    {quantity:16s} conserved {conserved or '-'} | "
                  f"directly changed {changed or '-'} | sign-flipped {flipped or '-'}")
    if "organ_concentration" in report:
        print("\n  METASTATIC ORGAN CONCENTRATION (shortcut screen)")
        for prototype, conc in report["organ_concentration"].items():
            if np.isfinite(conc["top_share"]) and conc["top_share"] >= CONCENTRATION_FLAG:
                print(f"    proto {prototype:>3s}  normalized mean-abundance concentration "
                      f"{conc['top_share']:.2f} in {conc['top_key']}  <- FLAG")

    dest = paths.EVAL_ROOT / f"e3b_transport_k{args.k}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {dest}")


# ── the final readout: specificity x conservation x shortcut ─────────────────
# The outcome groups of the design's step 10. They cannot be decided inside
# `specificity`, which runs before `transport` and so cannot see whether an
# association carries to metastatic tissue; this is where the two meet.
GROUPS = (
    "dependency_robust_conserved",
    "dependency_robust_changed",
    "dependency_robust_transport_heterogeneous",
    "dependency_robust_transport_inconclusive",
    "dependency_robust_transport_underpowered",
    "dependency_robust_transport_not_evaluated",
    "context_dependent",
    "restriction_sensitive_unexplained",
    "shortcut_technical",
    "not_kras_associated",
)
CLAIMED_GROUPS = GROUPS[:8]


def _conservation_state(prototype: int, transport: dict[str, Any]) -> dict[str, Any]:
    """Did a primary association carry, directly change, or remain unresolved?

    A non-significant metastatic effect is not evidence of a primary-to-
    metastatic difference. ``changed`` therefore requires the direct AUC
    contrast written by :func:`cmd_transport`; ``inconclusive`` means a primary
    effect existed but neither conservation nor change was demonstrated; and
    ``underpowered`` means neither transport cohort had a primary effect to test.
    """
    per_cohort: dict[str, Any] = {}
    for cohort, block in transport.get("conservation", {}).items():
        row = next(
            (r for r in block["by_quantity"]["abundance"] if int(r["prototype"]) == prototype),
            None,
        )
        if row is None:
            continue
        per_cohort[cohort] = {
            "auc_primary": row["auc_primary"],
            "auc_metastatic": row["auc_metastatic"],
            "primary_significant": bool(row["significant_primary"]),
            "metastatic_significant": bool(row["significant_metastatic"]),
            "same_direction": bool(row["same_direction"]),
            "conserved": bool(row["conserved"]),
            "changed": bool(row.get("changed", False)),
            "delta_auc_metastatic_minus_primary": row.get(
                "delta_auc_metastatic_minus_primary"
            ),
            "delta_ci_low": row.get("delta_ci_low"),
            "delta_ci_high": row.get("delta_ci_high"),
            "delta_q": row.get("delta_q"),
            "n_primary": block["n_primary"],
            "n_metastatic": block["n_metastatic"],
        }
    if not per_cohort:
        return {"state": "not_evaluated", "by_cohort": {}}
    conserved_anywhere = any(c["conserved"] for c in per_cohort.values())
    changed = [c for c in per_cohort.values() if c["changed"]]
    contradictory = any(
        c["primary_significant"] and not c["same_direction"]
        for c in per_cohort.values()
    )
    changed_signs = {
        int(np.sign(c["delta_auc_metastatic_minus_primary"]))
        for c in changed
        if c["delta_auc_metastatic_minus_primary"] is not None
        and np.isfinite(c["delta_auc_metastatic_minus_primary"])
    }
    if (conserved_anywhere and (changed or contradictory)) or len(changed_signs) > 1:
        state = "heterogeneous"
    elif conserved_anywhere:
        state = "conserved"
    elif changed:
        state = "changed"
    elif any(c["primary_significant"] for c in per_cohort.values()):
        state = "inconclusive"
    else:
        state = "underpowered"
    return {"state": state, "by_cohort": per_cohort}


def _shortcut_flags(prototype: int, block: dict[str, Any],
                    transport: dict[str, Any]) -> dict[str, list[str]]:
    """Source flags, split by WHICH claim each one can legitimately touch.

    The two are not interchangeable and must not be pooled into one list.

    `source` - cohort and acquisition concentration, both measured in the E0
    primary discovery arm. These bear on the primary KRAS association itself: if
    a prototype's tissue comes overwhelmingly from one institution or one
    scanner setting, its association and its source are not separable.

    `organ` - metastatic-organ concentration, measured in metastatic tissue
    ONLY. This bears on CONSERVATION and on nothing else. M04 makes the
    distinction concrete: its normalized mean-abundance concentration is 0.76
    for peritoneum in metastatic tissue, while its
    KRAS association was established across 1,486 primary tumours where no
    metastatic organ exists at all. Letting the organ screen suppress that
    primary finding would retract a result using evidence from a different
    population.
    """
    source = [f for f in block["classification"]["flags"]
              if f.startswith(("cohort_dominated", "acquisition_dominated"))]
    organ_flags: list[str] = []
    organ = transport.get("organ_concentration", {}).get(str(prototype), {})
    if organ and np.isfinite(organ.get("top_share", np.nan)) and (
        organ["top_share"] >= CONCENTRATION_FLAG
    ):
        organ_flags.append(f"organ_dominated({organ['top_key']} {organ['top_share']:.2f})")
    return {"source": source, "organ": organ_flags}


def final_readout(spec: dict[str, Any], transport: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """One row per prototype: association, specificity, conservation, shortcut.

    ORDER OF PRECEDENCE. The shortcut check is evaluated FIRST and dominates, on
    the same logic E3's underpowered check does: a prototype whose tissue comes
    overwhelmingly from one cohort, one scanner setting or one metastatic organ
    cannot be reported as morphology however strong its KRAS association looks,
    because the association and the source are not separable in this design. The
    underlying specificity and conservation are still recorded on the row, so
    nothing is hidden by the precedence - only the headline changes.
    """
    out: dict[int, dict[str, Any]] = {}
    # JSON written with ``sort_keys=True`` orders string prototype keys
    # lexicographically (0, 1, 10, ...).  Keep the scientific readout in
    # numeric prototype order regardless of the serialization used upstream.
    for key in sorted(spec["prototypes"], key=lambda value: int(value)):
        block = spec["prototypes"][key]
        prototype = int(key)
        a = block["abundance"].get("A", {})
        d = block["abundance"].get("D", {})
        attn = block["attn_mass_mean"].get("A", {})
        sig_a, sig_d = bool(a.get("significant")), bool(d.get("significant"))
        same_sign = (
            np.sign(a.get("auc", np.nan) - 0.5) == np.sign(d.get("auc", np.nan) - 0.5)
            if a and d else False
        )
        flags = _shortcut_flags(prototype, block, transport)
        conservation = _conservation_state(prototype, transport)
        bucket = block["classification"]["bucket"]

        # Source flags suppress the primary claim; an organ flag can only
        # withdraw a CONSERVED verdict, because that is the only claim it is
        # evidence about.
        organ_confounds_conservation = bool(flags["organ"]) and conservation["state"] == "conserved"
        if flags["source"] or organ_confounds_conservation:
            group = "shortcut_technical"
        elif sig_a and sig_d and same_sign:
            group = {
                "conserved": "dependency_robust_conserved",
                "changed": "dependency_robust_changed",
                "heterogeneous": "dependency_robust_transport_heterogeneous",
                "inconclusive": "dependency_robust_transport_inconclusive",
                "underpowered": "dependency_robust_transport_underpowered",
                "not_evaluated": "dependency_robust_transport_not_evaluated",
            }[conservation["state"]]
        elif sig_a:
            group = (
                "context_dependent"
                if bucket == "shared_msi_braf_context_morphology"
                else "restriction_sensitive_unexplained"
            )
        else:
            group = "not_kras_associated"

        direction = ("higher in mutants" if a.get("auc", 0.5) > 0.5 else "higher in wild-type")
        out[prototype] = {
            "prototype": prototype,
            "group": group,
            "specificity_bucket": bucket,
            "classification_flags": block["classification"].get("flags", []),
            "kras_association": {
                "auc_A": a.get("auc"), "q_A": a.get("q"), "significant": sig_a,
                "direction": direction if sig_a else None,
            },
            "model_attends": bool(attn.get("significant")),
            "attention_association": {
                "auc_A": attn.get("auc"),
                "q_A": attn.get("q"),
                "significant": bool(attn.get("significant")),
                "direction": (
                    "higher in mutants" if attn.get("auc", 0.5) > 0.5
                    else "higher in wild-type"
                ) if bool(attn.get("significant")) else None,
            },
            "specificity_effects": {
                quantity: block.get(quantity, {}) for quantity in QUANTITIES
            },
            "persists_in_D": (None if not sig_a else bool(sig_d and same_sign)),
            "conservation": conservation,
            "shortcut_flags": flags["source"],
            "organ_flags": flags["organ"],
            "organ_confounds_conservation": organ_confounds_conservation,
            "cohort_reproducibility": block.get("cohort_reproducibility", {}).get("abundance", {}),
            "attention_cohort_reproducibility": (
                block.get("cohort_reproducibility", {}).get("attn_mass_mean", {})
            ),
        }
    return out


def review_prototypes(readout: dict[int, dict[str, Any]],
                      n_technical: int = 6) -> dict[str, list[int]]:
    """Every prototype a claim depends on, plus enough of the rest to name it.

    Selection is driven by the readout rather than a fixed top-N, so nothing the
    final table asserts can reach the manuscript undescribed. Four strata:

    `claimed`    every prototype in a claimed group.
    `suppressed` prototypes whose source flag is DOING WORK - they carry a real
                 KRAS relationship (abundance association or model attention)
                 that the shortcut call is what withholds. These are the ones
                 where "is this morphology or is this the scanner?" decides a
                 result, so a pathologist has to look.
    `attention_followup` prototypes with significant attention differences that
                 are not already claimed or suppressed. Attention is a separate
                 model-behaviour result and must not promote one of these into an
                 abundance-based KRAS morphology group, but the morphology still
                 needs a blinded description.
    `technical`  the most source-dominated of the remainder, capped, so the
                 shortcut axis itself gets characterised - the design asks
                 explicitly whether these are normal organ tissue or artefact,
                 and that question cannot be answered from a concentration
                 statistic alone.

    A cohort-dominated prototype with no KRAS association and no attention is
    deliberately NOT reviewed in full: no claim rests on it, and montaging all
    of them would spend the pathologist's time proving something the
    concentration screen already reports.
    """
    claimed = sorted(p for p, r in readout.items() if r["group"] in CLAIMED_GROUPS)
    suppressed = sorted(
        p for p, r in readout.items()
        if r["group"] == "shortcut_technical"
        and (r["kras_association"]["significant"] or r["model_attends"])
    )
    already_selected = set(claimed) | set(suppressed)
    attention_followup = sorted(
        p for p, r in readout.items()
        if r["model_attends"] and p not in already_selected
    )
    rest = [p for p, r in readout.items()
            if r["group"] == "shortcut_technical" and p not in suppressed]

    def dominance(prototype: int) -> float:
        flags = readout[prototype]["shortcut_flags"]
        return max((float(f.split()[-1].rstrip(",)x")) for f in flags), default=0.0)

    # Prototype ID is the predeclared deterministic tie-break.  Relying on the
    # insertion order of ``readout`` made equal-dominance prototypes change
    # membership after a strict-JSON ``sort_keys=True`` round trip.
    technical = sorted(
        sorted(rest, key=lambda prototype: (-dominance(prototype), prototype))[
            :n_technical
        ]
    )
    return {
        "claimed": claimed,
        "suppressed": suppressed,
        "attention_followup": attention_followup,
        "technical": technical,
    }


# ── step 5: blinded montages ─────────────────────────────────────────────────
_WSI_SUFFIXES = (".svs", ".tiff", ".tif")
# Precedence mirrors the slide README: corrected copies win, backups of known
# defects are never read.
_SLIDE_DIRS = (
    "SURGEN", "rih_fixed", "rih", "TCGA", "CPTAC_COAD",
)


def slide_path_index(slide_root: Path, cache: Path) -> dict[str, str]:
    """slide stem -> file path, built once and cached.

    Directory precedence matters: SurGen's R/B-swapped originals and RIH's
    pre-repair SVS files are still on disk under quarantine/backup names, and a
    montage cut from those would show the wrong colours or the wrong tile grid.
    """
    if cache.is_file():
        return json.loads(cache.read_text())
    index: dict[str, str] = {}
    for name in _SLIDE_DIRS:
        directory = slide_root / name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in _WSI_SUFFIXES and path.stem not in index:
                index[path.stem] = str(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(index, indent=2))
    return index


def _patch_geometry(slide_id: str) -> dict[str, Any]:
    """Level-0 read size for one tile, from the patcher's own record."""
    path = paths.PINNED_FEATURE_DIR.parent / "patches" / f"{slide_id}_patches.h5"
    with h5py.File(path, "r") as handle:
        attrs = dict(handle["coords"].attrs)
    return {
        "patch_size_level0": int(attrs["patch_size_level0"]),
        "patch_size": int(attrs["patch_size"]),
        "coordinate_units": str(attrs.get("coordinate_units", "level0_pixels")),
    }


def cmd_montages(args: argparse.Namespace) -> None:
    """Blinded review packets, plus a key written where the reviewer will not read it."""
    packet = REVIEW_ROOT / f"k{args.k}"
    form = packet / "review_form.csv"
    if form.is_file() and _review_form_has_content(form) and not args.force:
        raise SystemExit(
            f"{form} contains review annotations; refusing to overwrite the completed "
            "blinded packet. Archive it first or pass `--force` only if replacement is "
            "intentional."
        )

    from PIL import Image

    try:
        import openslide  # noqa: F401
    except ImportError as exc:  # a missing reader must not look like unreadable slides
        raise SystemExit(
            "montages need a WSI reader: `uv sync --extra atlas` (openslide-python). "
            "Every other E4 step runs without it."
        ) from exc

    candidates = pd.read_parquet(candidates_path(args.k))
    spec_path = paths.EVAL_ROOT / f"e3b_specificity_k{args.k}.json"
    trans_path = paths.EVAL_ROOT / f"e3b_transport_k{args.k}.json"
    readout: dict[int, dict[str, Any]] = {}
    if spec_path.is_file():
        readout = final_readout(
            json.loads(spec_path.read_text()),
            json.loads(trans_path.read_text()) if trans_path.is_file() else {},
        )
    if readout and not args.top:
        strata = review_prototypes(readout, n_technical=args.technical)
        chosen = sorted({p for group in strata.values() for p in group})
        print(f"Readout-driven selection, {len(chosen)} montages:")
        print(f"  claimed              {strata['claimed'] or '-'}")
        print(f"  suppressed by a flag {strata['suppressed'] or '-'}")
        print(f"  attention follow-up  {strata['attention_followup'] or '-'}")
        print(f"  technical sample     {strata['technical'] or '-'}")
    else:
        ranking = _montage_ranking(args.k, candidates)
        chosen = ranking[: (args.top or 8)]
        print(f"Montages for prototypes {chosen} (top {args.top or 8} by |A effect| "
              "and attention; run `specificity` first for readout-driven selection).")

    index = slide_path_index(Path(args.slide_root), MONTAGE_DIR / "slide_path_index.json")
    blind_dir = packet / "montages"
    blind_dir.mkdir(parents=True, exist_ok=True)
    key_dir = MONTAGE_DIR / f"k{args.k}"
    key_dir.mkdir(parents=True, exist_ok=True)
    key_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(chosen))
    for position, prototype in enumerate(np.asarray(chosen)[order]):
        montage_id = f"M{position + 1:02d}"
        pool = candidates[candidates["prototype"].eq(int(prototype))]
        picks = _diverse_picks(pool, args.tiles_per_montage, rng)
        images, provenance = [], []
        for pick in picks.itertuples(index=False):
            path = index.get(pick.slide_id)
            if path is None:
                continue
            try:
                tile = _read_tile(Image, path, pick.slide_id, pick.x, pick.y, args.tile_px)
            except Exception as exc:  # noqa: BLE001 - one unreadable slide must not stop the run
                print(f"    skip {pick.slide_id} ({exc})")
                continue
            images.append(tile)
            provenance.append(pick)
        if not images:
            print(f"    prototype {prototype}: no readable tiles - skipped")
            continue
        grid = _grid(Image, images, args.tile_px)
        grid.save(blind_dir / f"{montage_id}.jpg", quality=92)
        review_rows.append({"montage_id": montage_id, "n_tiles": len(images),
                            "architecture": "", "mucin": "", "dirty_necrosis": "",
                            "differentiation": "", "desmoplasia_stroma": "",
                            "budding_invasion": "", "immune_infiltration": "",
                            "normal_organ_tissue": "", "artifact": "", "free_text": ""})
        for slot, pick in enumerate(provenance):
            key_rows.append({"montage_id": montage_id, "slot": slot,
                             "prototype": int(prototype), **pick._asdict()})
        print(f"  {montage_id}: prototype {prototype}, {len(images)} tiles")

    pd.DataFrame(review_rows).to_csv(packet / "review_form.csv", index=False)
    pd.DataFrame(key_rows).to_csv(key_dir / "KEY_do_not_open_before_review.csv", index=False)
    (packet / "README.md").write_text(_review_instructions(len(review_rows), args))
    print(f"\nReview packet  -> {packet}")
    print(f"  montages     -> {blind_dir}  ({len(review_rows)} images)")
    print(f"  form         -> {packet / 'review_form.csv'}")
    print(f"  instructions -> {packet / 'README.md'}")
    print(f"Key (NOT in the packet) -> {key_dir / 'KEY_do_not_open_before_review.csv'}")


def cmd_montage_addendum(args: argparse.Namespace) -> None:
    """Generate one add-only blinded montage without touching a completed packet."""
    montage_id = str(args.montage_id).upper()
    if not (
        len(montage_id) == 3
        and montage_id.startswith("M")
        and montage_id[1:].isdigit()
        and int(montage_id[1:]) > 0
    ):
        raise SystemExit("--montage-id must have the form M11")

    packet = REVIEW_ROOT / f"k{args.k}"
    image_path = packet / "montages" / f"{montage_id}.jpg"
    key_dir = MONTAGE_DIR / f"k{args.k}"
    key_path = key_dir / f"KEY_{montage_id}_followup_do_not_open_before_review.csv"
    canonical_key = key_dir / "KEY_do_not_open_before_review.csv"

    if canonical_key.is_file():
        locked = pd.read_csv(
            canonical_key, dtype={"montage_id": str, "prototype": int}
        )
        if montage_id in set(locked["montage_id"]):
            raise SystemExit(f"{montage_id} already belongs to the completed packet")
        if int(args.prototype) in set(locked["prototype"]):
            raise SystemExit(
                f"prototype {args.prototype} is already present in the completed packet"
            )
    existing = [path for path in (image_path, key_path) if path.exists()]
    if existing and not args.force:
        raise SystemExit(
            "refusing to overwrite existing montage addendum output(s): "
            + ", ".join(str(path) for path in existing)
        )

    spec_path = paths.EVAL_ROOT / f"e3b_specificity_k{args.k}.json"
    trans_path = paths.EVAL_ROOT / f"e3b_transport_k{args.k}.json"
    if not spec_path.is_file() or not trans_path.is_file():
        raise SystemExit("specificity and transport artifacts are required")
    readout = final_readout(
        json.loads(spec_path.read_text()), json.loads(trans_path.read_text())
    )
    attention_followup = review_prototypes(
        readout, n_technical=args.technical
    )["attention_followup"]
    if int(args.prototype) not in attention_followup:
        raise SystemExit(
            f"prototype {args.prototype} is not in the attention-follow-up selection "
            f"{attention_followup}"
        )

    from PIL import Image

    try:
        import openslide  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "montage addenda need a WSI reader: `uv sync --extra atlas` "
            "(openslide-python)"
        ) from exc

    candidates = pd.read_parquet(candidates_path(args.k))
    pool = candidates[candidates["prototype"].eq(int(args.prototype))]
    rng = np.random.default_rng(args.seed)
    picks = _diverse_picks(pool, args.tiles_per_montage, rng)
    site = ["slide_id", "x", "y"]
    if len(picks) != args.tiles_per_montage:
        raise SystemExit(
            f"{montage_id} selection returned {len(picks)} of "
            f"{args.tiles_per_montage} requested tiles"
        )
    if picks.duplicated(site).any():
        raise SystemExit(f"{montage_id} selection contains duplicate tile positions")
    if picks["slide_id"].nunique() != len(picks):
        raise SystemExit(f"{montage_id} requires one distinct slide per tile")
    if picks["patient_id"].nunique() != len(picks):
        raise SystemExit(f"{montage_id} requires one distinct patient per tile")

    index = slide_path_index(
        Path(args.slide_root), MONTAGE_DIR / "slide_path_index.json"
    )
    images: list[Any] = []
    provenance: list[Any] = []
    for pick in picks.itertuples(index=False):
        path = index.get(pick.slide_id)
        if path is None:
            raise SystemExit(f"no WSI path found for {pick.slide_id}")
        try:
            tile = _read_tile(
                Image, path, pick.slide_id, pick.x, pick.y, args.tile_px
            )
        except Exception as exc:  # noqa: BLE001 - report the exact unreadable input
            raise SystemExit(f"could not render {pick.slide_id}: {exc}") from exc
        images.append(tile)
        provenance.append(pick)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    grid = _grid(Image, images, args.tile_px)
    grid.save(image_path, quality=92)
    key_rows = [
        {
            "montage_id": montage_id,
            "slot": slot,
            "prototype": int(args.prototype),
            "selection_stratum": "attention_followup",
            **pick._asdict(),
        }
        for slot, pick in enumerate(provenance)
    ]
    pd.DataFrame(key_rows).to_csv(key_path, index=False)

    print(f"Blinded addendum -> {image_path} ({len(images)} distinct patients/slides)")
    print(f"Key (NOT in the packet) -> {key_path}")


def _review_instructions(n_montages: int, args: argparse.Namespace) -> str:
    """The reviewer's page. Written into the packet so it travels with the images."""
    return f"""# E4 - blinded morphology review

{n_montages} montages, each a grid of {args.tiles_per_montage} tiles drawn from ONE
morphology prototype. Prototypes were discovered by unsupervised clustering of
frozen UNIv1 embeddings; the clustering never saw KRAS, MSI, BRAF, cohort or
specimen role.

## You are blinded

The montage identifiers (M01, M02, ...) are in randomised order and carry no
information. Nothing in this folder records, for any montage:

* KRAS status
* MSI / dMMR or BRAF status
* cohort or institution
* primary versus metastatic origin

That mapping exists, but is deliberately stored outside this packet so it cannot
be read by accident. Please do not ask for it before the form is complete.

## What we are asking

Describe **what the morphology is**, not what you think it predicts. Sampling
spans institutions and specimen roles whenever that prototype is available in
them, so a montage may legitimately look heterogeneous - say so if it does.

Fill in `review_form.csv`, one row per montage. Free-text columns are welcome;
leave a field blank rather than guessing. The prompts are:

| column | what to record |
| ------ | -------------- |
| `architecture` | glandular, cribriform, solid, papillary, mixed |
| `mucin` | extracellular / intracellular mucin, signet-ring cells |
| `dirty_necrosis` | necrosis, karyorrhectic debris |
| `differentiation` | well / moderate / poor |
| `desmoplasia_stroma` | desmoplastic reaction, fibrosis, stromal density |
| `budding_invasion` | tumour budding, infiltrative margin, perineural or vascular invasion |
| `immune_infiltration` | TILs, Crohn-like reaction, lymphoid aggregates |
| `normal_organ_tissue` | non-tumour tissue - normal colon, liver parenchyma, lung, peritoneum |
| `artifact` | folds, blur, pen ink, bubbles, tissue tears, staining or scanner artefact |
| `free_text` | anything else, including "not interpretable" |

The last two columns matter as much as the first: part of this experiment is
finding out which prototypes are **not** morphology at all. A montage that is
mostly normal liver, or mostly a scanning artefact, is a useful and expected
answer.

Tiles are {args.tile_px} px squares rendered from their recorded level-0
coordinates.
"""


def _montage_ranking(k: int, candidates: pd.DataFrame) -> list[int]:
    """Prototypes worth a pathologist's time: strongest KRAS effect, then attention."""
    spec = paths.EVAL_ROOT / f"e3b_specificity_k{k}.json"
    if spec.is_file():
        blocks = json.loads(spec.read_text())["prototypes"]
        def score(item: tuple[str, dict]) -> float:
            a = item[1].get("abundance", {}).get("A", {})
            attn = item[1].get("attn_mass_mean", {}).get("A", {})
            return max(abs(a.get("auc", 0.5) - 0.5), abs(attn.get("auc", 0.5) - 0.5))
        return [int(p) for p, _ in sorted(blocks.items(), key=score, reverse=True)]
    return (
        candidates.groupby("prototype")["attention"].mean()
        .sort_values(ascending=False).index.astype(int).tolist()
    )


def _diverse_picks(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Spread a montage's tiles across cohorts, specimen roles and both kinds.

    Round-robin over (cohort, role, kind) strata rather than "top n by
    attention", which would return six tiles from the same institution and let a
    reviewer describe a scanner rather than a morphology.
    """
    if pool.empty:
        return pool
    pool = pool.assign(_stratum=pool["cohort"].astype(str) + "|" + pool["role"] + "|" + pool["kind"])
    picks: list[pd.DataFrame] = []
    strata = sorted(pool["_stratum"].unique())
    per = max(n // max(len(strata), 1), 1)
    for stratum in strata:
        block = pool[pool["_stratum"].eq(stratum)]
        take = min(per, len(block))
        picks.append(block.sample(n=take, random_state=int(rng.integers(0, 2**31))))
    out = pd.concat(picks, ignore_index=True)
    # A tile can be BOTH a medoid and a high-attention tile, and `kind` is part
    # of the stratum key, so the same (slide, x, y) can be drawn twice. Showing
    # a reviewer the same image twice wastes a slot and makes one pattern look
    # more prevalent than it is, so dedupe on position and top up from what is
    # left rather than shrinking the montage.
    site = ["slide_id", "x", "y"]
    out = out.drop_duplicates(subset=site)
    if len(out) < n:
        seen = set(map(tuple, out[site].to_numpy()))
        spare = pool[~pool[site].apply(tuple, axis=1).isin(seen)]
        if not spare.empty:
            take = min(n - len(out), len(spare))
            out = pd.concat(
                [out, spare.sample(n=take, random_state=int(rng.integers(0, 2**31)))],
                ignore_index=True,
            )
    if len(out) > n:
        out = out.sample(n=n, random_state=int(rng.integers(0, 2**31)))
    return out.drop(columns=["_stratum"])


def _read_tile(image_module: Any, path: str, slide_id: str, x: int, y: int, size: int) -> Any:
    """Read one tile at its recorded level-0 coordinates and rescale to ``size``.

    Coordinates are LEVEL-0 pixels and the read window is ``patch_size_level0``,
    both taken from the patcher's own attributes - the slides span 0.25 and 0.5
    micron/pixel, so a fixed window would cut a different amount of tissue per
    cohort and the montage would compare magnifications, not morphology.
    """
    import openslide

    geometry = _patch_geometry(slide_id)
    if geometry["coordinate_units"] != "level0_pixels":
        raise ValueError(f"{slide_id}: unexpected coordinate units")
    with openslide.OpenSlide(path) as slide:
        region = slide.read_region(
            (int(x), int(y)), 0, (geometry["patch_size_level0"],) * 2
        ).convert("RGB")
    return region.resize((size, size), image_module.Resampling.LANCZOS)


def _grid(image_module: Any, images: list[Any], size: int) -> Any:
    columns = int(np.ceil(np.sqrt(len(images))))
    rows = int(np.ceil(len(images) / columns))
    canvas = image_module.new("RGB", (columns * size, rows * size), "white")
    for i, tile in enumerate(images):
        canvas.paste(tile, ((i % columns) * size, (i // columns) * size))
    return canvas


# ── plan / report ────────────────────────────────────────────────────────────
def cmd_plan(args: argparse.Namespace) -> None:
    print("E4 - morphologic phenotype and transport atlas (0 new fits)\n")
    print(f"  {'arm':20s} {'role':11s} {'slides':>7s} {'pts':>6s} {'mut':>5s}  scored by")
    for name, arm in ARMS.items():
        try:
            manifest = arm_manifest(arm)
        except (FileNotFoundError, SystemExit) as exc:
            print(f"  {name:20s} MANIFEST MISSING: {exc}")
            continue
        patients = manifest.drop_duplicates("patient_id")
        by = "E0 fold models (OOF)" if arm.mode == "oof" else f"{arm.target}-held-out refit"
        print(f"  {name:20s} {arm.role:11s} {len(manifest):7d} {len(patients):6d} "
              f"{int(patients['target_label'].sum()):5d}  {by}")
    dual = rih_dual_role_patients()
    print(f"\n  RIH dual-role patients: {len(dual)} - retained in the standalone panels, "
          "excluded from the contrast")
    dev = dev_patients()
    print(f"  set A {len(dev)} | set D {int(dev['in_D'].sum())} | "
          f"KRAS-WT label-complete {int(((dev.is_mutant == 0) & (dev.label_complete == 1)).sum())}")

    print("\n  ARTEFACT AUDIT")
    ok = True
    for seed in SEEDS:
        try:
            e0_fold_checkpoints(seed)
            print(f"    E0 seed{seed} fold checkpoints        OK")
        except SystemExit as exc:
            ok = False
            print(f"    E0 seed{seed} fold checkpoints        MISSING: {exc}")
    for target in ("RIH", "SurGen"):
        for seed in SEEDS:
            present = refit_checkpoint(target, seed).is_file()
            ok &= present
            print(f"    E2a {target} refit seed{seed} cap{E3B_CAP}   "
                  f"{'OK' if present else 'MISSING'}")
    for name in ARMS:
        have = [s for s in SEEDS if attention_path(name, s).is_file()]
        print(f"    attention {name:20s} seeds {have or 'none exported yet'}")
    v = vocab_path(args.k)
    print(f"    vocabulary k={args.k}                {'OK' if v.is_file() else 'not built'} ({v})")
    for level in ("slide", "patient"):
        p = profile_path(args.k, level)
        print(f"    {level} profiles                    {'OK' if p.is_file() else 'not built'}")
    print("\n  Models are complete." if ok else "\n  Missing model artefacts above must be built first.")
    print("  Order: attention -> vocabulary -> assign -> specificity -> transport -> "
          "montages -> report")


def cmd_report(args: argparse.Namespace) -> None:
    spec_path = paths.EVAL_ROOT / f"e3b_specificity_k{args.k}.json"
    trans_path = paths.EVAL_ROOT / f"e3b_transport_k{args.k}.json"
    profiles_path = profile_path(args.k, "patient")
    if not spec_path.is_file():
        raise SystemExit(f"{spec_path} missing - run `specificity` first")
    if not trans_path.is_file():
        raise SystemExit(f"{trans_path} missing - run `transport` first")
    if not profiles_path.is_file():
        raise SystemExit(f"{profiles_path} missing - run `assign` first")
    profiles = pd.read_parquet(profiles_path)
    missing_arms = set(ARMS) - set(profiles["arm"].unique())
    if missing_arms:
        raise SystemExit(f"patient profiles are missing arm(s): {sorted(missing_arms)}")
    attention_columns = [f"attn_mass_seed{seed}" for seed in SEEDS]
    missing_attention = set(attention_columns) - set(profiles.columns)
    if missing_attention or profiles[attention_columns].isna().any().any():
        raise SystemExit(
            f"patient profiles must contain complete attention for all {len(SEEDS)} seeds"
        )
    if profiles[list(QUANTITIES)].isna().any().any():
        raise SystemExit("patient profiles contain missing abundance or attention values")
    spec = json.loads(spec_path.read_text())
    transport = json.loads(trans_path.read_text())
    missing_cohorts = {name for name, *_ in TRANSPORT_PAIRS} - set(
        transport.get("conservation", {})
    )
    if missing_cohorts:
        raise SystemExit(
            "transport artifact is incomplete; missing conservation arm(s): "
            + ", ".join(sorted(missing_cohorts))
        )
    readout = final_readout(spec, transport)

    groups: dict[str, list[int]] = {g: [] for g in GROUPS}
    for prototype, row in readout.items():
        groups[row["group"]].append(prototype)

    print(f"\n{'=' * 112}\nE4 - KRAS MORPHOLOGIC PHENOTYPE AND TRANSPORT ATLAS  "
          f"(k={args.k}, A n={spec['populations']['A']}, D n={spec['populations']['D']})"
          f"\n{'=' * 112}")

    # ── the per-prototype table (design step 9) ──────────────────────────────
    print("\n  PER-PROTOTYPE SUMMARY")
    print(f"  {'proto':>5s} {'KRAS association':>26s} {'in D?':>7s} {'P->M':>13s} "
          f"{'shortcut':>34s}  pathology")
    print(f"  {'-' * 5} {'-' * 26} {'-' * 7} {'-' * 13} {'-' * 34}  {'-' * 9}")
    base_review = _load_review(args.k)
    if base_review["status"] != "complete":
        raise SystemExit(
            f"blinded pathology review is {base_review['status']} "
            f"({base_review['n_reviewed']}/{base_review['n_expected']} montages); "
            "complete the review form before finalizing E4"
        )
    review_selection = review_prototypes(readout)
    base_strata = ("claimed", "suppressed", "technical")
    expected_base = {
        prototype
        for stratum in base_strata
        for prototype in review_selection[stratum]
    }
    base_reviewed = set(base_review["annotations"])
    if base_reviewed != expected_base:
        raise SystemExit(
            "completed base review does not match its locked readout-driven selection: "
            f"missing={sorted(expected_base - base_reviewed)}, "
            f"unexpected={sorted(base_reviewed - expected_base)}"
        )

    followup = _load_completed_followup(args.k)
    if followup["status"] != "complete":
        raise SystemExit(
            "the attention-only pathology follow-up is pending; complete and curate "
            "the M11 review before finalizing E4"
        )
    followup_reviewed = set(followup["annotations"])
    expected_review = {
        prototype for prototypes in review_selection.values() for prototype in prototypes
    }
    reviewed = base_reviewed | followup_reviewed
    if reviewed != expected_review:
        raise SystemExit(
            "combined completed review does not match the current readout selection: "
            f"missing={sorted(expected_review - reviewed)}, "
            f"unexpected={sorted(reviewed - expected_review)}"
        )
    missing_attention_followup = (
        set(review_selection["attention_followup"]) - followup_reviewed
    )
    if missing_attention_followup:
        raise SystemExit(
            "attention-only prototype(s) lack completed follow-up review: "
            f"{sorted(missing_attention_followup)}"
        )

    combined_annotations = dict(base_review["annotations"])
    combined_annotations.update(followup["annotations"])
    montage_by_prototype = dict(base_review.get("montage_by_prototype", {}))
    montage_by_prototype.update(followup["montage_by_prototype"])
    review_stage: dict[int, str] = {}
    attention_followup = set(review_selection["attention_followup"])
    for prototype in combined_annotations:
        if prototype in attention_followup:
            review_stage[prototype] = "attention_followup"
        elif prototype in followup_reviewed:
            review_stage[prototype] = "targeted_followup"
        else:
            review_stage[prototype] = "base"
    followup_meta = {
        key: value
        for key, value in followup.items()
        if key not in ("annotations", "montage_by_prototype")
    }
    review = {
        "schema_version": 2,
        "status": "complete",
        "n_expected": len(expected_review),
        "n_reviewed": len(reviewed),
        # Kept at top level for backward compatibility with the verifier.
        "form_sha256": base_review["form_sha256"],
        "key_sha256": base_review["key_sha256"],
        "base_packet": {
            "status": base_review["status"],
            "n_expected": base_review["n_expected"],
            "n_reviewed": base_review["n_reviewed"],
            "form_sha256": base_review["form_sha256"],
            "key_sha256": base_review["key_sha256"],
        },
        "followup": followup_meta,
        "selection": review_selection,
        "annotations": combined_annotations,
        "base_annotations": base_review["annotations"],
        "montage_by_prototype": montage_by_prototype,
        "review_stage_by_prototype": review_stage,
    }
    annotations = {
        prototype: _format_annotation(fields)
        for prototype, fields in review["annotations"].items()
    }
    for prototype, row in readout.items():
        fields = review["annotations"].get(prototype)
        row["pathology_reviewed"] = fields is not None
        row["montage_id"] = montage_by_prototype.get(prototype)
        row["pathology_review_stage"] = review_stage.get(prototype)
        row["pathology"] = fields
        row["pathology_flags"] = {
            column: fields[column]
            for column in (
                "normal_organ_tissue",
                "normal_non_tumour",
                "artifact",
                "artifact_concern",
            )
            if fields and fields.get(column)
        }
    for prototype in sorted(readout):
        row = readout[prototype]
        assoc = row["kras_association"]
        if assoc["significant"]:
            arrow = "^" if assoc["auc_A"] > 0.5 else "v"
            association = f"{arrow} {assoc['auc_A']:.3f} (q={assoc['q_A']:.3f})"
        else:
            association = f"- {assoc['auc_A']:.3f}" if assoc["auc_A"] is not None else "-"
        persists = "-" if row["persists_in_D"] is None else ("yes" if row["persists_in_D"] else "no")
        conserved = {"conserved": "conserved", "changed": "changed",
                     "heterogeneous": "heterogeneous", "inconclusive": "inconclusive",
                     "underpowered": "underpowered", "not_evaluated": "-"}[
            row["conservation"]["state"]]
        marks = list(row["shortcut_flags"])
        marks += [f"[met-only] {f}" for f in row.get("organ_flags", [])]
        shortcut = "; ".join(marks) or "-"
        print(f"  {prototype:5d} {association:>26s} {persists:>7s} {conserved:>13s} "
              f"{shortcut[:34]:>34s}  {annotations.get(prototype, 'not selected')[:60]}")

    # ── outcome groups (design step 10) ──────────────────────────────────────
    print("\n  MAIN READOUT - outcome groups")
    labels = {
        "dependency_robust_conserved":
            "dependency-robust AND conserved   - best candidate KRAS morphology",
        "dependency_robust_changed":
            "dependency-robust, directly changed across specimen roles",
        "dependency_robust_transport_heterogeneous":
            "dependency-robust, transport heterogeneous across cohorts",
        "dependency_robust_transport_inconclusive":
            "dependency-robust, transport inconclusive - conservation criterion missed",
        "dependency_robust_transport_underpowered":
            "dependency-robust, transport underpowered - no primary effect to test",
        "dependency_robust_transport_not_evaluated":
            "dependency-robust, transport not evaluated - incomplete analysis",
        "context_dependent":
            "context-dependent                 - lost after MSI/BRAF restriction",
        "restriction_sensitive_unexplained":
            "restriction-sensitive, not explained by measured MSI/BRAF context",
        "shortcut_technical":
            "source-confounded / technical     - organ, cohort, scanner or artefact",
        "not_kras_associated":
            "no KRAS association               - not claimed",
    }
    for group in GROUPS:
        print(f"    {labels[group]}  {sorted(groups[group]) or '-'}")

    claimed = sum(len(groups[g]) for g in CLAIMED_GROUPS)
    print(f"\n    {claimed} of {len(readout)} prototypes carry a claim; "
          f"{len(groups['shortcut_technical'])} are source-confounded.")

    # ── transport detail, kept beside the readout ────────────────────────────
    print("\n  TRANSPORT DETAIL (abundance effect, dual-role patients excluded)")
    for cohort, block in transport.get("conservation", {}).items():
        conserved = [r["prototype"] for r in block["by_quantity"]["abundance"] if r["conserved"]]
        print(f"    {cohort:10s} n_P={block['n_primary']:3d} n_M={block['n_metastatic']:3d}  "
              f"conserved {conserved or '-'}")
    print(f"\n  LABEL-MASKED PATHOLOGY REVIEW: {review['status']} "
          f"({review['n_reviewed']}/{review['n_expected']} unique montages; "
          f"{followup['n_assessments']} follow-up assessments)")

    print("\n  CLAIM LIMITS carried into the manuscript:")
    print("    - attention is region prioritisation, not causal explanation;")
    print("    - abundance and attention mass are separate claims;")
    print("    - surviving set D is robustness to the MEASURED dependencies only;")
    print("    - underpowered or inconclusive transport is not evidence of change;")
    print("    - a shortcut flag suppresses the headline, not the underlying statistic.")

    table_rows: list[dict[str, Any]] = []
    for prototype, row in sorted(readout.items()):
        abundance = row["specificity_effects"]["abundance"]
        attention_effect = row["specificity_effects"]["attn_mass_mean"]["A"]
        attention_effect_d = row["specificity_effects"]["attn_mass_mean"]["D"]
        a_effect = abundance["A"]
        d_effect = abundance["D"]
        context_effect = abundance["context_in_wt"]
        table_rows.append({
            "prototype": prototype,
            "montage_id": row["montage_id"],
            "group": row["group"],
            "specificity_bucket": row["specificity_bucket"],
            "auc_A": a_effect.get("auc"),
            "ci_low_A": a_effect.get("ci_low"),
            "ci_high_A": a_effect.get("ci_high"),
            "q_A": a_effect.get("q"),
            "kras_associated_A": a_effect.get("significant"),
            "auc_D": d_effect.get("auc"),
            "ci_low_D": d_effect.get("ci_low"),
            "ci_high_D": d_effect.get("ci_high"),
            "q_D": d_effect.get("q"),
            "kras_associated_D": d_effect.get("significant"),
            "context_auc_in_wt": context_effect.get("auc"),
            "context_q_in_wt": context_effect.get("q"),
            "context_associated_in_wt": context_effect.get("significant"),
            "classification_flags": "; ".join(row["classification_flags"]),
            "direction": row["kras_association"]["direction"],
            "attention_auc_A": attention_effect.get("auc"),
            "attention_ci_low_A": attention_effect.get("ci_low"),
            "attention_ci_high_A": attention_effect.get("ci_high"),
            "attention_q_A": attention_effect.get("q"),
            "attention_associated_A": attention_effect.get("significant"),
            "attention_direction": row["attention_association"]["direction"],
            "attention_auc_D": attention_effect_d.get("auc"),
            "attention_ci_low_D": attention_effect_d.get("ci_low"),
            "attention_ci_high_D": attention_effect_d.get("ci_high"),
            "attention_q_D": attention_effect_d.get("q"),
            "attention_associated_D": attention_effect_d.get("significant"),
            "attention_cohort_directions_agreeing": (
                row["attention_cohort_reproducibility"].get("agreeing")
            ),
            "attention_cohorts_total": (
                row["attention_cohort_reproducibility"].get("n_cohorts")
            ),
            "attention_cohort_direction": (
                row["attention_cohort_reproducibility"].get("direction")
            ),
            "persists_in_D": row["persists_in_D"],
            "conservation": row["conservation"]["state"],
            "transport_by_cohort": json.dumps(row["conservation"]["by_cohort"]),
            "shortcut_flags": "; ".join(row["shortcut_flags"]),
            "organ_flags_metastatic_only": "; ".join(row.get("organ_flags", [])),
            "pathology_reviewed": row["pathology_reviewed"],
            "pathology_review_stage": row["pathology_review_stage"],
            "canonical_description": (
                row["pathology"].get("canonical_description", "")
                if row["pathology"] else ""
            ),
            "artifact_concern": (
                row["pathology"].get("artifact_concern", "")
                if row["pathology"] else ""
            ),
            "biological_interpretability": (
                row["pathology"].get("biological_interpretability", "")
                if row["pathology"] else ""
            ),
            "pathology_confidence": (
                row["pathology"].get("confidence", "")
                if row["pathology"] else ""
            ),
            "pathology_quality_flags": json.dumps(
                row["pathology"].get("quality_flags", [])
                if row["pathology"] else []
            ),
            "pathology_flags": json.dumps(row["pathology_flags"]),
            "pathology": annotations.get(prototype, ""),
        })
    table = pd.DataFrame(table_rows)
    csv_dest = paths.EVAL_ROOT / f"e3b_atlas_k{args.k}_prototypes.csv"
    table.to_csv(csv_dest, index=False)

    dest = paths.EVAL_ROOT / f"e3b_atlas_k{args.k}.json"
    dest.write_text(json.dumps(
        {"k": args.k, "groups": groups, "readout": readout,
         "specificity": str(spec_path), "transport": str(trans_path),
         "prototype_table": str(csv_dest), "pathology_review": review},
        indent=2, default=str,
    ))
    print(f"\nWrote {dest}")
    print(f"Wrote {csv_dest}")


def _review_form_has_content(form: Path) -> bool:
    """Whether a review form contains any human-entered annotation."""
    filled = pd.read_csv(form, dtype=str, keep_default_na=False)
    described = [c for c in filled.columns if c not in ("montage_id", "n_tiles")]
    return any(str(value).strip() for value in filled[described].to_numpy().ravel())


def _format_annotation(fields: dict[str, Any]) -> str:
    """Losslessly render one structured review row for the flat CSV table."""
    if fields.get("canonical_description"):
        return str(fields["canonical_description"])
    return "; ".join(f"{column}: {value}" for column, value in fields.items())


def _load_review(k: int) -> dict[str, Any]:
    """Validate and unblind a completed morphology-review packet.

    A fully blank generated form is a legitimate pre-review state. Once any
    annotation is present, however, every montage must be described and the
    form/key relationship must be one-to-one. A partial or malformed form must
    not silently become a partly reviewed final report.
    """
    form = REVIEW_ROOT / f"k{k}" / "review_form.csv"
    key = MONTAGE_DIR / f"k{k}" / "KEY_do_not_open_before_review.csv"
    if not (form.is_file() and key.is_file()):
        return {"status": "pending", "n_expected": 0, "n_reviewed": 0,
                "annotations": {}}

    filled = pd.read_csv(form, dtype=str, keep_default_na=False)
    keyed = pd.read_csv(key, dtype={"montage_id": str, "prototype": int})
    required = {"montage_id", "n_tiles"}
    if missing := required - set(filled.columns):
        raise SystemExit(f"review form is missing required column(s): {sorted(missing)}")
    described = [c for c in filled.columns if c not in required]
    if not described:
        raise SystemExit("review form has no morphology-description columns")
    if filled["montage_id"].duplicated().any():
        raise SystemExit("review form contains duplicate montage_id rows")

    prototype_counts = keyed.groupby("montage_id")["prototype"].nunique()
    if (prototype_counts != 1).any():
        bad = prototype_counts[prototype_counts != 1].index.tolist()
        raise SystemExit(f"unblinding key maps montage(s) to multiple prototypes: {bad}")
    key_ids = set(prototype_counts.index)
    form_ids = set(filled["montage_id"])
    if key_ids != form_ids:
        raise SystemExit(
            "review form/key montage mismatch: "
            f"missing={sorted(key_ids - form_ids)}, unexpected={sorted(form_ids - key_ids)}"
        )
    montage_prototypes = keyed[["montage_id", "prototype"]].drop_duplicates()
    if montage_prototypes["prototype"].duplicated().any():
        duplicates = sorted(
            montage_prototypes.loc[
                montage_prototypes["prototype"].duplicated(keep=False), "prototype"
            ].unique().tolist()
        )
        raise SystemExit(f"unblinding key repeats prototype(s) across montages: {duplicates}")

    keyed_tiles = keyed.groupby("montage_id").size().to_dict()
    declared_tiles = pd.to_numeric(filled.set_index("montage_id")["n_tiles"], errors="coerce")
    bad_counts = [
        montage_id for montage_id, n_tiles in keyed_tiles.items()
        if not np.isfinite(declared_tiles.get(montage_id, np.nan))
        or int(declared_tiles[montage_id]) != int(n_tiles)
    ]
    if bad_counts:
        raise SystemExit(f"review form/key tile-count mismatch for: {sorted(bad_counts)}")

    rows: dict[str, dict[str, str]] = {}
    for record in filled.to_dict("records"):
        rows[record["montage_id"]] = {
            column: str(record[column]).strip()
            for column in described if str(record[column]).strip()
        }
    n_reviewed = sum(bool(fields) for fields in rows.values())
    if n_reviewed == 0:
        return {"status": "pending", "n_expected": len(rows), "n_reviewed": 0,
                "annotations": {}}
    if n_reviewed != len(rows):
        blank = sorted(montage_id for montage_id, fields in rows.items() if not fields)
        raise SystemExit(f"review form is only partially completed; blank montage(s): {blank}")

    mapping = montage_prototypes.set_index("montage_id")["prototype"].to_dict()
    annotations = {int(mapping[montage_id]): fields for montage_id, fields in rows.items()}
    montage_by_prototype = {
        int(prototype): str(montage_id) for montage_id, prototype in mapping.items()
    }
    return {"status": "complete", "n_expected": len(rows),
            "n_reviewed": n_reviewed,
            "form_sha256": hashlib.sha256(form.read_bytes()).hexdigest(),
            "key_sha256": hashlib.sha256(key.read_bytes()).hexdigest(),
            "annotations": annotations,
            "montage_by_prototype": montage_by_prototype}


def _load_completed_followup(k: int) -> dict[str, Any]:
    """Validate the curated extraction of the immutable completed follow-up.

    The submitted Markdown is retained byte-for-byte because several tile-level
    answers conflict. The structured JSON contains only montage-level conclusions
    that are supported across the response, plus explicit quality flags. It uses
    blinded M identifiers only; prototype IDs enter only here, through keys kept
    outside the reviewer packet.
    """
    packet = REVIEW_ROOT / f"k{k}"
    structured_path = packet / "completed_review_structured.json"
    if not structured_path.is_file():
        return {"status": "pending", "n_assessments": 0, "annotations": {}}

    structured = json.loads(structured_path.read_text())
    if int(structured.get("schema_version", -1)) != 1:
        raise SystemExit("completed follow-up has an unsupported schema_version")
    extraction_status = str(structured.get("extraction_status", "")).strip()
    if not extraction_status:
        raise SystemExit("completed follow-up lacks extraction_status")
    source_name = str(structured.get("source_document", ""))
    if not source_name or Path(source_name).name != source_name:
        raise SystemExit("completed follow-up source_document must be a local file name")
    source_path = packet / source_name
    if not source_path.is_file():
        raise SystemExit(f"completed follow-up source is missing: {source_path}")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha != structured.get("source_sha256"):
        raise SystemExit("completed follow-up source checksum does not match extraction")

    provenance = structured.get("provenance", {})
    base_form = packet / "review_form.csv"
    key_dir = MONTAGE_DIR / f"k{k}"
    base_key = key_dir / "KEY_do_not_open_before_review.csv"
    addendum_name = str(provenance.get("addendum_key", ""))
    if not addendum_name or Path(addendum_name).name != addendum_name:
        raise SystemExit("completed follow-up addendum_key must be a local file name")
    addendum_key = key_dir / addendum_name
    required_files = (base_form, base_key, addendum_key)
    if missing := [str(path) for path in required_files if not path.is_file()]:
        raise SystemExit(f"completed follow-up provenance file(s) missing: {missing}")
    expected_hashes = {
        base_form: provenance.get("base_form_sha256"),
        base_key: provenance.get("base_key_sha256"),
        addendum_key: provenance.get("addendum_key_sha256"),
    }
    for path, expected in expected_hashes.items():
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if not expected or observed != expected:
            raise SystemExit(f"completed follow-up provenance checksum mismatch: {path}")

    base_key_rows = pd.read_csv(
        base_key, dtype={"montage_id": str, "prototype": int}
    )
    addendum_key_rows = pd.read_csv(
        addendum_key, dtype={"montage_id": str, "prototype": int}
    )
    overlapping_ids = sorted(
        set(base_key_rows["montage_id"]) & set(addendum_key_rows["montage_id"])
    )
    if overlapping_ids:
        raise SystemExit(
            "base and addendum keys reuse montage ID(s): "
            f"{overlapping_ids}"
        )
    if "selection_stratum" not in addendum_key_rows.columns or set(
        addendum_key_rows["selection_stratum"].dropna().astype(str).str.strip()
    ) != {"attention_followup"}:
        raise SystemExit(
            "completed follow-up addendum key must contain only the "
            "attention_followup selection stratum"
        )
    keyed = pd.concat([base_key_rows, addendum_key_rows], ignore_index=True)
    counts = keyed.groupby("montage_id")["prototype"].nunique()
    if (counts != 1).any():
        bad = counts[counts != 1].index.tolist()
        raise SystemExit(f"follow-up key maps montage(s) to multiple prototypes: {bad}")
    montage_map = (
        keyed[["montage_id", "prototype"]]
        .drop_duplicates()
        .set_index("montage_id")["prototype"]
        .to_dict()
    )

    montages = structured.get("montages", {})
    if not isinstance(montages, dict) or not montages:
        raise SystemExit("completed follow-up contains no montage assessments")
    unknown = sorted(set(montages) - set(montage_map))
    if unknown:
        raise SystemExit(f"completed follow-up has unmapped montage(s): {unknown}")
    followup_prototypes = [int(montage_map[montage_id]) for montage_id in montages]
    if len(followup_prototypes) != len(set(followup_prototypes)):
        raise SystemExit("completed follow-up maps multiple montages to one prototype")

    expected_images = provenance.get("montage_sha256", {})
    tile_counts = keyed.groupby("montage_id").size().to_dict()
    annotations: dict[int, dict[str, Any]] = {}
    montage_by_prototype: dict[int, str] = {}
    image_hashes: dict[str, str] = {}
    for montage_id, fields in montages.items():
        if not isinstance(fields, dict) or not str(
            fields.get("canonical_description", "")
        ).strip():
            raise SystemExit(
                f"completed follow-up {montage_id} lacks canonical_description"
            )
        declared = fields.get("n_tiles")
        if int(declared) != int(tile_counts.get(montage_id, -1)):
            raise SystemExit(f"completed follow-up tile-count mismatch: {montage_id}")
        image_path = packet / "montages" / f"{montage_id}.jpg"
        if not image_path.is_file():
            raise SystemExit(f"completed follow-up montage is missing: {image_path}")
        image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if image_sha != expected_images.get(montage_id):
            raise SystemExit(f"completed follow-up montage checksum mismatch: {montage_id}")
        prototype = int(montage_map[montage_id])
        annotations[prototype] = dict(fields)
        montage_by_prototype[prototype] = montage_id
        image_hashes[montage_id] = image_sha

    return {
        "status": "complete",
        "schema_version": int(structured["schema_version"]),
        "extraction_status": extraction_status,
        "n_assessments": len(annotations),
        "source_document": source_name,
        "source_sha256": source_sha,
        "structured_sha256": hashlib.sha256(structured_path.read_bytes()).hexdigest(),
        "base_form_sha256": expected_hashes[base_form],
        "base_key_sha256": expected_hashes[base_key],
        "addendum_key": addendum_name,
        "addendum_key_sha256": expected_hashes[addendum_key],
        "montage_sha256": image_hashes,
        "reviewer_metadata": structured.get("reviewer_metadata", {}),
        "cross_montage": structured.get("cross_montage", {}),
        "quality_flags": structured.get("global_quality_flags", []),
        "annotations": annotations,
        "montage_by_prototype": montage_by_prototype,
    }


def _load_annotations(k: int) -> dict[int, str]:
    """Backward-compatible flat view of the validated structured review."""
    review = _load_review(k)
    return {
        prototype: _format_annotation(fields)
        for prototype, fields in review["annotations"].items()
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--k", type=int, default=atlas.N_PROTOTYPES,
                   help="prototype vocabulary size (24-40 band)")
    sub = p.add_subparsers(dest="command", required=True)

    x = sub.add_parser("plan")
    x.set_defaults(func=cmd_plan)

    x = sub.add_parser("attention")
    x.add_argument("--arm", choices=list(ARMS))
    x.add_argument("--seed", type=int, choices=list(SEEDS))
    x.add_argument("--top-fraction", type=float, default=0.10)
    x.add_argument("--limit", type=int, help="first N slides only, for a smoke test")
    x.add_argument("--force", action="store_true")
    x.add_argument("--dry-run", action="store_true")
    x.set_defaults(func=cmd_attention)

    x = sub.add_parser("vocabulary")
    x.add_argument("--tiles", type=int, default=atlas.TOTAL_SAMPLE_TILES)
    x.add_argument("--components", type=int, default=atlas.N_COMPONENTS)
    x.add_argument("--normalize", choices=("l2", "none"), default="l2")
    x.add_argument("--seed", type=int, default=atlas.VOCAB_SEED)
    x.add_argument("--force", action="store_true")
    x.add_argument("--dry-run", action="store_true")
    x.set_defaults(func=cmd_vocabulary)

    x = sub.add_parser("assign")
    x.add_argument("--arm", choices=list(ARMS))
    x.add_argument("--limit", type=int)
    x.add_argument("--seeds", help="comma-separated subset of 42,43,44 (default: all three)")
    x.add_argument("--allow-missing-attention", action="store_true",
                   help="profile abundance for slides with no exported attention "
                        "(their attention mass is NaN, not zero)")
    x.set_defaults(func=cmd_assign)

    x = sub.add_parser("specificity")
    x.add_argument("--n-bootstrap", type=int, default=2000)
    x.add_argument("--seed", type=int, default=atlas.VOCAB_SEED)
    x.set_defaults(func=cmd_specificity)

    x = sub.add_parser("transport")
    x.add_argument("--n-bootstrap", type=int, default=2000)
    x.add_argument("--seed", type=int, default=atlas.VOCAB_SEED)
    x.set_defaults(func=cmd_transport)

    x = sub.add_parser("montages")
    # 0 = readout-driven: every prototype the final table makes a claim about,
    # plus every shortcut suspect. A positive --top forces the old fixed-N
    # ranking instead, which is only correct before `specificity` has run.
    x.add_argument("--top", type=int, default=0)
    x.add_argument("--technical", type=int, default=6,
                   help="how many source-dominated prototypes to include for "
                        "characterisation (readout-driven selection only)")
    x.add_argument("--tiles-per-montage", type=int, default=12)
    x.add_argument("--tile-px", type=int, default=256)
    x.add_argument("--slide-root", default="/mnt/d/YC.Liu/slides/colon")
    x.add_argument("--seed", type=int, default=atlas.VOCAB_SEED)
    x.add_argument("--force", action="store_true",
                   help="replace an existing completed review packet")
    x.set_defaults(func=cmd_montages)

    x = sub.add_parser(
        "montage-addendum",
        help="generate one isolated blinded follow-up montage",
    )
    x.add_argument("--prototype", type=int, required=True,
                   help="internal prototype ID; never written into the review packet")
    x.add_argument("--montage-id", required=True,
                   help="new blinded reader-facing ID, for example M11")
    x.add_argument("--technical", type=int, default=6,
                   help="technical-sample size used to reproduce review selection")
    x.add_argument("--tiles-per-montage", type=int, default=12)
    x.add_argument("--tile-px", type=int, default=256)
    x.add_argument("--slide-root", default="/mnt/d/YC.Liu/slides/colon")
    x.add_argument("--seed", type=int, default=atlas.VOCAB_SEED)
    x.add_argument("--force", action="store_true",
                   help="replace only this addendum's image and separate key")
    x.set_defaults(func=cmd_montage_addendum)

    x = sub.add_parser("report")
    x.set_defaults(func=cmd_report)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
