#!/usr/bin/env python3
"""E2c - few-shot local adaptation of the CLASSIFIER HEAD (Aim 2, 0 new MIL fits).

THE QUESTION, REVISED 2026-08-19. Can a small number of local labelled cases
recover the metastatic DISCRIMINATION loss E2b measured - and do those cases
need to be metastatic?

WHY THIS IS NOT A RECALIBRATION EXPERIMENT ANY MORE. The first E2c design asked
whether few local labels could repair target PROBABILITIES. E2a answered most of
that for free: a source-only calibrator, fitted on source out-of-fold
predictions with NO target label, already cuts target log loss by 20-51 % and
pulls every calibration intercept to within 0.3 of zero. And E2b showed the
primary -> metastatic loss is NOT calibration: calibrated Brier moves +0.018
(RIH) and +0.001 (SurGen) while AUROC falls 0.108 and 0.050. Recalibration
provably cannot restore ranking, so an experiment built on it cannot answer the
question that is actually open. Calibration is retained here as a SECONDARY
CONTROL - reported, never the endpoint.

WHAT IS ADAPTED, AND WHAT IS NOT.

    UNIv1 (frozen) -> ABMIL patch projection + gated attention (FROZEN)
                   -> slide embedding, 512-d
                   -> classifier head  <-- THE ONLY THING THAT MOVES

The encoder is not touched, the attention is not touched, and no MIL model is
retrained: the 12 frozen E2a refits produce the embeddings once, and every arm
below is a 513-parameter logistic fit on top of them. Adapting attention or the
encoder on 4-16 examples is a different, much stronger claim and is deliberately
not attempted first.

SHRINKAGE IS WHAT MAKES IT A FEW-SHOT METHOD. 513 free parameters on 4-16
observations is 32-128 parameters per observation, which is not a credible
few-shot mechanism. The head is therefore fitted with an L2-SP penalty toward
the SOURCE head (Li et al., explicit inductive bias for transfer learning):

    min_w,b  (1/n) sum_i BCE(y_i, sigmoid(w.h_i + b))
             + (lambda/2) * ( ||w - w_src||^2 + (b - b_src)^2 )

As lambda -> infinity the arm collapses EXACTLY to S0, so the learning curve is
interpretable by construction: any gain is a measured departure from the
transported head, not a free refit. Lambda is selected by leave-one-out INSIDE
THE SUPPORT SET ONLY - never on the held-out fold - and the selection is logged.

PATIENT-LEVEL AGGREGATION IS ALGEBRAICALLY EXACT. The head is linear, so
for a patient with slides h_1..h_k

    mean_i (w.h_i + b) = w.(mean_i h_i) + b

The patient mean EMBEDDING therefore defines the adapted linear predictor
without approximation. The frozen S0 and every lambda=infinity prediction use
the stored native checkpoint logits, rather than an fp32 reconstruction of a
head originally evaluated under bf16. `embeddings --verify` asserts that the
native logits exactly reproduce E2a/E2b and that the reconstructed AUROC agrees
to tolerance.

THE THREE CONDITIONS. One question, three arms, matched label budgets:

    S0  zero-shot           no target label at all; the E2b baseline
    S1  local PRIMARY       k/class target-cohort PRIMARY patients
    S2  local METASTATIC    k/class target-cohort METASTATIC patients

    RIH:     S1 from RIH-P      -> evaluate RIH-M
    SurGen:  S1 from SR1482-P   -> evaluate SurGen-M (= SR1482-M)

SR1482-P, not SR386+SR1482: E2b's comparator is SR1482-P vs SR1482-M, and
changing the primary pool between the two experiments would confound the
specimen-role question with a case-mix change.

BUDGETS: 2, 4, 8 per class (4, 8, 16 total). 16/class is deliberately absent -
it is the fallback if everything is noisy, not part of the design.

THE ENDPOINT IS DISCRIMINATION.

    primary:   AUROC(S2) - AUROC(S1)   at matched budget - do the cases need to
                                        be metastatic?
    secondary: AUROC(arm) - AUROC(S0)  - does either kind of case help at all?
    control:   AUPRC, Brier, calibration intercept/slope - reported so a new
               calibration problem introduced by head adaptation would be seen.

ROTATING FOLDS, BECAUSE THE TARGETS ARE SMALL. RIH-M has 85 patients and
SurGen-M has 74. Permanently reserving support cases would delete a fifth of the
evaluation set, so support is drawn from OUTSIDE a rotating test fold:

    5 patient-grouped folds over the target METASTATIC patients, stratified on
    KRAS. For each fold: support drawn from the other 4 folds (S2) or from the
    target primary pool (S1) -> adapt -> evaluate on the untouched fold.
    R repetitions of the complete cross-fitted support procedure. Every
    metastatic patient is evaluated once per repetition. The headline is the
    MEAN OF THE R REPETITION METRICS. Predictions are never averaged across
    repetitions: that would create a 5R-head ensemble whose union had seen far
    more than k labels per class and could not be deployed at the stated budget.

All slides of a patient stay together. A patient's PRIMARY is removed from the
S1 support pool whenever that patient's METASTASIS is in the test fold - the 8
RIH dual-role patients, which would otherwise leak the test label into support.

INFERENCE HAS TWO RANDOM COMPONENTS. Expected procedural performance and all
contrasts use a two-way bootstrap: one shared patient resample across arms plus
resampling of the support/training repetitions. S1 and S2 repetitions are
resampled independently because they draw from different support populations.
The final default is 100 repetitions; 20 is too sparse to characterize the
tails of support-draw variability.
Support-draw percentiles are reported as procedural variability, not confidence
intervals. Full repetition-by-patient predictions and support IDs are retained
in the immutable result artifact for audit.

Usage:
    python aim2_head_adaptation_base.py embeddings --cap 8192 [--verify]
    python aim2_head_adaptation_base.py verify     --cap 8192
    python aim2_head_adaptation_base.py run        --cap 8192 [--reps 100]
    python aim2_head_adaptation_base.py report     --cap 8192
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_loco_transport  # noqa: E402
from oceanpath.aim1 import lineage, paths  # noqa: E402
from oceanpath.eval.external import sigmoid  # noqa: E402

# (target cohort, primary support pool, primary subcohort filter or None)
COHORTS: tuple[tuple[str, str | None], ...] = (("RIH", None), ("SurGen", "SR1482"))
BUDGETS: tuple[int, ...] = (2, 4, 8)  # per class
ARMS: tuple[str, ...] = ("S0", "S1", "S2")
N_FOLDS = 5
DEFAULT_REPS = 100
DEFAULT_N_BOOTSTRAP = 10_000
# lambda = inf IS a grid point, and it is the first one, so ties go to it.
# The docstring's guarantee — "as lambda -> infinity the arm collapses EXACTLY
# to S0" — is only true if infinity is reachable. Without it the method cannot
# decline to adapt: it is forced to move the head even when the support set says
# it should not, and the learning curve stops being interpretable as a departure
# from the transported model. Declining is the correct answer when 4 labels
# carry no usable information.
LAMBDA_GRID = (np.inf, 1e4, 3e3, 1e3, 3e2, 1e2, 3e1, 1e1, 3.0, 1.0)
EPS = 1e-6


def emb_path(target: str, kind: str, cap: int) -> Path:
    return (
        lineage.component_root("e2c") / "embeddings" / f"cap{cap}_{target.lower()}_{kind}.parquet"
    )


def head_path(target: str, cap: int) -> Path:
    return lineage.component_root("e2c") / "heads" / f"cap{cap}_{target.lower()}.npz"


def emb_receipt_path(target: str, kind: str, cap: int) -> Path:
    return emb_path(target, kind, cap).with_suffix(".receipt.json")


def head_receipt_path(target: str, cap: int) -> Path:
    return head_path(target, cap).with_suffix(".receipt.json")


def result_path(cap: int) -> Path:
    return lineage.component_root("e2c") / "analysis" / f"e2c_true_kshot_cap{cap}.json"


# ── embeddings ───────────────────────────────────────────────────────────────
def _embedding_inputs(target: str, kind: str, cap: int) -> dict:
    manifest = aim2_loco_transport.target_manifest(target, kind)
    score_contracts = {
        str(seed): aim2_loco_transport._score_inputs(target, seed, kind, cap)  # noqa: SLF001
        for seed in aim2_loco_transport.SEEDS
    }
    feature_identities = {
        json.dumps(contract["feature_store"], sort_keys=True)
        for contract in score_contracts.values()
    }
    if len(feature_identities) != 1:
        raise RuntimeError(f"{target}/{kind}: feature identity changed during preflight")
    return {
        "lineage": lineage.lineage_name(),
        "code": lineage.artifact_identity(Path(__file__)),
        "manifest": lineage.artifact_identity(manifest),
        "feature_store": next(iter(score_contracts.values()))["feature_store"],
        "checkpoints": {
            seed: contract["checkpoint"] for seed, contract in score_contracts.items()
        },
    }


def _head_inputs(target: str, cap: int) -> dict:
    return {
        "lineage": lineage.lineage_name(),
        "code": lineage.artifact_identity(Path(__file__)),
        "checkpoints": {
            str(seed): lineage.artifact_identity(aim2_loco_transport.model_ckpt(target, seed, cap))
            for seed in aim2_loco_transport.SEEDS
        },
        "completion_receipts": {
            str(seed): lineage.artifact_identity(
                aim2_loco_transport.fit_summary_path(target, seed, cap)
            )
            for seed in aim2_loco_transport.SEEDS
        },
    }


def _validated_embedding(target: str, kind: str, cap: int) -> pd.DataFrame | None:
    artifact = emb_path(target, kind, cap)
    receipt_path = emb_receipt_path(target, kind, cap)
    if not artifact.exists() and not receipt_path.exists():
        return None
    if not artifact.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Partial immutable embedding artifact: {artifact}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("inputs") != _embedding_inputs(target, kind, cap):
        raise RuntimeError(f"Embedding input lineage mismatch: {artifact}")
    if receipt.get("artifact") != lineage.artifact_identity(artifact):
        raise RuntimeError(f"Embedding artifact hash mismatch: {artifact}")
    return pd.read_parquet(artifact)


def _validated_head(target: str, cap: int) -> bool:
    artifact = head_path(target, cap)
    receipt_path = head_receipt_path(target, cap)
    if not artifact.exists() and not receipt_path.exists():
        return False
    if not artifact.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Partial immutable source-head artifact: {artifact}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("inputs") != _head_inputs(target, cap):
        raise RuntimeError(f"Source-head input lineage mismatch: {artifact}")
    if receipt.get("artifact") != lineage.artifact_identity(artifact):
        raise RuntimeError(f"Source-head artifact hash mismatch: {artifact}")
    return True


def extract_embeddings(target: str, kind: str, seed: int, cap: int) -> pd.DataFrame:
    """Slide-level 512-d embeddings + the head's own logit, from one frozen refit.

    The logit is carried alongside so the linearity identity can be checked
    against E2b's frozen scores instead of trusted.
    """
    import torch
    from torch.utils.data import DataLoader

    from oceanpath.datasets.datamodule import SimpleMILCollator, SlideDataset
    from oceanpath.training.lightning import MILTrainModule

    manifest = pd.read_csv(aim2_loco_transport.target_manifest(target, kind))
    dataset = SlideDataset(
        feature_dir=str(paths.PINNED_FEATURE_DIR),
        slide_ids=manifest["slide_id"].tolist(),
        labels=dict(zip(manifest["slide_id"], manifest["target_label"], strict=True)),
        max_instances=None,  # FULL bags, exactly as E2a/E2b scored
        is_train=False,
        force_float32=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=SimpleMILCollator(max_instances=None),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # MATCH THE SCORING PATH EXACTLY, including bf16 autocast. Running this in
    # fp32 instead is *more* precise but produces embeddings whose head logits
    # differ from E2a/E2b's frozen scores by ~1.6e-2 — enough that S0 would no
    # longer be the E2b baseline it is defined to be, and every "delta vs S0"
    # would inherit that offset. Precision is matched to the deployed object.
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    module = MILTrainModule.load_from_checkpoint(
        str(aim2_loco_transport.model_ckpt(target, seed, cap)), map_location=device, weights_only=False
    )
    module.eval().to(device)
    rows = []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device) if batch.get("mask") is not None else None
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = module.model(feats, mask=mask)
            emb = out.slide_embedding.detach().float().cpu().numpy()
            lg = out.logits.detach().float().cpu().numpy().ravel()
            for i, sid in enumerate(batch["slide_ids"]):
                rows.append(
                    {
                        "slide_id": sid,
                        "seed": seed,
                        "logit": float(lg[i]),
                        **{f"e{j}": float(v) for j, v in enumerate(emb[i])},
                    }
                )
    del module
    if device == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def cmd_embeddings(args: argparse.Namespace) -> None:
    for target, _sub in COHORTS:
        for kind in ("primary", "metastatic"):
            dest = emb_path(target, kind, args.cap)
            cached = _validated_embedding(target, kind, args.cap)
            if cached is not None:
                print(f"  {target}/{kind}: hash-valid immutable embedding — skipping")
                continue
            inputs_before = _embedding_inputs(target, kind, args.cap)
            frames = [extract_embeddings(target, kind, s, args.cap) for s in aim2_loco_transport.SEEDS]
            out = pd.concat(frames, ignore_index=True)
            if _embedding_inputs(target, kind, args.cap) != inputs_before:
                raise RuntimeError(
                    f"{target}/{kind}: input identity changed during embedding extraction"
                )
            lineage.write_parquet_once(dest, out)
            lineage.write_json_once(
                emb_receipt_path(target, kind, args.cap),
                {
                    "schema_version": 2,
                    "inputs": inputs_before,
                    "artifact": lineage.artifact_identity(dest),
                    "n_rows": int(len(out)),
                    "n_slides": int(out["slide_id"].nunique()),
                },
            )
            print(
                f"  {target}/{kind}: {len(out)} rows "
                f"({out.slide_id.nunique()} slides x {len(aim2_loco_transport.SEEDS)} seeds)"
            )
        # the frozen source head, per seed
        hp = head_path(target, args.cap)
        if _validated_head(target, args.cap):
            print(f"  {target}: hash-valid immutable source head — skipping")
            continue
        head_inputs_before = _head_inputs(target, args.cap)
        from oceanpath.training.lightning import MILTrainModule

        W, B = [], []
        for s in aim2_loco_transport.SEEDS:
            m = MILTrainModule.load_from_checkpoint(
                str(aim2_loco_transport.model_ckpt(target, s, args.cap)), map_location="cpu", weights_only=False
            )
            W.append(m.model.head.weight.detach().numpy().ravel().copy())
            B.append(float(m.model.head.bias.detach().numpy().ravel()[0]))
            del m
        lineage.write_npz_once(hp, w=np.vstack(W), b=np.array(B), seeds=np.array(aim2_loco_transport.SEEDS))
        if _head_inputs(target, args.cap) != head_inputs_before:
            raise RuntimeError(f"{target}: checkpoint identity changed during head extraction")
        lineage.write_json_once(
            head_receipt_path(target, args.cap),
            {
                "schema_version": 2,
                "inputs": head_inputs_before,
                "artifact": lineage.artifact_identity(hp),
            },
        )
        print(f"  {target}: source head saved {np.vstack(W).shape}")
    if args.verify:
        verify_linearity(args.cap)


def _auroc_logits(y: np.ndarray, eta: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, eta))


def _auroc_rows(y: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Tie-correct AUROC for every score row, vectorized for the bootstrap."""
    from scipy.stats import rankdata

    y = np.asarray(y)
    scores = _as_prediction_draws(eta, len(y))
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC requires both classes")
    ranks = rankdata(scores, method="average", axis=1)
    rank_sum_pos = ranks[:, y == 1].sum(axis=1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def labels_of(target: str, kind: str) -> np.ndarray:
    man = pd.read_csv(aim2_loco_transport.target_manifest(target, kind))
    return man.drop_duplicates("patient_id").sort_values("patient_id")["target_label"].to_numpy()


def verify_linearity(cap: int) -> None:
    """head(mean_slides(h)) == mean_slides(head(h)), and it must reproduce E2b.

    If this fails, patient-level head adaptation is NOT operating on the same
    quantity E2a/E2b reported and nothing downstream is comparable.
    """
    print("\n  linearity + reproduction audit")
    for target, _sub in COHORTS:
        if not _validated_head(target, cap):
            raise FileNotFoundError(head_path(target, cap))
        head = np.load(head_path(target, cap))
        for kind in ("primary", "metastatic"):
            emb = _validated_embedding(target, kind, cap)
            if emb is None:
                raise FileNotFoundError(emb_path(target, kind, cap))
            man = pd.read_csv(aim2_loco_transport.target_manifest(target, kind))
            sid2pat = dict(zip(man["slide_id"], man["patient_id"], strict=True))
            emb = emb.assign(patient_id=emb["slide_id"].map(sid2pat))
            ecols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
            # per seed: patient mean embedding -> head, vs patient mean logit
            worst_head, worst_e2b = 0.0, 0.0
            per_seed_pat, recon_per_seed = {}, []
            for s in aim2_loco_transport.SEEDS:
                d = emb[emb["seed"].eq(s)]
                pat_e = d.groupby("patient_id")[ecols].mean()
                pat_l = d.groupby("patient_id")["logit"].mean()
                recon = (
                    pat_e.to_numpy() @ head["w"][list(aim2_loco_transport.SEEDS).index(s)]
                    + head["b"][list(aim2_loco_transport.SEEDS).index(s)]
                )
                worst_head = max(worst_head, float(np.abs(recon - pat_l.to_numpy()).max()))
                per_seed_pat[s] = pat_l
                recon_per_seed.append(recon)
            # 3-seed ensemble vs the frozen E2b/E2a patient logits
            ens = pd.concat(per_seed_pat.values(), axis=1).mean(axis=1)
            recon_ens = np.mean(np.vstack(recon_per_seed), axis=0)
            frozen, _ = aim2_loco_transport.seed_ensemble(target, kind, cap)
            frozen = frozen.set_index("patient_id")["mean_logit"]
            common = ens.index.intersection(frozen.index)
            worst_e2b = float(np.abs(ens.loc[common] - frozen.loc[common]).max())
            # Two different tolerances, for two different reasons.
            #
            # REPRODUCTION must be exact: S0 is defined to BE the E2b baseline,
            # so the 3-seed ensemble of head(mean embedding) has to equal the
            # frozen patient logit E2a/E2b reported.
            #
            # LINEARITY is exact in real arithmetic but the deployed model
            # applies the head under bf16 autocast, so the stored logit carries
            # ~1e-2 of bf16 rounding that an fp32 reconstruction does not. The
            # identity is therefore audited where it is used — on patient-level
            # AUROC, the quantity E2c actually reports.
            head_auroc = _auroc_logits(labels_of(target, kind), recon_ens)
            frozen_auroc = _auroc_logits(labels_of(target, kind), frozen.loc[common].to_numpy())
            print(
                f"    {target:7s} {kind:11s} n_pat={len(common):4d}  "
                f"max|ensemble - frozen E2a/E2b| = {worst_e2b:.2e}   "
                f"head(mean h) bf16 gap = {worst_head:.2e}   "
                f"AUROC {head_auroc:.6f} vs {frozen_auroc:.6f}"
            )
            if worst_e2b > 1e-6:
                raise SystemExit("reproduction audit FAILED — S0 would not equal E2b")
            if abs(head_auroc - frozen_auroc) > 1e-4:
                raise SystemExit("linearity audit FAILED at the AUROC level")
    print("    PASS — patient mean embedding reproduces the frozen E2a/E2b logits,")
    print("           and the fp32 head reconstruction agrees on AUROC to <1e-4")


# ── L2-SP head adaptation ────────────────────────────────────────────────────
def fit_l2sp(
    H: np.ndarray,
    y: np.ndarray,
    w0: np.ndarray,
    b0: float,
    lam: float,
    iters: int = 60,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """Logistic regression with an L2-SP penalty toward (w0, b0), solved EXACTLY.

    THE OPTIMUM LIES IN THE SPAN OF THE SUPPORT SET. Stationarity of

        (1/n) sum_i BCE(y_i, sigmoid(w.h_i + b)) + (lam/2)(||w - w0||^2 + (b - b0)^2)

    in w gives  (1/n) sum_i (p_i - y_i) h_i + lam (w - w0) = 0, hence

        w = w0 - (1/(lam n)) sum_i (p_i - y_i) h_i  =  w0 + H^T alpha

    for some alpha in R^n. With n <= 16 support patients and d = 512 embedding
    dimensions, solving in alpha turns a 512-dimensional problem into an
    (n+1)-dimensional one and lets full Newton use the EXACT Hessian.

    WHY THIS REPLACED A DIAGONAL GAUSS-NEWTON ITERATION. The previous solver
    approximated the 513x513 Hessian by its diagonal. On real ABMIL embeddings —
    512 strongly correlated features, 4-16 points — that approximation
    understates the curvature badly enough that the Newton step overshoots and
    the iteration enters a LIMIT CYCLE instead of converging: at lam = 1 the step
    size was still 1.48 after 200 iterations, w(50) equalled w(200) exactly, and
    the returned point had objective 29.63 against a true optimum of 0.396. Any
    arm whose lambda selection landed below ~100 was therefore not fitting the
    stated objective at all. The exact form converges in a handful of steps, is
    ~21x faster, and reaches gradient norms of 1e-16.
    """
    if not np.isfinite(lam):  # infinite shrinkage == the source head
        return w0.copy(), float(b0)
    n = len(y)
    G = H @ H.T  # [n, n] Gram matrix
    z0 = H @ w0
    a = np.zeros(n)
    b = float(b0)
    eye = np.eye(n + 1)

    def objective(av: np.ndarray, bv: float) -> float:
        zz = z0 + G @ av + bv
        pp = np.clip(1.0 / (1.0 + np.exp(-np.clip(zz, -30, 30))), 1e-12, 1 - 1e-12)
        return float(
            -np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp))
            + lam / 2 * (av @ G @ av + (bv - b0) ** 2)
        )

    f_cur = objective(a, b)
    for _ in range(iters):
        z = z0 + G @ a + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        s = p * (1 - p) + 1e-12
        ga = G @ ((p - y) / n + lam * a)
        gb = float(np.mean(p - y) + lam * (b - b0))
        Haa = G @ ((s / n)[:, None] * G) + lam * G
        Hab = G @ (s / n)
        Hbb = float(np.mean(s) + lam)
        K = np.block([[Haa, Hab[:, None]], [Hab[None, :], np.array([[Hbb]])]])
        g = np.concatenate([ga, [gb]])
        step = np.linalg.solve(K + 1e-10 * eye, g)
        # ARMIJO BACKTRACKING. The objective is convex in (alpha, b) because G is
        # PSD, so a damped Newton step converges globally — but an UNDAMPED one
        # does not: at n = 4 support points and lam = 3 the full step overshot far
        # enough to leave a gradient norm of 1.78. Halving until the objective
        # actually decreases costs a few extra evaluations and removes the failure.
        t = 1.0
        gts = float(g @ step)
        for _ in range(40):
            a_new, b_new = a - t * step[:n], b - t * float(step[n])
            f_new = objective(a_new, b_new)
            if f_new <= f_cur - 1e-4 * t * gts:
                break
            t *= 0.5
        else:
            break  # no decrease available: already optimal
        a, b, f_cur = a_new, b_new, f_new
        if t * np.abs(step).max() < tol:
            break
    return w0 + H.T @ a, b


def select_lambda(H: np.ndarray, y: np.ndarray, w0: np.ndarray, b0: float) -> float:
    """Leave-one-out INSIDE THE SUPPORT SET. Never touches the held-out fold.

    With one class absent from a LOO split the fold is skipped rather than
    scored, and if no split is scorable the strongest shrinkage wins — which is
    S0, the safe default.
    """
    best, best_ll = LAMBDA_GRID[0], np.inf
    for lam in LAMBDA_GRID:
        lls = []
        for i in range(len(y)):
            keep = np.arange(len(y)) != i
            if len(np.unique(y[keep])) < 2:
                continue
            w, b = fit_l2sp(H[keep], y[keep], w0, b0, lam)
            p = float(np.clip(1 / (1 + np.exp(-np.clip(H[i] @ w + b, -30, 30))), EPS, 1 - EPS))
            lls.append(-(y[i] * np.log(p) + (1 - y[i]) * np.log(1 - p)))
        if lls and float(np.mean(lls)) < best_ll:
            best_ll, best = float(np.mean(lls)), lam
    return best


def stratified_folds(pat: pd.DataFrame, seed: int) -> np.ndarray:
    """Patient-grouped, KRAS-stratified 5 folds over the evaluation cohort."""
    from sklearn.model_selection import StratifiedKFold

    y = pat["label"].to_numpy()
    out = np.empty(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(skf.split(np.zeros(len(y)), y)):
        out[te] = f
    return out


def draw_support(pool: pd.DataFrame, k: int, rng: np.random.Generator) -> np.ndarray | None:
    """k patients per class, balanced. None if either class cannot supply k."""
    idx = []
    for cls in (0, 1):
        avail = np.flatnonzero(pool["label"].to_numpy() == cls)
        if len(avail) < k:
            return None
        idx.extend(rng.choice(avail, size=k, replace=False).tolist())
    return np.array(idx)


# ── patient tables ───────────────────────────────────────────────────────────
def patient_table(
    target: str, kind: str, cap: int, subcohort: str | None = None
) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    """Patient-level label table plus {seed: [n_patients, 512] mean embedding}."""
    emb = pd.read_parquet(emb_path(target, kind, cap))
    man = pd.read_csv(aim2_loco_transport.target_manifest(target, kind))
    if subcohort is not None:
        man = man[man["subcohort"].eq(subcohort)]
        emb = emb[emb["slide_id"].isin(set(man["slide_id"]))]
    sid = man.drop_duplicates("slide_id").set_index("slide_id")
    emb = emb.assign(patient_id=emb["slide_id"].map(sid["patient_id"]))
    ecols = [c for c in emb.columns if c.startswith("e") and c[1:].isdigit()]
    pats = (
        man.drop_duplicates("patient_id")[["patient_id", "target_label"]]
        .rename(columns={"target_label": "label"})
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    mats: dict[int, np.ndarray] = {}
    for s in aim2_loco_transport.SEEDS:
        g = emb[emb["seed"].eq(s)].groupby("patient_id")[ecols].mean()
        mats[s] = g.loc[pats["patient_id"]].to_numpy(dtype=np.float64)
    return pats, mats


def patient_native_logits(
    target: str, kind: str, cap: int, subcohort: str | None = None
) -> dict[int, np.ndarray]:
    """Stored checkpoint logits, averaged over slides and aligned by patient."""
    emb = pd.read_parquet(emb_path(target, kind, cap))
    man = pd.read_csv(aim2_loco_transport.target_manifest(target, kind))
    if subcohort is not None:
        man = man[man["subcohort"].eq(subcohort)]
        emb = emb[emb["slide_id"].isin(set(man["slide_id"]))]
    sid = man.drop_duplicates("slide_id").set_index("slide_id")
    emb = emb.assign(patient_id=emb["slide_id"].map(sid["patient_id"]))
    patients = man.drop_duplicates("patient_id")["patient_id"].sort_values().reset_index(drop=True)
    out: dict[int, np.ndarray] = {}
    for seed in aim2_loco_transport.SEEDS:
        logits = emb[emb["seed"].eq(seed)].groupby("patient_id")["logit"].mean()
        out[seed] = logits.loc[patients].to_numpy(dtype=np.float64)
    return out


def metrics(y: np.ndarray, eta: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    from oceanpath.eval.core import compute_calibration_intercept_slope

    p = np.clip(sigmoid(eta), EPS, 1 - EPS)
    if len(np.unique(y)) < 2:
        return {}
    cal = compute_calibration_intercept_slope(y, p)
    return {
        "auroc": float(roc_auc_score(y, eta)),  # logit scale: exact, tie-free
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": float(cal["calibration_intercept"]),
        "calibration_slope": float(cal["calibration_slope"]),
    }


def _as_prediction_draws(eta: np.ndarray, n_patients: int) -> np.ndarray:
    """Return a validated ``[procedure draw, patient]`` prediction matrix."""
    out = np.asarray(eta, dtype=np.float64)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2 or out.shape[1] != n_patients or out.shape[0] < 1:
        raise ValueError(
            "prediction draws must have shape [n_draws, n_patients]; "
            f"got {out.shape} for {n_patients} patients"
        )
    if not np.isfinite(out).all():
        raise ValueError("prediction draws contain non-finite values")
    return out


def _bootstrap_patient_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ordinary patient bootstrap, rejecting the rare single-class draw."""
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            return idx
    raise RuntimeError("could not draw a two-class patient bootstrap sample")


def _stable_seed(*parts: object) -> int:
    """Stable independent random streams, unaffected by loop or arm ordering."""
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _json_lambda(value: float) -> float | str:
    """Represent the decline-to-adapt grid point in standards-compliant JSON."""
    return float(value) if np.isfinite(value) else "infinity"


def procedure_summary(
    y: np.ndarray,
    prediction_draws: np.ndarray,
    *,
    n_boot: int = paths.N_BOOTSTRAP,
    random_seed: int = paths.BOOTSTRAP_SEED,
) -> dict:
    """Estimate expected repeated-cross-fit few-shot procedural performance.

    A row of ``prediction_draws`` is one complete repeated-cross-fit procedure:
    every held-out patient is scored once by a head fitted with exactly ``k``
    cases per class.  The headline is the mean of the per-draw metrics.  It is
    deliberately *not* the metric of a predictor averaged across draws, because
    such a predictor would have consumed the union of all support labels.

    The expected-performance CI is a two-way non-parametric bootstrap.  It
    resamples patients and the empirical support/training-procedure draws.  The
    single-procedure-draw interval samples one repetition per bootstrap and is
    reported separately: it describes support-procedure variability, rather
    than how precisely expected cross-fitted performance is estimated.  A
    repetition contains five fold-specific heads and is not called one deployed
    head.
    """
    y = np.asarray(y)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if y.ndim != 1 or len(np.unique(y)) != 2:
        raise ValueError("y must be a one-dimensional two-class patient label array")
    draws = _as_prediction_draws(prediction_draws, len(y))
    per_draw = [metrics(y, eta) for eta in draws]
    metric_names = tuple(per_draw[0])
    expected = {name: float(np.mean([row[name] for row in per_draw])) for name in metric_names}
    observed_aurocs = np.array([row["auroc"] for row in per_draw], dtype=float)

    rng = np.random.default_rng(random_seed)
    mean_boot: list[float] = []
    deployment_boot: list[float] = []
    for _ in range(n_boot):
        patient_idx = _bootstrap_patient_indices(y, rng)
        # Evaluate all observed support draws on the SAME patient resample.
        aucs = _auroc_rows(y[patient_idx], draws[:, patient_idx])
        rep_idx = rng.integers(0, len(draws), len(draws))
        mean_boot.append(float(np.mean(aucs[rep_idx])))
        deployment_boot.append(float(aucs[rng.integers(0, len(draws))]))

    return {
        "estimand": "expected metric of the exact-budget cross-fitted adaptation procedure",
        "n_procedure_draws": int(len(draws)),
        "metrics": expected,
        "expected_auroc_ci": [
            float(np.percentile(mean_boot, 2.5)),
            float(np.percentile(mean_boot, 97.5)),
        ],
        "single_procedure_draw_auroc": {
            "observed_sd": float(np.std(observed_aurocs, ddof=1))
            if len(observed_aurocs) > 1
            else 0.0,
            "observed_p2.5": float(np.percentile(observed_aurocs, 2.5)),
            "observed_p97.5": float(np.percentile(observed_aurocs, 97.5)),
            "patient_and_support_variability_interval": [
                float(np.percentile(deployment_boot, 2.5)),
                float(np.percentile(deployment_boot, 97.5)),
            ],
        },
        "per_draw_metrics": per_draw,
    }


def procedure_contrast(
    y: np.ndarray,
    prediction_draws_a: np.ndarray,
    prediction_draws_b: np.ndarray,
    *,
    n_boot: int = paths.N_BOOTSTRAP,
    random_seed: int = paths.BOOTSTRAP_SEED,
    support_resampling: str = "independent",
) -> dict:
    """Difference in expected AUROC with patient + support-draw uncertainty.

    Patients are always resampled once and shared between arms.  S1 and S2 use
    unrelated support pools, so their procedure draws are resampled
    independently.  ``paired`` is available only for genuinely paired
    procedures (same random support draw under two deterministic variants).
    A fixed S0 reference is represented by a one-row prediction matrix.
    """
    y = np.asarray(y)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if y.ndim != 1 or len(np.unique(y)) != 2:
        raise ValueError("y must be a one-dimensional two-class patient label array")
    a = _as_prediction_draws(prediction_draws_a, len(y))
    b = _as_prediction_draws(prediction_draws_b, len(y))
    if support_resampling not in {"independent", "paired"}:
        raise ValueError("support_resampling must be 'independent' or 'paired'")
    if support_resampling == "paired" and len(a) != len(b):
        raise ValueError("paired procedure draws must have equal lengths")

    auc_a = np.array([_auroc_logits(y, eta) for eta in a])
    auc_b = np.array([_auroc_logits(y, eta) for eta in b])
    point = float(np.mean(auc_a) - np.mean(auc_b))
    rng = np.random.default_rng(random_seed)
    mean_boot: list[float] = []
    deployment_boot: list[float] = []
    for _ in range(n_boot):
        patient_idx = _bootstrap_patient_indices(y, rng)
        boot_a = _auroc_rows(y[patient_idx], a[:, patient_idx])
        boot_b = _auroc_rows(y[patient_idx], b[:, patient_idx])
        if support_resampling == "paired":
            rep = rng.integers(0, len(a), len(a))
            mean_boot.append(float(np.mean(boot_a[rep] - boot_b[rep])))
            one = int(rng.integers(0, len(a)))
            deployment_boot.append(float(boot_a[one] - boot_b[one]))
        else:
            rep_a = rng.integers(0, len(a), len(a))
            rep_b = rng.integers(0, len(b), len(b))
            mean_boot.append(float(np.mean(boot_a[rep_a]) - np.mean(boot_b[rep_b])))
            deployment_boot.append(
                float(boot_a[rng.integers(0, len(a))] - boot_b[rng.integers(0, len(b))])
            )

    boot = np.asarray(mean_boot)
    return {
        "estimand": "difference in expected AUROC of exact-budget cross-fitted procedures",
        "delta": point,
        "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_gt_0": float(np.mean(boot > 0)),
        "patient_resampling": "paired",
        "support_resampling": support_resampling,
        "single_procedure_draw_delta_variability_interval": [
            float(np.percentile(deployment_boot, 2.5)),
            float(np.percentile(deployment_boot, 97.5)),
        ],
    }


def run_cohort(
    target: str,
    subcohort: str | None,
    cap: int,
    reps: int,
    n_bootstrap: int = paths.N_BOOTSTRAP,
) -> dict:
    """One cohort: 5 rotating folds x {S0, S1, S2} x budgets x reps."""
    if reps < 2:
        raise ValueError("reps must be at least 2 to estimate support-procedure variability")
    met_pat, met_emb = patient_table(target, "metastatic", cap)
    pri_pat, pri_emb = patient_table(target, "primary", cap, subcohort=subcohort)
    met_native = patient_native_logits(target, "metastatic", cap)
    head = np.load(head_path(target, cap))
    seeds = list(aim2_loco_transport.SEEDS)

    # dual-role patients: their PRIMARY must leave the S1 pool whenever their
    # METASTASIS is in the test fold, or the test label enters the support set.
    dual = set(met_pat["patient_id"]) & set(pri_pat["patient_id"])

    y_met = met_pat["label"].to_numpy().astype(float)
    y_pri = pri_pat["label"].to_numpy().astype(float)
    folds = stratified_folds(met_pat, paths.PRIMARY_SEED)

    # S0 — one frozen head, no support, no repetition needed
    eta0 = np.mean([met_native[s] for s in seeds], axis=0)
    out: dict = {
        "target": target,
        "primary_pool": subcohort or target,
        "n_metastatic": int(len(met_pat)),
        "n_metastatic_mut": int(y_met.sum()),
        "n_primary_pool": int(len(pri_pat)),
        "n_primary_mut": int(y_pri.sum()),
        "n_dual_role": len(dual),
        "reps": reps,
        "folds": N_FOLDS,
        "S0": metrics(y_met, eta0),
        "arms": {},
        "prediction_draws": {"S0": [eta0.tolist()]},
        "support_draws": {},
    }
    out["S0_inference"] = procedure_summary(
        y_met,
        eta0,
        n_boot=n_bootstrap,
        random_seed=_stable_seed(paths.BOOTSTRAP_SEED, target, "S0"),
    )

    for arm in ("S1", "S2"):
        for k in BUDGETS:
            eta_reps, lams, failures = [], [], 0
            support_reps: list[dict] = []
            for rep in range(reps):
                eta = np.full(len(y_met), np.nan)
                rep_audit: dict = {"repetition": rep, "folds": []}
                for f in range(N_FOLDS):
                    test = np.flatnonzero(folds == f)
                    if arm == "S2":
                        pool_idx = np.flatnonzero(folds != f)
                        pool = met_pat.iloc[pool_idx]
                        emb_src, y_src = met_emb, y_met
                    else:
                        blocked = {p for p in met_pat.iloc[test]["patient_id"] if p in dual}
                        pool_mask = ~pri_pat["patient_id"].isin(blocked).to_numpy()
                        pool_idx = np.flatnonzero(pool_mask)
                        pool = pri_pat.iloc[pool_idx]
                        emb_src, y_src = pri_emb, y_pri
                    support_seed = _stable_seed(
                        paths.BOOTSTRAP_SEED, "support", target, arm, k, rep, f
                    )
                    rng = np.random.default_rng(support_seed)
                    pick = draw_support(pool.reset_index(drop=True), k, rng)
                    if pick is None:
                        raise RuntimeError(
                            f"{target} {arm} k={k} repetition={rep} fold={f}: "
                            "support pool cannot supply the exact class-balanced budget"
                        )
                    sup = pool_idx[pick]
                    support_patients = pool.iloc[pick]["patient_id"].astype(str).tolist()
                    support_labels = y_src[sup].astype(int).tolist()
                    test_patients = set(met_pat.iloc[test]["patient_id"].astype(str))
                    if (
                        len(support_patients) != 2 * k
                        or support_labels.count(0) != k
                        or support_labels.count(1) != k
                    ):
                        raise RuntimeError("support draw violated the exact k-per-class budget")
                    if test_patients.intersection(support_patients):
                        raise RuntimeError("support/test patient leakage detected")
                    per_seed = []
                    fold_lams: list[float | str] = []
                    for i, s in enumerate(seeds):
                        Hs, ys = emb_src[s][sup], y_src[sup]
                        lam = select_lambda(Hs, ys, head["w"][i], head["b"][i])
                        w, b = fit_l2sp(Hs, ys, head["w"][i], head["b"][i], lam)
                        per_seed.append(
                            met_native[s][test]
                            if not np.isfinite(lam)
                            else met_emb[s][test] @ w + b
                        )
                        lams.append(lam)
                        fold_lams.append(_json_lambda(lam))
                    eta[test] = np.mean(per_seed, axis=0)
                    rep_audit["folds"].append(
                        {
                            "fold": f,
                            "support_seed": support_seed,
                            "support_patient_ids": support_patients,
                            "support_labels": support_labels,
                            "test_n": int(len(test)),
                            "lambda_by_model_seed": dict(
                                zip(map(str, seeds), fold_lams, strict=True)
                            ),
                        }
                    )
                if not np.isnan(eta).any():
                    eta_reps.append(eta)
                    support_reps.append(rep_audit)
            if len(eta_reps) != reps:
                raise RuntimeError(
                    f"{target} {arm} k={k}: completed {len(eta_reps)}/{reps} procedure draws"
                )
            E = np.vstack(eta_reps)
            inference = procedure_summary(
                y_met,
                E,
                n_boot=n_bootstrap,
                random_seed=_stable_seed(paths.BOOTSTRAP_SEED, target, arm, k, "summary"),
            )
            out["arms"][f"{arm}_k{k}"] = {
                "arm": arm,
                "k_per_class": k,
                "n_support": 2 * k,
                "reps_completed": len(eta_reps),
                "support_failures": failures,
                "lambda_frac_declined": float(np.mean(~np.isfinite(np.array(lams)))),
                "lambda_median_finite": float(np.median([x for x in lams if np.isfinite(x)]))
                if any(np.isfinite(x) for x in lams)
                else None,
                "performance": inference,
            }
            out["prediction_draws"][f"{arm}_k{k}"] = E.tolist()
            out["support_draws"][f"{arm}_k{k}"] = support_reps
    out["labels"] = y_met.tolist()
    out["fold_of_patient"] = folds.tolist()
    out["patient_ids"] = met_pat["patient_id"].astype(str).tolist()
    return out


def cmd_run(args: argparse.Namespace) -> None:
    dest = result_path(args.cap)
    lineage.ensure_absent(dest)
    for target, _subcohort in COHORTS:
        if not _validated_head(target, args.cap):
            raise FileNotFoundError(head_path(target, args.cap))
        for kind in ("primary", "metastatic"):
            if _validated_embedding(target, kind, args.cap) is None:
                raise FileNotFoundError(emb_path(target, kind, args.cap))
    report: dict = {
        "lineage": lineage.lineage_name(),
        "cap": args.cap,
        "reps": args.reps,
        "code": lineage.artifact_identity(Path(__file__)),
        "inputs": {
            target: {
                "primary_embeddings": lineage.artifact_identity(
                    emb_path(target, "primary", args.cap)
                ),
                "metastatic_embeddings": lineage.artifact_identity(
                    emb_path(target, "metastatic", args.cap)
                ),
                "source_heads": lineage.artifact_identity(head_path(target, args.cap)),
                "primary_embedding_receipt": lineage.artifact_identity(
                    emb_receipt_path(target, "primary", args.cap)
                ),
                "metastatic_embedding_receipt": lineage.artifact_identity(
                    emb_receipt_path(target, "metastatic", args.cap)
                ),
                "source_head_receipt": lineage.artifact_identity(
                    head_receipt_path(target, args.cap)
                ),
                "target_primary_manifest": lineage.artifact_identity(
                    aim2_loco_transport.target_manifest(target, "primary")
                ),
                "target_metastatic_manifest": lineage.artifact_identity(
                    aim2_loco_transport.target_manifest(target, "metastatic")
                ),
                "refit_checkpoints": {
                    str(seed): lineage.artifact_identity(aim2_loco_transport.model_ckpt(target, seed, args.cap))
                    for seed in aim2_loco_transport.SEEDS
                },
            }
            for target, _subcohort in COHORTS
        },
        "design": {
            "adapted": "classifier head only (513 params); encoder and ABMIL attention frozen",
            "penalty": "L2-SP toward the source head; lambda by LOO inside the support set",
            "budgets_per_class": list(BUDGETS),
            "endpoint": "difference in expected per-procedure AUROC(S2) - AUROC(S1)",
            "headline_estimator": "mean of repetition AUROCs; predictions are never averaged across repetitions",
            "uncertainty": "two-way bootstrap of patients and support/training-procedure repetitions",
            "contrast_support_resampling": "independent for S1 versus S2; patient resample shared",
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_seed": paths.BOOTSTRAP_SEED,
            "fold_seed": paths.PRIMARY_SEED,
            "support_seed_scheme": "SHA256(bootstrap_seed,target,arm,k,repetition,fold)",
        },
        "cohorts": {},
    }
    for target, sub in COHORTS:
        print(f"\n=== {target} (S1 pool = {sub or target}-P) ...")
        block = run_cohort(target, sub, args.cap, args.reps, args.n_bootstrap)
        y = np.array(block["labels"])
        prediction_draws = block["prediction_draws"]
        block["contrasts"] = {}
        for k in BUDGETS:
            s1, s2 = f"S1_k{k}", f"S2_k{k}"
            if s1 in prediction_draws and s2 in prediction_draws:
                block["contrasts"][f"S2_minus_S1_k{k}"] = procedure_contrast(
                    y,
                    np.array(prediction_draws[s2]),
                    np.array(prediction_draws[s1]),
                    n_boot=args.n_bootstrap,
                    random_seed=_stable_seed(paths.BOOTSTRAP_SEED, target, "S2_minus_S1", k),
                    support_resampling="independent",
                )
            for arm in (s1, s2):
                if arm in prediction_draws:
                    block["contrasts"][f"{arm}_minus_S0"] = procedure_contrast(
                        y,
                        np.array(prediction_draws[arm]),
                        np.array(prediction_draws["S0"]),
                        n_boot=args.n_bootstrap,
                        random_seed=_stable_seed(paths.BOOTSTRAP_SEED, target, f"{arm}_minus_S0"),
                        support_resampling="independent",
                    )
        report["cohorts"][target] = block
        print(f"    S0 AUROC {block['S0']['auroc']:.4f}")
        for name, a in block["arms"].items():
            if a.get("failed"):
                print(f"    {name}: FAILED ({a['support_failures']} support failures)")
            else:
                perf = a["performance"]
                print(
                    f"    {name}: expected AUROC {perf['metrics']['auroc']:.4f}  "
                    f"(support-draw SD {perf['single_procedure_draw_auroc']['observed_sd']:.4f})"
                    f"  declined {a['lambda_frac_declined']:.0%}"
                )
    lineage.write_json_once(dest, report)
    print(f"\nWrote {dest}")
    print_report(report)


def cmd_report(args: argparse.Namespace) -> None:
    print_report(json.loads(result_path(args.cap).read_text()))


def cmd_verify(args: argparse.Namespace) -> None:
    verify_linearity(args.cap)


def print_report(rep: dict) -> None:
    print(
        f"\n{'=' * 118}\nE2c — FEW-SHOT CLASSIFIER-HEAD ADAPTATION · cap {rep['cap']} · "
        f"{rep['reps']} support draws · {N_FOLDS} rotating folds\n{'=' * 118}"
    )
    print("  Adapted: classifier head only (513 params). Encoder and ABMIL attention FROZEN.")
    print("  S0 = zero-shot · S1 = local PRIMARY support · S2 = local METASTATIC support\n")
    for t, b in rep["cohorts"].items():
        print(
            f"{'-' * 118}\n{t}  (evaluate {b['n_metastatic']} metastatic patients, "
            f"{b['n_metastatic_mut']} mutant; S1 pool = {b['primary_pool']}-P, "
            f"n={b['n_primary_pool']}; {b['n_dual_role']} dual-role blocked)\n{'-' * 118}"
        )
        print(
            f"  {'arm':10s} {'support':>8s} {'exp AUROC [95% CI]':>28s} {'dAUROC vs S0':>26s} "
            f"{'AUPRC':>7s} {'Brier':>7s} {'cal int/slope':>15s} {'declined':>8s}"
        )
        s0 = b["S0"]
        s0_ci = b["S0_inference"]["expected_auroc_ci"]
        print(
            f"  {'S0':10s} {'0':>8s} "
            f"{s0['auroc']:.4f} [{s0_ci[0]:.4f},{s0_ci[1]:.4f}] {'—':>26s} "
            f"{s0['auprc']:7.4f} {s0['brier']:7.4f} "
            f"{s0['calibration_intercept']:+7.3f}/{s0['calibration_slope']:.3f} {'—':>8s}"
        )
        for arm in ("S1", "S2"):
            for k in BUDGETS:
                a = b["arms"].get(f"{arm}_k{k}")
                if not a or a.get("failed"):
                    continue
                d = b["contrasts"].get(f"{arm}_k{k}_minus_S0", {})
                ci = d.get("ci", [float("nan")] * 2)
                perf = a["performance"]
                m = perf["metrics"]
                aci = perf["expected_auroc_ci"]
                print(
                    f"  {arm + f' k={k}':10s} {a['n_support']:8d} "
                    f"{m['auroc']:.4f} [{aci[0]:.4f},{aci[1]:.4f}] "
                    f"{d.get('delta', float('nan')):+9.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] "
                    f"{m['auprc']:7.4f} {m['brier']:7.4f} "
                    f"{m['calibration_intercept']:+7.3f}/"
                    f"{m['calibration_slope']:.3f} {a['lambda_frac_declined']:7.0%}"
                )
                dep = perf["single_procedure_draw_auroc"]
                print(
                    f"    support-draw AUROC 2.5–97.5%: "
                    f"[{dep['observed_p2.5']:.4f}, {dep['observed_p97.5']:.4f}] "
                    "(variability, not a CI)"
                )
        print("\n  PRIMARY ENDPOINT — does the support have to be metastatic?")
        for k in BUDGETS:
            c = b["contrasts"].get(f"S2_minus_S1_k{k}")
            if not c:
                continue
            print(
                f"    k={k}/class ({2 * k} cases):  AUROC(S2) - AUROC(S1) = "
                f"{c['delta']:+.4f}  95% CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
                f"   P(S2 > S1) = {c['p_gt_0']:.3f}"
            )
        print()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    x = sub.add_parser("embeddings", help="extract frozen slide embeddings + source heads")
    x.add_argument("--cap", type=int, required=True, choices=aim2_loco_transport.E2A_CAPS)
    x.add_argument("--verify", action="store_true", help="audit the linearity identity")
    x.set_defaults(func=cmd_embeddings)
    x = sub.add_parser("verify", help="read-only audit of saved embeddings and native logits")
    x.add_argument("--cap", type=int, required=True, choices=aim2_loco_transport.E2A_CAPS)
    x.set_defaults(func=cmd_verify)
    x = sub.add_parser("run")
    x.add_argument("--cap", type=int, required=True, choices=aim2_loco_transport.E2A_CAPS)
    x.add_argument("--reps", type=int, default=DEFAULT_REPS)
    x.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    x.set_defaults(func=cmd_run)
    x = sub.add_parser("report")
    x.add_argument("--cap", type=int, required=True, choices=aim2_loco_transport.E2A_CAPS)
    x.set_defaults(func=cmd_report)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
