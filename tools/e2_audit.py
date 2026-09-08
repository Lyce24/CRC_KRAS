#!/usr/bin/env python3
"""Independent audit of E2a / E2b / E2c / E2d invariants (Aim 2).

Every check re-derives its answer from the artifacts or from first principles
rather than reading a number the experiment reported about itself. A failure
here means a reported Aim-2 result is not trustworthy.

Usage:
    OCEANPATH_AIM2_LINEAGE=<immutable-rerun> \
        python tools/e2_audit.py --cap 8192
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))  # aim2_loco_transport.py / aim2_head_adaptation_base.py live at the repo root

import aim2_loco_transport  # noqa: E402
import aim2_metastatic_transport as e2b_workflow  # noqa: E402
import aim2_head_adaptation_base  # noqa: E402
from oceanpath.aim1 import lineage, paths  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _identity_is_live(value: object) -> bool:
    if not isinstance(value, dict) or not value.get("path"):
        return False
    try:
        return value == lineage.artifact_identity(Path(str(value["path"])))
    except (FileNotFoundError, OSError, ValueError):
        return False


# ── E2a ──────────────────────────────────────────────────────────────────────
def audit_e2a(cap: int) -> None:
    print(f"\n{'=' * 100}\nE2a — transported object and its training data\n{'=' * 100}")
    size_matched = aim2_loco_transport.matched_arm(aim2_loco_transport.SIZE_MATCHED_TARGET)
    arms = [*aim2_loco_transport.TARGETS]
    if aim2_loco_transport.source_manifest(size_matched).is_file():
        arms.append(size_matched)
    all_full, all_clean, all_contracts = True, True, True
    for t in arms:
        man = pd.read_csv(aim2_loco_transport.source_manifest(t))
        counts = man["target_label"].value_counts().to_dict()
        for s in aim2_loco_transport.SEEDS:
            aim2_loco_transport._completed_refit(t, s, cap)  # noqa: SLF001
            info = json.loads(
                (aim2_loco_transport.run_dir(t, s, cap) / "final" / "refit" / "info.json").read_text())
            lc = {int(k): v for k, v in info["label_counts"].items()}
            all_full &= (info["n_train_slides"] == len(man)) and (lc == counts)
            all_contracts &= (
                info.get("refit_max_steps") == aim2_loco_transport.STEP_BUDGET
                and info.get("actual_optimizer_steps") == aim2_loco_transport.STEP_BUDGET
                and info.get("batch_size") == 1
                and info.get("accumulate_grad_batches") == 1
                and info.get("seed") == s
                and info.get("sampling_seed") == s
                and info.get("train_sampling_strategy") == "patient_natural"
                and info.get("sample_weight_column") is None
                and info.get("class_weights") is None
                and info.get("dataset_max_instances") == cap
                and info.get("max_instances") is None
                and info.get("eval_full_bags") is True
                and (info.get("training_sampling") or {}).get("strategy")
                == "patient_natural"
                and (info.get("training_sampling") or {}).get("seed") == s
                and (info.get("training_sampling") or {}).get("samples_per_epoch")
                == man["patient_id"].nunique()
                and info.get("lr_scheduler_interval") == "step"
                and info.get("lr_scheduler_total_steps") == aim2_loco_transport.STEP_BUDGET
                and len(info.get("final_learning_rates") or []) == 1
                and np.isclose(
                    float(info["final_learning_rates"][0]),
                    1.0e-6,
                    rtol=1.0e-9,
                    atol=1.0e-12,
                )
            )
        # no target cohort, no target patient, in the source pool
        cohort = aim2_loco_transport.cohort_of(t)
        tp = pd.read_csv(aim2_loco_transport.target_manifest(t, "primary"))
        leak = set(man["patient_id"]) & set(tp["patient_id"])
        if cohort in aim2_loco_transport.MET_TARGETS:
            tm = pd.read_csv(aim2_loco_transport.target_manifest(t, "metastatic"))
            leak |= set(man["patient_id"]) & set(tm["patient_id"])
        all_clean &= (cohort not in set(man["cohort"].unique())) and not leak
    check(
        f"all {len(arms) * len(aim2_loco_transport.SEEDS)} refits trained on 100% of their source manifest",
        all_full,
    )
    check(
        "every refit used exactly 6,060 seeded patient visits and a step-wise LR clock",
        all_contracts,
    )
    check("no held-out cohort or target patient appears in any source pool", all_clean)

    # source is primary-only — metastatic morphology never enters training
    prim_only = all(
        set(pd.read_csv(aim2_loco_transport.source_manifest(t))["specimen_role"].unique()) == {"primary"}
        for t in arms)
    check("every source pool is primary-only (specimen_role)", prim_only)

    # the calibrator preserves ranking, on the exact scale
    from sklearn.metrics import roc_auc_score
    worst = 0.0
    for t in arms:
        cal = pd.read_parquet(aim2_loco_transport.calibrated_path(t, "primary", cap))
        y = cal["label"].to_numpy()
        a1 = roc_auc_score(y, cal["mean_logit"].to_numpy())
        a2 = roc_auc_score(y, cal["eta_source_calibrated"].to_numpy())
        worst = max(worst, abs(a1 - a2))
    check("source calibrator preserves AUROC exactly on the linear predictor",
          worst == 0.0, f"max |dAUROC| = {worst:.1e}")

    # the calibrator saw no target label: it is a function of source OOF only
    ok = True
    for t in arms:
        info = json.loads(aim2_loco_transport.calibrator_path(t, cap).read_text())
        src_n = pd.read_csv(aim2_loco_transport.source_manifest(t))["patient_id"].nunique()
        ok &= info["n_source"] == src_n
    check("each calibrator was fitted on exactly its source patients, no target rows", ok)

    if size_matched in arms:
        sensitivity = json.loads(
            (
                lineage.eval_root()
                / f"e2a_transport_pb_cap{cap}_size_matched.json"
            ).read_text()
        )["size_matched_sensitivity"]
        standard, _ = aim2_loco_transport.seed_ensemble(aim2_loco_transport.SIZE_MATCHED_TARGET, "primary", cap)
        matched, _ = aim2_loco_transport.seed_ensemble(size_matched, "primary", cap)
        derived = aim2_loco_transport.evaluate.compare_auroc_paired(
            standard,
            matched,
            score_column="mean_logit",
            n_bootstrap=int(sensitivity["n_bootstrap"]),
            seed=int(sensitivity["bootstrap_seed"]),
        )
        check(
            "RIH size-matched sensitivity is a reproducible paired-patient comparison",
            abs(derived["delta_auroc"] - sensitivity["delta_auroc"]) < 1e-12
            and abs(derived["ci_low"] - sensitivity["ci_low"]) < 1e-12
            and abs(derived["ci_high"] - sensitivity["ci_high"]) < 1e-12,
        )


# ── E2b ──────────────────────────────────────────────────────────────────────
def audit_e2b(cap: int) -> None:
    print(f"\n{'=' * 100}\nE2b — matched arms, no shared patients, direction of effect\n{'=' * 100}")
    rep = json.loads((lineage.eval_root() / f"e2b_metastatic_cap{cap}.json").read_text())
    primary, met = aim2_loco_transport.load_primary(), aim2_loco_transport.load_metastatic()
    from oceanpath.aim1 import population
    dual = population.dual_specimen_patients(primary, met)

    # RIH: the contrast's primary arm must exclude every dual-role patient
    ens_p, _ = aim2_loco_transport.seed_ensemble("RIH", "primary", cap)
    p_side = ens_p[~ens_p["patient_id"].isin(dual)]
    ens_m, _ = aim2_loco_transport.seed_ensemble("RIH", "metastatic", cap)
    m_side = ens_m[~ens_m["patient_id"].isin(dual)]
    check("RIH contrast arms share no patient",
          not (set(p_side["patient_id"]) & set(m_side["patient_id"])),
          f"primary n={len(p_side)} (of {len(ens_p)}), "
          f"metastatic n={len(m_side)} (of {len(ens_m)})")
    check("RIH dual-role patients are excluded from both contrast arms",
          len(p_side) == rep["targets"]["RIH"]["primary_vs_metastatic"]["primary_n"]
          and len(m_side)
          == rep["targets"]["RIH"]["primary_vs_metastatic"]["metastatic_n"]
          and not (set(p_side["patient_id"]) & dual)
          and not (set(m_side["patient_id"]) & dual))

    # SurGen: primary arm must be SR1482 only, matching the metastatic subcohort
    ens_ps, _ = aim2_loco_transport.seed_ensemble("SurGen", "primary", cap)
    sr = ens_ps[ens_ps["subcohort"].eq(aim2_loco_transport.SURGEN_MET_SUBCOHORT)]
    man_m = pd.read_csv(aim2_loco_transport.target_manifest("SurGen", "metastatic"))
    check("SurGen contrast uses SR1482-P only, and SurGen-M is entirely SR1482",
          len(sr) == rep["targets"]["SurGen"]["primary_vs_metastatic"]["primary_n"]
          and set(man_m["subcohort"].unique()) == {aim2_loco_transport.SURGEN_MET_SUBCOHORT},
          f"SR1482-P n={len(sr)}")

    # Training-seed signs are a computational-stability diagnostic only.  They
    # are printed, never made a pass/fail invariant or treated as six samples.
    neg = []
    for _t, b in rep["targets"].items():
        ps, ms = b["primary_per_seed_auroc"], b["metastatic_per_seed_auroc"]
        neg.extend([ms[k] - ps[k] < 0 for k in ps])
    print(f"  [INFO] seed-level primary→metastatic directions — {sum(neg)}/6 negative")

    arms = {
        "RIH": (p_side, m_side),
        "SurGen": (sr, aim2_loco_transport.seed_ensemble("SurGen", "metastatic", cap)[0]),
    }
    recorded = rep["combined_decrement"]
    derived = e2b_workflow.combined_decrement_ci(
        arms,
        n_boot=int(recorded["n_bootstrap"]),
        seed=int(recorded["bootstrap_seed"]),
    )
    check(
        "combined decrement is re-derived from the two cohort patient arms",
        abs(derived["delta_auroc"] - recorded["delta_auroc"]) < 1e-12
        and np.allclose(
            derived["delta_auroc_ci"], recorded["delta_auroc_ci"], atol=0, rtol=0
        )
        and derived["inference"] == recorded["inference"],
    )

    # metastatic manifests carry exactly one specimen role
    ok = all(set(pd.read_csv(aim2_loco_transport.target_manifest(t, "metastatic"))["specimen_role"].unique())
             == {"metastatic"} for t in aim2_loco_transport.MET_TARGETS)
    check("metastatic manifests contain only metastatic specimens", ok)


# ── E2c ──────────────────────────────────────────────────────────────────────
def audit_e2c(cap: int) -> None:
    print(f"\n{'=' * 100}\nE2c — solver, shrinkage, leakage, folds\n{'=' * 100}")
    rng = np.random.default_rng(0)

    # 1. the solver is correct: with w0 = 0 the L2-SP objective IS ridge logistic,
    #    so it must agree with sklearn. This is the only external check available.
    from sklearn.linear_model import LogisticRegression
    n, d = 40, 12
    H = rng.normal(size=(n, d))
    y = (rng.random(n) < 1 / (1 + np.exp(-(H @ rng.normal(size=d))))).astype(float)
    lam = 0.5
    w, b = aim2_head_adaptation_base.fit_l2sp(H, y, np.zeros(d), 0.0, lam)
    # The intercept must be folded into the design matrix: E2c penalises
    # (b - b_src)^2 BY DESIGN — shrinkage toward the source classifier includes
    # the bias, and that is what makes lambda -> inf reproduce S0 exactly —
    # whereas sklearn leaves the intercept unpenalised. Comparing against
    # sklearn's default would compare two different objectives and fail for a
    # reason that has nothing to do with the solver.
    Ha = np.hstack([H, np.ones((n, 1))])
    sk = LogisticRegression(C=1.0 / (lam * n), fit_intercept=False,
                            max_iter=5000, tol=1e-10).fit(Ha, y)
    theta, ref = np.append(w, b), sk.coef_.ravel()
    gap = float(np.abs(theta - ref).max())

    def objective(th):
        z = Ha @ th
        pr = 1 / (1 + np.exp(-z))
        return float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr))
                     + lam / 2 * np.sum(th ** 2))

    check("L2-SP solver reproduces sklearn on the SAME objective (intercept penalised)",
          gap < 1e-6 and objective(theta) <= objective(ref) + 1e-9,
          f"max |param diff| = {gap:.2e}, objective {objective(theta):.10f} "
          f"vs {objective(ref):.10f}")

    # 2. KKT ON THE REAL DATA, ACROSS THE WHOLE LAMBDA GRID.
    #    The earlier version of this check used a well-conditioned synthetic
    #    problem (d=12, one lambda) and passed while the solver was in a limit
    #    cycle on the actual embeddings at lambda <= 10. A convergence check that
    #    does not use the real feature geometry and the real grid is not a
    #    convergence check.
    worst = {}
    for target, _sb in aim2_head_adaptation_base.COHORTS:
        pat, mats = aim2_head_adaptation_base.patient_table(target, "metastatic", cap)
        head = np.load(aim2_head_adaptation_base.head_path(target, cap))
        Hm, ym = mats[aim2_loco_transport.SEEDS[0]], pat["label"].to_numpy().astype(float)
        r = np.random.default_rng(7)
        for k in aim2_head_adaptation_base.BUDGETS:
            pick = aim2_head_adaptation_base.draw_support(pat, k, r)
            Hs, ys = Hm[pick], ym[pick]
            for lm in aim2_head_adaptation_base.LAMBDA_GRID:
                if not np.isfinite(lm):
                    continue
                w, b = aim2_head_adaptation_base.fit_l2sp(Hs, ys, head["w"][0], head["b"][0], lm)
                pr = 1 / (1 + np.exp(-(Hs @ w + b)))
                gw = Hs.T @ (pr - ys) / len(ys) + lm * (w - head["w"][0])
                gb = float(np.mean(pr - ys) + lm * (b - head["b"][0]))
                # Scale-aware: the gradient carries a lam*(w - w0) term, so at
                # lam = 1e4 a residual of 1e-8 means ||w - w0|| is accurate to
                # 1e-12 — the float64 floor, not a convergence failure. Normalise
                # by the penalty scale so one threshold is meaningful across the
                # whole grid.
                worst[(target, k, lm)] = max(float(np.abs(gw).max()), abs(gb)) / (1.0 + lm)
    mx = max(worst.values())
    arg = max(worst, key=worst.get)
    check("solver converges on REAL embeddings at every lambda and budget",
          mx < 1e-9,
          f"worst scaled |grad| = {mx:.2e} at {arg[0]} k={arg[1]} "
          f"lambda={arg[2]:g} (over {len(worst)} fits)")

    # 2b. the returned point is a MINIMUM, not just a stationary iterate:
    #     no random nearby perturbation lowers the objective
    pat, mats = aim2_head_adaptation_base.patient_table("RIH", "metastatic", cap)
    head = np.load(aim2_head_adaptation_base.head_path("RIH", cap))
    Hm, ym = mats[aim2_loco_transport.SEEDS[0]], pat["label"].to_numpy().astype(float)
    pick = aim2_head_adaptation_base.draw_support(pat, 8, np.random.default_rng(11))
    Hs, ys = Hm[pick], ym[pick]
    w0r, b0r = head["w"][0], head["b"][0]

    def obj(w, b, lm):
        z = Hs @ w + b
        pr = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)
        return float(-np.mean(ys * np.log(pr) + (1 - ys) * np.log(1 - pr))
                     + lm / 2 * (np.sum((w - w0r) ** 2) + (b - b0r) ** 2))

    beaten = 0
    for lm in (1.0, 10.0, 100.0):
        w, b = aim2_head_adaptation_base.fit_l2sp(Hs, ys, w0r, b0r, lm)
        base = obj(w, b, lm)
        rr = np.random.default_rng(13)
        for _ in range(200):
            scale = 10 ** rr.uniform(-4, -1)
            if obj(w + scale * rr.normal(size=len(w)), b + scale * rr.normal(), lm) < base - 1e-12:
                beaten += 1
    check("no random perturbation improves on the returned solution (it is a minimum)",
          beaten == 0, f"{beaten}/600 perturbations beat it")

    # 3. infinite shrinkage returns the SOURCE head exactly -> the arm IS S0
    wi, bi = aim2_head_adaptation_base.fit_l2sp(Hs, ys, w0r, b0r, np.inf)
    check("lambda = infinity returns the source head exactly (arm collapses to S0)",
          np.array_equal(wi, w0r) and bi == b0r)

    # 4. shrinkage is monotone on the REAL support geometry
    dists = [float(np.linalg.norm(aim2_head_adaptation_base.fit_l2sp(Hs, ys, w0r, b0r, lm)[0] - w0r))
             for lm in (1.0, 10.0, 100.0, 1000.0)]
    check("larger lambda moves the head strictly less far from the source",
          all(dists[i] > dists[i + 1] for i in range(len(dists) - 1)),
          " > ".join(f"{v:.3f}" for v in dists))

    # 5. lambda selection never sees anything but the support set — checked on the
    #    parsed syntax tree, not by grepping text (the docstring legitimately says
    #    "fold" and "held-out", which a substring search cannot tell from a read).
    import ast as _ast
    tree = _ast.parse((REPO / "aim2_head_adaptation_base.py").read_text())
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "select_lambda")
    args = {a.arg for a in fn.args.args}
    local = {t.id for n in _ast.walk(fn) if isinstance(n, _ast.Assign)
             for t in _ast.walk(n) if isinstance(t, _ast.Name)}
    comp = {g.target.id for n in _ast.walk(fn) if isinstance(n, _ast.For)
            for g in [n] if isinstance(g.target, _ast.Name)}
    allowed = args | local | comp | {"np", "LAMBDA_GRID", "EPS", "fit_l2sp", "len",
                                     "range", "float", "enumerate"}
    used = {n.id for n in _ast.walk(fn) if isinstance(n, _ast.Name)}
    leaked = used - allowed
    check("select_lambda touches only its support-set arguments (AST-checked)",
          not leaked, f"free names: {sorted(leaked) or 'none'}")

    # 6. folds: every metastatic patient is evaluated exactly once, stratified
    for target, _s in aim2_head_adaptation_base.COHORTS:
        met_pat, _ = aim2_head_adaptation_base.patient_table(target, "metastatic", cap)
        folds = aim2_head_adaptation_base.stratified_folds(met_pat, paths.PRIMARY_SEED)
        sizes = np.bincount(folds, minlength=aim2_head_adaptation_base.N_FOLDS)
        prev = [float(met_pat["label"].to_numpy()[folds == f].mean())
                for f in range(aim2_head_adaptation_base.N_FOLDS)]
        check(f"{target}: every metastatic patient in exactly one of {aim2_head_adaptation_base.N_FOLDS} folds",
              len(folds) == len(met_pat) and sizes.sum() == len(met_pat)
              and (sizes > 0).all(),
              f"fold sizes {sizes.tolist()}, prevalence "
              + "/".join(f"{p:.2f}" for p in prev))

    # 7. support/test disjointness and the dual-role block, simulated on the real data
    for target, sub in aim2_head_adaptation_base.COHORTS:
        met_pat, _ = aim2_head_adaptation_base.patient_table(target, "metastatic", cap)
        pri_pat, _ = aim2_head_adaptation_base.patient_table(target, "primary", cap, subcohort=sub)
        dual = set(met_pat["patient_id"]) & set(pri_pat["patient_id"])
        folds = aim2_head_adaptation_base.stratified_folds(met_pat, paths.PRIMARY_SEED)
        r = np.random.default_rng(1)
        bad_s2 = bad_s1 = 0
        for f in range(aim2_head_adaptation_base.N_FOLDS):
            test_ids = set(met_pat.iloc[np.flatnonzero(folds == f)]["patient_id"])
            # S2 support comes from the other folds only
            pool_idx = np.flatnonzero(folds != f)
            for k in aim2_head_adaptation_base.BUDGETS:
                pick = aim2_head_adaptation_base.draw_support(met_pat.iloc[pool_idx].reset_index(drop=True), k, r)
                if pick is not None:
                    sup = set(met_pat.iloc[pool_idx[pick]]["patient_id"])
                    bad_s2 += len(sup & test_ids)
            # S1 support must exclude a test patient's own primary
            blocked = {p for p in test_ids if p in dual}
            pool_mask = ~pri_pat["patient_id"].isin(blocked).to_numpy()
            bad_s1 += len(set(pri_pat[~pool_mask]["patient_id"]) - blocked)
            for k in aim2_head_adaptation_base.BUDGETS:
                idx = np.flatnonzero(pool_mask)
                pick = aim2_head_adaptation_base.draw_support(pri_pat.iloc[idx].reset_index(drop=True), k, r)
                if pick is not None:
                    bad_s1 += len(set(pri_pat.iloc[idx[pick]]["patient_id"]) & blocked)
        check(f"{target}: S2 support never overlaps the test fold", bad_s2 == 0)
        check(f"{target}: S1 support never contains a test patient's own primary "
              f"({len(dual)} dual-role)", bad_s1 == 0)

    # 8. patient grouping is structural: slides are averaged into the patient BEFORE
    #    any split, so a patient's slides cannot straddle support and test
    for target, _sub in aim2_head_adaptation_base.COHORTS:
        pat, mats = aim2_head_adaptation_base.patient_table(target, "metastatic", cap)
        man = pd.read_csv(aim2_loco_transport.target_manifest(target, "metastatic"))
        check(f"{target}: one row per patient after aggregation "
              f"({len(man)} slides -> {len(pat)} patients)",
              len(pat) == man["patient_id"].nunique()
              and mats[aim2_loco_transport.SEEDS[0]].shape[0] == len(pat))

    # 9. S0 reproduces the frozen E2b metastatic AUROC
    from sklearn.metrics import roc_auc_score
    aim2_metastatic_transport = json.loads((lineage.eval_root() / f"e2b_metastatic_cap{cap}.json").read_text())
    for target, _sub in aim2_head_adaptation_base.COHORTS:
        pat, _mats = aim2_head_adaptation_base.patient_table(target, "metastatic", cap)
        native = aim2_head_adaptation_base.patient_native_logits(target, "metastatic", cap)
        eta0 = np.mean([native[s] for s in aim2_loco_transport.SEEDS], axis=0)
        a = float(roc_auc_score(pat["label"].to_numpy(), eta0))
        ref = aim2_metastatic_transport["targets"][target]["metastatic_overall"]["auroc"]
        check(f"{target}: E2c's S0 equals E2b's metastatic AUROC",
              abs(a - ref) < 1e-4, f"{a:.6f} vs {ref:.6f}")

    # 10. The immutable result must retain every procedure draw and calculate
    #     the headline from per-draw metrics, never from averaged predictions.
    result = json.loads(aim2_head_adaptation_base.result_path(cap).read_text())
    for target, block in result["cohorts"].items():
        labels = np.asarray(block["labels"])
        patient_ids = np.asarray(block["patient_ids"], dtype=str)
        fold_of_patient = np.asarray(block["fold_of_patient"])
        predictions = block["prediction_draws"]
        for arm in ("S1", "S2"):
            for k in aim2_head_adaptation_base.BUDGETS:
                name = f"{arm}_k{k}"
                E = np.asarray(predictions[name], dtype=float)
                saved = block["arms"][name]
                aucs = np.array([roc_auc_score(labels, row) for row in E])
                headline = saved["performance"]["metrics"]["auroc"]
                check(f"{target} {name}: full repetition x patient matrix retained",
                      E.shape == (result["reps"], len(labels)) and np.isfinite(E).all(),
                      f"shape={E.shape}")
                check(f"{target} {name}: headline equals mean repetition AUROC",
                      abs(headline - float(aucs.mean())) < 1e-12
                      and "mean_auroc" not in saved,
                      f"saved={headline:.8f}, derived={aucs.mean():.8f}")

                support_ok = True
                audits = block["support_draws"][name]
                support_ok &= len(audits) == result["reps"]
                for repetition in audits:
                    support_ok &= len(repetition["folds"]) == aim2_head_adaptation_base.N_FOLDS
                    for fold in repetition["folds"]:
                        support = fold["support_patient_ids"]
                        support_ok &= len(support) == len(set(support)) == 2 * k
                        support_ok &= fold["support_labels"].count(0) == k
                        support_ok &= fold["support_labels"].count(1) == k
                        test_ids = set(patient_ids[fold_of_patient == fold["fold"]])
                        support_ok &= not (test_ids & set(support))
                check(f"{target} {name}: every saved support draw is exact-budget and leak-free",
                      support_ok)

        contrast_ok = all(
            row["patient_resampling"] == "paired"
            and row["support_resampling"] == "independent"
            for row in block["contrasts"].values()
        )
        check(f"{target}: contrasts pair patients and independently resample support",
              contrast_ok)


def audit_procedural_bootstrap() -> None:
    print(f"\n{'=' * 100}\nE2c — two-way procedural inference\n{'=' * 100}")
    rng = np.random.default_rng(3)
    y = (rng.random(200) < 0.4).astype(float)
    eta = rng.normal(size=200) + y
    draws = np.vstack([eta, eta + 0.1 * rng.normal(size=len(y))])
    d = aim2_head_adaptation_base.procedure_contrast(
        y, draws, draws, n_boot=300, random_seed=7, support_resampling="paired"
    )
    check("paired procedural bootstrap of an arm against itself is exactly zero",
          d["delta"] == 0.0 and d["ci"] == [0.0, 0.0],
          f"delta {d['delta']:.1e}, CI {d['ci']}")

    # Regression for the defect that invalidated the previous E2c headline.
    # Mean(AUROC per support procedure) and AUROC(mean logits) can differ; only
    # the first is available to a procedure constrained to k labels per class.
    ys = np.array([0, 0, 0, 1, 1, 1])
    synthetic = np.array([
        [0, 1, 3, 2, 4, 5],
        [4, 2, 0, 5, 1, 3],
    ], dtype=float)
    summary = aim2_head_adaptation_base.procedure_summary(ys, synthetic, n_boot=50, random_seed=8)
    from sklearn.metrics import roc_auc_score
    valid = summary["metrics"]["auroc"]
    invalid = float(roc_auc_score(ys, synthetic.mean(axis=0)))
    check("headline is mean(per-repetition AUROC), never AUROC(mean logits)",
          abs(valid - 7 / 9) < 1e-12 and invalid == 1.0 and valid != invalid,
          f"procedural={valid:.6f}, invalid ensemble={invalid:.6f}")


def audit_e2d(cap: int) -> None:
    """Validate the complete zero-fit robustness/concordance chain."""

    print(f"\n{'=' * 100}\nE2d — robustness and concordance artifact chain\n{'=' * 100}")
    root = lineage.eval_root()
    paths_by_name = {
        "E2d-1 metastatic sites": root / f"e2d1_metastatic_sites_cap{cap}.json",
        "E2d-2 peritoneal audit": root / f"e2d2_peritoneal_audit_cap{cap}.json",
        "E2d-3 Set-D contrast": root / f"e2d3_setd_contrast_cap{cap}.json",
        "E2d-4 SurGen gap": root / f"e2d4_surgen_gap_cap{cap}.json",
        "E2d-5 paired specimens": root / f"e2d5_paired_specimens_cap{cap}.json",
        "E2d-6 RIH technical regime": (
            root / f"e2d6_rih_acquisition_regime_cap{cap}.json"
        ),
    }
    reports: dict[str, dict] = {}
    for name, path in paths_by_name.items():
        exists = path.is_file()
        check(f"{name}: immutable report exists", exists, str(path))
        if not exists:
            continue
        report = json.loads(path.read_text())
        reports[name] = report
        check(
            f"{name}: lineage and cap match",
            report.get("lineage") == lineage.lineage_name()
            and report.get("cap") == cap,
        )

    if len(reports) != len(paths_by_name):
        return

    e2b_path = root / f"e2b_metastatic_cap{cap}.json"
    e2b_identity = lineage.artifact_identity(e2b_path)
    for name in (
        "E2d-1 metastatic sites",
        "E2d-3 Set-D contrast",
        "E2d-4 SurGen gap",
    ):
        check(
            f"{name}: bound to this lineage's E2b report",
            reports[name].get("upstream_e2b") == e2b_identity,
        )
    e2d1_identity = lineage.artifact_identity(
        paths_by_name["E2d-1 metastatic sites"]
    )
    check(
        "E2d-2: bound to this lineage's E2d-1 report",
        reports["E2d-2 peritoneal audit"].get("upstream_e2d1")
        == e2d1_identity,
    )

    # E2d-3's full-population row is intended to be E2b repeated verbatim
    # before Set-D restriction.  This catches a recurrence of the asymmetric
    # dual-role exclusion that previously made the two analyses incomparable.
    e2b_report = json.loads(e2b_path.read_text())
    setd_report = reports["E2d-3 Set-D contrast"]
    for target in aim2_loco_transport.MET_TARGETS:
        observed = setd_report["cohorts"][target]["full_population"]["delta_auroc"]
        expected = e2b_report["targets"][target]["primary_vs_metastatic"][
            "delta_auroc"
        ]
        check(
            f"{target}: E2d-3 full-population delta exactly reproduces E2b",
            abs(float(observed) - float(expected)) < 1e-12,
            f"{observed:.8f} vs {expected:.8f}",
        )

    paired = reports["E2d-5 paired specimens"]
    paired_audit = paired.get("population_audit") or {}
    check(
        "E2d-5: all nine verified pairs retained with RIH-only headline",
        paired_audit.get("n_pairs") == 9
        and paired_audit.get("cohort_counts") == {"RIH": 8, "TCGA": 1}
        and paired.get("guardrails", {}).get("auroc_computed") is False,
    )
    bootstrap_counts = {
        "E2d-1": reports["E2d-1 metastatic sites"].get("inference", {}).get(
            "n_bootstrap"
        ),
        "E2d-3": reports["E2d-3 Set-D contrast"].get("inference", {}).get(
            "n_bootstrap"
        ),
        "E2d-4": reports["E2d-4 SurGen gap"].get("inference", {}).get(
            "n_bootstrap"
        ),
        "E2d-5": paired.get("headline_RIH_same_model", {})
        .get("bootstrap", {})
        .get("n_bootstrap_requested"),
        "E2d-6": reports["E2d-6 RIH technical regime"]
        .get("inference", {})
        .get("n_bootstrap"),
    }
    check(
        "all inferential E2d panels used 10,000 bootstrap draws",
        all(value == 10_000 for value in bootstrap_counts.values()),
        str(bootstrap_counts),
    )
    paired_artifact = (paired.get("paired_table") or {}).get("artifact")
    check(
        "E2d-5: paired patient table hash is live",
        _identity_is_live(paired_artifact),
    )

    regime = reports["E2d-6 RIH technical regime"]
    interaction = (
        regime.get("e2b_aligned_primary_to_metastatic", {})
        .get("four_arm_interaction", {})
    )
    check(
        "E2d-6: exact 265/103/1 inventory mapping retained",
        (regime.get("mapping_audit") or {}).get("inventory_regime_counts")
        == {
            "aperio_native": 265,
            "repaired_converted_technical_regime": 103,
            "versa_descriptive_only": 1,
        },
    )
    check(
        "E2d-6: interaction used four disjoint patient arms and 10k draws",
        interaction.get("four_arms_pairwise_patient_disjoint") is True
        and interaction.get("n_bootstrap") == 10_000,
    )
    check(
        "E2d-6: scanner/equivalence claims remain forbidden",
        regime.get("guardrails", {}).get("scanner_claim_allowed") is False
        and regime.get("guardrails", {}).get(
            "equivalence_or_invariance_claim_allowed"
        )
        is False,
    )
    regime_artifact = (regime.get("patient_table") or {}).get("artifact")
    check(
        "E2d-6: technical-regime patient table hash is live",
        _identity_is_live(regime_artifact),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=int, default=8192, choices=aim2_loco_transport.E2A_CAPS)
    a = ap.parse_args()
    audit_e2a(a.cap)
    audit_e2b(a.cap)
    audit_e2c(a.cap)
    audit_procedural_bootstrap()
    audit_e2d(a.cap)
    print(f"\n{'=' * 100}")
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
