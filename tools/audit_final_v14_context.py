#!/usr/bin/env python3
"""Read-only numeric replay of completed FINAL-v14 context association panels."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import final_v14_context as context  # noqa: E402


def run() -> dict:
    io = context.io_tools
    summary = context.sealed(context.OUT / "results.json")
    contract = context.sealed(context.CONTRACT_PATH)
    assert summary["status"] == "CONTEXT_ASSOCIATIONS_COMPLETE"
    for record in contract["input_identities"] + contract["code_identities"]:
        context.verify_pin(record)
    for record in summary["artifacts"].values():
        context.verify_pin(record)
    frame = context.m1.patient_frame(context.PRE)
    full = pd.read_parquet(context.PRE / "teachers/oof_native_logits_recomputed.parquet").set_index("patient_id")
    frame["full_logit"] = full.loc[frame.patient_id, "mean_logit_5seed"].to_numpy(float)
    mapping = pd.read_csv(context.PRE / "mappings/outer_to_reference.csv")
    dictionary = context.sealed(context.PRE / "analysis_dictionary.json")
    prototypes = contract["ATTRIBUTABLE_REF"]
    panel_counts, tests = {}, 0
    with threadpool_limits(limits=1):
        for representation in ("abundance", "attention"):
            local_path = context.PRE / ("profiles/oof_patient_profiles.parquet" if representation == "abundance" else "teachers/oof_attention_patients_5seed_mean.parquet")
            Yall = context.aligned_context(frame, mapping, pd.read_parquet(local_path), prototypes,
                                          boundary_tolerance=2e-6 if representation == "attention" else 1e-12)
            for variable in context.COVARIATES if representation == "abundance" else context.COVARIATES[:2]:
                panel = context.sealed(context.OUT / "panels" / f"{representation}__{variable}.json")
                design = context.association_design(frame, variable, dictionary)
                X, kept, selected = design["X"], design["kept"], design["tested"]
                Y = Yall[kept]
                assert len(panel["rows"]) == len(prototypes) == 25
                assert panel["n_complete_cases"] == len(kept)
                orthogonal, triangular = np.linalg.qr(X, mode="reduced")
                qr_inverse = np.linalg.solve(triangular, orthogonal.T)
                leverage = np.sum(orthogonal**2, axis=1)
                replay_p = []
                for index, row in enumerate(panel["rows"]):
                    assert row["prototype_id"] == prototypes[index]
                    coefficients = qr_inverse @ Y[:, index]
                    residual = Y[:, index] - X @ coefficients
                    omega = (residual / (1 - leverage)) ** 2
                    covariance = (qr_inverse * omega) @ qr_inverse.T
                    b, cov = coefficients[selected], covariance[np.ix_(selected, selected)]
                    if row["status"] == "NOT_ESTIMABLE":
                        assert np.linalg.matrix_rank(cov) < len(selected)
                        assert row["p_value"] == 1 and all(c["adjusted_coefficient"] is None for c in row["contrasts"])
                        replay_p.append(1.0)
                        tests += 1
                        continue
                    p = float(2 * norm.sf(abs(b[0] / np.sqrt(cov[0, 0])))) if len(selected) == 1 else float(chi2.sf(b @ np.linalg.solve(cov, b), len(selected)))
                    np.testing.assert_allclose(row["p_value"], p, atol=1e-10, rtol=1e-8,
                                               err_msg=f"{representation}/{variable}/p{prototypes[index]} condition={np.linalg.cond(X)}")
                    for c, contrast in enumerate(row["contrasts"]):
                        np.testing.assert_allclose(contrast["adjusted_coefficient"], b[c], atol=1e-10)
                        np.testing.assert_allclose(contrast["standardized_effect"], b[c] / np.std(Y[:, index], ddof=1), atol=1e-10)
                    replay_p.append(p)
                    tests += 1
                np.testing.assert_allclose([r["q_value_bh"] for r in panel["rows"]], context.bh_adjust(np.asarray(replay_p)), atol=1e-10)
                with np.load(panel["bootstrap_artifact"]["path"], allow_pickle=False) as archive:
                    assert archive["raw_contrasts"].shape == (2000, len(selected), len(prototypes))
                    assert archive["patient_indices"].shape == (2000, len(kept))
                    assert set(archive["patient_indices"].ravel()).issubset(set(kept))
                    lookup = {int(global_index): i for i, global_index in enumerate(kept)}
                    for draw in (0, 997, 1999):
                        sampled = np.array([lookup[int(i)] for i in archive["patient_indices"][draw]])
                        np.testing.assert_allclose(archive["raw_contrasts"][draw], np.linalg.lstsq(X[sampled], Y[sampled], rcond=None)[0][selected], atol=1e-10)
                    for index, row in enumerate(panel["rows"]):
                        for c, contrast in enumerate(row["contrasts"]):
                            assert contrast["adjusted_coefficient_bootstrap"] == context.finite_interval(archive["raw_contrasts"][:, c, index])
                            assert contrast["standardized_effect_bootstrap"] == context.finite_interval(archive["standardized_contrasts"][:, c, index])
                panel_counts[f"{representation}__{variable}"] = {"prototypes": len(prototypes), "n_complete_cases": len(kept),
                    "bh_q_below_0_05": sum(r["q_value_bh"] < 0.05 for r in panel["rows"])}
    coefficients = context.sealed(context.OUT / "coefficient_stability.json")
    assert all(r["ridge"]["available"] == 25 and r["logistic"]["available"] == 5 for r in coefficients["rows"])
    result = {"status": "PASS", "created_utc": io.now(), "results": io.identity(context.OUT / "results.json"),
              "analysis_contract": io.identity(context.CONTRACT_PATH), "auditor": io.identity(Path(__file__)),
              "checks": {"all_source_artifact_hashes": "PASS", "independent_scalar_HC3_replayed_tests": tests,
                         "bootstrap_draws_each_panel": 2000, "three_raw_OLS_draws_replayed_each_panel": [0, 997, 1999],
                         "all_reported_bootstrap_effect_CIs_replayed": "PASS", "coefficient_counts": "25 ridge and 5 logistic per prototype"},
              "panels": panel_counts}
    path = context.OUT / "verification.json"
    if not path.exists():
        io.publish(path, result)
    else:
        old = context.sealed(path)
        assert old["checks"] == result["checks"] and old["panels"] == result["panels"]
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "checks": result["checks"], "panels": result["panels"]}, indent=2))
