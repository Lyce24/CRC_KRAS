import numpy as np
import pandas as pd
import pytest

from tools import final_v14_module2_analysis as a


def test_tie_aware_auc_and_patient_paired_bootstrap():
    y = np.array([0, 0, 1, 1])
    s = np.array([0., 1., 1., 2.])
    assert a.auc(y, s) == .875
    boot = a.bootstrap_auc_vectors(y, {"concept": s, "full": s}, rng=np.random.default_rng(6), n_boot=100)
    np.testing.assert_array_equal(boot["concept"], boot["full"])
    again = a.bootstrap_auc_vectors(y, {"concept": s, "full": s}, rng=np.random.default_rng(6), n_boot=100)
    np.testing.assert_array_equal(boot["concept"], again["concept"])


def test_ratio_ci_counts_undefined_without_redrawing_or_truncation():
    concept = np.repeat(.8, 10000)
    full = np.repeat(.6, 10000)
    full[:500] = .5
    r = a.ratio_summary(.8, .6, concept, full)
    assert r["finite_draws"] == 9500 and r["undefined_draws"] == 500
    assert r["status"] == "ESTIMABLE"
    assert r["point"] > 1
    full[500] = .49
    r = a.ratio_summary(.4, .5, concept, full)
    assert r["point"] is None
    assert r["finite_draws"] == 9499
    assert r["status"] == "RATIO_CI_NOT_ESTIMABLE"
    assert r["ci95"] is None


def test_hierarchical_named_gate_cannot_rescue_all32_or_name_gate():
    def row(lo):
        return {"cohorts": {c: {"concept_one_sided_97_5_lower": lo} for c in a.CONTROLLING}}
    results = {"ALL32": row(.49), "NAMED_REF": row(.6)}
    assert a.hierarchical_gates(results, "NAME_GATE_PASS")["NAMED_REF"].endswith("NOT_EVALUABLE")
    results["ALL32"] = row(.51)
    assert a.hierarchical_gates(results, "NAME_GATE_PASS")["NAMED_REF"].endswith("SUPPORTED")
    assert a.hierarchical_gates(results, "NAME_GATE_FAIL")["NAMED_REF"].endswith("NOT_EVALUABLE")
    results["ALL32"]["cohorts"][a.CONTROLLING[0]]["concept_one_sided_97_5_lower"] = .5
    assert a.hierarchical_gates(results, "NAME_GATE_PASS")["ALL32"].endswith("NOT_ESTABLISHED")
    assert a.hierarchical_gates({}, "NAME_GATE_FAIL")["ALL32"].endswith("NOT_EVALUABLE")


def test_role_interaction_matches_hand_means_and_exact_linear_logit_change():
    rows = []
    for role, label, p in [("primary", 0, .1), ("primary", 1, .3), ("metastatic", 0, .2), ("metastatic", 1, .6)]:
        for n in range(3):
            x = np.zeros(32)
            x[:2] = [p, 1 - p]
            rows.append({"patient_id": f"{role}{label}{n}", "role": role, "label": label,
                         **dict(zip(a.scoring.COLS, x, strict=True))})
    model = {"status": "ESTIMABLE", "coordinates": list(range(32)), "scaler_mean": [.12] * 32,
             "scaler_scale": [2.] * 32, "coef": [3., -1.] + [0.] * 30, "intercept": 7.}
    draws = {}
    panel, result = a.interaction_panel(pd.DataFrame(rows), model, rng=np.random.default_rng(2), n_boot=100, n_perm=100,
                                        draw_sink=draws)
    assert panel.iloc[0].interaction == pytest.approx(.2)
    assert panel.iloc[0].role_shift_WT == pytest.approx(.1)
    assert panel.iloc[0].role_shift_mutant == pytest.approx(.3)
    assert result["sum_fixed_contributions"] == pytest.approx(.4)
    assert result["direct_logit_separation_change"] == pytest.approx(.4)
    np.testing.assert_allclose(draws["bootstrap_logit_separation_change"], .4)
    np.testing.assert_allclose(draws["bootstrap_interactions"][:, 0], .2)
    np.testing.assert_allclose((1 + draws["permutation_exceedance_counts"]) / 101, panel.permutation_p)
    assert len(panel) == 32
    assert ((panel.permutation_p >= 1 / 101) & (panel.permutation_p <= 1)).all()
    assert (panel.bh_q >= panel.permutation_p).all()
    assert (panel.iloc[2:].permutation_p == 1).all()


def test_role_interaction_rejects_dual_role_patients():
    with pytest.raises(ValueError, match="strictly patient-disjoint"):
        a.interaction_panel(pd.DataFrame({"patient_id": ["p", "p"]}), {}, rng=np.random.default_rng(1), n_boot=1, n_perm=1)


def test_missing_label_blind_score_seal_blocks_before_outcome_read(tmp_path, monkeypatch):
    monkeypatch.setattr(a.scoring, "verify_bundle", lambda *args, **kwargs: {})
    opened = []
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: opened.append(args))
    with pytest.raises(FileNotFoundError):
        a._joined_input(tmp_path, {"full_scores": {"path": "/not/allowed"}})
    assert opened == []


def test_bh_family_includes_every_prototype():
    p = np.r_[.001, np.repeat(1., 31)]
    assert a.bh(p)[0] == pytest.approx(.032)
    np.testing.assert_array_equal(a.bh(p)[1:], np.ones(31))
