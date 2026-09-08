from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools import final_v14_module2 as m


def test_median_penalty_requires_all_five_estimable_folds():
    folds = {str(f): {"status": "ESTIMABLE", "selected_penalty": c}
             for f, c in enumerate([100, 0.01, 1, 0.1, 1000])}
    assert m.source_penalty(folds) == 1
    folds["4"]["status"] = "NOT_ESTIMABLE"
    assert m.source_penalty(folds) is None
    assert m.source_penalty({k: v for k, v in folds.items() if k != "4"}) is None


def test_zero_variance_coordinate_cannot_change_target_logit():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    X[:, 2] = 0.1
    model = m.fit_logistic(X, np.arange(30) % 2, 0.01)
    assert model["coef"][2] == 0
    assert model["scaler_scale"][2] == 1
    target = X.copy()
    target[:, 2] = 100
    np.testing.assert_array_equal(m.score_matrix(model, target), m.score_matrix(model, X))


def test_equal_slide_ood_does_not_impute_empty_cells_or_pool_tiles():
    q99 = np.repeat(1.0, 32)
    a, ad = m.slide_diagnostics(np.array([0] * 100), np.array([2.] * 100), q99)
    b, bd = m.slide_diagnostics(np.array([1]), np.array([0.25]), q99)
    slides = pd.DataFrame([{"cohort": "c", "patient_id": "p", **dict(zip(m.COLS, v, strict=True))} for v in [a, b]])
    cells = pd.DataFrame([{"cohort": "c", "patient_id": "p", **d} for d in ad + bd])
    profiles, ood = m.aggregate_slides(slides, cells)
    np.testing.assert_array_equal(profiles[m.COLS].to_numpy()[0, :2], [0.5, 0.5])
    assert ood.iloc[0].fraction_beyond_source_q99 == 1
    assert ood.iloc[0].nonempty_slides == 1
    assert ood.iloc[0].assigned_tiles == 100
    assert ood.iloc[2].nonempty_slides == 0
    assert pd.isna(ood.iloc[2].median_distance)


def test_bad_bundle_blocks_before_any_target_feature_access(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(np, "memmap", lambda *a, **kw: opened.append(a))
    with pytest.raises(FileNotFoundError):
        m.score_targets(tmp_path)
    assert opened == []


def test_sealed_bundle_change_detected(tmp_path):
    p = tmp_path / "input.json"
    m.write_json(p, {"a": 1})
    m.seal(p)
    assert m.read_sealed(p) == {"a": 1}
    p.write_text('{"a":2}')
    with pytest.raises(m.ContractError, match="seal"):
        m.read_sealed(p)


def test_source_freeze_uses_no_target_paths_and_no_named_fit_on_gate_fail(tmp_path, monkeypatch):
    pre = tmp_path / "pre"
    output = tmp_path / "reports/reruns/final_v14_additions_test/e4m2"
    source = pre / "inputs/tcga_surgen_primary.csv"
    source.parent.mkdir(parents=True)
    ids = [f"P{j:04d}" for j in range(1239)]
    pd.DataFrame({"patient_id": ids, "target_label": [1] * 501 + [0] * 738}).to_csv(source, index=False)
    X = np.random.default_rng(5).dirichlet(np.ones(32), len(ids))
    profiles = pd.DataFrame(X, columns=m.COLS)
    profiles.insert(0, "patient_id", ids)
    profile = pre / "profiles/reference/patient_profiles.parquet"
    profile.parent.mkdir(parents=True)
    profiles.to_parquet(profile, index=False)
    quantiles = pre / "profiles/reference_distance_quantiles.csv"
    pd.DataFrame({"prototype_id": range(32), "q99_distance": np.ones(32)}).to_csv(quantiles, index=False)
    vocab = pre / "vocabularies/reference/variants/k32_seed20260819/vocabulary.npz"
    vocab.parent.mkdir(parents=True)
    np.savez(vocab, centroids=np.zeros((32, 2)), pca_mean=np.zeros(3), pca_components=np.zeros((2, 3)))
    m.write_json(pre / "receipts/profiles.json", {"artifacts": [m.identity(profile), m.identity(quantiles)]})
    m.write_json(pre / "receipts/vocabularies.json", {"artifacts": [m.identity(vocab)]})
    m.write_json(pre / "receipts/preflight.json", {"source": m.identity(source), "pack": {}})
    summary = tmp_path / "summary.json"
    checkpoint = tmp_path / "checkpoint.json"
    m.write_json(checkpoint, {"status": "ESTIMABLE"})
    m.seal(checkpoint)
    m.write_json(summary, {"status": "SOURCE_MODELS_SEALED", "contract": m.identity(checkpoint),
                          "representations": {"ALL32": {"outer_folds": {
        str(f): {"status": "ESTIMABLE", "selected_penalty": 0.01,
                 "checkpoint": m.identity(checkpoint)} for f in range(5)}}}})
    m.seal(summary)
    naming = tmp_path / "naming.json"
    m.write_json(naming, {"name_gate_status": "NAME_GATE_FAIL", "NAMED_REF": [6]})
    m.seal(naming)
    real_open = Path.open
    def guarded_open(path, *args, **kwargs):
        if str(path).startswith("/mnt/"):
            raise AssertionError("source freeze opened an external target/source feature path")
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    bundle = m.freeze_source(pre, summary, naming, output)
    assert set(bundle["models"]) == {"ALL32"}
    assert bundle["models"]["ALL32"]["selected_penalty"] == .01
    assert m.read_sealed(output / "source_dry_run_receipt.json")["representations"]["ALL32"]["max_absolute_logit_error"] == 0
    assert m.verify_bundle(output, require_snapshot=False)["status"] == "SOURCE_SCORING_BUNDLE_SEALED"
