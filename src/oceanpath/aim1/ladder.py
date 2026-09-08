"""E5 — the molecular-resolution ladder and the controls that make a null mean something.

An allele task at chance is uninterpretable on its own: "G12C is at chance" is
fully explained by n=51. The claim only becomes evidence when the SAME
pipeline, at the SAME sample size and class balance, demonstrably learns the
gene-level task. That is what the sample-matched control provides.

Primary specimens only, and that restriction is load-bearing: an all-role
allele pool would break the shared-fold claim, put Aim-2 target-domain
metastatic specimens into an Aim-3 training set, and make the matched control
differ from the allele task in specimen-role composition — the exact confound
the control exists to remove.

The reported statistic for a null rung is not "we failed to reject". It is the
one-sided 95% UPPER confidence limit on the AUROC — a bounded, positive claim
about where histologic resolution stops.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Tier-1 rungs first; G12V/G13D are Tier 2, RAS is E3b.
LADDER = {
    "codon12": ("codon 12 vs non-codon-12", 1),
    "g12d": ("G12D vs other mutants", 1),
    "g12c": ("G12C vs other mutants (underpowered by design)", 1),
    "g12v": ("G12V vs other mutants", 2),
    "g13d": ("G13D vs other mutants", 2),
}

N_MATCHED_CONTROLS = 25  # resolves the control distribution to ~+/-0.02
EQUIVALENCE_MARGIN = 0.60  # TOST: "AUROC is bounded below this"


def matched_control_tasks(
    gene_population: pd.DataFrame,
    n_positive: int,
    n_negative: int,
    n_controls: int = N_MATCHED_CONTROLS,
    seed: int = 20260818,
) -> list[pd.DataFrame]:
    """Gene-level (KRAS mut vs WT) tasks subsampled to an allele task's exact shape.

    Same n+/n-, same folds, same encoder, same hyperparameters — the only
    difference is which label is being predicted. If these are learnable and
    the allele task is not, sample size is excluded as the explanation.
    """
    rng = np.random.default_rng(seed)
    patients = gene_population.drop_duplicates("patient_id")
    positives = patients[patients["target_label"] == 1]["patient_id"].to_numpy()
    negatives = patients[patients["target_label"] == 0]["patient_id"].to_numpy()
    if len(positives) < n_positive or len(negatives) < n_negative:
        raise ValueError(
            f"cannot match {n_positive}/{n_negative} from {len(positives)}/{len(negatives)}"
        )
    out = []
    for _ in range(n_controls):
        keep = set(rng.choice(positives, n_positive, replace=False).tolist())
        keep |= set(rng.choice(negatives, n_negative, replace=False).tolist())
        out.append(gene_population[gene_population["patient_id"].isin(keep)].copy())
    return out


def upper_confidence_limit(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 20260818,
) -> dict[str, float]:
    """One-sided 95% upper confidence limit on AUROC.

    This is the number to quote for a null rung. "Minimum detectable AUROC" is
    a design quantity, not a confidence bound, and quoting it as though it
    bounded the observed result would overstate the claim.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(labels)
    values = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, n, n)
        if len(np.unique(labels[index])) < 2:
            continue
        values.append(roc_auc_score(labels[index], scores[index]))
    point = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "auroc": point,
        "upper_95_one_sided": float(np.percentile(values, 95)) if values else float("nan"),
        "n": int(n),
        "n_positive": int(labels.sum()),
    }


def interpret(
    allele: dict[str, float], control_aurocs: list[float], null_aurocs: list[float]
) -> dict[str, Any]:
    """Map an allele rung onto the four pre-registered outcomes."""
    control = float(np.mean(control_aurocs)) if control_aurocs else float("nan")
    null_upper = float(np.percentile(null_aurocs, 95)) if null_aurocs else float("nan")
    observed = allele["auroc"]
    control_learnable = control > null_upper
    allele_learnable = observed > null_upper
    if control_learnable and not allele_learnable:
        verdict = "genuine molecular-resolution limit"
    elif not control_learnable:
        verdict = "sample-size limited — no claim"
    elif allele_learnable:
        verdict = "limited but real allele phenotype — reframe as positive"
    else:
        verdict = "indeterminate"
    return {
        "allele_auroc": observed,
        "allele_upper_95": allele["upper_95_one_sided"],
        "matched_control_mean_auroc": control,
        "permutation_null_upper_95": null_upper,
        "bounded_below": allele["upper_95_one_sided"] < EQUIVALENCE_MARGIN,
        "verdict": verdict,
    }
