"""Prespecified interpretation bands and the pooled external summary (§1D).

Two jobs, both of which must happen before unblinding to mean anything:

``bands``   the smallest observed AUROC whose 95% CI lower bound clears each
            reference line, at each cohort's actual size and prevalence. Fixed
            in advance, every outcome maps to a claim; fixed afterwards,
            nothing does.
``pooled``  a random-effects (DerSimonian-Laird) summary across the three
            external cohorts with I-squared, reported ALONGSIDE the per-cohort
            estimates rather than instead of them.

Reference lines (§1D):

    0.50   chance
    0.54   clinical-only baseline — routine age/sex/site information
    0.61   covariate ceiling — the best a non-image model over subcohort,
           site, age, sex, stage, MSI, and BRAF achieves. A floor on what a
           metadata confound could explain, not a hard cap: the image observes
           these constructs continuously where the labels are categorical,
           which is why 1C-R is still required.

The band calculation uses the Hanley-McNeil standard error, which depends only
on AUROC and the two class counts — so a band can be computed before any model
is scored, which is the entire point.

Caveat carried in the output: with k = 3 cohorts, tau-squared rests on two
degrees of freedom and I-squared is close to uninformative. The pooled number
is a co-summary, never a substitute for the per-cohort table.
"""

from __future__ import annotations

from typing import Any

import numpy as np

REFERENCE_LINES: dict[str, float] = {
    "chance": 0.50,
    "clinical_only": 0.54,
    "covariate_ceiling": 0.61,
}

Z95 = 1.959963985


def hanley_mcneil_se(auroc: float, n_positive: int, n_negative: int) -> float:
    """Standard error of an AUROC under the exponential-scores approximation."""
    if n_positive < 1 or n_negative < 1:
        return float("nan")
    q1 = auroc / (2.0 - auroc)
    q2 = 2.0 * auroc**2 / (1.0 + auroc)
    variance = (
        auroc * (1.0 - auroc)
        + (n_positive - 1) * (q1 - auroc**2)
        + (n_negative - 1) * (q2 - auroc**2)
    ) / (n_positive * n_negative)
    return float(np.sqrt(max(variance, 0.0)))


def minimum_clearing_auroc(
    reference: float, n_positive: int, n_negative: int, resolution: float = 0.0005
) -> float:
    """Smallest AUROC whose 95% CI lower bound exceeds ``reference``.

    Solved by scan rather than in closed form: the Hanley-McNeil variance is
    itself a function of the AUROC, so the lower bound is not monotone in any
    algebraically convenient way.
    """
    for value in np.arange(reference, 1.0, resolution):
        if value - Z95 * hanley_mcneil_se(float(value), n_positive, n_negative) > reference:
            return float(round(value, 4))
    return float("nan")


def bands(populations: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """Interpretation table for every prespecified population.

    ``populations`` maps a name to ``(n_positive, n_negative)``.
    """
    table: dict[str, Any] = {}
    for name, (n_positive, n_negative) in populations.items():
        table[name] = {
            "n": int(n_positive + n_negative),
            "n_mutant": int(n_positive),
            "n_wild_type": int(n_negative),
            **{
                f"clears_{line}": minimum_clearing_auroc(value, n_positive, n_negative)
                for line, value in REFERENCE_LINES.items()
            },
        }
    return {"reference_lines": REFERENCE_LINES, "populations": table}


def classify(auroc: float, ci_low: float) -> str:
    """The highest reference line this observed interval actually clears."""
    if not np.isfinite(ci_low):
        return "unresolved"
    for name in ("covariate_ceiling", "clinical_only", "chance"):
        if ci_low > REFERENCE_LINES[name]:
            return f"above_{name}"
    return "not_above_chance"


# ── Pooled external summary ───────────────────────────────────────────────────


def random_effects_pool(estimates: list[tuple[float, float]]) -> dict[str, Any]:
    """DerSimonian-Laird pooling of per-cohort AUROCs.

    ``estimates`` is a list of ``(auroc, standard_error)``. Pooling happens on
    the logit scale, where the sampling distribution is far closer to normal
    near the boundaries, and is transformed back for reporting.
    """
    if len(estimates) < 2:
        return {"note": "fewer than two cohorts — nothing to pool"}

    values = np.array([value for value, _ in estimates], dtype=float)
    errors = np.array([error for _, error in estimates], dtype=float)
    # Delta-method transfer of the SE onto the logit scale.
    theta = np.log(values / (1.0 - values))
    se_theta = errors / (values * (1.0 - values))

    weights = 1.0 / se_theta**2
    fixed = float((weights * theta).sum() / weights.sum())
    q = float((weights * (theta - fixed) ** 2).sum())
    df = len(estimates) - 1
    c = float(weights.sum() - (weights**2).sum() / weights.sum())
    tau2 = max(0.0, (q - df) / c) if c > 0 else 0.0

    re_weights = 1.0 / (se_theta**2 + tau2)
    pooled = float((re_weights * theta).sum() / re_weights.sum())
    se_pooled = float(np.sqrt(1.0 / re_weights.sum()))
    i2 = float(max(0.0, (q - df) / q) * 100.0) if q > 0 else 0.0

    def _inverse(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-value)))

    return {
        "pooled_auroc": _inverse(pooled),
        "ci_low": _inverse(pooled - Z95 * se_pooled),
        "ci_high": _inverse(pooled + Z95 * se_pooled),
        "tau2_logit": tau2,
        "q": q,
        "df": df,
        "i2_percent": i2,
        "k_cohorts": len(estimates),
        "per_cohort_auroc": values.tolist(),
        "caveat": (
            f"k={len(estimates)}: tau-squared rests on {df} degrees of freedom and "
            "I-squared is close to uninformative at this k. Read the per-cohort "
            "table as the result; this is a co-summary."
        ),
    }
