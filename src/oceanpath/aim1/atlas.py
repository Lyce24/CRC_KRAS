"""E3b — the label-blind morphology prototype vocabulary and its statistics.

E3b trains nothing. It asks a different question of models that already exist:
WHICH MORPHOLOGY carries the gene-level KRAS signal, is that morphology specific
to KRAS rather than to its molecular neighbourhood, and does it survive the move
to metastatic tissue?

Three design rules are enforced here rather than left to the caller.

LABEL-BLIND VOCABULARY. The prototypes are clustered from frozen UNIv1 tile
embeddings with no KRAS, MSI, BRAF, cohort or specimen-role label in sight. A
vocabulary fitted with any of those in the loop would be a classifier, and every
downstream "prototype k is enriched in KRAS-mutant tissue" would be circular.

ONE FROZEN VOCABULARY FOR EVERY ARM. The same centroids assign primary and
metastatic tiles, development and transport arms. Refitting per arm would make
"prototype 7" a different object in each panel and every cross-arm comparison
meaningless.

TWO QUANTITIES, NEVER COLLAPSED.

    abundance      tiles assigned to prototype k / eligible tiles
                   -> is this morphology physically more common in KRAS-mutant
                      tumours?
    attention mass sum of the frozen classifier's attention over tiles in k
                   -> is the KRAS model RELYING on this morphology?

They answer different questions and can disagree. The strong result is a
prototype that is both more abundant in mutant tissue AND more heavily attended;
either alone is a weaker, differently-worded claim.

Attention is a region-prioritisation signal, not a causal explanation: "the
model attends to X" is supported by these numbers, "X causes the KRAS signal" is
not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

# Sampling defaults. The vocabulary corpus is bounded by TOTAL_SAMPLE_TILES
# because 19.5M development tiles at 1,024 dimensions do not fit in memory and
# do not need to: a prototype vocabulary of a few dozen centroids is resolved
# long before that.
TOTAL_SAMPLE_TILES = 400_000
N_PROTOTYPES = 32          # inside the pre-specified 24-40 band
N_COMPONENTS = 64          # PCA rank before clustering
VOCAB_SEED = 20260819


# ── vocabulary ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Vocabulary:
    """Frozen prototype centroids plus the projection they live in.

    Stored together on purpose: a centroid is only meaningful in the PCA basis
    it was fitted in, and separating them is how an atlas silently starts
    assigning tiles in the wrong space.
    """

    centroids: np.ndarray        # [K, C] cluster centres in PCA space
    pca_mean: np.ndarray         # [D]    feature mean removed before projection
    pca_components: np.ndarray   # [C, D] projection rows
    normalize: str               # "l2" or "none", applied BEFORE centring
    config: dict[str, Any]

    @property
    def n_prototypes(self) -> int:
        return int(self.centroids.shape[0])

    def project(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if self.normalize == "l2":
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x = x / np.maximum(norms, 1e-12)
        return (x - self.pca_mean) @ self.pca_components.T

    def assign(self, features: np.ndarray) -> np.ndarray:
        """Nearest centroid per tile, as integer prototype ids."""
        z = self.project(features)
        # argmin ||z - c||^2 == argmax (z.c - 0.5||c||^2); the ||z||^2 term is
        # constant per tile and drops out.
        scores = z @ self.centroids.T - 0.5 * np.sum(self.centroids**2, axis=1)
        return np.asarray(np.argmax(scores, axis=1), dtype=np.int16)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            centroids=self.centroids,
            pca_mean=self.pca_mean,
            pca_components=self.pca_components,
        )
        path.with_suffix(".json").write_text(
            json.dumps({"normalize": self.normalize, **self.config}, indent=2, default=str)
        )

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        blob = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        return cls(
            centroids=blob["centroids"],
            pca_mean=blob["pca_mean"],
            pca_components=blob["pca_components"],
            normalize=meta.get("normalize", "l2"),
            config=meta,
        )


def sample_plan(
    slides: pd.DataFrame,
    total_tiles: int = TOTAL_SAMPLE_TILES,
    group_column: str = "atlas_group",
) -> pd.DataFrame:
    """How many tiles to draw from each slide, allocated evenly, not proportionally.

    Three equal splits in sequence: the budget is shared equally across GROUPS
    (subcohort x specimen role), then equally across PATIENTS inside a group,
    then equally across that patient's SLIDES. Proportional allocation would let
    SurGen — 737 patients at ~15,000 tiles a slide — define the vocabulary by
    itself, and the metastatic arms (74-85 patients) would contribute almost
    nothing, leaving normal liver parenchyma with no prototype of its own to be
    flagged as a shortcut.

    Equal allocation over-represents the small arms, and that is harmless: it
    only decides how many centroids each morphology earns. Every downstream
    statistic is computed on the full evaluated population, so the sampling
    weights cannot leak into an effect estimate.

    Slides with fewer tiles than their share give back what they cannot supply,
    and the shortfall is redistributed inside the same group.

    ``slides`` needs ``slide_id``, ``patient_id``, ``n_tiles`` and
    ``group_column``.
    """
    required = {"slide_id", "patient_id", "n_tiles", group_column}
    missing = required - set(slides.columns)
    if missing:
        raise ValueError(f"sample_plan needs columns {sorted(missing)}")

    groups = sorted(slides[group_column].unique())
    per_group = total_tiles // max(len(groups), 1)
    take: dict[str, int] = {}
    for group in groups:
        block = slides[slides[group_column].eq(group)]
        patients = sorted(block["patient_id"].unique())
        budget = per_group
        # Two passes: the first hands every patient an equal share capped by
        # what they have, the second re-offers the unspent remainder.
        for _ in range(2):
            if budget <= 0 or not patients:
                break
            share = max(budget // len(patients), 1)
            spent = 0
            for patient in patients:
                rows = block[block["patient_id"].eq(patient)]
                per_slide = max(share // len(rows), 1)
                for _, row in rows.iterrows():
                    already = take.get(row["slide_id"], 0)
                    room = int(row["n_tiles"]) - already
                    add = int(min(per_slide, max(room, 0)))
                    if add > 0:
                        take[row["slide_id"]] = already + add
                        spent += add
            budget -= spent
            if spent == 0:
                break
    plan = slides[["slide_id", "patient_id", group_column, "n_tiles"]].copy()
    plan["n_sample"] = plan["slide_id"].map(take).fillna(0).astype(int)
    return plan[plan["n_sample"] > 0].reset_index(drop=True)


def collect_sample(
    plan: pd.DataFrame,
    feature_dir: Path,
    seed: int = VOCAB_SEED,
    progress: bool = True,
) -> np.ndarray:
    """Read the planned tiles into one [T, D] float32 array.

    Tiles are drawn uniformly WITHOUT replacement inside each slide, so a slide
    contributes a spatially unbiased sample of its own tissue rather than a
    contiguous block of one region.

    Each slide is read whole and then indexed in memory: h5py point-selection on
    a scattered index list is roughly twice as slow as reading the (at most
    ~150 MB) contiguous block, and the tile order in the file is spatial, so a
    contiguous read is not a shortcut that biases the sample.
    """
    rng = np.random.default_rng(seed)
    blocks: list[np.ndarray] = []
    for index, row in enumerate(plan.itertuples(index=False), start=1):
        with h5py.File(Path(feature_dir) / f"{row.slide_id}.h5", "r") as handle:
            features = handle["features"][:]
        available = features.shape[0]
        n = int(min(row.n_sample, available))
        picks = rng.choice(available, size=n, replace=False)
        blocks.append(features[picks, :].astype(np.float32))
        if progress and index % 200 == 0:
            print(f"    sampled {index}/{len(plan)} slides", flush=True)
    return np.concatenate(blocks, axis=0)


def fit_vocabulary(
    sample: np.ndarray,
    n_prototypes: int = N_PROTOTYPES,
    n_components: int = N_COMPONENTS,
    seed: int = VOCAB_SEED,
    normalize: str = "l2",
    config: dict[str, Any] | None = None,
) -> Vocabulary:
    """PCA then k-means on frozen embeddings. No label is consulted anywhere.

    L2-normalising first makes the clustering effectively spherical, which
    matches the geometry UNIv1 was trained under; PCA is a rank reduction for
    speed and conditioning, not a denoiser, so it is not whitened.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    x = np.asarray(sample, dtype=np.float32)
    if normalize == "l2":
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    elif normalize != "none":
        raise ValueError(f"normalize must be 'l2' or 'none', got {normalize!r}")

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(x)
    kmeans = KMeans(n_clusters=n_prototypes, n_init=10, random_state=seed).fit(z)
    return Vocabulary(
        centroids=kmeans.cluster_centers_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        normalize=normalize,
        config={
            "n_prototypes": int(n_prototypes),
            "n_components": int(n_components),
            "seed": int(seed),
            "n_sample_tiles": int(len(x)),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "inertia": float(kmeans.inertia_),
            "cluster_sizes": np.bincount(
                kmeans.labels_, minlength=n_prototypes
            ).tolist(),
            **(config or {}),
        },
    )


def slide_profile(
    labels: np.ndarray,
    attention: dict[int, np.ndarray],
    n_prototypes: int,
) -> dict[str, np.ndarray]:
    """Abundance and per-seed attention mass for one slide.

    ``attention`` maps seed -> per-tile weights that already sum to 1 over the
    slide, so an attention mass is a share of that slide's attention and is
    directly comparable across slides with very different tile counts.
    """
    counts = np.bincount(labels, minlength=n_prototypes).astype(np.float64)
    profile = {"abundance": counts / max(counts.sum(), 1.0), "n_tiles": counts}
    for seed, weights in attention.items():
        if len(weights) != len(labels):
            raise ValueError(
                f"seed {seed}: {len(weights)} attention values for {len(labels)} tiles"
            )
        mass = np.zeros(n_prototypes, dtype=np.float64)
        np.add.at(mass, labels, weights.astype(np.float64))
        profile[f"attn_mass_seed{seed}"] = mass
    return profile


# ── statistics ───────────────────────────────────────────────────────────────
def auc_effect(values: np.ndarray, positive: np.ndarray) -> float:
    """P(value | positive > value | negative), ties counted as one half.

    An AUC rather than a mean difference because prototype abundances are
    compositional and heavy-tailed: a rank statistic is scale-free, comparable
    across prototypes of wildly different prevalence, and unaffected by the
    occasional slide that is 40% one morphology.

    Computed from average ranks rather than ``roc_auc_score``: identical to
    machine precision, ties included (many prototypes are absent from most
    slides, so exact zeros are the common case), and ~6x faster — which matters
    because the specificity panel alone evaluates this ~900,000 times.
    """
    from scipy.stats import rankdata

    y = np.asarray(positive).astype(int)
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(np.asarray(values, dtype=float))
    return float(
        (ranks[y == 1].sum() - n_positive * (n_positive + 1) / 2.0)
        / (n_positive * n_negative)
    )


def bootstrap_auc_effect(
    values: np.ndarray,
    positive: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 20260819,
) -> dict[str, float]:
    """Patient-level percentile bootstrap of ``auc_effect``.

    The caller must pass ONE row per patient. Resampling slides would treat a
    two-block patient as two independent observations, which is the same error
    the whole study avoids at the prediction stage.
    """
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    y = np.asarray(positive).astype(int)
    n = len(v)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(auc_effect(v[idx], y[idx]))
    values_arr = np.asarray(draws, dtype=float)
    point = auc_effect(v, y)
    return {
        "auc": point,
        "delta": point - 0.5,
        "ci_low": float(np.percentile(values_arr, 2.5)) if values_arr.size else float("nan"),
        "ci_high": float(np.percentile(values_arr, 97.5)) if values_arr.size else float("nan"),
        "n": int(n),
        "n_positive": int(y.sum()),
    }


def mannwhitney_p(values: np.ndarray, positive: np.ndarray) -> float:
    """Two-sided Mann-Whitney p, used only to order prototypes for FDR."""
    from scipy import stats

    v = np.asarray(values, dtype=float)
    y = np.asarray(positive).astype(bool)
    if y.sum() < 2 or (~y).sum() < 2:
        return float("nan")
    return float(stats.mannwhitneyu(v[y], v[~y], alternative="two-sided").pvalue)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values, NaNs passed through untouched.

    A prototype vocabulary is 24-40 simultaneous tests per quantity per
    population, so an unadjusted screen would manufacture "KRAS-associated
    morphology" at roughly two prototypes per panel by construction.
    """
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    values = p[finite]
    order = np.argsort(values)
    ranked = values[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(m, dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    out[finite] = restored
    return out


def concentration(shares: pd.Series) -> dict[str, Any]:
    """Largest single share and its owner — the shortcut screen's core number.

    A prototype that exists almost entirely in one cohort, one scanner batch or
    one metastatic organ is a candidate technical artefact regardless of how
    strong its KRAS association looks.

    ``shares`` must be a per-group MEAN ABUNDANCE, not a tile count. Raw tile
    sums would make the screen a measure of cohort size: SurGen contributes
    about half the study's tiles, so most prototypes would read as
    "SurGen-dominated" whether or not their morphology is actually specific to
    it. Mean abundance asks the size-free question — how much of a typical
    patient's tissue in this group is this morphology — and a prototype present
    equally everywhere then splits evenly across the groups.
    """
    total = float(shares.sum())
    if total <= 0:
        return {"top_share": float("nan"), "top_key": None}
    fractions = (shares / total).sort_values(ascending=False)
    return {
        "top_share": float(fractions.iloc[0]),
        "top_key": str(fractions.index[0]),
        "shares": {str(k): float(v) for k, v in fractions.items()},
    }
