"""Pure, deterministic concept-dictionary helpers for FINAL-v14 Aim 4.

This module intentionally performs no file I/O.  It collects the parts of the
pre-reader computation whose exact behaviour must be easy to unit-test:

* balanced source-only sampling;
* the frozen PCA/k-means numerical contract;
* vocabulary assignment and semantic-coordinate matching;
* equal-slide patient abundance profiles;
* representative-tile and duplicate-montage selection; and
* blinded HMAC codes and presentation order.

Identifiers are compared lexicographically after conversion to strings.  All
prototype identifiers are integer, zero-based coordinates unless an explicit
``prototype_index`` is requested; that index is always the position in the
sorted prototype inventory, never an arbitrary input-row position.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

VOCABULARY_SEED = 20260819
TOTAL_SAMPLE_TILES = 400_000
N_PCA_COMPONENTS = 64
N_REFERENCE_PROTOTYPES = 32
NAME_MAPPING_COSINE = 0.80
MONTAGE_TILES = 12
MONTAGE_PATIENTS_PER_SUBCOHORT = 3

_MONTAGE_STAGES = ("decile", "quartile", "all")
_MONTAGE_TIER = {name: index for index, name in enumerate(_MONTAGE_STAGES)}


def pcg64_rng(seed: int) -> np.random.Generator:
    """Return the explicitly pinned generator used by ``default_rng``.

    NumPy currently makes ``default_rng(seed)`` a ``Generator(PCG64(seed))``.
    Spelling out the bit generator here makes that campaign dependency
    auditable and prevents a future NumPy default change from changing draws.
    """

    return np.random.Generator(np.random.PCG64(int(seed)))


def balanced_integer_quotas(
    capacities: Mapping[str, int], budget: int
) -> dict[str, int]:
    """Allocate an integer budget equally with sorted round-robin overflow.

    The initial quotient is offered to every lexicographically sorted unit and
    the remainder to the first units in that same order.  A unit never receives
    more than its capacity.  Any resulting shortfall is redistributed one item
    at a time over the nonexhausted units, restarting at the first sorted unit.

    The returned allocation always sums to ``min(budget, sum(capacities))``.
    """

    if int(budget) != budget or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    normalized: dict[str, int] = {}
    for raw_key, raw_capacity in capacities.items():
        key = str(raw_key)
        if key in normalized:
            raise ValueError(f"duplicate unit after string normalization: {key!r}")
        if int(raw_capacity) != raw_capacity or raw_capacity < 0:
            raise ValueError(f"capacity for {key!r} must be a nonnegative integer")
        normalized[key] = int(raw_capacity)

    keys = sorted(normalized)
    if not keys:
        if budget:
            return {}
        return {}

    target = min(int(budget), sum(normalized.values()))
    quotient, remainder = divmod(target, len(keys))
    allocation = {
        key: min(normalized[key], quotient + int(index < remainder))
        for index, key in enumerate(keys)
    }
    shortfall = target - sum(allocation.values())

    # One append per allocated remainder is bounded by the requested campaign
    # cap (400k), while avoiding repeated scans across exhausted units.
    active = [key for key in keys if allocation[key] < normalized[key]]
    while shortfall:
        if not active:  # pragma: no cover - guarded by the target definition
            raise AssertionError("quota redistribution exhausted all capacity")
        next_active: list[str] = []
        for key in active:
            if shortfall == 0:
                next_active.extend(
                    candidate
                    for candidate in active[active.index(key) :]
                    if allocation[candidate] < normalized[candidate]
                )
                break
            allocation[key] += 1
            shortfall -= 1
            if allocation[key] < normalized[key]:
                next_active.append(key)
        active = next_active

    return allocation


def hierarchical_sample_plan(
    slides: pd.DataFrame,
    *,
    cap: int = TOTAL_SAMPLE_TILES,
    subcohort_column: str = "subcohort",
    patient_column: str = "patient_id",
    slide_column: str = "slide_id",
    tile_count_column: str = "n_tiles",
) -> pd.DataFrame:
    """Build the exact subcohort -> patient -> slide balanced sample plan.

    Balancing is applied independently at every level with
    :func:`balanced_integer_quotas`.  Consequently, small subcohorts, patients,
    or slides return unused quota to peers at the *same* level before the next
    level is allocated.  The result contains every input slide, including
    exhausted or zero-allocation slides, in canonical hierarchy order.
    """

    required = {
        subcohort_column,
        patient_column,
        slide_column,
        tile_count_column,
    }
    missing = required - set(slides.columns)
    if missing:
        raise ValueError(f"slides are missing required columns: {sorted(missing)}")
    if int(cap) != cap or cap < 0:
        raise ValueError("cap must be a nonnegative integer")

    frame = slides.copy()
    for column in (subcohort_column, patient_column, slide_column):
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing identifiers")
        frame[column] = frame[column].astype(str)

    counts = pd.to_numeric(frame[tile_count_column], errors="coerce")
    if counts.isna().any() or (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError(f"{tile_count_column} must contain nonnegative integers")
    frame[tile_count_column] = counts.astype(np.int64)
    if frame[slide_column].duplicated().any():
        duplicates = sorted(frame.loc[frame[slide_column].duplicated(), slide_column].unique())
        raise ValueError(f"slide identifiers are not unique: {duplicates[:5]}")
    patient_groups = frame.groupby(patient_column, sort=False)[subcohort_column].nunique()
    if (patient_groups > 1).any():
        raise ValueError("a patient cannot belong to more than one subcohort")

    hierarchy = [subcohort_column, patient_column, slide_column]
    frame = frame.sort_values(hierarchy, kind="stable").reset_index(drop=True)
    total_available = int(frame[tile_count_column].sum())
    target = min(int(cap), total_available)

    subcohort_capacity = (
        frame.groupby(subcohort_column, sort=True)[tile_count_column].sum().astype(int).to_dict()
    )
    subcohort_quota = balanced_integer_quotas(subcohort_capacity, target)
    slide_quota: dict[str, int] = {}

    for subcohort in sorted(subcohort_quota):
        subcohort_rows = frame.loc[frame[subcohort_column].eq(subcohort)]
        patient_capacity = (
            subcohort_rows.groupby(patient_column, sort=True)[tile_count_column]
            .sum()
            .astype(int)
            .to_dict()
        )
        patient_quota = balanced_integer_quotas(
            patient_capacity, subcohort_quota[subcohort]
        )
        for patient in sorted(patient_quota):
            patient_rows = subcohort_rows.loc[subcohort_rows[patient_column].eq(patient)]
            capacities = dict(
                zip(
                    patient_rows[slide_column].astype(str),
                    patient_rows[tile_count_column].astype(int),
                    strict=True,
                )
            )
            slide_quota.update(
                balanced_integer_quotas(capacities, patient_quota[patient])
            )

    frame["n_sample"] = frame[slide_column].map(slide_quota).fillna(0).astype(np.int64)
    if int(frame["n_sample"].sum()) != target:
        raise AssertionError("hierarchical plan did not exhaust the attainable cap")
    if (frame["n_sample"] > frame[tile_count_column]).any():
        raise AssertionError("hierarchical plan exceeded a slide capacity")
    return frame


def _canonical_tile_ids(tile_ids: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([str(value) for value in tile_ids], dtype=object)
    if len(set(values.tolist())) != len(values):
        raise ValueError("tile identifiers must be unique within a slide")
    order = np.asarray(sorted(range(len(values)), key=lambda index: values[index]), dtype=np.int64)
    return values, order


def deterministic_sample_indices(
    tile_ids: Sequence[Any],
    n_sample: int,
    *,
    seed: int = VOCABULARY_SEED,
) -> np.ndarray:
    """Select original row indices after lexically sorting tile identifiers.

    Sampling is without replacement with a pinned PCG64 generator.  Returned
    indices are in lexical tile-ID order, which is also the order in which a
    fit matrix should be assembled.
    """

    _, canonical_order = _canonical_tile_ids(tile_ids)
    if int(n_sample) != n_sample or n_sample < 0 or n_sample > len(canonical_order):
        raise ValueError("n_sample must be an integer between zero and the tile count")
    rng = pcg64_rng(seed)
    chosen = rng.choice(len(canonical_order), size=int(n_sample), replace=False)
    original_positions = canonical_order[np.asarray(chosen, dtype=np.int64)]
    rank = {int(position): index for index, position in enumerate(canonical_order)}
    return np.asarray(sorted(original_positions, key=lambda position: rank[int(position)]))


def sample_id_table(
    plan: pd.DataFrame,
    tile_ids_by_slide: Mapping[str, Sequence[Any]],
    *,
    seed: int = VOCABULARY_SEED,
    subcohort_column: str = "subcohort",
    patient_column: str = "patient_id",
    slide_column: str = "slide_id",
    tile_count_column: str = "n_tiles",
    sample_count_column: str = "n_sample",
) -> pd.DataFrame:
    """Materialize the exact sampled tile-ID census using one PCG64 stream.

    The shared generator advances over slides in canonical hierarchy order.
    The final table is sorted by subcohort, patient, slide, and tile identifier,
    making the input row order irrelevant and making the PCA input order
    explicit.  ``tile_index`` is the tile's original zero-based row position.
    """

    required = {
        subcohort_column,
        patient_column,
        slide_column,
        tile_count_column,
        sample_count_column,
    }
    missing = required - set(plan.columns)
    if missing:
        raise ValueError(f"plan is missing required columns: {sorted(missing)}")

    sources = {str(key): value for key, value in tile_ids_by_slide.items()}
    if len(sources) != len(tile_ids_by_slide):
        raise ValueError("duplicate slide key after string normalization")
    frame = plan.copy()
    for column in (subcohort_column, patient_column, slide_column):
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing identifiers")
        frame[column] = frame[column].astype(str)
    frame = frame.sort_values(
        [subcohort_column, patient_column, slide_column], kind="stable"
    ).reset_index(drop=True)

    rng = pcg64_rng(seed)
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        values = row._asdict()
        slide = str(values[slide_column])
        if slide not in sources:
            raise ValueError(f"tile identifiers are missing for slide {slide!r}")
        ids, canonical_order = _canonical_tile_ids(sources[slide])
        declared_count = int(values[tile_count_column])
        n_sample = int(values[sample_count_column])
        if len(ids) != declared_count:
            raise ValueError(
                f"slide {slide!r} declares {declared_count} tiles but has {len(ids)} IDs"
            )
        if n_sample < 0 or n_sample > len(ids):
            raise ValueError(f"invalid sample count for slide {slide!r}: {n_sample}")
        draws = np.asarray(
            rng.choice(len(ids), size=n_sample, replace=False), dtype=np.int64
        )
        chosen_positions = canonical_order[draws]
        for draw_index, original_position in enumerate(chosen_positions):
            records.append(
                {
                    subcohort_column: str(values[subcohort_column]),
                    patient_column: str(values[patient_column]),
                    slide_column: slide,
                    "tile_id": str(ids[int(original_position)]),
                    "tile_index": int(original_position),
                    "within_slide_draw_index": int(draw_index),
                }
            )

    columns = [
        subcohort_column,
        patient_column,
        slide_column,
        "tile_id",
        "tile_index",
        "within_slide_draw_index",
        "sample_index",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame.from_records(records)
    result = result.sort_values(
        [subcohort_column, patient_column, slide_column, "tile_id"], kind="stable"
    ).reset_index(drop=True)
    result["sample_index"] = np.arange(len(result), dtype=np.int64)
    return result[columns]


def pca64_parameters(*, seed: int = VOCABULARY_SEED) -> dict[str, Any]:
    """Return the complete, unwhitened randomized PCA contract."""

    return {
        "n_components": N_PCA_COMPONENTS,
        "whiten": False,
        "svd_solver": "randomized",
        "random_state": int(seed),
    }


def lloyd_kmeans_parameters(
    n_prototypes: int = N_REFERENCE_PROTOTYPES,
    *,
    seed: int = VOCABULARY_SEED,
) -> dict[str, Any]:
    """Return the complete Lloyd k-means contract."""

    if int(n_prototypes) != n_prototypes or n_prototypes <= 0:
        raise ValueError("n_prototypes must be a positive integer")
    return {
        "n_clusters": int(n_prototypes),
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4,
        "algorithm": "lloyd",
        "random_state": int(seed),
    }


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize rows, leaving an all-zero row all zero."""

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.float32(1e-12))


@dataclass(frozen=True)
class V14Vocabulary:
    """A frozen PCA projection and centroids in its unwhitened coordinates."""

    centroids: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        centroids = np.asarray(self.centroids)
        mean = np.asarray(self.pca_mean)
        components = np.asarray(self.pca_components)
        if centroids.ndim != 2 or components.ndim != 2 or mean.ndim != 1:
            raise ValueError("vocabulary arrays have invalid dimensions")
        if centroids.shape[1] != components.shape[0]:
            raise ValueError("centroid and PCA component dimensions do not match")
        if components.shape[1] != mean.shape[0]:
            raise ValueError("PCA components and mean feature dimensions do not match")
        if len(centroids) == 0:
            raise ValueError("a vocabulary must contain at least one centroid")

    @property
    def n_prototypes(self) -> int:
        return int(np.asarray(self.centroids).shape[0])

    @property
    def n_features(self) -> int:
        return int(np.asarray(self.pca_mean).shape[0])

    def project(self, features: np.ndarray) -> np.ndarray:
        """L2-normalize features and apply the frozen PCA transform."""

        matrix = np.asarray(features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features:
            raise ValueError(
                f"features must have shape [n, {self.n_features}], got {matrix.shape}"
            )
        normalized = l2_normalize_rows(matrix)
        return (normalized - np.asarray(self.pca_mean)) @ np.asarray(
            self.pca_components
        ).T

    def assign_with_distances(
        self, features: np.ndarray, *, batch_size: int = 65_536
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return nearest prototype IDs and Euclidean PCA-space distances."""

        if int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        projected = self.project(features)
        labels = np.empty(len(projected), dtype=np.int16)
        distances = np.empty(len(projected), dtype=np.float32)
        centers = np.asarray(self.centroids, dtype=np.float32)
        center_norm = np.sum(centers * centers, axis=1)
        for start in range(0, len(projected), int(batch_size)):
            stop = min(start + int(batch_size), len(projected))
            block = np.asarray(projected[start:stop], dtype=np.float32)
            squared = (
                np.sum(block * block, axis=1, keepdims=True)
                + center_norm[None, :]
                - 2.0 * block @ centers.T
            )
            np.maximum(squared, 0.0, out=squared)
            block_labels = np.argmin(squared, axis=1)
            labels[start:stop] = block_labels.astype(np.int16)
            distances[start:stop] = np.sqrt(
                squared[np.arange(len(block_labels)), block_labels]
            ).astype(np.float32)
        return labels, distances

    def assign(self, features: np.ndarray, *, batch_size: int = 65_536) -> np.ndarray:
        """Return nearest integer prototype IDs."""

        labels, _ = self.assign_with_distances(features, batch_size=batch_size)
        return labels

    def backprojected_centroids(self) -> np.ndarray:
        """PCA-inverse-transform centroids and L2-normalize them."""

        restored = np.asarray(self.centroids) @ np.asarray(self.pca_components)
        restored = restored + np.asarray(self.pca_mean)[None, :]
        return l2_normalize_rows(restored)


def fit_v14_vocabulary(
    sample: np.ndarray,
    *,
    n_prototypes: int = N_REFERENCE_PROTOTYPES,
    seed: int = VOCABULARY_SEED,
    config: Mapping[str, Any] | None = None,
) -> V14Vocabulary:
    """Fit the fixed L2 -> randomized PCA64 -> Lloyd k-means pipeline."""

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    matrix = np.asarray(sample, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < N_PCA_COMPONENTS:
        raise ValueError(f"sample must be a 2D matrix with at least {N_PCA_COMPONENTS} rows")
    if matrix.shape[1] < N_PCA_COMPONENTS:
        raise ValueError(f"sample must have at least {N_PCA_COMPONENTS} feature columns")
    normalized = l2_normalize_rows(matrix)
    pca = PCA(**pca64_parameters(seed=seed))
    # Randomized PCA's fit_transform returns the approximate truncated U*S
    # scores, whereas transform performs the exact projection onto the fitted
    # components.  All future assignment uses transform(), so train k-means in
    # that same coordinate system.
    pca.fit(normalized)
    projected = np.asarray(pca.transform(normalized), dtype=np.float32)
    model = KMeans(**lloyd_kmeans_parameters(n_prototypes, seed=seed)).fit(projected)
    metadata = {
        "normalize": "l2",
        "pca": pca64_parameters(seed=seed),
        "kmeans": lloyd_kmeans_parameters(n_prototypes, seed=seed),
        "n_sample_tiles": int(len(matrix)),
        "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
        "inertia": float(model.inertia_),
        "n_iter": int(model.n_iter_),
        "cluster_sizes": np.bincount(
            model.labels_, minlength=int(n_prototypes)
        ).astype(int).tolist(),
        **dict(config or {}),
    }
    return V14Vocabulary(
        centroids=np.asarray(model.cluster_centers_, dtype=np.float32),
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        config=metadata,
    )


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise pairwise cosine similarities, with zero rows mapped to 0."""

    a = l2_normalize_rows(left)
    b = l2_normalize_rows(right)
    if a.shape[1] != b.shape[1]:
        raise ValueError("cosine matrices must share a feature dimension")
    return np.asarray(a @ b.T, dtype=np.float64)


def maximum_cosine_hungarian_mapping(
    source_centroids: np.ndarray,
    reference_centroids: np.ndarray,
    *,
    source_prototype_ids: Sequence[int] | None = None,
    reference_prototype_ids: Sequence[int] | None = None,
    name_threshold: float = NAME_MAPPING_COSINE,
) -> pd.DataFrame:
    """Solve a maximum-cosine one-to-one source-to-reference correspondence.

    Inputs must already be in the same feature space (normally inverse-PCA,
    L2-normalized UNI-v1 space).  Rectangular inventories are allowed; any
    unmatched source row receives a nullable reference ID and is never name
    mappable.  IDs are sorted before matching to make tie handling canonical.
    """

    source = np.asarray(source_centroids, dtype=np.float32)
    reference = np.asarray(reference_centroids, dtype=np.float32)
    if source.ndim != 2 or reference.ndim != 2:
        raise ValueError("centroid inputs must be two-dimensional")
    if source.shape[1] != reference.shape[1]:
        raise ValueError("source and reference centroids have different dimensions")
    if not 0 <= name_threshold <= 1:
        raise ValueError("name_threshold must lie in [0, 1]")

    source_ids = np.asarray(
        list(range(len(source))) if source_prototype_ids is None else source_prototype_ids,
        dtype=np.int64,
    )
    reference_ids = np.asarray(
        list(range(len(reference)))
        if reference_prototype_ids is None
        else reference_prototype_ids,
        dtype=np.int64,
    )
    if len(source_ids) != len(source) or len(reference_ids) != len(reference):
        raise ValueError("prototype-ID inventories do not match centroid rows")
    if len(np.unique(source_ids)) != len(source_ids) or len(np.unique(reference_ids)) != len(
        reference_ids
    ):
        raise ValueError("prototype identifiers must be unique")

    source_order = np.argsort(source_ids, kind="stable")
    reference_order = np.argsort(reference_ids, kind="stable")
    source = source[source_order]
    reference = reference[reference_order]
    source_ids = source_ids[source_order]
    reference_ids = reference_ids[reference_order]
    similarities = cosine_similarity_matrix(source, reference)
    matched_rows, matched_columns = linear_sum_assignment(-similarities)
    matches = {
        int(row): (int(column), float(similarities[row, column]))
        for row, column in zip(matched_rows, matched_columns, strict=True)
    }

    records: list[dict[str, Any]] = []
    for row, source_id in enumerate(source_ids):
        if row in matches:
            column, similarity = matches[row]
            reference_id: int | None = int(reference_ids[column])
        else:
            reference_id = None
            similarity = float("nan")
        records.append(
            {
                "source_prototype_id": int(source_id),
                "reference_prototype_id": reference_id,
                "cosine_similarity": similarity,
                "name_mappable": bool(
                    reference_id is not None and similarity >= float(name_threshold)
                ),
            }
        )
    result = pd.DataFrame.from_records(records)
    result["reference_prototype_id"] = result["reference_prototype_id"].astype("Int64")
    return result


def map_vocabulary_to_reference(
    source: V14Vocabulary,
    reference: V14Vocabulary,
    *,
    name_threshold: float = NAME_MAPPING_COSINE,
) -> pd.DataFrame:
    """Back-project, renormalize, and Hungarian-map a vocabulary to reference."""

    return maximum_cosine_hungarian_mapping(
        source.backprojected_centroids(),
        reference.backprojected_centroids(),
        name_threshold=name_threshold,
    )


def slide_abundance(labels: np.ndarray, n_prototypes: int) -> np.ndarray:
    """Return the tile-assignment mass of every prototype on one slide."""

    values = np.asarray(labels)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array")
    if int(n_prototypes) != n_prototypes or n_prototypes <= 0:
        raise ValueError("n_prototypes must be a positive integer")
    if len(values) and ((values < 0).any() or (values >= n_prototypes).any()):
        raise ValueError("a prototype label lies outside the requested inventory")
    counts = np.bincount(values.astype(np.int64), minlength=int(n_prototypes)).astype(
        np.float64
    )
    return counts / len(values) if len(values) else counts


@dataclass(frozen=True)
class PatientProfiles:
    """Equal-slide patient profiles in lexical patient order."""

    patient_ids: np.ndarray
    profiles: np.ndarray
    n_slides: np.ndarray


def equal_slide_patient_profiles(
    slide_profiles: np.ndarray, patient_ids: Sequence[Any]
) -> PatientProfiles:
    """Average slide profile rows equally within patient.

    No tile-count weighting occurs here: a patient with two slides contributes
    the arithmetic mean of those two already-normalized slide vectors.
    """

    matrix = np.asarray(slide_profiles, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("slide_profiles must be a two-dimensional matrix")
    patients = np.asarray([str(value) for value in patient_ids], dtype=object)
    if len(patients) != len(matrix):
        raise ValueError("patient_ids length does not match slide profile rows")
    ordered_patients = np.asarray(sorted(set(patients.tolist())), dtype=object)
    profiles = np.empty((len(ordered_patients), matrix.shape[1]), dtype=np.float64)
    n_slides = np.empty(len(ordered_patients), dtype=np.int64)
    for index, patient in enumerate(ordered_patients):
        mask = patients == patient
        profiles[index] = matrix[mask].mean(axis=0)
        n_slides[index] = int(mask.sum())
    return PatientProfiles(ordered_patients, profiles, n_slides)


def zero_based_prototype_index(
    prototype_id: int, prototype_ids: Sequence[int]
) -> int:
    """Return a prototype's zero-based position in the sorted ID inventory."""

    inventory = sorted({int(value) for value in prototype_ids})
    if len(inventory) != len(prototype_ids):
        raise ValueError("prototype inventory contains duplicates")
    try:
        return inventory.index(int(prototype_id))
    except ValueError as error:
        raise ValueError(f"prototype {prototype_id} is absent from the inventory") from error


def closest_per_patient_candidates(
    assignments: pd.DataFrame,
    *,
    patient_column: str = "patient_id",
    subcohort_column: str = "subcohort",
    slide_column: str = "slide_id",
    tile_column: str = "tile_id",
    distance_column: str = "distance",
) -> pd.DataFrame:
    """Retain each patient's closest assigned tile and label expansion tiers.

    The decile and quartile thresholds are computed from *all tile assignments*
    supplied for the prototype using NumPy's ``method='linear'`` quantile.
    Patient representatives are then the closest tile per patient, with lexical
    slide and tile identifiers breaking exact distance ties.
    """

    required = {
        patient_column,
        subcohort_column,
        slide_column,
        tile_column,
        distance_column,
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"assignments are missing required columns: {sorted(missing)}")
    if assignments.empty:
        result = assignments.copy()
        result["eligibility_tier"] = pd.Series(dtype=np.int8)
        result["eligibility_stage"] = pd.Series(dtype=object)
        return result

    frame = assignments.copy()
    for column in (patient_column, subcohort_column, slide_column, tile_column):
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing identifiers")
        frame[column] = frame[column].astype(str)
    distances = pd.to_numeric(frame[distance_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("distances must be finite and nonnegative")
    frame[distance_column] = distances
    patient_groups = frame.groupby(patient_column, sort=False)[subcohort_column].nunique()
    if (patient_groups > 1).any():
        raise ValueError("a patient cannot belong to multiple subcohorts")

    q10, q25 = np.quantile(distances, [0.10, 0.25], method="linear")
    frame = frame.sort_values(
        [patient_column, distance_column, slide_column, tile_column], kind="stable"
    )
    closest = frame.drop_duplicates(patient_column, keep="first").copy()
    closest["eligibility_tier"] = np.select(
        [closest[distance_column] <= q10, closest[distance_column] <= q25],
        [0, 1],
        default=2,
    ).astype(np.int8)
    closest["eligibility_stage"] = closest["eligibility_tier"].map(
        dict(enumerate(_MONTAGE_STAGES))
    )
    closest["prototype_q10_distance"] = float(q10)
    closest["prototype_q25_distance"] = float(q25)
    return closest.sort_values(
        [distance_column, patient_column, slide_column, tile_column], kind="stable"
    ).reset_index(drop=True)


@dataclass(frozen=True)
class MontageSelection:
    """One deterministic montage occurrence and its audit metadata."""

    tiles: pd.DataFrame
    stage: str
    seed: int
    support_status: str
    n_available_patients: int
    n_eligible_patients: int
    overlap_count: int


def _validate_subcohort_order(subcohort_order: Sequence[Any]) -> tuple[str, str, str, str]:
    groups = tuple(str(value) for value in subcohort_order)
    if len(groups) != 4 or len(set(groups)) != 4:
        raise ValueError("subcohort_order must contain exactly four unique subcohorts")
    if groups != tuple(sorted(groups)):
        raise ValueError("subcohort_order must be the sealed lexicographic order")
    return groups  # type: ignore[return-value]


def _montage_quotas(
    capacities: Mapping[str, int],
    total: int,
    groups: Sequence[str],
    *,
    existing: Mapping[str, int] | None = None,
) -> dict[str, int]:
    """Target three/group, then redistribute by largest unused pool."""

    allocated = {group: 0 for group in groups}
    current = {group: int((existing or {}).get(group, 0)) for group in groups}
    remaining = min(int(total), sum(int(capacities.get(group, 0)) for group in groups))
    for group in groups:
        wanted = max(0, MONTAGE_PATIENTS_PER_SUBCOHORT - current[group])
        add = min(wanted, int(capacities.get(group, 0)), remaining)
        allocated[group] += add
        remaining -= add
    while remaining:
        available = {
            group: int(capacities.get(group, 0)) - allocated[group]
            for group in groups
            if allocated[group] < int(capacities.get(group, 0))
        }
        if not available:  # pragma: no cover - guarded by remaining definition
            raise AssertionError("montage quota exceeded candidate capacity")
        largest = max(available.values())
        group = next(group for group in groups if available.get(group) == largest)
        allocated[group] += 1
        remaining -= 1
    return allocated


def _sample_candidate_rows(
    pool: pd.DataFrame,
    quotas: Mapping[str, int],
    groups: Sequence[str],
    rng: np.random.Generator,
    *,
    subcohort_column: str,
    distance_column: str,
    patient_column: str,
    slide_column: str,
    tile_column: str,
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for group in groups:
        count = int(quotas.get(group, 0))
        if count == 0:
            continue
        candidates = pool.loc[pool[subcohort_column].eq(group)].sort_values(
            [distance_column, patient_column, slide_column, tile_column], kind="stable"
        )
        positions = np.asarray(
            rng.choice(len(candidates), size=count, replace=False), dtype=np.int64
        )
        blocks.append(candidates.iloc[positions].copy())
    if not blocks:
        return pool.iloc[0:0].copy()
    chosen = pd.concat(blocks, ignore_index=True)
    return chosen.iloc[np.asarray(rng.permutation(len(chosen)), dtype=np.int64)].reset_index(
        drop=True
    )


def select_montage_tiles(
    candidates: pd.DataFrame,
    *,
    prototype_index: int,
    occurrence: int,
    subcohort_order: Sequence[Any],
    avoid_patient_ids: Sequence[Any] = (),
    seed_base: int = VOCABULARY_SEED,
    total_tiles: int = MONTAGE_TILES,
    patient_column: str = "patient_id",
    subcohort_column: str = "subcohort",
    slide_column: str = "slide_id",
    tile_column: str = "tile_id",
    distance_column: str = "distance",
) -> MontageSelection:
    """Select one balanced montage, optionally minimizing prior-patient overlap.

    Candidate rows must come from :func:`closest_per_patient_candidates`.  The
    smallest decile/quartile/all tier able to supply the montage is used.  With
    an avoidance set (the first duplicate occurrence), the smallest tier able
    to supply ``total_tiles`` *nonoverlapping* patients is used; if none can,
    the all-patient tier is used and every possible nonoverlapping patient is
    selected before the mathematically unavoidable overlaps.
    """

    groups = _validate_subcohort_order(subcohort_order)
    if int(prototype_index) != prototype_index or prototype_index < 0:
        raise ValueError("prototype_index must be the nonnegative zero-based sorted index")
    if int(occurrence) != occurrence or occurrence < 0:
        raise ValueError("occurrence must be a nonnegative integer")
    if int(total_tiles) != total_tiles or total_tiles <= 0:
        raise ValueError("total_tiles must be a positive integer")
    required = {
        patient_column,
        subcohort_column,
        slide_column,
        tile_column,
        distance_column,
        "eligibility_tier",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates are missing required columns: {sorted(missing)}")

    frame = candidates.copy()
    for column in (patient_column, subcohort_column, slide_column, tile_column):
        frame[column] = frame[column].astype(str)
    if frame[patient_column].duplicated().any():
        raise ValueError("candidates contain more than one tile for a patient")
    unknown_groups = sorted(set(frame[subcohort_column]) - set(groups))
    if unknown_groups:
        raise ValueError(f"candidate subcohorts are outside the sealed order: {unknown_groups}")
    tiers = pd.to_numeric(frame["eligibility_tier"], errors="coerce")
    if tiers.isna().any() or (~tiers.isin([0, 1, 2])).any():
        raise ValueError("eligibility_tier must contain only 0, 1, or 2")
    frame["eligibility_tier"] = tiers.astype(np.int8)

    avoid = {str(value) for value in avoid_patient_ids}
    chosen_tier = 2
    for tier in range(3):
        pool = frame.loc[frame["eligibility_tier"] <= tier]
        available = pool.loc[~pool[patient_column].isin(avoid)] if avoid else pool
        if len(available) >= int(total_tiles):
            chosen_tier = tier
            break
    pool = frame.loc[frame["eligibility_tier"] <= chosen_tier].copy()
    target = min(int(total_tiles), len(pool))
    nonoverlap = pool.loc[~pool[patient_column].isin(avoid)].copy()
    overlap = pool.loc[pool[patient_column].isin(avoid)].copy()
    seed = int(seed_base) + 100 * int(prototype_index) + int(occurrence)
    rng = pcg64_rng(seed)

    if len(nonoverlap) >= target:
        capacities = nonoverlap[subcohort_column].value_counts().to_dict()
        quotas = _montage_quotas(capacities, target, groups)
        selected = _sample_candidate_rows(
            nonoverlap,
            quotas,
            groups,
            rng,
            subcohort_column=subcohort_column,
            distance_column=distance_column,
            patient_column=patient_column,
            slide_column=slide_column,
            tile_column=tile_column,
        )
    else:
        # Every nonoverlap row is required for the minimum possible overlap.
        selected_nonoverlap = nonoverlap.copy()
        current = (
            selected_nonoverlap[subcohort_column]
            .value_counts()
            .reindex(groups, fill_value=0)
            .astype(int)
            .to_dict()
        )
        remaining = target - len(selected_nonoverlap)
        overlap_capacity = overlap[subcohort_column].value_counts().to_dict()
        overlap_quotas = _montage_quotas(
            overlap_capacity, remaining, groups, existing=current
        )
        selected_overlap = _sample_candidate_rows(
            overlap,
            overlap_quotas,
            groups,
            rng,
            subcohort_column=subcohort_column,
            distance_column=distance_column,
            patient_column=patient_column,
            slide_column=slide_column,
            tile_column=tile_column,
        )
        selected = pd.concat([selected_nonoverlap, selected_overlap], ignore_index=True)
        if len(selected):
            selected = selected.iloc[
                np.asarray(rng.permutation(len(selected)), dtype=np.int64)
            ].reset_index(drop=True)

    if selected[patient_column].duplicated().any():
        raise AssertionError("montage selection repeated a patient")
    selected["montage_slot"] = np.arange(len(selected), dtype=np.int64)
    selected["prototype_index"] = int(prototype_index)
    selected["occurrence"] = int(occurrence)
    selected["selection_seed"] = seed
    selected["selection_stage"] = _MONTAGE_STAGES[chosen_tier]
    selected["overlaps_avoidance_set"] = selected[patient_column].isin(avoid)

    overlap_count = int(selected[patient_column].isin(avoid).sum())
    support = (
        "MONTAGE_SUPPORT_SUFFICIENT"
        if len(frame) >= int(total_tiles)
        else "MONTAGE_SUPPORT_INSUFFICIENT"
    )
    return MontageSelection(
        tiles=selected,
        stage=_MONTAGE_STAGES[chosen_tier],
        seed=seed,
        support_status=support,
        n_available_patients=int(len(frame)),
        n_eligible_patients=int(len(pool)),
        overlap_count=overlap_count,
    )


def select_duplicate_montages(
    candidates: pd.DataFrame,
    *,
    prototype_index: int,
    subcohort_order: Sequence[Any],
    seed_base: int = VOCABULARY_SEED,
    **columns: Any,
) -> tuple[MontageSelection, MontageSelection]:
    """Select original and duplicate occurrences with minimum patient overlap."""

    original = select_montage_tiles(
        candidates,
        prototype_index=prototype_index,
        occurrence=0,
        subcohort_order=subcohort_order,
        seed_base=seed_base,
        **columns,
    )
    patient_column = str(columns.get("patient_column", "patient_id"))
    duplicate = select_montage_tiles(
        candidates,
        prototype_index=prototype_index,
        occurrence=1,
        subcohort_order=subcohort_order,
        avoid_patient_ids=original.tiles[patient_column].tolist(),
        seed_base=seed_base,
        **columns,
    )
    expected_minimum = max(0, 2 * min(MONTAGE_TILES, len(candidates)) - len(candidates))
    if duplicate.overlap_count != expected_minimum:
        raise AssertionError(
            "duplicate montage did not attain the minimum possible patient overlap"
        )
    return original, duplicate


class HMACCodeExhaustedError(RuntimeError):
    """All eight complete six-character windows collided."""


def choose_hmac_code(encoded_digest: str, used_codes: set[str]) -> tuple[str, int]:
    """Choose the first unused nonoverlapping six-character digest window."""

    encoded = str(encoded_digest)
    valid = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
    if encoded != encoded.upper() or any(character not in valid for character in encoded):
        raise ValueError("encoded_digest must be unpadded uppercase RFC 4648 Base32")
    for window_index in range(8):
        start = 6 * window_index
        code = encoded[start : start + 6]
        if len(code) == 6 and code not in used_codes:
            return code, window_index
    raise HMACCodeExhaustedError("all eight complete HMAC code windows collided")


def _hmac_sha256(salt: bytes, message: str) -> bytes:
    return hmac.new(salt, message.encode("utf-8"), hashlib.sha256).digest()


def _unpadded_base32(payload: bytes) -> str:
    return base64.b32encode(payload).decode("ascii").rstrip("=")


def blinded_occurrences(
    prototype_ids: Sequence[int], duplicated_prototype_ids: Sequence[int]
) -> list[tuple[int, int]]:
    """Build the canonical occurrence inventory (all originals, then duplicates)."""

    prototypes = sorted({int(value) for value in prototype_ids})
    if len(prototypes) != len(prototype_ids):
        raise ValueError("prototype inventory contains duplicates")
    duplicates = sorted({int(value) for value in duplicated_prototype_ids})
    if len(duplicates) != len(duplicated_prototype_ids):
        raise ValueError("duplicate-prototype inventory contains duplicates")
    if not set(duplicates).issubset(prototypes):
        raise ValueError("a duplicate prototype is absent from the canonical inventory")
    return [(prototype, 0) for prototype in prototypes] + [
        (prototype, 1) for prototype in duplicates
    ]


def hmac_blinding_table(
    occurrences: Sequence[tuple[int, int]], salt: bytes
) -> pd.DataFrame:
    """Assign unique six-character codes and deterministic presentation order.

    Code collision resolution is performed in ascending
    ``(prototype_id, occurrence)`` order.  The returned table itself is in the
    independently HMAC-derived presentation order and therefore belongs in the
    embargoed key, not in the public reader packet.
    """

    if not isinstance(salt, bytes) or len(salt) != 32:
        raise ValueError("salt must contain exactly 32 bytes")
    normalized = [(int(prototype), int(occurrence)) for prototype, occurrence in occurrences]
    if any(prototype < 0 or occurrence < 0 for prototype, occurrence in normalized):
        raise ValueError("prototype IDs and occurrence numbers must be nonnegative")
    if len(set(normalized)) != len(normalized):
        raise ValueError("occurrence inventory contains duplicates")

    used: set[str] = set()
    records: list[dict[str, Any]] = []
    for prototype, occurrence in sorted(normalized):
        code_digest = _hmac_sha256(salt, f"code|{prototype}|{occurrence}")
        code, window_index = choose_hmac_code(_unpadded_base32(code_digest), used)
        used.add(code)
        order_digest = _hmac_sha256(salt, f"order|{prototype}|{occurrence}")
        records.append(
            {
                "prototype_id": prototype,
                "occurrence": occurrence,
                "code": code,
                "code_window_index": window_index,
                "order_digest_hex": order_digest.hex(),
                "is_controlling_read": occurrence == 0,
                "_order_digest": order_digest,
            }
        )

    records.sort(
        key=lambda record: (
            record["_order_digest"],
            record["prototype_id"],
            record["occurrence"],
        )
    )
    for position, record in enumerate(records):
        record["presentation_order"] = position
        del record["_order_digest"]
    columns = [
        "presentation_order",
        "prototype_id",
        "occurrence",
        "code",
        "code_window_index",
        "is_controlling_read",
        "order_digest_hex",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def draw_collision_free_hmac_salt(
    occurrences: Sequence[tuple[int, int]],
    *,
    token_bytes: Callable[[int], bytes] = secrets.token_bytes,
    max_attempts: int = 1_000,
) -> tuple[bytes, pd.DataFrame]:
    """Draw a 256-bit OS-random salt, redrawing only on window exhaustion."""

    if int(max_attempts) != max_attempts or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    for _ in range(int(max_attempts)):
        salt = token_bytes(32)
        if not isinstance(salt, bytes) or len(salt) != 32:
            raise ValueError("token_bytes must return exactly the requested byte count")
        try:
            return salt, hmac_blinding_table(occurrences, salt)
        except HMACCodeExhaustedError:
            continue
    raise HMACCodeExhaustedError("could not draw a collision-free salt")


def salt_sha256(salt: bytes) -> str:
    """Return the only salt-derived value allowed in the public manifest."""

    if not isinstance(salt, bytes) or len(salt) != 32:
        raise ValueError("salt must contain exactly 32 bytes")
    return hashlib.sha256(salt).hexdigest()
