#!/usr/bin/env python3
"""Audit the disconnected SR386, SR1482, and RIH UNI-v1 UMAP islands."""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import sqlite3
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu  # noqa: E402
from sklearn.cluster import DBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

COORDINATES = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides/coordinates_univ1.parquet"
)
SLIDE_MEANS = Path(
    "/mnt/wsl/oceanpath-hot/outputs/eval/cohort_umap_all_slides/slide_means_univ1.npz"
)
H5_DIR = Path(
    "/mnt/wsl/oceanpath-hot/features/colon_stream/20x_256px_0px_overlap_mpp0.5/features_uni_v1"
)
THUMBNAIL_DIR = Path("/mnt/wsl/oceanpath-hot/features/colon_stream/thumbnails")
QUEUE_DB = Path("/home/yc_liu/projects/OceanPath-colon/outputs/colon_stream/queue.sqlite")
MANIFEST = Path("/mnt/d/YC.Liu/manifests/colon/crc_final_v4.csv")
SURGEN_STATE = Path("/mnt/d/YC.Liu/slides/colon/fix_surgen_state.jsonl")
RIH_WORKBOOK = Path("/mnt/d/YC.Liu/manifests/colon/RIH_clinical_reconciliation.xlsx")
OUTPUT_DIR = Path("/mnt/wsl/oceanpath-hot/outputs/eval/univ1_umap_separation_audit")

SUBCOHORTS = ("SR386", "SR1482", "RIH-Colon")
EXPECTED_COUNTS = {"SR386": 427, "SR1482": 593, "RIH-Colon": 369}
DBSCAN_EPS = 0.4
INK = "#1b1b1a"
MUTED = "#696860"
NEGATIVE_COLOR = "#2878b5"
POSITIVE_COLOR = "#d95f02"
SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
OFFICE_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _thumbnail_hue(path: Path) -> tuple[float, int]:
    """Return median tissue hue using the frozen HEST thumbnail."""
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail(
        (192, 192),
        resample=Image.Resampling.BICUBIC,
        reducing_gap=2.0,
    )
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (maximum - minimum) / (maximum + 1e-6)
    mask = (maximum < 0.94) & (saturation > 0.06)
    if int(mask.sum()) < 50:
        mask = maximum < 0.90
    if int(mask.sum()) < 50:
        raise ValueError(f"Thumbnail has too few tissue-like pixels: {path}")
    return float(np.median(hsv[:, :, 0][mask])), int(mask.sum())


def _load_h5_and_thumbnail_stats(slide_ids: pd.Series) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, slide_id in enumerate(slide_ids.astype(str), start=1):
        h5_path = H5_DIR / f"{slide_id}.h5"
        thumbnail_path = THUMBNAIL_DIR / f"{slide_id}.jpg"
        if not h5_path.is_file() or not thumbnail_path.is_file():
            raise FileNotFoundError(f"Missing H5/thumbnail for {slide_id}")
        with h5py.File(h5_path, "r") as handle:
            coordinates = handle["coords"]
            features = handle["features"]
            if len(coordinates) != len(features):
                raise ValueError(f"Coordinate/feature count mismatch: {slide_id}")
            attrs = coordinates.attrs
            patch_count = len(features)
            width = int(attrs["level0_width"])
            height = int(attrs["level0_height"])
            native_mpp = float(attrs["level0_mpp"])
            patch_level0 = int(attrs["patch_size_level0"])
            sampling_mode = str(attrs["sampling_mode"])
            target_mpp = float(attrs["target_mpp"])
        hue, hue_pixels = _thumbnail_hue(thumbnail_path)
        rows.append(
            {
                "slide_id": slide_id,
                "patch_count": patch_count,
                "level0_width": width,
                "level0_height": height,
                "native_mpp": native_mpp,
                "patch_size_level0": patch_level0,
                "sampling_mode": sampling_mode,
                "target_mpp": target_mpp,
                "thumbnail_hue_median": hue,
                "thumbnail_tissue_pixels": hue_pixels,
            }
        )
        if index % 250 == 0:
            print(f"Read H5/thumbnail metadata for {index:,}/{len(slide_ids):,} slides", flush=True)
    return pd.DataFrame(rows)


def _load_surgen_state() -> pd.DataFrame:
    records = [json.loads(line) for line in SURGEN_STATE.read_text().splitlines() if line.strip()]
    state = pd.DataFrame(records)
    state = state[state["status"].eq("done")].drop_duplicates("file", keep="last")
    state["slide_id"] = state["file"].str.replace(".tiff", "", regex=False)
    return state[["slide_id", "ssim_min", "chan_dmean_max"]]


def _read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read a worksheet without adding an openpyxl runtime dependency."""
    with ZipFile(path) as archive:
        try:
            shared_root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = [
                "".join(node.text or "" for node in item.findall(f".//{SHEET_NS}t"))
                for item in shared_root
            ]
        except KeyError:
            shared = []
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        matches = [
            sheet
            for sheet in workbook.findall(f".//{SHEET_NS}sheet")
            if sheet.attrib.get("name") == sheet_name
        ]
        if len(matches) != 1:
            raise ValueError(f"No unique worksheet {sheet_name!r} in {path}")
        relationship_id = matches[0].attrib[OFFICE_REL_ID]
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = [
            relation.attrib["Target"]
            for relation in relationships.findall(f"{REL_NS}Relationship")
            if relation.attrib.get("Id") == relationship_id
        ]
        if len(targets) != 1:
            raise ValueError(f"No unique relationship for worksheet {sheet_name!r}")
        target = targets[0]
        worksheet = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath(posixpath.join("xl", target))
        )
        root = ElementTree.fromstring(archive.read(worksheet))

    rows: list[dict[str, str]] = []
    for row in root.findall(f".//{SHEET_NS}row"):
        values: dict[str, str] = {}
        for cell in row.findall(f"{SHEET_NS}c"):
            reference = cell.attrib.get("r", "")
            match = re.match(r"[A-Z]+", reference)
            if match is None:
                continue
            column = match.group()
            value_node = cell.find(f"{SHEET_NS}v")
            value = "" if value_node is None or value_node.text is None else value_node.text
            if cell.attrib.get("t") == "s":
                value = shared[int(value)]
            elif cell.attrib.get("t") == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall(f".//{SHEET_NS}t"))
            values[column] = value
        rows.append(values)
    columns = sorted(
        set().union(*(row.keys() for row in rows)),
        key=lambda value: (len(value), value),
    )
    headers = [rows[0].get(column, "") for column in columns]
    return pd.DataFrame(
        [[row.get(column, "") for column in columns] for row in rows[1:]],
        columns=headers,
    )


def _load_rih_source_metadata() -> pd.DataFrame:
    source = _read_xlsx_sheet(RIH_WORKBOOK, "Raw_Source_1")[["De-ID", "Type", "Biopsy"]]
    for column in ("De-ID", "Type", "Biopsy"):
        source[column] = source[column].astype("string").str.strip().replace("", pd.NA)
    source = source.dropna(subset=["De-ID"])
    if len(source) != 301:
        raise ValueError(f"Unexpected RIH Raw_Source_1 row count: {len(source)}")
    if source["De-ID"].duplicated().any():
        raise ValueError("RIH Raw_Source_1 contains duplicate nonblank De-IDs")
    source = source.rename(
        columns={
            "De-ID": "slide_id",
            "Type": "rih_source_type",
            "Biopsy": "rih_source_biopsy",
        }
    )
    numeric_biopsy = pd.to_numeric(source["rih_source_biopsy"], errors="coerce")
    invalid_biopsy = source["rih_source_biopsy"].notna() & ~numeric_biopsy.isin([0, 1])
    if invalid_biopsy.any():
        invalid = source.loc[invalid_biopsy, "rih_source_biopsy"].unique().tolist()
        raise ValueError(f"Unexpected RIH Biopsy encodings: {invalid}")
    source["rih_source_biopsy_flag"] = numeric_biopsy.map({1: True, 0: False})
    return source


def assemble_frame() -> pd.DataFrame:
    coordinates = pd.read_parquet(COORDINATES)
    required_coordinate_columns = {"slide_id", "cohort", "n_patches", "umap_1", "umap_2"}
    missing_coordinate_columns = required_coordinate_columns - set(coordinates.columns)
    if missing_coordinate_columns:
        raise ValueError(f"Coordinate table missing columns: {sorted(missing_coordinate_columns)}")
    if coordinates["slide_id"].astype(str).duplicated().any():
        raise ValueError("Coordinate table contains duplicate slide IDs")
    if not np.isfinite(coordinates[["umap_1", "umap_2"]].to_numpy(dtype=float)).all():
        raise ValueError("Coordinate table contains non-finite UMAP values")
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    if manifest["output_id"].astype(str).duplicated().any():
        raise ValueError("Canonical manifest contains duplicate output IDs")
    manifest_columns = [
        "output_id",
        "patient_uid",
        "subcohort",
        "specimen_role",
        "tumor_site_group",
        "metastatic_site_group",
        "tumor_site_raw",
        "age_at_diagnosis",
        "sex",
        "image_format",
    ]
    manifest = manifest[manifest_columns]
    with sqlite3.connect(f"file:{QUEUE_DB}?mode=ro", uri=True) as connection:
        queue = pd.read_sql_query(
            "SELECT output_id, cohort, mpp, source_size, source_abspath, "
            "source_mtime_ns, readiness_json FROM stream_jobs",
            connection,
        )
    if queue[["output_id", "cohort"]].duplicated().any():
        raise ValueError("Queue contains duplicate output_id/cohort pairs")
    frame = coordinates.merge(
        queue,
        left_on=["slide_id", "cohort"],
        right_on=["output_id", "cohort"],
        how="left",
        validate="one_to_one",
        indicator="_queue_merge",
    )
    missing_queue = frame.loc[frame["_queue_merge"].ne("both"), "slide_id"].astype(str)
    if not missing_queue.empty:
        raise ValueError(f"Coordinates missing from queue: {missing_queue.head().tolist()}")
    frame = frame.drop(columns="_queue_merge")
    frame = frame.merge(manifest, on="output_id", how="left", validate="one_to_one")
    frame.loc[frame["subcohort"].isna() & frame["cohort"].eq("RIH"), "subcohort"] = "RIH-Colon"
    frame = frame[frame["subcohort"].isin(SUBCOHORTS)].copy()
    counts = frame["subcohort"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected subcohort counts: {counts}")

    stats = _load_h5_and_thumbnail_stats(frame["slide_id"])
    frame = frame.merge(stats, on="slide_id", validate="one_to_one")
    if not frame["patch_count"].eq(frame["n_patches"]).all():
        raise ValueError("Live H5 patch counts do not match the frozen UMAP coordinate table")
    if not np.allclose(frame["native_mpp"], frame["mpp"], rtol=0.0, atol=1e-6):
        raise ValueError("Live H5 native MPP does not match the frozen queue metadata")
    frame = frame.merge(_load_surgen_state(), on="slide_id", how="left", validate="one_to_one")
    frame = frame.merge(
        _load_rih_source_metadata(), on="slide_id", how="left", validate="one_to_one"
    )
    frame["slide_area_pixels"] = frame["level0_width"] * frame["level0_height"]
    frame["tissue_grid_occupancy"] = (
        frame["patch_count"] * frame["patch_size_level0"] ** 2 / frame["slide_area_pixels"]
    )
    frame["source_bytes_per_patch"] = frame["source_size"] / frame["patch_count"]
    frame["source_bytes_per_pixel"] = frame["source_size"] / frame["slide_area_pixels"]
    frame["slide_number"] = pd.to_numeric(
        frame["slide_id"].str.extract(r"(?:T|SL-)(\d+)")[0], errors="coerce"
    )
    frame["section_number"] = pd.to_numeric(
        frame["slide_id"].str.extract(r"_(\d+)$")[0], errors="coerce"
    )
    frame["sr1482_biopsy_text"] = (
        frame["tumor_site_raw"]
        .fillna("")
        .str.contains(r"biops|biosp", case=False, regex=True, na=False)
    )
    frame["sr1482_polyp_text"] = (
        frame["tumor_site_raw"].fillna("").str.contains("polyp", case=False, regex=False, na=False)
    )
    if not frame["sampling_mode"].eq("exact_mpp").all() or not np.allclose(
        frame["target_mpp"], 0.5
    ):
        raise ValueError("Relevant H5s do not share the expected exact-MPP extraction policy")
    return frame


def assign_visible_islands(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = frame.copy()
    frame["visible_island"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    audit: dict[str, Any] = {}
    for subcohort in SUBCOHORTS:
        selection = frame["subcohort"].eq(subcohort)
        subset = frame.loc[selection]
        labels = DBSCAN(eps=DBSCAN_EPS, min_samples=3).fit_predict(subset[["umap_1", "umap_2"]])
        frame.loc[selection, "dbscan_cluster"] = labels
        clustered = frame.loc[selection].copy()
        centroids = clustered.groupby("dbscan_cluster")[["umap_1", "umap_2"]].mean()
        sizes = clustered["dbscan_cluster"].value_counts().sort_index()
        audit[subcohort] = {
            "eps": DBSCAN_EPS,
            "min_samples": 3,
            "cluster_sizes": {str(int(key)): int(value) for key, value in sizes.items()},
            "centroids": {
                str(int(key)): {"umap_1": float(row.umap_1), "umap_2": float(row.umap_2)}
                for key, row in centroids.iterrows()
            },
        }
        if subcohort == "SR386":
            detached = int(centroids["umap_1"].idxmin())
            frame.loc[selection, "visible_island"] = np.where(
                labels == detached, "cool_stain_island", "main_stain_island"
            )
        elif subcohort == "SR1482":
            non_noise = centroids.drop(index=-1, errors="ignore")
            small = int(non_noise["umap_1"].idxmax())
            large = int(non_noise["umap_1"].idxmin())
            semantic = np.select(
                [labels == small, labels == large],
                ["small_fragment_island", "large_section_island"],
                default="atypical_outlier",
            )
            frame.loc[selection, "visible_island"] = semantic
        else:
            fixed_mpp = np.isclose(subset["native_mpp"], 0.5016)
            fixed_centroids = centroids.loc[
                sorted(set(labels[fixed_mpp]) - {-1}), ["umap_1", "umap_2"]
            ]
            large = int(fixed_centroids["umap_1"].idxmax())
            semantic = np.where(
                ~fixed_mpp,
                "different_mpp",
                np.where(labels == large, "large_section_island", "small_fragment_island"),
            )
            frame.loc[selection, "visible_island"] = semantic
    frame["dbscan_cluster"] = frame["dbscan_cluster"].astype(int)
    expected_semantic_counts = {
        "SR386": {"main_stain_island": 379, "cool_stain_island": 48},
        "SR1482": {
            "large_section_island": 443,
            "small_fragment_island": 149,
            "atypical_outlier": 1,
        },
        "RIH-Colon": {
            "large_section_island": 177,
            "small_fragment_island": 88,
            "different_mpp": 104,
        },
    }
    for subcohort, expected in expected_semantic_counts.items():
        observed = (
            frame.loc[frame["subcohort"].eq(subcohort), "visible_island"]
            .value_counts()
            .astype(int)
            .to_dict()
        )
        if observed != expected:
            raise ValueError(
                f"Frozen semantic clusters changed for {subcohort}: "
                f"observed={observed}, expected={expected}"
            )
        audit[subcohort]["semantic_sizes"] = observed
    return frame, audit


def _binary_rule_metrics(
    frame: pd.DataFrame,
    *,
    predictor: str,
    positive_island: str,
    negative_island: str,
) -> tuple[dict[str, Any], np.ndarray]:
    binary = frame[frame["visible_island"].isin([positive_island, negative_island])].copy()
    binary = binary[binary[predictor].notna()].copy()
    values = binary[predictor].to_numpy(dtype=float)
    target = binary["visible_island"].eq(positive_island).astype(int).to_numpy()
    groups = binary["patient_uid"].fillna(binary["slide_id"]).astype(str).to_numpy()
    model = DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42)
    model.fit(values[:, None], target)
    predictions = model.predict(values[:, None])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(values[:, None], target, groups))
    cross_validated = cross_val_predict(
        DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=42),
        values[:, None],
        target,
        cv=folds,
    )
    fold_balanced_accuracies = [
        balanced_accuracy_score(target[test_indices], cross_validated[test_indices])
        for _, test_indices in folds
    ]
    raw_auc = roc_auc_score(target, values)
    statistic = mannwhitneyu(values[target == 0], values[target == 1], alternative="two-sided")
    threshold = float(model.tree_.threshold[0])
    left_class = int(np.argmax(model.tree_.value[1][0]))
    operator = "<=" if left_class == 1 else ">"
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    metrics = {
        "predictor": predictor,
        "positive_island": positive_island,
        "negative_island": negative_island,
        "n": int(len(binary)),
        "n_patients": int(pd.Series(groups).nunique()),
        "negative_n": int((target == 0).sum()),
        "positive_n": int((target == 1).sum()),
        "threshold": threshold,
        "positive_rule_operator": operator,
        "roc_auc_oriented": float(max(raw_auc, 1.0 - raw_auc)),
        "mann_whitney_u": float(statistic.statistic),
        "mann_whitney_p": float(statistic.pvalue),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "cv_balanced_accuracy": float(balanced_accuracy_score(target, cross_validated)),
        "cv_fold_balanced_accuracy_mean": float(np.mean(fold_balanced_accuracies)),
        "cv_fold_balanced_accuracy_sd": float(np.std(fold_balanced_accuracies, ddof=1)),
        "accuracy": float((predictions == target).mean()),
        "confusion_matrix_actual_negative_positive": matrix.astype(int).tolist(),
        "negative_median": float(np.median(values[target == 0])),
        "positive_median": float(np.median(values[target == 1])),
    }
    predicted_all = model.predict(frame[[predictor]].to_numpy(dtype=float))
    return metrics, predicted_all


def _high_dimensional_confirmation(
    frame: pd.DataFrame,
    *,
    positive_island: str,
    negative_island: str,
) -> dict[str, float | int]:
    binary = frame[frame["visible_island"].isin([positive_island, negative_island])].copy()
    with np.load(SLIDE_MEANS, allow_pickle=False) as archive:
        slide_ids = archive["slide_ids"].astype(str)
        means = archive["means"]
        if len(np.unique(slide_ids)) != len(slide_ids):
            raise ValueError("Slide-mean archive contains duplicate slide IDs")
        if means.shape != (len(slide_ids), 1024) or not np.isfinite(means).all():
            raise ValueError(f"Invalid UNI-v1 slide-mean matrix: {means.shape}")
        index = {slide_id: position for position, slide_id in enumerate(slide_ids)}
        missing = sorted(set(binary["slide_id"].astype(str)) - set(index))
        if missing:
            raise ValueError(f"Slides missing from UNI-v1 mean archive: {missing[:5]}")
        embeddings = means[[index[slide_id] for slide_id in binary["slide_id"].astype(str)]].copy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("UNI-v1 slide means contain zero or non-finite norms")
    embeddings /= norms
    target = binary["visible_island"].eq(positive_island).astype(int).to_numpy()
    groups = binary["patient_uid"].fillna(binary["slide_id"]).astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(embeddings, target, groups))
    model = make_pipeline(
        PCA(n_components=50, random_state=42),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
    )
    predictions = cross_val_predict(model, embeddings, target, cv=folds)
    fold_balanced_accuracies = [
        balanced_accuracy_score(target[test_indices], predictions[test_indices])
        for _, test_indices in folds
    ]
    pca_embeddings = PCA(n_components=50, random_state=42).fit_transform(embeddings)
    unsupervised_labels = KMeans(n_clusters=2, n_init=50, random_state=42).fit_predict(
        pca_embeddings
    )
    return {
        "n": int(len(binary)),
        "n_patients": int(pd.Series(groups).nunique()),
        "cosine_silhouette": float(silhouette_score(embeddings, target, metric="cosine")),
        "pca50_logistic_cv_balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "pca50_logistic_cv_fold_balanced_accuracy_mean": float(np.mean(fold_balanced_accuracies)),
        "pca50_logistic_cv_fold_balanced_accuracy_sd": float(
            np.std(fold_balanced_accuracies, ddof=1)
        ),
        "pca50_kmeans_adjusted_rand_index": float(adjusted_rand_score(target, unsupervised_labels)),
    }


def _screen_continuous_variables(
    contrasts: dict[str, pd.DataFrame],
    targets: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    variables = [
        "thumbnail_hue_median",
        "patch_count",
        "tissue_grid_occupancy",
        "level0_width",
        "level0_height",
        "slide_area_pixels",
        "source_size",
        "source_bytes_per_patch",
        "source_bytes_per_pixel",
        "chan_dmean_max",
        "slide_number",
        "age_at_diagnosis",
    ]
    rows: list[dict[str, Any]] = []
    for name, subset in contrasts.items():
        positive, negative = targets[name]
        binary = subset[subset["visible_island"].isin([positive, negative])]
        target = binary["visible_island"].eq(positive).astype(int)
        for variable in variables:
            complete = binary[variable].notna()
            if complete.sum() < 20 or target[complete].nunique() != 2:
                continue
            values = binary.loc[complete, variable].to_numpy(dtype=float)
            labels = target[complete].to_numpy()
            raw_auc = roc_auc_score(labels, values)
            test = mannwhitneyu(values[labels == 0], values[labels == 1])
            rows.append(
                {
                    "contrast": name,
                    "variable": variable,
                    "n": int(complete.sum()),
                    "roc_auc_oriented": float(max(raw_auc, 1.0 - raw_auc)),
                    "mann_whitney_p": float(test.pvalue),
                    "negative_median": float(np.median(values[labels == 0])),
                    "positive_median": float(np.median(values[labels == 1])),
                }
            )
    results = pd.DataFrame(rows).sort_values("mann_whitney_p").reset_index(drop=True)
    results["bh_fdr_q"] = _bh_fdr(results["mann_whitney_p"].to_numpy())
    return results


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted values in the input order."""
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty_like(ranked)
    q_values[order] = np.clip(ranked, 0.0, 1.0)
    return q_values


def _categorical_tests(frame: pd.DataFrame) -> dict[str, Any]:
    sr1482 = frame[
        frame["subcohort"].eq("SR1482")
        & frame["visible_island"].isin(["small_fragment_island", "large_section_island"])
    ]
    role_table = pd.crosstab(sr1482["visible_island"], sr1482["specimen_role"])
    role_chi2 = chi2_contingency(role_table)
    section_table = pd.crosstab(sr1482["visible_island"], sr1482["section_number"].eq(2))
    section_fisher = fisher_exact(section_table)
    biopsy_table_array = np.array(
        [
            [
                int(
                    (
                        sr1482["visible_island"].eq("small_fragment_island")
                        & sr1482["sr1482_biopsy_text"]
                    ).sum()
                ),
                int(
                    (
                        sr1482["visible_island"].eq("small_fragment_island")
                        & ~sr1482["sr1482_biopsy_text"]
                    ).sum()
                ),
            ],
            [
                int(
                    (
                        sr1482["visible_island"].eq("large_section_island")
                        & sr1482["sr1482_biopsy_text"]
                    ).sum()
                ),
                int(
                    (
                        sr1482["visible_island"].eq("large_section_island")
                        & ~sr1482["sr1482_biopsy_text"]
                    ).sum()
                ),
            ],
        ]
    )
    biopsy_fisher = fisher_exact(biopsy_table_array)

    rih = frame[
        frame["subcohort"].eq("RIH-Colon")
        & frame["visible_island"].isin(["small_fragment_island", "large_section_island"])
        & frame["specimen_role"].isin(["primary", "metastatic"])
    ]
    rih_table = pd.crosstab(rih["visible_island"], rih["specimen_role"])
    ordered_rih = np.array(
        [
            [
                int(
                    (
                        (rih["visible_island"] == "small_fragment_island")
                        & (rih["specimen_role"] == "metastatic")
                    ).sum()
                ),
                int(
                    (
                        (rih["visible_island"] == "small_fragment_island")
                        & (rih["specimen_role"] == "primary")
                    ).sum()
                ),
            ],
            [
                int(
                    (
                        (rih["visible_island"] == "large_section_island")
                        & (rih["specimen_role"] == "metastatic")
                    ).sum()
                ),
                int(
                    (
                        (rih["visible_island"] == "large_section_island")
                        & (rih["specimen_role"] == "primary")
                    ).sum()
                ),
            ],
        ]
    )
    rih_fisher = fisher_exact(ordered_rih)
    rih_source = frame[
        frame["subcohort"].eq("RIH-Colon")
        & frame["visible_island"].isin(["small_fragment_island", "large_section_island"])
        & frame["rih_source_biopsy_flag"].notna()
    ]
    source_biopsy_table = np.array(
        [
            [
                int(
                    (
                        rih_source["visible_island"].eq("small_fragment_island")
                        & rih_source["rih_source_biopsy_flag"].eq(True)  # noqa: E712
                    ).sum()
                ),
                int(
                    (
                        rih_source["visible_island"].eq("small_fragment_island")
                        & rih_source["rih_source_biopsy_flag"].eq(False)  # noqa: E712
                    ).sum()
                ),
            ],
            [
                int(
                    (
                        rih_source["visible_island"].eq("large_section_island")
                        & rih_source["rih_source_biopsy_flag"].eq(True)  # noqa: E712
                    ).sum()
                ),
                int(
                    (
                        rih_source["visible_island"].eq("large_section_island")
                        & rih_source["rih_source_biopsy_flag"].eq(False)  # noqa: E712
                    ).sum()
                ),
            ],
        ]
    )
    source_biopsy_fisher = fisher_exact(source_biopsy_table)
    rih_source_type = frame[
        frame["subcohort"].eq("RIH-Colon")
        & frame["visible_island"].isin(["small_fragment_island", "large_section_island"])
        & frame["rih_source_type"].notna()
    ].copy()
    rih_source_type["type_is_biopsy"] = rih_source_type["rih_source_type"].eq("Biopsy")
    source_type_table = np.array(
        [
            [
                int(
                    (
                        rih_source_type["visible_island"].eq("small_fragment_island")
                        & rih_source_type["type_is_biopsy"]
                    ).sum()
                ),
                int(
                    (
                        rih_source_type["visible_island"].eq("small_fragment_island")
                        & ~rih_source_type["type_is_biopsy"]
                    ).sum()
                ),
            ],
            [
                int(
                    (
                        rih_source_type["visible_island"].eq("large_section_island")
                        & rih_source_type["type_is_biopsy"]
                    ).sum()
                ),
                int(
                    (
                        rih_source_type["visible_island"].eq("large_section_island")
                        & ~rih_source_type["type_is_biopsy"]
                    ).sum()
                ),
            ],
        ]
    )
    source_type_fisher = fisher_exact(source_type_table)
    expected_tables = {
        "SR1482 biopsy text": (
            biopsy_table_array,
            np.array([[118, 31], [18, 425]]),
        ),
        "RIH Raw_Source_1 Biopsy": (
            source_biopsy_table,
            np.array([[56, 2], [4, 158]]),
        ),
        "RIH Raw_Source_1 Type": (
            source_type_table,
            np.array([[55, 3], [5, 157]]),
        ),
    }
    for name, (observed, expected) in expected_tables.items():
        if not np.array_equal(observed, expected):
            raise ValueError(
                f"Frozen {name} table changed: observed={observed.tolist()}, "
                f"expected={expected.tolist()}"
            )
    return {
        "sr1482_specimen_role": {
            "table": role_table.astype(int).to_dict(orient="index"),
            "chi_square": float(role_chi2.statistic),
            "p": float(role_chi2.pvalue),
        },
        "sr1482_second_section": {
            "table": section_table.astype(int).to_dict(orient="index"),
            "odds_ratio": float(section_fisher.statistic),
            "p": float(section_fisher.pvalue),
        },
        "sr1482_biopsy_text": {
            "table_rows_small_large_columns_biopsy_other": biopsy_table_array.tolist(),
            "odds_ratio": float(biopsy_fisher.statistic),
            "p": float(biopsy_fisher.pvalue),
        },
        "rih_specimen_role_known_only": {
            "table": rih_table.astype(int).to_dict(orient="index"),
            "metastatic_vs_primary_small_over_large_odds_ratio": float(rih_fisher.statistic),
            "p": float(rih_fisher.pvalue),
        },
        "rih_raw_source_1_biopsy_known_only": {
            "table_rows_small_large_columns_biopsy_nonbiopsy": source_biopsy_table.tolist(),
            "odds_ratio": float(source_biopsy_fisher.statistic),
            "p": float(source_biopsy_fisher.pvalue),
            "known_n": int(len(rih_source)),
        },
        "rih_raw_source_1_type_biopsy_known_only": {
            "table_rows_small_large_columns_biopsy_other": source_type_table.tolist(),
            "odds_ratio": float(source_type_fisher.statistic),
            "p": float(source_type_fisher.pvalue),
            "known_n": int(len(rih_source_type)),
        },
    }


def _rule_text(metrics: dict[str, Any]) -> str:
    threshold = metrics["threshold"]
    if metrics["predictor"] == "thumbnail_hue_median":
        return f"median hue {metrics['positive_rule_operator']} {threshold:.4f}"
    return f"UNI patches {metrics['positive_rule_operator']} {threshold:,.1f}"


def save_figure(
    contrasts: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, Any]],
    predictions: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.1), facecolor="white")
    panel_specs = {
        "SR386": ("Residual tissue-color signature", "main-like", "cool/blue-shifted"),
        "SR1482": ("Tissue-section size", "large/full-section", "small/fragment"),
        "RIH fixed MPP": (
            "Tissue-section size at MPP 0.5016",
            "large/full-section",
            "small/fragment",
        ),
    }
    for index, (name, ax) in enumerate(zip(panel_specs, axes, strict=True)):
        subset = contrasts[name].copy()
        subset["rule_prediction"] = predictions[name]
        positive_island = {
            "SR386": "cool_stain_island",
            "SR1482": "small_fragment_island",
            "RIH fixed MPP": "small_fragment_island",
        }[name]
        subset["actual_island"] = subset["visible_island"].eq(positive_island).astype(int)
        title, negative_label, positive_label = panel_specs[name]
        for value, color, label in (
            (0, NEGATIVE_COLOR, negative_label),
            (1, POSITIVE_COLOR, positive_label),
        ):
            selected = subset[subset["actual_island"].eq(value)]
            ax.scatter(
                selected["umap_1"],
                selected["umap_2"],
                s=30,
                c=color,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.3,
                label=f"{label} (n={len(selected):,})",
                rasterized=True,
            )
        mismatches = subset[subset["rule_prediction"].ne(subset["actual_island"])]
        ax.scatter(
            mismatches["umap_1"],
            mismatches["umap_2"],
            s=48,
            facecolors="none",
            edgecolors=INK,
            linewidths=0.7,
            label=f"rule mismatch (n={len(mismatches):,})",
            rasterized=True,
        )
        ax.set_title(
            f"{chr(ord('A') + index)}  {name}\n{title}",
            loc="left",
            fontsize=11.5,
            weight="semibold",
            color=INK,
            pad=8,
        )
        ax.text(
            0.02,
            -0.08,
            f"{_rule_text(metrics[name])}\n"
            f"patient-grouped CV balanced accuracy {metrics[name]['cv_balanced_accuracy']:.3f} · "
            f"AUC {metrics[name]['roc_auc_oriented']:.3f}",
            transform=ax.transAxes,
            fontsize=8.5,
            color=MUTED,
            va="top",
            clip_on=False,
        )
        ax.legend(
            loc="best",
            fontsize=8,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
            labelcolor=MUTED,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")
        for spine in ax.spines.values():
            spine.set_color("#d8d7d0")
        ax.set_facecolor("#fbfbf9")

    fig.suptitle(
        "What separates the disconnected UNI-v1 UMAP islands?",
        x=0.04,
        y=0.98,
        ha="left",
        fontsize=17,
        weight="semibold",
        color=INK,
    )
    fig.text(
        0.04,
        0.91,
        "Rules are tested against stable within-subcohort DBSCAN islands; RIH is restricted to identical native MPP. Coordinates are unchanged.",
        fontsize=9.5,
        color=MUTED,
    )
    fig.text(
        0.99,
        0.012,
        "Patch count and dimensions were not explicit UMAP inputs; they proxy specimen composition/section size",
        ha="right",
        fontsize=8,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.79, bottom=0.20, wspace=0.15)
    png = output_dir / "umap_univ1_separation_rules.png"
    pdf = output_dir / "umap_univ1_separation_rules.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Replace this script's outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame, cluster_audit = assign_visible_islands(assemble_frame())
    contrasts = {
        "SR386": frame[frame["subcohort"].eq("SR386")].copy(),
        "SR1482": frame[
            frame["subcohort"].eq("SR1482")
            & frame["visible_island"].isin(["small_fragment_island", "large_section_island"])
        ].copy(),
        "RIH fixed MPP": frame[
            frame["subcohort"].eq("RIH-Colon")
            & frame["visible_island"].isin(["small_fragment_island", "large_section_island"])
        ].copy(),
    }
    targets = {
        "SR386": ("cool_stain_island", "main_stain_island"),
        "SR1482": ("small_fragment_island", "large_section_island"),
        "RIH fixed MPP": ("small_fragment_island", "large_section_island"),
    }
    predictors = {
        "SR386": "thumbnail_hue_median",
        "SR1482": "patch_count",
        "RIH fixed MPP": "patch_count",
    }
    rule_metrics: dict[str, dict[str, Any]] = {}
    predictions: dict[str, np.ndarray] = {}
    high_dimensional: dict[str, dict[str, float | int]] = {}
    for name, subset in contrasts.items():
        positive, negative = targets[name]
        rule_metrics[name], predictions[name] = _binary_rule_metrics(
            subset,
            predictor=predictors[name],
            positive_island=positive,
            negative_island=negative,
        )
        high_dimensional[name] = _high_dimensional_confirmation(
            subset,
            positive_island=positive,
            negative_island=negative,
        )

    frame["best_simple_rule_prediction"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["best_simple_rule_correct"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    for name, subset in contrasts.items():
        positive, _ = targets[name]
        predicted = predictions[name].astype(int)
        actual = subset["visible_island"].eq(positive).astype(int).to_numpy()
        frame.loc[subset.index, "best_simple_rule_prediction"] = predicted
        frame.loc[subset.index, "best_simple_rule_correct"] = predicted == actual

    screen = _screen_continuous_variables(contrasts, targets)
    for name, predictor in predictors.items():
        selected = screen[screen["contrast"].eq(name) & screen["variable"].eq(predictor)]
        if len(selected) != 1:
            raise ValueError(f"Missing selected-rule multiplicity test for {name}/{predictor}")
        rule_metrics[name]["continuous_screen_bh_fdr_q"] = float(selected.iloc[0]["bh_fdr_q"])
    categorical = _categorical_tests(frame)
    categorical_names = [name for name, result in categorical.items() if "p" in result]
    categorical_q = _bh_fdr(
        np.array([categorical[name]["p"] for name in categorical_names], dtype=float)
    )
    for name, q_value in zip(categorical_names, categorical_q, strict=True):
        categorical[name]["bh_fdr_q_across_reported_tests"] = float(q_value)
    figures = save_figure(contrasts, rule_metrics, predictions, args.output_dir)

    output_columns = [
        "slide_id",
        "subcohort",
        "patient_uid",
        "umap_1",
        "umap_2",
        "visible_island",
        "dbscan_cluster",
        "native_mpp",
        "mpp",
        "n_patches",
        "patch_count",
        "tissue_grid_occupancy",
        "level0_width",
        "level0_height",
        "source_size",
        "source_bytes_per_patch",
        "source_bytes_per_pixel",
        "thumbnail_hue_median",
        "thumbnail_tissue_pixels",
        "ssim_min",
        "chan_dmean_max",
        "specimen_role",
        "image_format",
        "age_at_diagnosis",
        "tumor_site_group",
        "metastatic_site_group",
        "tumor_site_raw",
        "sr1482_biopsy_text",
        "sr1482_polyp_text",
        "rih_source_biopsy",
        "rih_source_biopsy_flag",
        "rih_source_type",
        "section_number",
        "best_simple_rule_prediction",
        "best_simple_rule_correct",
    ]
    assignments = args.output_dir / "slide_cluster_assignments.parquet"
    frame[output_columns].to_parquet(assignments, index=False)
    associations = args.output_dir / "continuous_association_tests.csv"
    screen.to_csv(associations, index=False)

    summary = {
        "analysis": "univ1_umap_separation_audit",
        "n_slides_audited": int(len(frame)),
        "source_sha256": {
            "audit_script": _sha256(Path(__file__).resolve()),
            "coordinates": _sha256(COORDINATES),
            "slide_means": _sha256(SLIDE_MEANS),
            "manifest": _sha256(MANIFEST),
            "surgen_repair_state": _sha256(SURGEN_STATE),
            "rih_reconciliation_workbook": _sha256(RIH_WORKBOOK),
        },
        "cluster_audit": cluster_audit,
        "rule_metrics": rule_metrics,
        "high_dimensional_confirmation": high_dimensional,
        "categorical_tests": categorical,
        "artifacts": {
            "assignments": str(assignments),
            "continuous_associations": str(associations),
            "report": str(args.output_dir / "report.md"),
            "figures": {key: str(value) for key, value in figures.items()},
            "sha256": {
                "assignments": _sha256(assignments),
                "continuous_associations": _sha256(associations),
                "figure_png": _sha256(figures["png"]),
                "figure_pdf": _sha256(figures["pdf"]),
            },
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    sr386 = rule_metrics["SR386"]
    sr1482 = rule_metrics["SR1482"]
    rih = rule_metrics["RIH fixed MPP"]
    sr1482_biopsy = categorical["sr1482_biopsy_text"]
    rih_biopsy = categorical["rih_raw_source_1_biopsy_known_only"]
    rih_type = categorical["rih_raw_source_1_type_biopsy_known_only"]
    sr386_age = screen[
        screen["contrast"].eq("SR386") & screen["variable"].eq("age_at_diagnosis")
    ].iloc[0]
    highd_sr386 = high_dimensional["SR386"]
    highd_sr1482 = high_dimensional["SR1482"]
    highd_rih = high_dimensional["RIH fixed MPP"]
    report = args.output_dir / "report.md"
    report.write_text(
        "\n".join(
            [
                "# UNI-v1 disconnected-island audit",
                "",
                "The three visual separations do not share one cause.",
                "",
                "| Subcohort | Strongest observed rule | Patient-grouped OOF balanced accuracy | ROC AUC | MW BH q-value | Interpretation |",
                "|---|---|---:|---:|---:|---|",
                f"| SR386 | {_rule_text(sr386)} | {sr386['cv_balanced_accuracy']:.3f} | {sr386['roc_auc_oriented']:.3f} | {sr386['continuous_screen_bh_fdr_q']:.2e} | Residual tissue-color signature |",
                f"| SR1482 | {_rule_text(sr1482)} | {sr1482['cv_balanced_accuracy']:.3f} | {sr1482['roc_auc_oriented']:.3f} | {sr1482['continuous_screen_bh_fdr_q']:.2e} | Small fragment versus large/full section |",
                f"| RIH at MPP 0.5016 | {_rule_text(rih)} | {rih['cv_balanced_accuracy']:.3f} | {rih['roc_auc_oriented']:.3f} | {rih['continuous_screen_bh_fdr_q']:.2e} | Biopsy/small-fragment specimens versus large resections |",
                "",
                "## Interpretation",
                "",
                f"- SR386 is fixed for primary role, TIFF format, native MPP 0.25, and extraction settings. The detached island is cooler/bluer by thumbnail hue; patch burden is not explanatory. This is a residual color phenotype consistent with an unrecorded staining/acquisition batch, but no recorded field proves the batch identity. Age is associated but weak (median {sr386_age['positive_median']:.0f} versus {sr386_age['negative_median']:.0f} years; AUC {sr386_age['roc_auc_oriented']:.3f}, BH q={sr386_age['bh_fdr_q']:.2e}) and is not a separation rule.",
                f"- SR1482 is a biopsy and tissue-size axis: biopsy wording occurs in {sr1482_biopsy['table_rows_small_large_columns_biopsy_other'][0][0]}/149 small-island slides versus {sr1482_biopsy['table_rows_small_large_columns_biopsy_other'][1][0]}/443 large-island slides (OR {sr1482_biopsy['odds_ratio']:.1f}, Fisher p={sr1482_biopsy['p']:.2e}, categorical BH q={sr1482_biopsy['bh_fdr_q_across_reported_tests']:.2e}). Specimen role is not associated with the two islands.",
                f"- RIH was restricted to native MPP 0.5016 before testing the additional split. The raw-source Biopsy field is positive in {rih_biopsy['table_rows_small_large_columns_biopsy_nonbiopsy'][0][0]}/{sum(rih_biopsy['table_rows_small_large_columns_biopsy_nonbiopsy'][0])} known small-island slides versus {rih_biopsy['table_rows_small_large_columns_biopsy_nonbiopsy'][1][0]}/{sum(rih_biopsy['table_rows_small_large_columns_biopsy_nonbiopsy'][1])} known large-island slides (45/265 are missing; OR {rih_biopsy['odds_ratio']:.1f}, Fisher p={rih_biopsy['p']:.2e}, categorical BH q={rih_biopsy['bh_fdr_q_across_reported_tests']:.2e}). The near-duplicate Type coding gives the same sensitivity conclusion (OR {rih_type['odds_ratio']:.1f}); metastatic role is a weaker correlate.",
                "- Patch counts and slide dimensions were not explicit UMAP inputs. They proxy slide composition/preparation; the number and composition of tile embeddings can still affect the slide mean.",
                f"- The partitions remain separable in the original normalized 1,024-D slide means. Unsupervised PCA50+k-means ARI is {highd_sr386['pca50_kmeans_adjusted_rand_index']:.3f} (SR386), {highd_sr1482['pca50_kmeans_adjusted_rand_index']:.3f} (SR1482), and {highd_rih['pca50_kmeans_adjusted_rand_index']:.3f} (RIH); patient-grouped PCA/logistic OOF balanced accuracy is {highd_sr386['pca50_logistic_cv_balanced_accuracy']:.3f}, {highd_sr1482['pca50_logistic_cv_balanced_accuracy']:.3f}, and {highd_rih['pca50_logistic_cv_balanced_accuracy']:.3f}.",
                "- Thresholds are descriptive full-data stumps; cross-validation relearns each threshold within its training fold. Mann-Whitney and Fisher tests are slide-level descriptive tests; patient-grouped CV is the leakage-resistant performance check.",
                "",
            ]
        )
    )
    print(f"Wrote UNI-v1 separation audit to {args.output_dir}")


if __name__ == "__main__":
    main()
