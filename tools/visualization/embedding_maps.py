#!/usr/bin/env python3
"""PCA/UMAP maps of slide-level (mean-pooled) foundation embeddings.

Population: TCGA + SurGen primary (the v2 DEV master, 1,389 slides).
Contrasts: KRAS mutant vs WT; G12D/V vs other mutants; G12D vs other
mutants; plus cohort coloring (batch-structure check). One PCA/UMAP fit per
encoder on the full population; mutant-only panels subset the same
coordinates. Each panel is annotated with a 5-fold logistic AUROC computed
ON THE 2D COORDINATES shown (what the eye can separate), with the full-dim
probe AUROC for reference in the title block.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from oceanpath.kras import study  # noqa: E402
from oceanpath.kras.labels import subvariant_tokens  # noqa: E402

OUT = Path("/mnt/wsl/oceanpath-hot/outputs/eval/kras_v2/embedding_maps")
FEATURE_DIRS = {
    "UNIv1": study.PINNED_FEATURE_DIR,
    "CONCH v1.5": study.FEATURE_ROOT / "20x_512px_0px_overlap_mpp0.5" / "features_conch_v15",
}

# Validated categorical palette (dataviz reference, slots 1-3).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#1a1a19", "#6b6a62"


def slide_means(feature_dir: Path, slide_ids: list[str], cache: Path) -> np.ndarray:
    if cache.is_file():
        data = np.load(cache, allow_pickle=True)
        if list(data["slide_ids"]) == slide_ids:
            return data["X"]
    rows = []
    for sid in slide_ids:
        with h5py.File(feature_dir / f"{sid}.h5", "r") as h:
            d = h["features"]
            stride = max(1, d.shape[0] // 2048)
            rows.append(d[::stride][:2048].astype(np.float32).mean(0))
    X = np.stack(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, slide_ids=np.array(slide_ids, dtype=object))
    return X


def auroc_2d(coords: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    scores = cross_val_predict(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        coords,
        y,
        cv=5,
        method="decision_function",
    )
    return float(roc_auc_score(y, scores))


def scatter_panel(ax, coords, groups, palette, title, order):
    for name, color in zip(order, palette, strict=True):
        m = groups == name
        ax.scatter(
            coords[m, 0],
            coords[m, 1],
            s=7,
            c=color,
            alpha=0.55,
            linewidths=0,
            label=f"{name} (n={int(m.sum())})",
        )
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color("#dddcd3")
    ax.legend(
        loc="best", fontsize=7, frameon=False, labelcolor=MUTED, handletextpad=0.2, markerscale=1.8
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    import umap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    master = pd.read_csv(study.MANIFEST_ROOT / "crc_kras_master_dev_v2.csv")
    slide_ids = master["slide_id"].tolist()
    toks = master["kras_subvariant"].map(subvariant_tokens)
    master["is_mut"] = master["kras"] == "mutant"
    master["g12dv"] = toks.map(lambda t: bool({"G12D", "G12V"} & set(t)))
    master["g12d"] = toks.map(lambda t: "G12D" in t)
    mut_mask = (master["is_mut"] & toks.map(len).gt(0)).to_numpy()

    for enc, feature_dir in FEATURE_DIRS.items():
        tag = enc.split()[0].lower()
        X = slide_means(feature_dir, slide_ids, OUT / f"means_{tag}.npz")
        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        P = pca.fit_transform(Xs)
        var = pca.explained_variance_ratio_ * 100
        U = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42).fit_transform(
            Xs
        )

        fig, axes = plt.subplots(2, 4, figsize=(16, 7.6), facecolor="white")
        contrasts = [
            (
                "KRAS: mutant vs wild-type",
                np.ones(len(master), bool),
                np.where(master["is_mut"], "mutant", "wild-type"),
                ["wild-type", "mutant"],
                [BLUE, ORANGE],
            ),
            (
                "G12D/V vs other mutants",
                mut_mask,
                np.where(master["g12dv"], "G12D/V", "other mutant"),
                ["other mutant", "G12D/V"],
                [BLUE, ORANGE],
            ),
            (
                "G12D vs other mutants",
                mut_mask,
                np.where(master["g12d"], "G12D", "other mutant"),
                ["other mutant", "G12D"],
                [BLUE, ORANGE],
            ),
            (
                "cohort (batch check)",
                np.ones(len(master), bool),
                master["cohort_group"].to_numpy(),
                ["SR386", "SR1482", "TCGA"],
                [BLUE, ORANGE, AQUA],
            ),
        ]
        for row, (coords, row_name) in enumerate([(P, "PCA"), (U, "UMAP")]):
            for col, (name, mask, groups, order, palette) in enumerate(contrasts):
                ax = axes[row, col]
                c, g = coords[mask], groups[mask]
                extra = ""
                if len(order) == 2:
                    y = (g == order[1]).astype(int)
                    extra = f"  ·  2D AUROC {auroc_2d(c, y):.2f}"
                sub = f"PC1 {var[0]:.0f}% / PC2 {var[1]:.0f}%" if row == 0 else "UMAP"
                scatter_panel(ax, c, g, palette, f"{row_name}: {name}\n{sub}{extra}", order)
        fig.suptitle(
            f"{enc} slide embeddings (mean-pooled, TCGA + SurGen primary, n=1,389) — "
            "full-dim probe AUROC: mut-vs-WT ~0.63–0.68, allele tasks ~0.5",
            fontsize=11,
            color=INK,
            x=0.01,
            ha="left",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out_png = OUT / f"embedding_maps_{tag}.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
