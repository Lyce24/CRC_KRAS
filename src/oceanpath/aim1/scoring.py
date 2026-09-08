"""Scoring a frozen fold ensemble on a manifest (Aim1_Setup.md §4.5).

Lifted out of the allele study so Aim 1 and Aim 3 share no code: the two have
different populations, folds, recipes, and freeze dates, and a shared helper
would let a change made for one silently alter the other.

Every manifest slide is scored by every fold model with FULL bags; the
per-model rows are kept so any later re-slice is free, and the deployed score
is the MEAN LOGIT across models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from oceanpath.eval.external import sigmoid

logger = logging.getLogger(__name__)


def score_manifest_with_checkpoints(
    checkpoints: list[tuple[int, int, Path]],
    manifest_csv: Path,
    feature_dir: Path,
    num_classes: int,
    batch_size: int = 1,
    num_workers: int = 4,
    device: str | None = None,
) -> pd.DataFrame:
    """Score every manifest slide with every fold model. FULL bags, fp32/bf16.

    Returns one row per (slide, seed, fold) with logit columns — the caller
    averages logits into the deployed ensemble score, and keeping the
    per-model scores makes every sensitivity re-slice (phase 8) free.
    """
    import torch
    from torch.utils.data import DataLoader

    from oceanpath.datasets.datamodule import SimpleMILCollator, SlideDataset
    from oceanpath.training.lightning import MILTrainModule

    manifest = pd.read_csv(manifest_csv)
    dataset = SlideDataset(
        feature_dir=str(feature_dir),
        slide_ids=manifest["slide_id"].tolist(),
        labels=dict(zip(manifest["slide_id"], manifest["target_label"], strict=True)),
        max_instances=None,  # full bags for the scientific result (§6.3)
        is_train=False,
        force_float32=True,
    )
    missing = set(manifest["slide_id"]) - set(dataset.slide_ids)
    if missing:
        raise SystemExit(f"{len(missing)} manifest slides lack features: {sorted(missing)[:5]}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=SimpleMILCollator(max_instances=None),
    )

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = resolved_device == "cuda" and torch.cuda.is_bf16_supported()
    rows: list[dict] = []
    for seed, fold_idx, ckpt in checkpoints:
        try:
            module = MILTrainModule.load_from_checkpoint(
                str(ckpt), map_location=resolved_device, weights_only=False
            )
        except TypeError:
            module = MILTrainModule.load_from_checkpoint(str(ckpt), map_location=resolved_device)
        module.eval().to(resolved_device)
        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(resolved_device, non_blocking=True)
                mask = batch["mask"].to(resolved_device) if batch.get("mask") is not None else None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    output = module.model(features, mask=mask)
                logits = output.logits.detach().float().cpu().numpy()
                for i, slide_id in enumerate(batch["slide_ids"]):
                    row = {"slide_id": slide_id, "seed": seed, "fold": fold_idx}
                    values = np.atleast_1d(logits[i]).ravel()
                    if num_classes == 2 and values.size == 1:
                        row["logit"] = float(values[0])
                    else:
                        for c in range(values.size):
                            row[f"logit_{c}"] = float(values[c])
                    rows.append(row)
        del module
        if resolved_device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Scored %s with seed %d fold %d", manifest_csv.name, seed, fold_idx)

    scores = pd.DataFrame(rows)
    n_models = len(checkpoints)
    if len(scores) != n_models * len(dataset):
        raise SystemExit("Scoring produced an unexpected number of rows")
    return scores


def ensemble_slide_predictions(scores: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Per-model logits → deployed slide score (mean logit over the 15 models),
    emitted as prob_* columns so patient aggregation is shared with OOF."""
    logit_cols = [c for c in scores.columns if c == "logit" or c.startswith("logit_")]
    mean_logits = scores.groupby("slide_id")[logit_cols].mean().reset_index()
    out = mean_logits.merge(
        manifest[["slide_id", "target_label"]], on="slide_id", validate="one_to_one"
    ).rename(columns={"target_label": "label"})
    if logit_cols == ["logit"]:
        out["prob_1"] = sigmoid(out["logit"].to_numpy())
    else:
        raw = out[logit_cols].to_numpy()
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        for index, col in enumerate(logit_cols):
            out[f"prob_{col.split('_')[1]}"] = probs[:, index]
    return out
