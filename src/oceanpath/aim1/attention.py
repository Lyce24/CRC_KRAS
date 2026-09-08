"""Per-tile attention export (E0's second output, E3b's input).

Attention is exported in a dedicated inference pass rather than during
training, for two reasons: the training loop only requests attention on canary
steps, and — more importantly — a slide's attention must come from the model
that is entitled to score it. Exporting mid-training would give every slide the
attention of a model that had already seen it.

Two entitlement rules, and they are NOT interchangeable:

``export_attention``        OUT-OF-FOLD. Each slide is scored by the single fold
                            model whose outer test fold it belongs to — the one
                            model that never saw it. This is E0's rule.
``export_attention_refit``  ONE FROZEN REFIT scores every slide in the manifest.
                            This is E2a/E2b's rule: the transported object is a
                            full-source refit with no folds and no held-out
                            relationship to the target, so there is no fold to
                            look a slide up in. Using an E0 fold model on a
                            target slide instead would be a different model
                            trained on different cohorts under a different
                            objective, and the attention would not correspond to
                            the score that slide was actually given.

One HDF5 per model holds a group per slide with the tile-level attention
weights and the tile coordinates, so a prototype atlas can go straight from a
high-attention tile back to its location on the slide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from oceanpath.aim1 import paths


def _write_slide_attention(
    module: Any,
    slide_ids: list[str],
    out: h5py.File,
    feature_dir: Path,
    top_fraction: float,
    device: str,
    attrs: dict[str, Any],
) -> int:
    """Score ``slide_ids`` with one loaded module and write one group each.

    Softmax is applied HERE because the aggregator returns pre-softmax logits;
    normalising per slide makes the weights sum to 1 over that slide and so
    comparable across slides of very different tile counts. Attention mass over
    a tile subset is then a share of the slide, which is exactly the quantity
    E3b aggregates.
    """
    import torch

    written = 0
    for slide_id in slide_ids:
        if slide_id in out:
            written += 1
            continue
        with h5py.File(feature_dir / f"{slide_id}.h5", "r") as handle:
            features = handle["features"][:].astype(np.float32)
            coords = handle["coords"][:].astype(np.int32)
        with torch.no_grad():
            result = module.model(
                torch.from_numpy(features).unsqueeze(0).to(device),
                mask=None,
                return_attention=True,
            )
        logits = result.extras["attention_weights"].detach().float().cpu().numpy().reshape(-1)
        if logits.size != len(features):
            raise SystemExit(
                f"{slide_id}: {logits.size} attention values for {len(features)} tiles"
            )
        weights = np.exp(logits - logits.max())
        weights = weights / weights.sum()
        cutoff = np.quantile(weights, 1.0 - top_fraction)

        group = out.create_group(slide_id)
        group.create_dataset("attention", data=weights.astype(np.float32), compression="gzip")
        group.create_dataset("coords", data=coords, compression="gzip")
        group.create_dataset("top_decile", data=(weights >= cutoff), compression="gzip")
        for key, value in attrs.items():
            group.attrs[key] = value
        group.attrs["n_tiles"] = int(len(weights))
        written += 1
    return written


def export_attention(
    checkpoints: list[tuple[int, int, Path]],
    manifest: pd.DataFrame,
    splits: pd.DataFrame,
    destination: Path,
    feature_dir: Path | None = None,
    top_fraction: float = 0.10,
) -> dict:
    """Write OUT-OF-FOLD per-tile attention for every slide in ``manifest``.

    Each slide is scored by the single fold model whose OUTER TEST fold it
    belongs to, i.e. the one model that never saw it. ``top_fraction`` records
    which tiles fall in the top decile of attention, the selection E3b's
    montages draw from, so that step does not have to re-read every weight.
    """
    import torch

    from oceanpath.training.lightning import MILTrainModule

    feature_dir = feature_dir or paths.PINNED_FEATURE_DIR
    fold_of_slide = dict(zip(splits["slide_id"], splits["fold"], strict=True))
    missing = set(manifest["slide_id"]) - set(fold_of_slide)
    if missing:
        raise SystemExit(
            f"{len(missing)} manifest slide(s) have no fold assignment, so no model is "
            f"entitled to score them: {sorted(missing)[:5]}"
        )
    by_fold: dict[int, list[str]] = {}
    for slide_id in manifest["slide_id"]:
        by_fold.setdefault(int(fold_of_slide[slide_id]), []).append(slide_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "mode": "out_of_fold",
        "n_slides": 0,
        "top_fraction": top_fraction,
        "per_fold": {},
    }

    # Append mode: a slide already written is skipped by _write_slide_attention,
    # so an interrupted export resumes instead of re-running every forward pass.
    with h5py.File(destination, "a") as out:
        for seed, fold, checkpoint in checkpoints:
            slides = by_fold.get(int(fold), [])
            if not slides:
                continue
            module = MILTrainModule.load_from_checkpoint(
                str(checkpoint), map_location=device, weights_only=False
            )
            module.eval().to(device)
            summary["n_slides"] += _write_slide_attention(
                module, slides, out, feature_dir, top_fraction, device,
                {"fold": int(fold), "seed": int(seed), "checkpoint": str(checkpoint)},
            )
            summary["per_fold"][str(fold)] = len(slides)
            del module
            if device == "cuda":
                torch.cuda.empty_cache()

    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary


def export_attention_refit(
    checkpoint: Path,
    slide_ids: list[str],
    destination: Path,
    feature_dir: Path | None = None,
    top_fraction: float = 0.10,
    seed: int | None = None,
) -> dict:
    """Write per-tile attention from ONE frozen refit for every listed slide.

    This is the E2a/E2b entitlement rule. The transported object there is a
    single full-source refit: it has no folds, and the target cohort was held
    out of its training entirely, so every target slide is equally unseen and
    one model scores all of them. There is deliberately no fold lookup — a
    caller that wants out-of-fold attention wants ``export_attention``.

    The caller is responsible for passing the checkpoint that actually produced
    the scores being explained. Attention from any other model would describe a
    prediction that was never made.
    """
    import torch

    from oceanpath.training.lightning import MILTrainModule

    feature_dir = feature_dir or paths.PINNED_FEATURE_DIR
    if not Path(checkpoint).is_file():
        raise SystemExit(f"checkpoint missing: {checkpoint}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    destination.parent.mkdir(parents=True, exist_ok=True)

    module = MILTrainModule.load_from_checkpoint(
        str(checkpoint), map_location=device, weights_only=False
    )
    module.eval().to(device)
    attrs: dict[str, Any] = {"checkpoint": str(checkpoint), "mode": "refit"}
    if seed is not None:
        attrs["seed"] = int(seed)
    with h5py.File(destination, "a") as out:
        written = _write_slide_attention(
            module, list(slide_ids), out, feature_dir, top_fraction, device, attrs
        )
    del module
    if device == "cuda":
        torch.cuda.empty_cache()

    summary = {
        "mode": "refit",
        "n_slides": written,
        "top_fraction": top_fraction,
        "checkpoint": str(checkpoint),
        "seed": seed,
    }
    destination.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary
