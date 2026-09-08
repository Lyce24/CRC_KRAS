"""The three trained Aim-1 models and their frozen definitions.

Every model shares one recipe (UNI-v1 features, ABMIL, the locked
hyperparameters of §4.2, the same five patient folds, the same early-stopping
rule) and differs only in which development rows it trains on and how those
rows are weighted:

    1A     all 909 development patients, patient-normalised weights
           -> the locked primary model; everything in 1B re-reads its
              predictions, nothing is retrained there
    1C-R   the 745 MSS/pMMR + BRAF-wild-type patients, obtained by FILTERING
           the 1A fold assignment (never re-drawing folds), so 1A and 1C-R are
           compared on identical external patients with paired bootstraps

1C-B (covariate-balanced training via overlap weighting) is deliberately NOT
here. It was built, diagnosed, and cut in v2: the only covariate meaningfully
imbalanced between KRAS classes is BRAF (standardized difference 0.530; every
other covariate <=0.164), overlap weighting retains 87.6% of the effective
sample size, and 1C-R removes BRAF-mutant patients outright — so the two are
near-redundant. The diagnostics are still computed and reported
(``draft/aim1_superseded/aim1_phase6_controlled_retraining.py diagnostics``);
only the training run
is cut.

Robustness encoders (Virchow2, CONCH v1.5) and the appendix architectures are
deliberately absent: this run is UNI-v1 + ABMIL only, per the 2026-08-17
decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from oceanpath.aim1 import paths


@dataclass(frozen=True)
class Aim1Model:
    model_id: str
    title: str
    manifest_name: str
    # Restrict the development rows before training. None = all 909 patients.
    restrict: str | None = None
    # Manifest column carrying the per-slide training weight.
    weight_column: str = "slide_weight"
    notes: str = ""

    @property
    def manifest_path(self):
        return paths.MANIFEST_DIR / self.manifest_name

    @property
    def manifest_stem(self) -> str:
        """Filename stem Hydra composes into data.csv_path."""
        return self.manifest_name.removesuffix(".csv")

    @property
    def data_name(self) -> str:
        return f"aim1_{self.model_id}"

    def run_dir(self, seed: int = paths.PRIMARY_SEED):
        return paths.TRAIN_ROOT / self.model_id / paths.PINNED_ENCODER / f"seed{seed}"


MODELS: dict[str, Aim1Model] = {
    model.model_id: model
    for model in (
        Aim1Model(
            model_id="1a",
            title="KRAS mutant vs wild-type — locked primary model",
            manifest_name="aim1_dev.csv",
            notes="The model 1B interrogates; its five fold models are the frozen predictor.",
        ),
        Aim1Model(
            model_id="1cr",
            title="Molecular-context-restricted model (MSS/pMMR + BRAF-WT)",
            manifest_name="aim1_dev_1cr.csv",
            restrict="mss_brafwt",
            notes="Same folds as 1A, filtered — not re-drawn.",
        ),
    )
}

TRAINING_ORDER: tuple[str, ...] = ("1a", "1cr")
