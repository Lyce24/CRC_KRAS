"""Generic CLI entry point for feature extraction.

This is the supported entry point for feature extraction, used by the
foundation DAG and by ``tools/run_cptac_priority_encoding.py``. The superseded
allele study's duplicate wrapper is archived at
``draft/kras_allele_prereg/phase1_feature_extraction.py``.
"""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.extraction import run_extraction


@hydra.main(config_path="../configs", config_name="extract", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(run_extraction(cfg).to_json())


if __name__ == "__main__":
    main()
