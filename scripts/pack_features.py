"""Generic CLI entry point for the optional packed-feature optimization.

This is the supported entry point. The superseded allele study's duplicate
wrapper is archived at ``draft/kras_allele_prereg/phase2_packing.py``.
"""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.packing import run_packing


@hydra.main(config_path="../configs", config_name="pack", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(run_packing(cfg).to_json())


if __name__ == "__main__":
    main()
