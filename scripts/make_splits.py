"""Generic CLI entry point for split generation.

Study-specific manifest construction lives in ``tools/phase2_manifests_splits.py``; this
thin launcher is the split stage invoked by the reusable foundation DAG.
"""

import hydra
from omegaconf import DictConfig

from oceanpath.workflows.splitting import run_split_generation


@hydra.main(config_path="../configs", config_name="make_splits", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(run_split_generation(cfg).to_json())


if __name__ == "__main__":
    main()
