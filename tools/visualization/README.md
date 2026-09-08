# Visualization commands

These tools were moved from `tools/` into this directory. Their input and output
locations are unchanged. Run a command from the repository root, for example:

```bash
.venv/bin/python tools/visualization/epoch_trends.py --help
.venv/bin/python -m tools.visualization.cohort_umap_all_slides --help
```

| Command | Purpose |
|---|---|
| `epoch_trends.py` | Plot training metrics across epochs and compare arms |
| `embedding_maps.py` | Generate KRAS embedding maps |
| `cohort_umap_all_slides.py` | Compare cohorts using matched UNI/CONCH slide embeddings |
| `cohort_umap_virchow2_snapshot.py` | Plot a frozen set of completed Virchow2 embeddings |
| `univ1_controlled_contrast_umaps.py` | Plot scanner, collection, and specimen-role contrasts |
| `univ1_umap_subcohort_mpp.py` | Plot UNI subcohorts and native resolution |
| `univ1_umap_separation_audit.py` | Analyze the factors associated with UNI UMAP separation |
| `univ1_umap_comprehensive.py` | Combine UNI plots with audited tissue rules |
| `conch_umap_comprehensive.py` | Plot CONCH subcohorts, resolution, and tissue classes |

Both direct script execution and `python -m tools.visualization.<module>` are
supported. `--help` displays usage without reading study data or generating
outputs. Historical reports retain their original command paths; use the paths
above for new invocations.
