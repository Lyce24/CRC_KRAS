# Tests

[`unit/`](unit/) covers the reusable library, configuration, datasets, models,
training, and pipeline contracts. [`extraction/`](extraction/) covers slide
inventory, tiling, streaming, and feature preparation with small fixtures.
Experiment, campaign, and report tests live directly in this directory.

Root experiment imports use the canonical `aim{1,2,3,4}_<purpose>.py` names;
see the complete [experiment entry-point tables](../tools/README.md#canonical-experiment-entry-points).
Existing test filenames may retain earlier experiment labels. Test references
were updated with the source renames, so these published files do not preserve
the original checksums recorded in historical receipts.

This source publication excludes datasets, feature stores, trained models,
outputs, reports, review packets, and archived code. Some experiment and report
tests require those external artifacts or a particular local environment.
Frozen receipts and source snapshots must be checked against their original
artifacts and source bytes. Archived tests under `draft/legacy_tests/` are not
included in this publication.

Root dependency metadata (`pyproject.toml` and `uv.lock`) is included for
environment verification and campaign source snapshots; see the
[environment setup](../README.md#environment). Some artifact-backed tests read preregistrations under
`reports/`, frozen splits under `outputs/`, or sealed runs on configured data
volumes; they are not standalone tests of this source tree. Extraction test
collection additionally requires the optional `shapely` dependency.

With a Python environment containing the required project and test
dependencies, run a selected reusable group from the repository root:

```bash
python -m pytest tests/unit -m "not slow and not gpu and not integration"
python -m pytest tests/extraction -m "not slow and not gpu and not integration"
```

These commands exclude explicitly marked slow, GPU, and integration tests;
other tests can still require optional dependencies or external artifacts.
The extraction group uses WSI and geometry dependencies where needed. To
select an experiment test, pass its published path to `python -m pytest`.
This page does not prescribe environment installation.

Shared import setup in [`conftest.py`](conftest.py) applies throughout the test
tree. Add reusable tests to the appropriate group and create fixture data
under pytest's `tmp_path` when possible.
