# CRC_KRAS

Code for the colorectal KRAS histomorphology study, built on the OceanPath
multiple-instance learning pipeline. The 42 main experiment entries use
`aim1_` through `aim4_` prefixes; see the complete
[experiment index and filename migration table](tools/README.md#canonical-experiment-entry-points).

## Environment

The package version, supported Python range, dependencies, optional groups,
and development-tool settings are defined in [`pyproject.toml`](pyproject.toml).
[`.python-version`](.python-version) selects Python 3.10, and
[`uv.lock`](uv.lock) records the resolved dependency versions, hashes, and
pinned Git sources. Keep the project file and lockfile together when recording
or preparing a campaign source snapshot.

With `uv` available, run from the repository root:

```bash
uv sync --frozen
uv lock --check
```

The default environment includes the development dependency group. Select the
optional capabilities required by your workflow:

```bash
uv sync --frozen --group extract                 # TRIDENT and WSI extraction
uv sync --frozen --extra atlas --extra reporting # slide reading and reports
uv sync --frozen --extra track                   # experiment tracking
```

Combine all required groups and extras in the same sync command to retain
them. Slide extraction and atlas workflows also require their native slide
reader libraries, model access, and study inputs. Configure data locations
through [`configs/platform/`](configs/platform/) and the study configurations.

## Repository layout

| Location | Contents |
|---|---|
| `aim1_*.py`–`aim4_*.py` | Main experiment and campaign entries |
| [`configs/`](configs/) | Hydra experiment, data, model, and platform settings |
| [`scripts/`](scripts/) | Generic pipeline launchers |
| [`src/oceanpath/`](src/oceanpath/) | Reusable Python library |
| [`tests/`](tests/README.md) | Unit, extraction, and experiment tests |
| [`tools/`](tools/README.md) | Preparation, analysis, audit, and visualization tools |
| [`Makefile`](Makefile) | Environment, checks, tests, and package-build commands |
| [`.pre-commit-config.yaml`](.pre-commit-config.yaml) | Pinned development hooks |
| [`.github/workflows/ci.yml`](.github/workflows/ci.yml) | Project CI configuration |

## Validation

```bash
make config
make test-unit
make build
```

`make lint`, `make typecheck`, and `make precommit` run the configured
development checks. `make test` and the existing CI test job also cover study
tests that may require external artifacts; the full suite is not guaranteed
to run from a source-only clone. Extraction test collection requires optional
geometry and slide dependencies, including `shapely`. See the
[test guide](tests/README.md) for the distinction between reusable tests and
artifact-backed study checks.

## External study material

Datasets, feature stores, checkpoints, outputs, reports, review packets,
deployment files, manuscript material, caches, credentials, and installed
virtual environments are excluded from Git. Restore the required study
inputs separately before launching experiments. Historical receipts and
source snapshots require their original source bytes and paths; renamed
publication files do not replace those frozen artifacts.
