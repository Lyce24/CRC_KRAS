#!/usr/bin/env python3
"""Final-v8 E2-CPHT-A: Orion target-internal few-shot residual adaptation.

This additive runner consumes the label-blind patient-level export made by
:mod:`aim2_confirmatory_transfer`.  ``seal`` parses KRAS outcomes only to
materialize immutable realized fold and support manifests; it does not open
native logits/embeddings or compute predictions/metrics.  ``run`` is the
first logit-analysis boundary.  The runner never reopens patches and never
updates the encoder, attention pooler, or MIL classifier.  For each frozen
all-conventional model seed it fits the prespecified residual head

    eta_adapted = eta_native + H @ delta_w + delta_b

inside five fixed KRAS-stratified folds.  Each complete support procedure uses
2, 4, or 8 patients per class from the other four folds.  There are exactly
100 procedures per budget.  The reported point is the mean of their 100
AUROCs, not the AUROC of predictions averaged over procedures.

The command boundary is strict::

    python aim2_a_orion_residual_adaptation.py preflight --output-root <lineage>
    python aim2_a_orion_residual_adaptation.py seal      --output-root <lineage>
    python aim2_a_orion_residual_adaptation.py run       --output-root <lineage>
    python aim2_a_orion_residual_adaptation.py report    --output-root <lineage>
    python aim2_a_orion_residual_adaptation.py verify    --output-root <lineage>

``seal`` hashes the frozen inference export, implementation, numerical
runtime, outcome source, and realized outcome-informed design. One immutable,
resumable artifact and receipt is written for every budget/procedure pair
(300 pairs in total).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import aim2_residual_adaptation as e2c_offset  # noqa: E402
import aim2_confirmatory_transfer as cpht  # noqa: E402
import aim2_cross_protocol_transfer as prior_cpht  # noqa: E402
from oceanpath.aim1 import lineage  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/wsl/oceanpath-hot/outputs/aim1/reruns/aim2_final_v8_complete_v1_20260822"
)
LABEL_SOURCE = cpht.LABEL_SOURCE
DEFAULT_COMPONENT_NAME = "cpht_a"
_ACTIVE_COMPONENT_NAME = DEFAULT_COMPONENT_NAME
SEEDS: tuple[int, ...] = (42, 43, 44)
N_FOLDS = 5
BUDGETS: tuple[int, ...] = (2, 4, 8)
N_PROCEDURES = 100
LAYOUT_SEED = 20_260_822
SUPPORT_SEED = 20_260_822
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_822
EMBED_DIM = 512
EXPECTED_PATIENTS = 40
EXPECTED_MUTANT = 15
EXPECTED_WILD_TYPE = 25
LAMBDA_GRID: tuple[float, ...] = (
    np.inf,
    10_000.0,
    3_000.0,
    1_000.0,
    300.0,
    100.0,
    30.0,
    10.0,
    3.0,
    1.0,
)
EPS = 1e-6
SOLVER_GRAD_TOL = 1e-6
OBJECTIVE_TOL = 1e-9
FORBIDDEN_FEATURE_COLUMNS = {
    "label",
    "target_label",
    "kras",
    "kras_subvariant",
    "used_kras",
    "v600e",
    "nras",
    "braf",
}
OUTCOME_ACCESS_CAVEAT = (
    "Orion outcomes were accessed by the earlier zero-shot sensitivity analysis before "
    "the all-conventional artifact-level inference seal was created. E2-CPHT-A was "
    "specified in final-v8 before this run, but is exploratory target-internal adaptation, "
    "not pristine prospective or external validation."
)


class ContractError(RuntimeError):
    """A governed CPHT-A input, implementation, or immutable result changed."""


@dataclass(frozen=True)
class PatientData:
    """Outcome-joined arrays in one stable patient order."""

    patient_ids: NDArray[Any]
    labels: NDArray[Any]
    folds: NDArray[Any]
    features: NDArray[Any]
    native: NDArray[Any]


@dataclass(frozen=True)
class RealizedDesign:
    """Outcome-informed folds/supports frozen before any CPHT-A logit analysis."""

    folds: pd.DataFrame
    supports: pd.DataFrame


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _activate_component_name(name: str) -> None:
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None:
        raise ValueError(
            "--component-name must contain only lowercase letters, digits, underscores, or hyphens"
        )
    global _ACTIVE_COMPONENT_NAME  # noqa: PLW0603
    _ACTIVE_COMPONENT_NAME = name


def component_root(output_root: Path) -> Path:
    return output_root / _ACTIVE_COMPONENT_NAME


def contract_path(output_root: Path) -> Path:
    return component_root(output_root) / "inputs/execution_contract.json"


def fold_manifest_path(output_root: Path) -> Path:
    return component_root(output_root) / "inputs/realized_folds.csv"


def support_manifest_path(output_root: Path) -> Path:
    return component_root(output_root) / "inputs/realized_supports.csv"


def procedure_path(output_root: Path, budget: int, draw: int) -> Path:
    return component_root(output_root) / f"procedures/k{budget}/draw_{draw:03d}.npz"


def procedure_receipt_path(output_root: Path, budget: int, draw: int) -> Path:
    return procedure_path(output_root, budget, draw).with_suffix(".receipt.json")


def run_receipt_path(output_root: Path) -> Path:
    return component_root(output_root) / "run_receipt.json"


def analysis_root(output_root: Path) -> Path:
    return component_root(output_root) / "analysis"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], lineage.artifact_identity(path))


def _validate_identity(identity: dict[str, Any], *, label: str) -> None:
    path = Path(str(identity.get("path", "")))
    if identity != _artifact(path):
        raise ContractError(f"{label} changed after it was sealed: {path}")


def _publish_bytes(path: Path, payload: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace non-identical artifact: {path}")
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish_json(path: Path, value: Any) -> None:
    _publish_bytes(path, _canonical_json(value).encode("utf-8"))


def _csv_payload(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _publish_csv(path: Path, frame: pd.DataFrame) -> None:
    _publish_bytes(path, _csv_payload(frame))


def _publish_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.is_file():
        if not pd.read_parquet(path).equals(frame):
            raise FileExistsError(f"Refusing to replace non-identical artifact: {path}")
        return
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        frame.to_parquet(temporary, index=False)
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _publish_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _lambda_key(value: float) -> str:
    return "infinity" if not np.isfinite(value) else format(float(value), ".12g")


def _analysis_environment() -> dict[str, Any]:
    import scipy
    import sklearn

    sources = (
        Path(__file__).resolve(),
        REPO / "aim2_residual_adaptation.py",
        REPO / "aim2_head_adaptation_base.py",
        REPO / "aim2_confirmatory_transfer.py",
        REPO / "aim2_cross_protocol_transfer.py",
        REPO / "src/oceanpath/aim1/lineage.py",
    )
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "pyarrow_version": importlib.metadata.version("pyarrow"),
        "blas": str(np.show_config(mode="dicts")),
        "implementation_sources": [_artifact(path) for path in sources],
    }


def _design_contract() -> dict[str, Any]:
    if tuple(LAMBDA_GRID) != tuple(e2c_offset.LAMBDA_GRID) or EPS != e2c_offset.EPS:
        raise ContractError("CPHT-A lambda grid/EPS differ from the exact v7 E2c reference")
    return {
        "experiment": "E2-CPHT-A Orion target-internal few-shot adaptation",
        "frozen_model": "raw all-conventional CPHT gated-ABMIL seeds 42/43/44",
        "adapter": "eta_native + H @ delta_w + delta_b per frozen model seed",
        "embedding_dimension": EMBED_DIM,
        "encoder_updates": 0,
        "mil_updates": 0,
        "outer_folds": N_FOLDS,
        "layout_seed": LAYOUT_SEED,
        "expected_test_per_fold": {"mutant": 3, "wild_type": 5},
        "support_source": "other four outer folds only",
        "budgets_per_class": list(BUDGETS),
        "complete_support_procedures_per_budget": N_PROCEDURES,
        "support_seed": SUPPORT_SEED,
        "support_seed_scheme": ("SHA256(20260822,E2-CPHT-A,support,budget,procedure,outer_fold)"),
        "lambda_selection": "leave-one-support-patient-out logistic loss",
        "lambda_grid_ordered_tie_preferred": [_lambda_key(value) for value in LAMBDA_GRID],
        "lambda_tie_rule": "strictly lower mean loss wins; exact ties retain earlier grid value",
        "loso_probability_clip": EPS,
        "loso_reference_implementation": "aim2_residual_adaptation.py",
        "infinity_rule": "exact native-logit copy; no residual arithmetic",
        # The preregistration inventory calls these 4,500 fold x seed fits
        # "outer procedures". They sit inside 300 complete procedures (100 per
        # budget), each of which supplies one AUROC.
        "outer_procedures": (len(BUDGETS) * N_PROCEDURES * N_FOLDS * len(SEEDS)),
        "outer_procedure_definition": "budget x support draw x outer fold x model seed",
        "point_estimand": "mean of 100 complete-procedure patient AUROCs",
        "forbidden_estimand": "AUROC of predictions averaged across support procedures",
        "primary_contrast": "expected adapted procedure AUROC minus native ensemble AUROC",
        "bootstrap": "paired ordinary-patient and support-procedure bootstrap",
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def _validate_label_blind_features(frame: pd.DataFrame) -> pd.DataFrame:
    forbidden = FORBIDDEN_FEATURE_COLUMNS & set(frame.columns)
    if forbidden:
        raise ContractError(f"Patient feature export leaks outcomes: {sorted(forbidden)}")
    required = {
        "seed",
        "patient_id",
        "logit",
        "n_slides",
        "exclude_neoadjuvant",
        "exclude_ambiguous_crc15",
        *(f"e{index}" for index in range(EMBED_DIM)),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError(f"Patient feature export lacks columns: {missing[:8]}")
    frame = frame.copy()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    if len(frame) != len(SEEDS) * EXPECTED_PATIENTS:
        raise ContractError("Patient feature export must contain 3 x 40 rows")
    if set(frame["seed"]) != set(SEEDS):
        raise ContractError("Patient feature export seed roster differs from 42/43/44")
    if frame.duplicated(["seed", "patient_id"]).any():
        raise ContractError("Patient feature export duplicates a seed/patient row")
    patient_sets = [
        set(frame.loc[frame["seed"].eq(seed), "patient_id"].astype(str)) for seed in SEEDS
    ]
    if any(len(items) != EXPECTED_PATIENTS for items in patient_sets):
        raise ContractError("Each seed must export exactly 40 patients")
    if any(items != patient_sets[0] for items in patient_sets[1:]):
        raise ContractError("Frozen model seeds cover different Orion patients")
    numeric = ["logit", *(f"e{index}" for index in range(EMBED_DIM))]
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise ContractError("Patient feature export contains non-finite values")
    flags = ["n_slides", "exclude_neoadjuvant", "exclude_ambiguous_crc15"]
    if frame.groupby("patient_id")[flags].nunique().to_numpy().max() != 1:
        raise ContractError("Patient-level flags differ across model seeds")
    per_patient = frame.drop_duplicates("patient_id")
    doubled = set(per_patient.loc[per_patient["n_slides"].eq(2), "patient_id"].astype(str))
    if (
        doubled != {"ORION:C33"}
        or not per_patient.loc[~per_patient["patient_id"].eq("ORION:C33"), "n_slides"].eq(1).all()
    ):
        raise ContractError("C33 must be the only two-slide Orion patient")
    return frame.sort_values(["seed", "patient_id"], kind="stable").reset_index(drop=True)


def _validate_upstream(output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate only sealed identities/receipts; do not read native logits or H."""

    seal_path = cpht.inference_seal_path(output_root)
    seal = _read_json(seal_path)
    if seal.get("target_outcomes_present") is not False:
        raise ContractError("CPHT inference seal is not label-blind")
    if seal.get("confirmatory_cpht_model_present") is not True:
        raise ContractError("CPHT inference seal lacks the all-conventional scorer")
    if seal.get("complete_eight_model_matrix") is not True:
        raise ContractError("CPHT inference seal lacks its declared complete score matrix")
    for key in (
        "execution_contract",
        "orion_manifest",
        "patient_features",
        "patient_features_receipt",
    ):
        identity = seal.get(key)
        if not isinstance(identity, dict):
            raise ContractError(f"CPHT inference seal lacks {key}")
        _validate_identity(identity, label=f"CPHT {key}")
    feature_path = Path(seal["patient_features"]["path"])
    receipt = _read_json(Path(seal["patient_features_receipt"]["path"]))
    if receipt.get("target_outcomes_opened") is not False:
        raise ContractError("CPHT patient feature receipt is not label-blind")
    if receipt.get("artifact") != _artifact(feature_path):
        raise ContractError("CPHT patient feature receipt does not bind its export")
    if int(receipt.get("n_patients", -1)) != EXPECTED_PATIENTS:
        raise ContractError("CPHT patient feature receipt has the wrong patient census")
    if (
        int(receipt.get("n_rows", -1)) != len(SEEDS) * EXPECTED_PATIENTS
        or int(receipt.get("embedding_dim", -1)) != EMBED_DIM
    ):
        raise ContractError("CPHT patient feature receipt has the wrong row/dimension census")
    return seal, receipt


def _upstream_contract(output_root: Path, seal: dict[str, Any]) -> dict[str, Any]:
    return {
        "cpht_inference_seal": _artifact(cpht.inference_seal_path(output_root)),
        "cpht_execution_contract": seal["execution_contract"],
        "cpht_orion_manifest": seal["orion_manifest"],
        "patient_features": seal["patient_features"],
        "patient_features_receipt": seal["patient_features_receipt"],
    }


def build_execution_contract(
    output_root: Path,
    label_source: Path,
    fold_manifest: pd.DataFrame,
    support_manifest: pd.DataFrame,
) -> dict[str, Any]:
    seal, feature_receipt = _validate_upstream(output_root)
    _validate_realized_design(fold_manifest, support_manifest)
    published = _load_realized_design(output_root)
    if not published.folds.equals(fold_manifest) or not published.supports.equals(support_manifest):
        raise ContractError("Published realized design differs from the in-memory design")
    return {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "sealed_realized_design_before_cpht_a_logit_analysis",
        "component": "aim2_a_orion_residual_adaptation",
        "component_output_name": _ACTIVE_COMPONENT_NAME,
        "output_root": str(output_root.resolve()),
        "upstream": _upstream_contract(output_root, seal),
        "label_source": _artifact(label_source),
        "realized_design": {
            "fold_manifest": _artifact(fold_manifest_path(output_root)),
            "support_manifest": _artifact(support_manifest_path(output_root)),
            "n_fold_rows": int(len(fold_manifest)),
            "n_support_rows": int(len(support_manifest)),
            "n_support_sets": int(
                support_manifest[["budget_per_class", "procedure_draw", "outer_fold"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "label_blind_feature_census": {
            "rows": int(feature_receipt["n_rows"]),
            "patients": int(feature_receipt["n_patients"]),
            "seeds": list(SEEDS),
            "embedding_dimension": int(feature_receipt["embedding_dim"]),
        },
        "design": _design_contract(),
        "analysis_environment": _analysis_environment(),
        "target_outcomes_parsed_during_seal": True,
        "outcome_use_during_seal": (
            "materialize immutable KRAS-stratified folds and seeded support manifests only"
        ),
        "patient_features_or_native_logits_parsed_during_seal": False,
        "prediction_metrics_computed_during_seal": False,
        "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
    }


def _load_execution_contract(output_root: Path) -> dict[str, Any]:
    contract = _read_json(contract_path(output_root))
    if contract.get("status") != "sealed_realized_design_before_cpht_a_logit_analysis":
        raise ContractError("CPHT-A execution contract has the wrong status")
    if (
        contract.get("target_outcomes_parsed_during_seal") is not True
        or contract.get("patient_features_or_native_logits_parsed_during_seal") is not False
        or contract.get("prediction_metrics_computed_during_seal") is not False
    ):
        raise ContractError("CPHT-A execution contract has a dishonest seal boundary")
    if contract.get("output_root") != str(output_root.resolve()):
        raise ContractError("--output-root differs from the CPHT-A execution contract")
    if contract.get("component_output_name") != _ACTIVE_COMPONENT_NAME:
        raise ContractError("--component-name differs from the CPHT-A execution contract")
    if contract.get("design") != _design_contract():
        raise ContractError("Current CPHT-A design constants differ from the seal")
    if contract.get("analysis_environment") != _analysis_environment():
        raise ContractError("CPHT-A code or numerical runtime changed after seal")
    for key, identity in contract["upstream"].items():
        _validate_identity(identity, label=key)
    _validate_identity(contract["label_source"], label="Orion label source")
    for key, identity in contract["realized_design"].items():
        if isinstance(identity, dict):
            _validate_identity(identity, label=f"realized design {key}")
    seal, feature_receipt = _validate_upstream(output_root)
    if contract["upstream"] != _upstream_contract(output_root, seal):
        raise ContractError("Current CPHT upstream identities differ from the CPHT-A seal")
    census = contract["label_blind_feature_census"]
    if (
        census.get("rows") != feature_receipt.get("n_rows")
        or census.get("patients") != feature_receipt.get("n_patients")
        or census.get("embedding_dimension") != feature_receipt.get("embedding_dim")
    ):
        raise ContractError("Current label-blind feature schema/census differs from the seal")
    _load_realized_design(output_root, contract)
    return contract


def stratified_folds(labels: np.ndarray, seed: int = LAYOUT_SEED) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels, dtype=int)
    if labels.shape != (EXPECTED_PATIENTS,) or int(labels.sum()) != EXPECTED_MUTANT:
        raise ValueError("The fixed Orion layout requires 40 patients, including 15 mutants")
    folds: np.ndarray = np.empty(len(labels), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        folds[test] = fold
    for fold in range(N_FOLDS):
        test_labels = labels[folds == fold]
        if len(test_labels) != 8 or int(test_labels.sum()) != 3:
            raise RuntimeError("Fixed Orion fold is not exactly 3 mutant / 5 wild type")
    return folds


def _generate_support_for_seal(
    labels: np.ndarray,
    folds: np.ndarray,
    budget: int,
    draw: int,
    outer_fold: int,
) -> tuple[np.ndarray, int]:
    if budget not in BUDGETS or draw not in range(N_PROCEDURES):
        raise ValueError("Support budget/draw is outside the governed roster")
    seed = stable_seed(SUPPORT_SEED, "E2-CPHT-A", "support", budget, draw, outer_fold)
    rng = np.random.default_rng(seed)
    picked: list[int] = []
    for label in (0, 1):
        candidates = np.flatnonzero((folds != outer_fold) & (labels == label))
        if len(candidates) < budget:
            raise RuntimeError("Other-fold support pool cannot provide exact balanced support")
        picked.extend(rng.choice(candidates, size=budget, replace=False).tolist())
    support = np.asarray(picked, dtype=np.int16)
    if (
        len(np.unique(support)) != 2 * budget
        or int(labels[support].sum()) != budget
        or np.any(folds[support] == outer_fold)
    ):
        raise RuntimeError("Exact balanced other-fold support contract failed")
    return support, seed


def realize_design(label_source: Path) -> RealizedDesign:
    """Parse outcomes only to freeze folds/supports; never open CPHT-A logits."""

    labels = prior_cpht._patient_labels(pd.read_csv(label_source))  # noqa: SLF001
    labels = labels.sort_values("patient_id", kind="stable").reset_index(drop=True)
    y = labels["label"].to_numpy(dtype=np.int8)
    folds = stratified_folds(y)
    fold_manifest = labels.assign(outer_fold=folds)[["patient_id", "label", "outer_fold"]].copy()
    fold_manifest["label"] = fold_manifest["label"].astype(int)
    fold_manifest["outer_fold"] = fold_manifest["outer_fold"].astype(int)
    rows: list[dict[str, Any]] = []
    patient_ids = fold_manifest["patient_id"].astype(str).to_numpy()
    for budget in BUDGETS:
        for draw in range(N_PROCEDURES):
            for outer_fold in range(N_FOLDS):
                support, seed = _generate_support_for_seal(y, folds, budget, draw, outer_fold)
                for ordinal, patient_index in enumerate(support):
                    rows.append(
                        {
                            "budget_per_class": budget,
                            "procedure_draw": draw,
                            "outer_fold": outer_fold,
                            "support_seed": str(seed),
                            "model_seed_roster": "42|43|44",
                            "support_ordinal": ordinal,
                            "patient_id": str(patient_ids[patient_index]),
                            "label": int(y[patient_index]),
                        }
                    )
    support_manifest = pd.DataFrame(rows)
    _validate_realized_design(fold_manifest, support_manifest)
    return RealizedDesign(folds=fold_manifest, supports=support_manifest)


def _validate_realized_design(folds: pd.DataFrame, supports: pd.DataFrame) -> None:
    expected_fold_columns = ["patient_id", "label", "outer_fold"]
    expected_support_columns = [
        "budget_per_class",
        "procedure_draw",
        "outer_fold",
        "support_seed",
        "model_seed_roster",
        "support_ordinal",
        "patient_id",
        "label",
    ]
    if list(folds.columns) != expected_fold_columns:
        raise ContractError("Realized fold manifest has an unexpected schema")
    if list(supports.columns) != expected_support_columns:
        raise ContractError("Realized support manifest has an unexpected schema")
    if (
        len(folds) != EXPECTED_PATIENTS
        or folds["patient_id"].duplicated().any()
        or int(folds["label"].sum()) != EXPECTED_MUTANT
        or set(folds["label"].astype(int)) != {0, 1}
        or set(folds["outer_fold"].astype(int)) != set(range(N_FOLDS))
    ):
        raise ContractError("Realized fold manifest violates the Orion census")
    if not folds["patient_id"].astype(str).is_monotonic_increasing:
        raise ContractError("Realized fold manifest patient order is not stable")
    for outer_fold in range(N_FOLDS):
        block = folds[folds["outer_fold"].eq(outer_fold)]
        if len(block) != 8 or int(block["label"].sum()) != 3:
            raise ContractError("Realized outer fold is not exactly 3 mutant / 5 wild type")

    expected_rows = N_PROCEDURES * N_FOLDS * sum(2 * budget for budget in BUDGETS)
    if len(supports) != expected_rows:
        raise ContractError(
            f"Realized support manifest has {len(supports)} rows, expected {expected_rows}"
        )
    fold_lookup = folds.set_index("patient_id")
    group_columns = ["budget_per_class", "procedure_draw", "outer_fold"]
    groups = supports.groupby(group_columns, sort=False)
    if groups.ngroups != len(BUDGETS) * N_PROCEDURES * N_FOLDS:
        raise ContractError("Realized support manifest lacks the complete 1,500-set roster")
    for (budget, draw, outer_fold), block in groups:
        budget = int(budget)
        draw = int(draw)
        outer_fold = int(outer_fold)
        if (
            budget not in BUDGETS
            or draw not in range(N_PROCEDURES)
            or outer_fold not in range(N_FOLDS)
        ):
            raise ContractError("Realized support manifest contains an out-of-roster key")
        expected_seed = stable_seed(SUPPORT_SEED, "E2-CPHT-A", "support", budget, draw, outer_fold)
        patient_ids = block["patient_id"].astype(str)
        if (
            len(block) != 2 * budget
            or patient_ids.duplicated().any()
            or list(block["support_ordinal"].astype(int)) != list(range(2 * budget))
            or set(block["support_seed"].astype(str)) != {str(expected_seed)}
            or set(block["model_seed_roster"].astype(str)) != {"42|43|44"}
            or set(patient_ids) - set(fold_lookup.index.astype(str))
        ):
            raise ContractError("Realized support set has invalid identities/seeds/ordinals")
        expected_labels = fold_lookup.loc[patient_ids, "label"].to_numpy(dtype=int)
        if (
            not np.array_equal(block["label"].to_numpy(dtype=int), expected_labels)
            or int(expected_labels.sum()) != budget
            or np.any(fold_lookup.loc[patient_ids, "outer_fold"].to_numpy(dtype=int) == outer_fold)
        ):
            raise ContractError("Realized support set is not balanced and other-fold-only")


def _load_realized_design(
    output_root: Path, contract: dict[str, Any] | None = None
) -> RealizedDesign:
    folds = pd.read_csv(fold_manifest_path(output_root), dtype={"patient_id": "string"})
    supports = pd.read_csv(
        support_manifest_path(output_root),
        dtype={
            "patient_id": "string",
            "support_seed": "string",
            "model_seed_roster": "string",
        },
    )
    folds["patient_id"] = folds["patient_id"].astype(str)
    supports["patient_id"] = supports["patient_id"].astype(str)
    supports["support_seed"] = supports["support_seed"].astype(str)
    supports["model_seed_roster"] = supports["model_seed_roster"].astype(str)
    _validate_realized_design(folds, supports)
    if contract is not None:
        design = contract["realized_design"]
        if (
            design.get("n_fold_rows") != len(folds)
            or design.get("n_support_rows") != len(supports)
            or design.get("n_support_sets") != len(BUDGETS) * N_PROCEDURES * N_FOLDS
        ):
            raise ContractError("Realized design census differs from its contract")
    return RealizedDesign(folds=folds, supports=supports)


def _support_from_manifest(
    design: RealizedDesign,
    data: PatientData,
    budget: int,
    draw: int,
    outer_fold: int,
) -> tuple[np.ndarray, int]:
    block = design.supports[
        design.supports["budget_per_class"].eq(budget)
        & design.supports["procedure_draw"].eq(draw)
        & design.supports["outer_fold"].eq(outer_fold)
    ].sort_values("support_ordinal", kind="stable")
    if len(block) != 2 * budget:
        raise ContractError("Frozen support manifest lookup is incomplete")
    index = {str(patient): ordinal for ordinal, patient in enumerate(data.patient_ids)}
    try:
        support = np.asarray(
            [index[str(patient)] for patient in block["patient_id"]], dtype=np.int16
        )
    except KeyError as error:
        raise ContractError("Frozen support patient is absent from CPHT features") from error
    return support, int(str(block.iloc[0]["support_seed"]))


def fit_residual(
    features: np.ndarray,
    offset: np.ndarray,
    labels: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Call the exact frozen v7 E2c native-offset residual implementation."""

    features = np.asarray(features, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if (
        features.ndim != 2
        or offset.shape != (len(features),)
        or labels.shape != (len(features),)
        or not np.isfinite(features).all()
        or not np.isfinite(offset).all()
        or not np.isfinite(labels).all()
    ):
        raise ValueError("Residual fit inputs are malformed or non-finite")
    weights, bias = e2c_offset.fit_native_offset(features, labels, offset, lam)
    reference = e2c_offset.native_offset_fit_diagnostics(
        features, labels, offset, weights, bias, lam
    )
    if reference["status"] == "native_exact":
        return (
            weights,
            bias,
            {
                "status": "native_exact",
                "gradient_inf_norm": None,
                "objective_decrease": 0.0,
                "delta_weight_l2": 0.0,
                "delta_bias": 0.0,
                "prediction_rule": "eta_native (bit-identical copy; no residual arithmetic)",
                "reference_implementation": "aim2_residual_adaptation.fit_native_offset",
            },
        )
    return (
        weights,
        bias,
        {
            "status": "finite_optimum",
            "gradient_inf_norm": float(reference["kkt_gradient_inf_norm"]),
            "objective_at_zero": float(reference["objective_at_zero_delta"]),
            "objective_at_fit": float(reference["objective_at_fit"]),
            "objective_decrease": float(reference["objective_decrease"]),
            "delta_weight_l2": float(reference["delta_weight_l2"]),
            "delta_bias": float(reference["delta_bias"]),
            "prediction_rule": "eta_native + H @ delta_w + delta_b",
            "accepted_by": "exact_v7_e2c_native_offset_solver",
            "reference_implementation": "aim2_residual_adaptation.fit_native_offset",
        },
    )


def _tie_preferred_index(mean_losses: np.ndarray) -> int:
    mean_losses = np.asarray(mean_losses, dtype=np.float64)
    if mean_losses.shape != (len(LAMBDA_GRID),) or not np.isfinite(mean_losses).all():
        raise ValueError("Lambda mean-loss vector is malformed")
    selected = 0
    best = float("inf")
    for index, loss in enumerate(mean_losses):
        if float(loss) < best:
            selected, best = index, float(loss)
    return selected


def select_lambda(
    features: np.ndarray,
    offset: np.ndarray,
    labels: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Exact leave-one-support-patient-out selection on the frozen grid."""

    labels = np.asarray(labels, dtype=np.float64)
    n_support = len(labels)
    if n_support not in {2 * value for value in BUDGETS}:
        raise ValueError("LOSO support must match a governed balanced budget")
    losses: np.ndarray = np.empty((len(LAMBDA_GRID), n_support), dtype=np.float64)
    for held_out in range(n_support):
        train = np.arange(n_support) != held_out
        if set(np.unique(labels[train])) != {0.0, 1.0}:
            raise RuntimeError("A LOSO training split lost a KRAS class")
        for lambda_index, lam in enumerate(LAMBDA_GRID):
            weights, bias, _ = fit_residual(features[train], offset[train], labels[train], lam)
            if np.isfinite(lam):
                eta = float(offset[held_out] + features[held_out] @ weights + bias)
            else:
                eta = float(offset[held_out])
            probability = float(
                np.clip(
                    1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0))),
                    EPS,
                    1.0 - EPS,
                )
            )
            losses[lambda_index, held_out] = float(
                -(
                    labels[held_out] * np.log(probability)
                    + (1.0 - labels[held_out]) * np.log(1.0 - probability)
                )
            )
    return _tie_preferred_index(losses.mean(axis=1)), losses


def _outcome_joined_data(features: pd.DataFrame, design: RealizedDesign) -> PatientData:
    """Join frozen feature rows to sealed realized labels/folds without regenerating them."""

    fold_manifest = design.folds.sort_values("patient_id", kind="stable").reset_index(drop=True)
    patient_ids = fold_manifest["patient_id"].astype(str).to_numpy()
    ecols = [f"e{index}" for index in range(EMBED_DIM)]
    matrices: list[np.ndarray] = []
    native: list[np.ndarray] = []
    for seed in SEEDS:
        block = features[features["seed"].eq(seed)].set_index("patient_id")
        try:
            aligned = block.loc[patient_ids]
        except KeyError as error:
            raise ContractError("Outcome and label-blind feature patient rosters differ") from error
        matrices.append(aligned[ecols].to_numpy(dtype=np.float64))
        native.append(aligned["logit"].to_numpy(dtype=np.float64))
    y = fold_manifest["label"].to_numpy(dtype=np.int8)
    return PatientData(
        patient_ids=patient_ids,
        labels=y,
        folds=fold_manifest["outer_fold"].to_numpy(dtype=np.int8),
        features=np.stack(matrices),
        native=np.stack(native),
    )


def compute_procedure(
    data: PatientData, design: RealizedDesign, budget: int, draw: int
) -> dict[str, np.ndarray]:
    """Compute one complete five-fold procedure for all three frozen seeds."""

    support_indices: np.ndarray = np.empty((N_FOLDS, 2 * budget), dtype=np.int16)
    support_seeds: np.ndarray = np.empty(N_FOLDS, dtype=np.uint64)
    lambda_indices: np.ndarray = np.empty((N_FOLDS, len(SEEDS)), dtype=np.int8)
    loso_losses: np.ndarray = np.empty(
        (N_FOLDS, len(SEEDS), len(LAMBDA_GRID), 2 * budget), dtype=np.float64
    )
    weights: np.ndarray = np.empty((N_FOLDS, len(SEEDS), EMBED_DIM), dtype=np.float64)
    biases: np.ndarray = np.empty((N_FOLDS, len(SEEDS)), dtype=np.float64)
    gradient_inf: np.ndarray = np.empty((N_FOLDS, len(SEEDS)), dtype=np.float64)
    objective_decrease: np.ndarray = np.empty((N_FOLDS, len(SEEDS)), dtype=np.float64)
    adapted = np.full_like(data.native, np.nan, dtype=np.float64)
    for fold in range(N_FOLDS):
        test = np.flatnonzero(data.folds == fold)
        support, seed = _support_from_manifest(design, data, budget, draw, fold)
        support_indices[fold] = support
        support_seeds[fold] = seed
        for seed_index, _model_seed in enumerate(SEEDS):
            selected, losses = select_lambda(
                data.features[seed_index, support],
                data.native[seed_index, support],
                data.labels[support],
            )
            lam = LAMBDA_GRID[selected]
            delta_w, delta_b, diagnostic = fit_residual(
                data.features[seed_index, support],
                data.native[seed_index, support],
                data.labels[support],
                lam,
            )
            if np.isfinite(lam):
                prediction = (
                    data.native[seed_index, test]
                    + data.features[seed_index, test] @ delta_w
                    + delta_b
                )
            else:
                prediction = data.native[seed_index, test].copy()
            adapted[seed_index, test] = prediction
            lambda_indices[fold, seed_index] = selected
            loso_losses[fold, seed_index] = losses
            weights[fold, seed_index] = delta_w
            biases[fold, seed_index] = delta_b
            gradient_inf[fold, seed_index] = (
                -1.0
                if diagnostic["gradient_inf_norm"] is None
                else float(diagnostic["gradient_inf_norm"])
            )
            objective_decrease[fold, seed_index] = float(diagnostic["objective_decrease"])
    if not np.isfinite(adapted).all():
        raise RuntimeError("A complete support procedure left patients unscored")
    return {
        "schema_version": np.asarray([1], dtype=np.int16),
        "budget_per_class": np.asarray([budget], dtype=np.int16),
        "procedure_draw": np.asarray([draw], dtype=np.int16),
        "patient_ids": data.patient_ids.astype("U"),
        "labels": data.labels.astype(np.int8),
        "folds": data.folds.astype(np.int8),
        "model_seeds": np.asarray(SEEDS, dtype=np.int16),
        "native_by_seed": data.native.astype(np.float64),
        "adapted_by_seed": adapted,
        "adapted_ensemble": adapted.mean(axis=0),
        "support_indices": support_indices,
        "support_seeds": support_seeds,
        "lambda_indices": lambda_indices,
        "loso_losses": loso_losses,
        "delta_weights": weights,
        "delta_biases": biases,
        "outer_gradient_inf": gradient_inf,
        "outer_objective_decrease": objective_decrease,
    }


def _expected_array_keys() -> set[str]:
    return {
        "schema_version",
        "budget_per_class",
        "procedure_draw",
        "patient_ids",
        "labels",
        "folds",
        "model_seeds",
        "native_by_seed",
        "adapted_by_seed",
        "adapted_ensemble",
        "support_indices",
        "support_seeds",
        "lambda_indices",
        "loso_losses",
        "delta_weights",
        "delta_biases",
        "outer_gradient_inf",
        "outer_objective_decrease",
    }


def audit_procedure(
    arrays: dict[str, np.ndarray],
    data: PatientData,
    design: RealizedDesign,
    budget: int,
    draw: int,
) -> dict[str, Any]:
    if set(arrays) != _expected_array_keys():
        raise ContractError("Procedure artifact has an unexpected array inventory")
    scalar_checks = {
        "schema_version": 1,
        "budget_per_class": budget,
        "procedure_draw": draw,
    }
    for key, expected in scalar_checks.items():
        if arrays[key].shape != (1,) or int(arrays[key][0]) != expected:
            raise ContractError(f"Procedure {key} differs from its governed value")
    if not np.array_equal(arrays["model_seeds"], np.asarray(SEEDS)):
        raise ContractError("Procedure model seed roster changed")
    for key, expected_array in (
        ("patient_ids", data.patient_ids),
        ("labels", data.labels),
        ("folds", data.folds),
        ("native_by_seed", data.native),
    ):
        if not np.array_equal(arrays[key], expected_array):
            raise ContractError(f"Procedure {key} differs from the sealed analysis input")
    expected_shapes = {
        "adapted_by_seed": (len(SEEDS), EXPECTED_PATIENTS),
        "adapted_ensemble": (EXPECTED_PATIENTS,),
        "support_indices": (N_FOLDS, 2 * budget),
        "support_seeds": (N_FOLDS,),
        "lambda_indices": (N_FOLDS, len(SEEDS)),
        "loso_losses": (N_FOLDS, len(SEEDS), len(LAMBDA_GRID), 2 * budget),
        "delta_weights": (N_FOLDS, len(SEEDS), EMBED_DIM),
        "delta_biases": (N_FOLDS, len(SEEDS)),
        "outer_gradient_inf": (N_FOLDS, len(SEEDS)),
        "outer_objective_decrease": (N_FOLDS, len(SEEDS)),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ContractError(f"Procedure {key} shape is {arrays[key].shape}, expected {shape}")
    numeric = [key for key in expected_shapes if key not in {"outer_gradient_inf"}]
    if any(not np.isfinite(arrays[key]).all() for key in numeric):
        raise ContractError("Procedure artifact contains non-finite numeric values")
    if not np.array_equal(arrays["adapted_ensemble"], arrays["adapted_by_seed"].mean(axis=0)):
        raise ContractError("Procedure ensemble is not the mean of three frozen seed logits")

    finite_outer = 0
    declined_outer = 0
    seen_test: list[int] = []
    for fold in range(N_FOLDS):
        test = np.flatnonzero(data.folds == fold)
        seen_test.extend(test.tolist())
        expected_support, expected_seed = _support_from_manifest(design, data, budget, draw, fold)
        support = arrays["support_indices"][fold]
        if not np.array_equal(support, expected_support):
            raise ContractError("Procedure support patients differ from the seeded draw")
        if int(arrays["support_seeds"][fold]) != expected_seed:
            raise ContractError("Procedure support seed differs from the seeded draw")
        for seed_index, _model_seed in enumerate(SEEDS):
            selected = int(arrays["lambda_indices"][fold, seed_index])
            if selected not in range(len(LAMBDA_GRID)):
                raise ContractError("Procedure selected a lambda outside the frozen grid")
            losses = arrays["loso_losses"][fold, seed_index]
            if selected != _tie_preferred_index(losses.mean(axis=1)):
                raise ContractError("Procedure violated ordered tie-preferred lambda selection")
            lam = LAMBDA_GRID[selected]
            delta_w = arrays["delta_weights"][fold, seed_index]
            delta_b = float(arrays["delta_biases"][fold, seed_index])
            if not np.isfinite(lam):
                declined_outer += 1
                if (
                    not np.array_equal(delta_w, np.zeros(EMBED_DIM))
                    or delta_b != 0.0
                    or float(arrays["outer_gradient_inf"][fold, seed_index]) != -1.0
                    or float(arrays["outer_objective_decrease"][fold, seed_index]) != 0.0
                    or not np.array_equal(
                        arrays["adapted_by_seed"][seed_index, test],
                        data.native[seed_index, test],
                    )
                ):
                    raise ContractError("Infinity did not use the bit-identical native branch")
                continue
            finite_outer += 1
            reference = e2c_offset.native_offset_fit_diagnostics(
                data.features[seed_index, support],
                data.labels[support],
                data.native[seed_index, support],
                delta_w,
                delta_b,
                lam,
            )
            gradient_observed = float(reference["kkt_gradient_inf_norm"])
            decrease = float(reference["objective_decrease"])
            if (
                gradient_observed > SOLVER_GRAD_TOL
                or decrease < -OBJECTIVE_TOL
                or abs(float(arrays["outer_gradient_inf"][fold, seed_index]) - gradient_observed)
                > 1e-12
                or abs(float(arrays["outer_objective_decrease"][fold, seed_index]) - decrease)
                > 1e-12
            ):
                raise ContractError("Finite outer residual fit fails its KKT/objective audit")
            expected_prediction = (
                data.native[seed_index, test] + data.features[seed_index, test] @ delta_w + delta_b
            )
            if not np.allclose(
                arrays["adapted_by_seed"][seed_index, test],
                expected_prediction,
                atol=1e-12,
                rtol=1e-12,
            ):
                raise ContractError("Stored adapted predictions do not match the residual head")
    if sorted(seen_test) != list(range(EXPECTED_PATIENTS)):
        raise ContractError("A procedure does not test every patient exactly once")
    return {
        "status": "PASS",
        "outer_adapter_decisions": N_FOLDS * len(SEEDS),
        "finite_outer_optimizer_fits": finite_outer,
        "declined_exact_native": declined_outer,
        "inner_loso_grid_evaluations": N_FOLDS * len(SEEDS) * 2 * budget * len(LAMBDA_GRID),
        "inner_finite_optimizer_fits": N_FOLDS * len(SEEDS) * 2 * budget * (len(LAMBDA_GRID) - 1),
    }


def _procedure_inputs(
    output_root: Path, contract: dict[str, Any], budget: int, draw: int
) -> dict[str, Any]:
    return {
        "execution_contract": _artifact(contract_path(output_root)),
        "patient_features": contract["upstream"]["patient_features"],
        "realized_fold_manifest": contract["realized_design"]["fold_manifest"],
        "realized_support_manifest": contract["realized_design"]["support_manifest"],
        "budget_per_class": budget,
        "procedure_draw": draw,
        "layout_seed": LAYOUT_SEED,
        "support_seed": SUPPORT_SEED,
    }


def _ensure_procedure(
    output_root: Path,
    contract: dict[str, Any],
    data: PatientData,
    design: RealizedDesign,
    budget: int,
    draw: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    artifact = procedure_path(output_root, budget, draw)
    receipt_path = procedure_receipt_path(output_root, budget, draw)
    expected_inputs = _procedure_inputs(output_root, contract, budget, draw)
    if receipt_path.exists() and not artifact.is_file():
        raise ContractError(f"Procedure receipt exists without its artifact: {receipt_path}")
    if artifact.is_file():
        arrays = _load_npz(artifact)
        audit = audit_procedure(arrays, data, design, budget, draw)
        if receipt_path.is_file():
            receipt = _read_json(receipt_path)
            if receipt.get("inputs") != expected_inputs or receipt.get("artifact") != _artifact(
                artifact
            ):
                raise ContractError(f"Procedure receipt identity mismatch: {receipt_path}")
            if receipt.get("audit") != audit:
                raise ContractError(f"Procedure receipt semantic audit mismatch: {receipt_path}")
            return receipt, arrays
        # Recover a fully-written, valid procedure after interruption between
        # the artifact and receipt links. The artifact itself is never replaced.
        receipt = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "complete",
            "inputs": expected_inputs,
            "artifact": _artifact(artifact),
            "audit": audit,
            "target_outcomes_opened": True,
            "encoder_updates": 0,
            "mil_updates": 0,
        }
        _publish_json(receipt_path, receipt)
        return receipt, arrays
    arrays = compute_procedure(data, design, budget, draw)
    audit = audit_procedure(arrays, data, design, budget, draw)
    _publish_npz(artifact, arrays)
    receipt = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "complete",
        "inputs": expected_inputs,
        "artifact": _artifact(artifact),
        "audit": audit,
        "target_outcomes_opened": True,
        "encoder_updates": 0,
        "mil_updates": 0,
    }
    _publish_json(receipt_path, receipt)
    return receipt, arrays


def _load_analysis_data(
    output_root: Path, label_source: Path
) -> tuple[dict[str, Any], PatientData, RealizedDesign]:
    contract = _load_execution_contract(output_root)
    if _artifact(label_source) != contract["label_source"]:
        raise ContractError("--label-source differs from the frozen outcome source")
    design = _load_realized_design(output_root, contract)
    feature_path = Path(contract["upstream"]["patient_features"]["path"])
    features = _validate_label_blind_features(pd.read_parquet(feature_path))
    return contract, _outcome_joined_data(features, design), design


def _current_procedure_roster(output_root: Path) -> list[dict[str, Any]]:
    roster: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for draw in range(N_PROCEDURES):
            artifact = procedure_path(output_root, budget, draw)
            receipt = procedure_receipt_path(output_root, budget, draw)
            if not artifact.is_file() or not receipt.is_file():
                raise ContractError(f"Incomplete procedure roster at k={budget}, draw={draw}")
            roster.append(
                {
                    "budget_per_class": budget,
                    "procedure_draw": draw,
                    "artifact": _artifact(artifact),
                    "receipt": _artifact(receipt),
                }
            )
    return roster


def verify_run_receipt(output_root: Path) -> dict[str, Any]:
    receipt = _read_json(run_receipt_path(output_root))
    if receipt.get("status") != "complete" or receipt.get("target_outcomes_opened") is not True:
        raise ContractError("CPHT-A run receipt is not complete")
    if receipt.get("execution_contract") != _artifact(contract_path(output_root)):
        raise ContractError("CPHT-A run receipt binds a different execution contract")
    roster = _current_procedure_roster(output_root)
    if receipt.get("procedures") != roster:
        raise ContractError("CPHT-A run receipt procedure roster changed")
    if (
        receipt.get("outer_procedures") != len(BUDGETS) * N_PROCEDURES * N_FOLDS * len(SEEDS)
        or receipt.get("mil_updates") != 0
        or receipt.get("encoder_updates") != 0
    ):
        raise ContractError("CPHT-A run receipt fit inventory is invalid")
    return receipt


def _auroc_rows(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = scores[None, :]
    negative = scores[:, labels == 0]
    positive = scores[:, labels == 1]
    pairwise = positive[:, :, None] - negative[:, None, :]
    result: np.ndarray = (
        (pairwise > 0).sum(axis=(1, 2)) + 0.5 * (pairwise == 0).sum(axis=(1, 2))
    ) / (positive.shape[1] * negative.shape[1])
    return result


def _bootstrap_layout(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    patients: np.ndarray = np.empty((N_BOOTSTRAP, len(labels)), dtype=np.int16)
    for bootstrap in range(N_BOOTSTRAP):
        for _ in range(1_000):
            candidate = rng.integers(0, len(labels), len(labels), dtype=np.int16)
            if len(np.unique(labels[candidate])) == 2:
                patients[bootstrap] = candidate
                break
        else:
            raise RuntimeError("Could not draw a two-class ordinary patient bootstrap")
    procedures = rng.integers(0, N_PROCEDURES, size=(N_BOOTSTRAP, N_PROCEDURES), dtype=np.int16)
    return patients, procedures


def _bootstrap_expected_auc(
    labels: np.ndarray,
    adapted: np.ndarray,
    native: np.ndarray,
    patient_indices: np.ndarray,
    procedure_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adapted_draw: np.ndarray = np.empty(N_BOOTSTRAP, dtype=np.float64)
    native_draw: np.ndarray = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for bootstrap, patient_index in enumerate(patient_indices):
        y = labels[patient_index]
        procedure_aurocs = _auroc_rows(y, adapted[:, patient_index])
        adapted_draw[bootstrap] = float(np.mean(procedure_aurocs[procedure_indices[bootstrap]]))
        native_draw[bootstrap] = float(_auroc_rows(y, native[patient_index])[0])
    return adapted_draw, native_draw, adapted_draw - native_draw


def _interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def analyse_procedures(output_root: Path, data: PatientData) -> tuple[dict[str, Any], pd.DataFrame]:
    native = data.native.mean(axis=0)
    native_auc = float(_auroc_rows(data.labels, native)[0])
    patient_boot, procedure_boot = _bootstrap_layout(data.labels)
    budget_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        matrices: list[np.ndarray] = []
        lambda_indices: list[np.ndarray] = []
        for draw in range(N_PROCEDURES):
            arrays = _load_npz(procedure_path(output_root, budget, draw))
            matrices.append(arrays["adapted_ensemble"])
            lambda_indices.append(arrays["lambda_indices"])
        adapted = np.vstack(matrices)
        per_procedure = _auroc_rows(data.labels, adapted)
        expected = float(np.mean(per_procedure))
        adapted_boot, native_boot, delta_boot = _bootstrap_expected_auc(
            data.labels, adapted, native, patient_boot, procedure_boot
        )
        selected = np.concatenate([item.ravel() for item in lambda_indices])
        census = {
            _lambda_key(value): int(np.sum(selected == index))
            for index, value in enumerate(LAMBDA_GRID)
        }
        budget_results[str(budget)] = {
            "budget_per_class": budget,
            "support_patients_total": 2 * budget,
            "n_complete_procedures": N_PROCEDURES,
            "outer_procedures": N_PROCEDURES * N_FOLDS * len(SEEDS),
            "support_failures": 0,
            "solver_failures": 0,
            "support_and_test_fold_integrity": "PASS",
            "native_auroc": native_auc,
            "native_auroc_ci95": _interval(native_boot),
            "expected_adapted_auroc": expected,
            "expected_adapted_auroc_ci95": _interval(adapted_boot),
            "adapted_minus_native_auroc": expected - native_auc,
            "adapted_minus_native_auroc_ci95": _interval(delta_boot),
            "bootstrap_probability_delta_above_zero": float(np.mean(delta_boot > 0)),
            "per_procedure_auroc_sd": float(np.std(per_procedure, ddof=1)),
            "per_procedure_auroc_p2p5_p97p5": _interval(per_procedure),
            "lambda_selection_census_across_1500_fold_seed_decisions": census,
            "lambda_fraction_infinity": float(np.mean(selected == 0)),
        }
        for draw, auroc in enumerate(per_procedure):
            rows.append(
                {
                    "budget_per_class": budget,
                    "procedure_draw": draw,
                    "adapted_auroc": float(auroc),
                    "native_auroc": native_auc,
                    "adapted_minus_native_auroc": float(auroc - native_auc),
                    "lambda_fraction_infinity": float(np.mean(lambda_indices[draw] == 0)),
                }
            )
    core = {
        "population": {
            "patients": EXPECTED_PATIENTS,
            "mutant": EXPECTED_MUTANT,
            "wild_type": EXPECTED_WILD_TYPE,
        },
        "fold_integrity": {
            "status": "PASS",
            "folds": N_FOLDS,
            "layout_seed": LAYOUT_SEED,
            "test_patients_per_fold": 8,
            "test_mutant_per_fold": 3,
            "test_wild_type_per_fold": 5,
        },
        "statistics": {
            "estimand": "mean of 100 complete-procedure AUROCs",
            "not_estimand": "AUROC of predictions averaged across support procedures",
            "bootstrap": "paired ordinary-patient and empirical support-procedure resampling",
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "shared_patient_and_procedure_indices_across_budgets": True,
        },
        "budgets": budget_results,
    }
    return core, pd.DataFrame(rows)


def _analysis_paths(output_root: Path) -> dict[str, Path]:
    root = analysis_root(output_root)
    return {
        "results": root / "results.json",
        "table": root / "results.csv",
        "procedure_aurocs": root / "procedure_aurocs.parquet",
        "receipt": root / "receipt.json",
    }


def _compact_results(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        block = results["budgets"][str(budget)]
        rows.append(
            {
                "budget_per_class": budget,
                "support_patients_total": 2 * budget,
                "n_complete_procedures": N_PROCEDURES,
                "native_auroc": block["native_auroc"],
                "expected_adapted_auroc": block["expected_adapted_auroc"],
                "adapted_minus_native_auroc": block["adapted_minus_native_auroc"],
                "delta_ci_low": block["adapted_minus_native_auroc_ci95"][0],
                "delta_ci_high": block["adapted_minus_native_auroc_ci95"][1],
                "lambda_fraction_infinity": block["lambda_fraction_infinity"],
            }
        )
    return pd.DataFrame(rows)


def _verify_analysis(output_root: Path, data: PatientData) -> dict[str, Any]:
    paths = _analysis_paths(output_root)
    if not all(path.is_file() for path in paths.values()):
        raise ContractError("CPHT-A analysis artifact set is incomplete")
    receipt = _read_json(paths["receipt"])
    if receipt.get("status") != "PASS":
        raise ContractError("CPHT-A analysis receipt is not PASS")
    if receipt.get("run_receipt") != _artifact(run_receipt_path(output_root)):
        raise ContractError("CPHT-A analysis binds a different run receipt")
    for key in ("results", "table", "procedure_aurocs"):
        if receipt.get("outputs", {}).get(key) != _artifact(paths[key]):
            raise ContractError(f"CPHT-A analysis {key} identity changed")
    results = _read_json(paths["results"])
    fresh_core, fresh_rows = analyse_procedures(output_root, data)
    for key in ("population", "fold_integrity", "statistics", "budgets"):
        if results.get(key) != fresh_core[key]:
            raise ContractError(f"CPHT-A analysis {key} does not recompute")
    stored_rows = pd.read_parquet(paths["procedure_aurocs"])
    if not stored_rows.equals(fresh_rows):
        raise ContractError("CPHT-A per-procedure AUROCs do not recompute")
    expected_table = _csv_payload(_compact_results(results))
    if paths["table"].read_bytes() != expected_table:
        raise ContractError("CPHT-A compact results table does not recompute")
    return receipt


def cmd_preflight(args: argparse.Namespace) -> None:
    seal, feature_receipt = _validate_upstream(args.output_root)
    design = realize_design(args.label_source)
    label_identity = _artifact(args.label_source)
    print(
        f"READY: sealed CPHT export {feature_receipt['n_rows']} rows / "
        f"{feature_receipt['n_patients']} patients / seeds {list(SEEDS)}"
    )
    print(f"  inference seal: {cpht.inference_seal_path(args.output_root)}")
    print(
        f"  outcome-informed design preview: {len(design.folds)} fold rows / "
        f"{len(design.supports)} support rows from {label_identity['path']}"
    )
    print(
        "  CPHT patient-feature bytes were identity-hashed but logits/embeddings were not "
        "semantically parsed or interpreted"
    )
    print(
        f"  governed analysis: {len(BUDGETS) * N_PROCEDURES} complete procedures; "
        f"{_design_contract()['outer_procedures']} preregistered outer procedures"
    )
    print(f"  upstream status: {seal['status']}")
    print(f"  caveat: {OUTCOME_ACCESS_CAVEAT}")


def cmd_seal(args: argparse.Namespace) -> None:
    destination = contract_path(args.output_root)
    if destination.is_file():
        _load_execution_contract(args.output_root)
        print(f"hash-valid realized-design CPHT-A contract already exists: {destination}")
        return
    design = realize_design(args.label_source)
    _publish_csv(fold_manifest_path(args.output_root), design.folds)
    _publish_csv(support_manifest_path(args.output_root), design.supports)
    contract = build_execution_contract(
        args.output_root, args.label_source, design.folds, design.supports
    )
    _publish_json(destination, contract)
    print(f"sealed outcome-informed folds/supports before CPHT-A logit analysis: {destination}")
    print(
        "CPHT patient-feature bytes were identity-hashed but logits/embeddings were not "
        "semantically parsed or interpreted"
    )
    print(f"caveat: {OUTCOME_ACCESS_CAVEAT}")


def cmd_run(args: argparse.Namespace) -> None:
    contract, data, design = _load_analysis_data(args.output_root, args.label_source)
    print(
        "ANALYSIS BOUNDARY: consuming sealed realized labels/folds/supports with frozen logits.",
        flush=True,
    )
    roster: list[dict[str, Any]] = []
    finite_outer = 0
    declined_outer = 0
    for budget in BUDGETS:
        print(f"  budget {budget}/class", flush=True)
        for draw in range(N_PROCEDURES):
            receipt, _arrays = _ensure_procedure(
                args.output_root, contract, data, design, budget, draw
            )
            finite_outer += int(receipt["audit"]["finite_outer_optimizer_fits"])
            declined_outer += int(receipt["audit"]["declined_exact_native"])
            roster.append(
                {
                    "budget_per_class": budget,
                    "procedure_draw": draw,
                    "artifact": _artifact(procedure_path(args.output_root, budget, draw)),
                    "receipt": _artifact(procedure_receipt_path(args.output_root, budget, draw)),
                }
            )
            if (draw + 1) % 10 == 0:
                print(f"    complete {draw + 1}/{N_PROCEDURES}", flush=True)
    value = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "complete",
        "execution_contract": _artifact(contract_path(args.output_root)),
        "realized_fold_manifest": contract["realized_design"]["fold_manifest"],
        "realized_support_manifest": contract["realized_design"]["support_manifest"],
        "target_outcomes_opened": True,
        "procedures": roster,
        "n_complete_procedures": len(BUDGETS) * N_PROCEDURES,
        "outer_procedures": len(BUDGETS) * N_PROCEDURES * N_FOLDS * len(SEEDS),
        "finite_outer_optimizer_fits": finite_outer,
        "declined_exact_native": declined_outer,
        "inner_loso_grid_evaluations": N_PROCEDURES
        * N_FOLDS
        * len(SEEDS)
        * sum(2 * budget for budget in BUDGETS)
        * len(LAMBDA_GRID),
        "inner_finite_optimizer_fits": N_PROCEDURES
        * N_FOLDS
        * len(SEEDS)
        * sum(2 * budget for budget in BUDGETS)
        * (len(LAMBDA_GRID) - 1),
        "encoder_updates": 0,
        "mil_updates": 0,
        "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
    }
    destination = run_receipt_path(args.output_root)
    if destination.is_file():
        recorded = _read_json(destination)
        for key, expected in value.items():
            if key != "created_utc" and recorded.get(key) != expected:
                raise ContractError(f"Existing CPHT-A run receipt differs at {key}")
    else:
        _publish_json(destination, value)
    verify_run_receipt(args.output_root)
    print(f"PASS: {len(roster)} immutable procedures; run receipt {destination}")


def cmd_report(args: argparse.Namespace) -> None:
    _contract, data, _design = _load_analysis_data(args.output_root, args.label_source)
    verify_run_receipt(args.output_root)
    paths = _analysis_paths(args.output_root)
    if paths["receipt"].is_file():
        _verify_analysis(args.output_root, data)
        print(f"hash-valid CPHT-A analysis already exists: {paths['results']}")
        return
    core, procedure_rows = analyse_procedures(args.output_root, data)
    if paths["results"].is_file():
        results = _read_json(paths["results"])
        for key, expected in core.items():
            if results.get(key) != expected:
                raise ContractError(f"Partial CPHT-A results differ at {key}")
        if (
            results.get("status") != "PASS"
            or results.get("analysis_environment") != _analysis_environment()
        ):
            raise ContractError("Partial CPHT-A results metadata is invalid")
    else:
        results = {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "PASS",
            "experiment": "E2-CPHT-A Orion target-internal few-shot residual adaptation",
            "interpretation": (
                "Exploratory target-internal adaptation feasibility; not zero-shot transport "
                "and not validation in a new cohort."
            ),
            "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
            "model_updates": {"encoder": 0, "mil": 0, "residual_linear_head_only": True},
            "analysis_environment": _analysis_environment(),
            **core,
        }
        _publish_json(paths["results"], results)
    _publish_parquet(paths["procedure_aurocs"], procedure_rows)
    _publish_csv(paths["table"], _compact_results(results))
    _publish_json(
        paths["receipt"],
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "status": "PASS",
            "run_receipt": _artifact(run_receipt_path(args.output_root)),
            "realized_fold_manifest": _artifact(fold_manifest_path(args.output_root)),
            "realized_support_manifest": _artifact(support_manifest_path(args.output_root)),
            "outputs": {key: _artifact(path) for key, path in paths.items() if key != "receipt"},
            "analysis_environment": _analysis_environment(),
            "outcome_access_caveat": OUTCOME_ACCESS_CAVEAT,
        },
    )
    _verify_analysis(args.output_root, data)
    for budget in BUDGETS:
        block = results["budgets"][str(budget)]
        low, high = block["adapted_minus_native_auroc_ci95"]
        print(
            f"k={budget}/class: adapted {block['expected_adapted_auroc']:.4f}; "
            f"native {block['native_auroc']:.4f}; delta "
            f"{block['adapted_minus_native_auroc']:+.4f} [{low:+.4f}, {high:+.4f}]"
        )
    print(f"complete CPHT-A analysis: {paths['results']}")


def cmd_verify(args: argparse.Namespace) -> None:
    _contract, data, design = _load_analysis_data(args.output_root, args.label_source)
    run_receipt = verify_run_receipt(args.output_root)
    finite_outer = 0
    declined_outer = 0
    for budget in BUDGETS:
        for draw in range(N_PROCEDURES):
            arrays = _load_npz(procedure_path(args.output_root, budget, draw))
            audit = audit_procedure(arrays, data, design, budget, draw)
            receipt = _read_json(procedure_receipt_path(args.output_root, budget, draw))
            if receipt.get("audit") != audit:
                raise ContractError("Stored procedure audit does not reproduce")
            finite_outer += int(audit["finite_outer_optimizer_fits"])
            declined_outer += int(audit["declined_exact_native"])
    if (
        finite_outer != run_receipt["finite_outer_optimizer_fits"]
        or declined_outer != run_receipt["declined_exact_native"]
    ):
        raise ContractError("Aggregate outer residual-fit census does not reproduce")
    analysis_status = "not_yet_reported"
    if analysis_root(args.output_root).exists():
        _verify_analysis(args.output_root, data)
        analysis_status = "PASS"
    print(
        "PASS: outcome-informed realized-design seal, 300 procedure receipts, "
        "4,500 fold-seed adapter "
        f"decisions, exact infinity branches, and analysis={analysis_status}"
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--label-source", type=Path, default=LABEL_SOURCE)
    parser.add_argument("--component-name", default=DEFAULT_COMPONENT_NAME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        "preflight": cmd_preflight,
        "seal": cmd_seal,
        "run": cmd_run,
        "report": cmd_report,
        "verify": cmd_verify,
    }
    for name, handler in commands.items():
        child = subparsers.add_parser(name)
        _add_common(child)
        child.set_defaults(func=handler)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _activate_component_name(args.component_name)
    args.func(args)


if __name__ == "__main__":
    main()
