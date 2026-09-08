"""Mount-independent tests for the FINAL-v14 pre-reader coordinator."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "tools/final_v14_pre_reader.py"
SPEC = importlib.util.spec_from_file_location("final_v14_pre_reader_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def _support(counts: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prototype_id": np.arange(RUNNER.CANONICAL_K, dtype=np.int64),
            "distinct_patient_support": counts,
        }
    )


def _candidate_table(*, patients_per_subcohort: int = 6) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for prototype in range(RUNNER.CANONICAL_K):
        for group_index, subcohort in enumerate(RUNNER.SOURCE_SUBCOHORTS):
            for patient_index in range(patients_per_subcohort):
                patient_id = f"P{prototype:02d}-{group_index}-{patient_index:02d}"
                records.append(
                    {
                        "prototype_id": prototype,
                        "patient_id": patient_id,
                        "subcohort": subcohort,
                        "slide_id": f"S-{patient_id}",
                        "tile_id": f"{patient_index:012d}",
                        "tile_index": patient_index,
                        "x": 256 * patient_index,
                        "y": 256 * group_index,
                        "distance": float(group_index * 100 + patient_index),
                        # Force selection through the prespecified all-patient tier.
                        "eligibility_tier": 2,
                        "eligibility_stage": "all",
                    }
                )
    return pd.DataFrame.from_records(records)


def _code(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    digits = ["A"] * 6
    value = index
    for position in range(5, -1, -1):
        digits[position] = alphabet[value % len(alphabet)]
        value //= len(alphabet)
    return "".join(digits)


def _jpeg_payload() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 768), (242, 242, 242)).save(
        buffer,
        format="JPEG",
        quality=92,
        subsampling=0,
        optimize=False,
        progressive=False,
        exif=b"",
    )
    return buffer.getvalue()


def _public_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, bytes]]:
    codes = [_code(index) for index in range(40)]
    blinding = pd.DataFrame(
        {
            "presentation_order": np.arange(40, dtype=np.int64),
            "code": codes,
        }
    )
    provenance = pd.DataFrame(
        {
            "code": np.repeat(codes, 12),
            "montage_slot": np.tile(np.arange(12, dtype=np.int64), 40),
        }
    )
    payload = _jpeg_payload()
    rendered = {code: payload for code in codes}
    return blinding, provenance, rendered


def test_duplicate_preflight_selection_is_frozen_and_order_independent() -> None:
    support = _support([24] * RUNNER.CANONICAL_K)
    expected = sorted(
        int(value)
        for value in np.random.Generator(np.random.PCG64(RUNNER.VOCAB_SEED)).choice(
            np.arange(RUNNER.CANONICAL_K, dtype=np.int64), size=8, replace=False
        )
    )

    selected, status = RUNNER._reader_duplicate_selection(support)
    shuffled, shuffled_status = RUNNER._reader_duplicate_selection(
        support.sample(frac=1, random_state=91)
    )

    assert selected == expected
    assert shuffled == expected
    assert status == shuffled_status == "DUPLICATE_PREFLIGHT_PASS"


def test_duplicate_preflight_fails_sticky_and_uses_best_remaining_support() -> None:
    counts = [12] * 6 + [11, 10] + [1] * (RUNNER.CANONICAL_K - 8)
    selected, status = RUNNER._reader_duplicate_selection(_support(counts))

    assert selected == list(range(8))
    assert status == "NAME_GATE_FAIL_INSUFFICIENT_DUPLICATE_SUPPORT"


def test_reader_selections_are_deterministic_balanced_and_disjoint() -> None:
    candidates = _candidate_table()
    duplicated, _ = RUNNER._reader_duplicate_selection(
        _support([24] * RUNNER.CANONICAL_K)
    )

    selections, audit = RUNNER._reader_selections(candidates, duplicated)
    replay, replay_audit = RUNNER._reader_selections(
        candidates.sample(frac=1, random_state=20260819), duplicated
    )

    assert len(selections) == len(replay) == 40
    pd.testing.assert_frame_equal(audit, replay_audit)
    assert sorted(audit["n_displayed_tiles"].unique().tolist()) == [12]
    assert set(audit["selection_stage"]) == {"all"}
    for key, selection in selections.items():
        replay_selection = replay[key]
        pd.testing.assert_frame_equal(selection.tiles, replay_selection.tiles)
        assert selection.tiles["patient_id"].is_unique
        assert selection.tiles["subcohort"].value_counts().to_dict() == {
            subcohort: 3 for subcohort in RUNNER.SOURCE_SUBCOHORTS
        }

    for prototype in duplicated:
        original = selections[(prototype, 0)]
        duplicate = selections[(prototype, 1)]
        assert set(original.tiles["patient_id"]).isdisjoint(
            set(duplicate.tiles["patient_id"])
        )
        assert duplicate.overlap_count == 0


def test_patient_profile_frame_uses_equal_slide_not_tile_weighting() -> None:
    roster = pd.DataFrame(
        {
            "slide_id": ["s-z", "s-a", "s-b"],
            "patient_id": ["patient-2", "patient-1", "patient-1"],
            "subcohort": ["SR1482", "SR386", "SR386"],
            "fold": [1, 0, 0],
            "n_tiles": [100, 1000, 1],
        }
    )
    slide_profiles = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    slides = RUNNER._profile_frame(
        roster, slide_profiles, vocabulary_id="reference"
    )
    patients = RUNNER._patient_profile_frame(slides, vocabulary_id="reference")

    assert patients["patient_id"].tolist() == ["patient-1", "patient-2"]
    assert patients["n_slides"].tolist() == [2, 1]
    np.testing.assert_allclose(
        patients[["prototype_00", "prototype_01"]].to_numpy(),
        np.asarray([[0.5, 0.5], [1.0, 0.0]]),
        rtol=0,
        atol=0,
    )
    assert np.allclose(
        patients[["prototype_00", "prototype_01"]].sum(axis=1), 1.0
    )


def test_write_once_and_atomic_checkpoint_replays_are_fail_closed(
    tmp_path: Path,
) -> None:
    immutable = tmp_path / "nested/immutable.bin"
    RUNNER.write_once(immutable, b"sealed")
    RUNNER.write_once(immutable, b"sealed")
    with pytest.raises(RUNNER.ContractError, match="nonidentical artifact"):
        RUNNER.write_once(immutable, b"drifted")
    assert immutable.read_bytes() == b"sealed"

    checkpoint = tmp_path / "nested/checkpoint.bin"
    RUNNER.atomic_write(checkpoint, b"complete")
    RUNNER.atomic_write(checkpoint, b"complete")
    with pytest.raises(RUNNER.ContractError, match="existing checkpoint"):
        RUNNER.atomic_write(checkpoint, b"partial")
    assert checkpoint.read_bytes() == b"complete"


def test_assignment_checkpoint_recovers_crash_between_npz_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeStore:
        def length_of(self, slide_id: str) -> int:
            assert slide_id == "slide-1"
            return 3

        def position(self, slide_id: str) -> int:
            assert slide_id == "slide-1"
            return 0

        def read_features(self, position: int) -> np.ndarray:
            assert position == 0
            return np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)

    class FakeVocabulary:
        def __init__(self, offset: int) -> None:
            self.offset = offset

        def assign_with_distances(
            self, features: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
            labels = (
                np.arange(len(features), dtype=np.int16) + self.offset
            ) % RUNNER.CANONICAL_K
            return labels, np.arange(len(features), dtype=np.float32)

    shard_root = tmp_path / "assignments"
    vocabularies = {
        "reference": FakeVocabulary(0),
        **{
            f"outer_fold_{fold}": FakeVocabulary(fold + 1)
            for fold in RUNNER.FOLDS
        },
    }
    monkeypatch.setattr(RUNNER, "_ASSIGNMENT_STORE", FakeStore())
    monkeypatch.setattr(RUNNER, "_ASSIGNMENT_SHARD_ROOT", shard_root)
    monkeypatch.setattr(RUNNER, "_ASSIGNMENT_VOCABULARIES", vocabularies)
    monkeypatch.setattr(
        RUNNER,
        "_ASSIGNMENT_VOCABULARY_DIGESTS",
        {key: f"digest-{key}" for key in vocabularies},
    )
    monkeypatch.setattr(
        RUNNER,
        "_ASSIGNMENT_DEPENDENCIES",
        {"contract": "frozen"},
    )

    first = RUNNER._assign_one_slide("slide-1")
    checkpoint = Path(first["path"])
    receipt = Path(first["receipt_path"])
    with np.load(checkpoint, allow_pickle=False) as bundle:
        original_arrays = {key: bundle[key].copy() for key in bundle.files}
    receipt.unlink()

    recovered = RUNNER._assign_one_slide("slide-1")
    assert recovered["action"] == "computed"
    assert receipt.is_file()
    with np.load(checkpoint, allow_pickle=False) as bundle:
        assert set(bundle.files) == set(original_arrays)
        for key, expected in original_arrays.items():
            np.testing.assert_array_equal(bundle[key], expected)

    replay = RUNNER._assign_one_slide("slide-1")
    assert replay["action"] == "replayed"


def test_publish_directory_accepts_exact_replay_and_rejects_drift(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"
    first_stage = tmp_path / "first-stage"
    first_stage.mkdir()
    (first_stage / "artifact.txt").write_text("sealed\n", encoding="utf-8")
    RUNNER._publish_directory(first_stage, destination)
    assert not first_stage.exists()

    replay_stage = tmp_path / "replay-stage"
    replay_stage.mkdir()
    (replay_stage / "artifact.txt").write_text("sealed\n", encoding="utf-8")
    RUNNER._publish_directory(replay_stage, destination)
    assert not replay_stage.exists()

    drift_stage = tmp_path / "drift-stage"
    drift_stage.mkdir()
    (drift_stage / "artifact.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RUNNER.ContractError, match="nonidentical directory"):
        RUNNER._publish_directory(drift_stage, destination)
    assert (destination / "artifact.txt").read_text(encoding="utf-8") == "sealed\n"


def test_public_reader_package_builds_validates_and_detects_tampering(
    tmp_path: Path,
) -> None:
    blinding, provenance, rendered = _public_inputs()
    public = tmp_path / "FOR_PATHOLOGIST"
    RUNNER._build_public_reader_directory(
        public,
        blinding,
        provenance,
        rendered,
        "a" * 64,
        sealed_utc="2026-09-03T12:00:00+00:00",
    )
    RUNNER._build_public_reader_directory(
        public,
        blinding,
        provenance,
        rendered,
        "a" * 64,
        sealed_utc="2026-09-03T12:00:00+00:00",
    )
    manifest = RUNNER.json.loads(
        (public / "HANDOFF_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["sealed_utc"] == "2026-09-03T12:00:00+00:00"
    with pytest.raises(RUNNER.ContractError, match="timestamp differs"):
        RUNNER._build_public_reader_directory(
            public,
            blinding,
            provenance,
            rendered,
            "a" * 64,
            sealed_utc="2026-09-03T12:00:01+00:00",
        )
    roster = pd.DataFrame(
        {
            "patient_id": ["PATIENT-IDENTIFIER-0001"],
            "slide_id": ["SLIDE-IDENTIFIER-0001"],
        }
    )

    validation = RUNNER._validate_public_reader_directory(public, roster)
    assert validation["status"] == "PASS"
    assert validation["montages"] == validation["form_rows"] == 40
    form = pd.read_csv(public / "review_form.csv", keep_default_na=False)
    assert form["presentation_order"].tolist() == list(range(1, 41))
    assert form["n_tiles"].tolist() == [12] * 40
    assert all(value == "" for value in form["primary_category"])
    instructions = public / "INSTRUCTIONS.md"
    instruction_text = instructions.read_text(encoding="utf-8")
    assert "`review_status` to exactly `complete`" in instruction_text
    assert "`artifact_uninterpretable` to exactly `yes` or `no`" in instruction_text
    assert "`blinding_attestation` to exactly `confirmed_no_key_access`" in instruction_text
    assert "`YYYY-MM-DD`" in instruction_text

    instructions.write_text(
        instructions.read_text(encoding="utf-8") + "\npatient_id\n",
        encoding="utf-8",
    )
    with pytest.raises(RUNNER.ContractError):
        RUNNER._validate_public_reader_directory(public, roster)


def test_public_tree_inventory_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    target = root / "target.txt"
    target.write_text("data", encoding="utf-8")
    (root / "link.txt").symlink_to(target)

    with pytest.raises(RUNNER.ContractError, match="symlink"):
        RUNNER._tree_inventory(root)


def test_receipt_replay_validates_bound_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"frozen")
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(
        RUNNER.json_bytes(
            {
                "status": "PASS",
                "artifacts": {"result": RUNNER.identity(artifact)},
            }
        )
    )

    replay = RUNNER._load_json_receipt_if_valid(
        receipt, status="PASS", artifact_keys=("result",)
    )
    assert replay is not None
    assert replay["artifacts"]["result"]["sha256"] == RUNNER.sha256_file(artifact)

    artifact.write_bytes(b"tampered")
    with pytest.raises(RUNNER.ContractError, match="fails replay"):
        RUNNER._load_json_receipt_if_valid(
            receipt, status="PASS", artifact_keys=("result",)
        )
