from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools import aim1_tcga_surgen_two_encoder_recovery_v2 as recovery  # noqa: E402


def _identity(path: str = "/tmp/example", *, sha: str = "a" * 64, size: int = 1):
    return {"path": path, "sha256": sha, "size_bytes": size}


def _synthetic_census(root: Path, records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "root": str(root.resolve()),
        "excluded_prefix": "recovery_v1/",
        "artifact_count": len(records),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "tree_sha256": recovery._canonical_sha256(records),
        "artifacts": records,
    }


def _synthetic_v1(census: dict[str, object]) -> dict[str, object]:
    return {
        "adjudication": {
            "raw_artifact_hash_census": {
                "before": census,
                "after_expected_identical": copy.deepcopy(census),
                "publication_exclusion": "recovery_v1/",
            }
        }
    }


def _synthetic_evidence(root: Path) -> dict[str, object]:
    terminal = _identity(str(root / "recovery_v1/receipts/training_complete_recovered.json"))
    adjudication = _identity(str(root / "recovery_v1/receipts/validator_adjudication.json"))
    scope = {
        "root": str(root.resolve()),
        **recovery.ORIGINAL_CENSUS,
        "original_census_source": adjudication,
        "original_census_json_pointer": "/raw_artifact_hash_census/before",
        "all_original_records_rehashed_at_certification": True,
        "closed_roster_outside_exclusions": True,
    }
    baseline = {
        **recovery.PREPARED_BASELINE,
        "root": str(root.resolve()),
        "canonical_record_schema": "sorted campaign-relative {path,sha256,size_bytes}",
        "artifacts": [
            {
                "path": str((root / relative).resolve()),
                "sha256": sha256,
                "size_bytes": size,
            }
            for relative, (sha256, size) in sorted(recovery.PINNED_PREPARED_BASELINE.items())
        ],
    }
    return {
        "root": root,
        "v1": {
            "implementation": {},
            "artifacts": {
                "receipts/training_complete_recovered.json": terminal,
                "receipts/validator_adjudication.json": adjudication,
            },
        },
        "scope": scope,
        "baseline": baseline,
    }


def test_public_surface_and_exact_terminal_roster() -> None:
    assert Path("recovery_v2/receipts/training_complete_scoped.json") == (
        recovery.RECOVERY_V2_TERMINAL
    )
    assert recovery.RECOVERY_STATUS == "complete_and_certified_via_scoped_census_erratum"
    assert recovery.AUTHORIZED_EXCLUDED_NAMESPACES == (
        "recovery_v1",
        "recovery_v2",
        "downstream",
        "downstream_v2",
    )
    assert {
        "schema_version",
        "recovery",
        "status",
        "created_utc",
        "base_campaign",
        "scope_contract",
        "scope_adjudication",
        "recovery_implementation",
        "predecessor_v1_terminal",
        "training_scoped_census",
        "prepared_downstream_baseline",
        "namespace_policy",
        "fit_accounting",
        "execution_accounting",
        "concurrency",
        "certification_boundary",
    } == recovery.TERMINAL_FIELDS


def test_original_and_baseline_frozen_scalars() -> None:
    assert recovery.ORIGINAL_CENSUS == {
        "artifact_count": 441,
        "total_size_bytes": 989_858_771,
        "tree_sha256": "a2ffd5b61eaf65dc8d0c5df6a1b29867ffabc57f0af869120603ea37591ac261",
    }
    records = [
        {"path": relative, "sha256": sha256, "size_bytes": size}
        for relative, (sha256, size) in sorted(recovery.PINNED_PREPARED_BASELINE.items())
    ]
    assert recovery._canonical_sha256(records) == recovery.PREPARED_BASELINE["tree_sha256"]
    assert sum(record["size_bytes"] for record in records) == 95_802


def test_frozen_v1_implementation_bytes_match() -> None:
    for spec in recovery.PINNED_V1_IMPLEMENTATION.values():
        observed = recovery._artifact(Path(spec["path"]))
        assert observed["sha256"] == spec["sha256"]
        assert observed["size_bytes"] == spec["size_bytes"]


@pytest.mark.parametrize(
    "bad_path",
    ("../escape.json", "/absolute.json", "a//b.json", "a/./b.json", ".", "downstream/x"),
)
def test_original_census_rejects_traversal_and_excluded_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_path: str
) -> None:
    records = [{"path": bad_path, "sha256": "a" * 64, "size_bytes": 1}]
    census = _synthetic_census(tmp_path, records)
    monkeypatch.setattr(
        recovery,
        "ORIGINAL_CENSUS",
        {
            "artifact_count": 1,
            "total_size_bytes": 1,
            "tree_sha256": recovery._canonical_sha256(records),
        },
    )
    with pytest.raises(recovery.ContractError, match="escapes|malformed"):
        recovery._original_census(_synthetic_v1(census), tmp_path)


def test_original_census_accepts_exact_sorted_unique_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        {"path": "a.json", "sha256": "a" * 64, "size_bytes": 1},
        {"path": "train/b.bin", "sha256": "b" * 64, "size_bytes": 2},
    ]
    census = _synthetic_census(tmp_path, records)
    monkeypatch.setattr(
        recovery,
        "ORIGINAL_CENSUS",
        {
            "artifact_count": 2,
            "total_size_bytes": 3,
            "tree_sha256": recovery._canonical_sha256(records),
        },
    )
    assert recovery._original_census(_synthetic_v1(census), tmp_path) == census


@pytest.mark.parametrize("mutation", ("before_after", "duplicate", "unsorted", "extra_key"))
def test_original_census_rejects_structural_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    records = [
        {"path": "a", "sha256": "a" * 64, "size_bytes": 1},
        {"path": "b", "sha256": "b" * 64, "size_bytes": 2},
    ]
    census = _synthetic_census(tmp_path, records)
    monkeypatch.setattr(
        recovery,
        "ORIGINAL_CENSUS",
        {
            "artifact_count": 2,
            "total_size_bytes": 3,
            "tree_sha256": recovery._canonical_sha256(records),
        },
    )
    v1 = _synthetic_v1(census)
    raw = v1["adjudication"]["raw_artifact_hash_census"]
    if mutation == "before_after":
        raw["after_expected_identical"]["tree_sha256"] = "0" * 64
    elif mutation == "duplicate":
        duplicated = [records[0], records[0]]
        raw["before"].update(_synthetic_census(tmp_path, duplicated))
        raw["after_expected_identical"] = copy.deepcopy(raw["before"])
    elif mutation == "unsorted":
        reversed_records = list(reversed(records))
        raw["before"].update(_synthetic_census(tmp_path, reversed_records))
        raw["after_expected_identical"] = copy.deepcopy(raw["before"])
    else:
        raw["before"]["unexpected"] = True
        raw["after_expected_identical"] = copy.deepcopy(raw["before"])
    with pytest.raises(recovery.ContractError):
        recovery._original_census(v1, tmp_path)


def test_scoped_census_prunes_only_exact_authorized_namespaces(tmp_path: Path) -> None:
    (tmp_path / "contract.json").write_text("alpha", encoding="utf-8")
    for name in recovery.AUTHORIZED_EXCLUDED_NAMESPACES:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "ignored.bin").write_text(name, encoding="utf-8")
    near_miss = tmp_path / "downstream_extra"
    near_miss.mkdir()
    (near_miss / "governed.bin").write_text("beta", encoding="utf-8")
    census = recovery._current_scoped_census(tmp_path)
    assert [record["path"] for record in census["artifacts"]] == [
        "contract.json",
        "downstream_extra/governed.bin",
    ]


def test_scoped_census_rejects_symlink_in_immutable_scope(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_text("bytes", encoding="utf-8")
    (tmp_path / "alias.bin").symlink_to(target)
    with pytest.raises(recovery.ContractError, match="symlink"):
        recovery._current_scoped_census(tmp_path)


def test_shallow_namespace_boundary_rejects_root_analysis_and_unknown_top_level(
    tmp_path: Path,
) -> None:
    (tmp_path / "contract.json").write_text("sealed", encoding="utf-8")
    (tmp_path / "recovery_v1").mkdir()
    (tmp_path / "downstream").mkdir()
    original = {"artifacts": [{"path": "contract.json"}]}
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    with pytest.raises(recovery.ContractError, match="Root analysis"):
        recovery._validate_namespace_boundaries(tmp_path, original, require_v2_absent=False)
    analysis.rmdir()
    (tmp_path / "unknown").mkdir()
    with pytest.raises(recovery.ContractError, match="roster is open"):
        recovery._validate_namespace_boundaries(tmp_path, original, require_v2_absent=False)


def test_prepared_baseline_is_exact_and_only_downstream_v2_may_grow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downstream = tmp_path / "downstream"
    downstream.mkdir()
    baseline = downstream / "contract.json"
    baseline.write_text("sealed", encoding="utf-8")
    spec = {"downstream/contract.json": (recovery.campaign._sha256(baseline), 6)}
    records = [
        {
            "path": "downstream/contract.json",
            "sha256": spec["downstream/contract.json"][0],
            "size_bytes": 6,
        }
    ]
    monkeypatch.setattr(recovery, "PINNED_PREPARED_BASELINE", spec)
    monkeypatch.setattr(
        recovery,
        "PREPARED_BASELINE",
        {
            "artifact_count": 1,
            "total_size_bytes": 6,
            "tree_sha256": recovery._canonical_sha256(records),
        },
    )
    expected_absolute = [{**records[0], "path": str(baseline.resolve())}]
    assert (
        recovery._validate_downstream_baseline(tmp_path, require_continuation_absent=True)[
            "artifacts"
        ]
        == expected_absolute
    )
    (downstream / "future.score").write_text("delegated", encoding="utf-8")
    with pytest.raises(recovery.ContractError, match="exactly the immutable seven-file"):
        recovery._validate_downstream_baseline(tmp_path, require_continuation_absent=False)
    (downstream / "future.score").unlink()
    continuation = tmp_path / "downstream_v2"
    continuation.mkdir()
    (continuation / "future.score").write_text("delegated", encoding="utf-8")
    assert (
        recovery._validate_downstream_baseline(tmp_path, require_continuation_absent=False)[
            "artifacts"
        ]
        == expected_absolute
    )
    with pytest.raises(recovery.ContractError, match="requires absent downstream_v2"):
        recovery._validate_downstream_baseline(tmp_path, require_continuation_absent=True)
    baseline.write_text("drift!", encoding="utf-8")
    with pytest.raises(recovery.ContractError, match="SHA drifted|size drifted"):
        recovery._validate_downstream_baseline(tmp_path, require_continuation_absent=False)


def test_delegated_growth_rejects_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    downstream = tmp_path / "downstream"
    downstream.mkdir()
    baseline = downstream / "contract.json"
    baseline.write_text("sealed", encoding="utf-8")
    record = {
        "path": "downstream/contract.json",
        "sha256": recovery.campaign._sha256(baseline),
        "size_bytes": 6,
    }
    monkeypatch.setattr(
        recovery,
        "PINNED_PREPARED_BASELINE",
        {record["path"]: (record["sha256"], record["size_bytes"])},
    )
    monkeypatch.setattr(
        recovery,
        "PREPARED_BASELINE",
        {
            "artifact_count": 1,
            "total_size_bytes": 6,
            "tree_sha256": recovery._canonical_sha256([record]),
        },
    )
    target = tmp_path / "target"
    target.write_text("x", encoding="utf-8")
    (downstream / "future-link").symlink_to(target)
    with pytest.raises(recovery.ContractError, match="symlink"):
        recovery._validate_downstream_baseline(tmp_path, require_continuation_absent=False)


def test_shallow_audit_does_not_rehash_scope_or_replay_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = {"root": str(tmp_path), "artifacts": [], **recovery.ORIGINAL_CENSUS}
    v1 = {"artifacts": {}, "terminal": {}, "adjudication": {}}
    baseline = {"artifacts": [], **recovery.PREPARED_BASELINE}
    monkeypatch.setattr(recovery, "_assert_root", lambda _: tmp_path)
    monkeypatch.setattr(recovery, "_pinned_v1_namespace", lambda _: v1)
    monkeypatch.setattr(recovery, "_original_census", lambda _v1, _root: original)
    monkeypatch.setattr(
        recovery,
        "_validate_downstream_baseline",
        lambda _root, **_kwargs: baseline,
    )
    monkeypatch.setattr(recovery, "_validate_namespace_boundaries", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(recovery, "_scope_summary", lambda _: {"scope": "summary"})
    monkeypatch.setattr(
        recovery,
        "_current_scoped_census",
        lambda _: pytest.fail("shallow audit rehashed the immutable scope"),
    )
    monkeypatch.setattr(
        recovery,
        "_reconstruct_and_verify_v1",
        lambda *_: pytest.fail("shallow audit replayed the full v1 graph"),
    )
    evidence = recovery._audit_scope(tmp_path, deep_scope=False, require_v2_absent=False)
    assert evidence["baseline"] == baseline


def test_deep_audit_rejects_changed_or_added_scoped_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_record = {"path": "a", "sha256": "a" * 64, "size_bytes": 1}
    original = {
        "root": str(tmp_path),
        "artifacts": [original_record],
        **recovery.ORIGINAL_CENSUS,
    }
    v1 = {"artifacts": {}, "terminal": {}, "adjudication": {}}
    monkeypatch.setattr(recovery, "_assert_root", lambda _: tmp_path)
    monkeypatch.setattr(recovery, "_pinned_v1_namespace", lambda _: v1)
    monkeypatch.setattr(recovery, "_original_census", lambda _v1, _root: original)
    monkeypatch.setattr(
        recovery,
        "_validate_downstream_baseline",
        lambda _root, **_kwargs: {"artifacts": [], **recovery.PREPARED_BASELINE},
    )
    monkeypatch.setattr(recovery, "_validate_namespace_boundaries", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        recovery,
        "_current_scoped_census",
        lambda _: {
            "artifact_count": 1,
            "total_size_bytes": 1,
            "tree_sha256": "0" * 64,
            "artifacts": [{**original_record, "sha256": "b" * 64}],
        },
    )
    with pytest.raises(recovery.ContractError, match="changed"):
        recovery._audit_scope(tmp_path, deep_scope=True, require_v2_absent=False)


def test_materialized_three_file_graph_replays(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    evidence = _synthetic_evidence(root)
    recovery._materialize(root, stage, evidence, created_utc="2026-08-25T00:00:00+00:00")
    assert {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()} == {
        "contract_scope_erratum.json",
        "receipts/scope_adjudication.json",
        "receipts/training_complete_scoped.json",
    }
    stage.rename(recovery.recovery_v2_dir(root))
    terminal = recovery._verify_v2_files(root, evidence)
    assert set(terminal) == recovery.TERMINAL_FIELDS
    assert terminal["status"] == recovery.RECOVERY_STATUS


def test_staged_semantic_replay_rejects_tamper(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    stage = tmp_path / "stage"
    stage.mkdir()
    evidence = _synthetic_evidence(root)
    recovery._materialize(root, stage, evidence, created_utc="2026-08-25T00:00:00+00:00")
    contract_path = stage / "contract_scope_erratum.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["status"] = "tampered"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(recovery.ContractError, match="scope contract does not replay"):
        recovery._verify_staged(root, stage, evidence)


def test_atomic_publication_is_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    evidence = _synthetic_evidence(root)
    monkeypatch.setattr(
        recovery,
        "_audit_scope",
        lambda *_args, **_kwargs: evidence,
    )
    verified = []
    original_verify = recovery._verify_v2_files

    def verify(*args, **kwargs):
        verified.append(True)
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(recovery, "_verify_v2_files", verify)
    terminal = recovery._publish_atomic(root, evidence)
    assert recovery.scoped_terminal_path(root).is_file()
    assert terminal["status"] == recovery.RECOVERY_STATUS
    assert verified == [True]
    with pytest.raises(recovery.ContractError, match="exactly once"):
        recovery._publish_atomic(root, evidence)


def test_alias_forwards_deep_pack_without_swallowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        recovery,
        "validate_scoped_terminal",
        lambda root, *, deep_scope: calls.append((root, deep_scope)) or {"ok": True},
    )
    assert recovery.validate_recovered_terminal_v2(tmp_path, deep_pack=False) == {"ok": True}
    assert calls == [(tmp_path, False)]


def test_certify_requires_explicit_apply() -> None:
    args = recovery.build_parser().parse_args(["certify"])
    with pytest.raises(recovery.ContractError, match="requires --apply"):
        recovery.cmd_certify(args)


LIVE_ROOT = recovery.DEFAULT_OUTPUT_ROOT


@pytest.mark.skipif(
    not recovery.recovery_v1.recovered_terminal_path(LIVE_ROOT).is_file(),
    reason="published recovery-v1 terminal unavailable",
)
def test_live_prepublication_scope_audit_is_read_only() -> None:
    if recovery.recovery_v2_dir(LIVE_ROOT).exists():
        terminal = recovery.validate_scoped_terminal(LIVE_ROOT, deep_scope=True)
        assert terminal["status"] == recovery.RECOVERY_STATUS
    else:
        evidence = recovery._audit_scope(
            LIVE_ROOT,
            deep_scope=True,
            require_v2_absent=True,
        )
        assert evidence["scope"]["artifact_count"] == 441
        assert evidence["scope"]["tree_sha256"] == recovery.ORIGINAL_CENSUS["tree_sha256"]
        assert evidence["baseline"]["artifact_count"] == 7
        assert not recovery.recovery_v2_dir(LIVE_ROOT).exists()


def test_no_production_apply() -> None:
    assert not recovery.recovery_v2_dir(recovery.DEFAULT_OUTPUT_ROOT).exists()
