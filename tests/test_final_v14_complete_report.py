"""The renderer must not overwrite complete or partially sealed reports."""
import pytest

from tools import final_v14_complete_report as report


@pytest.mark.parametrize("marker", ["source_manifest.json", "final_bundle_receipt.json", "final_bundle_receipt.json.seal.json"])
def test_sealed_or_partially_sealed_report_blocks_before_read_or_write(tmp_path, monkeypatch, marker):
    (tmp_path / marker).write_bytes(b"immutable marker")
    (tmp_path / "Results.md").write_bytes(b"preserved scientific report")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(report, "load_inputs", lambda: pytest.fail("must block before reading or rendering inputs"))
    with pytest.raises(ValueError, match="completed or partially sealed"):
        report.render(tmp_path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
