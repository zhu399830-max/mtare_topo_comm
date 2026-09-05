"""Post-run synthetic hardening, separate from the frozen metadata test file."""
from pathlib import Path

from tests.v3.unit.test_gse_head_development_selection import (
    test_real_runner_freezes180_rows_actual_population_and_seal as _run_synthetic_e2e,
)


def test_unregistered_and_nonmetadata_seal_entries_are_never_resolved(tmp_path, monkeypatch):
    original_write = Path.write_text
    original_resolve = Path.resolve
    source_seal = tmp_path / "fixtures/evidence_sha256.txt"
    seal_written = False

    def append_unregistered_entry(path, text, *args, **kwargs):
        nonlocal seal_written
        if path == source_seal:
            text += "0" * 64 + "  fixtures/teacher/fit/S99_unregistered_C07__c1_mixed.zarr/frame_row/0.0\n"
            seal_written = True
        return original_write(path, text, *args, **kwargs)

    def forbid_nonmetadata_resolution(path, *args, **kwargs):
        # Fixture creation is allowed to create forbidden dummy arrays. Once
        # its source seal exists, executing the audit must not resolve them.
        if seal_written:
            assert "S99_unregistered" not in str(path)
            assert "range_m" not in path.parts
            assert "axis_control_current_sensor_m" not in path.parts
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", append_unregistered_entry)
    monkeypatch.setattr(Path, "resolve", forbid_nonmetadata_resolution)
    _run_synthetic_e2e(tmp_path, monkeypatch)
    assert seal_written
