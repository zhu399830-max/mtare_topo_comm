"""V1R regression: preserve venv invocation, resolve only binary identity.

Every executable here is an inert temporary fixture file; subprocess calls are
mocked. No actual virtual environment, source corpus, or array payload is read.
"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from tests.v3.unit import test_gse_surface_input_runner_v1 as original


def load_runner_v1r():
    folder = Path(__file__).resolve().parents[3] / "tools/v3"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location("surface_input_v1r_symlink_regression", folder / "run_gse_surface_input_export_v1r.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def symlink_fixture(tmp_path, monkeypatch):
    real = tmp_path / "base/bin/python"
    real.parent.mkdir(parents=True); real.write_bytes(b"inert synthetic Python binary")
    invoked = tmp_path / "venv/bin/python"
    invoked.parent.mkdir(parents=True); invoked.symlink_to(real)
    assert invoked.resolve() == real and invoked != real
    monkeypatch.setattr(sys, "executable", str(invoked))
    monkeypatch.setattr(original, "load_runner", load_runner_v1r)
    f = original.fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(f.module.platform, "platform", lambda: "synthetic-platform")
    assert f.spec["python_executable_sha256"] == f.module.sha(real) == f.module.sha(invoked)
    seen = []
    def environment_sensitive_pip(command, **kwargs):
        seen.append(command)
        assert command[1:] == ["-m", "pip", "freeze", "--all"]
        assert kwargs == {"text":True}
        if command[0] == str(invoked):
            return "synthetic-test-only==0\n"  # already frozen by original fixture
        if command[0] == str(real):
            return "different-base-environment==0\n"
        raise AssertionError("unexpected interpreter path")
    monkeypatch.setattr(f.module.subprocess, "check_output", environment_sensitive_pip)
    return f, invoked, real, seen


def test_pip_uses_invoked_symlink_not_resolved_binary_and_full_export_completes(tmp_path, monkeypatch):
    f, invoked, real, seen = symlink_fixture(tmp_path, monkeypatch)
    assert f.module.execute(f.spec, f.run, tmp_path) == 0
    assert len(seen) == 1 and seen[0][0] == sys.executable == str(invoked)
    assert seen[0][0] != str(real)
    assert f.calls == {"compiler":1,"reader":1,"tasks":1}
    environment = json.loads((f.run/"config/execution_environment.json").read_text())
    assert environment["executable_sha256"] == f.module.sha(real)
    assert environment["sidecar_freeze_sha256"] == original.digest_bytes(b"synthetic-test-only==0\n")
    assert json.loads((f.run/"RUN_STATE.json").read_text())["state"] == "COMPLETED"
    sealed = original.assert_sealed(f)
    with pytest.raises(ValueError, match="no overwrite/retry"):
        f.module.execute(f.spec, f.run, tmp_path)
    assert (f.run/"artifacts/evidence_sha256.txt").read_bytes() == sealed


def test_same_binary_hash_does_not_excuse_wrong_package_environment(tmp_path, monkeypatch):
    f, invoked, real, seen = symlink_fixture(tmp_path, monkeypatch)
    def drift(command, **kwargs):
        seen.append(command)
        assert command[0] == sys.executable == str(invoked)
        return "different-base-environment==0\n"
    monkeypatch.setattr(f.module.subprocess, "check_output", drift)
    assert f.module.execute(f.spec, f.run, tmp_path) == 1
    assert f.module.sha(real) == f.spec["python_executable_sha256"]
    assert f.calls == {"compiler":0,"reader":0,"tasks":0}
    assert "data-sidecar environment drift" in (f.run/"logs/error.log").read_text()
    assert json.loads((f.run/"RUN_STATE.json").read_text())["state"] == "FAILED"
    original.assert_sealed(f)


def test_binary_integrity_still_checked_independently_of_correct_venv_pip(tmp_path, monkeypatch):
    f, invoked, real, seen = symlink_fixture(tmp_path, monkeypatch)
    real.write_bytes(b"changed underlying binary, same venv path")
    assert f.module.execute(f.spec, f.run, tmp_path) == 1
    assert seen[0][0] == str(invoked)
    assert f.calls == {"compiler":0,"reader":0,"tasks":0}
    assert "data-sidecar environment drift" in (f.run/"logs/error.log").read_text()
    original.assert_sealed(f)
