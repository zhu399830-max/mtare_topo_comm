from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_replacement():
    import sys

    tools = ROOT / "tools/v3"
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "run_aee_domain_sensor_export_v1r",
        tools / "run_aee_domain_sensor_export_v1r.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_replacement_binds_exact_verified_host_environment() -> None:
    module = load_replacement()
    module.verify_host_python()
    assert module.RUN_ID.endswith("aee_domain_sensor_export_v1r_seed20260820")
    assert module.EXPECTED_PYTHON == Path(
        "/home/zeng-workstation/anaconda3/bin/python"
    ).resolve()
    assert module.EXPECTED_NUMPY_VERSION == "2.1.3"


def test_replacement_rejects_interpreter_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_replacement()
    monkeypatch.setattr(module.sys, "executable", "/usr/bin/python3")
    with pytest.raises(RuntimeError, match="replacement requires"):
        module.verify_host_python()


def test_replacement_rejects_numpy_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_replacement()
    monkeypatch.setattr(module.np, "__version__", "0.0.0")
    with pytest.raises(RuntimeError, match="NumPy identity drift"):
        module.verify_host_python()


def test_replacement_main_forwards_once_with_only_new_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_replacement()
    calls: list[str] = []
    monkeypatch.setattr(module, "verify_host_python", lambda: None)
    monkeypatch.setattr(
        module.implementation,
        "main",
        lambda: calls.append(module.implementation.RUN_ID) or 7,
    )

    assert module.main() == 7
    assert calls == [module.RUN_ID]
    assert module.RUN_ID != "gate2_20260820_aee_domain_sensor_export_v1_seed20260820"


def test_environment_failure_never_enters_core_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_replacement()
    calls: list[bool] = []
    monkeypatch.setattr(module.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(module.implementation, "main", lambda: calls.append(True) or 0)

    with pytest.raises(RuntimeError, match="replacement requires"):
        module.main()
    assert calls == []


def test_replacement_proposal_keeps_v1_contract_and_hashes() -> None:
    proposal = json.loads(
        (ROOT / "configs/v3/gate2/aee_domain_sensor_export_v1r.json").read_text(
            encoding="utf-8"
        )
    )
    command = proposal["command"]
    assert "/home/zeng-workstation/anaconda3/bin/python" in command
    assert "tools/v3/run_aee_domain_sensor_export_v1r.py" in command
    assert command[-1].endswith("aee_domain_sensor_export_v1r_seed20260820")
    assert proposal["scientific_contract_changes"] == []
    assert proposal["replacement_for"].endswith(
        "aee_domain_sensor_export_v1_seed20260820"
    )
    for item in proposal["frozen_tools"].values():
        payload = (ROOT / item["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
