"""Bounded Unix-socket integration tests for the native token stream service.

All payloads are the existing synthetic five-frame fixture.  The extractor is
replaced at the loader boundary, so these tests exercise the service, transport,
source binding, and evidence persistence without loading a real checkpoint.
"""

import hashlib
import importlib
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from mtare_topo.integration.structural_token_transport import StructuralTokenClient
from tests.v3.unit.test_frozen_structural_token_worker import FakeOutput
from tests.v3.unit.test_streaming_structural_tokens import request


def _service_module():
    return importlib.import_module("native_structure_token_process")


def _start_service(tmp_path, monkeypatch, *, max_requests=2, max_wall_seconds=5.0):
    service = _service_module()
    se3 = importlib.import_module("mtare_topo.integration.frozen_structural_se3_worker")
    calls = []

    class FakeExtractor:
        def extract(self, student, **kwargs):
            calls.append(kwargs)
            return FakeOutput(np.arange(2), np.ones((2, 128), dtype=np.float32))

    def load(root, *, device):
        calls.append(("load", root, device))
        return FakeExtractor()

    monkeypatch.setattr(service, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(se3, "load_registered_extractor", load)
    output = tmp_path / "service-output"
    args = SimpleNamespace(
        preprocessing="full_relative_se3_v1",
        manifest=None,
        manifest_sha256=None,
        max_requests=max_requests,
        max_wall_seconds=max_wall_seconds,
        stream_socket=tmp_path / "native-token.sock",
        stream_input_root=tmp_path,
        output=output,
    )
    result = []
    errors = []

    def serve():
        try:
            result.append(service.serve_stream(args))
        except BaseException as exc:  # surface worker failures in the test thread
            errors.append(exc)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    ready = output / "ready.json"
    deadline = time.monotonic() + 2
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists(), "stream service did not publish readiness"
    return thread, result, errors, calls, args


def _join_service(thread, result, errors):
    thread.join(timeout=3)
    assert not thread.is_alive(), "bounded stream service did not stop"
    assert not errors, errors
    assert len(result) == 1
    return result[0]


def test_socket_stream_two_requests_one_load_persists_outputs_and_stops(tmp_path, monkeypatch):
    thread, result, errors, calls, args = _start_service(tmp_path, monkeypatch)

    first = StructuralTokenClient(args.stream_socket, timeout_s=2).request(request(tmp_path, 0))
    second = StructuralTokenClient(args.stream_socket, timeout_s=2).request(request(tmp_path, 1))
    summary = _join_service(thread, result, errors)

    assert sum(item[0] == "load" for item in calls if isinstance(item, tuple)) == 1
    assert len(calls) == 3  # one loader call and one synthetic forward per request
    assert first["response"]["status"] == second["response"]["status"] == "PROCESSED"
    assert summary["counts"] == {
        "received": 2,
        "responses_delivered": 2,
        "client_disconnected": 0,
        "rejected_requests": 0,
        "model_forwards": 2,
        "processed_windows": 2,
    }
    assert summary["stop_reason"] == "REQUEST_LIMIT"
    assert json.loads((args.output / "ready.json").read_text())["weights_loaded"] is True

    for client_result in (first, second):
        response = client_result["response"]
        record = Path(response["output_record"])
        assert record.exists()
        assert hashlib.sha256(record.read_bytes()).hexdigest() == response["output_record_sha256"]
        assert json.loads(record.read_text())["status"] == "PROCESSED"
        assert response["token_ref"] is not None
        token = Path(response["token_ref"]["path"])
        assert hashlib.sha256(token.read_bytes()).hexdigest() == response["token_ref"]["sha256"]

    stream_rows = [json.loads(line) for line in (args.output / "stream.jsonl").read_text().splitlines()]
    assert len(stream_rows) == 2
    assert all(row["delivered"] for row in stream_rows)
    assert json.loads((args.output / "stream_summary.json").read_text())["stop_reason"] == "REQUEST_LIMIT"


def test_socket_stream_preserves_record_rejection_and_continues_within_bound(tmp_path, monkeypatch):
    thread, result, errors, calls, args = _start_service(tmp_path, monkeypatch)

    rejected = StructuralTokenClient(args.stream_socket, timeout_s=2).request(
        request(tmp_path, 0, bad_pose=True)
    )["response"]
    accepted = StructuralTokenClient(args.stream_socket, timeout_s=2).request(request(tmp_path, 1))["response"]
    summary = _join_service(thread, result, errors)

    assert rejected["status"] == "REJECTED_RECORD"
    assert "proper sensor rotations" in rejected["reason"]
    assert rejected["token_ref"] is None
    assert accepted["status"] == "PROCESSED"
    assert summary["counts"]["received"] == 2
    assert summary["counts"]["processed_windows"] == 1
    assert summary["counts"]["model_forwards"] == 1
    assert summary["stop_reason"] == "REQUEST_LIMIT"
    assert len(calls) == 2  # preload plus the one accepted synthetic forward
    assert (args.output / "window_000000" / "record_000000.json").exists()
    assert json.loads((args.output / "window_000000" / "record_000000.json").read_text())["status"] == "REJECTED_RECORD"
