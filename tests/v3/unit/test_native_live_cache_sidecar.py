import json
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace as NS

from tests.v3.unit.test_live_structural_token_cache import fixture


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/v3"))


class _Stamp:
    def __init__(self, value):
        self.value = value

    def to_nsec(self):
        return self.value


def _ros_modules(callbacks, published, shutdown, spin, publisher_topics):
    rospy = types.ModuleType("rospy")
    rospy.Time = NS(now=lambda: _Stamp(1100))
    rospy.init_node = lambda *args, **kwargs: None
    def publisher(topic, *args, **kwargs):
        publisher_topics.append(topic)
        return NS(publish=lambda message: published.append(message.data))

    rospy.Publisher = publisher

    def subscribe(topic, msgtype, callback, **kwargs):
        callbacks[topic] = callback
        return NS(unregister=lambda: None)

    rospy.Subscriber = subscribe
    rospy.spin = spin
    rospy.signal_shutdown = lambda reason: shutdown.append(reason)
    rospy.logerr = rospy.logwarn = lambda *args: None
    std_msgs = types.ModuleType("std_msgs")
    messages = types.ModuleType("std_msgs.msg")
    messages.String = lambda data: NS(data=data)
    return {"rospy": rospy, "std_msgs": std_msgs, "std_msgs.msg": messages}


def test_live_ready_token_is_ingested_before_identity_advice(tmp_path, monkeypatch):
    _cache, event, snapshot = fixture(tmp_path)
    import native_structure_advice_node as node
    from mtare_topo.integration.live_structural_token_cache import LiveStructuralTokenCache

    callbacks, published, shutdown, publisher_topics = {}, [], [], []
    ingested = threading.Event()
    original_ingest = LiveStructuralTokenCache.ingest

    def ingest_and_signal(self, value, *, received_monotonic_ns):
        result = original_ingest(self, value, received_monotonic_ns=received_monotonic_ns)
        ingested.set()
        return result

    monkeypatch.setattr(LiveStructuralTokenCache, "ingest", ingest_and_signal)

    def spin():
        callbacks["/native_structure/token_ready"](
            NS(data=json.dumps(event), _connection_header={"callerid": "/native_structure_live_inputs"})
        )
        assert ingested.wait(timeout=2.0), "live token loader did not ingest within bound"
        callbacks["/sensor_coverage_planner/native_structure/candidates"](
            NS(data=json.dumps(snapshot))
        )

    for name, module in _ros_modules(callbacks, published, shutdown, spin, publisher_topics).items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(node, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "sidecar"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_structure_advice_node",
            "--output",
            str(output),
            "--segment",
            "live",
            "--live-token-artifacts",
            "tokens",
            "--live-token-inputs",
            "inputs",
            "--live-token-max-entries",
            "2",
        ],
    )

    assert node.main() == 0
    assert not shutdown
    assert len(published) == 1
    assert publisher_topics == ["/sensor_coverage_planner/native_structure/advice"]
    advice = json.loads(published[0])
    assert advice["route"] == snapshot["original_route"]

    rows = [json.loads(row) for row in (output / "trace.jsonl").read_text().splitlines()]
    candidate = next(row for row in rows if row["kind"] == "candidate_response")
    assert candidate["identity_fallback"] is True
    assert candidate["live_token_status"]["ready"] is True
    assert candidate["live_token_used_for_route"] is False
    summary = json.loads((output / "summary.json").read_text())
    assert summary["live_token_counts"]["loaded"] == 1
    assert summary["live_token_counts"]["lookup_ready"] == 1
    assert summary["counts"]["advice_published"] == 1
    assert summary["waypoint_published"] is False
    assert json.loads((output / "ready.json").read_text())["control_topics"] == []


def test_wrong_live_token_publisher_rejected_without_advice(tmp_path, monkeypatch):
    _cache, event, _snapshot = fixture(tmp_path)
    import native_structure_advice_node as node

    callbacks, published, shutdown, publisher_topics = {}, [], [], []

    def spin():
        callbacks["/native_structure/token_ready"](
            NS(data=json.dumps(event), _connection_header={"callerid": "/wrong"})
        )

    for name, module in _ros_modules(callbacks, published, shutdown, spin, publisher_topics).items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(node, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "sidecar"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "native_structure_advice_node",
            "--output",
            str(output),
            "--segment",
            "live",
            "--live-token-artifacts",
            "tokens",
            "--live-token-inputs",
            "inputs",
            "--live-token-max-entries",
            "2",
        ],
    )

    assert node.main() == 1
    assert not published
    assert publisher_topics == ["/sensor_coverage_planner/native_structure/advice"]
    summary = json.loads((output / "summary.json").read_text())
    assert summary["live_token_counts"]["failures"] == 1
    assert summary["live_token_counts"]["loaded"] == 0
    assert summary["counts"]["advice_published"] == 0
