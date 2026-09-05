"""Method-independent ROS bag recording and post-run inventory contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


METHODS = ("original_mtare", "m1d_topology", "layered_gt_map_oracle")


@dataclass(frozen=True)
class TopicContract:
    name: str
    type: str
    required: bool
    purpose: str


@dataclass(frozen=True)
class TopicObservation:
    type: str
    messages: int

    def __post_init__(self) -> None:
        if not self.type or self.messages < 0:
            raise ValueError("topic observations require a type and non-negative message count")


@dataclass(frozen=True)
class RecordingAudit:
    passed: bool
    missing_topics: tuple[str, ...]
    empty_required_topics: tuple[str, ...]
    type_mismatches: tuple[str, ...]
    observed_required_messages: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_topic_contract(path: str | Path) -> tuple[TopicContract, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "mtare_closed_loop_recording_topics_v1":
        raise ValueError("unexpected closed-loop recording schema")
    if tuple(payload.get("methods", ())) != METHODS:
        raise ValueError("method ordering or identity drift")
    topics = tuple(TopicContract(**item) for item in payload.get("topic_contract", ()))
    names = [topic.name for topic in topics]
    if not topics or len(names) != len(set(names)):
        raise ValueError("topic contract must be non-empty and unique")
    if any(not topic.name.startswith("/") or not topic.type or not topic.purpose for topic in topics):
        raise ValueError("every topic requires an absolute name, type and purpose")
    required_names = {topic.name for topic in topics if topic.required}
    invariant = {
        "/registered_scan",
        "/state_estimation_at_scan",
        "/way_point",
        "/runtime",
        "/sensor_coverage_planner/exploration_finish",
        "/free_paths",
        "/path",
        "/cmd_vel",
    }
    if not invariant.issubset(required_names):
        raise ValueError("recording contract omits a closed-loop invariant topic")
    return topics


def rosbag_record_command(
    contract: Sequence[TopicContract],
    *,
    output_bag: str | Path,
) -> tuple[str, ...]:
    destination = Path(output_bag)
    if destination.suffix != ".bag":
        raise ValueError("recording output must end in .bag")
    if destination.exists():
        raise FileExistsError(destination)
    names = tuple(topic.name for topic in contract)
    if len(names) != len(set(names)):
        raise ValueError("duplicate recording topics")
    return ("rosbag", "record", "--lz4", "--buffsize=2048", "-O", str(destination), *names)


def audit_topic_inventory(
    contract: Sequence[TopicContract],
    observed: Mapping[str, TopicObservation],
) -> RecordingAudit:
    missing: list[str] = []
    empty: list[str] = []
    mismatches: list[str] = []
    required_messages = 0
    for topic in contract:
        observation = observed.get(topic.name)
        if observation is None:
            if topic.required:
                missing.append(topic.name)
            continue
        if observation.type != topic.type:
            mismatches.append(f"{topic.name}:{observation.type}!={topic.type}")
        if topic.required:
            required_messages += observation.messages
            if observation.messages == 0:
                empty.append(topic.name)
    return RecordingAudit(
        passed=not missing and not empty and not mismatches,
        missing_topics=tuple(sorted(missing)),
        empty_required_topics=tuple(sorted(empty)),
        type_mismatches=tuple(sorted(mismatches)),
        observed_required_messages=required_messages,
    )


__all__ = [
    "METHODS",
    "RecordingAudit",
    "TopicContract",
    "TopicObservation",
    "audit_topic_inventory",
    "load_topic_contract",
    "rosbag_record_command",
]
