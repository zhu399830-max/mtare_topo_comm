"""Read-only contracts for auditing causal structural-event supervision units."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class StructuralEpisode:
    traversal_id: str
    identity: str
    event: str
    start_sequence_index: int
    end_sequence_index: int
    frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def contiguous_structural_episodes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[StructuralEpisode, ...]:
    """Partition structural labels into same-traversal, same-identity runs."""

    ordered = sorted(
        rows,
        key=lambda row: (str(row["traversal_id"]), int(row["sequence_index"])),
    )
    seen: set[tuple[str, int]] = set()
    episodes: list[StructuralEpisode] = []
    active: list[Mapping[str, Any]] = []

    def flush() -> None:
        if not active:
            return
        first = active[0]
        episodes.append(
            StructuralEpisode(
                traversal_id=str(first["traversal_id"]),
                identity=str(first["identity"]),
                event=str(first["event"]),
                start_sequence_index=int(first["sequence_index"]),
                end_sequence_index=int(active[-1]["sequence_index"]),
                frame_count=len(active),
            )
        )
        active.clear()

    for row in ordered:
        key = (str(row["traversal_id"]), int(row["sequence_index"]))
        if key in seen:
            raise ValueError("duplicate traversal/sequence Teacher row")
        seen.add(key)
        event = str(row["event"])
        identity = row.get("identity")
        if (event == "corridor") != (identity is None):
            raise ValueError("corridor/structural identity contract drift")
        if event == "corridor":
            flush()
            continue
        continues = bool(
            active
            and str(active[-1]["traversal_id"]) == key[0]
            and int(active[-1]["sequence_index"]) + 1 == key[1]
            and str(active[-1]["identity"]) == str(identity)
            and str(active[-1]["event"]) == event
        )
        if not continues:
            flush()
        active.append(row)
    flush()
    return tuple(episodes)


def transition_supervision_alignment(
    *,
    teacher_rows: Sequence[Mapping[str, Any]],
    proof_points: Sequence[Mapping[str, Any]],
    proof_labels: Sequence[Mapping[str, Any]],
    history_lengths: Sequence[int] = (5, 8, 10, 12),
) -> dict[str, Any]:
    """Measure frame labels against the causal confirmation interval.

    Identity and bidirectional evidence are used only for this Teacher audit,
    never as student inputs.
    """

    histories = tuple(int(value) for value in history_lengths)
    if not histories or any(value < 2 for value in histories) or len(set(histories)) != len(histories):
        raise ValueError("history lengths must be unique integers >=2")

    teacher_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    traversal_rows: dict[str, list[Mapping[str, Any]]] = {}
    final_transition_keys: set[tuple[str, int]] = set()
    final_transition_identities: set[str] = set()
    for row in teacher_rows:
        key = (str(row["traversal_id"]), int(row["sequence_index"]))
        if key in teacher_by_key:
            raise ValueError("duplicate Teacher traversal/sequence key")
        teacher_by_key[key] = row
        traversal_rows.setdefault(key[0], []).append(row)
        if str(row["event"]) == "geometry_transition":
            if row.get("identity") is None:
                raise ValueError("transition row lacks identity")
            final_transition_keys.add(key)
            final_transition_identities.add(str(row["identity"]))

    episode_by_identity_traversal: dict[tuple[str, str], Mapping[str, Any]] = {}
    emitted_points = 0
    for point in proof_points:
        if not bool(point.get("emitted")):
            continue
        emitted_points += 1
        identity = str(point["identity_assignment"]["identity"])
        matches = point.get("causal_matches", ())
        if len(matches) != 2 or any(len(group) != 1 for group in matches):
            raise ValueError("emitted point lacks one causal episode per direction")
        for group in matches:
            episode = group[0]
            key = (identity, str(episode["traversal_id"]))
            if key in episode_by_identity_traversal:
                raise ValueError("identity/traversal has multiple causal episodes")
            episode_by_identity_traversal[key] = episode

    timing_rows: list[dict[str, Any]] = []
    proof_key_count: dict[tuple[str, int], int] = {}
    for label in proof_labels:
        key = (str(label["traversal_id"]), int(label["sequence_index"]))
        if key not in final_transition_keys:
            continue
        proof_key_count[key] = proof_key_count.get(key, 0) + 1
        teacher = teacher_by_key[key]
        if str(teacher["identity"]) != str(label["identity"]):
            raise ValueError("proof/final Teacher transition identity mismatch")
        episode = episode_by_identity_traversal[(str(label["identity"]), key[0])]
        arc = float(label["traversal_arc_m"])
        emission = float(episode["emission_arc_m"])
        boundary = emission - float(episode["causal_delay_m"])
        timing_rows.append(
            {
                "parent_id": str(label["parent_id"]),
                "traversal_id": key[0],
                "sequence_index": key[1],
                "identity": str(label["identity"]),
                "traversal_arc_m": arc,
                "emission_arc_m": emission,
                "boundary_traversal_arc_m": boundary,
                "metres_before_confirmation": emission - arc,
                "metres_after_boundary": arc - boundary,
                "before_closed_confirmation": arc < emission - 1e-9,
                "at_closed_confirmation": abs(arc - emission) <= 1e-9,
                "boundary_inside_five_frame_history": arc - 4.0 - 1e-9 <= boundary <= arc + 1e-9,
            }
        )
    if set(proof_key_count) != final_transition_keys or any(value != 1 for value in proof_key_count.values()):
        raise ValueError("proof labels do not cover the final transition Teacher exactly")

    availability = {str(value): 0 for value in histories}
    directional_episodes = 0
    for (identity, traversal_id), episode in episode_by_identity_traversal.items():
        if identity not in final_transition_identities:
            continue
        directional_episodes += 1
        rows = traversal_rows[traversal_id]
        emission = float(episode["emission_arc_m"])
        candidates = [row for row in rows if abs(float(row["traversal_arc_m"]) - emission) <= 1e-9]
        if len(candidates) != 1:
            raise ValueError("causal emission does not map to exactly one Teacher frame")
        available_frames = int(candidates[0]["frame_index"]) + 1
        for history in histories:
            availability[str(history)] += int(available_frames >= history)

    label_count = len(timing_rows)
    before = sum(bool(row["before_closed_confirmation"]) for row in timing_rows)
    at_confirmation = sum(bool(row["at_closed_confirmation"]) for row in timing_rows)
    inside_five = sum(bool(row["boundary_inside_five_frame_history"]) for row in timing_rows)
    return {
        "emitted_change_points": emitted_points,
        "directional_change_episodes": directional_episodes,
        "final_transition_labels": label_count,
        "final_transition_identities": len(final_transition_identities),
        "labels_before_closed_confirmation": before,
        "labels_at_closed_confirmation": at_confirmation,
        "labels_after_closed_confirmation": label_count - before - at_confirmation,
        "labels_with_boundary_inside_five_frame_history": inside_five,
        "labels_with_boundary_outside_five_frame_history": label_count - inside_five,
        "history_available_at_confirmation": availability,
        "timing_rows": timing_rows,
    }


__all__ = [
    "StructuralEpisode",
    "contiguous_structural_episodes",
    "transition_supervision_alignment",
]
