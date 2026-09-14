"""Evidence-bound structural task bookkeeping; no planner, model or map access.

Similarity is a proposal only. Retrieval-set-scoped matches remain reversible
records, never globally unique identities, task/anchor unions or transferable
completion evidence. Evidence
producers must verify the referenced observations: this module validates their
typed contract and causal scope, not the contents of external evidence files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import math


class TaskState(str, Enum):
    PENDING = "pending_exploration"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed_with_evidence"
    BLOCKED_RETRY = "blocked_pending_retry"
    UNKNOWN = "unknown"


class EvidenceKind(str, Enum):
    DISPATCH = "dispatch"
    POSITION_REACHED = "position_reached"
    EXPLORATION_COMPLETE = "exploration_complete"
    BLOCKED = "blocked"
    RETRY = "retry"
    NEW_OBSERVATION = "new_observation"
    UNKNOWN = "unknown"


class DirectionFrameKind(str, Enum):
    COMMON_METRIC = "common_metric"
    SENSOR_LOCAL = "sensor_local"
    UNKNOWN = "unknown"


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _order(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("causal order must be a nonnegative integer")


def _refs(value: tuple[str, ...]) -> None:
    if not isinstance(value, tuple) or not value or len(set(value)) != len(value):
        raise ValueError("immutable, nonempty, unique source references required")
    for ref in value:
        _text(ref, "source reference")


def _xyz(value: tuple[float, float, float], name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{name} must be an immutable XYZ triple")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"{name} must contain finite numbers")


def _score(value: float | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
        raise ValueError("proposal score must be finite or explicit unknown")


@dataclass(frozen=True)
class NativeSnapshotBinding:
    """Exact native observation bound at task ingestion, not a frame-name alias.

    Causal task order and native ROS timestamp use different units. Their
    relationship is supplied explicitly by the trusted ingest adapter, never
    inferred from a timestamp or by renaming registered/raw scan references.
    """

    epoch: str
    snapshot_stamp_ns: int
    source_frame_keys: tuple[str, ...]
    task_order: int

    def __post_init__(self) -> None:
        _text(self.epoch, "native snapshot epoch")
        _order(self.snapshot_stamp_ns)
        _order(self.task_order)
        _refs(self.source_frame_keys)


@dataclass(frozen=True)
class NativeTaskCandidate:
    task_id: str
    native_candidate_id: str
    segment: str
    anchor_id: str
    direction_id: str
    direction_xyz: tuple[float, float, float]
    target_xyz_m: tuple[float, float, float]
    level_id: str | None
    order: int
    source_refs: tuple[str, ...]
    model_score: float | None = None
    direction_frame_id: str | None = None
    direction_frame_kind: DirectionFrameKind = DirectionFrameKind.UNKNOWN
    native_snapshot_binding: NativeSnapshotBinding | None = None

    def __post_init__(self) -> None:
        for name in ("task_id", "native_candidate_id", "segment", "anchor_id", "direction_id"):
            _text(getattr(self, name), name)
        if self.level_id is not None:
            _text(self.level_id, "level_id")
        if not isinstance(self.direction_frame_kind, DirectionFrameKind):
            raise ValueError("explicit direction frame kind required")
        if self.direction_frame_id is not None:
            _text(self.direction_frame_id, "direction_frame_id")
        if self.direction_frame_kind is not DirectionFrameKind.UNKNOWN and self.direction_frame_id is None:
            raise ValueError("a known direction frame needs an explicit frame ID")
        _order(self.order)
        _refs(self.source_refs)
        if self.native_snapshot_binding is not None:
            if not isinstance(self.native_snapshot_binding, NativeSnapshotBinding):
                raise ValueError("typed immutable native snapshot binding required")
            if self.native_snapshot_binding.task_order != self.order:
                raise ValueError("native snapshot binding must match the task's own causal order")
            if self.native_snapshot_binding.source_frame_keys != self.source_refs:
                raise ValueError("native snapshot binding must match the task's own source references")
        _xyz(self.direction_xyz, "direction_xyz")
        _xyz(self.target_xyz_m, "target_xyz_m")
        if not math.isclose(math.hypot(*self.direction_xyz), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("direction_xyz must be an oriented 3D unit vector")
        _score(self.model_score)


@dataclass(frozen=True)
class TaskEvidence:
    evidence_id: str
    task_id: str
    segment: str
    direction_id: str
    level_id: str | None
    order: int
    kind: EvidenceKind
    source_refs: tuple[str, ...]
    reason: str
    trajectory_ref: str | None = None
    exploration_feedback_ref: str | None = None

    def __post_init__(self) -> None:
        for name in ("evidence_id", "task_id", "segment", "direction_id", "reason"):
            _text(getattr(self, name), name)
        if not isinstance(self.kind, EvidenceKind):
            raise ValueError("explicit EvidenceKind required; a score is not evidence")
        if self.level_id is not None:
            _text(self.level_id, "level_id")
        _order(self.order)
        _refs(self.source_refs)
        for name in ("trajectory_ref", "exploration_feedback_ref"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        if self.kind is EvidenceKind.EXPLORATION_COMPLETE:
            if self.trajectory_ref is None or self.exploration_feedback_ref is None:
                raise ValueError("completion requires task-specific exploration feedback and trajectory evidence")


@dataclass(frozen=True)
class AssociationEvidence:
    evidence_id: str
    source_task_id: str
    target_task_id: str
    segment: str
    order: int
    registration_ref: str
    direction_ref: str
    trajectory_ref: str
    level_ref: str
    registration_verified: bool
    direction_verified: bool
    trajectory_verified: bool
    level_verified: bool

    def __post_init__(self) -> None:
        for name in ("evidence_id", "source_task_id", "target_task_id", "segment", "registration_ref", "direction_ref", "trajectory_ref", "level_ref"):
            _text(getattr(self, name), name)
        _order(self.order)
        for name in ("registration_verified", "direction_verified", "trajectory_verified", "level_verified"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an explicit boolean")


@dataclass(frozen=True)
class AnchorVisit:
    visit_id: str
    anchor_id: str
    segment: str
    trajectory_id: str
    order: int
    position_xyz_m: tuple[float, float, float]
    source_refs: tuple[str, ...]
    actual_passage_verified: bool
    previous_visit_id: str | None = None
    continuous_trajectory_ref: str | None = None

    def __post_init__(self) -> None:
        for name in ("visit_id", "anchor_id", "segment", "trajectory_id"):
            _text(getattr(self, name), name)
        _order(self.order)
        _xyz(self.position_xyz_m, "position_xyz_m")
        _refs(self.source_refs)
        if type(self.actual_passage_verified) is not bool:
            raise ValueError("actual passage verdict must be an explicit boolean")
        if (self.previous_visit_id is None) != (self.continuous_trajectory_ref is None):
            raise ValueError("a previous visit and its continuous trajectory evidence are required together")
        if self.previous_visit_id is not None:
            _text(self.previous_visit_id, "previous_visit_id")
            _text(self.continuous_trajectory_ref, "continuous_trajectory_ref")


@dataclass(frozen=True)
class _Task:
    candidate: NativeTaskCandidate
    state: TaskState
    completion_evidence_id: str | None = None


class StructuralTaskLifecycle:
    """Append-only evidence history with conservative, per-direction task states.

    ``add_native_candidate`` never filters scores. ``apply_task_evidence`` is the
    only state-transition API. All evidence and visits are supplied by callers;
    registering a task/association cannot manufacture physical edges.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, _Task] = {}
        self._associations: dict[str, dict] = {}
        self._association_resolutions: dict[str, dict] = {}
        self._visits: list[AnchorVisit] = []
        self._edges: list[dict] = []
        self._history: list[dict] = []
        self._evidence_ids: set[str] = set()
        self._last_order = -1

    def _check_order(self, order: int) -> None:
        _order(order)
        if order < self._last_order:
            raise ValueError("evidence must not revise an earlier causal prefix")

    def _check_evidence(self, evidence_id: str, order: int) -> None:
        _text(evidence_id, "evidence_id")
        self._check_order(order)
        if evidence_id in self._evidence_ids:
            raise ValueError("evidence ID already consumed")

    def _record(self, order: int, event: dict, evidence_id: str | None = None) -> None:
        self._history.append(dict(order=order, **event))
        self._last_order = order
        if evidence_id is not None:
            self._evidence_ids.add(evidence_id)

    def add_native_candidate(self, candidate: NativeTaskCandidate, *, initial_state: TaskState = TaskState.PENDING) -> None:
        if not isinstance(candidate, NativeTaskCandidate):
            raise ValueError("typed native candidate required")
        self._check_order(candidate.order)
        if candidate.task_id in self._tasks:
            raise ValueError("task ID already exists; observations must not overwrite tasks")
        if not isinstance(initial_state, TaskState) or initial_state not in (TaskState.PENDING, TaskState.UNKNOWN):
            raise ValueError("new tasks must be pending or explicit unknown")
        self._tasks[candidate.task_id] = _Task(candidate, initial_state)
        self._record(candidate.order, dict(event="candidate_preserved", candidate=asdict(candidate), state=initial_state.value))

    def apply_task_evidence(self, evidence: TaskEvidence) -> TaskState:
        if not isinstance(evidence, TaskEvidence):
            raise ValueError("typed task evidence required")
        self._check_evidence(evidence.evidence_id, evidence.order)
        task = self._tasks[evidence.task_id]
        candidate = task.candidate
        if (evidence.segment, evidence.direction_id, evidence.level_id) != (candidate.segment, candidate.direction_id, candidate.level_id):
            raise ValueError("evidence must bind this exact segment, oriented task and level")
        kind, old = evidence.kind, task.state
        if kind is EvidenceKind.POSITION_REACHED:
            new = old  # An arrival is deliberately not exploration completion.
        elif kind is EvidenceKind.NEW_OBSERVATION:
            new = TaskState.PENDING  # Explicit reopening, with old evidence retained.
        elif kind is EvidenceKind.RETRY and old is TaskState.BLOCKED_RETRY:
            new = TaskState.PENDING
        elif kind is EvidenceKind.DISPATCH and old in (TaskState.PENDING, TaskState.UNKNOWN):
            new = TaskState.IN_PROGRESS
        elif kind is EvidenceKind.EXPLORATION_COMPLETE and old is TaskState.IN_PROGRESS:
            new = TaskState.COMPLETED
        elif kind is EvidenceKind.BLOCKED and old is TaskState.IN_PROGRESS:
            new = TaskState.BLOCKED_RETRY
        elif kind is EvidenceKind.UNKNOWN and old is not TaskState.COMPLETED:
            new = TaskState.UNKNOWN
        else:
            raise ValueError(f"invalid task transition: {old.value} via {kind.value}")
        completion_id = task.completion_evidence_id if new is TaskState.COMPLETED else None
        if kind is EvidenceKind.EXPLORATION_COMPLETE:
            completion_id = evidence.evidence_id
        self._tasks[evidence.task_id] = replace(task, state=new, completion_evidence_id=completion_id)
        self._record(evidence.order, dict(event="task_evidence", previous_state=old.value, state=new.value, evidence=asdict(evidence)), evidence.evidence_id)
        return new

    def propose_association(self, *, association_id: str, source_task_id: str, target_task_id: str, order: int, descriptor_ref: str, model_score: float | None) -> None:
        for name, value in (("association_id", association_id), ("descriptor_ref", descriptor_ref)):
            _text(value, name)
        self._check_order(order)
        _score(model_score)
        source, target = self._tasks[source_task_id], self._tasks[target_task_id]
        if source_task_id == target_task_id or source.candidate.segment != target.candidate.segment:
            raise ValueError("association requires distinct tasks in one segment")
        if association_id in self._associations:
            raise ValueError("association ID already exists")
        proposal = dict(association_id=association_id, source_task_id=source_task_id, target_task_id=target_task_id, descriptor_ref=descriptor_ref, model_score=model_score, state="proposed", verification_evidence_id=None)
        self._associations[association_id] = proposal
        self._record(order, dict(event="association_proposed", association=dict(proposal)))

    def verify_association(self, association_id: str, evidence: AssociationEvidence) -> None:
        """Pairwise checking cannot establish identity before competitors are seen."""
        raise ValueError("pairwise verification is forbidden; use resolve_associations on the complete declared retrieval set")

    def resolve_associations(self, *, resolution_id: str, source_task_id: str,
                             association_ids: tuple[str, ...],
                             evidence: tuple[AssociationEvidence, ...],
                             order: int, retrieval_set_ref: str) -> dict:
        """Evaluate a complete declared retrieval set atomically, without ranking.

        The set must contain every currently proposed association for this
        source and exactly one check result per target. A sole supported match
        is unique *only within this declared retrieval set*, never certified
        globally unique task identity. More than one supported match leaves
        every association unknown. Neither outcome merges or completes tasks.
        """
        self._check_evidence(resolution_id, order)
        _text(retrieval_set_ref, "retrieval_set_ref")
        _refs(association_ids)
        source = self._tasks[source_task_id].candidate
        pending = {key for key, proposal in self._associations.items()
                   if proposal["source_task_id"] == source_task_id and proposal["state"] == "proposed"}
        if set(association_ids) != pending:
            raise ValueError("the complete currently proposed retrieval set must be resolved together")
        if not isinstance(evidence, tuple) or any(not isinstance(item, AssociationEvidence) for item in evidence):
            raise ValueError("immutable typed evidence for every candidate required")
        targets = [self._associations[key]["target_task_id"] for key in association_ids]
        if len(set(targets)) != len(targets):
            raise ValueError("one proposal per historical target is required")
        if len(evidence) != len(targets) or {item.target_task_id for item in evidence} != set(targets):
            raise ValueError("exactly one result for every candidate is required; missing checks remain unresolved")
        evidence_ids = [item.evidence_id for item in evidence]
        if len(set(evidence_ids)) != len(evidence_ids) or resolution_id in evidence_ids:
            raise ValueError("batch and candidate evidence IDs must be unique")
        checks = {item.target_task_id: item for item in evidence}
        rows = []
        for association_id in sorted(association_ids):
            proposal = self._associations[association_id]
            item = checks[proposal["target_task_id"]]
            self._check_evidence(item.evidence_id, item.order)
            if item.order > order:
                raise ValueError("a resolution cannot consume future candidate evidence")
            if item.source_task_id != source_task_id:
                raise ValueError("verification must bind the exact proposed source task")
            target = self._tasks[item.target_task_id].candidate
            if item.segment != source.segment or item.segment != target.segment:
                raise ValueError("cross-segment verification is forbidden")
            reasons = []
            if not all((item.registration_verified, item.direction_verified, item.trajectory_verified, item.level_verified)):
                reasons.append("registration_direction_trajectory_or_level_not_verified")
            # Compare directions only in the same declared fixed metric frame.
            # Equal sensor-local headings cannot establish a common direction.
            if (source.direction_frame_kind is not DirectionFrameKind.COMMON_METRIC
                    or target.direction_frame_kind is not DirectionFrameKind.COMMON_METRIC):
                reasons.append("direction_not_in_common_metric_frame")
            elif source.direction_frame_id != target.direction_frame_id:
                reasons.append("different_direction_frames")
            elif sum(a * b for a, b in zip(source.direction_xyz, target.direction_xyz)) <= 0:
                reasons.append("opposite_or_orthogonal_direction")
            if source.level_id is None or source.level_id != target.level_id:
                reasons.append("unknown_or_different_levels")
            rows.append(dict(association_id=association_id, checks_supported=not reasons,
                             unknown_reasons=reasons, evidence=asdict(item)))
        supported = [row["association_id"] for row in rows if row["checks_supported"]]
        selected = supported[0] if len(supported) == 1 else None
        result = dict(resolution_id=resolution_id, source_task_id=source_task_id,
                      retrieval_set_ref=retrieval_set_ref, association_ids=sorted(association_ids),
                      candidate_checks=rows, supported_association_ids=supported,
                      selected_association_id=selected,
                      state="unique_in_declared_retrieval_set" if selected is not None else "unknown",
                      unknown_reason=None if selected is not None else ("multiple_supported_candidates" if supported else "no_supported_candidate"),
                      complete_declared_set_evaluated=True, search_scope="declared_retrieval_set_only",
                      exhaustive_history_search=False, global_identity_verified=False)
        # All checks above are read-only; commit the whole batch at once.
        for row in rows:
            association_id = row["association_id"]
            self._associations[association_id] = dict(
                self._associations[association_id],
                state="unique_in_declared_retrieval_set" if association_id == selected else "unknown",
                resolution_id=resolution_id, checks_supported=row["checks_supported"],
                verification_evidence_id=row["evidence"]["evidence_id"] if association_id == selected else None,
                global_identity_verified=False)
        self._association_resolutions[resolution_id] = result
        self._record(order, dict(event="association_set_resolved", resolution=result), resolution_id)
        self._evidence_ids.update(evidence_ids)
        from copy import deepcopy
        return deepcopy(result)

    def revoke_association(self, association_id: str, *, evidence_id: str, order: int, source_refs: tuple[str, ...], reason: str) -> None:
        self._check_evidence(evidence_id, order)
        _refs(source_refs)
        _text(reason, "reason")
        proposal = self._associations[association_id]
        if proposal["state"] == "revoked":
            raise ValueError("association is already revoked")
        self._associations[association_id] = dict(proposal, state="revoked")
        self._record(order, dict(event="association_revoked", association_id=association_id, evidence_id=evidence_id, source_refs=source_refs, reason=reason), evidence_id)

    def record_anchor_visit(self, visit: AnchorVisit) -> dict | None:
        if not isinstance(visit, AnchorVisit):
            raise ValueError("typed actual anchor visit required")
        self._check_evidence(visit.visit_id, visit.order)
        if not visit.actual_passage_verified:
            raise ValueError("predicted reachability or a target is not an actual anchor passage")
        previous = self._visits[-1] if self._visits else None
        edge = None
        if visit.previous_visit_id is not None:
            if previous is None or previous.visit_id != visit.previous_visit_id:
                raise ValueError("only consecutive actual anchor visits may create edges")
            if previous.segment != visit.segment or previous.trajectory_id != visit.trajectory_id:
                raise ValueError("continuous edges cannot cross a segment or trajectory boundary")
            if visit.order <= previous.order:
                raise ValueError("traversal endpoints need strictly increasing causal order")
            if previous.anchor_id != visit.anchor_id:
                edge = dict(segment=visit.segment, trajectory_id=visit.trajectory_id, from_anchor=previous.anchor_id, to_anchor=visit.anchor_id, from_visit=previous.visit_id, to_visit=visit.visit_id, trajectory_ref=visit.continuous_trajectory_ref, physical_passage_verified=True)
        self._visits.append(visit)
        if edge is not None:
            self._edges.append(edge)
        self._record(visit.order, dict(event="anchor_visited", visit=asdict(visit), edge=None if edge is None else dict(edge)), visit.visit_id)
        return None if edge is None else dict(edge)

    def completion_report(self, *, native_finish: bool | None, at_home: bool | None, budget_exhausted: bool = False, timed_out: bool = False) -> dict:
        for name, value in (("native_finish", native_finish), ("at_home", at_home)):
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be a boolean or explicit unknown")
        if type(budget_exhausted) is not bool or type(timed_out) is not bool:
            raise ValueError("budget and timeout flags must be explicit booleans")
        counts = {state.value: sum(task.state is state for task in self._tasks.values()) for state in TaskState}
        outstanding = len(self._tasks) - counts[TaskState.COMPLETED.value]
        all_recorded = bool(self._tasks) and outstanding == 0
        conditions_met = all_recorded and native_finish is True and at_home is True and not (budget_exhausted or timed_out)
        if timed_out:
            status = "timeout_incomplete"
        elif budget_exhausted:
            status = "budget_exhausted_incomplete"
        elif conditions_met:
            status = "recorded_tasks_completed_native_finished_home"
        elif native_finish is True and outstanding:
            status = "native_finished_with_outstanding_tasks"
        elif native_finish is True and at_home is not True:
            status = "native_finished_home_unconfirmed"
        elif not self._tasks:
            status = "no_task_evidence"
        else:
            status = "running_or_completion_unconfirmed"
        return dict(status=status, native_finish=native_finish, at_home=at_home, budget_exhausted=budget_exhausted, timed_out=timed_out, task_counts=counts, total_tasks=len(self._tasks), outstanding_tasks=outstanding, all_recorded_tasks_completed=all_recorded, recorded_completion_conditions_met=conditions_met, full_exploration_proven=False)

    def snapshot(self) -> dict:
        from copy import deepcopy
        return deepcopy(dict(tasks={key: asdict(task) for key, task in self._tasks.items()}, associations=self._associations, association_resolutions=self._association_resolutions, visits=[asdict(visit) for visit in self._visits], confirmed_edges=self._edges, history=self._history))
