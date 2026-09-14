"""Pure, fail-closed fusion of ground-path reference and causal observation evidence.

This module does NOT produce evidence from rays/meshes, certify continuous safety,
search 3-D air, or label a whole portal unreachable from one blocked candidate.
Its label applies to exactly the supplied discrete ground path. Explicit source
bindings are producer attestations, not cryptographic or physical validation.
"""
from dataclasses import dataclass
from enum import Enum
import math


class Truth(Enum):
    PASS = "PASS"
    BLOCKED = "BLOCKED"
    UNKNOWN = "UNKNOWN"


class Observation(Enum):
    SUPPORTED_PATH = "SUPPORTED_PATH"
    OBSERVED_OBSTRUCTION = "OBSERVED_OBSTRUCTION"
    UNKNOWN = "UNKNOWN"


class Label(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN"


class ContractScope(Enum):
    SYNTHETIC_ONLY = "SYNTHETIC_ONLY"
    DEPLOYMENT_BOUND = "DEPLOYMENT_BOUND"


class Obstruction(Enum):
    COLLISION = "COLLISION"
    OBSERVED_DROP_OFF = "OBSERVED_DROP_OFF"
    EXCESS_SLOPE = "EXCESS_SLOPE"
    EXCESS_STEP = "EXCESS_STEP"


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _bools(value, length, name):
    if not isinstance(value, tuple) or len(value) != length or any(type(x) is not bool for x in value):
        raise ValueError(f"{name} requires {length} explicit bool entries")


@dataclass(frozen=True)
class GroundRobotContract:
    """No default scientific values; rectangular envelope includes safety margin.

    Dimensions describe the whole collision envelope, NOT the visual body or
    sensor height. Limits describe supported capability, NOT a terrain fitter.
    Five bindings identify body, support, slope, step and sensor-ground contract.
    A caller cannot promote synthetic values merely by passing allow_synthetic.
    """
    scope: ContractScope
    envelope_length_m: float
    envelope_width_m: float
    envelope_height_m: float
    max_slope_deg: float
    max_step_m: float
    source_bindings: tuple[str, str, str, str, str]

    def __post_init__(self):
        if not isinstance(self.scope, ContractScope):
            raise ValueError("scope must be ContractScope")
        for name in ("envelope_length_m", "envelope_width_m", "envelope_height_m", "max_slope_deg", "max_step_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
            if value < 0 or (name.startswith("envelope") and value == 0):
                raise ValueError(f"invalid {name}")
        if self.max_slope_deg >= 90:
            raise ValueError("ground slope must be below 90 degrees")
        if not isinstance(self.source_bindings, tuple) or len(self.source_bindings) != 5:
            raise ValueError("all five physical source bindings are required")
        for source in self.source_bindings:
            _text(source, "source binding")


@dataclass(frozen=True)
class GroundPath:
    """Local geometry state IDs, never construction/TNG identities.

    Every transition is explicit. layer_transition_supported certifies an
    actual supported ramp/step transition when the adjacent layer IDs differ;
    equal XY coordinates never create adjacency. Positions are ground-contact
    states (not sensor positions). Producer must bind reference to this object.
    """
    state_ids: tuple[str, ...]
    ground_xyz_m: tuple[tuple[float, float, float], ...]
    layer_ids: tuple[str, ...]
    layer_transition_supported: tuple[bool, ...]

    def __post_init__(self):
        n = len(self.state_ids)
        if not isinstance(self.state_ids, tuple) or n < 2 or len(set(self.state_ids)) != n:
            raise ValueError("path needs at least two distinct ordered states")
        if not isinstance(self.ground_xyz_m, tuple) or not isinstance(self.layer_ids, tuple):
            raise ValueError("path fields must be immutable tuples")
        if len(self.ground_xyz_m) != n or len(self.layer_ids) != n:
            raise ValueError("ground state fields must align")
        for state, layer, xyz in zip(self.state_ids, self.layer_ids, self.ground_xyz_m):
            _text(state, "state")
            _text(layer, "ground layer")
            if not isinstance(xyz, tuple) or len(xyz) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in xyz
            ):
                raise ValueError("ground contact XYZ must be finite triples")
        _bools(self.layer_transition_supported, n - 1, "layer_transition_supported")


@dataclass(frozen=True)
class PhysicalReference:
    path: GroundPath
    status: Truth
    contract: GroundRobotContract | None
    evidence_id: str

    def __post_init__(self):
        if not isinstance(self.path, GroundPath) or not isinstance(self.status, Truth):
            raise ValueError("typed path and physical status required")
        if self.contract is not None and not isinstance(self.contract, GroundRobotContract):
            raise ValueError("typed physical contract required")
        _text(self.evidence_id, "physical evidence id")


@dataclass(frozen=True)
class FiveFrameSupport:
    path: GroundPath
    frame_indices: tuple[int, int, int, int, int]
    current_frame_index: int
    status: Observation
    ground_supported: tuple[bool, ...]
    transition_supported: tuple[bool, ...]
    witness_frame_indices: tuple[int, ...]
    witness_ids: tuple[str, ...]
    obstruction: Obstruction | None

    def __post_init__(self):
        if not isinstance(self.path, GroundPath) or not isinstance(self.status, Observation):
            raise ValueError("typed observed path and status required")
        f = self.frame_indices
        if not isinstance(f, tuple) or len(f) != 5 or any(type(x) is not int or x < 0 for x in f):
            raise ValueError("exactly five source frame indices required")
        if type(self.current_frame_index) is not int or tuple(sorted(set(f))) != f or f[-1] != self.current_frame_index:
            raise ValueError("strictly causal ordered five-frame window ending at current frame required")
        _bools(self.ground_supported, len(self.path.state_ids), "ground_supported")
        _bools(self.transition_supported, len(self.path.state_ids) - 1, "transition_supported")
        if not isinstance(self.witness_frame_indices, tuple) or not isinstance(self.witness_ids, tuple):
            raise ValueError("witness fields must be tuples")
        if len(self.witness_frame_indices) != len(self.witness_ids):
            raise ValueError("witness IDs and source frames must align")
        for frame, witness in zip(self.witness_frame_indices, self.witness_ids):
            if type(frame) is not int or frame not in f:
                raise ValueError("witness outside the five causal frames")
            _text(witness, "observation witness")
        if self.status is not Observation.UNKNOWN and not self.witness_ids:
            raise ValueError("determinate observation requires actual witness provenance")
        if self.status is Observation.OBSERVED_OBSTRUCTION:
            if not isinstance(self.obstruction, Obstruction):
                raise ValueError("negative observation requires an observed physical obstruction, not search failure")
        elif self.obstruction is not None:
            raise ValueError("obstruction reason only allowed with observed obstruction")


@dataclass(frozen=True)
class FusedPathTarget:
    label: Label
    reason: str
    path: GroundPath
    synthetic_only: bool
    label_scope: str = "THIS_DISCRETE_GROUND_PATH_ONLY"


def fuse_ground_path(reference: PhysicalReference, observed: FiveFrameSupport, *, allow_synthetic: bool = False) -> FusedPathTarget:
    """No physical values or labels are inferred from missing information.

    BLOCKED here means this candidate path, not root unreachability, a closed
    portal, terminal event, or permission to delete an exploration frontier.
    Upstream evidence extraction/collision checking remains a separate task.
    """
    if not isinstance(reference, PhysicalReference) or not isinstance(observed, FiveFrameSupport):
        raise ValueError("typed reference and observation required")
    if type(allow_synthetic) is not bool:
        raise ValueError("allow_synthetic must be bool")
    if reference.path != observed.path:
        raise ValueError("physical confirmation and observed support must refer to the exact same ground path")
    contract = reference.contract
    synthetic = contract is not None and contract.scope is ContractScope.SYNTHETIC_ONLY

    def result(label, reason):
        return FusedPathTarget(label, reason, reference.path, synthetic)

    if contract is None:
        return result(Label.UNKNOWN, "ROBOT_PHYSICAL_CONTRACT_UNBOUND")
    if synthetic and not allow_synthetic:
        return result(Label.UNKNOWN, "SYNTHETIC_CONTRACT_NOT_DEPLOYABLE")
    if observed.status is Observation.UNKNOWN:
        return result(Label.UNKNOWN, "OBSERVATION_SUPPORT_INSUFFICIENT")
    if reference.status is Truth.UNKNOWN:
        return result(Label.UNKNOWN, "PHYSICAL_REFERENCE_UNKNOWN")
    if observed.status is Observation.OBSERVED_OBSTRUCTION:
        if reference.status is Truth.PASS:
            return result(Label.UNKNOWN, "OBSERVATION_REFERENCE_CONFLICT")
        return result(Label.NEGATIVE, "OBSERVED_OBSTRUCTION_CONFIRMED_ON_THIS_PATH")
    if reference.status is Truth.BLOCKED:
        return result(Label.UNKNOWN, "HIDDEN_OR_CONFLICTING_REFERENCE_OBSTRUCTION")
    path = reference.path
    if not all(observed.ground_supported) or not all(observed.transition_supported):
        return result(Label.UNKNOWN, "GROUND_OR_BODY_TRANSITION_SUPPORT_INCOMPLETE")
    for left, right, supported in zip(path.layer_ids, path.layer_ids[1:], path.layer_transition_supported):
        if left != right and not supported:
            return result(Label.UNKNOWN, "UNSUPPORTED_CROSS_LAYER_TRANSITION")
    return result(Label.POSITIVE, "OBSERVED_GROUND_PATH_PHYSICALLY_CONFIRMED")
