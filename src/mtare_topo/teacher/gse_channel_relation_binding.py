"""Loss-side indexing of independently supported channel relations.

This is NOT a source-to-channel classifier or evidence qualifier. The caller
must supply independently justified memberships and per-attribute references.
No identity in this module is passed to GeometryStructureEncoder.forward.
Missing, multiple or unqualified memberships remain unknown. In particular,
same-region membership does not manufacture a positive correspondence label.
"""
from dataclasses import dataclass
import math
import torch
from mtare_topo.representation.gse_observable_relation_losses import ObservableRelationTargets


@dataclass(frozen=True)
class PatchChannelMembership:
    region_ids: tuple[str, ...]
    evidence_ref: str = ''
    evidence_kind: str = 'unqualified'


@dataclass(frozen=True)
class SupportedRelationValue:
    value: float
    evidence_ref: str


@dataclass(frozen=True)
class SupportedChannelPair:
    observation_index: int
    receiver_region: str
    neighbor_region: str
    axis: SupportedRelationValue | None = None
    height: SupportedRelationValue | None = None
    width_ratio: SupportedRelationValue | None = None
    height_ratio: SupportedRelationValue | None = None
    correspondence: SupportedRelationValue | None = None


def bind_channel_relation_targets(prediction, memberships, pairs):
    """Return patch-indexed targets and a per-pair audit; never qualify evidence.

    Ratios are logarithms, height is neighbor minus receiver in meters and
    axis is unsigned dot product. Ordered relation records must be explicit:
    no reverse relation, self relation, negative or interpolation is invented.
    """
    neighbors = prediction.neighbor_index
    valid = prediction.computation_valid
    if neighbors.ndim != 3 or neighbors.dtype != torch.long or valid.shape != neighbors.shape or valid.dtype != torch.bool:
        raise ValueError('invalid prediction index contract')
    b, m, k = neighbors.shape
    if bool((valid & ((neighbors < 0) | (neighbors >= m))).any()):
        raise ValueError('valid pair has out-of-range neighbor')
    if not isinstance(memberships, tuple) or len(memberships) != b:
        raise ValueError('one immutable membership row per observation required')
    for row in memberships:
        if not isinstance(row, tuple) or len(row) != m:
            raise ValueError('membership order must match all model patches')
        for item in row:
            if type(item) is not PatchChannelMembership or not isinstance(item.region_ids, tuple):
                raise ValueError('typed immutable membership required')
            if any(type(r) is not str or not r.strip() for r in item.region_ids) or len(set(item.region_ids)) != len(item.region_ids):
                raise ValueError('invalid or duplicated region identity')
            if item.evidence_kind not in ('unqualified', 'observed_channel_support'):
                raise ValueError('source identity or input geometry is not channel support')
            if item.region_ids and item.evidence_kind == 'observed_channel_support' and not item.evidence_ref.strip():
                raise ValueError('qualified membership lacks provenance')
    names = ('axis', 'height', 'width_ratio', 'height_ratio', 'correspondence')
    lookup = {}
    for pair in pairs:
        if type(pair) is not SupportedChannelPair or type(pair.observation_index) is not int or not 0 <= pair.observation_index < b:
            raise ValueError('invalid channel pair record')
        if any(type(r) is not str or not r.strip() for r in (pair.receiver_region, pair.neighbor_region)):
            raise ValueError('invalid channel pair identity')
        key = (pair.observation_index, pair.receiver_region, pair.neighbor_region)
        if key in lookup:
            raise ValueError('duplicate ordered channel relation')
        lookup[key] = pair
        for name in names:
            value = getattr(pair, name)
            if value is None:
                continue
            if (type(value) is not SupportedRelationValue or type(value.value) not in (float, int)
                    or not math.isfinite(value.value) or not isinstance(value.evidence_ref, str) or not value.evidence_ref.strip()):
                raise ValueError('known attribute requires finite value and independent evidence')
            if name == 'axis' and not 0 <= value.value <= 1:
                raise ValueError('axis dot outside [0,1]')
            if name == 'correspondence' and value.value not in (0, 1):
                raise ValueError('correspondence must be binary')
    values = {n: torch.zeros_like(prediction.axis_abs_dot) for n in names}
    masks = {n: torch.zeros_like(valid) for n in names}
    refs, audit = set(), []
    # Static label compilation only, outside forward/optimizer and detached
    # from predictions. CPU copies prevent per-item CUDA synchronization.
    neighbor_rows = neighbors.detach().cpu().tolist()
    valid_rows = valid.detach().cpu().tolist()
    for obs in range(b):
        for receiver in range(m):
            for slot in range(k):
                if not valid_rows[obs][receiver][slot]:
                    continue
                neighbor = neighbor_rows[obs][receiver][slot]
                a, c = memberships[obs][receiver], memberships[obs][neighbor]
                entry = {'index': (obs, receiver, slot), 'neighbor_patch': neighbor,
                         'receiver_regions': a.region_ids, 'neighbor_regions': c.region_ids,
                         'known_attributes': (), 'evidence_refs': ()}
                if len(a.region_ids) != 1 or len(c.region_ids) != 1:
                    entry['reason'] = 'MISSING_OR_AMBIGUOUS_MEMBERSHIP'
                elif a.evidence_kind != 'observed_channel_support' or c.evidence_kind != 'observed_channel_support':
                    entry['reason'] = 'UNQUALIFIED_MEMBERSHIP'
                else:
                    pair = lookup.get((obs, a.region_ids[0], c.region_ids[0]))
                    if pair is None:
                        entry['reason'] = 'NO_ORDERED_RELATION_EVIDENCE'
                    else:
                        known, local_refs = [], set()
                        for name in names:
                            value = getattr(pair, name)
                            if value is not None:
                                values[name][obs, receiver, slot] = value.value
                                masks[name][obs, receiver, slot] = True
                                known.append(name)
                                local_refs.update((a.evidence_ref, c.evidence_ref, value.evidence_ref))
                        refs.update(local_refs)
                        entry.update(reason='ATTRIBUTE_MASKS_APPLIED', known_attributes=tuple(known), evidence_refs=tuple(sorted(local_refs)))
                audit.append(entry)
    targets = ObservableRelationTargets(
        values['axis'], masks['axis'], values['height'], masks['height'],
        torch.stack((values['width_ratio'], values['height_ratio']), -1),
        torch.stack((masks['width_ratio'], masks['height_ratio']), -1),
        values['correspondence'], masks['correspondence'], tuple(sorted(refs)))
    return targets, tuple(audit)
