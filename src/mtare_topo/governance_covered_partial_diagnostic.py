"""Metadata-only input inventory for the fixed partial-teacher diagnostic.

This is a scope builder, not an executor or training authorization.
"""
import hashlib
import json
from .governance_source_compatibility import scope as source_scope
from .governance_covered_sensor_export import ROOT


AUDIT = 'results/gate3_semantics/gate3_20260908_gse_source_compatibility_v1_seed20260906'
AUDIT_SEAL = 'aca5d582085a41a4b2912fc38d2b5efaf4dc71fd2a13ae2b3bb5010ee59d969f'
SCHEMA = 'v3_covered_partial_diagnostic_card_v1'
SLUG = 'gse_covered_partial_diagnostic_v1'


def scope():
    original = source_scope()
    path = AUDIT + '/SHA256_SEAL.json'
    raw = (ROOT/path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != AUDIT_SEAL:
        raise ValueError('source compatibility seal drift')
    seal = json.loads(raw)
    inputs = dict(original['input_sha256'])
    inputs[path] = AUDIT_SEAL
    for name in original['cases']:
        for frame in range(5):
            suffix = f'artifacts/{name}_frame{frame}.npz'
            inputs[AUDIT+'/'+suffix] = seal[suffix]
    return dict(cases=original['cases'], input_sha256=inputs,
        source_run=original['source_run'], export_run=original['export_run'],
        source_audit_run=AUDIT, observations=12, frames=60, rays=691200,
        independent_topologies=1, section_realizations=3, views_per_section=4,
        real_worlds_read=[], spacing=original['spacing'], split='synthetic diagnosis only',
        source_permission_mode='diagnostic', source_permission_required=True,
        construction='original sealed case declaration; no new geometry or ray rendering',
        teacher='existing V8 source-precision partial references with explicit source permission',
        comparison='per-case supported anchors/openings/memberships and unknown evidence; not full detection F1',
        training_eligible=False, complete_background=False, optimizer_steps=0,
        limitation='Approximate-source agreement is not uniqueness certification; no physical reachability claim.')


def validate_card(card):
    from .governance import ValidationReport
    from .governance_covered_sensor_export import digest
    try:
        expected=scope();approval=card['approval']
        if (card['schema_version']!=SCHEMA or card['card_id']!=SLUG
                or card['operation']!='teacher_generation' or card['scope']!=expected
                or card['scope_sha256']!=digest(expected)
                or approval['status']!='APPROVED' or approval['scope_sha256']!=digest(expected)
                or approval['authorized_operations']!=['teacher_generation']
                or approval['authorized_gates']!=[3]
                or any(not isinstance(approval.get(k),str) or not approval[k].strip()
                    for k in ('approved_by','approved_at','scope','confirmation_reference'))):
            raise ValueError('exact partial diagnostic scope and authorization required')
    except (KeyError,TypeError,ValueError,OSError) as exc:
        return ValidationReport(False,(str(exc),))
    return ValidationReport(True,())
