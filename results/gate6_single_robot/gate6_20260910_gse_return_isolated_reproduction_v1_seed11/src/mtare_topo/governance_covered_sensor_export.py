"""Exact sealed synthetic sensor conversion, no teacher or training scope."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'v3_covered_sensor_export_card_v1'
SLUG = 'gse_covered_sensor_export_v1'
SOURCE = 'results/gate3_semantics/gate3_20260908_gse_double_population_covered_v1_seed20260906'
SEAL_SHA = '687cd803127e479a88cd1605c3c35bd4ffb5cc2dfa19649af8e80864fa3474e4'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def scope():
    seal_path = ROOT/SOURCE/'SHA256_SEAL.json'
    raw = seal_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SEAL_SHA:
        raise ValueError('source seal changed')
    seal = json.loads(raw)
    inputs = {SOURCE+'/'+p: seal[p] for p in
              ('metrics/rays.jsonl', 'metrics/summary.json', 'config/source_config')}
    inputs[SOURCE+'/SHA256_SEAL.json'] = SEAL_SHA
    return dict(source_run=SOURCE, input_sha256=inputs,
        cases=[f'double_junction__{section}__view{view}' for section in
               ('circle', 'ellipse', 'rounded_rectangle') for view in range(4)],
        observations=12, frames=60, rays=691200, independent_topologies=1,
        section_realizations=3, viewpoints_per_realization=4,
        history_indices=[0,1,2,3,4], nominal_history_spacing_m=.1,
        angular_shape=[16,720], spatial_source='sealed exact float64 original poses',
        split='synthetic diagnostic only; no train/validation/test payload',
        real_worlds_read=[], teacher_labels_generated=0, optimizer_steps=0,
        training_eligible=False, source_completeness_qualified=False,
        operation='lossless_reported_provenance_and_float32_sensor_conversion_no_rerender')


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        expected = scope()
        if (card['schema_version'] != SCHEMA or card['card_id'] != SLUG
            or card['operation'] != 'data_export' or card['scope'] != expected
            or card['scope_sha256'] != digest(expected)):
            raise ValueError('exact scope mismatch')
        a = card['approval']
        if (a['status'] != 'APPROVED' or a['authorized_operations'] != ['data_export']
            or a['authorized_gates'] != [3] or a['scope_sha256'] != digest(expected)
            or any(not isinstance(a.get(k), str) or not a[k].strip() for k in
                   ('approved_by','approved_at','scope','confirmation_reference'))):
            raise ValueError('exact approval binding missing')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return ValidationReport(False, ('covered sensor export: '+str(exc),))
    return ValidationReport(True, ())
