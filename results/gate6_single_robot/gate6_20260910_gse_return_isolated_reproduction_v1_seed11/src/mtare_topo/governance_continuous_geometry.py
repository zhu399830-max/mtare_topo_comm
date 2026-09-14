"""Exact existing-package scope for the nonlearning geometry integration run."""
import hashlib
import json
from pathlib import Path

PARENT='results/gate3_semantics/gate3_20260909_gse_continuous_model_input_export_v1_seed20260906'
SCHEMA='v3_continuous_geometry_card_v1'
SLUG='gse_continuous_geometry_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def compile_scope(root):
    base=root/PARENT
    manifest=base/'artifacts/input_manifest.json'
    payload=manifest.read_bytes(); data=json.loads(payload)
    seal=(base/'artifacts/evidence_sha256.txt').read_text()
    expected=hashlib.sha256(payload).hexdigest()
    if expected+'  '+str(manifest.relative_to(root)) not in seal.splitlines():
        raise ValueError('parent manifest not in source seal')
    tasks=[]
    for item in data['tasks']:
        row={k:item[k] for k in ('task','observations','bytes','sha256','source_frames','source_sequence_ids')}
        row['path']=str((base/item['file']).relative_to(root))
        if row['sha256']+'  '+row['path'] not in seal.splitlines():
            raise ValueError('package absent from parent seal')
        tasks.append(row)
    names={'S09_flat_complex_C04__'+v for v in ('ellipse','rounded_rectangle','c1_mixed')}
    if {r['task'] for r in tasks}!=names or len(tasks)!=3 or any(r['observations']!=10 for r in tasks):
        raise ValueError('exact30 existing development windows required')
    return dict(tasks=tasks,parent_manifest_sha256=expected,world='S09_flat_complex_C04',
        independent_routes=1,variants=3,observations=30,unique_variant_frames=42,
        split='historically_used_fit',spacing='original consecutive approximately1m positions; five-frame history',
        timing='source order only; no seconds invented',teacher='none read',
        protected_test_access=False,training_updates=0,
        composition_policy=dict(max_residual_m=.01,min_crossing_sine=.1,
            endpoint_tolerance_m=1e-8,maximum_candidates=32))


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        scope=compile_scope(Path(__file__).resolve().parents[2])
        if card.get('schema_version')!=SCHEMA or card.get('card_id')!=SLUG or card.get('operation')!='audit':
            errors.append('wrong geometry-only operation')
        if card.get('scope')!=scope or card.get('scope_sha256')!=digest(scope):
            errors.append('exact geometry scope drift')
        a=card.get('approval',{})
        if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=digest(scope)
                or a.get('authorized_operations')!=['audit'] or a.get('authorized_gates')!=[3]
                or not a.get('confirmation_reference')):
            errors.append('exact operation-bound user authority required')
    except Exception as e:
        errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
