"""Exact metadata-only scope construction for the fixed source audit."""
import hashlib
import json
from .governance_covered_sensor_export import ROOT, SOURCE, SEAL_SHA, digest

SCHEMA='v3_source_compatibility_card_v1'
SLUG='gse_source_compatibility_v1'
EXPORT='results/gate3_semantics/gate3_20260908_gse_covered_sensor_export_v1_seed20260906'
EXPORT_SEAL='b972519dca54a36466f56b7922b6ed2b120d44da725bbb31ce1c66ba2486f2d9'


def scope():
    cases=[f'double_junction__{s}__view{v}' for s in ('circle','ellipse','rounded_rectangle') for v in range(4)]
    inputs={}
    for source, expected, files in [
        (SOURCE,SEAL_SHA,['config/source_config']),
        (EXPORT,EXPORT_SEAL,['artifacts/'+c+ext for c in cases for ext in ('.student.npz','.diagnostic.npz','.json')])]:
        p=source+'/SHA256_SEAL.json';raw=(ROOT/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('source seal drift')
        seal=json.loads(raw);inputs[p]=expected
        inputs.update({source+'/'+f:seal[f] for f in files})
    return dict(cases=cases,source_run=SOURCE,export_run=EXPORT,input_sha256=inputs,
        observations=12,frames=60,rays=691200,independent_topologies=1,section_realizations=3,
        spacing='original five poses at nominal0.1m;16x720 angles',split='synthetic diagnosis only',
        real_worlds_read=[],teacher_labels_generated=0,optimizer_steps=0,
        numerical_geometry_certified=False,training_eligible=False,
        comparison='reported exits versus closed prism surface candidates in float32 rounding cells; aggregate discrepancies, do not relabel')


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    try:
        s=scope();a=card['approval']
        if (card['schema_version']!=SCHEMA or card['card_id']!=SLUG or card['operation']!='audit'
            or card['scope']!=s or card['scope_sha256']!=digest(s)
            or a['status']!='APPROVED' or a['scope_sha256']!=digest(s)
            or a['authorized_operations']!=['audit'] or a['authorized_gates']!=[3]
            or any(not isinstance(a.get(k),str) or not a[k].strip() for k in
                   ('approved_by','approved_at','scope','confirmation_reference'))):
            raise ValueError('exact audit scope and authorization required')
    except (KeyError,TypeError,ValueError,OSError) as exc:
        return ValidationReport(False,(str(exc),))
    return ValidationReport(True,())
