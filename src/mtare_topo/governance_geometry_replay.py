import hashlib
from pathlib import Path
from .governance_continuous_geometry import digest

SCHEMA='v3_saved_geometry_replay_card_v1'
SLUG='gse_saved_geometry_replay_v1'
PARENT='results/gate3_semantics/gate3_20260910_gse_continuous_geometry_v1_seed0'


def scope(root):
    seal=(root/PARENT/'artifacts/evidence_sha256.txt').read_text()
    entries=dict((line.split('  ',1)[1],line.split('  ',1)[0]) for line in seal.splitlines())
    groups={}
    for v in ('c1_mixed','ellipse','rounded_rectangle'):
        task='S09_flat_complex_C04__'+v
        groups[task]=[]
        for i in range(10):
            p=f'{PARENT}/artifacts/{task}_{i:02d}.json'
            groups[task].append(dict(path=p,sha256=entries[p]))
    return dict(groups=groups,parent='S09_flat_complex_C04',split='historically_used_fit',
        observations=30,unique_source_frames=42,independent_route=1,variants=3,
        original_spacing_m=1.,timing='frame_order_not_wall_time',teacher_reads=0,
        anchor_spacing_m=4.,lookahead_m=4.,scope='given_pose_metric_graph_and_geometry_targets_not_closed_loop')


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    try:
        s=scope(Path(__file__).resolve().parents[2]);a=card.get('approval',{})
        if card.get('schema_version')!=SCHEMA or card.get('scope')!=s:errors.append('exact saved30 scope required')
        if (a.get('scope_sha256')!=digest(s) or a.get('status')!='APPROVED'
                or a.get('authorized_operations')!=['topology_replay'] or a.get('authorized_gates')!=[4]
                or not a.get('confirmation_reference')):errors.append('exact user-authorized replay required')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
