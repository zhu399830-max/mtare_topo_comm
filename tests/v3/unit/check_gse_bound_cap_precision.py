"""Actual CSG scans and bound interfaces; synthetic data only, no target calls."""
import copy
import json
import numpy as np
from check_gse_joint_producer_backend import main as make_bundle
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.evaluation.gse_bound_cap_precision import audit_bound_cap_precision
from mtare_topo.teacher.gse_reference_anchor_targets_v1 import _produce_junction_reference_targets
from mtare_topo.teacher.gse_reference_anchor_targets_v3 import produce_junction_reference_targets


def check(name,positions):
    bundle=make_bundle(return_bundle=True,sensor_positions=positions)
    raw=diagnose_observation(bundle)
    snapshot=copy.deepcopy(raw)
    result=audit_bound_cap_precision(bundle,raw,axial_spacing_m=.05,angular_segments=64)
    assert raw==snapshot
    assert not result['label_or_physical_connectivity_certified']
    cap_only=_produce_junction_reference_targets(bundle,raw,
        cap_source_settings=dict(axial_spacing_m=.05,angular_segments=64))
    assert len(cap_only['record']['anchors'])==int(result['nodes']['junction0']['three_cap_directions_numerically_supported'])
    settings=dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.01)
    legacy=produce_junction_reference_targets(bundle,raw,**settings)
    guarded=produce_junction_reference_targets(bundle,raw,qualify_cap_precision=True,**settings)
    assert legacy['record']==guarded['record']
    assert 'cap_precision_audit' not in legacy and 'cap_precision_audit' in guarded
    assert raw==snapshot
    print(json.dumps(dict(scene=name,nodes=result['nodes'],interfaces={str(k):dict(
        saved=v['saved_ray_count'],stable=len(v['stable_entering_ray_indices']),
        unknown=v['unknown_hit_count']) for k,v in result['interfaces'].items()},
        cap_only_anchors=len(cap_only['record']['anchors']),
        with_interior_lateral_anchors=len(guarded['record']['anchors']))),flush=True)
    damaged=copy.deepcopy(raw);damaged['source']['source_sequence_id']+=1
    try:audit_bound_cap_precision(bundle,damaged,axial_spacing_m=.05,angular_segments=64)
    except ValueError:pass
    else:raise AssertionError('foreign observation accepted')
    return result


if __name__=='__main__':
    inside=check('source_interior_without_cap_entrance',[[0.,-3.+.1*i,.04] for i in range(5)])
    assert inside['nodes']['junction0']['stable_distinct_cap_directions']==0
    stable=check('explicit_cap_crossing_views',[[-3.,0.,.04],[-2.,0.,.04],[2.,0.,.04],[3.,0.,.04],[0.,1.5,.04]])
    assert stable['nodes']['junction0']['stable_distinct_cap_directions']==3
    boundary=check('on_one_cap',[[1.,0.,.04] for _ in range(5)])
    assert not boundary['nodes']['junction0']['three_cap_directions_numerically_supported']
    print('BOUND_CAP_SOURCE_SYNTHETIC_CHECKS_PASS')
