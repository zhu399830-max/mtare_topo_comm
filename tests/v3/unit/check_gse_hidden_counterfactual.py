"""One fixed observed sequence, two constructions; real synthetic target path."""
import json
from copy import deepcopy
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.gse_synthetic_sensor import SyntheticSensorScene
from mtare_topo.data.gse_hidden_counterfactual import remove_unobserved_hidden_operand
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets

case=next(c for c in matrix() if c['case_id']=='hidden_branch__circle__view0')
a=SyntheticSensorScene(case).render(case);b=remove_unobserved_hidden_operand(case,a)
assert all(np.array_equal(v,b['student'][k]) for k,v in a['student'].items())
settings=dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025,qualify_cap_precision=True)
old=produce_joint_reference_targets(a,diagnose_observation(a),**settings)
new=produce_joint_reference_targets(b,diagnose_observation(b),**settings)
assert old['record']==new['record'],'same fixed observation gained a contradictory local target'
assert not old['record']['anchors'] and not new['record']['anchors']
assert not old['full_training_gate_eligible'] and not new['full_training_gate_eligible']
for damage in ('wrong_case','invalid_ray'):
    bad=deepcopy(a)
    if damage=='wrong_case':bad['source']['case_id']='foreign'
    else:
        bad['student']['valid_mask'][0,0,0]=0
        bad['sensor_teacher_only']['primitive_membership_code'][0,0,0]=0
    try:remove_unobserved_hidden_operand(case,bad)
    except ValueError:pass
    else:raise AssertionError('counterfactual must reject '+damage)
print(json.dumps(dict(status='FIXED_OBSERVATION_HIDDEN_CONTROL_PASS',anchors=0,
    openings=len(new['record']['openings']),student_bitwise_unchanged=True,
    provenance=b['counterfactual_provenance'],tamper_cases_rejected=2)),flush=True)
