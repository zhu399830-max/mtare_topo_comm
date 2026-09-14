"""Three fixed actual synthetic positive controls; no real data reads."""
import json
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.gse_synthetic_sensor import SyntheticSensorScene
from mtare_topo.evaluation.gse_synthetic_field_scoring import score_case
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets

cases={c['case_id']:c for c in matrix()};failures=[]
for key in ('straight__circle__view2','terminal__circle__view2','T__circle__view2'):
    case=cases[key];bundle=SyntheticSensorScene(case).render(case)
    target=produce_joint_reference_targets(bundle,diagnose_observation(bundle),
        axial_spacing_m=.05,angular_segments=64,field_spacing_m=.025,qualify_cap_precision=True)
    result=score_case(case,target['record'])
    print(json.dumps(result),flush=True)
    if result['status']!='FIXTURE_GEOMETRY_PASS':failures.append(key)
assert not failures, 'independent positive controls failed: '+str(failures)
