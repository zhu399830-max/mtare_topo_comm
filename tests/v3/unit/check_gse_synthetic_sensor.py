"""Four fixed synthetic observations; no labels or repository output assets."""
import json
import numpy as np
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.data.gse_synthetic_sensor import SyntheticSensorScene
from mtare_topo.evaluation.gse_synthetic_straight_oracle import visible_straight_cap_rays

rows={r['case_id']:r for r in matrix()}
for key in ('straight__circle__view2','terminal__circle__view2'):
    case=rows[key];scene=SyntheticSensorScene(case);bundle=scene.render(case)
    oracle=visible_straight_cap_rays(case)
    if key.startswith('straight'):assert not any(oracle.values())
    else:
        assert len(oracle[1])>0 and not oracle[0]
        assert bundle['student']['valid_mask'].reshape(-1)[oracle[1]].all()
    print(json.dumps(dict(case=key,valid_returns=int(bundle['student']['valid_mask'].sum()),
        analytic_cap_support={str(k):len(v) for k,v in oracle.items()})),flush=True)
case=rows['hidden_branch__circle__view0']
a=SyntheticSensorScene(case).render(case)
b=SyntheticSensorScene(case,hidden_control=True).render(case)
same={k:bool(np.array_equal(v,b['student'][k])) for k,v in a['student'].items()}
print(json.dumps(dict(case=case['case_id'],paired_student_fields_equal=same)),flush=True)
ra=a['student']['ranges_m'];rb=b['student']['ranges_m'];different=ra!=rb
print(json.dumps(dict(different_ranges=int(different.sum()),max_difference_m=float(np.abs(ra-rb).max()),
    differing_range_min_m=float(np.minimum(ra,rb)[different].min()) if different.any() else None,
    differing_range_max_m=float(np.maximum(ra,rb)[different].max()) if different.any() else None)),flush=True)
assert all(same.values()),'hidden control is not observationally equivalent; do not edit rays'
print('FOUR_SYNTHETIC_OBSERVATION_ADAPTER_CHECKS_PASS')
