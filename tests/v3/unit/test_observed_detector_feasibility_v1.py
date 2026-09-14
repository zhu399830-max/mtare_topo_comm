import numpy as np
import pytest
from mtare_topo.evaluation.observed_detector_feasibility_v1 import fit_template,fixed_correspondence_shuffle,decision


def test_fit_template_never_pads_missing_slots_or_accepts_calibration():
    rows=[dict(split='fit',position_m=np.asarray([[1.,0,0],[3,0,0]])),
          dict(split='fit',position_m=np.asarray([[2.,0,0]]))]
    t=fit_template(rows)
    assert t['slot_support_counts'].tolist()==[2,1]
    assert t['position_m'][:,0].tolist()==[1.5,3.]
    rows[1]['split']='calibration'
    with pytest.raises(ValueError):fit_template(rows)


def test_shuffle_deterministic_without_self_donors_or_label_access():
    sources=[dict(task=str(i)) for i in range(20)]
    a=fixed_correspondence_shuffle(sources)
    assert np.array_equal(a,fixed_correspondence_shuffle(sources))
    assert np.all(a!=np.arange(20)) and len(set(a))==20
    with pytest.raises(ValueError):fixed_correspondence_shuffle([sources[0],sources[0]])


def arguments():
    return dict(observations=282,anchor_tp=180,anchor_fn=6,branch_tp=560,branch_fn=38,
                anchor_known_fp=2,branch_known_fp=3,unresolved_anchors=10,unresolved_branches=20,
                candidate_mean_error_m=.3,template_mean_error_m=.7,shuffled_anchor_tp=130,shuffled_anchor_fn=56)


def test_pass_is_only_pilot_and_all_reject_cannot_pass():
    a=arguments();r=decision(**a)
    assert r['pilot_pass'] and not r['relation_advantage'] and not r['graph_advantage']
    a.update(anchor_tp=0,anchor_fn=186,shuffled_anchor_tp=0,shuffled_anchor_fn=186)
    assert not decision(**a)['pilot_pass']


@pytest.mark.parametrize('change',[dict(branch_known_fp=30),dict(template_mean_error_m=.31),
    dict(shuffled_anchor_tp=175,shuffled_anchor_fn=11),dict(branch_tp=500,branch_fn=98)])
def test_every_required_line_enforced(change):
    a=arguments();a.update(change)
    assert not decision(**a)['pilot_pass']
