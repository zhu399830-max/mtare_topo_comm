from copy import deepcopy
import pytest
from mtare_topo.topology.direction_task_ledger import DirectionTaskLedger
from mtare_topo.evaluation.direction_task_maintenance import score_task_history

def candidates():return [dict(candidate=i,source_frame_keys=['observed_frame'],target_xyz_m=[1,float(i),0]) for i in range(2)]
def reference(order,labels=None):return dict(segment='s',order=order,candidate_labels=labels or {0:'left',1:'right'},observable_tasks=['left','right'],evidence_ref='synthetic fixture, not dataset label')

def replay(matches):
    g=DirectionTaskLedger()
    for i,m in enumerate(matches):g.update(segment='s',order=i,candidates=candidates(),previous_matches=m)
    return g

def test_correct_correspondence_reduces_fragmentation_without_false_merge():
    fresh=replay([{0:None,1:None}]*3)
    correct=replay([{0:None,1:None},{0:0,1:1},{0:0,1:1}])
    refs=[reference(i) for i in range(3)]
    a=score_task_history(fresh.history,refs)['totals'];b=score_task_history(correct.history,refs)['totals']
    assert a['fragmentation_excess']==4 and b['fragmentation_excess']==0
    assert b['conflicting_task_ids']==0 and b['reference_tasks_not_recovered']==0
    assert len(correct.tasks)==2 and correct.snapshot()['confirmed_edges']==[]

def test_bad_merge_is_not_rewarded_for_fewer_tasks():
    g=replay([{0:None,1:None},{0:0,1:0}])
    score=score_task_history(g.history,[reference(0),reference(1)])['totals']
    assert score['conflicting_task_ids']==1

def test_unknown_is_neither_negative_nor_silently_deleted():
    g=replay([{0:None,1:None}]);ref=reference(0,{0:'left',1:None});ref['observable_tasks']=['left']
    t=score_task_history(g.history,[ref])['totals']
    assert t['unknown_candidate_outputs']==1 and len(g.history[0]['assignments'])==2
    assert t['reference_tasks_not_recovered']==0

def test_no_three_direction_requirement_and_missing_frame_preserves_old_tasks():
    g=replay([{0:None,1:None}]);g.update(segment='s',order=1,candidates=[],previous_matches={})
    assert len(g.tasks)==2 and g.tasks[0]['state']=='provisional_unattempted'

def test_cross_segment_reference_rejected_atomically():
    g=replay([{0:None,1:None}]);before=g.snapshot()
    with pytest.raises(ValueError):g.update(segment='other',order=1,candidates=candidates(),previous_matches={0:0,1:1})
    assert g.snapshot()==before
    g.update(segment='other',order=1,candidates=candidates(),previous_matches={0:None,1:None})
    assert len(g.tasks)==4

def test_prefix_is_immutable():
    g=replay([{0:None,1:None}]);first=deepcopy(g.history)
    g.update(segment='s',order=1,candidates=candidates(),previous_matches={0:0,1:1})
    assert g.history[:1]==first
    snap=g.snapshot();snap['history'][0]['order']=99;assert g.history[0]['order']==0

def test_teacher_rejected_from_online_input():
    g=DirectionTaskLedger();c=candidates();c[0]['true_task']='hidden'
    with pytest.raises(ValueError):g.update(segment='s',order=0,candidates=c,previous_matches={0:None,1:None})

def test_missing_covered_task_is_counted_separately():
    g=DirectionTaskLedger();g.update(segment='s',order=0,candidates=candidates()[:1],previous_matches={0:None})
    t=score_task_history(g.history,[reference(0)])['totals']
    assert t['reference_tasks_not_recovered']==1 and t['conflicting_task_ids']==0
