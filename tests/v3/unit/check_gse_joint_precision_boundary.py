"""Actual synthetic joint pipeline at a translated/rotated cap intersection."""
import json
from check_gse_joint_producer_backend import main
from mtare_topo.teacher.gse_junction_interface_diagnostic_v1 import diagnose_observation
from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets


if __name__=='__main__':
    bundle=main(return_bundle=True,sensor_positions=[[1.,1.,.04]]*5,
        rigid_yaw_deg=17.,translation_xyz_m=(70.000003,70.000003,0.))
    raw=diagnose_observation(bundle)
    settings=dict(axial_spacing_m=.05,angular_segments=64,field_spacing_m=.01)
    old=produce_joint_reference_targets(bundle,raw,**settings)
    new=produce_joint_reference_targets(bundle,raw,qualify_cap_precision=True,**settings)
    report=dict(old_anchors=len(old['record']['anchors']),new_anchors=len(new['record']['anchors']),
        old_membership=old['record']['membership'],new_membership=new['record']['membership'],
        cap_reports={kind:{str(k):dict(saved=v['saved_ray_count'],
            stable=len(v['stable_'+kind+'_ray_indices']),unknown=v['unknown_hit_count'])
            for k,v in audit['interfaces'].items()}
            for kind,audit in new['terminal_exclusion_cap_precision'].items()})
    print(json.dumps(report),flush=True)
    assert any(v['unknown'] for rows in report['cap_reports'].values() for v in rows.values()), 'must exercise actual quantization boundary candidates'
    assert report['old_anchors']>report['new_anchors'], 'must demonstrate whole-chain anchor withdrawal, not merely a no-regression case'
    assert new['record']['score_region']==old['record']['score_region']
    assert not new['full_training_gate_eligible']
    print('JOINT_BOUNDARY_WITHDRAWAL_SYNTHETIC_PASS')
