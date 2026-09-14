"""Read-only coverage inventory; never upgrades unscored cases to passes."""
import _bootstrap
from collections import Counter,defaultdict
import gzip,hashlib,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/gate3_semantics/gate3_20260908_gse_synthetic_matrix_v1_seed20260906'


def main():
    pins={p:h for h,p in (l.split('  ',1) for l in (RUN/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    groups=defaultdict(Counter);count=0;counterexample=None
    for p in sorted((RUN/'artifacts').glob('*.json.gz')):
        if p.name.endswith('_control.json.gz'):continue
        b=p.read_bytes()
        if hashlib.sha256(b).hexdigest()!=pins[str(p.relative_to(ROOT))]:raise ValueError('evidence drift')
        d=json.loads(gzip.decompress(b))
        if d['independent_score']['status']!='UNSCORED_NOT_PASS':continue
        t=d['produced_targets']['record'];kind=d['case']['program']['type'];count+=1
        groups[kind][(len(t['anchors']),len(t['openings']))]+=1
        if p.stem=='overlap_ambiguity__circle__view2.json':counterexample=d
    if count!=99 or counterexample is None:raise ValueError('fixed coverage population drift')
    for kind,counts in groups.items():
        print(json.dumps(dict(kind=kind,conditions=sum(counts.values()),
            outputs=[dict(anchors=a,openings=o,count=n) for (a,o),n in sorted(counts.items())])))
    # Independent certificate for the declared circular central prototype:
    # at world z=0 the two negative-Y tube axes are x=0 and x=.5;
    # along x in [0,.5], y=-sqrt(100-.04^2-x^2), the distance to either
    # axis is <=.5. This curve lies on the sensor's 10m sphere and is
    # inside BOTH actual 64-gon prisms (inradius 2*cos(pi/64) > .5).
    c=counterexample['case'];edges={e['id']:e for e in c['program']['edges']}
    assert c['poses_world_m'][-1]==[0.,0.,.04] and c['half_axes_m']==[2.,2.] and c['shape_exponent']==2.
    assert edges['p2']['points']==[[0.,1.,0.],[0.,-30.,0.]]
    assert edges['other_p2']['points']==[[.5,1.,0.],[.5,-30.,0.]]
    radius=10.;height=.04;delta=.5;inradius=2*math.cos(math.pi/64)
    ymin=-math.sqrt(radius**2-height**2);ymax=-math.sqrt(radius**2-height**2-delta**2)
    assert -30<ymin<=ymax<1 and delta<inradius
    record=counterexample['produced_targets']['record'];assert len(record['openings'])==2
    print(json.dumps(dict(case_id=c['case_id'],emitted_openings=2,
        both_openings_share_connected_free_window_component=True,
        shared_curve_world_x_interval=[0.,delta],shared_curve_world_y_bounds=[ymin,ymax],world_z=0.,
        maximum_distance_to_either_axis_m=delta,actual_polygon_inradius_m=inradius,
        statement='Two construction-reference targets are not two disconnected physical apertures.',
        visibility_completeness_certified=False,full_label_qualification=False,new_labels=0)))

if __name__=='__main__':main()
