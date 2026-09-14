"""One complete original-ten precision diagnostic, with explicit differences."""
import argparse
from pathlib import Path
import v8_original_ten_probe as runner
from mtare_topo.governance_ten_precision import SCHEMA,SLUG,POLICY,SCOPE_SHA,validate_card

runner.SCHEMA=SCHEMA;runner.SLUG=SLUG;runner.POLICY=POLICY;runner.SCOPE_SHA=SCOPE_SHA
runner.validate_card=validate_card
runner.CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
runner.SPEC='configs/v3/gate3/'+SLUG+'.json'
runner.RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed20260906'
runner.ENTRYPOINT='tools/v3/original_ten_precision.py'
runner.COMPLETION_STATUS='ORIGINAL_TEN_PRECISION_DIAGNOSTIC_COMPLETE_NOT_QUALIFICATION'


def produce(bundle,raw,card):
    from mtare_topo.teacher.gse_joint_reference_targets_v8 import produce_joint_reference_targets
    return produce_joint_reference_targets(bundle,raw,qualify_cap_precision=True,**card['scope']['geometry_settings'])


def compare(old,new):
    a,b=old['record'],new['record']
    if old['source_binding']!=new['source_binding']:raise ValueError('source binding changed')
    for k in ('openings','score_region','source_frame_indices','coordinate_frame'):
        if a[k]!=b[k]:raise ValueError('unexpected '+k+' change')
    old_pos=[tuple(x['position_m']) for x in a['anchors']]
    new_pos=[tuple(x['position_m']) for x in b['anchors']]
    if len(set(old_pos))!=len(old_pos) or len(set(new_pos))!=len(new_pos):raise ValueError('ambiguous anchor geometry')
    changes=[]
    for pos in sorted(set(old_pos)|set(new_pos)):
        i=old_pos.index(pos) if pos in old_pos else None
        j=new_pos.index(pos) if pos in new_pos else None
        changes.append(dict(position_m=list(pos),old_anchor=i,new_anchor=j,
            status='ADDED' if i is None else 'WITHDRAWN' if j is None else 'RETAINED',
            old_memberships=None if i is None else [r[i] for r in a['membership']],
            new_memberships=None if j is None else [r[j] for r in b['membership']]))
    return dict(old_anchors=len(old_pos),new_anchors=len(new_pos),changes=changes,
        old_counts=runner.counts(a),new_counts=runner.counts(b),complete_annotation=False)


runner.produce_output=produce;runner.compare=compare

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--freeze',action='store_true');parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.freeze:
        runner.freeze(spec_overrides=dict(
            question='After correcting source precision, what valid partial supervision remains on the original ten cases?',
            method='V8 with explicit entering/leaving source-interval policy and original .05/64/.025 settings; exact duplicate sampling fix; no first-return rerender.',
            baseline='Original sealed V6 per-case targets. Old unsupported answers may be withdrawn; every addition, withdrawal and membership change is reported, not forced unchanged.',
            acceptance_criteria=['Exact original ten scope and source hashes; no new data, training or C08-C10.',
                'Opening geometry, poses, frame identity and score region unchanged; all anchor and membership differences preserved.',
                'Complete all ten or stop and seal failure; no retry or expansion.',
                'Diagnostic completion is not population label qualification or scientific PASS.']))
    elif args.spec and args.run_dir:raise SystemExit(runner.execute(runner.load_json(args.spec),args.run_dir))
    else:parser.error('--freeze or --spec/--run-dir required')
