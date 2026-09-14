"""One registered allocator-only correction; unchanged population and teacher."""
import argparse
import copy
import gzip
import json
import os
from pathlib import Path
import multiview_historical_teacher_v1 as original
from development_grids_v1 import write
from surface_features_v1 import sha

ROOT=original.ROOT
SLUG='gse_multiview_historical_teacher_v1r'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed20260906'


def freeze():
    if any((ROOT/p).exists() for p in (SPEC,RUN)):raise FileExistsError('no corrective overwrite/refreeze')
    old=original.load_json(ROOT/original.SPEC)
    previous=ROOT/original.RUN
    if original.load_json(previous/'RUN_STATE.json')['state']!='FAILED':raise ValueError('terminal failed predecessor required')
    spec=copy.deepcopy(old);spec['slug']=SLUG
    spec['command']=['env','MALLOC_ARENA_MAX=2',*old['command'][1:]]
    command=spec['command'];command[command.index('tools/v3/multiview_historical_teacher_v1.py')]='tools/v3/multiview_historical_teacher_v1r.py'
    command[command.index('--spec')+1]=str(ROOT/SPEC)
    command[command.index('--run-dir')+1]=str(ROOT/RUN)
    spec['method']+=' Registered executor-only correction1: MALLOC_ARENA_MAX=2 inherited by historical worker; identical archive, algorithms,2259 windows and address-space limits. Reevaluate all windows once; compare prior52 targets and raw evidence, no selective sample removal.'
    spec['fallback']='Fail and seal; no further automatic allocator correction, retry, resampling or teacher modification.'
    spec['executor_correction']=dict(index=1,environment={'MALLOC_ARENA_MAX':'2'},
        predecessor=str(previous.relative_to(ROOT)),predecessor_seal_sha256=sha(previous/'artifacts/evidence_sha256.txt'),
        reason='Synthetic57600-ray Open3D box expands default allocator virtual space above3GiB at~315MiBRSS;2arenas preserves all output hashes and fits original address-space cap.')
    sealed={p:h for h,p in (line.split('  ',1) for line in (previous/'artifacts/evidence_sha256.txt').read_text().splitlines())}
    prefix=[]
    for index in range(52):
        path=str((previous/'artifacts'/f'observation_{index:05d}.json.gz').relative_to(ROOT))
        if sha(ROOT/path)!=sealed[path]:raise ValueError('predecessor observation drift')
        prefix.append(dict(path=path,sha256=sealed[path]))
    spec['unchanged_prefix_evidence']=prefix
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{original.CARD})
    spec['source_sha256']={p:sha(ROOT/p) for p in files}
    # Same approved card and exact data operation; record standing correction authority explicitly.
    spec['user_authorization']['confirmation_reference']+=' One evidence-backed executor-only resource correction under standing autonomous authorization; failed predecessor retained.'
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,observations=2259,executed=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path);args=parser.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:
        spec=original.load_json(args.spec)
        if os.environ.get('MALLOC_ARENA_MAX')!='2' or spec['executor_correction']['environment']!={'MALLOC_ARENA_MAX':'2'}:
            raise ValueError('registered allocator environment required')
        previous=ROOT/spec['executor_correction']['predecessor']
        if sha(previous/'artifacts/evidence_sha256.txt')!=spec['executor_correction']['predecessor_seal_sha256']:
            raise ValueError('predecessor seal drift')
        class ComparingClient(original.HistoricalTeacherClient):
            completed=0
            def request(self,bundle):
                response=super().request(bundle)
                if self.completed<len(spec['unchanged_prefix_evidence']):
                    pin=spec['unchanged_prefix_evidence'][self.completed]
                    old=json.loads(gzip.decompress(original.read_pinned(ROOT,pin['path'],pin['sha256'])))
                    for key in ('raw_interfaces','produced_targets'):
                        if original.digest(response[key])!=original.digest(old[key]):
                            raise ValueError('allocator correction changed prior output: '+key)
                self.completed+=1
                return response
        original.HistoricalTeacherClient=ComparingClient
        raise SystemExit(original.execute(spec,args.run_dir))
    else:parser.error('--freeze or --spec/--run-dir required')
