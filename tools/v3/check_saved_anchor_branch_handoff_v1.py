"""Read-only 3x307 saved prediction handoff; no model or teacher execution."""
from _bootstrap import PROJECT_ROOT as ROOT
from collections import Counter
import hashlib
import io
import json
import numpy as np
from mtare_topo.topology.saved_anchor_branch_adapter_v1 import convert_saved_prediction


def main():
    run=ROOT/'results/gate3_semantics/gate3_20260909_gse_development_corrective_train_v1_seed0'
    seal=(run/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(seal).hexdigest()!='750372954133fcfa17c03cdc294aac6bc0fa3365abad687989af1a00cb7210cb':
        raise ValueError('saved model evidence seal drift')
    hashes=dict((line.split('  ',1)[1],line.split('  ',1)[0]) for line in seal.decode().splitlines())
    def read(path):
        raw=path.read_bytes();name=str(path.relative_to(ROOT))
        if hashlib.sha256(raw).hexdigest()!=hashes[name]:raise ValueError('saved output drift')
        return raw
    native=ROOT/'results/gate3_semantics/gate3_20260909_gse_native_graph_population_v2_seed0'
    native_seal=(native/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(native_seal).hexdigest()!='a6235d5c5f2062e0133ecd086529cd3abe36fdafb1601295a76b9e0287871949':
        raise ValueError('native source seal drift')
    nh=dict((line.split('  ',1)[1],line.split('  ',1)[0]) for line in native_seal.decode().splitlines())
    path=native/'artifacts/native_graph_manifest.json';raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=nh[str(path.relative_to(ROOT))]:raise ValueError('source mapping drift')
    def key(source):return json.dumps(source,sort_keys=True,separators=(',',':'))
    bindings={key(e['source']):e for e in json.loads(raw)['observations']}
    totals={};count=0
    for method in ('r0','r1','r2'):
        rows=[json.loads(s) for s in read(run/'metrics'/f'{method}_final.jsonl').decode().splitlines()]
        if len(rows)!=307 or {key(r['source']) for r in rows}!=set(bindings):raise ValueError('population drift')
        metric=Counter()
        for row in rows:
            source=row['source'];b=bindings[key(source)]
            with np.load(io.BytesIO(read(run/'artifacts'/row['prediction_file'])),allow_pickle=False) as p:
                arrays={k:p[k] for k in p.files}
            # Independent fragments deliberately have independent opaque stream keys.
            stream=hashlib.sha256(key(source).encode()).hexdigest()
            obs=convert_saved_prediction(arrays,stream_key=stream,decision_index=0,
                frame_orders=source['frame_rows'],input_binding_sha256=b['input_binding_sha256'])
            selected=[a.anchor_query_index for a in obs.anchors]
            if any(s['selected_anchor_query_indices']!=selected or s['source']!=source
                   or s['split']!=b['split'] for s in row['scores']):raise ValueError('saved scoring selection drift')
            for a in obs.anchors:
                expected=np.flatnonzero(arrays['branch_logits'][a.anchor_query_index]>=0).tolist()
                if [d.branch_query_index for d in a.branches]!=expected:raise ValueError('branch selection drift')
                for d in a.branches:
                    if tuple(arrays['directions'][a.anchor_query_index,d.branch_query_index].astype(float))!=d.direction_current_sensor:
                        raise ValueError('direction changed')
            metric[b['split']+'_anchors']+=len(obs.anchors)
            metric[b['split']+'_directions']+=sum(len(a.branches) for a in obs.anchors)
            count+=1
        totals[method]=dict(metric)
    print(json.dumps(dict(status='SAVED_PREDICTION_INTERFACE_IDENTICAL_NOT_GRAPH_SUCCESS',
        converted=count,observations_per_method=307,methods=totals,model_invocations=0,
        teacher_invocations=0,cross_fragment_connections=0,
        scope='Selected local query indices and geometry preserved; score anchor selection checked in all saved variants; branch indices checked against original logits. No new detection scores.'),indent=2))


if __name__=='__main__':main()
