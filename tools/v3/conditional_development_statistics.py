"""Paired parent-level statistics, without best-seed/threshold selection.

Input rows are complete per-observation results, never only successful cases.
The three geometric realizations remain repeated measures of one parent.
"""
import math
import random
from statistics import mean

METRICS=('axis_mae','height_mae_m','constant_axis_mae','constant_height_mae_m')


def summarize(rows,identities):
    expected={i['case']:i for i in identities}
    if len(expected)!=len(identities):raise ValueError('duplicate frozen case')
    indexed={}
    for r in rows:
        key=(r['variant'],r['case'])
        if key in indexed:raise ValueError('duplicate prediction case')
        if r['variant'] not in 'ABC' or len(r['variant'])!=1 or r['case'] not in expected:raise ValueError('unexpected prediction case')
        i=expected[r['case']]
        if (r['parent'],r['task'],r['physical_edge'])!=(i['parent_id'],i['task'],i['physical_edge_id']):raise ValueError('prediction identity mismatch')
        for category in ('all','same','cross'):
            m=r['metrics'][category];count=m['count']
            if type(count) is not int or count<0:raise ValueError('invalid known count')
            for metric in METRICS:
                v=m[metric]
                if (count==0 and v is not None) or (count>0 and (v is None or not math.isfinite(v) or v<0)):
                    raise ValueError('missing/nonfinite score or unknown treated as zero')
        indexed[key]=r
    if set(indexed)!={(v,c) for v in 'ABC' for c in expected}:raise ValueError('incomplete paired population')
    for case in expected:
        a=indexed['A',case]
        for v in 'BC':
            b=indexed[v,case]
            if (a['unknown_relations'],a['output_relations'])!=(b['unknown_relations'],b['output_relations']):raise ValueError('unequal query population')
            for cat in ('all','same','cross'):
                for k in ('count','constant_axis_mae','constant_height_mae_m'):
                    if a['metrics'][cat][k]!=b['metrics'][cat][k]:raise ValueError('unequal reference scoring')
    parents=sorted({i['parent_id'] for i in identities});table=[]
    for parent in parents:
        cases=[c for c,i in expected.items() if i['parent_id']==parent]
        for variant in 'ABC':
            rr=[indexed[variant,c] for c in cases];categories={}
            for cat in ('all','same','cross'):
                mm=[r['metrics'][cat] for r in rr if r['metrics'][cat]['count']>0]
                categories[cat]=dict(observations_scored=len(mm),observations_total=len(cases),
                    **{k:mean(m[k] for m in mm) if mm else None for k in METRICS})
            table.append(dict(parent=parent,variant=variant,metrics=categories,
                output_relations=sum(r['output_relations'] for r in rr),unknown_relations=sum(r['unknown_relations'] for r in rr)))
    pairs={};by_parent={(r['variant'],r['parent']):r for r in table}
    for left,right in [('C','B'),('B','A'),('C','A')]:
        for metric in ('axis_mae','height_mae_m'):
            differences=[];included=[]
            for parent in parents:
                x=by_parent[left,parent]['metrics']['cross'][metric];y=by_parent[right,parent]['metrics']['cross'][metric]
                if x is not None and y is not None:included.append(parent);differences.append(x-y)
            # Fixed parent bootstrap, descriptive only with this small/exposed
            # development set. Never resample patches as independent worlds.
            rng=random.Random(20260911);n=len(differences)
            boot=sorted(mean(rng.choices(differences,k=n)) for _ in range(5000)) if n>=2 else []
            pairs[f'{left}_minus_{right}_{metric}']=dict(parent_ids=included,parents=n,
                differences=differences,mean=mean(differences) if n else None,
                descriptive_bootstrap_95=[boot[124],boot[4874]] if boot else None,
                improved_parents=sum(d<0 for d in differences),unchanged_parents=sum(d==0 for d in differences))
    return dict(observations=len(expected),parents=len(parents),per_parent=table,paired_cross_component=pairs,
        interpretation='Conditional-head parent-held-out C07 development only; old encoder selected on C07. No strict test, topology, observability or exploration claim.',
        aggregation='Original within-observation component-pair weights, then equal eligible observations within parent, then equal parents. Missing support reported, never zero-filled.',
        bootstrap='5000 fixed-seed paired-parent resamples; descriptive small-population interval, not evidence of reliable deployment')
