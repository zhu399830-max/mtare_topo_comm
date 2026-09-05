#!/usr/bin/env python3
"""Read-only B0 metric freeze on the sealed validation split; no tuning."""

from __future__ import annotations

import argparse,json,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_contract_pilot import match_headings
from mtare_topo.governance import write_json
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.monotonic()
    root=a.dataset_run.resolve();records=[json.loads(x) for x in open(root/'artifacts/manifest.jsonl') if json.loads(x)['split']=='validation'];by_parent=defaultdict(list)
    for x in records:by_parent[x['parent_id']].append(x)
    baseline=RangeExitBaseline();total=Counter();errors=[];count_correct=0;worlds=[];elevation=np.arange(-15,16,2,dtype=np.float64)
    for parent,items in sorted(by_parent.items()):
        g=zarr.open_group(str(root/f'artifacts/dataset/validation/{parent}.zarr'),mode='r');local=Counter();local_errors=[];local_correct=0
        for item in items:
            row=int(item['zarr_row']);pred=baseline.predict(g['range_m'][row],g['valid_mask'][row],elevation);match=match_headings(pred['headings_robot_deg'],item['headings_robot_deg'],20.0)
            local.update({k:match[k] for k in ('matched','predicted','truth')});local_errors.extend(match['angular_errors_deg']);local_correct+=int(pred['branch_count']==item['branch_count'])
        precision=local['matched']/max(local['predicted'],1);recall=local['matched']/max(local['truth'],1);f1=2*precision*recall/max(precision+recall,1e-12)
        worlds.append({'parent_id':parent,'frames':len(items),'precision':precision,'recall':recall,'f1':f1,'mean_matched_angular_error_deg':float(np.mean(local_errors)) if local_errors else None,'branch_count_accuracy':local_correct/len(items)})
        total.update(local);errors.extend(local_errors);count_correct+=local_correct
    precision=total['matched']/max(total['predicted'],1);recall=total['matched']/max(total['truth'],1);f1=2*precision*recall/max(precision+recall,1e-12)
    result={'schema_version':'phase3_b0_validation_baseline_audit_v1','status':'PASS_READ_ONLY_B0_VALIDATION_AUDIT','dataset':str(root.relative_to(PROJECT_ROOT)),'split':'validation','worlds':len(worlds),'frames':len(records),'configuration':baseline.config.to_dict(),'matching_tolerance_deg':20.0,'matched':total['matched'],'predicted':total['predicted'],'truth':total['truth'],'precision':precision,'recall':recall,'f1':f1,'mean_matched_angular_error_deg':float(np.mean(errors)) if errors else None,'median_matched_angular_error_deg':float(np.median(errors)) if errors else None,'branch_count_accuracy':count_correct/len(records),'world_metrics':worlds,'duration_seconds':time.monotonic()-started,'training_samples_consumed':0,'models':0}
    write_json(a.output.resolve(),result);print(json.dumps(result,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
