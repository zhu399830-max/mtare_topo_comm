"""Verify frozen underground cross-validation artifacts without retraining."""
from __future__ import annotations
import argparse,json,torch
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=[]
 for fold in sorted(a.run.glob('fold*')):
  split=json.loads((fold/'split.json').read_text());val=json.loads((fold/'validation.json').read_text());latest=torch.load(fold/'semantic_bottleneck_latest.pt',map_location='cpu',weights_only=False);best=torch.load(fold/'semantic_bottleneck_checkpoint.pt',map_location='cpu',weights_only=False)
  rows.append({'fold':fold.name,'train_token':split['train_token'],'validation_token':split['validation_token'],'tokens_disjoint':split['train_token']!=split['validation_token'],'latest_epoch':latest['epoch'],'best_epoch':best['epoch'],'topology_spearman':val['metrics']['topology_spearman'],'coverage_spearman':val['metrics']['coverage_spearman'],'pair_ranking_success':val['pairs']['ranking_success'],'top10':val['pairs']['retrieval']['top10'],'random_top10':val['pairs']['retrieval']['random_top10']})
 checks={'two_folds':len(rows)==2,'trajectory_disjoint':all(x['tokens_disjoint'] for x in rows),'at_least_60_epochs':all(x['latest_epoch']>=60 for x in rows),'topology_above_coverage':all(x['topology_spearman']>x['coverage_spearman'] for x in rows),'pair_order_correct':all(x['pair_ranking_success']>.5 for x in rows),'retrieval_above_random':all(x['top10']>x['random_top10'] for x in rows)};out={'folds':rows,'checks':checks,'all_checks_pass':all(checks.values())};(a.run/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));raise SystemExit(0 if out['all_checks_pass'] else 2)
if __name__=='__main__':main()
