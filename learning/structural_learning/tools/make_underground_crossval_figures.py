"""Aggregate underground cross-validation and create beginner-friendly figures."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def read(p): return json.loads(p.read_text())
def emit(fig,out,name):
 for ext in ('png','pdf'): fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();folds=sorted(a.run.glob('fold*'));vals=[read(x/'validation.json') for x in folds];out=a.run/'figures';out.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25})
 fig,axs=plt.subplots(1,2,figsize=(8.5,3.1),sharey=True)
 for ax,fd in zip(axs,folds):
  h=read(fd/'semantic_bottleneck_training.json');ax.plot([x['epoch'] for x in h],[x['forest_explicit_topology_spearman'] for x in h],label='Topology');ax.plot([x['epoch'] for x in h],[x['forest_coverage_spearman_audit'] for x in h],label='Coverage');ax.set_title(fd.name.replace('_',' '));ax.set_xlabel('Epoch')
 axs[0].set_ylabel('Held-out underground Spearman');axs[1].legend(frameon=False);emit(fig,out,'underground_training_curves')
 x=np.arange(2);top=[v['metrics']['topology_spearman'] for v in vals];cov=[v['metrics']['coverage_spearman'] for v in vals];fig,ax=plt.subplots(figsize=(4.8,3.2));ax.bar(x-.18,top,.36,label='Topology');ax.bar(x+.18,cov,.36,label='Coverage');ax.set_xticks(x,['Fold 1','Fold 2']);ax.set_ylim(0,1);ax.set_ylabel('Spearman');ax.set_title('Underground trajectory holdout');ax.legend(frameon=False);emit(fig,out,'underground_topology_vs_coverage')
 fig,ax=plt.subplots(figsize=(5.2,3.2));ks=['Top-1','Top-5','Top-10'];x=np.arange(3)
 for i,v in enumerate(vals):
  q=v['pairs']['retrieval'];ax.bar(x+(.22*i-.22),[q['top1'],q['top5'],q['top10']],.22,label=f'Fold {i+1}')
 q=vals[0]['pairs']['retrieval'];ax.bar(x+.22,[q['random_top1'],q['random_top5'],q['random_top10']],.22,label='Random');ax.set_xticks(x,ks);ax.set_ylabel('Recall');ax.set_title('Underground cross-position retrieval');ax.legend(frameon=False);emit(fig,out,'underground_retrieval')
 fig,ax=plt.subplots(figsize=(5.2,3.2));pos=[v['pairs']['positive_distance_mean'] for v in vals];neg=[v['pairs']['hard_negative_distance_mean'] for v in vals];ax.bar(x[:2]-.18,pos,.36,label='Same topology / different coverage');ax.bar(x[:2]+.18,neg,.36,label='Different topology / same coverage');ax.set_xticks(x[:2],['Fold 1','Fold 2']);ax.set_ylabel('Role cosine distance');ax.set_title('Underground counterexample separation');ax.legend(frameon=False,fontsize=7);emit(fig,out,'underground_pair_separation')
 keys=['topology_spearman','coverage_spearman','direction_mae','exit_mae','role_cosine'];summary={'protocol':'two-fold trajectory-disjoint validation inside tunnel+garage','folds':vals,'mean':{k:float(np.mean([v['metrics'][k] for v in vals])) for k in keys},'mean_pair_ranking_success':float(np.mean([v['pairs']['ranking_success'] for v in vals]))}
 (a.run/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True));print(json.dumps(summary['mean'],indent=2))
if __name__=='__main__':main()
