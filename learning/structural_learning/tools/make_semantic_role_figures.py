"""Create publication figures from the frozen semantic-role run (no test inference)."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load(p): return json.loads(p.read_text())
def savefig(fig,root,name):
 for ext in ('png','pdf'): fig.savefig(root/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;out=r/'previews';out.mkdir(parents=True,exist_ok=True)
 s=load(r/'summary.json');camp=load(r/'campus_test/campus.json');ind=load(r/'indoor_test/indoor.json');dep=load(r/'dependency_audit/audit.json');err=load(r/'error_propagation_audit.json');logs=load(r/'semantic_bottleneck_training.json')
 plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.25,'figure.dpi':130})
 fig,ax=plt.subplots(figsize=(5.2,3.2));e=[x['epoch'] for x in logs];ax.plot(e,[x.get('forest_explicit_topology_spearman',np.nan) for x in logs],label='Topology');ax.plot(e,[x.get('forest_coverage_spearman_audit',np.nan) for x in logs],label='Coverage');ax.axhline(.25,color='k',ls='--',lw=.8,label='PASS threshold');ax.set(xlabel='Epoch',ylabel='Forest Spearman',title='Semantic bottleneck validation trajectory');ax.legend(frameon=False);savefig(fig,out,'fig1_training_correlations')
 worlds=['Forest','Campus','Indoor'];direct=[s['forest']['direct']['topology_spearman'],camp['direct_role']['topology_spearman'],ind['direct_role']['topology_spearman']];sem=[s['forest']['semantic']['topology_spearman'],camp['semantic_bottleneck']['topology_spearman'],ind['semantic_bottleneck']['topology_spearman']];cov=[s['forest']['semantic']['coverage_spearman'],camp['semantic_bottleneck']['coverage_spearman'],ind['semantic_bottleneck']['coverage_spearman']];x=np.arange(3);fig,ax=plt.subplots(figsize=(5.5,3.3));ax.bar(x-.25,direct,.25,label='Direct: topology');ax.bar(x,sem,.25,label='Semantic: topology');ax.bar(x+.25,cov,.25,label='Semantic: coverage');ax.set_xticks(x,worlds);ax.set_ylabel('Spearman');ax.set_title('Topology signal versus coverage nuisance');ax.legend(frameon=False,fontsize=8);savefig(fig,out,'fig2_topology_vs_coverage')
 ks=['Top-1','Top-5','Top-10'];ret=s['forest']['semantic_pairs']['retrieval'];dr=s['forest']['direct_pairs']['retrieval'];fig,ax=plt.subplots(figsize=(5.2,3.2));x=np.arange(3);ax.bar(x-.25,[dr['top1'],dr['top5'],dr['top10']],.25,label='Direct');ax.bar(x,[ret['top1'],ret['top5'],ret['top10']],.25,label='Semantic');ax.bar(x+.25,[ret['random_top1'],ret['random_top5'],ret['random_top10']],.25,label='Random');ax.set_xticks(x,ks);ax.set_ylabel('Recall');ax.set_title('Forest cross-position retrieval');ax.legend(frameon=False);savefig(fig,out,'fig3_forest_retrieval')
 models=['Direct','Semantic'];pos=[s['forest']['direct_pairs']['positive_distance_mean'],s['forest']['semantic_pairs']['positive_distance_mean']];neg=[s['forest']['direct_pairs']['hard_negative_distance_mean'],s['forest']['semantic_pairs']['hard_negative_distance_mean']];fig,ax=plt.subplots(figsize=(4.8,3.2));x=np.arange(2);ax.bar(x-.18,pos,.36,label='Topology-same / coverage-different');ax.bar(x+.18,neg,.36,label='Topology-different / coverage-same');ax.set_xticks(x,models);ax.set_ylabel('Role cosine distance');ax.set_title('Forest counterexample pairs');ax.legend(frameon=False,fontsize=7);savefig(fig,out,'fig4_pair_separation')
 names=['normal','shuffle','zero','teacher'];vals=[dep[k]['topology_spearman'] if dep[k]['topology_spearman'] is not None else 0 for k in names];fig,ax=plt.subplots(figsize=(4.8,3.2));ax.bar(names,vals);ax.set_ylabel('Topology Spearman');ax.set_title('Explicit-semantic dependency audit');savefig(fig,out,'fig5_dependency_audit')
 names=list(err);vals=[err[k]['improvement_over_student'] for k in names];fig,ax=plt.subplots(figsize=(5.5,3.2));ax.bar(names,vals);ax.set_ylabel('Role cosine-error improvement');ax.set_title('Single-field teacher replacement');ax.tick_params(axis='x',rotation=30);savefig(fig,out,'fig6_error_propagation')
 rows=[]
 for w,d in [('forest',s['forest']),('campus',camp),('indoor',ind)]:
  for model,key in [('direct','direct' if w=='forest' else 'direct_role'),('semantic','semantic' if w=='forest' else 'semantic_bottleneck')]:
   m=d[key];pairs=d[(model+'_pairs') if w!='forest' else (model+'_pairs')];rows.append([w,model,m['topology_spearman'],m['coverage_spearman'],pairs['ranking_success'],pairs['retrieval']['top1'],pairs['retrieval']['top5'],pairs['retrieval']['top10']])
 with (out/'paper_metrics.csv').open('w',newline='') as f:
  q=csv.writer(f);q.writerow(['split','model','topology_spearman','coverage_spearman','pair_ranking_success','top1','top5','top10']);q.writerows(rows)
 (r/'campus_test/high_confidence_subset_status.json').write_text(json.dumps({'available':False,'reason':'v4 samples contain no teacher-confidence field; frozen test predictions were not persisted, so no post-hoc second inference was performed.'},indent=2))
 (r/'indoor_test/high_confidence_subset_status.json').write_text(json.dumps({'available':False,'reason':'v4 samples contain no teacher-confidence field; frozen test predictions were not persisted, so no post-hoc second inference was performed.'},indent=2))
 print(json.dumps({'figures':6,'table':str(out/'paper_metrics.csv')},indent=2))
if __name__=='__main__':main()
