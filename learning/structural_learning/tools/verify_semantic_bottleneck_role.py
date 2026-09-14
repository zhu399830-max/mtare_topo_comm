"""Recompute the frozen forest claims and audit the semantic-only role boundary."""
from __future__ import annotations
import argparse,json,sys,torch,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from learning.structural_learning.tools.run_semantic_bottleneck_role import Data,Semantic,predict,metrics,pair_retrieval,canonical_roles_numpy
from sklearn.metrics import balanced_accuracy_score,average_precision_score

def binary_report(y,p):
 y=(np.asarray(y).reshape(-1)>=.5).astype(int);q=(np.asarray(p).reshape(-1)>=.5).astype(int);tp=int(((y==1)&(q==1)).sum());fp=int(((y==0)&(q==1)).sum());fn=int(((y==1)&(q==0)).sum())
 return {'f1':2*tp/max(2*tp+fp+fn,1),'balanced_accuracy':float(balanced_accuracy_score(y,q)),'pr_auc':float(average_precision_score(y,np.asarray(p).reshape(-1)))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();d=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 m=Semantic().to(d);ck=torch.load(a.run/'semantic_bottleneck_checkpoint.pt',map_location=d,weights_only=False);m.load_state_dict(ck['state']);ds=Data(a.dataset_root,'val');x=predict(m,ds,d);mm=metrics(x);pairs=pair_retrieval(x)
 oracle=canonical_roles_numpy(x['y_direction'],x['y_distance'],x['y_exit']);yn=np.linalg.norm(x['y_role'],axis=1);on=np.linalg.norm(oracle,axis=1);valid=(yn>1e-8)|(on>1e-8);cos=np.sum(oracle[valid]*x['y_role'][valid],1)/np.maximum(on[valid]*yn[valid],1e-8)
 result={'checkpoint_epoch':ck['epoch'],'forest_count':len(ds),'fixed_splits':{'train':['tunnel','garage'],'val':['forest'],'test':['campus','indoor']},'forest_metrics':mm,'forest_direction_classification':binary_report(x['y_direction'],x['p_direction']),'forest_exit_classification':binary_report(x['y_exit'],x['p_exit']),'forest_pairs':pairs,'teacher_reconstruction':{'nonzero_union_count':int(valid.sum()),'mean_cosine_error':float(np.mean(1-cos)),'both_zero_count':int((~valid).sum())},'architecture':{'role_module_parameter_count':sum(q.numel() for q in m.role_module.parameters()),'role_module_input':'explicit semantic dictionary only','encoder_reference_in_role_module':hasattr(m.role_module,'enc')},'checks':{}}
 result['checks']={'teacher_exact':result['teacher_reconstruction']['mean_cosine_error']<1e-6,'topology_above_coverage':mm['topology_spearman']>mm['coverage_spearman'],'retrieval_above_random':pairs['retrieval']['top10']>pairs['retrieval']['random_top10'],'pair_order_correct':pairs['ranking_success']>.5,'no_trainable_role_or_bypass':result['architecture']['role_module_parameter_count']==0 and not result['architecture']['encoder_reference_in_role_module']}
 result['all_checks_pass']=all(result['checks'].values());q=a.run/'verification.json';q.write_text(json.dumps(result,indent=2,sort_keys=True));print(json.dumps(result,indent=2));raise SystemExit(0 if result['all_checks_pass'] else 2)
if __name__=='__main__':main()
