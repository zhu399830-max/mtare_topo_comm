"""Clean underground trajectory cross-validation for the semantic role model.

This does not alter or copy teacher data.  Each fold trains from scratch so
the held-out underground trajectories have never influenced the weights.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from learning.structural_learning.tools.run_semantic_bottleneck_role import Data,Semantic,train,predict,metrics,pair_retrieval,seed,save

def trajectory_subset(root:Path, token:str):
 ds=Data(root,'train'); keep=[]
 for p in ds.files:
  with np.load(p,allow_pickle=False) as z:
   if token in str(z['trajectory']): keep.append(p)
 ds.files=keep;return ds

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=1);p.add_argument('--seed',type=int,default=20260808);p.add_argument('--fold',type=int,choices=(1,2));p.add_argument('--resume',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');folds=[]
 specs=[('traj01','traj02'),('traj02','traj01')]
 for idx,(tr_token,va_token) in enumerate(specs,1):
  if a.fold is not None and idx!=a.fold: continue
  seed(a.seed+idx);tr=trajectory_subset(a.dataset_root,tr_token);va=trajectory_subset(a.dataset_root,va_token);fo=a.out/f'fold{idx}_{tr_token}_to_{va_token}';fo.mkdir(parents=True,exist_ok=True)
  save(fo/'split.json',{'train_token':tr_token,'validation_token':va_token,'train_samples':len(tr),'validation_samples':len(va),'worlds_in_both':['tunnel','garage'],'clean_from_scratch':True})
  model=Semantic().to(d)
  if a.resume and (fo/'semantic_bottleneck_checkpoint.pt').exists(): model.load_state_dict(torch.load(fo/'semantic_bottleneck_checkpoint.pt',map_location=d,weights_only=False)['state'])
  train(model,tr,va,d,a.epochs,'semantic_bottleneck',fo,a.resume);x=predict(model,va,d);row={'fold':idx,'train_token':tr_token,'validation_token':va_token,'checkpoint':torch.load(fo/'semantic_bottleneck_checkpoint.pt',map_location='cpu',weights_only=False)['epoch'],'metrics':metrics(x),'pairs':pair_retrieval(x)};save(fo/'validation.json',row);folds.append(row);print(json.dumps({'finished_fold':idx,'metrics':row['metrics']}),flush=True)
 keys=['topology_spearman','coverage_spearman','direction_mae','exit_mae','role_cosine'];summary={'protocol':'two-fold underground trajectory holdout; both tunnel and garage appear in train and validation but trajectories are disjoint','epochs_max':a.epochs,'folds':folds,'mean':{k:float(np.mean([r['metrics'][k] for r in folds])) for k in keys}}
 if a.fold is None: save(a.out/'summary.json',summary)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
