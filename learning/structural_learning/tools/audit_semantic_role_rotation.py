"""Forest-only rotation audit for a frozen semantic-bottleneck checkpoint."""
from pathlib import Path
import argparse, json, sys, torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from learning.structural_learning.tools.run_semantic_bottleneck_role import Data, collate, Direct, Semantic, devbatch

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
 d=torch.device('cpu'); direct=Direct(); sem=Semantic()
 direct.load_state_dict(torch.load(a.run/'direct_role_baseline_checkpoint.pt',map_location=d,weights_only=False)['state'])
 sem.load_state_dict(torch.load(a.run/'semantic_bottleneck_checkpoint.pt',map_location=d,weights_only=False)['state'])
 direct.eval();sem.eval(); dd=[];sd=[]
 with torch.no_grad():
  for raw in DataLoader(Data(a.dataset_root,'val'),64,False,collate_fn=collate):
   b=devbatch(raw,d); rot=dict(b);rot['cur']=torch.rot90(b['cur'],1,(-2,-1));rot['hist']=torch.rot90(b['hist'],1,(-2,-1))
   dd.extend((1-F.cosine_similarity(direct(b)['role'],direct(rot)['role'])).tolist())
   sd.extend((1-F.cosine_similarity(sem(b)['role'],sem(rot)['role'])).tolist())
 out={'split':'forest','rotation':'90_degree_surface_rotation','teacher_canonical_role_distance':0.0,'direct_role_distance_mean':sum(dd)/len(dd),'semantic_bottleneck_role_distance_mean':sum(sd)/len(sd),'note':'This measures end-to-end rotational consistency; semantic role itself is circular-autocorrelation invariant once its predicted directional fields rotate consistently.'}
 q=a.run/'rotation_audit/forest.json';q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
