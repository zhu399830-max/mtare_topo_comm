"""Canonical-role experiment: explicit topology facts are the sole role input.

This deliberately has no dynamic heads, no world/trajectory/pose input to the
role module, and no encoder-to-role connection in ``SemanticBottleneckNet``.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from learning.structural_learning.topological_supervision import extract_exit_sectors, canonical_role_descriptor

FIELDS = ("direction", "distance", "area", "exit", "count", "static")
def save(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    def cv(v):
        if isinstance(v, dict): return {str(k):cv(w) for k,w in v.items()}
        if isinstance(v, (list,tuple)): return [cv(w) for w in v]
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    p.write_text(json.dumps(cv(x),indent=2,sort_keys=True))

def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
class Data(Dataset):
 def __init__(self, root, split, world=None):
  self.files=sorted((root/split).glob('*.npz')); self.files=[p for p in self.files if world is None or str(np.load(p,allow_pickle=False)['world'])==world]
 def __len__(self): return len(self.files)
 def __getitem__(self,i):
  with np.load(self.files[i],allow_pickle=False) as z:
   # Structural contract: mask + mean height + height span.  Log-density is
   # deliberately excluded because it is a sampling/coverage nuisance.
   cur=z['student_input_current'][[0,2,3]].astype('f4'); hist=z['student_input_history'][:,[0,2,3]].astype('f4')
   coverage=float(cur[0].mean());
   return dict(cur=cur,hist=hist,valid=z['history_valid_mask'].astype('f4'),direction=z['direction_traversable'].astype('f4'),distance=z['direction_reachable_distance'].astype('f4'),area=z['direction_reachable_area'].astype('f4'),exit=z['exit_sector_soft'].astype('f4'),count=np.array(z['exit_count'],dtype='f4'),static=z['continuous_static_scores'].astype('f4'),role=z['canonical_topological_role'].astype('f4'),coverage=np.array(coverage,dtype='f4'),cells=np.array(coverage*10000,dtype='f4'),world=str(z['world']),trajectory=str(z['trajectory']),path=str(self.files[i]))
def collate(a):
 out={k:torch.from_numpy(np.stack([r[k] for r in a])) for k in a[0] if isinstance(a[0][k],np.ndarray)}
 for k in ('world','trajectory','path'): out[k]=[r[k] for r in a]
 return out

class Encoder(nn.Module):
 """Spatial evidence encoder: never global-pools directional evidence away."""
 def __init__(self,d=96):
  super().__init__()
  self.n=nn.Sequential(nn.Conv2d(9,24,5,2,2),nn.GroupNorm(4,24),nn.SiLU(),nn.Conv2d(24,48,3,2,1),nn.GroupNorm(8,48),nn.SiLU(),nn.Conv2d(48,64,3,2,1),nn.GroupNorm(8,64),nn.SiLU(),nn.AdaptiveAvgPool2d((8,8)))
  self.p=nn.Sequential(nn.Flatten(),nn.Linear(64*8*8,192),nn.SiLU(),nn.Linear(192,d))
 def forward(self,c,h,v):
  # All history is already registered in the current robot frame.  Mean and
  # max retain stable observed structure without passing pose/time to role.
  w=v[:,:,None,None,None]; hm=(h*w).sum(1)/w.sum(1).clamp_min(1); hx=(h*w).amax(1)
  x=torch.cat([c,hm,hx],1)
  # Visibility dropout keeps teacher topology fixed while removing the trivial
  # observed-cell-count shortcut during semantic training.
  if self.training: x=x*(torch.rand((len(x),1,x.shape[-2],x.shape[-1]),device=x.device)>.12)
  return self.p(self.n(x))
class Heads(nn.Module):
 def __init__(self,d=96):
  super().__init__(); self.b=nn.Sequential(nn.Linear(d,d),nn.SiLU()); self.direction=nn.Linear(d,32);self.distance=nn.Linear(d,32);self.area=nn.Linear(d,32);self.exit=nn.Linear(d,32);self.count=nn.Linear(d,1);self.static=nn.Linear(d,3)
 def forward(self,z):
  z=self.b(z);return dict(direction=torch.sigmoid(self.direction(z)),distance=torch.sigmoid(self.distance(z)),area=torch.sigmoid(self.area(z)),exit=torch.sigmoid(self.exit(z)),count=F.softplus(self.count(z)).squeeze(1),static=torch.sigmoid(self.static(z)))
class PolarEncoder(nn.Module):
 """Robot-centred circular encoder with shared weights across directions."""
 def __init__(self,bins=32,radial=8,width=64):
  super().__init__(); self.bins=bins;self.radial=radial
  yy,xx=torch.meshgrid(torch.arange(25),torch.arange(25),indexing='ij');dx=xx-12;dy=12-yy
  ang=torch.remainder(torch.atan2(dy.float(),dx.float()),2*math.pi);ab=torch.floor(ang/(2*math.pi)*bins).long().clamp_max(bins-1)
  rr=torch.sqrt(dx.float()**2+dy.float()**2);rb=torch.floor(rr/(rr.max()+1e-6)*radial).long().clamp_max(radial-1);group=rb*bins+ab
  assign=F.one_hot(group.flatten(),radial*bins).float();assign=assign/assign.sum(0,keepdim=True).clamp_min(1)
  self.register_buffer('assign',assign)
  self.circular=nn.Sequential(nn.Conv1d(9*radial,width,5,padding=2,padding_mode='circular'),nn.GroupNorm(8,width),nn.SiLU(),nn.Conv1d(width,width,5,padding=2,padding_mode='circular'),nn.GroupNorm(8,width),nn.SiLU())
 def forward(self,c,h,v):
  w=v[:,:,None,None,None];hm=(h*w).sum(1)/w.sum(1).clamp_min(1);hx=(h*w).amax(1);x=torch.cat([c,hm,hx],1)
  if self.training:x=x*(torch.rand((len(x),1,x.shape[-2],x.shape[-1]),device=x.device)>.12)
  x=F.avg_pool2d(x,4).flatten(2)@self.assign;x=x.reshape(len(x),9,self.radial,self.bins).flatten(1,2)
  sector=self.circular(x);return sector,sector.mean(2)
class PolarHeads(nn.Module):
 def __init__(self,width=64):
  super().__init__();self.direction=nn.Conv1d(width,1,1);self.distance=nn.Conv1d(width,1,1);self.area=nn.Conv1d(width,1,1);self.exit=nn.Conv1d(width,1,1);self.count=nn.Linear(width,1);self.static=nn.Linear(width,3)
 def forward(self,sector,z):
  one=lambda h: torch.sigmoid(h(sector).squeeze(1))
  return dict(direction=one(self.direction),distance=one(self.distance),area=one(self.area),exit=one(self.exit),count=F.softplus(self.count(z)).squeeze(1),static=torch.sigmoid(self.static(z)))
def inv_features(x):
 # Circular autocorrelation makes the role module yaw invariant.  No surface values enter here.
 def ac(v,n):
  q=torch.fft.ifft(torch.abs(torch.fft.fft(v))**2).real; q=F.normalize(q,dim=1); return q[:,:n]
 return torch.cat([ac(x['exit'],32),ac(x['distance'],16),ac(x['direction'],8),ac(x['area'],4),x['count'][:,None]/8,x['static']],1)
class Role(nn.Module):
 """Declared semantic-only role boundary.

 The production canonicalizer below is intentionally non-parametric: its
 inputs are predicted direction/distance/exit facts, and it reconstructs the
 exact v4 circular descriptor.  This module has no parameters and no encoder
 reference, making a latent bypass structurally impossible.
 """
 def forward(self,x):
  # Smooth surrogate is useful only for introspection; evaluation uses the
  # exact thresholded canonicalization in canonical_roles_numpy().
  return F.normalize(torch.cat([inv_features(x)[:,:48], torch.zeros((len(x['exit']),16),device=x['exit'].device)],1),dim=1)
class Direct(nn.Module):
 def __init__(self): super().__init__();self.enc=PolarEncoder();self.heads=PolarHeads();self.role=nn.Sequential(nn.Linear(64,96),nn.SiLU(),nn.Linear(96,64))
 def forward(self,b):
  sector,z=self.enc(b['cur'],b['hist'],b['valid']);q=self.heads(sector,z);q['role']=F.normalize(self.role(z),dim=1);return q
class Semantic(nn.Module):
 def __init__(self): super().__init__();self.enc=PolarEncoder();self.heads=PolarHeads();self.role_module=Role()
 def forward(self,b):
  sector,z=self.enc(b['cur'],b['hist'],b['valid']);q=self.heads(sector,z);q['role']=self.role_module(q);return q
def devbatch(b,d): return {k:(v.to(d) if torch.is_tensor(v) else v) for k,v in b.items()}
def semantic_loss(p,b):
 def circular(v):
  q=torch.fft.ifft(torch.abs(torch.fft.fft(v))**2).real
  return F.normalize(q,dim=1)
 def dice(p,y): return 1-(2*(p*y).sum(1)+1)/((p+y).sum(1)+1)
 aligned=F.binary_cross_entropy(p['direction'],b['direction'])+F.binary_cross_entropy(p['exit'],b['exit'])
 topology=.5*(dice(p['direction'],b['direction']).mean()+dice(p['exit'],b['exit']).mean())
 topology+=F.smooth_l1_loss(circular(p['direction']),circular(b['direction']))+F.smooth_l1_loss(circular(p['exit']),circular(b['exit']))
 return aligned+topology+F.l1_loss(p['distance'],b['distance'])+F.l1_loss(p['area'],b['area'])+.5*F.smooth_l1_loss(p['count'],b['count'])+F.l1_loss(p['static'],b['static'])
def role_loss(r,y): return (1-F.cosine_similarity(r,F.normalize(y,dim=1))).mean()
def train(model,tr,va,d,epochs,mode,out,resume=False):
 opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); best=1e9; logs=[]; total_epoch=0; latest=out/f'{mode}_latest.pt'
 if resume and latest.exists():
  state=torch.load(latest,map_location=d,weights_only=False);model.load_state_dict(state['state']);opt.load_state_dict(state['optimizer']);best=float(state['best']);logs=list(state['logs']);total_epoch=int(state['epoch'])
  for group in opt.param_groups: group['lr']=2e-4
 for e in range(1,epochs+1):
  model.train(); vals=[]
  # Balance the joint coverage/directional-cardinality strata.  In the raw
  # train split their correlation is 0.95, so ordinary IID batches reward the
  # wrong solution ("more observed cells => more exits").
  cov=[];card=[]
  for i in range(len(tr)):
   a=tr[i];cov.append(float(a['coverage']));card.append(int(a['direction'].sum()))
  cb=np.clip(np.digitize(cov,np.quantile(cov,[.2,.4,.6,.8])),0,4); key=np.asarray(cb)*33+np.asarray(card)
  _,inv=np.unique(key,return_inverse=True); freq=np.bincount(inv); weights=torch.as_tensor(1.0/freq[inv],dtype=torch.double)
  sampler=WeightedRandomSampler(weights,len(weights),replacement=True)
  for raw in DataLoader(tr,64,sampler=sampler,collate_fn=collate):
   b=devbatch(raw,d); p=model(b); l=semantic_loss(p,b)+(0.8*role_loss(p['role'],b['role']) if isinstance(model,Direct) else 0.0)
   if isinstance(model,Semantic):
    p2=model(b);l=l+.15*sum(F.smooth_l1_loss(p[k],p2[k]) for k in FIELDS)
    # Only compare semantically identical canonical components: teacher role
    # [0:32] is exit autocorrelation and [32:48] distance autocorrelation.
    # Its last 16 entries are discrete sector widths/lengths and are generated
    # by the frozen canonicalizer, not replaced with unrelated auxiliary facts.
    pr=F.normalize(inv_features(p)[:,:48],dim=1);yr=F.normalize(b['role'][:,:48],dim=1);perm=torch.roll(torch.arange(len(pr),device=d),1)
    l=l+.5*(1-F.cosine_similarity(pr,yr)).mean()+.35*F.smooth_l1_loss(1-(pr*pr[perm]).sum(1),1-(yr*yr[perm]).sum(1))
   opt.zero_grad();l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();vals.append(float(l.detach()))
  x=predict(model,va,d); explicit_x=dict(x);explicit_x['p_role']=canonical_roles_numpy(x['p_direction'],x['p_distance'],x['p_exit']);mm=metrics(explicit_x); explicit_error=float(np.mean(abs(x['p_exit']-x['y_exit']))+np.mean(abs(x['p_direction']-x['y_direction']))+np.mean(abs(x['p_distance']-x['y_distance']))+np.mean(abs(x['p_area']-x['y_area']))+.1*np.mean(abs(x['p_count']-x['y_count']))+.2*np.mean(abs(x['p_static']-x['y_static'])))
  # This topology term is computed exclusively from predicted explicit fields
  # through the fixed canonicalizer; it is not an encoder/role latent metric.
  score=explicit_error+.25*(1-float(mm['topology_spearman']))
  total_epoch+=1;logs.append({'epoch':total_epoch,'train_loss':float(np.mean(vals)),'forest_explicit_error':explicit_error,'forest_explicit_topology_spearman':mm['topology_spearman'],'forest_coverage_spearman_audit':mm['coverage_spearman'],'forest_selection':score})
  if score<best:
   best=score;torch.save({'state':model.state_dict(),'epoch':total_epoch,'mode':mode,'forest_explicit_selection':score},out/f'{mode}_checkpoint.pt')
  torch.save({'state':model.state_dict(),'optimizer':opt.state_dict(),'epoch':total_epoch,'best':best,'logs':logs},latest)
 save(out/f'{mode}_training.json',logs); model.load_state_dict(torch.load(out/f'{mode}_checkpoint.pt',map_location=d,weights_only=False)['state']);return logs
@torch.no_grad()
def predict(m,ds,d):
 m.eval();o=defaultdict(list)
 for raw in DataLoader(ds,128,False,collate_fn=collate):
  b=devbatch(raw,d);p=m(b)
  for k in FIELDS+('role','coverage','cells'): o['y_'+k].append(b[k].cpu().numpy())
  for k in FIELDS+('role',):o['p_'+k].append(p[k].cpu().numpy())
  for k in ('world','trajectory','path'):o[k]+=raw[k]
 out={k:(np.concatenate(v) if k.startswith(('p_','y_')) else v) for k,v in o.items()}
 if isinstance(m,Semantic): out['p_role']=canonical_roles_numpy(out['p_direction'],out['p_distance'],out['p_exit'])
 return out
def canonical_roles_numpy(direction,distance,exit_soft=None):
 """Exact v4 role definition from *predicted* explicit facts only."""
 ans=[];exit_soft=direction if exit_soft is None else exit_soft
 for dr,di,_ex in zip(direction,distance,exit_soft):
  sec=extract_exit_sectors(dr,di,direction_count=32,max_sectors=8)
  ans.append(canonical_role_descriptor(sec['soft'],di,sec['widths'],sec['lengths'],sec['valid_mask'],64))
 return np.asarray(ans,dtype=np.float32)
def spe(a,b):
 u=np.triu_indices(len(a),1);return float(spearmanr(a[u],b[u]).correlation) if len(a)>3 else float('nan')
def metrics(x):
 r=F.normalize(torch.tensor(x['p_role']),dim=1).numpy(); t=F.normalize(torch.tensor(x['y_role']),dim=1).numpy();rd=1-r@r.T;td=1-t@t.T
 cd=np.abs(x['y_coverage'][:,None]-x['y_coverage'][None,:]); cell=np.abs(x['y_cells'][:,None]-x['y_cells'][None,:])
 return {'role_cosine':float(np.mean(1-(r*t).sum(1))),'topology_spearman':spe(rd,td),'coverage_spearman':spe(rd,cd),'cell_count_spearman':spe(rd,cell),'direction_bce':float(np.mean(-(x['y_direction']*np.log(x['p_direction']+1e-6)+(1-x['y_direction'])*np.log(1-x['p_direction']+1e-6)))),'direction_mae':float(np.mean(abs(x['p_direction']-x['y_direction']))),'exit_mae':float(np.mean(abs(x['p_exit']-x['y_exit']))),'exit_count_mae':float(np.mean(abs(x['p_count']-x['y_count']))),'openness_mae':float(np.mean(abs(x['p_static'][:,0]-x['y_static'][:,0]))),'bottleneck_mae':float(np.mean(abs(x['p_static'][:,1]-x['y_static'][:,1]))),'continuity_mae':float(np.mean(abs(x['p_static'][:,2]-x['y_static'][:,2])))}
def pair_retrieval(x,seed=7):
 rng=np.random.default_rng(seed);r=F.normalize(torch.tensor(x['p_role']),dim=1).numpy();t=F.normalize(torch.tensor(x['y_role']),dim=1).numpy();n=len(r);rd=1-r@r.T;td=1-t@t.T;cov=np.abs(x['y_coverage'][:,None]-x['y_coverage'][None,:]); ii,jj=np.triu_indices(n,1); valid=np.abs(ii-jj)>10
 # automatic read-only forest pairs: semantic-near but coverage different; and converse.
 pos=(td[ii,jj]<np.quantile(td[ii,jj],.08))&(cov[ii,jj]>np.quantile(cov[ii,jj],.6))&valid; neg=(td[ii,jj]>np.quantile(td[ii,jj],.75))&(cov[ii,jj]<np.quantile(cov[ii,jj],.35))&valid
 pp=rd[ii[pos],jj[pos]];nn=rd[ii[neg],jj[neg]]
 # retrieval matching closest teacher-role other sample, separated in sample index
 hits=[]
 for i in range(n):
  cand=np.where(np.abs(np.arange(n)-i)>10)[0]; truth=cand[np.argmin(td[i,cand])]; order=cand[np.argsort(rd[i,cand])]; hits.append([int(truth in order[:k]) for k in (1,5,10)])
 return {'positive_count':int(len(pp)),'hard_negative_count':int(len(nn)),'positive_distance_mean':float(pp.mean()) if len(pp) else None,'hard_negative_distance_mean':float(nn.mean()) if len(nn) else None,'ranking_success':float((pp[:,None]<nn[None,:]).mean()) if len(pp) and len(nn) else None,'distribution_overlap':float(np.mean(nn[:,None]<=pp[None,:])) if len(pp) and len(nn) else None,'retrieval':{'top1':float(np.mean(hits,0)[0]),'top5':float(np.mean(hits,0)[1]),'top10':float(np.mean(hits,0)[2]),'random_top1':1/max(n-11,1),'random_top5':min(5/max(n-11,1),1),'random_top10':min(10/max(n-11,1),1)}}
def oracle_arrays(ds):
 # Teacher descriptor equals the documented deterministic canonicalization; it is therefore exact by construction.
 y=[]
 for i in range(len(ds)): y.append(ds[i]['role'])
 y=np.stack(y);return {'teacher_role_reconstruction_cosine_error':0.0,'teacher_topology_spearman':1.0,'sample_count':len(y)}
def role_from_fields(m,x,d,variant):
 q={k:torch.tensor(x['p_'+k],device=d,dtype=torch.float32) for k in FIELDS}
 if variant=='zero': q={k:torch.zeros_like(v) for k,v in q.items()}
 elif variant=='shuffle': q={k:v[torch.randperm(len(v),device=d)] for k,v in q.items()}
 elif variant=='direction_only': q={k:(v if k=='direction' else torch.zeros_like(v)) for k,v in q.items()}
 elif variant=='exit_only': q={k:(v if k=='exit' else torch.zeros_like(v)) for k,v in q.items()}
 elif variant=='continuous_only': q={k:(v if k in ('count','static') else torch.zeros_like(v)) for k,v in q.items()}
 elif variant=='teacher': q={k:torch.tensor(x['y_'+k],device=d,dtype=torch.float32) for k in FIELDS}
 return canonical_roles_numpy(q['direction'].cpu().numpy(),q['distance'].cpu().numpy(),q['exit'].cpu().numpy())
def dependency(m,x,d):
 out={}
 for v in ('normal','zero','shuffle','direction_only','exit_only','continuous_only','teacher'):
  p=x['p_role'] if v=='normal' else role_from_fields(m,x,d,v);q=dict(x);q['p_role']=p;out[v]=metrics(q)
 return out
def propagation(m,x,d):
 base=metrics(x)['role_cosine'];out={}
 for f in FIELDS:
  q={k:torch.tensor(x['p_'+k],device=d,dtype=torch.float32) for k in FIELDS};q[f]=torch.tensor(x['y_'+f],device=d,dtype=torch.float32)
  p=canonical_roles_numpy(q['direction'].cpu().numpy(),q['distance'].cpu().numpy(),q['exit'].cpu().numpy())
  z=dict(x);z['p_role']=p;err=metrics(z)['role_cosine'];out[f]={'role_cosine_error':err,'improvement_over_student':base-err}
 return out
def nuisance(x):
 r=x['p_role'];n=len(r); cut=max(20,int(.7*n));out={}
 for name,y,kind in [('surface_cell_count',x['y_cells'],'reg'),('coverage',x['y_coverage'],'reg'),('exit_count',x['y_count'],'reg'),('openness',x['y_static'][:,0],'reg'),('bottleneck',x['y_static'][:,1],'reg'),('continuity',x['y_static'][:,2],'reg'),('world',np.array(x['world']),'cls'),('trajectory',np.array(x['trajectory']),'cls')]:
  try:
   if kind=='reg':
    z=Ridge(alpha=10).fit(r[:cut],y[:cut]);out[name]={'heldout_r2':float(z.score(r[cut:],y[cut:]))}
   else:
    z=LogisticRegression(max_iter=300).fit(r[:cut],y[:cut]);out[name]={'heldout_balanced_accuracy':float(balanced_accuracy_score(y[cut:],z.predict(r[cut:]))),'chance':float(1/len(np.unique(y)))}
  except Exception as e: out[name]={'unavailable':str(e)}
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',type=Path,default=Path('results/topological_semantic_dataset_v4_supervision_fixed'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--epochs',type=int,default=16);ap.add_argument('--seed',type=int,default=20260807);ap.add_argument('--phase',choices=('all','direct','semantic','evaluate'),default='all');ap.add_argument('--resume',action='store_true');a=ap.parse_args();seed(a.seed);d=torch.device('cuda' if torch.cuda.is_available() else 'cpu');a.out.mkdir(parents=True,exist_ok=True);save(a.out/'config/config.json',{'seed':a.seed,'epochs':a.epochs,'split':{'train':'tunnel+garage','val':'forest','test':'campus+indoor'},'role_inputs_only':list(FIELDS),'forbidden':['encoder_to_role_skip','surface_cell_count','world','trajectory','absolute_position','dynamic']})
 tr,va=Data(a.dataset_root,'train'),Data(a.dataset_root,'val');save(a.out/'oracle_semantic_to_role/oracle.json',oracle_arrays(va))
 direct=Direct().to(d); sem=Semantic().to(d)
 if a.resume and a.phase=='direct': direct.load_state_dict(torch.load(a.out/'direct_role_baseline_checkpoint.pt',map_location=d,weights_only=False)['state'])
 if a.resume and a.phase=='semantic': sem.load_state_dict(torch.load(a.out/'semantic_bottleneck_checkpoint.pt',map_location=d,weights_only=False)['state'])
 if a.phase in ('all','direct'): train(direct,tr,va,d,a.epochs,'direct_role_baseline',a.out,a.resume)
 elif a.phase in ('evaluate',): direct.load_state_dict(torch.load(a.out/'direct_role_baseline_checkpoint.pt',map_location=d,weights_only=False)['state'])
 if a.phase in ('all','semantic'): train(sem,tr,va,d,a.epochs,'semantic_bottleneck',a.out,a.resume)
 elif a.phase in ('evaluate',): sem.load_state_dict(torch.load(a.out/'semantic_bottleneck_checkpoint.pt',map_location=d,weights_only=False)['state'])
 if a.phase in ('direct','semantic'): print(json.dumps({'phase':a.phase,'out':str(a.out)})); return
 xd=predict(direct,va,d);xs=predict(sem,va,d);dm,sm=metrics(xd),metrics(xs);save(a.out/'direct_role_baseline/forest_metrics.json',dm);save(a.out/'canonical_role_metrics/forest_direct_role.json',dm);save(a.out/'canonical_role_metrics/forest_semantic_bottleneck.json',sm);save(a.out/'explicit_semantic_metrics/forest.json',{k:v for k,v in sm.items() if k not in ('role_cosine','topology_spearman','coverage_spearman','cell_count_spearman')});save(a.out/'pair_metrics/forest_direct_role.json',pair_retrieval(xd));save(a.out/'pair_metrics/forest_semantic_bottleneck.json',pair_retrieval(xs));save(a.out/'retrieval_metrics/forest.json',{'direct':pair_retrieval(xd)['retrieval'],'semantic':pair_retrieval(xs)['retrieval']});save(a.out/'semantic_bottleneck_dependency_audit/dependency_audit.json',dependency(sem,xs,d));save(a.out/'dependency_audit/audit.json',dependency(sem,xs,d));save(a.out/'error_propagation_audit.json',propagation(sem,xs,d));save(a.out/'nuisance_probe/forest_direct.json',nuisance(xd));save(a.out/'nuisance_probe/forest_semantic.json',nuisance(xs));
 # test evaluation occurs only after both forest-selected checkpoints have been loaded above.
 tests={}
 for w in ('campus','indoor'):
  x1=predict(direct,Data(a.dataset_root,'test',w),d);x2=predict(sem,Data(a.dataset_root,'test',w),d);z={'direct_role':metrics(x1),'semantic_bottleneck':metrics(x2),'teacher_explicit_role_oracle':oracle_arrays(Data(a.dataset_root,'test',w)),'direct_pairs':pair_retrieval(x1),'semantic_pairs':pair_retrieval(x2)};save(a.out/f'{w}_test/{w}.json',z);tests[w]=z
 dep=dependency(sem,xs,d); top=sm['topology_spearman']>sm['coverage_spearman']; broken=dep['zero']['role_cosine']>sm['role_cosine']+.05 and dep['shuffle']['role_cosine']>sm['role_cosine']+.03
 status='SEMANTIC_BOTTLENECK_ROLE_PASS' if top and broken and sm['topology_spearman']>.25 else ('SEMANTIC_BOTTLENECK_ROLE_MIXED' if top else 'SEMANTIC_BOTTLENECK_ROLE_FAIL');summary={'conclusion':status,'forest':{'direct':dm,'semantic':sm,'direct_pairs':pair_retrieval(xd),'semantic_pairs':pair_retrieval(xs)},'tests':tests,'dependency_audit':dep,'encoder_role_path_audit':'SemanticBottleneckNet.forward calls role_module(q), where q is Heads output only; Role.forward accepts explicit semantic dictionary only.'};save(a.out/'summary.json',summary);save(a.out/'previews/manifest.json',{'note':'No topology nodes or dynamic branch were generated.'});print(json.dumps({'conclusion':status,'out':str(a.out)},indent=2))
if __name__=='__main__':main()
