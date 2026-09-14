"""Actual native intersections, three batches including a one-ray remainder."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools/v3'))
from observed_entries_batched_v1 import observed_entries_batched
from test_gse_observed_operand_entries_v1 import caster_for_box
from mtare_topo.teacher.gse_observed_operand_entries_v1 import observed_operand_entries


def main():
    caster=caster_for_box()
    ray=np.array([1.5,0.,0.,-2.5,3.,0.],np.float32)
    ray[3:]/=np.linalg.norm(ray[3:])
    rays=np.tile(ray,(2305,1));ranges=np.full(2305,np.linalg.norm([-2.5,3.,0.]),np.float32)
    valid=np.ones(2305,bool);owners=[['north'] for _ in rays]
    # Invalid, occluded and multisource rows at both sides of batch boundaries.
    valid[1151]=False;ranges[1152]=.01;owners[2303]=['north','other']
    args=dict(first_return=ranges,valid=valid,return_sources=owners,center_m=np.array([1.5,0.,0.]))
    old=observed_operand_entries(caster,rays,**args)
    new=observed_entries_batched(observed_operand_entries,caster,rays,**args)
    assert old==new
    ids=[e['ray_index'] for e in new['entries']]
    assert len(ids)==2302 and ids[-1]==2304
    assert not {1151,1152,2303}&set(ids)
    print('PASS full_vs_batched_exact_entries 2305 rays 2302 witnesses; boundary masks and global IDs preserved')


if __name__=='__main__':main()
