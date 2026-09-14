"""Read-only original same45 reference checks, no labels exported or changed."""
import _bootstrap
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from mtare_topo.data.gse_synthetic_fit_scope import declared_cases
from mtare_topo.evaluation.gse_synthetic_field_scoring import expected_geometry
from mtare_topo.governance_surface_selection import digest
from mtare_topo.representation.gse_block_structure_readout_v2 import smooth_anchor_coordinates

def main():
    root=Path(__file__).resolve().parents[2]
    run=root/'results/gate3_semantics/gate3_20260908_gse_block_representation_fit_v1_seed0'
    seal=(run/'artifacts/evidence_sha256.txt').read_bytes()
    assert hashlib.sha256(seal).hexdigest()=='9993ec4cbbc08521a099ceba8b81fdfafeef7b91477b2f5fd9a0d3bb383bdfdf'
    pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def read(rel):
        p=run/rel;b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==pins[str(p.relative_to(root))]
        return json.loads(b)
    spec=read('config/run_spec.json');population=read('config/population.json')
    for rel in ('src/mtare_topo/data/gse_synthetic_fit_scope.py','src/mtare_topo/evaluation/gse_synthetic_field_scoring.py'):
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==spec['source_sha256'][rel]
    cases={c['case_id']:c for c in declared_cases()};positions=[];openings=0
    assert len(cases)==len(population)==45
    for row in population:
        ref=expected_geometry(cases[row['case_id']]);assert digest(ref)==row['reference_sha256']
        positions.extend(ref['anchors']);openings+=len(ref['openings'])
    xyz=torch.tensor(positions,dtype=torch.float64);r=torch.linalg.vector_norm(xyz,dim=1)
    assert len(xyz)==27 and openings==84
    if (r>=10).any():raise ValueError('exact boundary anchor not finitely representable; stop')
    raw=xyz/torch.sqrt(100-(xyz*xyz).sum(1,keepdim=True))
    error=torch.linalg.vector_norm(smooth_anchor_coordinates(raw)-xyz,dim=1)
    assert error.max()<1e-10
    print(json.dumps(dict(observations=45,anchors=27,openings=84,anchor_radius_min_m=float(r.min()),anchor_radius_max_m=float(r.max()),inverse_raw_radius_max=float(torch.linalg.vector_norm(raw,dim=1).max()),inverse_reconstruction_error_max_m=float(error.max()),optimizer_steps=0,new_labels=0)))

if __name__=='__main__':main()
