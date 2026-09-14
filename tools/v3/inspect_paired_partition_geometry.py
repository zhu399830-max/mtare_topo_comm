"""Read-only information-reduction diagnostic, not a method ranking metric."""
import _bootstrap
import hashlib
import io
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RUN='results/gate3_semantics/gate3_20260908_gse_spg_paired_extraction_v1_seed0'
SEAL='66e056fa4f235694d9fb7b39f155a6a8696324514febf47ca263acf584a0cadd'


def describe(p,method):
    x=p['points_xyz_m'].astype(float)
    g=p[method+'_point_to_group'];centers=p[method+'_centers_m']
    normals=p[method+'_normals'];valid=p[method+'_normal_valid'][g]
    residual=np.abs(np.einsum('ij,ij->i',x-centers[g],normals[g]))[valid]
    counts=p[method+'_point_count']
    extent=np.linalg.norm(p[method+'_bounds_max_m']-p[method+'_bounds_min_m'],axis=1)
    rough=p[method+'_roughness_m']
    known=p[method+'_normal_valid']
    # The stored roughness is the smallest covariance eigenvalue's square
    # root. Independently recomputing point-to-plane RMS checks this meaning.
    rms=float(np.sqrt(np.mean(residual**2))) if len(residual) else None
    reconstructed=float(np.sqrt(np.sum(counts[known]*rough[known]**2)/counts[known].sum())) if known.any() else None
    if rms is not None and not np.isclose(rms,reconstructed,rtol=1e-7,atol=1e-9):
        raise ValueError('stored roughness does not match raw residual')
    largest=int(np.argmax(counts))
    return dict(groups=len(counts),returns=len(x),known_normal_returns=int(valid.sum()),
        unknown_normal_returns=int((~valid).sum()),plane_rms_m=rms,
        plane_abs_p95_m=float(np.quantile(residual,.95)) if len(residual) else None,
        maximum_observed_bbox_diagonal_m=float(extent.max()),
        largest_group=dict(index=largest,returns=int(counts[largest]),bbox_diagonal_m=float(extent[largest]),
            roughness_m=float(rough[largest]),center_m=centers[largest].tolist()),
        squared_residual_sum=float(np.sum(residual**2)))


def main():
    seal=(ROOT/RUN/'artifacts/evidence_sha256.txt').read_bytes()
    if hashlib.sha256(seal).hexdigest()!=SEAL:raise ValueError('seal drift')
    pins={path:h for h,path in (s.split('  ',1) for s in seal.decode().splitlines())}
    def read(path):
        raw=(ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=pins[path]:raise ValueError('evidence drift')
        return raw
    rows=[json.loads(line) for line in read(RUN+'/logs/observations.jsonl').decode().splitlines()]
    results=[]
    for row in rows:
        case=row['case_id']
        with np.load(io.BytesIO(read(RUN+'/artifacts/partitions/'+case+'.npz')),allow_pickle=False) as p:
            results.append(dict(case_id=case,**{m:describe(p,m) for m in ('r1','r2')}))
    if len(results)!=45:raise ValueError('exact45 required')
    summary={}
    for m in ('r1','r2'):
        n=sum(r[m]['known_normal_returns'] for r in results)
        summary[m]=dict(groups=sum(r[m]['groups'] for r in results),known_normal_returns=n,
            unknown_normal_returns=sum(r[m]['unknown_normal_returns'] for r in results),
            point_weighted_plane_rms_m=float(np.sqrt(sum(r[m]['squared_residual_sum'] for r in results)/n)),
            maximum_observed_bbox_diagonal_m=max(r[m]['maximum_observed_bbox_diagonal_m'] for r in results))
    print(json.dumps(dict(scope='Same sealed45; conditional plane-reduction diagnostic, NOT geometry MAE against ground truth or ranking',
        source_seal_sha256=SEAL,summary=summary,cases=results),allow_nan=False))


if __name__=='__main__':main()
