"""Explicit metadata-only inventory; no scans, weights or Zarr chunks."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'results/gate3_semantics'
SOURCES={
 'endpoint':'gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0',
 'selection':'gate3_20260907_gse_surface_identity_selection_v1_seed20260906',
 'terminal':'gate3_20260907_gse_population_terminal_v2_seed20260906',
 'partial':'gate3_20260908_gse_supplement_joint_v3_seed20260906'}

def main():
    data={};hashes={}
    for key,run in SOURCES.items():
        path=ROOT/run/'metrics/summary.json';raw=path.read_bytes()
        data[key]=json.loads(raw);hashes[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    split=data['endpoint']['split']
    for value in split.values():
        assert value['positive_attachment_pairs']==value['supported_positive_attachment_pairs']+value['hidden_positive_attachment_pairs']
    rows=data['partial']['observations'];assert len(rows)==210
    assert all('_C0'+str(i) not in r['task'] for i in (8,9) for r in rows)
    assert all('_C10' not in r['task'] for r in rows)
    by_split={}
    for name in ('C01_C06','C07'):
        selected=[r for r in rows if ('_C07' in r['task'])==(name=='C07')]
        by_split[name]={key:sum(r[key] for r in selected) for key in
            ('observations','anchors','openings','positive_memberships')}
    print(json.dumps(dict(status='EXISTING_TARGET_METADATA_INVENTORY',source_sha256=hashes,
        endpoint_proxy=split,partial_structure_counts=by_split,
        selected_population=data['selection']['result'],
        original_terminal_observations=sum(r['observations'] for r in data['terminal']['observations']),
        original_observed_terminal_occurrences=sum(r['observed_terminals'] for r in data['terminal']['observations']),
        partial_run_declared_labels=data['partial']['labels'],new_training_qualified=False,
        note='counts are occurrences not independent places; endpoint proxy is not surface membership',
        payload_reads=0,optimizer_steps=0),ensure_ascii=False))

if __name__=='__main__':main()
