"""Portable, non-mutating copy of a completed cache and its exact dependencies."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha
import json,zipfile

RUN='results/gate3_semantics/gate3_20260912_gse_direction_task_model_cache_v1_seed0'
OUTPUT='docs/exports/gse_direction_task_portable_20260912.zip'

def main():
    run=ROOT/RUN
    assert json.loads((run/'RUN_STATE.json').read_text())['state']=='COMPLETED'
    seal=run/'artifacts/evidence_sha256.txt'
    checked=0
    for line in seal.read_text().splitlines():
        h,p=line.split('  ',1);assert sha(ROOT/p)==h,p;checked+=1
    spec=json.loads((run/'config/run_spec.json').read_text())
    files=set(spec['input_sha256'])
    files.update(str(p.relative_to(ROOT)) for p in run.rglob('*') if p.is_file())
    source=ROOT/'results/gate3_semantics/gate3_20260912_gse_direction_task_pilot_inputs_v1_seed20260912'
    files.update(str(p.relative_to(ROOT)) for p in source.rglob('*') if p.is_file())
    figures=ROOT/'docs/figures/gse_conditional_geometry_fit_v1/direction_task_pilot'
    files.update(str(p.relative_to(ROOT)) for p in figures.rglob('*') if p.is_file())
    files.add('docs/GSE_CONDITIONAL_GEOMETRY_FIT_RESULT.md')
    files.add('tools/v3/package_direction_task_cache.py')
    manifest={p:sha(ROOT/p) for p in sorted(files)}
    output=ROOT/OUTPUT;output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_STORED) as archive:
        for p in sorted(files):archive.write(ROOT/p,p)
        archive.writestr('PORTABLE_MANIFEST.json',json.dumps(manifest,indent=2))
        archive.writestr('READ_ME.txt',
            'This is a local portable copy, NOT an off-machine backup.\n'
            'Contains all12 student windows, source evidence (teacher-only), encoder/checkpoints, common features, and36ABC outputs.\n'
            'Full source is artifacts/source_snapshot.zip in the model cache run. Restore paths relative to a new project root.\n'
            'NPZ arrays use allow_pickle=False and can be analyzed on CPU with NumPy.\n'
            'Training task was conditional geometry, NOT correspondence. No accuracy or exploration benefit established.\n'
            'Teacher source evidence must never enter model forward. Fit-only3parents, not unseen testing.\n'
            'For inference reproduction use the saved environment and checkpoint hashes; the old absolute command path requires relocation.\n')
    with zipfile.ZipFile(output) as archive:
        import hashlib
        for p,h in manifest.items():
            assert hashlib.sha256(archive.read(p)).hexdigest()==h,p
    print(json.dumps(dict(output=OUTPUT,bytes=output.stat().st_size,sha256=sha(output),files=len(manifest),source_seal_entries_verified=checked,archive_entries_verified=len(manifest))))

if __name__=='__main__':main()
