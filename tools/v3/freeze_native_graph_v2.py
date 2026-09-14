"""Metadata-only one-time freeze for the fixed native candidate graph export."""
from _bootstrap import PROJECT_ROOT as ROOT
import os
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from native_graph_population_v2 import write
from mtare_topo.data.development_sensor_scope import compile_scope
from mtare_topo.governance_native_graph_v2 import SCHEMA, SLUG, validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.representation.gse_spg_runtime import sha


def freeze():
    card_path='configs/v3/gate3/data_cards/'+SLUG+'.json'
    spec_path='configs/v3/gate3/'+SLUG+'.json'
    runtime_path='configs/v3/environments/'+SLUG+'.json'
    if any((ROOT/p).exists() for p in (card_path,spec_path,runtime_path)):
        raise FileExistsError('one freeze only')
    scope=compile_scope(ROOT)
    approval=dict(status='APPROVED',approved_by='user-standing-scope-authorization',
        approved_at='2026-09-09',authorized_operations=['data_export'],authorized_gates=[3],
        scope_sha256=digest(scope),scope='307 C01-C07 selected observations; original full50m sensor input; candidate geometry only.',
        confirmation_reference='Standing autonomous development authorization and current PLAN native307 export; no labels, training, protected worlds, or semantic benchmark.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='data_export',scope=scope,
        scope_sha256=digest(scope),approval=approval)
    report=validate_card(card)
    if not report.passed: raise ValueError(report.errors)
    exe='build/gse_voxblox_cpu_v1/cmake/observation_graph'
    lib='build/gse_voxblox_cpu_v1/prefix/usr/lib/x86_64-linux-gnu'
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/lib))
    linked=subprocess.check_output(['ldd',str(ROOT/exe)],env=env,text=True)
    if 'not found' in linked: raise RuntimeError('unresolved native library')
    paths={Path(sys.executable).resolve(),ROOT/exe}
    paths.update(Path(p).resolve() for p in re.findall(r'(/[\w./+\-]+)',linked))
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and fields[-1].startswith('/') and Path(fields[-1]).is_file():
            paths.add(Path(fields[-1]).resolve())
    for p in Path(np.__file__).parent.rglob('*'):
        if p.is_file() and (p.suffix in ('.py','.so') or '.so.' in p.name): paths.add(p.resolve())
    runtime=dict(schema_version='native_candidate_runtime_v1',python=sys.version,numpy=np.__version__,
        python_executable=str(Path(sys.executable).resolve()),native_executable=exe,native_library_path=lib,
        files={str(p):sha(p) for p in sorted(paths)},ldd=linked,
        scope='Executable, resolved native dependencies, loaded Python libraries and NumPy sources; algorithm defaults separately source-pinned.',
        native_limits=dict(address_space_bytes=4*1024**3,cpu_s=120,wall_s=150,file_bytes=16*1024**2),
        parent_address_space_bytes=8*1024**3)
    write(ROOT/card_path,card,'x');write(ROOT/runtime_path,runtime,'x')
    sources={str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3')
             for p in (ROOT/folder).rglob('*.py')}
    for folder in ('tools/v3/native_voxblox','third_party/voxblox_reference/voxblox',
                   'third_party/mav_voxblox_planning_reference/voxblox_skeleton',
                   'third_party/minkindr_reference/minkindr/include'):
        sources.update(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob('*')
                       if p.is_file() and p.suffix in ('.h','.hpp','.cpp','.cc','.proto','.txt'))
    sources.update((card_path,runtime_path,'docs/GSE_NATIVE_OBSERVATION_RUNNER_CONTRACT_DRAFT.md'))
    run='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed0'
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','PYTHONHASHSEED=0',
        str(Path(sys.executable)), 'tools/v3/native_graph_population_v2.py',
        '--spec',str(ROOT/spec_path),'--run-dir',str(ROOT/run)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=0,
        operation='data_export',data_card=card_path,config_path=card_path,user_authorization=approval,
        command=command,question='What candidate geometry does the pinned native algorithm recover from the same307 full sensor observations?',
        method='Official SimpleTSDF to full-euclidean ESDF to native sparse skeleton; each fiveframe observation independent; all raw edges retained.',
        baseline='Nonlearning candidate geometric reference; not a semantic-junction detection score or ground-robot safety reference.',
        fallback='Stop and seal failure; no retry, pruning, sampling substitution, or parameter tuning.',
        wall_time_cap_s=43200,estimated_cost=dict(compute='CPU only;307 sequential native invocations',host_ram_gb=12,gpu_vram_gb=0,disk_gb=12,wall_time_hours=12),
        acceptance_criteria=['307 exact selected source bindings and raw graph outputs; no cross-observation edges.',
            'No hallucinated ESDF or missing TSDF support; finite native multigraph and consistent edge audit; self loops retained and counted, not travel evidence.',
            'Preserve low-clearance and unknown edge flags; completion is not semantic or safety PASS.'],
        expected_evidence=['Source snapshot,exact card,environment,command,per-observation native JSON and stderr,source hashes,summary,RUN_STATE,SHA256 seal.'],
        source_sha256={p:sha(ROOT/p) for p in sorted(sources)},
        environment_manifest=runtime_path,environment_sha256=sha(ROOT/runtime_path))
    write(ROOT/spec_path,spec,'x')
    print(spec_path,scope['counts'],flush=True)


if __name__=='__main__': freeze()

