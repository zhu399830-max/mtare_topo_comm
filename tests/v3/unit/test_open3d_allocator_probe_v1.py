import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_zarr2187_v1/bin/python'


def test_allocator_correction_under_original_worker_address_cap(tmp_path):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    env.pop('MALLOC_ARENA_MAX',None)
    command=[PYTHON,str(ROOT/'tools/v3/open3d_allocator_probe_v1.py')]
    reference=json.loads(subprocess.check_output(command,env=env,text=True,timeout=20))
    corrected=json.loads(subprocess.check_output(['prlimit','--as=3221225472','--',*command],
        env=dict(env,MALLOC_ARENA_MAX='2'),text=True,timeout=20))
    assert len(reference['rows'])==len(corrected['rows'])==6
    for before,after in zip(reference['rows'],corrected['rows'],strict=True):
        assert before['output_sha256']==after['output_sha256']
        assert after['memory_kib_except_threads']['VmSize']<3*1024**2
    # Evidence remains synthetic; this does not prove the real population fits.
    (tmp_path/'paired_probe.json').write_text(json.dumps(dict(reference=reference,corrected=corrected)))
