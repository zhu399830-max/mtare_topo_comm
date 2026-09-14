"""One declared scheduling-only override of the sealed lateral entry kernel."""
import argparse
from functools import partial
import hashlib
import importlib
from pathlib import Path
import zipfile
import historical_teacher_worker_v1 as transport
from v8_teacher_snapshot_probe_v1 import load_teacher
from observed_entries_batched_v1 import observed_entries_batched


def load_batched():
    diagnose,produce,modules,manifest=load_teacher()
    archive=Path(__file__).resolve().parents[2]/manifest['archive_path']
    transport.ZIP=archive;transport.SHA=manifest['archive_sha256']
    name='src/mtare_topo/teacher/gse_observed_operand_entries_v1.py'
    with zipfile.ZipFile(archive) as z:
        if hashlib.sha256(z.read(name)).hexdigest()!='a1e768bffee8d5b0e06dff9dca1b4c3ec0fa78799c28151a26b4196a7072f11a':
            raise ValueError('original intersection kernel drift')
    kernel=importlib.import_module('mtare_topo.teacher.gse_observed_operand_entries_v1').observed_operand_entries
    lateral=importlib.import_module('mtare_topo.teacher.gse_lateral_branch_evidence_v1')
    if lateral.observed_operand_entries is not kernel:raise ValueError('unexpected previous override')
    lateral.observed_operand_entries=partial(observed_entries_batched,kernel)
    return diagnose,produce,modules


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--echo',action='store_true');p.add_argument('--memory-bytes',type=int)
    args=p.parse_args();transport.load_teacher=load_batched
    raise SystemExit(transport.main(args.echo,args.memory_bytes))
