"""Probe isolated historical teacher imports; no dataset or label execution.

Deliberately does not import _bootstrap or current mtare_topo. The historical
ZIP is the first package path, so transitive teacher dependencies stay frozen.
"""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
ZIP=ROOT/'results/gate3_semantics/gate3_20260908_gse_supplement_joint_v3_seed20260906/artifacts/source_snapshot.zip'
SHA='c4c0d42d844bfd6e514949efe7ba9f70952727ff6afa59752e72749e6a51fe97'


def load_teacher():
    if any(k=='mtare_topo' or k.startswith('mtare_topo.') for k in sys.modules):
        raise RuntimeError('fresh process required; never mix live and frozen packages')
    if hashlib.sha256(ZIP.read_bytes()).hexdigest()!=SHA:raise ValueError('historical source archive drift')
    sys.path.insert(0,str(ZIP)+'/src')
    raw=importlib.import_module('mtare_topo.teacher.gse_junction_interface_diagnostic_v1')
    target=importlib.import_module('mtare_topo.teacher.gse_joint_reference_targets_v6')
    loaded={}
    with zipfile.ZipFile(ZIP) as archive:
        for name,module in sorted(sys.modules.items()):
            if name=='mtare_topo' or name.startswith('mtare_topo.'):
                path=str(getattr(module,'__file__',''))
                if not path.startswith(str(ZIP)+'/src/'):raise RuntimeError('nonhistorical project dependency: '+name)
                member=path[len(str(ZIP))+1:]
                loaded[name]=hashlib.sha256(archive.read(member)).hexdigest()
    return raw.diagnose_observation,target.produce_joint_reference_targets,loaded


if __name__=='__main__':
    _,_,loaded=load_teacher()
    print(json.dumps(dict(status='HISTORICAL_TEACHER_IMPORT_ISOLATED',archive_sha256=SHA,
        modules=loaded,data_reads=0,teacher_calls=0,model_calls=0)))
