"""Load the exact sealed V8 implementation without current package mixing."""
import hashlib
import importlib
import json
import sys
import zipfile
from v8_multiview_manifest_v1 import ROOT,compile_manifest


def load_teacher():
    if any(n=='mtare_topo' or n.startswith('mtare_topo.') for n in sys.modules):
        raise RuntimeError('fresh process required for historical V8')
    manifest=compile_manifest()
    archive=ROOT/manifest['archive_path']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=manifest['archive_sha256']:
        raise ValueError('V8 archive drift')
    sys.path.insert(0,str(archive)+'/src')
    raw=importlib.import_module('mtare_topo.teacher.gse_junction_interface_diagnostic_v1')
    target=importlib.import_module('mtare_topo.teacher.gse_joint_reference_targets_v8')
    loaded={}
    with zipfile.ZipFile(archive) as z:
        for name,module in sorted(sys.modules.items()):
            if name=='mtare_topo' or name.startswith('mtare_topo.'):
                path=str(getattr(module,'__file__',''))
                if not path.startswith(str(archive)+'/src/'):raise RuntimeError('mixed source: '+name)
                loaded[name]=hashlib.sha256(z.read(path[len(str(archive))+1:])).hexdigest()
    def produce(bundle,raw):
        return target.produce_joint_reference_targets(bundle,raw,qualify_cap_precision=True,
                                                       **manifest['geometry_settings'])
    return raw.diagnose_observation,produce,loaded,manifest


if __name__=='__main__':
    _,_,loaded,manifest=load_teacher()
    print(json.dumps(dict(status='SEALED_V8_IMPORT_PASS_NO_TEACHER_CALL',modules=loaded,
        archive_sha256=manifest['archive_sha256'],geometry_settings=manifest['geometry_settings'],
        counts=manifest['counts'],teacher_calls=0,model_calls=0)))
