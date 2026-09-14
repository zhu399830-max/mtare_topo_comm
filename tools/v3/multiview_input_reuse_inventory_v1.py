"""Save exact metadata-only input reuse/missing plan, without payload IO."""
import hashlib
import json
from pathlib import Path
from _bootstrap import PROJECT_ROOT as ROOT
from mtare_topo.data.multiview_input_reuse_v1 import compile_scope


if __name__=='__main__':
    path=ROOT/'configs/v3/gate3/multiview_input_reuse_inventory_v1.json'
    if path.exists():raise FileExistsError('no refreeze')
    scope=compile_scope(ROOT)
    scope['compiler_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),ROOT/'src/mtare_topo/data/multiview_input_reuse_v1.py')}
    with path.open('x') as stream:json.dump(scope,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(counts=scope['counts'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),file=str(path.relative_to(ROOT)))))
