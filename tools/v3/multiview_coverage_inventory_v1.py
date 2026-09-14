"""Produce metadata-only candidate coverage; never reads scan or target arrays."""
import hashlib
import json
from pathlib import Path
from _bootstrap import PROJECT_ROOT as ROOT
from mtare_topo.data.multiview_coverage_inventory_v1 import compile_inventory


if __name__=='__main__':
    out=ROOT/'configs/v3/gate3/multiview_coverage_inventory_v1.json'
    if out.exists():raise FileExistsError('no overwrite or refreeze')
    result=compile_inventory(ROOT)
    for p,h in result['source_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('source changed')
    result['compiler_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),ROOT/'src/mtare_topo/data/multiview_coverage_inventory_v1.py')}
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(file=str(out.relative_to(ROOT)),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),counts=result['counts']),ensure_ascii=False))
