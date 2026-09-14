"""Read-only exact scope compiler, no data export, inference or optimizer."""
import _bootstrap
import json
from pathlib import Path
from mtare_topo.data.gse_synthetic_fit_scope import compile_scope
from mtare_topo.data.gse_structure_review_v1 import canonical_sha

if __name__=='__main__':
    scope=compile_scope(Path(__file__).resolve().parents[2])
    print(json.dumps(dict(scope_sha256=canonical_sha(scope),scope=scope)),flush=True)
