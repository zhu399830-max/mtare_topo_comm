import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test_gse_surface_review_v1 import bundle
from mtare_topo.data.gse_surface_review_v1 import import_browser_review


def test_real_node_export_import_float_bytes_and_tampering():
    b=bundle();b['points_xyz_m']=[[1.0,2.0,3.0]]
    raw=json.dumps(b).encode();bh=hashlib.sha256(raw).hexdigest()
    ref=json.dumps(dict(bundle_file_sha256=bh,construction_reference={})).encode()
    root=Path(__file__).resolve().parents[3]
    script="""
const fs=require('fs');const {SurfaceBrowserReview,emptyTarget}=require('./tools/v3/review/gse_surface_review.js');
const x=JSON.parse(fs.readFileSync(0,'utf8'));const s=new SurfaceBrowserReview(x.bundle,x.bh,'synthetic_interop');
s.commit(emptyTarget(x.bundle));s.reveal(x.reference,x.rh);s.finish('synthetic only');process.stdout.write(JSON.stringify(s.export()));
"""
    record=json.loads(subprocess.check_output(['node','-e',script],cwd=root,input=json.dumps(dict(
        bundle=b,bh=bh,reference=json.loads(ref),rh=hashlib.sha256(ref).hexdigest())).encode()))
    imported=import_browser_review(raw,ref,record)
    assert imported['review_protocol_complete']
    assert not imported['automatic_training_eligibility']
    with pytest.raises(ValueError,match='hash'):import_browser_review(raw+b' ',ref,record)
    with pytest.raises(ValueError,match='duplicate'):import_browser_review(b'{"x":1,"x":2}',ref,record)
