"""Restore exact learned-pair sources and input copies; never execute a run."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import prepare_geometry_reproduction as restore


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',required=True)
    parser.add_argument('--common-domain',action='store_true');a=parser.parse_args()
    destination=Path(a.destination).absolute()
    restore.SPEC='configs/v3/gate6/gse_learned_geometry_pair_v1.json'
    restore.ARCHIVE=ROOT/'docs/reproduction/gse_learned_geometry_pair_v1'
    if a.common_domain:
        restore.SPEC='configs/v3/gate6/gse_common_geometry_domain_v1.json'
        restore.ARCHIVE=ROOT/'docs/reproduction/gse_common_geometry_domain_v1'
        restore.TOOLS=['preflight.py','create_run.py']  # bootstrap already pinned
    spec=json.loads((ROOT/restore.SPEC).read_text())
    # Fail before constructing a workspace if any existing input has drifted.
    for name,h in spec['input_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('input drift: '+name)
    restore.prepare(destination)
    copied={}
    for name,h in spec['input_sha256'].items():
        target=destination/name
        if target.exists():raise ValueError('input/source collision: '+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)  # independent copy, no mutable hard links
        if sha(target)!=h:raise ValueError('copied input drift: '+name)
        copied[name]=h
    with (destination/'INPUT_RESTORE_REPORT.json').open('x') as out:
        json.dump(dict(input_sha256=copied,experiment_executed=False,
            prerequisites='Exact local Docker image and absolute Python sidecar path from run spec; not bundled'),out,indent=2)
    print(json.dumps(dict(inputs_verified=len(copied),workspace=str(destination),experiment_executed=False)))


if __name__=='__main__':main()
