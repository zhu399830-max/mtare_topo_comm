"""Preserve exact still-matching source pins before later development edits."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    spec_path=(ROOT/a.spec).resolve();out=(ROOT/a.output).resolve()
    if ROOT not in spec_path.parents or ROOT not in out.parents:raise ValueError('project-local paths required')
    spec=json.loads(spec_path.read_text());pins=spec['source_sha256']
    for name,digest in pins.items():
        path=(ROOT/name).resolve()
        if ROOT not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('cannot archive drifted source: '+name)
    out.mkdir(parents=True,exist_ok=False)
    archive=out/'frozen_sources.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for name in sorted(set(pins)|{str(spec_path.relative_to(ROOT))}):
            t.add(ROOT/name,arcname=name,recursive=False)
    result=dict(spec=str(spec_path.relative_to(ROOT)),source_files=len(pins),
        image=spec.get('image'),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        experiment_rerun=False)
    with (out/'archive_manifest.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
