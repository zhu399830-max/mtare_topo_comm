"""Restore release evidence into a chosen directory without overwriting files.

Download evidence-*.zip from the private release first. No pickle/model loading.
Only allowlisted regular members are read, hash-checked, then written; duplicate
source files are reconstructed from the same member. Raw scans are not supplied.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',required=True,type=Path)
    parser.add_argument('--destination',required=True,type=Path)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).resolve().parents[2]/'docs/github_evidence_20260914/included.jsonl')
    args=parser.parse_args();root=args.destination.resolve()
    rows=[json.loads(l) for l in args.manifest.read_text().splitlines()]
    groups={}
    for row in rows:
        if row['destination']=='release':groups.setdefault(row['archive'],[]).append(row)
    for archive,entries in groups.items():
        with zipfile.ZipFile(args.assets/archive) as z:
            for row in entries:
                target=(root/row['path']).resolve();target.relative_to(root)
                if target.exists():
                    with target.open('rb') as f:existing=hashlib.file_digest(f,'sha256').hexdigest()
                    if existing!=row['sha256']:raise RuntimeError('Refusing overwrite: '+str(target))
                    continue
                member=row.get('duplicate_of',row['path']);info=z.getinfo(member)
                assert info.file_size==row['bytes']
                h=hashlib.sha256()
                with z.open(member) as f:
                    while block:=f.read(2**20):h.update(block)
                if h.hexdigest()!=row['sha256']:raise RuntimeError('Member hash mismatch '+member)
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(member) as source,target.open('xb') as out:shutil.copyfileobj(source,out)
        print('Restored',archive,flush=True)


if __name__=='__main__':main()
