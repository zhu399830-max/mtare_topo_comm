"""Inventory and package existing research evidence; no experiments or cleanup.

Git gets figures and small readable records. Releases get larger readable
records and saved PyTorch weights. Raw scans/arrays/datasets are inventoried,
not silently represented as backed up. All original paths and bytes survive.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections
import hashlib
import json
import os
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/github-evidence-20260914"
DOC = ROOT / "docs/github_evidence_20260914"
IMAGES = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.gif'}
TEXT = {'.json', '.jsonl', '.txt', '.log', '.stdout', '.stderr', '.md', '.html',
        '.csv', '.yaml', '.yml', '.sha256', '.py', '.js', '.cpp', '.h', '.xml',
        '.launch', '.world', '.sdf', '.diff', '.sh', '.toml'}
PATTERNS = [(name, re.compile(p)) for name, p in {
    'github_token': rb'gh[pousr]_[A-Za-z0-9]{36,255}',
    'github_fine_grained_token': rb'github_pat_[A-Za-z0-9_]{50,255}',
    'openai_project_token': rb'sk-proj-[A-Za-z0-9_-]{40,}',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'aws_access_key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'credential_in_url': rb'https?://[^\s/@:]+:[^\s/@]{8,}@',
}.items()]


def inspect(row):
    p = ROOT / row['path']; h = hashlib.sha256(); carry = b''; found = set()
    before = p.stat()
    with p.open('rb') as f:
        while data := f.read(2**20):
            h.update(data)
            block = carry + data
            for name, pattern in PATTERNS:
                if pattern.search(block): found.add(name)
            carry = block[-4096:]
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source changed during inspection: ' + row['path'])
    return dict(row, sha256=h.hexdigest(), patterns=sorted(found), mtime_ns=after.st_mtime_ns)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    DOC.mkdir(parents=True, exist_ok=True)
    selected=[]; omitted=[]
    for directory, dirs, files in os.walk(ROOT / 'results'):
        dirs[:] = sorted(d for d in dirs if d not in {'.git', '__pycache__'})
        for name in sorted(files):
            p=Path(directory)/name; rel=p.relative_to(ROOT).as_posix()
            if p.is_symlink():
                omitted.append(dict(path=rel,bytes=0,reason='symlink_not_followed'));continue
            size=p.stat().st_size; suffix=p.suffix.lower()
            if any(part.endswith('.zarr') for part in p.parts):
                kind='raw_dataset'
            elif suffix in IMAGES or suffix in TEXT or suffix=='.pt' or (not suffix and size<2**20):
                kind='git' if ((suffix in IMAGES and size<20*2**20) or (suffix!='.pt' and size<=256*1024)) else 'release'
            else:kind='raw_or_other_binary'
            row=dict(path=rel,bytes=size,destination=kind)
            if kind in {'git','release'}:selected.append(row)
            else:omitted.append(dict(path=rel,bytes=size,reason=kind))
    selected.sort(key=lambda r:r['path'])
    print(json.dumps({'phase':'inventory','selected_files':len(selected),'selected_bytes':sum(r['bytes'] for r in selected),'omitted_files':len(omitted),'omitted_bytes':sum(r['bytes'] for r in omitted)}),flush=True)
    inspected=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(inspect,selected)):
            inspected.append(row)
            if (i+1)%2000==0:print(json.dumps({'phase':'hash_and_secret_scan','completed':i+1,'total':len(selected)}),flush=True)
    findings=[r for r in inspected if r['patterns']]
    if findings:
        (OUT/'sensitive_findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
        raise RuntimeError('Potential credentials: inspect locations before staging; no uploads performed')
    gitrows=[r for r in inspected if r['destination']=='git']; released=[r for r in inspected if r['destination']=='release']
    (OUT/'git-paths.nul').write_bytes(b'\0'.join(r['path'].encode() for r in gitrows)+b'\0')
    archives=[]; duplicates={}; unique=[]
    for row in released:
        key=row['sha256']
        if key in duplicates:row['duplicate_of']=duplicates[key]
        else:duplicates[key]=row['path'];unique.append(row)
    # Independent ZIPs preserve paths; only identical bytes share a restore source.
    batches=[];batch=[];total=0
    for row in unique:
        if batch and total+row['bytes']>512*2**20:
            batches.append(batch);batch=[];total=0
        batch.append(row);total+=row['bytes']
    if batch:batches.append(batch)
    def package(item):
        index,rows=item; name=f'evidence-{index:03d}.zip'; target=OUT/name
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
            for row in rows:
                p=ROOT/row['path'];st=p.stat()
                if st.st_size!=row['bytes'] or st.st_mtime_ns!=row['mtime_ns']:raise RuntimeError('Source drift: '+row['path'])
                z.write(p,arcname=row['path'])
                row['archive']=name
        digest=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()
        if target.stat().st_size>=2*1024**3:raise RuntimeError('Release asset exceeds limit')
        return dict(name=name,bytes=target.stat().st_size,sha256=digest,files=len(rows))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for entry in pool.map(package,enumerate(batches,1)):
            archives.append(entry);print(json.dumps({'phase':'archive','completed':len(archives),'total':len(batches),**entry}),flush=True)
    lookup={r['path']:r['archive'] for r in unique}
    for row in released:
        if 'duplicate_of' in row:row['archive']=lookup[row['duplicate_of']]
    for row in inspected:row.pop('patterns');row.pop('mtime_ns')
    for name,rows in [('included.jsonl',inspected),('not_uploaded.jsonl',omitted)]:
        (DOC/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    summary=dict(state='PACKAGED_NOT_UPLOADED',git_files=len(gitrows),git_bytes=sum(r['bytes'] for r in gitrows),
      release_files=len(released),release_source_bytes=sum(r['bytes'] for r in released),duplicate_files=len(released)-len(unique),
      archives=archives,release_archive_bytes=sum(a['bytes'] for a in archives),
      not_uploaded_files=len(omitted),not_uploaded_bytes=sum(r['bytes'] for r in omitted),
      original_files_deleted=0,new_training_steps=0,new_simulations=0,
      secret_scan='Common-pattern heuristic on selected bytes; not comprehensive security certification',
      release_tag='research-evidence-20260914',private_repository='zhu399830-max/mtare_topo_comm')
    (DOC/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (OUT/'SHA256SUMS.txt').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in archives))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
