"""Upload the frozen evidence ZIP list and verify server SHA-256 digests.

Release must already exist in the intended private repository. Re-running only
skips assets whose server digest AND size match; it never overwrites assets.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import subprocess
import threading

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'docs/github_evidence_20260914'
OUT=ROOT/'build/github-evidence-20260914'
GH='/home/zeng-workstation/.local/bin/gh'
REPO='zhu399830-max/mtare_topo_comm'
TAG='research-evidence-20260914'


def gh(*args):
    return subprocess.check_output([GH,*args],text=True)


def assets(release_id):
    pages=json.loads(gh('api','--paginate','--slurp',f'repos/{REPO}/releases/{release_id}/assets?per_page=100'))
    return {a['name']:a for page in pages for a in page}


def main():
    assert json.loads(gh('repo','view',REPO,'--json','visibility'))['visibility']=='PRIVATE'
    pages=json.loads(gh('api','--paginate','--slurp',f'repos/{REPO}/releases?per_page=100'))
    matches=[r for page in pages for r in page if r['tag_name']==TAG]
    assert len(matches)==1, 'Expected exactly one draft or published release'
    release=matches[0]
    summary=json.loads((DOC/'summary.json').read_text())
    rows=summary['archives'][:]
    for p in [OUT/'SHA256SUMS.txt',DOC/'included.jsonl',DOC/'not_uploaded.jsonl']:
        rows.append(dict(name=p.name,bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),path=str(p)))
    existing=assets(release['id']); completed=[];lock=threading.Lock()
    def upload(row):
        name=row['name'];a=existing.get(name)
        if a is None:
            p=Path(row.get('path',str(OUT/name)))
            if hashlib.file_digest(p.open('rb'),'sha256').hexdigest()!=row['sha256']:raise RuntimeError('Local asset drift '+name)
            subprocess.run([GH,'release','upload',TAG,str(p),'--repo',REPO],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        elif a['size']!=row['bytes'] or a.get('digest')!='sha256:'+row['sha256']:
            raise RuntimeError('Existing asset differs or has no checksum; will not overwrite '+name)
        with lock:
            completed.append(name)
            print(json.dumps({'uploaded_or_preverified':len(completed),'total':len(rows),'asset':name,'asset_bytes':row['bytes']}),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(upload,row) for row in rows]):future.result()
    remote=assets(release['id']);verified=[]
    for row in rows:
        a=remote[row['name']]
        assert a['size']==row['bytes'],row['name']
        assert a.get('digest')=='sha256:'+row['sha256'],('Missing or different server digest',row['name'])
        verified.append(dict(name=row['name'],bytes=row['bytes'],sha256=row['sha256'],id=a['id'],url=a['browser_download_url']))
    receipt=dict(state='REMOTE_RELEASE_ASSETS_SIZE_AND_SHA256_VERIFIED',repository=REPO,visibility='PRIVATE',tag=TAG,
                 release_url=release['html_url'],assets=verified,asset_count=len(verified),bytes=sum(r['bytes'] for r in verified))
    (DOC/'upload_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'state':receipt['state'],'assets':len(verified),'bytes':receipt['bytes']}),flush=True)


if __name__=='__main__':main()
