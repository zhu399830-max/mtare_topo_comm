"""Copy only complete, hash-bound reference cases into a new immutable run."""
import hashlib,json
from pathlib import Path

FILES={'source_records.npz','references.json','targets.npz','summary.json'}

def copy_reference_prefix(root, output, cases, input_hashes):
    root=Path(root).resolve();output=Path(output).resolve();rows=[]
    if [c['case'] for c in cases]!=list(range(len(cases))):
        raise ValueError('reuse must be a complete contiguous prefix')
    for case in cases:
        if case.get('eligible') is not True:raise ValueError('ineligible reuse')
        artifacts=case['artifacts'];paths=[Path(p) for p in artifacts]
        if len(paths)!=4 or {p.name for p in paths}!=FILES:raise ValueError('complete four-artifact case required')
        data={}
        for path in paths:
            source=(root/path).resolve()
            if not source.is_relative_to(root) or source.is_relative_to(output):raise ValueError('invalid reuse source')
            raw=source.read_bytes();h=hashlib.sha256(raw).hexdigest()
            if h!=artifacts[str(path)] or h!=input_hashes.get(str(path)):raise ValueError('reuse hash drift')
            data[path.name]=raw
        row=json.loads(data['summary.json'])
        if row['case']!=case['case']:raise ValueError('summary identity drift')
        target=output/'artifacts'/f"case_{case['case']:03d}";target.mkdir()
        for name,raw in data.items():
            with (target/name).open('xb') as f:f.write(raw)
        with (target/'reuse_origin.json').open('x') as f:json.dump(case,f,indent=2)
        rows.append(dict(row,reused=True,reuse_origin=artifacts))
    return rows
