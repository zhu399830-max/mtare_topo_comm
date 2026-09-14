"""Deterministic storage only: reproduce the existing JSON bytes exactly."""
import gzip
import hashlib
import json
from pathlib import Path


def encode_evidence(value):
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode('utf8')
    packed=gzip.compress(raw,compresslevel=6,mtime=0)
    if gzip.decompress(packed)!=raw:
        raise ValueError('lossless evidence roundtrip mismatch')
    return packed,dict(codec='gzip6_mtime0_json_original_spacing',raw_bytes=len(raw),
        raw_sha256=hashlib.sha256(raw).hexdigest(),compressed_bytes=len(packed),
        compressed_sha256=hashlib.sha256(packed).hexdigest())


def write_evidence(path,value):
    path=Path(path)
    packed,metadata=encode_evidence(value)
    with path.open('xb') as stream:
        stream.write(packed)
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=metadata['compressed_sha256']:
            raise ValueError('stored evidence SHA mismatch')
    return metadata
