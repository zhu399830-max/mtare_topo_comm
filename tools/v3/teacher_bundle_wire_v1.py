"""Bounded NumPy/JSON wire format; no project imports, pickle or file IO."""
import hashlib
import io
import json
import struct
import zipfile
import numpy as np

MAX_PACKET=64*1024**2
ARRAYS={
 'student/ranges_m':((5,16,720),'float32'), 'student/valid_mask':((5,16,720),'uint8'),
 'student/relative_translation_current_sensor_m':((5,3),'float32'),
 'student/relative_yaw_current_sensor_deg':((5,),'float32'),
 'student/frame_rows':((5,),'int32'), 'student/source_sequence_ids':((),'int64'),
 'sensor_teacher_only/sensor_xyz_m':((5,3),'float64'),
 'sensor_teacher_only/yaw_deg':((5,),'float64'),
 'sensor_teacher_only/primitive_membership_code':((5,16,720),'uint16')}
DOCUMENTS=('source','construction_teacher_only','codebook_teacher_only')


def validate(bundle):
    if set(bundle)!=set(DOCUMENTS)|{'student','sensor_teacher_only'}:raise ValueError('closed bundle required')
    for section in ('student','sensor_teacher_only'):
        if set(bundle[section])!={k.split('/')[1] for k in ARRAYS if k.startswith(section+'/')}:
            raise ValueError('closed numeric section required')
    for k,(shape,dtype) in ARRAYS.items():
        section,name=k.split('/');a=bundle[section][name]
        if not isinstance(a,(np.ndarray,np.generic)) or a.shape!=shape or a.dtype!=np.dtype(dtype) or not np.isfinite(a).all():
            raise ValueError('array shape/dtype/numeric mismatch: '+k)
    if any(type(bundle[k]) is not dict for k in DOCUMENTS):raise ValueError('JSON dictionaries required')


def encode_bundle(bundle):
    validate(bundle)
    header=json.dumps({k:bundle[k] for k in DOCUMENTS},sort_keys=True,allow_nan=False).encode()
    if len(header)>16*1024**2:raise ValueError('metadata cap')
    arrays={k:bundle[k.split('/')[0]][k.split('/')[1]] for k in ARRAYS}
    stream=io.BytesIO();np.savez_compressed(stream,**arrays,metadata=np.frombuffer(header,dtype=np.uint8))
    payload=stream.getvalue()
    if len(payload)>MAX_PACKET:raise ValueError('packet cap')
    return payload


def decode_bundle(payload):
    if not isinstance(payload,bytes) or not 0<len(payload)<=MAX_PACKET:raise ValueError('bounded packet required')
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        entries=z.infolist()
        if len(entries)!=10 or {e.filename for e in entries}!={k+'.npy' for k in (*ARRAYS,'metadata')} or sum(e.file_size for e in entries)>MAX_PACKET:
            raise ValueError('closed bounded NPZ required')
    with np.load(io.BytesIO(payload),allow_pickle=False) as archive:
        meta=archive['metadata']
        if meta.dtype!=np.uint8 or meta.ndim!=1 or meta.size>16*1024**2:raise ValueError('bounded metadata bytes required')
        result=json.loads(meta.tobytes())
        if set(result)!=set(DOCUMENTS):raise ValueError('closed JSON header required')
        result.update(student={},sensor_teacher_only={})
        for k in ARRAYS:
            section,name=k.split('/');result[section][name]=archive[k].copy()
    validate(result);return result


def write_frame(stream,payload):
    if not 0<len(payload)<=MAX_PACKET:raise ValueError('bounded frame required')
    for part in (struct.pack('!Q',len(payload)),hashlib.sha256(payload).digest(),payload):
        view=memoryview(part)
        while view:
            n=stream.write(view)
            if n is None or n<=0:raise EOFError('frame write made no progress')
            view=view[n:]
    stream.flush()


def read_frame(stream):
    def exact(n):
        chunks=[]
        while n:
            piece=stream.read(n)
            if not piece:raise EOFError('truncated frame')
            chunks.append(piece);n-=len(piece)
        return b''.join(chunks)
    header=stream.read(1)
    if header==b'':return None
    header+=exact(7)
    n=struct.unpack('!Q',header)[0]
    if not 0<n<=MAX_PACKET:raise ValueError('frame cap')
    digest=exact(32);payload=exact(n)
    if hashlib.sha256(payload).digest()!=digest:raise ValueError('frame checksum mismatch')
    return payload
