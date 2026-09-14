"""Bounded local Unix-socket transport; arrays never use pickle."""
import io
import socket
import struct
import numpy as np

MAX_BYTES=8*1024*1024

def read_exact(stream,n):
    parts=[]
    while n:
        part=stream.recv(n)
        if not part:raise ValueError('truncated inference message')
        parts.append(part);n-=len(part)
    return b''.join(parts)

def receive(stream):
    size=struct.unpack('!I',read_exact(stream,4))[0]
    if not 0<size<=MAX_BYTES:raise ValueError('inference message exceeds bound')
    with np.load(io.BytesIO(read_exact(stream,size)),allow_pickle=False) as data:
        return {k:data[k] for k in data.files}

def send(stream,arrays):
    output=io.BytesIO();np.savez(output,**arrays);payload=output.getvalue()
    if not 0<len(payload)<=MAX_BYTES:raise ValueError('inference message exceeds bound')
    stream.sendall(struct.pack('!I',len(payload))+payload)

class LocalGeometryPredictor:
    def __init__(self,path,timeout_s=2.):
        if not 0<timeout_s<=2:raise ValueError('bounded timeout required')
        self.path=str(path);self.timeout=timeout_s

    def __call__(self,**request):
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
            stream.settimeout(self.timeout);stream.connect(self.path)
            send(stream,request);response=receive(stream)
        return dict(axes=response['axes'],probabilities=response['probabilities'],
            source_frame_keys=response['source_frame_keys'].tolist(),timestamp=float(response['timestamp']),
            checkpoint_sha256=str(response['checkpoint_sha256']),evidence=str(response['evidence']))
