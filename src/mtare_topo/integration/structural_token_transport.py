"""Bounded JSON-only local token transport, outside the control callback.

Raw arrays remain in SHA-bound input files on the shared workspace. There is
no arbitrary code serialization and no instruction to execute a supplied path.
The client response records receipt time; this is not scan/acquisition time.
"""
import json
import socket
import struct
import time

MAX_JSON_BYTES = 65536


def _read(stream, count):
    chunks=[]
    while count:
        value=stream.recv(count)
        if not value: raise ValueError('TRUNCATED_TOKEN_MESSAGE')
        chunks.append(value); count-=len(value)
    return b''.join(chunks)


def receive_json(stream):
    size=struct.unpack('!I',_read(stream,4))[0]
    if not 0<size<=MAX_JSON_BYTES: raise ValueError('TOKEN_MESSAGE_SIZE')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result: raise ValueError('DUPLICATE_TOKEN_JSON_KEY')
            result[key]=value
        return result
    return json.loads(_read(stream,size),object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('NONFINITE_TOKEN_JSON')))


def send_json(stream, value):
    payload=json.dumps(value,allow_nan=False,separators=(',',':')).encode()
    if not 0<len(payload)<=MAX_JSON_BYTES: raise ValueError('TOKEN_MESSAGE_SIZE')
    stream.sendall(struct.pack('!I',len(payload))+payload)


class StructuralTokenClient:
    """Call from an acquisition/inference worker, never block native advice."""
    def __init__(self,path,*,timeout_s):
        if not 0<timeout_s<=10: raise ValueError('BOUNDED_TOKEN_CLIENT_TIMEOUT')
        self.path=str(path); self.timeout_s=timeout_s

    def request(self,request):
        started=time.monotonic_ns()
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
            stream.settimeout(self.timeout_s); stream.connect(self.path)
            send_json(stream,request); response=receive_json(stream)
        received=time.monotonic_ns()
        if (not isinstance(response,dict) or response.get('request_id')!=request.get('request_id')
                or response.get('schema_version') not in ('streaming_structural_token_response_v1','streaming_structural_token_error_v1')):
            raise ValueError('TOKEN_RESPONSE_BINDING')
        return dict(response=response,client_started_monotonic_ns=started,
                    client_received_monotonic_ns=received,usable_for_native_advice=False)
