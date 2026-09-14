import socket
import struct
import threading
import pytest
from mtare_topo.integration.structural_token_transport import receive_json,send_json,StructuralTokenClient,MAX_JSON_BYTES


def test_json_round_trip():
    a,b=socket.socketpair()
    with a,b:
        send_json(a,{'request_id':'one','manifest':{'path':'raw.json','sha256':'abc'}})
        assert receive_json(b)['request_id']=='one'


@pytest.mark.parametrize('payload',[b'{"a":1,"a":2}',b'{"a":NaN}',b'{"bad":'])
def test_malformed_json_rejected(payload):
    a,b=socket.socketpair()
    with a,b:
        a.sendall(struct.pack('!I',len(payload))+payload)
        with pytest.raises(ValueError):receive_json(b)


def test_message_bound_rejected_before_payload_read():
    a,b=socket.socketpair()
    with a,b:
        a.sendall(struct.pack('!I',MAX_JSON_BYTES+1))
        with pytest.raises(ValueError,match='TOKEN_MESSAGE_SIZE'):receive_json(b)


def test_client_records_actual_receipt_without_implying_ready_for_decision(tmp_path):
    path=tmp_path/'test.sock'
    server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(path));server.listen(1)
    def run():
        connection,_=server.accept()
        with connection:
            request=receive_json(connection)
            send_json(connection,dict(schema_version='streaming_structural_token_response_v1',
                request_id=request['request_id'],status='PROCESSED'))
    thread=threading.Thread(target=run);thread.start()
    try:
        result=StructuralTokenClient(path,timeout_s=1).request({'request_id':'x'})
        assert result['client_received_monotonic_ns']>=result['client_started_monotonic_ns']
        assert not result['usable_for_native_advice']
    finally:thread.join(timeout=2);server.close()
