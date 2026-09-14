import pytest
from mtare_topo.integration.explicit_scan_sources import ObservedScan,ExplicitScanSources


def message(i):
    return dict(schema_version='native_scan_source_v1',session_id='s',source_seq=i+6,
        source_stamp_ns=100+i*10,registered_seq=i,registered_stamp_ns=99+i*10,
        raw_topic='/velodyne_points',registered_topic='/registered_scan',source_frame='velodyne',
        registered_frame='map',raw_frame_matches=True,physical_pose_verified=False)


def registry():
    r=ExplicitScanSources(segment_id='one',source_artifact_ref='source_sha_manifest.json')
    for i in range(5):
        # Intentionally inverted cross-subscriber receipts: explicit callback
        # evidence, not equal/nearest stamps or reception order, proves identity.
        r.add(message(i),raw=ObservedScan(i+6,100+i*10,103+i*10),
            registered=ObservedScan(i,99+i*10,102+i*10),evidence_receipt_ns=104+i*10,evidence_ref=f'row:{i}')
    return r


def window(r,receipt=144):
    return r.window([f'registered_scan:{99+i*10}' for i in range(5)],segment_id='one',
        snapshot_stamp_ns=139,snapshot_receipt_ns=receipt)


def test_exact_source_without_equal_stamp_or_capture_tail_assumption():
    result=window(registry())
    assert result['source_transport_bound'] and result['available_at_native_snapshot']
    assert result['raw_source_keys'][0]=='/velodyne_points:100'
    assert not result['registration_verified'] and not result['usable_for_native_advice']


def test_future_evidence_does_not_claim_live_availability():
    assert not window(registry(),receipt=143)['available_at_native_snapshot']


def test_missing_binding_no_nearest_fallback():
    r=registry();r.by_registered.pop(119)
    assert window(r)['reason']=='MISSING_EXPLICIT_SOURCE'


@pytest.mark.parametrize('key,value', [('source_seq',99),('registered_stamp_ns',111),
    ('raw_frame_matches',False),('session_id','different'),('physical_pose_verified',True)])
def test_mismatching_actual_header_frame_session_and_claim_rejected(key,value):
    r=registry();m=message(5);m[key]=value
    with pytest.raises(ValueError):
        r.add(m,raw=ObservedScan(11,150,153),registered=ObservedScan(5,149,152),
            evidence_receipt_ns=154,evidence_ref='row5')


def test_other_segment_or_reordered_native_keys_rejected():
    r=registry();keys=[f'registered_scan:{99+i*10}' for i in [0,1,3,2,4]]
    with pytest.raises(ValueError):
        r.window(keys,segment_id='one',snapshot_stamp_ns=139,snapshot_receipt_ns=144)
    with pytest.raises(ValueError):
        r.window(keys,segment_id='two',snapshot_stamp_ns=139,snapshot_receipt_ns=144)
