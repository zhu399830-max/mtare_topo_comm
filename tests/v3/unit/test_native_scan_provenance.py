import pytest
from dataclasses import replace
from mtare_topo.integration.native_scan_provenance import (
    IMAGE, SOURCE_SHA, ScanHeader, ModelWindowSource, audit_startup_sequence, bind_exact_window)


def headers():
    raw = [ScanHeader(i, 100 + 10*i, 101 + 10*i, '/gazebo') for i in range(14)]
    registered = [ScanHeader(i, 159 + 10*i, 162 + 10*i, '/vehicleSimulator') for i in range(8)]
    return raw, registered


def audit(raw=None, registered=None, **kwargs):
    r, g = headers()
    return audit_startup_sequence(r if raw is None else raw, g if registered is None else registered,
        **dict(dict(segment_id='one', bag_sha256='a'*64, image=IMAGE, source_sha256=SOURCE_SHA), **kwargs))


def bind(p=None, **kwargs):
    provenance = audit() if p is None else p
    keys = tuple(x[0] for x in provenance.bindings[:5])
    sources = tuple(x[1] for x in provenance.bindings[:5])
    return bind_exact_window(provenance, **dict(dict(segment_id='one', registered_source_keys=keys,
        snapshot_stamp_ns=199, snapshot_receipt_ns=202,
        windows=[window('w0', sources)]), **kwargs))


def window(name, sources, **kwargs):
    return ModelWindowSource(**dict(dict(window_id=name, raw_source_keys=sources,
        segment_id='one', bag_sha256='a'*64, input_sha256='b'*64), **kwargs))


def test_sequence_origin_does_not_require_equal_stamp_or_certify_registration():
    result = bind()
    assert result['status'] == 'EXACT_CACHE_MATCH_PENDING_CAPTURE_BOUNDARY'
    assert result['raw_source_keys'][0] == '/velodyne_points:160'
    assert result['registered_source_keys'][0] == 'registered_scan:159'
    assert not result['registration_verified'] and not result['task_identity_verified']
    assert not result['capture_boundary_verified'] and not result['usable_for_native_advice']


def test_missing_exact_window_never_uses_nearest_or_future_window():
    p = audit()
    sources = tuple(x[1] for x in p.bindings[1:6])
    assert bind(p, windows=[window('nearby', sources)])['reason'] == 'NO_EXACT_MODEL_WINDOW'
    with pytest.raises(ValueError, match='NONCAUSAL'):
        bind(p, snapshot_receipt_ns=201)
    # Sensor header is deliberately earlier than raw. Arrival proves causality.
    assert bind(p)['status'] == 'EXACT_CACHE_MATCH_PENDING_CAPTURE_BOUNDARY'


def test_ambiguous_cache_order_and_duplicate_id_are_explicit():
    p = audit(); sources = tuple(x[1] for x in p.bindings[:5])
    windows = [window('a', sources), window('b', sources)]
    assert bind(p, windows=windows) == bind(p, windows=windows[::-1])
    assert bind(p, windows=windows)['reason'] == 'AMBIGUOUS_MODEL_WINDOW'
    with pytest.raises(ValueError, match='unique'):
        bind(p, windows=[window('a', sources), window('a', sources)])


@pytest.mark.parametrize('damage', ['drop', 'restart', 'other_producer', 'duplicate_stamp', 'premature_output'])
def test_incomplete_or_ambiguous_record_never_certifies_mapping(damage):
    raw, registered = headers()
    if damage == 'drop': raw.pop()
    if damage == 'restart': raw[8] = ScanHeader(0, 180, 181, '/gazebo')
    if damage == 'other_producer': registered[4] = ScanHeader(4, 199, 202, '/other')
    if damage == 'duplicate_stamp': raw[8] = ScanHeader(8, 170, 181, '/gazebo')
    if damage == 'premature_output': registered[0] = ScanHeader(0, 159, 100, '/vehicleSimulator')
    with pytest.raises(ValueError): audit(raw, registered)


def test_source_contract_or_segment_drift_rejected():
    with pytest.raises(ValueError, match='SOURCE_CONTRACT'): audit(source_sha256={})
    with pytest.raises(ValueError, match='SEGMENT'): bind(segment_id='another')


def test_missing_source_and_wrong_order_not_silently_repaired():
    keys = tuple(x[0] for x in audit().bindings[:5])
    assert bind(registered_source_keys=('missing',)+keys[1:])['reason'] == 'UNMAPPED_REGISTERED_FRAME'
    with pytest.raises(ValueError, match='NONCAUSAL'): bind(registered_source_keys=keys[::-1])


def test_identical_timestamps_in_other_bag_or_segment_cannot_bind():
    sources = tuple(x[1] for x in audit().bindings[:5])
    for mismatch in (dict(bag_sha256='c'*64), dict(segment_id='other')):
        with pytest.raises(ValueError, match='MODEL_CACHE_SOURCE_MISMATCH'):
            bind(windows=[window('w', sources, **mismatch)])


def test_certificate_reconstruction_cannot_bypass_source_or_sequence_checks():
    p = audit()
    with pytest.raises(ValueError): replace(p, registered=p.registered[1:])
    with pytest.raises(ValueError): replace(p, image='wrong')
    with pytest.raises((TypeError, ValueError)): replace(p, capture_boundary_verified=True)
