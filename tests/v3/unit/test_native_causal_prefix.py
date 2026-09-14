from dataclasses import replace
import pytest
from mtare_topo.integration.native_scan_provenance import (
    CausalPrefixBinding, ScanHeader, audit_startup_sequence, IMAGE, SOURCE_SHA)


def certificate(changed=None):
    raw = [ScanHeader(i, 100+i*10, 101+i*10, '/gazebo') for i in range(14)]
    out = [ScanHeader(i, 159+i*10, 162+i*10, '/vehicleSimulator') for i in range(8)]
    if changed is not None:
        changed(raw, out)
    p = audit_startup_sequence(raw, out, segment_id='one', bag_sha256='a'*64,
                              image=IMAGE, source_sha256=SOURCE_SHA)
    return CausalPrefixBinding(p, 'clock.json', 'b'*64,
                              'pinned_gazebo_measurement_and_rosbag_sim_clock_v1')


def bind(c, start=0, receipt=222):
    keys = [f'registered_scan:{159+10*i}' for i in range(start, start+5)]
    return c.bind_native(keys, snapshot_stamp_ns=159+10*(start+4), snapshot_receipt_ns=receipt)


def test_prefix_proves_source_not_complete_tail_or_registration():
    c = certificate()
    assert c.prefix_count == 7
    result = bind(c, 2)
    assert result['raw_orders'] == [8, 9, 10, 11, 12]
    assert result['transport_source_bound']
    assert not result['usable_for_native_advice'] and not result['registration_verified']
    assert not result['complete_capture_boundary_verified']
    with pytest.raises(ValueError, match='OUTSIDE_CERTIFIED'):
        bind(c, 3, 232)


def test_missing_input_or_delayed_output_ends_prefix_no_later_recovery():
    def damage(raw, out):
        out[2] = replace(out[2], arrival_ns=raw[9].stamp_ns)
    c = certificate(damage)
    assert c.prefix_count == 2
    with pytest.raises(ValueError, match='OUTSIDE_CERTIFIED'):
        bind(c)


def test_untrusted_clock_early_snapshot_and_reordered_sources_fail():
    c = certificate()
    with pytest.raises(ValueError, match='COMMON_CLOCK'):
        replace(c, common_clock_contract='assumed')
    with pytest.raises(ValueError, match='NONCAUSAL'):
        bind(c, receipt=201)
    keys = [f'registered_scan:{159+10*i}' for i in [0, 1, 3, 2, 4]]
    with pytest.raises(ValueError, match='ORDERED'):
        c.bind_native(keys, snapshot_stamp_ns=199, snapshot_receipt_ns=202)
