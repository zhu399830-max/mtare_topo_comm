import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.evaluation.double_frame_diagnostic import frame_inputs, score_return


def test_actual_scan_interface_and_determinism():
    case, origin, directions, expected, manifest = frame_inputs()
    assert manifest == frame_inputs()[-1]
    assert manifest['rays'] == 11520
    np.testing.assert_array_equal(origin, [-4.4, 0., .04])
    np.testing.assert_array_equal(directions, world_directions(
        lidar_local_directions().reshape(-1, 3).astype(np.float64), case['yaw_deg'][0]))
    assert expected.shape == (11520,)
    assert (expected > 0).all()
    assert manifest['five_frame_observations'] == 0


def test_unknown_and_nonfinite_never_become_valid_scan():
    for result in ({'status': 'needs_reference'},
                   {'status': 'candidate', 'distance_m': float('nan')},
                   {'status': 'reference_return', 'distance_m': float('inf')}):
        score = score_return(result, 2.)
        assert not score['passed'] and not score['sensor_valid']


def test_fixed_comparison_and_sensor_range_are_separate():
    assert score_return({'status': 'candidate', 'distance_m': 2.}, 2.)['passed']
    assert not score_return({'status': 'candidate', 'distance_m': 2.001}, 2.)['passed']
    score = score_return({'status': 'reference_return', 'distance_m': .2}, .2)
    assert score['passed'] and not score['sensor_valid']
    assert not score_return({'status': 'out_of_range'}, 2.)['passed']
    assert score_return({'status': 'out_of_range'}, 51.)['passed']
