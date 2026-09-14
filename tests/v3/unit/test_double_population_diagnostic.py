import numpy as np
from mtare_topo.evaluation.double_frame_diagnostic import frame_inputs
from mtare_topo.evaluation.double_population_diagnostic import frames, input_manifest


def test_first_frame_exactly_preserves_passed_input():
    old = frame_inputs(); new = next(frames())
    assert old[0] == new[0]
    for a, b in zip(old[1:4], new[1:4]):
        np.testing.assert_array_equal(a, b)
    for k, v in new[-1].items():
        assert old[-1][k] == v


def test_complete_population_no_missing_or_duplicate_frames():
    manifest = input_manifest()
    rows = manifest['frame_inputs']
    assert len(rows) == 60
    assert len({(r['case_id'], r['frame_index']) for r in rows}) == 60
    assert sum(r['rays'] for r in rows) == 691200
    assert len({r['case_id'] for r in rows}) == 12
    assert manifest['comparison_atol_m'] == 1e-5
