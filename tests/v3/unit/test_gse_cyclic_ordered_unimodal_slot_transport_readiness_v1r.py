from execute_gse_cyclic_ordered_unimodal_slot_transport_readiness_v1r import replacement_network_check


def test_degree_diagnostic_is_retained_but_not_dimensionally_pooled():
    metrics = {
        "slot_mass_rotation": 2.7e-7,
        "raw_mass_rotation": 2e-9,
        "kappa_rotation": 6e-8,
        "concentration_rotation": 2e-7,
        "count_rotation": 3e-8,
        "slot_geometry_rotation": 6e-7,
        "cyclic_slot_loss": 0.0,
        "batch_permutation": 8e-6,
        "repeat": 0.0,
        "bearing_rotation_deg": 0.0036,
    }
    passed, error = replacement_network_check(metrics)
    assert passed
    assert error == 8e-6


def test_distribution_error_still_fails_at_the_original_threshold():
    metrics = {
        "slot_mass_rotation": 3.1e-5,
        "raw_mass_rotation": 0.0,
        "kappa_rotation": 0.0,
        "concentration_rotation": 0.0,
        "count_rotation": 0.0,
        "slot_geometry_rotation": 0.0,
        "cyclic_slot_loss": 0.0,
        "batch_permutation": 0.0,
        "repeat": 0.0,
    }
    passed, error = replacement_network_check(metrics)
    assert not passed
    assert error == 3.1e-5
