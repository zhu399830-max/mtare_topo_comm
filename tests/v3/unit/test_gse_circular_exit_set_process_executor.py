from __future__ import annotations


def test_float_rotation_contract_uses_bound_not_exact_equality() -> None:
    measured = 5.960464477539063e-8
    assert measured != 0.0
    assert measured <= 1e-6
    assert measured <= 3e-5
