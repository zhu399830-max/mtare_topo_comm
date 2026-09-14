import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools/v3'))
spec = importlib.util.spec_from_file_location('dev_scope', ROOT/'tools/v3/freeze_conditional_development_inputs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def header():
    return dict(shape=[10, 3], chunks=[4, 3], filters=None, order='C', dtype='<f4')


def test_exact_rows_collateral_and_padding():
    plan = module.plan_rows(header(), [9, 1, 1])
    assert plan['selected_rows'] == [1, 9]
    assert plan['chunk_keys'] == ['0.0', '2.0']
    assert plan['decoded_intervals'] == [[0,4], [8,10]]
    assert plan['collateral_rows'] == 4
    assert plan['decoded_padded_bytes'] == 96


@pytest.mark.parametrize('rows', [[], [-1], [10], [True], [1.0]])
def test_invalid_source_rows(rows):
    with pytest.raises(ValueError):
        module.plan_rows(header(), rows)


def test_no_partial_tail_chunks():
    h = header(); h['chunks'] = [4, 2]
    with pytest.raises(ValueError):
        module.plan_rows(h, [1])


def test_preserve_original_float64_pose_precision():
    h = header(); h['dtype'] = '<f8'
    plan = module.plan_rows(h, [1])
    assert plan['header']['dtype'] == '<f8'
    assert plan['decoded_padded_bytes'] == 96
