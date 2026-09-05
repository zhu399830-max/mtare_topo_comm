from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPEC = spec_from_file_location(
    "run_aee_composite_v9_stochastic_v3r",
    ROOT / "tools/v3/run_aee_composite_v9_stochastic_v3r.py",
)
module = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _cases():
    from mtare_topo.evaluation.closed_loop_matrix import enumerate_stochastic_cases, load_stochastic_matrix
    matrix = load_stochastic_matrix(ROOT / "configs/v3/gate6/mtare_single_robot_stochastic_matrix_v3.json")
    return enumerate_stochastic_cases(matrix)


def test_recovery_selects_whole_failed_block_and_tail():
    selected = module.recovery_cases(_cases())
    assert len(selected) == 21
    assert sum(case.block_id == "tunnel_env23" for case in selected) == 9
    assert [case.index for case in selected if case.index >= 77] == list(range(77, 90))


def test_source_reuse_excludes_whole_failed_block():
    selected = module.source_cases(_cases())
    assert len(selected) == 69
    assert all(case.index < 77 for case in selected)
    assert all(case.block_id != "tunnel_env23" for case in selected)


def test_recovery_and_source_form_exact_original_schedule():
    cases = _cases()
    ids = {case.case_id for case in module.recovery_cases(cases)} | {case.case_id for case in module.source_cases(cases)}
    assert ids == {case.case_id for case in cases}
