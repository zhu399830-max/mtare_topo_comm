import json
from types import SimpleNamespace
import pytest
from mtare_topo.evaluation import native_graph_population as population


def test_population_identity_logging_and_no_resume(tmp_path, monkeypatch):
    (tmp_path/'artifacts').mkdir(); (tmp_path/'logs').mkdir()
    rows=[dict(source=dict(task='synthetic', source_sequence_id=i, frame_rows=list(range(5))),
        input_path='synthetic.npz', input_sha256='fake', input_row=i,
        observation_count=307, split='fit', parent_id='synthetic') for i in range(307)]
    scope=dict(observations=rows, counts=dict(observations=307))
    seen=[]
    monkeypatch.setattr(population, 'read_pinned', lambda *a: b'synthetic container')
    def bind(raw, **kw):
        seen.append(kw['row'])
        return SimpleNamespace(range_valid=None, translation_m=None, yaw_deg=None,
                               input_binding_sha256='synthetic binding')
    monkeypatch.setattr(population, 'bind_feature_input', bind)
    monkeypatch.setattr(population, 'encode_observation', lambda *a: b'sensor only')
    def execute(command, packet, **kw):
        assert packet==b'sensor only'
        graph=dict(nodes=0, edges=0, rays=0, edge_audit=[])
        kw['stdout_path'].write_text(json.dumps(graph))
        kw['stderr_path'].write_text('native log')
        return graph, dict(elapsed_s=0)
    monkeypatch.setattr(population, 'execute_native', execute)
    result=population.export_population(tmp_path, tmp_path, scope, command=['fake'], environment={})
    assert seen==list(range(307))
    assert len(result['observations'])==307
    assert result['cross_observation_edges']==0
    events=[json.loads(s) for s in (tmp_path/'logs/native_observations.jsonl').read_text().splitlines()]
    assert len(events)==614
    assert [e['event'] for e in events[:2]]==['START', 'COMPLETE']
    with pytest.raises(FileExistsError):
        population.export_population(tmp_path, tmp_path, scope, command=['fake'], environment={})


def test_wrong_population_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match='307'):
        population.export_population(tmp_path, tmp_path, dict(observations=[], counts={}),
                                     command=[], environment={})
    assert not list(tmp_path.iterdir())
