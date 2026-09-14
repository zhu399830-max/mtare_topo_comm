import itertools
import zipfile
import pytest
from mtare_topo.data.gse_singleton_source_evidence import verify_singleton_producer


def test_singleton_implication_including_overlaps():
    universe = {0, 1, 2}
    subsets = [set(s) for n in range(1, 4) for s in itertools.combinations(universe, n)]
    accepted = [(face, active) for face in subsets for active in subsets if face <= active]
    for face, active in accepted:
        if len(active) == 1:
            assert face == active
    # Multi-source active sets do not imply a unique face-source set.
    assert len({tuple(sorted(f)) for f, a in accepted if a == {0, 1}}) == 3


def test_unbound_producer_rejected(tmp_path):
    p = tmp_path / 'unreviewed.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('src/mtare_topo/teacher/csg_mesh_provenance.py', 'modified')
    with pytest.raises(ValueError, match='unreviewed producer'):
        verify_singleton_producer(p)
