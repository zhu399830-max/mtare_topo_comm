"""Loss-side singleton source evidence for the pinned original ray producer.

For an accepted return the producer requires nonempty intersected-face source
set F to be a subset of saved active set A. If |A|=1, F=A. Thus the source
operand is identifiable without a residual threshold; the triangle itself is
not. This says nothing about junction identity, traversability, or whether two
different construction operands are parts of one semantic tunnel.
"""
import hashlib
import zipfile
import numpy as np


PRODUCER_MEMBERS = {
    'src/mtare_topo/teacher/csg_mesh_provenance.py':
        'a903fd9b7292ceedf21162a77d7bf087ba2f4f1f536894a28d7ad179360854ba',
    'tools/v3/run_primitive_relation_p1a_sensor_provenance_export_v1.py':
        '33c33d2db0b7e52cf3a35dfe431754b018de30162053fdae896baaf05f0d4049',
    'src/mtare_topo/data/primitive_relation_sensor_export.py':
        '23c361223b5bf5e16c83a72857bc28523254176592d6081610d15f470757e745',
    'src/mtare_topo/data/primitive_relation_dataset.py':
        '7e33f2fd012c511b03ec59fc1a60156e09f457adedbe42d845067069eece3728',
}


def verify_singleton_producer(archive_path):
    """Check the reviewed source chain, not the current mutable checkout."""
    with zipfile.ZipFile(archive_path) as archive:
        for name, expected in PRODUCER_MEMBERS.items():
            if archive.namelist().count(name) != 1:
                raise ValueError('missing or duplicate producer member: ' + name)
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('unreviewed producer: ' + name)


def singleton_return_evidence(joined, *, producer_archive):
    """Qualify original-return source only; caller must bind run/input hashes.

    Source inventories must come from the original codebook, as in the frozen
    new12 extraction. No nearest-residual elimination or candidate rewriting
    is permitted before this operation. IDs are local to one observation.
    """
    verify_singleton_producer(producer_archive)
    mask = np.asarray([len(candidates) == 1 for candidates in joined.candidates], dtype=bool)
    mask.setflags(write=False)
    return mask
