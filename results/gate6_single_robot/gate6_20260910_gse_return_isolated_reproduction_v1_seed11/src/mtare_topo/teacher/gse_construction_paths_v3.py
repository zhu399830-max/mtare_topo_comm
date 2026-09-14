"""Keep source axis-segment provenance separate from observation connectors."""
from mtare_topo.teacher.gse_construction_paths_v2 import construction_incident_paths as _load


def construction_incident_paths(document):
    groups = _load(document)
    return tuple({**group, "paths": tuple({
        **path,
        # The source adapter prepends exactly one connector iff offset != 0.
        # No epsilon, nearest-point repair, or path modification is introduced.
        "axis_start_index": int(path["anchor_axis_offset_m"] != 0.),
    } for path in group["paths"])} for group in groups)
