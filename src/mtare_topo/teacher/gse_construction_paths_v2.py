"""Strict P1a schema adapter; preserve the archived reference-path semantics.

The legacy path kernel uses an internal ``compositions`` key. On disk the
producer writes ``composition_operations``. Never accept the internal spelling
as a substitute for the producer schema or silently supply missing fields.
"""

from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from mtare_topo.teacher.gse_construction_paths_v1r import (
    construction_incident_paths as _reference_paths,
)


def construction_incident_paths(document):
    base = document.get("base_construction")
    if type(base) is not dict or base.get("coordinate_frame") != "cano_world":
        raise ValueError("explicit original cano_world source construction required")
    groups = base.get("composition_operations")
    if type(groups) is not list or "compositions" in base:
        raise ValueError("original composition_operations list required; no schema aliases")
    # Run strict raw incidence checks before the existing loader's numeric/string
    # conversions. This retains degree checks and forbids endpoint snapping.
    paths = _reference_paths({"base_construction": {
        **base, "compositions": groups,
    }})
    construction, realized = load_p1a_realized_construction(document)
    if len(construction.compositions) != len(paths) or not realized:
        raise ValueError("decoded construction differs from reference paths")
    return paths
