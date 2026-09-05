"""Source-independent positive SECTION evidence, not an opening/region teacher.

The existing source stores endpoint unions and full free-space operands, not
a decomposition into local region boundaries. This kernel therefore does not
invent region polyhedra or infer membership from absent returns. It removes
only exactly duplicated source sections before rerunning the existing finite
ray competition, preserving both original and deduplicated evidence.
"""
from dataclasses import dataclass

import numpy as np

from mtare_topo.teacher.gse_portal_ray_evidence import (
    CausalRaySegments, PortalRayEvidence, PortalSections, aperture_ray_evidence,
)


@dataclass(frozen=True)
class SectionPositiveEvidence:
    candidate_to_section: np.ndarray
    source_candidate_groups: tuple[tuple[int, ...], ...]
    original_evidence: PortalRayEvidence
    unique_section_evidence: PortalRayEvidence
    section_status: tuple[str, ...]
    region_membership_available: bool = False
    complete_event_labels_available: bool = False
    independent_opening_claim: bool = False


def _unoriented(vector):
    vector = np.asarray(vector)
    nonzero = np.flatnonzero(vector)
    return tuple(vector if not len(nonzero) or vector[nonzero[0]] > 0 else -vector)


def _exact_section_key(sections, index):
    center = sections.centers_m[index]
    normal = sections.inward_normals[index]
    width = sections.width_directions[index]
    height = np.cross(normal, width)
    polygon = sections.polygon_normalized_xy[index]
    vertices = (center + polygon[:, :1] * sections.half_axes_m[index, 0] * width
                + polygon[:, 1:] * sections.half_axes_m[index, 1] * height)
    # Exact equality only: no radius/plane tolerance collapses nearby caps.
    # Retain the ideal-section parameters too, since the existing witness
    # operator checks their interior in addition to the actual mesh polygon.
    return (tuple(center), _unoriented(normal), _unoriented(width),
            tuple(sections.half_axes_m[index]), float(sections.exponents[index]),
            tuple(sorted(tuple(vertex) for vertex in vertices)))


def section_positive_evidence(rays: CausalRaySegments, sections: PortalSections,
                              *, chunk_size: int = 2048) -> SectionPositiveEvidence:
    """Deduplicate exact source sections, never choose by composition/node ID.

    SECTION_POSITIVE means a finite valid ray crosses one unique supplied
    section before its first return. It does not establish that this is a
    region boundary, a distinct navigable opening, or a terminal/junction.
    Serial or overlapping sections remain in the shared competition. No-ray,
    occluded, grazing and multi-section evidence stays UNKNOWN.
    """
    rays.validate(); sections.validate()
    if sections.polygon_normalized_xy is None:
        raise ValueError("exact source mesh section polygon is required")
    original = aperture_ray_evidence(rays, sections, chunk_size=chunk_size)
    keys = {}; groups = []; mapping = []
    for index in range(len(sections.centers_m)):
        key = _exact_section_key(sections, index)
        if key not in keys:
            keys[key] = len(groups); groups.append([])
        group = keys[key]; groups[group].append(index); mapping.append(group)
    representatives = np.array([indices[0] for indices in groups], dtype=np.int64)
    unique = PortalSections(**{name: getattr(sections, name)[representatives].copy()
        for name in ("centers_m", "inward_normals", "width_directions", "half_axes_m",
                     "exponents", "polygon_normalized_xy")})
    evidence = aperture_ray_evidence(rays, unique, chunk_size=chunk_size)
    return SectionPositiveEvidence(np.asarray(mapping, dtype=np.int64),
        tuple(tuple(indices) for indices in groups), original, evidence,
        tuple("SECTION_POSITIVE" if yes else "UNKNOWN_NO_EXCLUSIVE_RAY" for yes in evidence.witnessed))
