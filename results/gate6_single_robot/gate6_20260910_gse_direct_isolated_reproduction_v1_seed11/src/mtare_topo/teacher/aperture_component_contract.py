"""Discrete boundary-component contract, NOT a mesh/observation classifier.

Caller must supply a conforming boundary cell complex and independently
qualified whole-cell states. Cell-center samples or reference IDs alone are
not sufficient. This module never turns construction sections into instances.
"""
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class CellState(IntEnum):
    UNKNOWN=-1
    BLOCKED=0
    FREE=1


@dataclass(frozen=True)
class BoundaryPartition:
    geometry_components: tuple[tuple[int,...],...]
    observed_components: tuple[tuple[int,...],...]
    # Geometry components may still join through unknown cells. False here
    # means genuinely separated in the supplied complex; True is not a merge.
    unresolved_component_pairs: tuple[tuple[int,int],...]
    conflicting_cells: tuple[int,...]
    complete_geometry_partition: bool
    complete_observed_partition: bool
    physical_surface_extraction_verified: bool=False
    training_labels_qualified: bool=False


def _components(adjacency,mask):
    remaining=set(np.flatnonzero(mask).tolist());out=[]
    while remaining:
        seed=min(remaining);remaining.remove(seed);stack=[seed];component=[]
        while stack:
            i=stack.pop();component.append(i)
            for j in adjacency[i]:
                if j in remaining:remaining.remove(j);stack.append(j)
        out.append(tuple(sorted(component)))
    return tuple(out)


def partition_boundary(adjacency,geometry_state,observed_state):
    n=len(adjacency);g=np.asarray(geometry_state);o=np.asarray(observed_state)
    if g.shape!=(n,) or o.shape!=(n,) or not np.isin(g,[-1,0,1]).all() or not np.isin(o,[-1,0,1]).all():
        raise ValueError('explicit FREE/BLOCKED/UNKNOWN state per boundary cell required')
    neighbors=[]
    for i,entries in enumerate(adjacency):
        values=tuple(entries)
        if any(type(j) is not int or j<0 or j>=n or j==i for j in values) or len(set(values))!=len(values):
            raise ValueError('invalid boundary adjacency')
        neighbors.append(set(values))
    if any(i not in neighbors[j] for i,values in enumerate(neighbors) for j in values):raise ValueError('adjacency must be symmetric')
    conflicts=tuple(np.flatnonzero((g!=-1)&(o!=-1)&(g!=o)).tolist())
    physical=_components(neighbors,g==1)
    # Observations are kept separate: hidden reference connectivity cannot
    # silently merge two observed fragments into a positive membership label.
    observed=_components(neighbors,o==1)
    possible=_components(neighbors,g!=0)
    possible_owner={cell:i for i,group in enumerate(possible) for cell in group}
    unresolved=tuple((i,j) for i,a in enumerate(physical) for j,b in enumerate(physical) if i<j and possible_owner[a[0]]==possible_owner[b[0]])
    return BoundaryPartition(physical,observed,unresolved,conflicts,
        bool(not conflicts and np.all(g!=-1)),bool(not conflicts and np.all(o!=-1)))


def map_reference_sections(partition,reference_cells):
    """Teacher bookkeeping only. Multiple sections may map to ONE component.

    No reference ID enters partition_boundary. A section crossing components
    or unknown/blocked cells is unresolved, never assigned by nearest center.
    """
    owner={cell:i for i,group in enumerate(partition.geometry_components) for cell in group}
    result=[]
    for cells in reference_cells:
        cells=tuple(cells);ids={owner.get(i) for i in cells}
        result.append(next(iter(ids)) if cells and None not in ids and len(ids)==1 else None)
    return tuple(result)
