"""Float32 range ambiguity diagnostic, not a surface-uniqueness certificate.

Closed endpoint cells conservatively retain round-to-even ties. The caller
must separately flag uncertain/coplanar geometry; approximate interval roots
are not silently treated as exact physical truth.
"""
import numpy as np


def float32_cells(ranges):
    values = np.asarray(ranges)
    if values.dtype != np.float32 or values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('one dimensional positive finite float32 ranges required')
    previous = np.nextafter(values, np.float32(-np.inf)).astype(np.float64)
    following = np.nextafter(values, np.float32(np.inf)).astype(np.float64)
    if not np.isfinite(previous).all() or not np.isfinite(following).all():
        raise ValueError('finite neighboring representable values required')
    middle = values.astype(np.float64)
    return np.column_stack(((previous + middle)/2., (middle + following)/2.))


def source_candidates(ranges, intervals, *, uncertain_operands):
    """Return possible entry/exit surfaces within stored range cells.

    intervals is NxKx2 closed line-solid intersection [entry, exit]. Empty
    intersections must be [inf,-inf]; tangency [t,t] must NOT be discarded.
    uncertain_operands NxK is mandatory and includes coplanarity/numerical
    uncertainty supplied by the geometric evaluator. No source IDs or model
    outputs influence this calculation.
    """
    cells = float32_cells(ranges)
    spans = np.asarray(intervals, dtype=np.float64)
    uncertain = np.asarray(uncertain_operands)
    if spans.ndim != 3 or spans.shape[0] != len(cells) or spans.shape[2] != 2 or spans.shape[1] < 1:
        raise ValueError('NxKx2 intervals required')
    if uncertain.shape != spans.shape[:2] or uncertain.dtype != bool or np.isnan(spans).any():
        raise ValueError('explicit boolean geometry uncertainty required')
    entry, exit = spans[...,0], spans[...,1]
    empty = np.isposinf(entry) & np.isneginf(exit)
    if ((entry > exit) & ~empty).any():
        raise ValueError('invalid reversed interval')
    low, high = cells[:,0,None], cells[:,1,None]
    possible_entry = ~empty & (entry >= low) & (entry <= high)
    possible_exit = ~empty & (exit >= low) & (exit <= high)
    possible = possible_entry | possible_exit | uncertain
    return dict(cell_m=cells, possible_entry=possible_entry, possible_exit=possible_exit,
                possible_surface=possible, uncertain_operands=uncertain.copy(),
                singleton_candidate=(possible.sum(axis=1)==1) & ~uncertain.any(axis=1),
                # Deliberately not synonymous with singleton_candidate.
                surface_uniqueness_certified=np.zeros(len(cells),dtype=bool))
