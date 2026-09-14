"""Covered-event diagnostic with reference fallback, not export authority."""
from .covered_event_diagnostic import CoveredEventDiagnostic


class CoveredReferenceDiagnostic(CoveredEventDiagnostic):
    def query(self, origin, direction, *, maximum_m=50.):
        result = super().query(origin, direction, maximum_m=maximum_m)
        if result['status'] != 'needs_reference':
            return result
        # The interval reference treats touching solids as a closed union.
        # It cannot certify open connectivity for an uncovered exact contact.
        if result.get('reason') == 'coincident_entry_exit':
            return dict(result, reference_attempted=False,
                        reference_status='blocked_uncovered_contact')
        checked = self._origin.check(origin)
        if checked.status != 'origin_checked':
            return result
        comparison = self.compare(origin, direction, checked.inside, maximum_m=maximum_m)
        if comparison.reference is None:
            return dict(result, reference_attempted=True, reference_status='unknown')
        hit = comparison.reference
        return dict(result, status='reference_return', reference_attempted=True,
                    reference_status='returned', distance_m=hit.distance_m,
                    sources=list(hit.source_primitive_ids))
