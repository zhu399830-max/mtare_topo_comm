"""Run the identical four native fixtures through the composed interface."""
import check_covered_event_native as fixtures
from mtare_topo.teacher.covered_reference_diagnostic import CoveredReferenceDiagnostic


if __name__ == '__main__':
    # Explicit test-harness substitution only; production modules are unchanged.
    fixtures.CoveredEventDiagnostic = CoveredReferenceDiagnostic
    fixtures.main()
