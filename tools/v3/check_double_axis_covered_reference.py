"""Existing 360-axis contract through the versioned covered/reference chain."""
import check_double_axis_exact_reference as fixtures
from mtare_topo.teacher.covered_reference_diagnostic import CoveredReferenceDiagnostic


if __name__ == '__main__':
    fixtures.ExactReferenceDiagnostic = CoveredReferenceDiagnostic
    fixtures.main()
