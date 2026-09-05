#!/usr/bin/env python3
"""V2R: run the V2 exporter with tunnel quotas derived after eligibility."""

from __future__ import annotations

from pathlib import Path

import execute_cano_phase2_supervised_range_dataset_v2 as base

from mtare_topo.data.cano_phase2_dataset import ROLE_ORDER
from mtare_topo.data.cano_phase2_selector_v2 import (
    length_proportional_tunnel_quotas,
    select_spatially_balanced_clusters as solve_selector,
)
from mtare_topo.governance import load_json, write_json


def select_v2r(candidates, role_quota, maximum_coverage_radius_m=10.0, *, tunnel_quota=None, constraint_reference=None):
    """Ignore pre-eligibility quotas and derive exact quotas from eligible capacity."""

    selected_count = sum(int(role_quota[role]) for role in ROLE_ORDER)
    eligible_quota = length_proportional_tunnel_quotas(candidates, selected_count)
    selected, audit = solve_selector(
        candidates,
        role_quota,
        maximum_coverage_radius_m,
        tunnel_quota=eligible_quota,
        constraint_reference=constraint_reference,
    )
    audit["selector_version"] = "V2R"
    audit["tunnel_quota_basis"] = "eligible_candidate_capacity_after_objective_audit"
    audit["pre_eligibility_tunnel_quota_ignored"] = (
        {str(key): int(value) for key, value in sorted((tunnel_quota or {}).items())}
        if tunnel_quota is not None else None
    )
    return selected, audit


def main() -> int:
    base.select_spatially_balanced_clusters = select_v2r
    result = base.main()
    if result == 0:
        import sys
        run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1]).resolve()
        summary = load_json(run_dir / "metrics/summary.json")
        summary["schema_version"] = "cano_phase2_dataset_summary_v2r"
        summary["selector_version"] = "V2R"
        summary["tunnel_quota_basis"] = "eligible_candidate_capacity_after_objective_audit"
        # The shared V2 runner validates cardinality/evidence before the V2R
        # runner gives the sealed run its final V2R identity.
        write_json(run_dir / "metrics/summary.json", summary)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
