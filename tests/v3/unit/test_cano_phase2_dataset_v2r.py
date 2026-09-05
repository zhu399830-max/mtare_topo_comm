import unittest

from mtare_topo.data.cano_phase2_selector_v2 import selector_v2_audit
from execute_cano_phase2_supervised_range_dataset_v2r import select_v2r


def candidate(index, role="interior"):
    return {
        "cluster_id": f"c{index}", "tunnel_id": 0,
        "center_arc_m": float(index * 5), "primary_role": role,
        "junction_event_ids": ["j0"] if role == "junction" else [],
        "terminal_event_ids": [],
    }


class Phase2DatasetV2RTest(unittest.TestCase):
    def test_quota_is_derived_from_eligible_capacity_after_rejection(self):
        reference = [candidate(index, "junction" if index == 3 else "interior") for index in range(7)]
        eligible = reference[1:]
        selected, audit = select_v2r(
            eligible, {"interior": 3, "junction": 1, "terminal": 0},
            tunnel_quota={0: 5}, constraint_reference=reference,
        )
        self.assertEqual(len(selected), 4)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["tunnel_quota"], {"0": 4})
        self.assertEqual(audit["pre_eligibility_tunnel_quota_ignored"], {"0": 5})
        self.assertEqual(audit["tunnel_quota_basis"], "eligible_candidate_capacity_after_objective_audit")


if __name__ == "__main__":
    unittest.main()
