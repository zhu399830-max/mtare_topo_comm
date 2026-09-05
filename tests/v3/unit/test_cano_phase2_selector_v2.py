import unittest

from mtare_topo.data.cano_phase2_selector_v2 import (
    length_proportional_tunnel_quotas,
    select_spatially_balanced_clusters,
)


def candidate(tunnel, index, role="interior", junction=(), terminal=()):
    return {"cluster_id":f"t{tunnel}_k{index}","tunnel_id":tunnel,"center_arc_m":2.5+5*index,"primary_role":role,"junction_event_ids":list(junction),"terminal_event_ids":list(terminal)}


class Phase2SelectorV2Test(unittest.TestCase):
    def test_length_quota_is_exact_and_covers_every_tunnel(self):
        rows=[candidate(1,i) for i in range(8)]+[candidate(2,i) for i in range(4)]
        quota=length_proportional_tunnel_quotas(rows,9)
        self.assertEqual(sum(quota.values()),9);self.assertGreaterEqual(min(quota.values()),1);self.assertGreater(quota[1],quota[2])

    def test_milp_preserves_roles_events_tunnels_and_coverage(self):
        rows=[]
        for tunnel in (1,2):
            for index in range(8):
                role="junction" if index==2 else "terminal" if index==7 else "interior"
                rows.append(candidate(tunnel,index,role,(f"j{tunnel}",) if role=="junction" else (), (f"t{tunnel}",) if role=="terminal" else ()))
        selected,audit=select_spatially_balanced_clusters(rows,{"interior":8,"junction":2,"terminal":2},10.0)
        self.assertEqual(len(selected),12);self.assertTrue(audit["passed"]);self.assertTrue(audit["all_tunnels_represented"]);self.assertFalse(audit["event_failures"]);self.assertLessEqual(audit["maximum_candidate_to_selected_same_tunnel_arc_distance_m"],10.0)

    def test_infeasible_role_quota_raises(self):
        rows=[candidate(1,i) for i in range(6)]
        with self.assertRaises(RuntimeError):
            select_spatially_balanced_clusters(rows,{"interior":5,"junction":1,"terminal":0},10.0)

    def test_eligible_subset_is_measured_against_frozen_reference(self):
        rows=[candidate(0,index,"junction" if index==3 else "interior",("j0",) if index==3 else ()) for index in range(7)]
        eligible=[row for row in rows if row["cluster_id"] != "t0_k0"]
        selected,audit=select_spatially_balanced_clusters(
            eligible,{"interior":3,"junction":1,"terminal":0},10.0,
            tunnel_quota={0:4},constraint_reference=rows,
        )
        self.assertEqual(len(selected),4)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["candidate_clusters"],7)
        self.assertEqual(audit["eligible_candidate_clusters"],6)
        self.assertLessEqual(audit["maximum_candidate_to_selected_same_tunnel_arc_distance_m"],10.0)


if __name__=="__main__": unittest.main()
