import unittest

from mtare_topo.data.phase3_multitask_dataset import validate_phase3_manifest


class Phase3ManifestTest(unittest.TestCase):
    def test_five_frame_multitask_contract(self):
        cluster={"cluster_id":"c0","parent_id":"p0","split":"train","primary_role":"junction","near_junction":True,"near_terminal":False,"tunnel_id":1}
        frames=[]
        for index in range(5):
            frames.append({**cluster,"frame_id":f"f{index}","branch_count":3,"headings_robot_deg":[0.0,90.0,180.0]})
        result=validate_phase3_manifest(frames,[cluster])
        self.assertTrue(result["passed"])
        self.assertEqual(result["frames"],5)

    def test_heading_count_mismatch_rejected(self):
        cluster={"cluster_id":"c0","parent_id":"p0","split":"train","primary_role":"junction","near_junction":True,"near_terminal":False,"tunnel_id":1}
        frame={**cluster,"frame_id":"f0","branch_count":3,"headings_robot_deg":[0.0]}
        with self.assertRaises(ValueError): validate_phase3_manifest([frame],[cluster])


if __name__=="__main__": unittest.main()
