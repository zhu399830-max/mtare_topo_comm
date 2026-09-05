import unittest
import numpy as np
from mtare_topo.evaluation.phase3_semantic_metrics import decode_direction_components,match_headings
try:
    from train_phase3_multitask_structural_semantics import BlockShuffleSampler
except ModuleNotFoundError:
    BlockShuffleSampler=None


class Phase3TrainingHelpersTest(unittest.TestCase):
    def test_circular_component_decode(self):
        logits=np.full(720,-20.0);logits[[719,0,1]]=20.0
        self.assertEqual(decode_direction_components(logits),[0.0])

    def test_heading_matching(self):
        matched,predicted,truth,errors=match_headings([2.0,92.0],[0.0,90.0],20.0)
        self.assertEqual((matched,predicted,truth),(2,2,2));self.assertEqual(errors,[2.0,2.0])

    @unittest.skipIf(BlockShuffleSampler is None,"PyTorch training environment only")
    def test_block_sampler_exact_and_deterministic(self):
        first=list(BlockShuffleSampler(103,16,7));second=list(BlockShuffleSampler(103,16,7))
        self.assertEqual(first,second);self.assertEqual(sorted(first),list(range(103)))


if __name__=="__main__":unittest.main()
