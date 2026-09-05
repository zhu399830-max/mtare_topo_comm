import unittest
import numpy as np
from mtare_topo.evaluation.phase3_semantic_metrics import confusion_matrix,per_class_scores,same_cluster_cosine,circular_direction_equivariance_error


class Phase3SemanticMetricsTest(unittest.TestCase):
    def test_classification_scores(self):
        matrix=confusion_matrix(np.array([0,1,2,2]),np.array([0,1,1,2]),3);scores=per_class_scores(matrix)
        self.assertEqual(matrix.tolist(),[[1,0,0],[0,1,0],[0,1,1]])
        self.assertAlmostEqual(scores["accuracy"],.75)

    def test_same_cluster_cosine(self):
        result=same_cluster_cosine(np.array([[1,0],[1,0],[0,1],[0,1]],float),["a","a","b","b"])
        self.assertEqual(result["pairs"],2);self.assertAlmostEqual(result["mean"],1.0)

    def test_circular_equivariance(self):
        values=np.arange(720,dtype=float)[None,:];rotated=np.roll(values,17,axis=-1)
        result=circular_direction_equivariance_error(values,rotated,17)
        self.assertEqual(result["maximum_absolute_logit_error"],0.0)


if __name__=="__main__":unittest.main()
