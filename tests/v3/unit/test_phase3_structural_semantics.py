import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet, multitask_loss


@unittest.skipIf(torch is None, "PyTorch is validated in the isolated Phase-3 training environment")
class Phase3StructuralSemanticNetTest(unittest.TestCase):
    def test_shapes_normalization_and_backward(self):
        model=StructuralSemanticNet();student=torch.rand(2,2,16,720)
        outputs=model(student)
        self.assertEqual(tuple(outputs["direction_logits"].shape),(2,720))
        self.assertEqual(tuple(outputs["count_logits"].shape),(2,6))
        self.assertEqual(tuple(outputs["role_logits"].shape),(2,3))
        self.assertEqual(tuple(outputs["z_role"].shape),(2,128))
        self.assertTrue(torch.allclose(outputs["z_role"].norm(dim=1),torch.ones(2),atol=1e-5))
        losses=multitask_loss(outputs,torch.rand(2,720),torch.tensor([1,2]),torch.tensor([0,1]))
        losses["total"].backward()
        self.assertTrue(torch.isfinite(losses["total"]))

    def test_rejects_forbidden_shape(self):
        with self.assertRaises(ValueError): StructuralSemanticNet()(torch.rand(1,3,16,720))

    def test_azimuth_rotation_equivariance_and_role_invariance(self):
        torch.manual_seed(3);model=StructuralSemanticNet().eval();values=torch.rand(1,2,16,720)
        with torch.no_grad():
            first=model(values);rotated=model(torch.roll(values,12,dims=-1))
        self.assertTrue(torch.allclose(torch.roll(first["direction_logits"],12,dims=-1),rotated["direction_logits"],atol=2e-5,rtol=1e-5))
        self.assertTrue(torch.allclose(first["z_role"],rotated["z_role"],atol=2e-5,rtol=1e-5))


if __name__=="__main__": unittest.main()
