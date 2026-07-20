import ast
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = ROOT / "src/deeplabv3plus_vainf_common.py"

try:
    import torch
except ImportError:
    torch = None


class DeepLabV3PlusVainFStaticContracts(unittest.TestCase):
    def test_public_interfaces_and_approved_constants_exist(self):
        self.assertTrue(COMMON_PATH.is_file(), COMMON_PATH)
        source = COMMON_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        classes = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
        self.assertIn("VainFDeepLabV3PlusResNet50", classes)
        self.assertTrue(
            {
                "find_resnet50_checkpoint",
                "build_vainf_deeplabv3plus",
                "build_vainf_official_optimizer",
                "deeplab_cross_entropy",
                "deeplab_ce_foreground_dice",
                "make_auglite_dataset_bundle",
            }.issubset(functions)
        )
        self.assertIn("output_stride = 16", source)
        self.assertIn("aspp_atrous_rates = (6, 12, 18)", source)
        self.assertIn("pretrained_backbone=False", source)
        self.assertIn("fc.weight", source)
        self.assertIn("fc.bias", source)


@unittest.skipIf(torch is None, "torch is unavailable in the local static-test Python")
class DeepLabV3PlusVainFFunctionalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "src"))
        import deeplabv3plus_vainf_common as common

        cls.common = common

    def test_model_shape_rates_and_recipe_builders(self):
        official = self.common.build_vainf_deeplabv3plus(pretrained=False).eval()
        medical = self.common.build_vainf_deeplabv3plus(pretrained=False).eval()
        self.assertEqual(set(official.state_dict()), set(medical.state_dict()))
        rates = tuple(
            branch[0].dilation[0]
            for branch in official.network.classifier.aspp.convs[1:4]
        )
        self.assertEqual(rates, (6, 12, 18))
        with torch.no_grad():
            output = official(torch.zeros(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 2, 64, 64))

        optimizer = self.common.build_vainf_official_optimizer(
            official,
            classifier_lr=0.0025,
            weight_decay=1e-4,
        )
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.00025)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.0025)

    def test_medical_loss_rewards_correct_foreground_logits(self):
        masks = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
        correct = torch.tensor(
            [[[[8.0, -8.0], [-8.0, 8.0]], [[-8.0, 8.0], [8.0, -8.0]]]]
        )
        inverted = -correct
        correct_loss = self.common.deeplab_ce_foreground_dice(correct, masks)
        inverted_loss = self.common.deeplab_ce_foreground_dice(inverted, masks)
        self.assertTrue(torch.isfinite(correct_loss))
        self.assertLess(float(correct_loss), float(inverted_loss))


if __name__ == "__main__":
    unittest.main()
