from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DeepLabExperimentContracts(unittest.TestCase):
    def test_official_recipe(self):
        script_path = ROOT / "src/train_deeplabv3plus_vainf_official.py"
        submit_path = (
            ROOT
            / "submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh"
        )
        self.assertTrue(script_path.is_file(), script_path)
        self.assertTrue(submit_path.is_file(), submit_path)
        script = script_path.read_text(encoding="utf-8")
        submit = submit_path.read_text(encoding="utf-8")

        for expected in (
            "deeplabv3plus_vainf_r50_os16_official_30k_bs4",
            'train_augmentation="deeplab_official"',
            "deeplab_cross_entropy",
            "IterationPolynomialLR",
            "power=0.9",
            'selection_metric="mean_iou"',
            "maximum validation mean IoU",
            "VainFDeepLabV3PlusResNet50",
        ):
            self.assertIn(expected, script)

        for expected in (
            "conda activate caries-train",
            "conda activate caries-baselines",
            "prepare_official_baseline_weights.py",
            "--model deeplab",
            "--verify-only",
            "validate_deeplabv3plus_vainf.py --device cuda",
            "--total-iters 30000",
            "--val-interval-iters 1000",
            "--batch-size 4",
            "--lr 0.0025",
            "--seed 1",
        ):
            self.assertIn(expected, submit)

    def test_medical_recipe(self):
        script_path = ROOT / "src/train_deeplabv3plus_vainf_medical_e260.py"
        submit_path = (
            ROOT
            / "submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh"
        )
        self.assertTrue(script_path.is_file(), script_path)
        self.assertTrue(submit_path.is_file(), submit_path)
        script = script_path.read_text(encoding="utf-8")
        submit = submit_path.read_text(encoding="utf-8")

        for expected in (
            "deeplabv3plus_vainf_r50_os16_medical_auglite_e260_bs6",
            "VainFDeepLabV3PlusResNet50",
            "make_auglite_dataset_bundle",
            "deeplab_ce_foreground_dice",
            "torch.optim.AdamW",
            '"protocol_family": "CariXray-adapted"',
            'selection_metric="dice"',
            '"scheduler_name": "constant"',
        ):
            self.assertIn(expected, script)

        for expected in (
            "conda activate caries-train",
            "conda activate caries-baselines",
            "prepare_official_baseline_weights.py",
            "--model deeplab",
            "--verify-only",
            "validate_deeplabv3plus_vainf.py --device cuda",
            "--epochs 260",
            "--batch-size 6",
            "--lr 3e-4",
            "--weight-decay 1e-4",
            "--seed 42",
        ):
            self.assertIn(expected, submit)


if __name__ == "__main__":
    unittest.main()
