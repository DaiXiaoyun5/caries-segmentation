from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENV_PREAMBLE = (
    "cd /share/home/u2515283028/caries_project",
    "module load anaconda3/4.12.0",
    'source "$(conda info --base)/etc/profile.d/conda.sh"',
    "conda activate caries-train",
)


class BaselineHandoffContract(unittest.TestCase):
    def test_deeplab_asset_preparation_uses_vainf_network(self):
        script = (
            ROOT / "scripts/setup/prepare_official_baseline_weights.py"
        ).read_text(encoding="utf-8")
        self.assertIn("build_vainf_deeplabv3plus", script)
        self.assertIn('"aspp_atrous_rates": [6, 12, 18]', script)
        self.assertIn('"loader": "VainFDeepLabV3PlusResNet50"', script)
        self.assertNotIn("decoder_atrous_rates=(12, 24, 36)", script)

    def test_followup_guide_contains_reproducible_commands(self):
        guide_path = ROOT / "docs/run_umamba_and_deeplab_followup.md"
        self.assertTrue(guide_path.is_file(), guide_path)
        guide = guide_path.read_text(encoding="utf-8")
        for expected in ENV_PREAMBLE:
            self.assertIn(expected, guide)
        for expected in (
            "bash scripts/setup/setup_umamba_official_env.sh",
            "sbatch submit/submitjob_umamba_prepare_caries_2d.sh",
            "sbatch submit/submitjob_umamba_bot_official_2d_fold0.sh",
            "python scripts/setup/prepare_official_baseline_weights.py",
            "--model deeplab",
            "sbatch submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh",
            "sbatch submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh",
            "deeplabv3plus_r50_os16_30k_bs4",
            "现已被新实验替代",
        ):
            self.assertIn(expected, guide)


if __name__ == "__main__":
    unittest.main()
