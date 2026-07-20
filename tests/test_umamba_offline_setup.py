from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UMambaOfflineSetupTests(unittest.TestCase):
    def test_setup_uses_exact_local_wheels_without_github_release_urls(self):
        text = (
            ROOT / "scripts/setup/setup_umamba_official_env.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("external_assets/umamba_wheels", text)
        self.assertIn(
            "causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-"
            "cp310-cp310-linux_x86_64.whl",
            text,
        )
        self.assertIn(
            "mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-"
            "cp310-cp310-linux_x86_64.whl",
            text,
        )
        self.assertNotIn(
            "github.com/Dao-AILab/causal-conv1d/releases",
            text,
        )
        self.assertNotIn("github.com/state-spaces/mamba/releases", text)

    def test_jobs_report_named_prerequisite_failures(self):
        for relative in (
            "submit/submitjob_umamba_prepare_caries_2d.sh",
            "submit/submitjob_umamba_bot_official_2d_fold0.sh",
        ):
            with self.subTest(script=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("require_file", text)
                self.assertIn("require_dir", text)
                self.assertIn("Missing required", text)

    def test_jobs_bind_and_verify_isolated_nnunet_paths(self):
        for relative in (
            "submit/submitjob_umamba_prepare_caries_2d.sh",
            "submit/submitjob_umamba_bot_official_2d_fold0.sh",
        ):
            with self.subTest(script=relative):
                text = (ROOT / relative).read_text(encoding="utf-8")
                for variable in (
                    "nnUNet_raw",
                    "nnUNet_preprocessed",
                    "nnUNet_results",
                ):
                    self.assertIn(f"export {variable}=", text)
                self.assertIn("nnU-Net path preflight: OK", text)


if __name__ == "__main__":
    unittest.main()
