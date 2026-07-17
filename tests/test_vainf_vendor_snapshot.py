import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = ROOT / "third_party/vainf_deeplabv3plus"
REQUIRED_UPSTREAM_FILES = (
    "LICENSE",
    "network/__init__.py",
    "network/_deeplab.py",
    "network/modeling.py",
    "network/utils.py",
    "network/backbone/__init__.py",
    "network/backbone/hrnetv2.py",
    "network/backbone/mobilenetv2.py",
    "network/backbone/resnet.py",
    "network/backbone/xception.py",
)


class VainFVendorSnapshotTests(unittest.TestCase):
    def test_manifest_covers_and_hashes_every_upstream_file(self):
        manifest_path = VENDOR_ROOT / "source_manifest.json"
        self.assertTrue(manifest_path.is_file(), manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["repository"],
            "https://github.com/VainF/DeepLabV3Plus-Pytorch",
        )
        self.assertEqual(manifest["ref"], "master")
        self.assertEqual(set(manifest["files"]), set(REQUIRED_UPSTREAM_FILES))

        for relative in REQUIRED_UPSTREAM_FILES:
            with self.subTest(path=relative):
                path = VENDOR_ROOT / relative
                self.assertTrue(path.is_file(), path)
                record = manifest["files"][relative]
                self.assertRegex(record["github_blob_sha"], r"^[0-9a-f]{40}$")
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual, record["sha256"])

    def test_resnet_os16_uses_reference_aspp_rates(self):
        modeling = (VENDOR_ROOT / "network/modeling.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("aspp_dilate = [6, 12, 18]", modeling)
        self.assertIn("aspp_dilate = [12, 24, 36]", modeling)


if __name__ == "__main__":
    unittest.main()
