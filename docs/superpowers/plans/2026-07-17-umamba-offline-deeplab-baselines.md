# U-Mamba Offline Setup and DeepLabV3+ Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make U-Mamba install reproducibly from the uploaded local CUDA wheels and add separately runnable strict-VainF and CariXray-adapted DeepLabV3+ ResNet50 experiments.

**Architecture:** Keep U-Mamba model code unchanged and repair only its dependency/preflight boundary. Vendor the attributed VainF network source, wrap one shared ResNet50-OS16 model builder, and expose two independent training entry points whose only difference is the approved training recipe. Reuse `official_baseline_common.py` for split handling, metrics, checkpoints, and evidence export.

**Tech Stack:** Bash/Slurm, Python 3.10, PyTorch 2.0.1, torchvision 0.15.2, VainF DeepLabV3+ source, Conda, `unittest` static contracts.

## Global Constraints

- Preserve B2, B30, completed experiments, and the old SMP DeepLab files.
- Do not create smoke-test submit scripts.
- Do not commit wheels, datasets, checkpoints, caches, predictions, or large run artifacts.
- Strict DeepLab: ResNet50 ImageNet, OS16, ASPP `6/12/18`, two-class CE, SGD/poly, 30,000 updates, batch 4, seed 1.
- Medical DeepLab: identical network, whole-image AugLite, CE plus foreground Dice, AdamW `3e-4` constant, 260 epochs, batch 6, seed 42.
- Every user-facing supercomputer command includes the complete project-directory and Conda activation preamble.
- Both experiments retain the standard project result files and 512-by-512 final metric evaluation.

---

### Task 1: U-Mamba offline wheel and preflight contracts

**Files:**
- Create: `tests/test_umamba_offline_setup.py`
- Modify: `scripts/setup/setup_umamba_official_env.sh`
- Modify: `submit/submitjob_umamba_prepare_caries_2d.sh`
- Modify: `submit/submitjob_umamba_bot_official_2d_fold0.sh`

**Interfaces:**
- Consumes: wheels in `external_assets/umamba_wheels/` with the two approved exact filenames.
- Produces: deterministic local-wheel installation and `require_file`/`require_dir` diagnostics before U-Mamba preprocessing or training.

- [ ] **Step 1: Write failing static contract tests**

Create `tests/test_umamba_offline_setup.py` using `unittest`. It must assert:

```python
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UMambaOfflineSetupTests(unittest.TestCase):
    def test_setup_uses_exact_local_wheels_without_github_release_urls(self):
        text = (ROOT / "scripts/setup/setup_umamba_official_env.sh").read_text()
        self.assertIn("external_assets/umamba_wheels", text)
        self.assertIn("causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl", text)
        self.assertIn("mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl", text)
        self.assertNotIn("github.com/Dao-AILab/causal-conv1d/releases", text)
        self.assertNotIn("github.com/state-spaces/mamba/releases", text)

    def test_jobs_report_named_prerequisite_failures(self):
        for relative in (
            "submit/submitjob_umamba_prepare_caries_2d.sh",
            "submit/submitjob_umamba_bot_official_2d_fold0.sh",
        ):
            text = (ROOT / relative).read_text()
            self.assertIn("require_file", text)
            self.assertIn("require_dir", text)
            self.assertIn("Missing required", text)
```

- [ ] **Step 2: Run tests and confirm the expected failure**

Run: `py -3 -m unittest tests.test_umamba_offline_setup -v`

Expected: FAIL because the setup still contains GitHub release URLs and submit jobs still use silent bare `test` commands.

- [ ] **Step 3: Implement the smallest offline setup change**

In `setup_umamba_official_env.sh`, define the wheel directory and exact paths, reject missing/empty files, print each selected asset, and install with:

```bash
python -m pip install --no-deps "${CAUSAL_CONV1D_WHEEL}"
python -m pip install --no-deps "${MAMBA_SSM_WHEEL}"
```

Replace each pre-echo bare prerequisite check in the two submit scripts with helpers whose failure format is `Missing required file: ${required_path}` or `Missing required directory: ${required_path}`.

- [ ] **Step 4: Re-run contract and shell syntax tests**

Run:

```powershell
py -3 -m unittest tests.test_umamba_offline_setup -v
& 'C:\Program Files\Git\bin\bash.exe' -n scripts/setup/setup_umamba_official_env.sh
& 'C:\Program Files\Git\bin\bash.exe' -n submit/submitjob_umamba_prepare_caries_2d.sh
& 'C:\Program Files\Git\bin\bash.exe' -n submit/submitjob_umamba_bot_official_2d_fold0.sh
```

Expected: 2 tests PASS and all Bash commands exit 0.

- [ ] **Step 5: Commit the U-Mamba fix**

```bash
git add tests/test_umamba_offline_setup.py scripts/setup/setup_umamba_official_env.sh submit/submitjob_umamba_prepare_caries_2d.sh submit/submitjob_umamba_bot_official_2d_fold0.sh
git commit -m "Support offline U-Mamba CUDA wheels"
```

### Task 2: Vendor the attributed VainF network snapshot

**Files:**
- Create: `third_party/__init__.py`
- Create: `third_party/vainf_deeplabv3plus/__init__.py`
- Create: `third_party/vainf_deeplabv3plus/LICENSE`
- Create: `third_party/vainf_deeplabv3plus/NOTICE.md`
- Create: `third_party/vainf_deeplabv3plus/source_manifest.json`
- Create: `third_party/vainf_deeplabv3plus/network/__init__.py`
- Create: `third_party/vainf_deeplabv3plus/network/_deeplab.py`
- Create: `third_party/vainf_deeplabv3plus/network/modeling.py`
- Create: `third_party/vainf_deeplabv3plus/network/utils.py`
- Create: `third_party/vainf_deeplabv3plus/network/backbone/__init__.py`
- Create: `third_party/vainf_deeplabv3plus/network/backbone/hrnetv2.py`
- Create: `third_party/vainf_deeplabv3plus/network/backbone/mobilenetv2.py`
- Create: `third_party/vainf_deeplabv3plus/network/backbone/resnet.py`
- Create: `third_party/vainf_deeplabv3plus/network/backbone/xception.py`
- Create: `tests/test_vainf_vendor_snapshot.py`

**Interfaces:**
- Consumes: text files fetched from `VainF/DeepLabV3Plus-Pytorch` branch `master` through the GitHub connector.
- Produces: importable `third_party.vainf_deeplabv3plus.network` plus a local SHA-256 manifest and license/attribution record.

- [ ] **Step 1: Write the failing vendor integrity test**

The test loads `source_manifest.json`, requires the ten upstream source/license files, computes each local SHA-256, and asserts the manifest value. It separately reads `modeling.py` and asserts that the ResNet OS16 branch contains `aspp_dilate = [6, 12, 18]`.

- [ ] **Step 2: Run and confirm failure because the vendor package is absent**

Run: `py -3 -m unittest tests.test_vainf_vendor_snapshot -v`

Expected: FAIL with missing `source_manifest.json`.

- [ ] **Step 3: Fetch and add the exact upstream files**

Use the GitHub connector's file fetch for each listed upstream path. Preserve source contents unchanged. Add package-marker files outside the upstream `network` tree, copy the upstream license, and record every upstream path in `source_manifest.json`. Each record has two concrete strings: `github_blob_sha`, copied from the connector response, and `sha256`, computed from the bytes written locally. The manifest also records repository `https://github.com/VainF/DeepLabV3Plus-Pytorch` and ref `master`.

- [ ] **Step 4: Run integrity and compile checks**

Run:

```powershell
py -3 -m unittest tests.test_vainf_vendor_snapshot -v
py -3 -m compileall -q third_party/vainf_deeplabv3plus
```

Expected: all integrity tests PASS and compilation exits 0 without importing torch.

- [ ] **Step 5: Commit the vendor snapshot**

```bash
git add third_party tests/test_vainf_vendor_snapshot.py
git commit -m "Vendor VainF DeepLabV3+ network source"
```

### Task 3: Shared VainF model, pretrained loader, losses, and runtime validator

**Files:**
- Create: `src/deeplabv3plus_vainf_common.py`
- Create: `scripts/validation/validate_deeplabv3plus_vainf.py`
- Create: `tests/test_deeplabv3plus_vainf_contract.py`

**Interfaces:**
- Produces: `VainFDeepLabV3PlusResNet50`, `build_vainf_deeplabv3plus`, `build_vainf_official_optimizer`, `deeplab_cross_entropy`, `deeplab_ce_foreground_dice`, and `make_auglite_dataset_bundle`.
- Consumes: cached `resnet50-*.pth`, VainF vendored package, `official_baseline_common`, and the existing exact `AugmentedCariesMaskDataset` AugLite implementation.

- [ ] **Step 1: Write failing common-module contracts**

Static tests must assert the approved constants and public function names. When torch is importable, functional tests must additionally build an uninitialized model, verify an input/output spatial match, verify two channels, assert ASPP rates `6/12/18`, compare official/medical model state-dict keys, confirm the official LR ratio is 0.1, and confirm perfect foreground logits have lower medical loss than inverted logits.

- [ ] **Step 2: Run and observe the missing-module failure**

Run: `py -3 -m unittest tests.test_deeplabv3plus_vainf_contract -v`

Expected: FAIL because `src/deeplabv3plus_vainf_common.py` does not exist.

- [ ] **Step 3: Implement the shared module**

Required public interfaces are `find_resnet50_checkpoint(torch_home: Path) -> Path`, `build_vainf_deeplabv3plus(pretrained: bool = True) -> nn.Module`, `build_vainf_official_optimizer(model, classifier_lr: float, weight_decay: float)`, `deeplab_cross_entropy(output, masks)`, `deeplab_ce_foreground_dice(output, masks)`, and `make_auglite_dataset_bundle(image_size: int)`. `VainFDeepLabV3PlusResNet50` exposes class constants `output_stride = 16` and `aspp_atrous_rates = (6, 12, 18)`.

Build the upstream model with `pretrained_backbone=False`, load only the cached ResNet50 backbone keys, allow only the expected unused `fc.weight` and `fc.bias`, and normalize zero-to-one RGB tensors with ImageNet mean/std inside the wrapper. The AugLite bundle must reuse `AugmentedCariesMaskDataset` for training and `CariesOfficialDataset` without augmentation for train-eval, validation, and test.

- [ ] **Step 4: Add the runtime validation command**

`validate_deeplabv3plus_vainf.py` must load cached weights, run a CPU or allocated-GPU forward pass on a small tensor, calculate both losses, inspect LR groups, and print `VainF DeepLabV3+ validation: OK` only after all assertions pass.

- [ ] **Step 5: Run static tests and compilation**

Run:

```powershell
py -3 -m unittest tests.test_deeplabv3plus_vainf_contract -v
py -3 -m compileall -q src/deeplabv3plus_vainf_common.py scripts/validation/validate_deeplabv3plus_vainf.py
```

Expected locally: static tests PASS; torch-dependent tests are explicitly skipped if torch is unavailable; compilation exits 0.

- [ ] **Step 6: Commit the shared implementation**

```bash
git add src/deeplabv3plus_vainf_common.py scripts/validation/validate_deeplabv3plus_vainf.py tests/test_deeplabv3plus_vainf_contract.py
git commit -m "Add shared VainF DeepLabV3+ baseline model"
```

### Task 4: Strict VainF training experiment

**Files:**
- Create: `src/train_deeplabv3plus_vainf_official.py`
- Create: `submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh`
- Create: `tests/test_deeplab_experiment_contracts.py`

**Interfaces:**
- Consumes: shared VainF builder, official `deeplab_official` dataset bundle, `IterationPolynomialLR`, and `run_official_experiment`.
- Produces: run `deeplabv3plus_vainf_r50_os16_official_30k_bs4` and all required result artifacts.

- [ ] **Step 1: Write failing strict-recipe contract tests**

Assert the training script and submit job declare `30000`, `1000`, `0.0025`, batch 4, seed 1, CE, poly power 0.9, `deeplab_official`, maximum `mean_iou`, and the exact run name. Assert the submit job activates `caries-baselines`, verifies pretrained assets, and runs the VainF runtime validator before training.

- [ ] **Step 2: Run and confirm missing-file failures**

Run: `py -3 -m unittest tests.test_deeplab_experiment_contracts.DeepLabExperimentContracts.test_official_recipe -v`

Expected: FAIL because the new strict files do not exist.

- [ ] **Step 3: Implement the strict entry point and submit job**

The Python entry point builds the model and optimizer from the shared module, creates the official data bundle, calculates epochs from train split size and 30,000 updates, records complete architecture/recipe/reference metadata, and calls `run_official_experiment` with batch-wise poly scheduling and mIoU checkpoint selection.

- [ ] **Step 4: Run tests, compile, and shell syntax checks**

Run:

```powershell
py -3 -m unittest tests.test_deeplab_experiment_contracts.DeepLabExperimentContracts.test_official_recipe -v
py -3 -m compileall -q src/train_deeplabv3plus_vainf_official.py
& 'C:\Program Files\Git\bin\bash.exe' -n submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh
```

Expected: PASS and exit 0.

- [ ] **Step 5: Commit the strict experiment**

```bash
git add src/train_deeplabv3plus_vainf_official.py submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh tests/test_deeplab_experiment_contracts.py
git commit -m "Add strict VainF DeepLabV3+ experiment"
```

### Task 5: CariXray-adapted DeepLab experiment

**Files:**
- Create: `src/train_deeplabv3plus_vainf_medical_e260.py`
- Create: `submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh`
- Modify: `tests/test_deeplab_experiment_contracts.py`

**Interfaces:**
- Consumes: the identical shared VainF model, exact existing AugLite dataset, medical CE-plus-foreground-Dice loss, and shared runner.
- Produces: run `deeplabv3plus_vainf_r50_os16_medical_auglite_e260_bs6` and all required result artifacts.

- [ ] **Step 1: Add the failing medical-recipe contract**

Assert both files declare 260 epochs, batch 6, AdamW `3e-4`, constant schedule, weight decay `1e-4`, seed 42, AugLite, CE plus foreground Dice, validation Dice selection, and the exact run name. Assert the script imports the same model class as the strict version.

- [ ] **Step 2: Run and confirm missing-file failure**

Run: `py -3 -m unittest tests.test_deeplab_experiment_contracts.DeepLabExperimentContracts.test_medical_recipe -v`

Expected: FAIL because the medical files do not exist.

- [ ] **Step 3: Implement the medical entry point and submit job**

Use one AdamW parameter group over the unchanged network. Call `run_official_experiment` with no scheduler, epoch-based validation, 260 epochs, and maximum Dice checkpoint selection. Metadata must explicitly label the run `CariXray-adapted` and not official.

- [ ] **Step 4: Run tests, compile, and shell syntax checks**

Run:

```powershell
py -3 -m unittest tests.test_deeplab_experiment_contracts -v
py -3 -m compileall -q src/train_deeplabv3plus_vainf_medical_e260.py
& 'C:\Program Files\Git\bin\bash.exe' -n submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh
```

Expected: all experiment contracts PASS and syntax checks exit 0.

- [ ] **Step 5: Commit the medical experiment**

```bash
git add src/train_deeplabv3plus_vainf_medical_e260.py submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh tests/test_deeplab_experiment_contracts.py
git commit -m "Add CariXray-adapted DeepLabV3+ experiment"
```

### Task 6: Offline asset verification and user handoff documentation

**Files:**
- Modify: `scripts/setup/prepare_official_baseline_weights.py`
- Modify: `docs/comparison_baseline_status.md`
- Create: `docs/run_umamba_and_deeplab_followup.md`
- Create: `tests/test_baseline_handoff_contract.py`

**Interfaces:**
- Consumes: shared VainF model/cache loader and all three submit workflows.
- Produces: compute-node-offline ResNet50 verification, corrected baseline status, and commands in setup/preprocess/train dependency order.

- [ ] **Step 1: Write failing asset and documentation contracts**

Assert the asset verifier uses the VainF common loader and records OS16 plus `6/12/18`; assert the status document marks the old SMP `12/24/36` run superseded; assert the handoff contains the exact environment preamble and these commands:

```bash
bash scripts/setup/setup_umamba_official_env.sh
sbatch submit/submitjob_umamba_prepare_caries_2d.sh
sbatch submit/submitjob_umamba_bot_official_2d_fold0.sh
sbatch submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh
sbatch submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh
```

- [ ] **Step 2: Run and confirm failure against old metadata/docs**

Run: `py -3 -m unittest tests.test_baseline_handoff_contract -v`

Expected: FAIL because the verifier and handoff have not been updated.

- [ ] **Step 3: Implement verifier and documentation updates**

Replace the SMP architecture-instantiation check with the shared VainF cached-backbone loader. Keep `--model deeplab --verify-only` network-free. Document that U-Mamba preprocessing must finish before GPU submission and that the two DeepLab runs may be submitted independently after asset verification.

- [ ] **Step 4: Re-run documentation and syntax checks**

Run:

```powershell
py -3 -m unittest tests.test_baseline_handoff_contract -v
py -3 -m compileall -q scripts/setup/prepare_official_baseline_weights.py
git diff --check
```

Expected: PASS and no whitespace errors.

- [ ] **Step 5: Commit handoff updates**

```bash
git add scripts/setup/prepare_official_baseline_weights.py docs/comparison_baseline_status.md docs/run_umamba_and_deeplab_followup.md tests/test_baseline_handoff_contract.py
git commit -m "Document U-Mamba and DeepLab baseline runs"
```

### Task 7: Full verification, review, and GitHub publication

**Files:**
- Verify: all files changed on `agent/umamba-offline-deeplab-baselines`

**Interfaces:**
- Produces: a pushed branch and draft PR against `main` with reproducible validation evidence.

- [ ] **Step 1: Run the complete local verification suite**

Run:

```powershell
py -3 -m unittest discover -s tests -v
py -3 -m compileall -q src scripts third_party tests
& 'C:\Program Files\Git\bin\bash.exe' -n scripts/setup/setup_umamba_official_env.sh
& 'C:\Program Files\Git\bin\bash.exe' -n submit/submitjob_umamba_prepare_caries_2d.sh
& 'C:\Program Files\Git\bin\bash.exe' -n submit/submitjob_umamba_bot_official_2d_fold0.sh
& 'C:\Program Files\Git\bin\bash.exe' -n submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh
& 'C:\Program Files\Git\bin\bash.exe' -n submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh
git diff main...HEAD --check
git status --short
```

Expected: all tests PASS with only documented torch-dependent skips, all syntax checks exit 0, no whitespace errors, and no uncommitted files.

- [ ] **Step 2: Inspect scope and preserve old experiments**

Run:

```bash
git diff --stat main...HEAD
git diff --name-status main...HEAD
git diff --exit-code main -- src/train_deeplabv3plus_official.py submit/submitjob_deeplabv3plus_r50_os16_30k_bs4.sh runs
```

Expected: only intended files are new/modified; old DeepLab scripts and tracked run results are unchanged.

- [ ] **Step 3: Confirm GitHub authentication and push**

Run:

```bash
gh auth status
git push -u origin agent/umamba-offline-deeplab-baselines
```

Expected: authenticated status and successful tracked remote branch.

- [ ] **Step 4: Open a draft pull request**

Create a draft PR targeting `main` with sections for U-Mamba root cause/fix, strict versus medical DeepLab design, validation evidence, and the supercomputer steps still required.
