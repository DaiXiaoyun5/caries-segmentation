# U-Mamba Offline Setup and DeepLabV3+ Baselines Design

## Decision

Implement one reliability fix and two new, separately named comparison
experiments:

1. make the existing official U-Mamba environment setup consume the two
   user-uploaded CUDA wheels without contacting GitHub;
2. add a VainF-faithful DeepLabV3+ ResNet50 baseline using the corrected
   output-stride-16 ASPP rates and the reference SGD/poly recipe;
3. add a CariXray-adapted training variant that keeps exactly the same
   DeepLabV3+ network and changes only the data and optimization recipe.

The existing DeepLabV3+ result and scripts remain in the repository for
auditability. They will be documented as superseded, not deleted or silently
overwritten.

## Motivation

The existing `deeplabv3plus_r50_os16_30k_bs4` run reached only 0.7175 global
test Dice and remained underfit at the end of 30,000 iterations. Its declared
OS16 model uses ASPP rates `12/24/36`; the VainF ResNet implementation uses
`6/12/18` for OS16 and reserves `12/24/36` for OS8. In addition, the existing
SMP decoder uses separable decoder convolutions and is not identical to the
default VainF implementation cited by the experiment.

The paired DeepLab experiments separate two questions:

- Can the correctly specified reference DeepLabV3+ recipe solve CariXray?
- How much of the previous deficit is caused by a natural-image training
  recipe that is poorly matched to sparse, weak-boundary caries masks?

The U-Mamba logs expose a separate infrastructure failure: setup stopped while
downloading `causal-conv1d` through an unavailable proxy, so preprocessing and
training never started. This is solved at the dependency boundary rather than
by changing the U-Mamba model.

## Scope and constraints

- Do not alter B2, B30, or any completed experiment.
- Do not delete the old SMP DeepLabV3+ scripts or result JSON files.
- Do not create smoke-test submit jobs.
- Do not commit wheel files, checkpoints, cached ImageNet weights, datasets,
  predictions, or large run artifacts.
- Both new DeepLab experiments use the existing train/validation/test split
  and export the project's standard metrics and evidence files.
- Every documented supercomputer command starts in
  `/share/home/u2515283028/caries_project`, loads Anaconda, initializes Conda,
  and activates the required environment.

## U-Mamba offline dependency design

The setup script will require these exact Linux wheels under
`external_assets/umamba_wheels/`:

- `causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`
- `mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl`

Before invoking pip, the script will print the resolved path and reject a
missing, empty, or differently named file with an actionable error. It will
install both files with `python -m pip install --no-deps <local-path>` and will
not use a network URL. Existing post-install checks remain responsible for
verifying the package versions, importing both CUDA extensions, importing the
official trainer, and confirming `pip check` succeeds.

The U-Mamba preprocessing and GPU submit scripts will replace silent bare
`test` commands with named preflight checks. A missing source file, raw
dataset, preprocessed plan, or fixed split must be printed before exit. The GPU
job remains blocked until the CPU preparation job has produced both
`nnUNetPlans_2d` and `splits_final.json`.

## DeepLabV3+ source and network

A small attributed third-party package will contain the unmodified network
components required by VainF's `deeplabv3plus_resnet50`. The vendored snapshot
will include its upstream URLs, license, and source hashes. Vendoring source is
necessary because compute nodes cannot reliably access GitHub; no pretrained
weights are vendored.

Both experiments use exactly the same network:

- encoder: ResNet50 with ImageNet initialization;
- task head: two classes, background and caries;
- output stride: 16;
- ASPP dilation rates: `6/12/18`;
- ASPP channels: 256;
- low-level projection: 48 channels;
- default VainF decoder convolutions, without enabling the optional
  `--separable_conv` conversion;
- BatchNorm momentum: 0.01;
- full-resolution logits returned to the shared evaluator.

The wrapper loads the existing cached torchvision-compatible ResNet50
ImageNet state dictionary from `.cache/torch/hub/checkpoints`. Asset
verification fails before allocating a training job if the cache is absent or
cannot load into the backbone.

## Experiment 1: strict VainF recipe

Files and naming:

- training entry point:
  `src/train_deeplabv3plus_vainf_official.py`;
- submit job:
  `submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh`;
- run name:
  `deeplabv3plus_vainf_r50_os16_official_30k_bs4`.

Recipe:

- input: 512 by 512 RGB;
- augmentation: random scale `0.5-2.0`, random crop/pad to 512, horizontal
  flip;
- loss: unweighted two-class cross entropy;
- optimizer: SGD, momentum 0.9, weight decay `1e-4`;
- classifier learning rate: `0.0025`;
- backbone learning rate: `0.00025`;
- schedule: iteration-wise polynomial decay, power 0.9;
- budget: 30,000 optimizer updates;
- batch size: 4, dropping the incomplete training batch;
- validation interval: 1,000 updates;
- seed: 1;
- checkpoint selection: maximum validation mean IoU.

This experiment is the paper's reference-recipe DeepLabV3+ result. Its only
dataset adaptations are two output classes, the CariXray split, and 512-pixel
input/crop size.

## Experiment 2: CariXray medical adaptation

Files and naming:

- training entry point:
  `src/train_deeplabv3plus_vainf_medical_e260.py`;
- submit job:
  `submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh`;
- run name:
  `deeplabv3plus_vainf_r50_os16_medical_auglite_e260_bs6`.

Recipe:

- input: 512 by 512 RGB;
- augmentation: the existing whole-image AugLite profile, without random
  crops that can remove sparse caries pixels;
- loss: equal sum of two-class cross entropy and foreground soft Dice loss;
- optimizer: AdamW, learning rate `3e-4`, weight decay `1e-4`;
- schedule: constant learning rate;
- budget: 260 epochs;
- batch size: 6;
- seed: 42;
- checkpoint selection: maximum validation foreground Dice.

This run is explicitly labeled task-adapted and is not described as an
official DeepLab recipe. Its purpose is to evaluate the same external network
under the proven CariXray training regime, not to introduce another proposed
model module.

## Shared data flow and outputs

Each training entry point constructs its own recipe but delegates split
loading, global confusion-matrix metrics, checkpoint restoration, per-case
metrics, and result export to the existing official-baseline runner.

Both runs must save:

- `config.json`;
- `history.csv`;
- `best_val_metrics.json`;
- `best_train_metrics.json`;
- `test_metrics.json`;
- `summary_metrics.json`;
- `run_status.json`;
- `per_case_metrics.csv` and `per_case_metrics_summary.json`;
- a best checkpoint and test preview under the ignored run directory.

Final metrics are evaluated at the shared 512 by 512 resolution. The strict
run uses two-class argmax; the medical run uses the same two-class argmax, so
thresholding differences cannot explain their comparison.

## Error handling

- U-Mamba setup fails before pip with the exact missing wheel path.
- U-Mamba jobs print every failed prerequisite instead of exiting silently.
- DeepLab asset verification reports the expected cache directory and does
  not attempt compute-node downloads.
- DeepLab wrappers validate input/output shape, output channel count, ASPP
  rates, optimizer parameter-group learning rates, and finite loss before the
  long training loop.
- Existing failure-report JSON behavior remains enabled for both DeepLab runs.

## Verification strategy

No smoke-training job is created. Local and static checks will cover:

- shell syntax for setup and submit scripts;
- Python compilation for all new and modified Python files;
- exact U-Mamba wheel filenames and absence of GitHub wheel URLs in the
  offline setup path;
- VainF OS16 ASPP rates `6/12/18` and output shape on a small CPU tensor;
- identical architecture state-dict keys for the official and medical model
  builders;
- strict optimizer backbone/classifier learning-rate ratio;
- medical CE-plus-Dice loss behavior on controlled logits;
- submit-job arguments, run names, environments, and required output paths;
- preservation of all old DeepLab and B2/B30 files.

GPU execution remains the user's supercomputer step. The handoff document will
give setup, preprocessing, U-Mamba training, and both DeepLab `sbatch` commands
in dependency order.

## References

- DeepLabV3+ paper: <https://arxiv.org/abs/1802.02611>
- VainF implementation: <https://github.com/VainF/DeepLabV3Plus-Pytorch>
- VainF output-stride/rate mapping:
  <https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/network/modeling.py>
- VainF SGD/poly/CE recipe:
  <https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/main.py>
