# U-Mamba and VainF DeepLabV3+ follow-up

This handoff covers the two U-Mamba offline CUDA wheels and the two new
DeepLabV3+ protocols. No smoke job is required. Pull the branch/PR on the
supercomputer before using these commands.

## 1. Put the U-Mamba files in their expected locations

If the zip and wheels were uploaded to the project root, run:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

mkdir -p external_assets/umamba_wheels
mv -n U-Mamba-main.zip external_assets/U-Mamba-main.zip
mv -n causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl external_assets/umamba_wheels/
mv -n mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl external_assets/umamba_wheels/
```

The setup script deliberately requires these exact CPython 3.10, CUDA 11.8,
torch 2.0 and old-ABI wheel names. Wheels, archives, extracted repositories,
environments, datasets and checkpoints remain ignored by Git.

## 2. Build the isolated U-Mamba environment

Run this on a login node. `torch.cuda.is_available() == False` there is not a
failure; the submitted GPU job performs the CUDA forward check.

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

bash scripts/setup/setup_umamba_official_env.sh
```

Do not submit training until the setup ends with
`U-Mamba official environment is ready: umamba-caries`.

## 3. Prepare U-Mamba data, then train fold 0

Submit preprocessing first:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_umamba_prepare_caries_2d.sh
```

After that Slurm job finishes successfully, submit training:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_umamba_bot_official_2d_fold0.sh
```

The exported small evidence files appear under
`runs/umamba_bot_official_2d_fold0/`; the large official checkpoint remains in
the isolated U-Mamba results directory.

## 4. Prepare the VainF DeepLabV3+ ResNet50 cache

Run this once on a login node. It only handles DeepLab, so an unrelated
SegFormer local-config hash mismatch cannot block it.

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
conda activate caries-baselines

export PYTHONNOUSERSITE=1
export TORCH_HOME=/share/home/u2515283028/caries_project/.cache/torch
python scripts/setup/prepare_official_baseline_weights.py \
  --model deeplab
python scripts/setup/prepare_official_baseline_weights.py \
  --model deeplab \
  --verify-only
```

Success must print `DeepLabV3+ ResNet50 cache: OK` twice. The manifest must
record VainF, output stride 16 and ASPP rates `6/12/18`.

## 5. Submit both DeepLabV3+ protocols

Strict VainF recipe:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh
```

CariXray-adapted medical recipe:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh
```

Both jobs use exactly the same VainF ResNet50/OS16/ASPP `6/12/18` network.
The strict run tests the recognized reference recipe; the medical run tests
whether task-appropriate optimization explains the earlier poor result.

Expected result directories:

```text
runs/deeplabv3plus_vainf_r50_os16_official_30k_bs4/
runs/deeplabv3plus_vainf_r50_os16_medical_auglite_e260_bs6/
```

Each contains `config.json`, `history.csv`, `best_train_metrics.json`,
`best_val_metrics.json`, `test_metrics.json`, `summary_metrics.json`, per-case
metrics, status records and predictions.

The historical run `deeplabv3plus_r50_os16_30k_bs4` used the SMP implementation
with ASPP `12/24/36`. It is superseded: keep it for auditability, but do not use
it as the DeepLabV3+ value in the final paper table.
