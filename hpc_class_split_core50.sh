#!/bin/bash
# LSF job array: class-split protocol on CORe50 (pretrain on 5 seen categories, evaluate on the 5 unseen ones).
# 6 configs (method x backbone) x 3 split seeds = 18 jobs, one GPU each.
#   mkdir -p logs && bsub < hpc_class_split_core50.sh   (run from the repo root; LSF does not create logs/)
# Job i (1..18): config = RUNS[(i-1) % 6], split seed = (i-1) / 6, i.e. jobs 1-6 seed 0, 7-12 seed 1, 13-18 seed 2.
#
# Add the rest of your usual #BSUB header (queue, GPU, walltime, memory, cores) below.
#BSUB -J "split5_core50[1-18]"
#BSUB -o logs/split5_core50_%J_%I.out
#BSUB -e logs/split5_core50_%J_%I.err

set -e

REPO=/zhome/48/a/213648/work/CNN-JEPA-core50
VENV=/zhome/48/a/213648/work/CNN-JEPA/.venv
DATA_ROOT=/zhome/48/a/213648/work/CNN-JEPA/datasets   # contains imagenette/ and core50/

RUNS=(
    "train_ijepacnn.py    split5_ijepacnn_core50_simplecnn"
    "train_ijepacnn.py    split5_ijepacnn_core50_simpleconvnext"
    "train_autoencoder.py split5_ae_core50_simplecnn"
    "train_autoencoder.py split5_ae_core50_simpleconvnext"
    "train_supervised.py  split5_supervised_core50_simplecnn"
    "train_supervised.py  split5_supervised_core50_simpleconvnext"
)
idx=$((LSB_JOBINDEX - 1))
read -r script config <<< "${RUNS[$((idx % ${#RUNS[@]}))]}"
seed=$((idx / ${#RUNS[@]}))

cd "$REPO"
source "$VENV/bin/activate"
export WANDB_MODE=offline   # no W&B login on the cluster; sync later with: wandb sync artifacts/wandb/offline-run-*

nvidia-smi
echo "=== [$LSB_JOBINDEX] $config, split seed $seed on $(hostname) ==="
PYTHONPATH=. python "pretrain/$script" --config-name "$config.yaml" \
    data.root="$DATA_ROOT" data.num_workers=8 data.class_split.seed="$seed"
