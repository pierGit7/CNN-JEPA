#!/bin/bash
# Class-split protocol: pretrain on 5 seen classes, evaluate on the 5 unseen ones
# (see pretrain/configs/split5_*_<dataset>_*.yaml). Runs the 6 method x backbone configs one after another, for each
# split seed. DATASET selects the dataset: imagenette (default) or core50.
#
# Usage (from the repo root, e.g. inside a 1-GPU SLURM job):
#   ./run_class_split.sh                                  # split seed 0
#   SEEDS="0 1 2" ./run_class_split.sh data.root=/path    # 3 folds; extra arguments are passed to every run as Hydra overrides
#   DATASET=core50 SEEDS="0 1 2" ./run_class_split.sh     # same on CORe50
set -e

SEEDS="${SEEDS:-0}"
DATASET="${DATASET:-imagenette}"
RUNS=(
    "train_ijepacnn.py    split5_ijepacnn_${DATASET}_simplecnn"
    "train_ijepacnn.py    split5_ijepacnn_${DATASET}_simpleconvnext"
    "train_autoencoder.py split5_ae_${DATASET}_simplecnn"
    "train_autoencoder.py split5_ae_${DATASET}_simpleconvnext"
    "train_supervised.py  split5_supervised_${DATASET}_simplecnn"
    "train_supervised.py  split5_supervised_${DATASET}_simpleconvnext"
)

for seed in $SEEDS; do
    for run in "${RUNS[@]}"; do
        read -r script config <<< "$run"
        echo "=== $config, split seed $seed ==="
        PYTHONPATH=. python "pretrain/$script" --config-name "$config.yaml" data.class_split.seed="$seed" "$@"
    done
done
