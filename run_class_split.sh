#!/bin/bash
# Class-split protocol: pretrain on 5 seen Imagenette classes, evaluate on the 5 unseen ones
# (see pretrain/configs/split5_*.yaml). Runs the 6 method x backbone configs one after another, for each split seed.
#
# Usage (from the repo root, e.g. inside a 1-GPU SLURM job):
#   ./run_class_split.sh                                  # split seed 0
#   SEEDS="0 1 2" ./run_class_split.sh data.root=/path    # 3 folds; extra arguments are passed to every run as Hydra overrides
set -e

SEEDS="${SEEDS:-0}"
RUNS=(
    "train_ijepacnn.py    split5_ijepacnn_imagenette_simplecnn"
    "train_ijepacnn.py    split5_ijepacnn_imagenette_simpleconvnext"
    "train_autoencoder.py split5_ae_imagenette_simplecnn"
    "train_autoencoder.py split5_ae_imagenette_simpleconvnext"
    "train_supervised.py  split5_supervised_imagenette_simplecnn"
    "train_supervised.py  split5_supervised_imagenette_simpleconvnext"
)

for seed in $SEEDS; do
    for run in "${RUNS[@]}"; do
        read -r script config <<< "$run"
        echo "=== $config, split seed $seed ==="
        PYTHONPATH=. python "pretrain/$script" --config-name "$config.yaml" data.class_split.seed="$seed" "$@"
    done
done
