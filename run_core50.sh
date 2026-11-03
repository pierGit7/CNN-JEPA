#!/bin/bash
# Cross-dataset protocol: pretrain on Imagenette, evaluate with the online benchmark on CORe50
# (see pretrain/configs/core50_*.yaml). Runs the 6 method x backbone configs one after another.
#
# Usage (from the repo root, e.g. inside a 1-GPU SLURM job):
#   ./run_core50.sh                                       # all 6 runs
#   ./run_core50.sh data.root=/path data.core50_frame_step=5   # extra arguments are passed to every run as Hydra overrides
set -e

RUNS=(
    "train_ijepacnn.py    core50_ijepacnn_imagenette_simplecnn"
    "train_ijepacnn.py    core50_ijepacnn_imagenette_simpleconvnext"
    "train_autoencoder.py core50_ae_imagenette_simplecnn"
    "train_autoencoder.py core50_ae_imagenette_simpleconvnext"
    "train_supervised.py  core50_supervised_imagenette_simplecnn"
    "train_supervised.py  core50_supervised_imagenette_simpleconvnext"
)

for run in "${RUNS[@]}"; do
    read -r script config <<< "$run"
    echo "=== $config ==="
    PYTHONPATH=. python "pretrain/$script" --config-name "$config.yaml" "$@"
done
