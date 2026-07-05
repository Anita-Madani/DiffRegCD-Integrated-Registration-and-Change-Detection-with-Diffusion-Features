#!/usr/bin/env bash
# Download pretrained weights for DiffRegCD from the Hugging Face Hub.
#
#   1) the frozen DDPM backbone (shared by all datasets), and
#   2) the trained DiffRegCD registration + change-detection heads per dataset.
#
# Weights are placed under ./checkpoints/ to match the paths in config/*.json.
#
# Usage:
#   bash scripts/download_weights.sh              # backbone + all datasets
#   bash scripts/download_weights.sh levir        # backbone + one dataset
#
# Requires the Hugging Face CLI:  pip install -U huggingface_hub
set -euo pipefail

HF_REPO="${DIFFREGCD_HF_REPO:-Anita1379m/DiffRegCD}"
DST="checkpoints"
mkdir -p "$DST"

dl() {  # dl <path-in-repo>
  echo ">> $1"
  hf download "$HF_REPO" "$1" --local-dir "$DST"
}

# 1) Frozen DDPM backbone (pretrained diffusion feature extractor)
dl "I200000_E3_gen.pth" || true

# 2) Per-dataset trained heads
DATASETS=("${@:-levir dsifn whu sysu vl_cmu_cd}")
for ds in ${DATASETS[@]}; do
  echo "== $ds =="
  dl "$ds/best_reg_model_gen.pth" || true
  dl "$ds/best_cd_model_gen.pth"  || true
done

echo "Done. Weights are under ./$DST/"
