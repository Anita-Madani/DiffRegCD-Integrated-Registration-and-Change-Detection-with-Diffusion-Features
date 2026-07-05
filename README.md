<div align="center">

# DiffRegCD: Integrated Registration and Change Detection with Diffusion Features

Seyedehanita Madani · Rama Chellappa · Vishal M. Patel

[![arXiv](https://img.shields.io/badge/arXiv-2511.07935-b31b1b.svg)](https://arxiv.org/abs/2511.07935)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
**WACV 2026**

</div>

DiffRegCD unifies **dense image registration** and **change detection** in a *single* model.
Instead of registering a pair of images and then detecting changes as two separate stages,
DiffRegCD does both jointly, which makes it robust to the parallax, viewpoint shifts, and
temporal misalignment found in real bi-temporal imagery.

**Key ideas**

- **Frozen diffusion features.** Multi-scale, multi-timestep features from a *frozen* pretrained
  denoising diffusion model (DDPM) serve as the backbone, giving robustness to illumination and
  viewpoint change without any backbone fine-tuning.
- **Correspondence as Gaussian-smoothed classification.** Flow is predicted as a classification
  over a discretized displacement grid with a Gaussian soft label, then converted to sub-pixel
  flow via soft-argmax — more stable to train than direct regression.
- **Affine-perturbation supervision.** Controlled affine perturbations of standard CD datasets
  yield paired ground truth for *both* flow and change, so no extra annotation is needed.
- **Joint objective.** The change head operates on the *registered* features:
  `total_loss = flow_loss + λ · change_loss`.

Evaluated on **LEVIR-CD, DSIFN-CD, WHU-CD, SYSU-CD** (aerial) and **VL-CMU-CD** (ground-level).

---

## Repository layout

```
DiffRegCD/
├── main.py                 # single entry point: train (-p train) and test (-p test)
├── config/                 # one JSON per dataset (+ *_test.json for evaluation)
├── model/
│   ├── model.py            # frozen DDPM feature extractor
│   ├── ddpm_modules/  sr3_modules/   # diffusion U-Nets (backbone)
│   ├── cd_model_256.py     # JOINT registration + change model (forward pass + joint loss)
│   ├── cd_modules/         # change-detection head (cd_head_v2)
│   └── networks.py
├── registration/           # flow head: classification decoder + soft-argmax utilities
├── losses/roma_flow_cls_256.py   # Gaussian-smoothed classification flow loss
├── data/CMUCDFlowDataset.py      # loads image pair + change mask + GT flow
├── scripts/
│   ├── prepare_gt_flow.py  # affine-perturbation GT flow generation
│   └── download_weights.sh # fetch checkpoints from Hugging Face
├── core/  misc/            # logging, metrics, utilities
├── requirements.txt  environment.yml
└── ACKNOWLEDGEMENTS.md  CITATION.cff  LICENSE
```

## Installation

```bash
git clone https://github.com/Anita-Madani/DiffRegCD-Integrated-Registration-and-Change-Detection-with-Diffusion-Features.git
cd DiffRegCD-Integrated-Registration-and-Change-Detection-with-Diffusion-Features

# conda (recommended)
conda env create -f environment.yml
conda activate diffregcd

# or pip
pip install -r requirements.txt
```

## Data preparation

Download the change-detection datasets from their original sources:

| Dataset     | Link |
|-------------|------|
| LEVIR-CD    | https://chenhao.in/LEVIR/ |
| DSIFN-CD    | https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images |
| WHU-CD      | https://gpcv.whu.edu.cn/data/building_dataset.html |
| SYSU-CD     | https://github.com/liumency/SYSU-CD |
| VL-CMU-CD   | https://ghsi.github.io/proj/RSS2016.html |

Each dataset must be arranged in the layout expected by `data/CMUCDFlowDataset.py`. All image
tensors are read from the `train/` subfolder and the split is selected by the list files:

```
<dataroot>/
├── list/
│   ├── train.txt        # one base filename (no extension) per line
│   ├── val.txt
│   └── test.txt
└── train/
    ├── t0/   <name>.png  # image B (un-warped reference)
    ├── t1/   <name>.png  # image A (affine-warped view)      <- produced in the next step
    ├── mask/ <name>.png  # binary change label
    └── flow/ <name>.npy  # dense GT flow, shape (2, H, W)     <- produced in the next step
```

Generate the affine-warped view (`t1`) and the dense GT flow (`flow`) from your reference
images (`t0`):

```bash
python scripts/prepare_gt_flow.py \
    --src        LEVIR-CD256/train/t0 \
    --warped-out LEVIR-CD256/train/t1 \
    --flow-out   LEVIR-CD256/train/flow \
    --angle 20 --scale 0.90 1.15 --trans 16 --seed 0
```

Then point the `dataroot` fields in the matching `config/*.json` at your dataset folder.

## Pretrained weights

Weights live on the Hugging Face Hub and download into `./checkpoints/` (the paths already
referenced by the configs):

```bash
pip install "huggingface_hub[cli]"
export DIFFREGCD_HF_REPO=<HF_USER>/DiffRegCD      # set once weights are uploaded
bash scripts/download_weights.sh                  # backbone + all datasets
```

This fetches:

- the **frozen DDPM backbone** → `checkpoints/I200000_E3_gen.pth` (used by every config via
  `path.resume_state`), and
- the **trained DiffRegCD heads** → `checkpoints/<dataset>/best_reg_model_*.pth` and
  `best_cd_model_*.pth` (used by the `*_test.json` configs).

## Training

```bash
python main.py -c config/levir.json -p train
python main.py -c config/dsifn.json -p train
python main.py -c config/whu.json   -p train
python main.py -c config/sysu.json  -p train
python main.py -c config/vl_cmu_cd.json -p train
```

Select GPUs with `-gpu 0,1` or edit `"gpu_ids"` in the config. Training logs to Weights &
Biases (project `DiffRegCD`); run `wandb offline` to disable.

## Evaluation

Use the matching `*_test.json` config with `-p test` (it loads the trained checkpoints from
`checkpoints/<dataset>/`):

```bash
python main.py -c config/levir_test.json     -p test
python main.py -c config/dsifn_test.json     -p test
python main.py -c config/whu_test.json       -p test
python main.py -c config/sysu_test.json      -p test
python main.py -c config/vl_cmu_cd_test.json -p test
```

## Results

See the [paper](https://arxiv.org/abs/2511.07935) for the full registration and change-detection
metrics on all five benchmarks.

## Citation

```bibtex
@article{madani2025diffregcd,
  title   = {DiffRegCD: Integrated Registration and Change Detection with Diffusion Features},
  author  = {Madani, Seyedehanita and Chellappa, Rama and Patel, Vishal M.},
  journal = {arXiv preprint arXiv:2511.07935},
  year    = {2025}
}
```

## Acknowledgements

DiffRegCD builds on **DDPM-CD**, **RoMa**, and **DINOv2** — see
[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md). Released under the [MIT License](LICENSE).
