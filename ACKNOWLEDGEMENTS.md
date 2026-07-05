# Acknowledgements

DiffRegCD builds on several excellent open-source projects. We thank their authors.

- **DDPM-CD** — Denoising Diffusion Probabilistic Models as Feature Extractors for
  Change Detection (WACV 2025), Bandara et al.
  <https://github.com/wgcban/ddpm-cd>
  The frozen diffusion feature extractor (`model/ddpm_modules`, `model/sr3_modules`,
  `model/model.py`) and the change-detection head (`model/cd_modules`) derive from this
  repository. DiffRegCD is licensed under the same MIT terms.

- **RoMa** — Robust Dense Feature Matching (CVPR 2024), Edstedt et al.
  <https://github.com/Parskatt/RoMa>
  The classification-based correspondence formulation and the Gaussian-Process /
  soft-argmax utilities in `registration/` are inspired by RoMa.

- **DINOv2** — Meta AI. The transformer attention blocks under
  `registration/transformer/layers/` originate from the DINOv2 codebase.

Datasets used for training and evaluation are the property of their respective
authors; please cite them and follow their licenses (see the README).
