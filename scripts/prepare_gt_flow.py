"""
Generate ground-truth optical flow for DiffRegCD by applying controlled affine
perturbations to change-detection images.

For each source image it produces (a) an affine-warped copy and (b) the dense
pixel-wise flow field induced by that affine transform, saved as a ``.npy`` of
shape ``(2, H, W)``.

Mapping to the dataset layout expected by ``data/CMUCDFlowDataset.py``:
    --src        -> <dataroot>/train/t0     (image B, the un-warped reference)
    --warped-out -> <dataroot>/train/t1     (image A, the affine-warped view)
    --flow-out   -> <dataroot>/train/flow   (per-image GT flow, <stem>.npy)

Example:
    python scripts/prepare_gt_flow.py \
        --src        LEVIR-CD256/train/t0 \
        --warped-out LEVIR-CD256/train/t1 \
        --flow-out   LEVIR-CD256/train/flow \
        --angle 20 --scale 0.90 1.15 --trans 16 --seed 0
"""
import os
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def safe_imread(path):
    """Read with OpenCV; fall back to PIL. Return None if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.array(im.convert("RGB"))   # RGB
        return arr[:, :, ::-1].copy()           # to BGR for OpenCV
    except Exception:
        return None


def apply_affine_transform_and_generate_flow(src_dir, warped_dir, flow_dir,
                                             angle_range=(-20, 20),
                                             scale_range=(0.90, 1.15),
                                             translation_range=(-16, 16),
                                             skip_log="skipped_images.txt"):
    Path(warped_dir).mkdir(parents=True, exist_ok=True)
    Path(flow_dir).mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    filenames = sorted(f for f in os.listdir(src_dir) if Path(f).suffix.lower() in exts)

    skipped = []
    for fname in tqdm(filenames, desc="Generating affine-warped images and flow"):
        img = safe_imread(os.path.join(src_dir, fname))
        if img is None:
            skipped.append((fname, "unreadable/libpng error"))
            continue
        try:
            h, w = img.shape[:2]

            # Random affine transform (rotation + scale + translation)
            angle = np.random.uniform(*angle_range)
            scale = np.random.uniform(*scale_range)
            tx = np.random.uniform(*translation_range)
            ty = np.random.uniform(*translation_range)

            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            M[:, 2] += [tx, ty]

            warped_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            out_img_path = os.path.join(warped_dir, Path(fname).with_suffix(".png").name)
            if not cv2.imwrite(out_img_path, warped_img):
                skipped.append((fname, "cv2.imwrite failed"))
                continue

            # Dense flow field induced by the affine transform
            grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
            coords = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3).T  # [3, H*W]
            warped_coords = M @ coords
            flow_x = warped_coords[0].reshape(h, w) - grid_x
            flow_y = warped_coords[1].reshape(h, w) - grid_y
            flow = np.stack([flow_x, flow_y], axis=0).astype(np.float32)  # (2, H, W)

            np.save(os.path.join(flow_dir, f"{Path(fname).stem}.npy"), flow)
        except Exception as e:
            skipped.append((fname, f"exception: {type(e).__name__}: {e}"))
            continue

    if skipped:
        with open(Path(flow_dir) / skip_log, "w") as f:
            for name, reason in skipped:
                f.write(f"{name}\t{reason}\n")
        print(f"[INFO] Skipped {len(skipped)} files. Logged to {Path(flow_dir) / skip_log}")
    else:
        print("[INFO] No files skipped.")


def main():
    p = argparse.ArgumentParser(description="Affine-perturbation GT flow generator for DiffRegCD.")
    p.add_argument("--src", required=True, help="Directory of source images (become image B / t0).")
    p.add_argument("--warped-out", required=True, help="Output dir for affine-warped images (image A / t1).")
    p.add_argument("--flow-out", required=True, help="Output dir for per-image GT flow (.npy).")
    p.add_argument("--angle", type=float, default=20.0, help="Max abs rotation in degrees (range: [-a, a]).")
    p.add_argument("--scale", type=float, nargs=2, default=(0.90, 1.15), help="Scale range (min max).")
    p.add_argument("--trans", type=float, default=16.0, help="Max abs translation in pixels (range: [-t, t]).")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    args = p.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    apply_affine_transform_and_generate_flow(
        src_dir=args.src,
        warped_dir=args.warped_out,
        flow_dir=args.flow_out,
        angle_range=(-args.angle, args.angle),
        scale_range=tuple(args.scale),
        translation_range=(-args.trans, args.trans),
    )


if __name__ == "__main__":
    main()
