import numpy as np
import cv2
import torch

def flow_to_rgb(flow_tensor):
    """
    Convert flow tensor (B, 2, H, W) to RGB image (H, W, 3) using HSV encoding.
    """
    flow = flow_tensor[0].detach().cpu().numpy()  # (2, H, W)
    flow = np.transpose(flow, (1, 2, 0))  # (H, W, 2)

    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2       # Hue
    hsv[..., 1] = 255                         # Saturation
    hsv[..., 2] = np.clip(mag * 8, 0, 255)    # Brightness (scale as needed)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
