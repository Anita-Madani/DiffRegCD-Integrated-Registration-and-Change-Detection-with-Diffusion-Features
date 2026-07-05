import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from data.util import transform_augment_cd_all  # unified transform

IMG_FOLDER_NAME = "train/t1"
IMG_POST_FOLDER_NAME = "train/t0"
FLOW_FOLDER_NAME = "train/flow"
LABEL_FOLDER_NAME = "train/mask"
LIST_FOLDER_NAME = "list"
label_suffix = ".png"


def get_img1_path(root_dir, fname):
    return os.path.join(root_dir, IMG_FOLDER_NAME, fname + ".png")

def get_img2_path(root_dir, fname):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, fname + ".png")

def get_label_path(root_dir, fname):
    return os.path.join(root_dir, LABEL_FOLDER_NAME, fname + ".png")

def get_flow_path(root_dir, fname):
    flow_fname = fname + ".npy"  # assuming flow is saved with same base name
    return os.path.join(root_dir, FLOW_FOLDER_NAME, flow_fname)


class CMUCDFlowDataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=-1, load_flow=True):
        self.res = resolution
        self.split = split
        self.data_len = data_len
        self.load_flow = load_flow
        self.root_dir = dataroot

        list_path = os.path.join(dataroot, LIST_FOLDER_NAME, f'{split}.txt')
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Missing split file: {list_path}")
        with open(list_path, 'r') as f:
            self.file_names = [line.strip() for line in f if line.strip()]
        self.dataset_len = len(self.file_names)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fname = self.file_names[index % self.data_len]

        # Load image pairs
        img_A = Image.open(get_img1_path(self.root_dir, fname)).convert("RGB")
        img_B = Image.open(get_img2_path(self.root_dir, fname)).convert("RGB")
        label = Image.open(get_label_path(self.root_dir, fname)).convert("L")

        # Load flow (npy format, shape: [H, W, 2])
        flow_path = get_flow_path(self.root_dir, fname)
        if self.load_flow and os.path.exists(flow_path):
            flow = np.load(flow_path).astype(np.float32)
            flow = torch.from_numpy(flow)
        else:
            if self.load_flow and not hasattr(self, "_warned_missing_flow"):
                print(f"[Warning] Flow file missing for some images. Using zero flow.")
                self._warned_missing_flow = True
            flow = torch.zeros(2, self.res, self.res)

        # Apply augmentations
        img_A, img_B, label, flow = transform_augment_cd_all(
            img_A, img_B, label, flow, split=self.split, min_max=(-1, 1)
        )
#        print("shape fof flwo in loading ", flow.shape )

        return {
            'A': img_A,         # [3, H, W]
            'B': img_B,         # [3, H, W]
            'L': label,         # [H, W]
            'flow': flow,    # renamed key to match model expectations
            'Index': index
        }
