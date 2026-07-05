import warnings
import numpy as np
import cv2
import math
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from PIL import Image
import kornia
from einops import rearrange

import torch
import torch.nn as nn


class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features = False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1),
                indexing = 'ij'),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2),
                indexing = 'ij'),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
#            return torch.cos(8 * math.pi * self.pos_conv(x))
            return torch.cos(8 * math.pi * self.pos_conv(x.to(self.pos_conv.weight.device)))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently im_Bed in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=y.device),
                torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=y.device),
            ),
            indexing = 'ij'
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[
            None
        ].expand(b, h, w, 2)
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        return coarse_embedded_coords

    def forward(self, x, y, **kwargs):
        b, c, h1, w1 = x.shape
        b, c, h2, w2 = y.shape
        f = self.get_pos_enc(y)
        b, d, h2, w2 = f.shape
        x, y, f = self.reshape(x.float()), self.reshape(y.float()), self.reshape(f)
        K_xx = self.K(x, x)
        K_yy = self.K(y, y)
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        with warnings.catch_warnings():
            K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)
        device = x.device
        K_xy = K_xy.to(device)
        K_yy_inv = K_yy_inv.to(device)
        f = f.to(device)
        mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
        if not self.no_cov:
            cov_x = K_xx - K_xy.matmul(K_yy_inv.matmul(K_yx))
            cov_x = rearrange(cov_x, "b (h w) (r c) -> b h w r c", h=h1, w=w1, r=h1, c=w1)
            local_cov_x = self.get_local_cov(cov_x)
            local_cov_x = rearrange(local_cov_x, "b h w K -> b K h w")
            gp_feats = torch.cat((mu_x, local_cov_x), dim=1)
        else:
            gp_feats = mu_x
        return gp_feats


class CosKernel(nn.Module):  # Similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K


def recover_pose(E, kpts0, kpts1, K0, K1, mask):
    best_num_inliers = 0
    K0inv = np.linalg.inv(K0[:2,:2])
    K1inv = np.linalg.inv(K1[:2,:2])

    kpts0_n = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
    kpts1_n = (K1inv @ (kpts1-K1[None,:2,2]).T).T

    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, kpts0_n, kpts1_n, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t, mask.ravel() > 0)
    return ret


def warp_features_all(feats, flow):
    all_warped = []

    for t_idx, feat_list in enumerate(feats):  # List[List[Tensor]]
        warped_per_timestep = []

        for s_idx, feat in enumerate(feat_list):  # Tensor expected
            # if torch.is_tensor(feat):
                # print(f"[Input] t={t_idx}, s={s_idx}, feat.type={type(feat)}, feat.shape={feat.shape}")
            # else:
                # print(f"[Input] t={t_idx}, s={s_idx}, feat.type={type(feat)} (no .shape), flow.shape={getattr(flow,'shape','?')}")
            warped_feat = warp_single_feature(feat, flow)

            # ⚠️ fix: check if warp_single_feature is returning a list
            # if isinstance(warped_feat, list):
                # print(f"⚠️ Got a list from warp_single_feature at t={t_idx}, s={s_idx}")
#                warped_feat = warped_feat[0]  # Or handle properly

            warped_per_timestep.append(warped_feat)

        all_warped.append(warped_per_timestep)

    return all_warped



def warp_features_all2(feats, flow):
    all_warped = []

    for t_idx, feat_list in enumerate(feats):  # loop over timesteps
        warped_per_timestep = []

        for s_idx, feat in enumerate(feat_list):  # loop over scales
            B, C, H, W = feat.shape
            # Apply flow to feat
            warped_feat = warp_single_feature(feat, flow)  # or your warp function
            warped_per_timestep.append(warped_feat)

        all_warped.append(warped_per_timestep)

    return all_warped



def warp_single_feature(feat, flow, full_img_size=(512, 512)):
    # print("shape of the features in single feature func:", feat.shape)

    B, H_img, W_img, _ = flow.shape
    _, _, H, W = feat.shape

    # Resize flow to feature resolution
    flow_resized = F.interpolate(
        flow.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True
    ).permute(0, 2, 3, 1)

    # Scale flow for current resolution
#    scale_x, scale_y = W / W_img, H / H_img
#    flow_resized[..., 0] *= scale_x
#    flow_resized[..., 1] *= scale_y

    # Generate grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=feat.device),
        torch.linspace(-1, 1, W, device=feat.device),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)[None].repeat(B, 1, 1, 1)

    # Warp
    warped_grid = grid + flow_resized
    warped = F.grid_sample(feat, warped_grid, align_corners=True, mode='bilinear', padding_mode='border')

    return warped


def warp_single_feature_old(features, flow, full_img_size=(256, 256)):
    warped_all = []
    # print("shape of the features in single feature func", features.shape)
    B, H_img, W_img, _ = flow.shape
#    for feat in features:
    # print("shape of the feats", feat.shape)
#        feat = feat.unsqueeze(0)
    _, _, H, W = feat.shape
    flow_resized = F.interpolate(flow.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=True)
    flow_resized = flow_resized.permute(0, 2, 3, 1)
    scale_x, scale_y = W / W_img, H / H_img
    flow_resized[..., 0] *= scale_x
    flow_resized[..., 1] *= scale_y
    grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H, device=feat.device),
                                        torch.linspace(-1, 1, W, device=feat.device), indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1)[None].repeat(B, 1, 1, 1)
    warped_grid = grid + flow_resized
    warped = F.grid_sample(feat, warped_grid, align_corners=True, mode='bilinear', padding_mode='border')
#    warped_all.append(warped)
    return warped


# Code taken from https://github.com/PruneTruong/DenseMatching/blob/40c29a6b5c35e86b9509e65ab0cd12553d998e5f/validation/utils_pose_estimation.py
# --- GEOMETRY ---
def estimate_pose(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2,:2])
    K1inv = np.linalg.inv(K1[:2,:2])

    kpts0 = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
    kpts1 = (K1inv @ (kpts1-K1[None,:2,2]).T).T
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf
    )

    ret = None
    if E is not None:
        best_num_inliers = 0

        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret

def estimate_pose_uncalibrated(kpts0, kpts1, K0, K1, norm_thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None
    method = cv2.USAC_ACCURATE
    F, mask = cv2.findFundamentalMat(
        kpts0, kpts1, ransacReprojThreshold=norm_thresh, confidence=conf, method=method, maxIters=10000
    )
    E = K1.T@F@K0
    ret = None
    if E is not None:
        best_num_inliers = 0
        K0inv = np.linalg.inv(K0[:2,:2])
        K1inv = np.linalg.inv(K1[:2,:2])

        kpts0_n = (K0inv @ (kpts0-K0[None,:2,2]).T).T 
        kpts1_n = (K1inv @ (kpts1-K1[None,:2,2]).T).T
 
        for _E in np.split(E, len(E) / 3):
            n, R, t, _ = cv2.recoverPose(_E, kpts0_n, kpts1_n, np.eye(3), 1e9, mask=mask)
            if n > best_num_inliers:
                best_num_inliers = n
                ret = (R, t, mask.ravel() > 0)
    return ret

def unnormalize_coords(x_n,h,w):
    x = torch.stack(
        (w * (x_n[..., 0] + 1) / 2, h * (x_n[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    return x


def rotate_intrinsic(K, n):
    base_rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rot = np.linalg.matrix_power(base_rot, n)
    return rot @ K


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array(
            [
                [np.cos(r), -np.sin(r), 0.0, 0.0],
                [np.sin(r), np.cos(r), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1.0 / scales[0], 1.0 / scales[1], 1.0])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t.squeeze(), t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e) / t)
    return aucs


# From Patch2Pix https://github.com/GrumpyZhou/patch2pix
def get_depth_tuple_transform_ops_nearest_exact(resize=None):
    ops = []
    if resize:
        ops.append(TupleResizeNearestExact(resize))
    return TupleCompose(ops)

def get_depth_tuple_transform_ops(resize=None, normalize=True, unscale=False):
    ops = []
    if resize:
        ops.append(TupleResize(resize, mode=InterpolationMode.BILINEAR))
    return TupleCompose(ops)


def get_tuple_transform_ops(resize=None, normalize=True, unscale=False, clahe = False, colorjiggle_params = None):
    ops = []
    if resize:
        ops.append(TupleResize(resize))
    ops.append(TupleToTensorScaled())
    if normalize:
        ops.append(
            TupleNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )  # Imagenet mean/std
    return TupleCompose(ops)

class ToTensorScaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor, scale the range to [0, 1]"""

    def __call__(self, im):
        if not isinstance(im, torch.Tensor):
            im = np.array(im, dtype=np.float32).transpose((2, 0, 1))
            im /= 255.0
            return torch.from_numpy(im)
        else:
            return im

    def __repr__(self):
        return "ToTensorScaled(./255)"


class TupleToTensorScaled(object):
    def __init__(self):
        self.to_tensor = ToTensorScaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorScaled(./255)"


class ToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __call__(self, im):
        return torch.from_numpy(np.array(im, dtype=np.float32).transpose((2, 0, 1)))

    def __repr__(self):
        return "ToTensorUnscaled()"


class TupleToTensorUnscaled(object):
    """Convert a RGB PIL Image to a CHW ordered Tensor"""

    def __init__(self):
        self.to_tensor = ToTensorUnscaled()

    def __call__(self, im_tuple):
        return [self.to_tensor(im) for im in im_tuple]

    def __repr__(self):
        return "TupleToTensorUnscaled()"

class TupleResizeNearestExact:
    def __init__(self, size):
        self.size = size
    def __call__(self, im_tuple):
        return [F.interpolate(im, size = self.size, mode = 'nearest-exact') for im in im_tuple]

    def __repr__(self):
        return "TupleResizeNearestExact(size={})".format(self.size)


class TupleResize(object):
    def __init__(self, size, mode=InterpolationMode.BICUBIC):
        self.size = size
        self.resize = transforms.Resize(size, mode)
    def __call__(self, im_tuple):
        return [self.resize(im) for im in im_tuple]

    def __repr__(self):
        return "TupleResize(size={})".format(self.size)
    
class Normalize:
    def __call__(self,im):
        mean = im.mean(dim=(1,2), keepdims=True)
        std = im.std(dim=(1,2), keepdims=True)
        return (im-mean)/std


class TupleNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, im_tuple):
        c,h,w = im_tuple[0].shape
        if c > 3:
            warnings.warn(f"Number of channels c={c} > 3, assuming first 3 are rgb")
        return [self.normalize(im[:3]) for im in im_tuple]

    def __repr__(self):
        return "TupleNormalize(mean={}, std={})".format(self.mean, self.std)


class TupleCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im_tuple):
        for t in self.transforms:
            im_tuple = t(im_tuple)
        return im_tuple

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

@torch.no_grad()
def cls_to_flow(cls, deterministic_sampling = True):
    B,C,H,W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[torch.linspace(-1+1/res, 1-1/res, steps = res, device = device) for _ in range(2)],
        indexing = 'ij'
        )
    G = torch.stack([G[1],G[0]],dim=-1).reshape(C,2)
    if deterministic_sampling:
        sampled_cls = cls.max(dim=1).indices
    else:
        sampled_cls = torch.multinomial(cls.permute(0,2,3,1).reshape(B*H*W,C).softmax(dim=-1), 1).reshape(B,H,W)
    flow = G[sampled_cls]
    return flow




def cls_to_flow_refine(cls_logits: torch.Tensor, tau: float = 1.0):
    """
    Convert class logits to *normalized* flow in [-1,1], shape [B,H,W,2].
    Works with AMP (fp16/bf16), CUDA, and MPS.
    """
    B, C, H, W = cls_logits.shape
    device = cls_logits.device
    dtype  = cls_logits.dtype

    # class grid resolution (r x r = C)
    r = int(math.sqrt(C))
    assert r * r == C, "C must be a perfect square."

    # build codebook of centers in normalized space [-1,1]
    # build in float32, then cast to logits dtype to avoid half/float mix
    yy, xx = torch.meshgrid(
        torch.linspace(-1 + 1/r, 1 - 1/r, steps=r, device=device, dtype=torch.float32),
        torch.linspace(-1 + 1/r, 1 - 1/r, steps=r, device=device, dtype=torch.float32),
        indexing='ij'
    )
    centers = torch.stack((xx, yy), dim=-1).reshape(C, 2)            # [C,2]
    centers = centers.to(device=device, dtype=dtype)                 # match probs dtype!

    # probabilities over classes (softmax with temperature)
    if device.type == 'mps':
        # Be safe on MPS: compute softmax in fp32, then cast back
        probs = torch.softmax((cls_logits.float() / tau), dim=1).to(dtype)
    else:
        probs = torch.softmax(cls_logits / tau, dim=1)               # [B,C,H,W]

    # expectation over all classes -> normalized flow
    # probs: [B,C,H,W], centers: [C,2]  -> flow_norm: [B,H,W,2]
    flow_norm = torch.einsum('bchw,cd->bhwd', probs, centers)
    return flow_norm

@torch.no_grad()
def cls_to_flow_refine_orig(cls):
    B,C,H,W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(
        *[torch.linspace(-1+1/res, 1-1/res, steps = res, device = device) for _ in range(2)],
        indexing = 'ij'
        )
    G = torch.stack([G[1],G[0]],dim=-1).reshape(C,2)
    # FIXME: below softmax line causes mps to bug, don't know why.
    if device.type == 'mps':
        cls = cls.log_softmax(dim=1).exp()
    else:
        cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices
    
    index = torch.stack((mode-1, mode, mode+1, mode - res, mode + res), dim = 1).clamp(0,C - 1).long()
    neighbours = torch.gather(cls, dim = 1, index = index)[...,None]
    flow = neighbours[:,0] * G[index[:,0]] + neighbours[:,1] * G[index[:,1]] + neighbours[:,2] * G[index[:,2]] + neighbours[:,3] * G[index[:,3]] + neighbours[:,4] * G[index[:,4]]
    tot_prob = neighbours.sum(dim=1)  
    flow = flow / tot_prob
    return flow


def get_gt_warp(depth1, depth2, T_1to2, K1, K2, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, H = None, W = None):
    
    if H is None:
        B,H,W = depth1.shape
    else:
        B = depth1.shape[0]
    with torch.no_grad():
        x1_n = torch.meshgrid(
            *[
                torch.linspace(
                    -1 + 1 / n, 1 - 1 / n, n, device=depth1.device
                )
                for n in (B, H, W)
            ],
            indexing = 'ij'
        )
        x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H * W, 2)
        mask, x2 = warp_kpts(
            x1_n.double(),
            depth1.double(),
            depth2.double(),
            T_1to2.double(),
            K1.double(),
            K2.double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        prob = mask.float().reshape(B, H, W)
        x2 = x2.reshape(B, H, W, 2)
        return x2, prob

@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, smooth_mask = False, return_relative_depth_error = False, depth_interpolation_mode = "bilinear", relative_depth_error_threshold = 0.05):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    # https://github.com/zju3dv/LoFTR/blob/94e98b695be18acb43d5d3250f52226a8e36f839/src/loftr/utils/geometry.py adapted from here
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>, should be normalized in (-1,1)
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":
        # Inspired by approach in inloc, try to fill holes from bilinear interpolation by nearest neighbour interpolation
        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "bilinear",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        valid_nearest, warp_nearest = warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "nearest-exact",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest) 
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp
        
        
    kpts0_depth = F.grid_sample(depth0[:, None], kpts0[:, :, None], mode = depth_interpolation_mode, align_corners=False)[
        :, 0, :, 0
    ]
    kpts0 = torch.stack(
        (w * (kpts0[..., 0] + 1) / 2, h * (kpts0[..., 1] + 1) / 2), dim=-1
    )  # [-1+1/h, 1-1/h] -> [0.5, h-0.5]
    # Sample depth, get calculable_mask on depth != 0
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_n = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)
    kpts0_cam = kpts0_n

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0 = torch.stack(
        (2 * w_kpts0[..., 0] / w - 1, 2 * w_kpts0[..., 1] / h - 1), dim=-1
    )  # from [0.5,h-0.5] -> [-1+1/h, 1-1/h]
    # w_kpts0[~covisible_mask, :] = -5 # xd

    w_kpts0_depth = F.grid_sample(
        depth1[:, None], w_kpts0[:, :, None], mode=depth_interpolation_mode, align_corners=False
    )[:, 0, :, 0]
    
    relative_depth_error = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error/smooth_mask).exp()
    valid_mask = nonzero_mask * covisible_mask * consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0

imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
imagenet_std = torch.tensor([0.229, 0.224, 0.225])


def numpy_to_pil(x: np.ndarray):
    """
    Args:
        x: Assumed to be of shape (h,w,c)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if x.max() <= 1.01:
        x *= 255
    x = x.astype(np.uint8)
    return Image.fromarray(x)


def tensor_to_pil(x, unnormalize=False):
    if unnormalize:
        x = x * (imagenet_std[:, None, None].to(x.device)) + (imagenet_mean[:, None, None].to(x.device))
    x = x.detach().permute(1, 2, 0).cpu().numpy()
    x = np.clip(x, 0.0, 1.0)
    return numpy_to_pil(x)


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cpu()
    return batch


def get_pose(calib):
    w, h = np.array(calib["imsize"])[0]
    return np.array(calib["K"]), np.array(calib["R"]), np.array(calib["T"]).T, h, w


def compute_relative_pose(R1, t1, R2, t2):
    rots = R2 @ (R1.T)
    trans = -rots @ t1 + t2
    return rots, trans

@torch.no_grad()
def reset_opt(opt):
    for group in opt.param_groups:
        for p in group['params']:
            if p.requires_grad:
                state = opt.state[p]
                # State initialization

                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p)
                # Exponential moving average of gradient difference
                state['exp_avg_diff'] = torch.zeros_like(p)


def flow_to_pixel_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                w1 * (flow[..., 0] + 1) / 2,
                h1 * (flow[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return flow

to_pixel_coords = flow_to_pixel_coords # just an alias

def flow_to_normalized_coords(flow, h1, w1):
    flow = (
        torch.stack(
            (
                2 * (flow[..., 0]) / w1 - 1,
                2 * (flow[..., 1]) / h1 - 1,
            ),
            axis=-1,
        )
    )
    return flow

to_normalized_coords = flow_to_normalized_coords # just an alias

def warp_to_pixel_coords(warp, h1, w1, h2, w2):
    warp1 = warp[..., :2]
    warp1 = (
        torch.stack(
            (
                w1 * (warp1[..., 0] + 1) / 2,
                h1 * (warp1[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    warp2 = warp[..., 2:]
    warp2 = (
        torch.stack(
            (
                w2 * (warp2[..., 0] + 1) / 2,
                h2 * (warp2[..., 1] + 1) / 2,
            ),
            axis=-1,
        )
    )
    return torch.cat((warp1,warp2), dim=-1)



def signed_point_line_distance(point, line, eps: float = 1e-9):
    r"""Return the distance from points to lines.

    Args:
       point: (possibly homogeneous) points :math:`(*, N, 2 or 3)`.
       line: lines coefficients :math:`(a, b, c)` with shape :math:`(*, N, 3)`, where :math:`ax + by + c = 0`.
       eps: Small constant for safe sqrt.

    Returns:
        the computed distance with shape :math:`(*, N)`.
    """

    if not point.shape[-1] in (2, 3):
        raise ValueError(f"pts must be a (*, 2 or 3) tensor. Got {point.shape}")

    if not line.shape[-1] == 3:
        raise ValueError(f"lines must be a (*, 3) tensor. Got {line.shape}")

    numerator = (line[..., 0] * point[..., 0] + line[..., 1] * point[..., 1] + line[..., 2])
    denominator = line[..., :2].norm(dim=-1)

    return numerator / (denominator + eps)


def signed_left_to_right_epipolar_distance(pts1, pts2, Fm):
    r"""Return one-sided epipolar distance for correspondences given the fundamental matrix.

    This method measures the distance from points in the right images to the epilines
    of the corresponding points in the left images as they reflect in the right images.

    Args:
       pts1: correspondences from the left images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       pts2: correspondences from the right images with shape
         :math:`(*, N, 2 or 3)`. If they are not homogeneous, converted automatically.
       Fm: Fundamental matrices with shape :math:`(*, 3, 3)`. Called Fm to
         avoid ambiguity with torch.nn.functional.

    Returns:
        the computed Symmetrical distance with shape :math:`(*, N)`.
    """
    import kornia
    if (len(Fm.shape) < 3) or not Fm.shape[-2:] == (3, 3):
        raise ValueError(f"Fm must be a (*, 3, 3) tensor. Got {Fm.shape}")

    if pts1.shape[-1] == 2:
        pts1 = kornia.geometry.convert_points_to_homogeneous(pts1)

    F_t = Fm.transpose(dim0=-2, dim1=-1)
    line1_in_2 = pts1 @ F_t

    return signed_point_line_distance(pts2, line1_in_2)

def get_grid(b, h, w, device):
    grid = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=device)
            for n in (b, h, w)
        ],
        indexing = 'ij'
    )
    grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
    return grid


def get_autocast_params(device=None, enabled=False, dtype=None):
    if device is None:
        autocast_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        #strip :X from device
        autocast_device = str(device).split(":")[0]
    if 'cuda' in str(device):
        out_dtype = dtype
        enabled = True
    else:
        out_dtype = torch.bfloat16
        enabled = False
        # mps is not supported
        autocast_device = "cpu"
    return autocast_device, enabled, out_dtype

def check_not_i16(im):
    if im.mode == "I;16":
        raise NotImplementedError("Can't handle 16 bit images")

def check_rgb(im):
    if im.mode != "RGB":
        raise NotImplementedError("Can't handle non-RGB images")
