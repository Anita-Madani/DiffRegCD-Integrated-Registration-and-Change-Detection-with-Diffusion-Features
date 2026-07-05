
import math, torch
import torch.nn.functional as F


# ---------- Helpers (aligned) ----------
@torch.no_grad()
def _make_codebook(r: int, device, dtype):
    """
    Codebook centers on the SAME lattice used by soft-argmax:
    edges grid linspace(-1, 1, r). Shape: [1, r*r, 2, 1, 1] as (x,y).
    """
    ys = torch.linspace(-1.0, 1.0, r, device=device, dtype=dtype)  # slow (rows)
    xs = torch.linspace(-1.0, 1.0, r, device=device, dtype=dtype)  # fast (cols)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')                  # [r,r]
    centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [r*r, 2] = (x,y)
    return centers.view(1, r*r, 2, 1, 1)                            # [1,r*r,2,1,1]

def _softargmax_flow(logits, r, temperature=None):
    if temperature is not None:
        logits = logits / temperature.clamp(0.3, 1.5)
    C = logits.shape[1]
    C_eff = r*r if C == r*r + 1 else C
    probs = torch.softmax(logits[:, :C_eff], dim=1)
    dev, dt = logits.device, logits.dtype
    ys = torch.linspace(-1.0, 1.0, r, device=dev, dtype=dt)
    xs = torch.linspace(-1.0, 1.0, r, device=dev, dtype=dt)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')    # SAME order as codebook
    uu = xx.reshape(1, -1, 1, 1)                      # x
    vv = yy.reshape(1, -1, 1, 1)                      # y
    u = (probs * uu).sum(1, keepdim=True)
    v = (probs * vv).sum(1, keepdim=True)
    return torch.cat([u, v], dim=1)   


# ---------- Loss ----------
def gm_cls_loss_from_gt(
    gt_flow,          # [B,2,H_in,W_in] in *pixels*
    logits,              # [B, C, Hc, Wc] where C = r*r or r*r+1
    *,
    sigma_bins=1.0,
    temperature=None,
    align_corners=True
):
    B, C, Hc, Wc = logits.shape
    device, dtype = logits.device, logits.dtype

    # infer r and whether there's a no-match
    r_sq = int(round(math.sqrt(C)))
    if r_sq * r_sq == C:
        r = r_sq; has_nomatch = False; C_eff = C
    else:
        r_sq1 = int(round(math.sqrt(C - 1)))
        if r_sq1 * r_sq1 + 1 == C:
            r = r_sq1; has_nomatch = True; C_eff = r*r
        else:
            raise AssertionError(f"C={C} is neither r^2 nor r^2+1")

    # resize GT to [Hc,Wc] and convert pixels -> normalized offsets
    gt_rs = F.interpolate(gt_flow.to(dtype), size=(Hc, Wc),
                      mode='bilinear', align_corners=align_corners)

    # NEW: normalize using the *input* image size, not the coarse size
    H_in, W_in = gt_flow.shape[-2:]
    if align_corners:
        sx = 2.0 / (W_in - 1) if W_in > 1 else 0.0
        sy = 2.0 / (H_in - 1) if H_in > 1 else 0.0
    else:
        sx = 2.0 / W_in
        sy = 2.0 / H_in

    gt_norm = torch.empty_like(gt_rs)
    gt_norm[:, 0] = gt_rs[:, 0] * sx
    gt_norm[:, 1] = gt_rs[:, 1] * sy
                                   # [B,2,Hc,Wc]

    # mask out-of-range targets
    overflow = (gt_norm.abs().max(dim=1, keepdim=True).values > 1)     # [B,1,Hc,Wc]
    valid = (~overflow).float()

    # soft labels over r*r lattice (edge-aligned); σ = 1 bin = 2/(r-1)
    code = _make_codebook(r, device, dtype)                             # [1,r*r,2,1,1]
    gtv  = gt_norm.unsqueeze(1)                                         # [B,1,2,Hc,Wc]
    d2   = ((gtv - code) ** 2).sum(dim=2)                               # [B,r*r,Hc,Wc]
    bin_delta = 2.0 / (r - 1) if r > 1 else 0.0
    sigma = bin_delta * sigma_bins
    P_eff = torch.exp(-0.5 * d2 / (sigma**2 + 1e-12))
    P_eff = P_eff / (P_eff.sum(1, keepdim=True) + 1e-8)                 # [B,r*r,Hc,Wc]

    # build target P (optionally add no-match)
    if has_nomatch:
        P_no = 1e-4 * torch.ones(B, 1, Hc, Wc, device=device, dtype=dtype)
        P = torch.cat([P_eff, P_no], dim=1)                              # [B,r*r+1,Hc,Wc]
        if overflow.any():
            P = P.clone()
            P[:, :C_eff] *= (~overflow).float()
            P[:, C_eff:]  = torch.where(overflow, P.new_ones(1), P[:, C_eff:])
        logp = torch.log_softmax(
            logits / (temperature.clamp(0.3, 1.5) if temperature is not None else 1.0),
            dim=1
        )
        kl = F.kl_div(logp, P.clamp_min(1e-8), reduction='none').sum(1, keepdim=True)  # [B,1,Hc,Wc]
        weight = (valid + overflow.float())
        ce = (kl * weight).sum() / weight.sum().clamp_min(1)
        no_match_prob = torch.softmax(logits, dim=1)[:, -1].mean()
    else:
        logp_eff = torch.log_softmax(
            logits[:, :C_eff] / (temperature.clamp(0.3, 1.5) if temperature is not None else 1.0),
            dim=1
        )
        kl = F.kl_div(logp_eff, P_eff.clamp_min(1e-8), reduction='none').sum(1, keepdim=True)
        ce = (kl * valid).sum() / valid.sum().clamp_min(1)
        no_match_prob = torch.tensor(0.0, device=device)

    # continuous metric (normalized coords) via aligned soft-argmax
    pred_norm = _softargmax_flow(logits, r, temperature)                # [B,2,Hc,Wc]
    # pred_norm[:, 0].mul_(-1)
    epe = ((pred_norm - gt_norm).abs() * valid).sum() / valid.sum().clamp_min(1)
    torch.save(pred_norm.permute(0,2,3,1).contiguous().cpu(), "pred_32x32x2.pt")

    # save GT in pixels at full res
    torch.save(gt_flow.detach().cpu(), "gt_flow_px_256.pt")

    return {
        "gm_cls_loss_fine": ce,
        "epe_norm": epe,
        "no_match_prob": no_match_prob.detach(),
        "overflow_ratio": overflow.float().mean().detach()
    }


# import torch
# import torch.nn.functional as F
# import math


# import math, torch
# import torch.nn.functional as F


# @torch.no_grad()
# def _make_codebook(r: int, device, dtype):
#     """Return codebook centers as [1, r*r, 2, 1, 1] in normalized coords [-1,1]."""
#     ys = torch.linspace(-1 + 1/r, 1 - 1/r, r, device=device, dtype=dtype)
#     xs = torch.linspace(-1 + 1/r, 1 - 1/r, r, device=device, dtype=dtype)
#     yy, xx = torch.meshgrid(ys, xs, indexing='ij')             # [r,r]
#     centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [r*r, 2]
#     return centers.view(1, r*r, 2, 1, 1)                       # [1, r*r, 2, 1, 1]

# def _softargmax_flow(logits, r, temperature=None):
#     """Decode logits to continuous flow in [-1,1] by expectation over r*r bins."""
#     if temperature is not None:
#         logits = logits / temperature.clamp(0.3, 1.5)

#     C = logits.shape[1]
#     C_eff = r*r if C == r*r + 1 else C           # drop no-match if present
#     probs = torch.softmax(logits[:, :C_eff], dim=1)  # [B, r*r, Hc, Wc]

#     dev, dt = logits.device, logits.dtype
#     # make flattened coordinates (use reshape, not view)
#     u1d = torch.linspace(-1, 1, r, device=dev, dtype=dt)
#     v1d = torch.linspace(-1, 1, r, device=dev, dtype=dt)
#     uu, vv = torch.meshgrid(u1d, v1d, indexing='ij')  # [r,r]
#     uu = uu.reshape(1, -1, 1, 1)                      # [1, r*r, 1, 1]
#     vv = vv.reshape(1, -1, 1, 1)                      # [1, r*r, 1, 1]

#     u = (probs * uu).sum(1, keepdim=True)             # [B,1,Hc,Wc]
#     v = (probs * vv).sum(1, keepdim=True)             # [B,1,Hc,Wc]
#     return torch.cat([u, v], dim=1)                   # [B,2,Hc,Wc]
#                              # [B,2,Hc,Wc]

# def gm_cls_loss_from_gt(
#     gt_flow_px,          # [B,2,H_in,W_in] in *pixels*
#     logits,              # [B, C, Hc, Wc] where C=r*r or r*r+1
#     *,
#     sigma_bins=1.0,
#     temperature=None,
#     align_corners=False
# ):
#     B, C, Hc, Wc = logits.shape
#     device, dtype = logits.device, logits.dtype

#     # --- infer r and whether there's a no-match channel ---
#     r_sq = int(round(math.sqrt(C)))
#     if r_sq * r_sq == C:
#         r = r_sq
#         has_nomatch = False
#         C_eff = C
#     else:
#         r_sq1 = int(round(math.sqrt(C - 1)))
#         if r_sq1 * r_sq1 + 1 == C:
#             r = r_sq1
#             has_nomatch = True
#             C_eff = r * r
#         else:
#             raise AssertionError(f"C={C} is neither r^2 nor r^2+1")

#     # --- resize GT to logits map and convert pixels -> normalized offsets at (Hc,Wc) ---
#     gt_rs = F.interpolate(gt_flow_px.to(dtype), size=(Hc, Wc),
#                           mode='bilinear', align_corners=align_corners)     # [B,2,Hc,Wc]
#     if align_corners:
#         sx = 2.0 / (Wc - 1) if Wc > 1 else 0.0
#         sy = 2.0 / (Hc - 1) if Hc > 1 else 0.0
#     else:
#         sx = 2.0 / Wc
#         sy = 2.0 / Hc
#     gt_norm = torch.empty_like(gt_rs)
#     gt_norm[:, 0] = gt_rs[:, 0] * sx
#     gt_norm[:, 1] = gt_rs[:, 1] * sy                                   # [B,2,Hc,Wc]

#     # --- overflow mask (targets outside representable range [-1,1]) ---
#     overflow = (gt_norm.abs().max(dim=1, keepdim=True).values > 1)     # [B,1,Hc,Wc]
#     valid = (~overflow).float()

#     # --- soft labels around GT on r*r grid ---
#     code = _make_codebook(r, device, dtype)                             # [1, r*r, 2, 1, 1]
#     gtv  = gt_norm.unsqueeze(1)                                         # [B, 1, 2, Hc, Wc]
#     d2   = ((gtv - code) ** 2).sum(dim=2)                               # [B, r*r, Hc, Wc]
#     sigma = (2.0 / r) * sigma_bins                                      # 1 bin std (normalized)
#     P_eff = torch.exp(-0.5 * d2 / (sigma ** 2))
#     P_eff = P_eff / (P_eff.sum(1, keepdim=True) + 1e-8)                 # [B, r*r, Hc, Wc]

#     # --- build full target distribution P (add no-match if present) ---
#     if has_nomatch:
#         P_no = 1e-4 * torch.ones(B, 1, Hc, Wc, device=device, dtype=dtype)
#         P = torch.cat([P_eff, P_no], dim=1)                              # [B, r*r+1, Hc, Wc]
#         # Hard no-match where overflow happens
#         if overflow.any():
#             P = P.clone()
#             P[:, :C_eff] *= (~overflow).float()
#             P[:, C_eff:]  = torch.where(overflow, P.new_ones(1), P[:, C_eff:])
#         logp = torch.log_softmax(
#             logits / (temperature.clamp(0.3, 1.5) if temperature is not None else 1.0),
#             dim=1
#         )
#         # KL per pixel, then mean; count overflow pixels too (as no-match)
#         kl = F.kl_div(logp, P.clamp_min(1e-8), reduction='none').sum(1, keepdim=True)  # [B,1,Hc,Wc]
#         weight = (valid + overflow.float())
#         ce = (kl * weight).sum() / weight.sum().clamp_min(1)
#         no_match_prob = torch.softmax(logits, dim=1)[:, -1].mean()
#     else:
#         # No no-match channel: exclude overflow pixels from the loss
#         logp_eff = torch.log_softmax(
#             logits[:, :C_eff] / (temperature.clamp(0.3, 1.5) if temperature is not None else 1.0),
#             dim=1
#         )
#         kl = F.kl_div(logp_eff, P_eff.clamp_min(1e-8), reduction='none').sum(1, keepdim=True)  # [B,1,Hc,Wc]
#         ce = (kl * valid).sum() / valid.sum().clamp_min(1)
#         no_match_prob = torch.tensor(0.0, device=device)

#     # --- continuous flow EPE from soft-argmax ---
#     pred_norm = _softargmax_flow(logits, r, temperature)                # [B,2,Hc,Wc]
#     epe = ((pred_norm - gt_norm).abs() * valid).sum() / valid.sum().clamp_min(1)

#     # --- stats ---
#     of_ratio = overflow.float().mean()
#     return {
#         "gm_cls_loss_fine": ce,
#         "epe_norm": epe,
#         "no_match_prob": no_match_prob.detach(),
#         "overflow_ratio": of_ratio.detach()
#     }










# ## THE LOSS THAT WORKS ##

# # def gm_cls_loss_from_gt(gt_flow, scale_gm_cls, H, W, scale="fine"):
# #     """
# #     Args:
# #         gt_flow: [B, 2, H, W] — ground truth flow (normalized in [-1, 1])
# #         scale_gm_cls: [B, C, H, W] — predicted logits for each flow class
# #         scale: str — just used for logging
# #     """
    
# #     H_img, W_img = 512, 512

# #     # Normalize gt_flow to [-1, 1] based on image size
# #     print("size of the gt flow", gt_flow.shape)
# #     gt_flow_down = F.interpolate(gt_flow, size=(H,W) , mode='bilinear', align_corners=True)
# # #    print("shape of the gt glow donw", gt_flow_down.shape)
# #     gt_flow_down = gt_flow_down
# #     gt_flow_norm = gt_flow_down.clone()
# #     gt_flow_norm[:, 0] = gt_flow_down[:, 0] / (W_img / 2)  # x flow
# #     gt_flow_norm[:, 1] = gt_flow_down[:, 1] / (H_img / 2)  # y flow
# # #    print("shape of logit - loss ", scale_gm_cls.shape)
# # #    print("shape input gt flow", gt_flow.shape)
# #     B, C, H, W = scale_gm_cls.shape
# #     device = gt_flow.device
# #     cls_res = round(math.sqrt(C))  # e.g., 16 → 16x16 class grid
# #     # build class centers grid: shape [C, 2]
# #     G = torch.meshgrid(
# #         *[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps=cls_res, device=device) for _ in range(2)],
# #         indexing='ij'
# #     )
# #     G = torch.stack((G[1], G[0]), dim=-1).reshape(C, 2)  # [C, 2]

# #     # Compute closest GT class index for each pixel
# #     # gt_flow: [B, 2, H, W] → [B, H, W, 2]
# #     gt_vec = gt_flow_norm.permute(0, 2, 3, 1).unsqueeze(3) # [B, H, W, 2]
# #     # Compute L2 distance from GT to each class center
# # #    print("shapes", G.shape , gt_vec.shape )
# #     G = G.view(1, 1, 1, -1, 2)
# # #    print("shapes", G.shape , gt_vec.shape )
# #     diff = G - gt_vec  # [B, H, W, C, 2]
# #     dists = torch.norm(diff, dim=-1)  # [B, H, W, C]
# #     target_cls = dists.argmin(dim=-1)  # [B, H, W]
# #     with torch.no_grad():
# #         uniq = target_cls.unique()
# #         print(f"[DEBUG] target_cls unique: {uniq.numel()} / {C}")
# #         if uniq.numel() < 10:
# #             print("[DEBUG] target bins:", uniq[:20].tolist())

# #     # compute cross entropy
# #     loss = F.cross_entropy(scale_gm_cls, target_cls, reduction='mean')

# #     return {f"gm_cls_loss_{scale}": loss}




# # import torch
# # import torch.nn.functional as F
# # import math


# # import torch, math
# # import torch.nn.functional as F

# # def _resize_flow(flow, out_h, out_w, align_corners=True):
# #     # flow: [B,2,H,W] in *pixels*
# #     B,_,H,W = flow.shape
# #     rs = F.interpolate(flow, size=(out_h,out_w), mode='bilinear', align_corners=align_corners)
# #     if align_corners:
# #         sx = (out_w-1)/(W-1) if W>1 else 1.0
# #         sy = (out_h-1)/(H-1) if H>1 else 1.0
# #     else:
# #         sx = out_w/W
# #         sy = out_h/H
# #     rs[:,0] *= sx
# #     rs[:,1] *= sy
# #     return rs

# # def _pixels_to_norm(flow_px, H, W, align_corners=True):
# #     if align_corners:
# #         sx = 2.0/(W-1) if W>1 else 0.0
# #         sy = 2.0/(H-1) if H>1 else 0.0
# #     else:
# #         sx = 2.0/W
# #         sy = 2.0/H
# #     out = torch.empty_like(flow_px)
# #     out[:,0] = flow_px[:,0] * sx
# #     out[:,1] = flow_px[:,1] * sy
# #     return out

# # def gm_cls_loss_from_gt(gt_flow_px, logits, align_corners=True, scale="fine"):
# #     """
# #     gt_flow_px: [B,2,256,256] in *pixels* (your case)
# #     logits    : [B,C,Hc,Wc] (classifier over flow classes on a normalized grid)
# #     """
# #     B,C,Hc,Wc = logits.shape
# #     device, dtype = logits.device, logits.dtype

# #     # 1) resize GT to logits resolution, scale vector magnitudes (still pixels)
# #     gt_rs_px = _resize_flow(gt_flow_px.to(dtype), Hc, Wc, align_corners=align_corners)

# #     # 2) convert to normalized offsets at (Hc,Wc)
# #     gt_norm = _pixels_to_norm(gt_rs_px, Hc, Wc, align_corners=align_corners)  # [B,2,Hc,Wc]

# #     # 3) build class centers in normalized coords
# #     res = round(math.sqrt(C))
# #     ys = torch.linspace(-1+1/res, 1-1/res, res, device=device, dtype=dtype)
# #     xs = torch.linspace(-1+1/res, 1-1/res, res, device=device, dtype=dtype)
# #     yy, xx = torch.meshgrid(ys, xs, indexing='ij')
# #     G = torch.stack([xx, yy], dim=-1).reshape(C,2)  # [C,2], normalized

# #     # 4) nearest class target
# #     gt_vec = gt_norm.permute(0,2,3,1).unsqueeze(3)  # [B,Hc,Wc,1,2]
# #     dists = torch.norm(G.view(1,1,1,C,2) - gt_vec, dim=-1)  # [B,Hc,Wc,C]
# #     target = dists.argmin(dim=-1)  # [B,Hc,Wc]

# #     # 5) CE on logits
# #     loss = F.cross_entropy(logits, target, reduction='mean')
# #     return {f"gm_cls_loss_{scale}": loss}


# # #def gm_cls_loss_from_gt(gt_flow, scale_gm_cls, H, W, scale="fine"):
# # #    """
# # #    Args:
# # #        gt_flow: [B, 2, H, W] — ground truth flow (normalized in [-1, 1])
# # #        scale_gm_cls: [B, C, H, W] — predicted logits for each flow class
# # #        scale: str — just used for logging
# # #    """
# # #    
# # #    H_img, W_img = 256, 256
# # #
# # #    # Normalize gt_flow to [-1, 1] based on image size
# # #    gt_flow_down = F.interpolate(gt_flow, size=(H,W) , mode='bilinear', align_corners=True)
# # #    print("shape of the gt glow donw", gt_flow_down.shape)
# # #    gt_flow_down = gt_flow_down
# # #    gt_flow_norm = gt_flow_down.clone()
# # #    gt_flow_norm[:, 0] = gt_flow_down[:, 0] / (W_img / 2)  # x flow
# # #    gt_flow_norm[:, 1] = gt_flow_down[:, 1] / (H_img / 2)  # y flow
# # #    print("shape of logit - loss ", scale_gm_cls.shape)
# # #    print("shape input gt flow", gt_flow.shape)
# # #    B, C, H, W = scale_gm_cls.shape
# # #    device = gt_flow.device
# # #    cls_res = round(math.sqrt(C))  # e.g., 16 → 16x16 class grid
# #     # build class centers grid: shape [C, 2]
# # #    G = torch.meshgrid(
# # #        *[torch.linspace(-1+1/cls_res, 1 - 1/cls_res, steps=cls_res, device=device) for _ in range(2)],
# # #        indexing='ij'
# # #    )
# # #    G = torch.stack((G[1], G[0]), dim=-1).reshape(C, 2)  # [C, 2]
# # #
# # #    # Compute closest GT class index for each pixel
# # #    # gt_flow: [B, 2, H, W] → [B, H, W, 2]
# # #    gt_vec = gt_flow_down.permute(0, 2, 3, 1).unsqueeze(3) # [B, H, W, 2]
# # #    # Compute L2 distance from GT to each class center
# # #    print("shapes", G.shape , gt_vec.shape )
# # #    G = G.view(1, 1, 1, -1, 2)
# # #    print("shapes", G.shape , gt_vec.shape )
# # #    diff = G - gt_vec  # [B, H, W, C, 2]
# # #    dists = torch.norm(diff, dim=-1)  # [B, H, W, C]
# # #    target_cls = dists.argmin(dim=-1)  # [B, H, W]

# #     # compute cross entropy
# # #    loss = F.cross_entropy(scale_gm_cls, target_cls, reduction='mean')

# # #    return {f"gm_cls_loss_{scale}": loss}
