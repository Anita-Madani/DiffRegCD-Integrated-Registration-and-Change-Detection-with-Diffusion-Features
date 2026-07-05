import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from misc.metric_tools import ConfuseMatrixMeter
from misc.torchutils import get_scheduler
logger = logging.getLogger('base')
from registration.registration_module import RegistrationModule
from registration.utils256 import warp_features_all,  cls_to_flow_refine
from losses.roma_flow_cls_256 import gm_cls_loss_from_gt
from registration.utils256 import decode_pred_norm_to_px_full

import torch
import torch.nn.functional as F

def infer_r_from_logits(logits: torch.Tensor) -> int:
    C = logits.shape[1]
    r = int((((C - 1) ** 0.5) - 1) / 2)
    assert (2*r + 1)**2 + 1 == C, f"Bad bins: C={C}, inferred r={r}"
    return r

from math import isqrt

def infer_bin_layout(C: int):
    """
    Infer search-window geometry from channel count C.
    Returns dict with keys: r, side, has_nomatch, parity ('odd'|'even').
    Supports:
      - C = (2r+1)^2 + 1      (odd grid + no-match)
      - C = (2r+1)^2          (odd grid, no 'no-match')
      - C = (2r)^2 + 1        (even grid + no-match)
      - C = (2r)^2            (even grid, no 'no-match')  <-- YOUR CASE (32x32)
    """
    # exact square?
    s = isqrt(C)
    if s * s == C:
        parity = 'odd' if (s % 2 == 1) else 'even'
        r = (s - 1) // 2 if parity == 'odd' else s // 2
        return dict(r=r, side=s, has_nomatch=False, parity=parity)

    # square after removing one 'no-match' channel?
    if C > 1:
        s1 = isqrt(C - 1)
        if s1 * s1 == (C - 1):
            parity = 'odd' if (s1 % 2 == 1) else 'even'
            r = (s1 - 1) // 2 if parity == 'odd' else s1 // 2
            return dict(r=r, side=s1, has_nomatch=True, parity=parity)

    raise AssertionError(f"Unrecognized bin layout for C={C}")


def resize_flow(flow, size, align_corners=True):
    """Resize flow (B,2,H,W) and scale vectors to new resolution."""
    B, C, H, W = flow.shape
    H2, W2 = size
    out = F.interpolate(flow, (H2, W2), mode="bilinear", align_corners=align_corners)
    out[:,0] *= (W2 / W)
    out[:,1] *= (H2 / H)
    return out

def epe_mean(a, b):
    return (a - b).pow(2).sum(1).sqrt().mean()

def mean_bias(a, b):
    d = (a - b).mean(dim=(0,2,3))  # avg vector bias
    return float(d[0]), float(d[1])

def corr_xy(a, b):
    ax, ay = a[:,0].flatten(), a[:,1].flatten()
    bx, by = b[:,0].flatten(), b[:,1].flatten()
    cx = torch.corrcoef(torch.stack([ax, bx]))[0,1]
    cy = torch.corrcoef(torch.stack([ay, by]))[0,1]
    return float(cx), float(cy)



class CD(BaseModel):
    def __init__(self, opt):
        super(CD, self).__init__(opt)
        self.global_step = 0
        self.warmup_steps = 100 # number of steps before starting CD training
        self.accum_steps = 4          # example: accumulate 4 micro-batches
        self._accum_ctr = 0 
        self.use_amp      = True
        self.scaler       = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        # define network and load pretrained models
        self.netCD = self.set_device(networks.define_CD(opt))
        self.netReg = self.set_device(RegistrationModule())
#        for param in self.netReg.gp.parameters():
#            param.requires_grad = False

        # set loss and load resume state
        self.loss_type = opt['model_cd']['loss_type']
        if self.loss_type == 'ce':
            self.loss_func =nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplementedError()
        for name, param in self.netReg.named_parameters():
            if param.requires_grad:
                print(f"Training: {name}")
            else:
                print(f"Frozen: {name}")

        if self.opt['phase'] == 'train':
            self.netCD.train()
            self.netReg.train()
            optim_reg_params = list(self.netReg.parameters())
            self.optReg = torch.optim.Adam(optim_reg_params, lr=opt['train']["optimizer"]["lr"])
            # find the parameters to optimize
            optim_cd_params = list(self.netCD.parameters())

            if opt['train']["optimizer"]["type"] == "adam":
                self.optCD = torch.optim.Adam(
                    optim_cd_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optCD = torch.optim.AdamW(
                    optim_cd_params, lr=opt['train']["optimizer"]["lr"])
            else:
                raise NotImplementedError(
                    'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))
            self.log_dict = OrderedDict()
            
            #Define learning rate sheduler
            self.exp_lr_scheduler_netCD = get_scheduler(optimizer=self.optCD, args=opt['train'])
        else:
            self.netCD.eval()
            self.netReg.eval()
            self.log_dict = OrderedDict()

        self.load_network()
        self.print_network()

        self.running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        self.len_train_dataloader = opt["len_train_dataloader"]
        self.len_val_dataloader = opt["len_val_dataloader"]

    # Feeding all data to the CD model
    def feed_data(self, feats_A, feats_B, data):
        self.feats_A = feats_A
        self.feats_B = feats_B
        self.data = self.set_device(data)
        self.flow_gt = data["flow"].to(self.device) if "flow" in data else None

    def optimize_parameters(self):
        # -------- config --------
        warmup_steps  = int(self.warmup_steps)     # use the __init__ field
        cd_ramp_steps = 1500
        base_lambda_cd = 6.0
        epe_w = 0.2
        clip_max = 5.0
        accum_steps = max(1, getattr(self, "accum_steps", 1))

        self.netReg.train()
        self.netCD.train()

        first_micro = (self._accum_ctr == 0)
        if first_micro:
            self.optReg.zero_grad(set_to_none=True)
            self.optCD.zero_grad(set_to_none=True)

        # ----- forward -----
        fA = self.feats_A[0][11]
        fB = self.feats_B[0][11]

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # registration
            self.pred_flow_coarse, self.logits = self.netReg(fA, fB)  # B→A flow, logits [B, r*r+1, Hc, Wc]
            print(" $$$$$$$4 shape of pred and gt  $$$$$44", self.pred_flow_coarse.shape , self.data["flow"].shape )
            # print("shape of coarse flow and gt flow",self.pred_flow_coarse.shape ,  )
            # if self._accum_ctr == 0 and self.global_step % 10 == 0:
            #     self._probe_shift(fA, fB, tag=f"#gs{self.global_step}")
           
            # warp **B→A** for CD
            warped_feats_B = warp_features_all(self.feats_B, self.pred_flow_coarse.detach().clone())

            # ---- flow loss (soft labels + soft-argmax EPE) ----
            flow_out = gm_cls_loss_from_gt(
                gt_flow=self.data["flow"],
                logits=self.logits,
                sigma_bins=1.0,
                temperature=getattr(self.netReg, "temperature", None),
                align_corners=True
            )
            flow_ce   = flow_out["gm_cls_loss_fine"]
            flow_epe  = flow_out["epe_norm"]
            flow_loss = flow_ce + 0.2 * flow_epe
            B, C, Hc, Wc = self.logits.shape

            # flow_out = gm_cls_loss_from_gt(
            #     gt_flow=self.data["flow"],          # [B,2,H,W] (your function handles the resize)
            #     scale_gm_cls=self.logits,           # logits [B,C,Hc,Wc]
            #     H=Hc,                               # coarse height
            #     W=Wc,                               # coarse width
            #     scale="fine"
            # )
            # flow_ce   = flow_out["gm_cls_loss_fine"]
            # flow_loss = flow_ce


            # ---- CD branch ----
            if self.global_step >= warmup_steps:
                after_wu  = self.global_step - warmup_steps
                ramp_frac = min(1.0, after_wu / cd_ramp_steps)
                lambda_cd = base_lambda_cd * ramp_frac

                # small stabilization for first ~500 steps after warm-up
                if after_wu < 10:
                    with torch.no_grad():
                        self.pred_cm = self.netCD(self.feats_A, warp_features_all(self.feats_B, self.pred_flow_coarse.detach()))
                else:
                    self.pred_cm = self.netCD(self.feats_A, warped_feats_B)

                l_cd = self.loss_func(self.pred_cm, self.data["L"].long())
            else:
                with torch.no_grad():
                    self.pred_cm = self.netCD(self.feats_A, warped_feats_B)
                    l_cd = self.loss_func(self.pred_cm, self.data["L"].long()).detach()
                lambda_cd = 0.0

            total_loss = flow_loss + lambda_cd * l_cd
            # self._quick_probe(fA, fB)
            # self.self_test_decoder(Hc=8, Wc=8)

        # ----- accumulation -----
        loss_scaled = total_loss / accum_steps
        self.scaler.scale(loss_scaled).backward()
        self._accum_ctr += 1

        # ----- step on real update -----
        if self._accum_ctr >= accum_steps:
            # unscale + clip then step
            self.scaler.unscale_(self.optReg)
            self.scaler.unscale_(self.optCD)
            torch.nn.utils.clip_grad_norm_(self.netReg.parameters(), clip_max)
            # torch.nn.utils.clip_grad_norm_(self.netCD.parameters(), clip_max)

            self.scaler.step(self.optReg)
            if self.global_step >= warmup_steps and (self.global_step - warmup_steps) >= 10:
                self.scaler.step(self.optCD)
            self.scaler.update()

            self.optReg.zero_grad(set_to_none=True)
            self.optCD.zero_grad(set_to_none=True)

            self._accum_ctr = 0
            self.global_step += 1


        # ----- logs (raw, un-divided) -----
        # self.log_dict['l_flow_ce']  = float(flow_ce)
        # self.log_dict['l_flow_epe'] = float(flow_epe) if isinstance(flow_epe, torch.Tensor) else float(flow_epe)
        # self.log_dict['l_flow']     = float(flow_loss)
        # self.log_dict['l_cd']       = float(l_cd)
        # self.log_dict['l_total']    = float(total_loss)
        self.log_dict['l_flow_ce']  = float(flow_ce)
        self.log_dict['l_flow']     = float(flow_loss)
        self.log_dict['l_cd']       = float(l_cd)
        self.log_dict['l_total']    = float(total_loss)
        # If your gm loss returns diagnostics:
        if "no_match_prob" in flow_out:   self.log_dict['no_match_prob']  = float(flow_out["no_match_prob"])
        if "overflow_ratio" in flow_out:  self.log_dict['overflow_ratio'] = float(flow_out["overflow_ratio"])




    # Optimize the parameters of the CD model
    # def optimize_parameters(self):

    #     warmup_steps = 1000
    #     self.netCD.train()
    #     self.netReg.train()
    #     lambda_flow_init = 1.0
    #     lambda_flow_after = 1.0
    #     lambda_cd_after = 0.0
    #     print("[debug] global step ",self.global_step ,"warm up step ", warmup_steps, "self.accum_steps", self.accum_steps,"self._accum_ctr", self._accum_ctr )
    #     # --------- start a new accumulation window? ---------
    #     first_micro = (self._accum_ctr == 0)
    #     if first_micro:
    #         # zero grads once per window (NOT every micro-batch)
    #         self.optCD.zero_grad(set_to_none=True)
    #         self.optReg.zero_grad(set_to_none=True)

    #     # --------- forward (same as before) ---------
    #     fA = self.feats_A[0][11]
    #     fB = self.feats_B[0][11]
    #     H, W = fA.shape[2], fA.shape[3]

    #     # registration / flow head
    #     self.pred_flow_coarse, self.logits = self.netReg(fA, fB)

    #     # warp B’s features using predicted flow
    #     self.warped_feats_A = warp_features_all(self.feats_A, self.pred_flow_coarse)

    #     # flow loss from GT (classification-style head)
    #     flow_loss = gm_cls_loss_from_gt(self.data["flow"], self.logits, H, W)["gm_cls_loss_fine"]

    #     # change-detection branch
    #     if self.global_step >= warmup_steps:
    #         print("After Warm Up!")
    #         lambda_flow = lambda_flow_after
    #         lambda_cd   = lambda_cd_after
    #         self.pred_cm = self.netCD(self.warped_feats_A, self.feats_B)
    #         l_cd = self.loss_func(self.pred_cm, self.data["L"].long())
    #         total_loss = lambda_flow * flow_loss # + lambda_cd * l_cd
    #     else:
    #         # during warmup, DO NOT build CD grads
    #         with torch.no_grad():
    #             self.pred_cm = self.netCD(self.warped_feats_A, self.feats_B)
    #             l_cd = self.loss_func(self.pred_cm, self.data["L"].long()).detach()
    #         lambda_flow = lambda_flow_init
    #         total_loss = lambda_flow * flow_loss

    #     # --------- scale loss for accumulation ---------
    #     total_loss = total_loss / max(1, self.accum_steps)

    #     # --------- backward (accumulate grads) ---------
    #     total_loss.backward()
    #     self._accum_ctr += 1
    #     def grad_norm(module):
    #         vals = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    #         return float(torch.norm(torch.stack(vals)).item()) if vals else 0.0

    #     gn_reg = grad_norm(self.netReg)
    #     gn_cd  = grad_norm(self.netCD)
    #     print(f"[dbg] grad_norm  Reg={gn_reg:.2e}  CD={gn_cd:.2e}")

    #     # --------- conditionally step optimizers ---------
    #     if self._accum_ctr >= max(1, self.accum_steps):
    #         # (optional) clip grads here, e.g.:
    #         # torch.nn.utils.clip_grad_norm_(self.netReg.parameters(), 1.0)
    #         # if self.global_step >= warmup_steps:
    #         #     torch.nn.utils.clip_grad_norm_(self.netCD.parameters(), 1.0)

    #         # step: always Reg; CD only after warmup
    #         self.optReg.step()
    #         # if self.global_step >= warmup_steps:
    #         #     self.optCD.step()
    #         # else: no CD step (and we didn’t accumulate CD grads anyway)

    #         # reset accumulation window & advance "real" step
    #         self._accum_ctr = 0
    #         self.global_step += 1

    #         # ---- logging (raw, un-divided losses) ----
    #     self.log_dict['l_cd']    = float(l_cd)
    #     self.log_dict['l_flow']  = float(flow_loss)
    #     self.log_dict['l_total'] = float(total_loss)

    # Testing on given data

    def _probe_shift(self, fA, fB, tag=""):
        from registration.utils import cls_to_flow_refine, warp_features_all

        # 0) Basic
        H, W = self.data["flow"].shape[-2:]
        gt_full = self.data["flow"].detach().float()
        pred_c  = self.pred_flow_coarse.detach().float()
        logits  = self.logits.detach().float()

        layout = infer_bin_layout(logits.shape[1])   # <-- NEW
        r, side, has_nomatch, parity = layout["r"], layout["side"], layout["has_nomatch"], layout["parity"]
        print(f"\n[SHIFT{tag}] gt {H}x{W} | pred_c={list(pred_c.shape)} | logits C={logits.shape[1]} "
            f"| side={side} ({parity}) has_no_match={has_nomatch} r={r}")

        # 1) Coarse plane (downsample GT correctly)
        gt_c = resize_flow(gt_full, pred_c.shape[-2:], align_corners=True)
        epe_c = epe_mean(pred_c, gt_c)
        bx_c, by_c = mean_bias(pred_c, gt_c)
        print(f"[SHIFT{tag}] Coarse:  EPE={epe_c:.3f}  bias=({bx_c:.3f},{by_c:.3f})  pred|min/max=({pred_c.min():.2f},{pred_c.max():.2f})  gt|min/max=({gt_c.min():.2f},{gt_c.max():.2f})")

        # 2) Upsample to full with proper vector scaling
        pred_full = resize_flow(pred_c, (H, W), align_corners=True)
        epe_f = epe_mean(pred_full, gt_full)
        bx_f, by_f = mean_bias(pred_full, gt_full)
        cx, cy = corr_xy(pred_full, gt_full)
        print(f"[SHIFT{tag}] Full↑:   EPE={epe_f:.3f}  bias=({bx_f:.3f},{by_f:.3f})  corr(x)={cx:.3f} corr(y)={cy:.3f}")

        # 3) Direction check
        epe_flip = epe_mean(-pred_full, gt_full)
        print(f"[SHIFT{tag}] Sign-:   EPE={epe_flip:.3f}  (if << Full↑, your direction is inverted)")

        # 4) Decoder consistency sweep (find best align_corners/r)
        def decode_to_full(al_c, rr):
            flow_c = cls_to_flow_refine(logits, align_corners=al_c, r=rr)
            return resize_flow(flow_c, (H, W), align_corners=al_c)

        best = (1e9, None)
        for al_c in (True, False):
            pf = decode_to_full(al_c, r)
            e = epe_mean(pf, gt_full)
            b0, b1 = mean_bias(pf, gt_full)
            print(f"[SHIFT{tag}] Decode ac={al_c} r={r}: EPE={e:.3f} bias=({b0:.3f},{b1:.3f})")
            if e < best[0]: best = (e, f"ac={al_c}, r={r}")
        # small r-perturb
        if r-1 >= 1:
            pf = decode_to_full(True, r-1); e = epe_mean(pf, gt_full)
            print(f"[SHIFT{tag}] Decode ac=True r={r-1}: EPE={e:.3f}")
            if e < best[0]: best = (e, f"ac=True, r={r-1}")
        pf = decode_to_full(True, r+1); e = epe_mean(pf, gt_full)
        print(f"[SHIFT{tag}] Decode ac=True r={r+1}: EPE={e:.3f}")
        if e < best[0]: best = (e, f"ac=True, r={r+1}")

        print(f"[SHIFT{tag}] BEST decode = {best[1]}  (if not ac=True,r={r}, your decode/loss geometry mismatched)")

        # 5) Warper mutation + alignment check
        sum_before = float(self.pred_flow_coarse.detach().sum())
        _ = warp_features_all(self.feats_B, self.pred_flow_coarse.detach().clone())  # safe
        sum_after  = float(self.pred_flow_coarse.detach().sum())
        print(f"[SHIFT{tag}] Warper mutates flow? {'YES' if abs(sum_after-sum_before)>1e-4 else 'NO'}")

        # 6) Channel swap test (catch dx/dy swapped in some path)
        pred_swapped = pred_full.clone()
        pred_swapped[:,0], pred_swapped[:,1] = pred_full[:,1], pred_full[:,0]
        epe_swapped = epe_mean(pred_swapped, gt_full)
        print(f"[SHIFT{tag}] Swap(x,y): EPE={epe_swapped:.3f}  (if << Full↑, channels are swapped somewhere)")


    def self_test_decoder(self, Hc=8, Wc=8):
        """
        Quick unit test for cls_to_flow_refine bin centers & scale.
        Runs once; no real data needed.
        """
        import torch
        from registration.utils import cls_to_flow_refine

        self.netReg.eval()
        device = self.device

        # (Most setups use 1024-ch features feeding GP (512) + f0 (1024))
        FEAT_CH = 1024

        with torch.no_grad():
            # dummy features on the coarse grid
            fA = torch.zeros(1, FEAT_CH, Hc, Wc, device=device)
            fB = torch.zeros_like(fA)

            # get shapes/r straight from your reg head
            pred_flow, logits = self.netReg(fA, fB)
            C = logits.shape[1]
            r = infer_r_from_logits(logits)
            print(f"[SELFTEST] logits C={C}, inferred r={r}")

            # build synthetic logits with a single bin "on"
            B = 1
            logits_synth = torch.full((B, C, Hc, Wc), -50.0, device=device)

            def bin_idx(dx, dy):
                # bins over [-r..r]x[-r..r], +1 for "no-match"
                return (dx + r) + (2*r + 1) * (dy + r)

            # center bin should decode to ~zero flow
            k0 = bin_idx(0, 0)
            logits_synth[:] = -50.0
            logits_synth[:, k0] = 50.0
            flow_c = cls_to_flow_refine(logits_synth, align_corners=True, r=r)
            m = float(flow_c.abs().max())
            print(f"[SELFTEST] center bin -> max|flow|={m:.6f}  (expect ~0.0)")

            # +x bin should decode to a positive dx on coarse plane
            logits_synth.fill_(-50.0)
            kx = bin_idx(+1, 0)
            logits_synth[:, kx] = 50.0
            flow_c = cls_to_flow_refine(logits_synth, align_corners=True, r=r)
            avg_dx = float(flow_c[:, 0].mean())
            avg_dy = float(flow_c[:, 1].mean())
            print(f"[SELFTEST] +x bin -> avg(dx,dy)=({avg_dx:.4f},{avg_dy:.4f})")

    def test(self):
        self.netCD.eval()
        self.netReg.eval()

        with torch.no_grad():
            fA = self.feats_A[0][11]
            fB = self.feats_B[0][11]
            if isinstance(self.netCD, nn.DataParallel):
                self.pred_cm = self.netCD.module.forward(self.feats_A, self.feats_B)
                self.pred_corase_flow, self.logits = self.netReg(fA, fB)
            else:
                self.pred_cm = self.netCD(self.feats_A, self.feats_B)
                self.pred_corase_flow, self.logits = self.netReg(fA, fB)
            l_cd = self.loss_func(self.pred_cm, self.data["L"].long())
            flow_out = gm_cls_loss_from_gt(
                gt_flow=self.data["flow"],
                logits=self.logits,
                sigma_bins=1.0,
                temperature=getattr(self.netReg, "temperature", None),
                align_corners=True
            )
            flow_ce   = flow_out["gm_cls_loss_fine"]
            flow_epe  = flow_out["epe_norm"]
            flow_loss = flow_ce + 0.2 * flow_epe
            self.log_dict['l_cd'] = l_cd.item()
            self.log_dict['l_flow'] = float(flow_loss)
            self.log_dict['flow_ce'] = float(flow_ce)
            self.log_dict['flow_epe'] = float(flow_epe)

        self.netCD.train()
        self.netReg.train()

    # Get current log
    def get_current_log(self):
        return self.log_dict

    import torch



    # Get current visuals

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred_cm'] = torch.argmax(self.pred_cm, dim=1, keepdim=False)
        out_dict['gt_cm']   = self.data['L']

        # Convert pred (norm, coarse) -> pixels at GT size
        H, W = self.data['flow'].shape[-2:]
        pred_px_full = decode_pred_norm_to_px_full(self.pred_flow_coarse.detach(), H, W, align_corners=True)
        # right after you compute pred_px_full
        print(
            "[viz] pred u/v p99:",
            float(torch.quantile(pred_px_full[:, 0].float().abs().flatten(), 0.99)),
            float(torch.quantile(pred_px_full[:, 1].float().abs().flatten(), 0.99)),
            "| gt u/v p99:",
            float(torch.quantile(self.data['flow'][:, 0].float().abs().flatten(), 0.99)),
            float(torch.quantile(self.data['flow'][:, 1].float().abs().flatten(), 0.99)),
        )



        out_dict['pred_flow'] = pred_px_full      # [B,2,H,W] pixels
        out_dict['gt_flow']   = self.data['flow'] # [B,2,H,W] pixels
        return out_dict


    # Printing the CD network
    def print_network(self):
        s, n = self.get_network_description(self.netCD)
        if isinstance(self.netCD, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netCD.__class__.__name__,
                                             self.netCD.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netCD.__class__.__name__)

        logger.info(
            'Change Detection Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    # Saving the network parameters
    def save_network(self, epoch, is_best_model = False):
        cd_gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'cd_model_E{}_gen.pth'.format(epoch))
        cd_opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'cd_model_E{}_opt.pth'.format(epoch))
        


        reg_gen_path = os.path.join(self.opt['path']['checkpoint'], f'reg_model_E{epoch}_gen.pth')
        reg_opt_path = os.path.join(self.opt['path']['checkpoint'], f'reg_model_E{epoch}_opt.pth')
        if is_best_model:
            best_cd_gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_cd_model_gen.pth'.format(epoch))
            best_cd_opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_cd_model_opt.pth'.format(epoch))
            best_reg_gen_path = os.path.join(self.opt['path']['checkpoint'], 'best_reg_model_gen.pth')
            best_reg_opt_path = os.path.join(self.opt['path']['checkpoint'], 'best_reg_model_opt.pth')

        # Save CD model pareamters
        network = self.netCD
        if isinstance(self.netCD, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, cd_gen_path)
        if is_best_model:
            torch.save(state_dict, best_cd_gen_path)


        # Save CD optimizer paramers
        opt_state = {'epoch': epoch,
                     'scheduler': None, 
                     'optimizer': None}
        opt_state['optimizer'] = self.optCD.state_dict()
        torch.save(opt_state, cd_opt_path)
        if is_best_model:
            torch.save(opt_state, best_cd_opt_path)



        # === Save Registration ===
        reg = self.netReg.module if isinstance(self.netReg, nn.DataParallel) else self.netReg
        torch.save(reg.state_dict(), reg_gen_path)
        torch.save({'epoch': epoch, 'optimizer': self.optReg.state_dict()}, reg_opt_path)

        if is_best_model:
            torch.save(reg.state_dict(), best_reg_gen_path)
            torch.save({'epoch': epoch, 'optimizer': self.optReg.state_dict()}, best_reg_opt_path)

        # Print info
        logger.info(
            'Saved current CD model in [{:s}] ...'.format(cd_gen_path))
        logger.info(f"Saved Registration model to [{reg_gen_path}]")
        if is_best_model:
            logger.info(
            'Saved best CD model in [{:s}] ...'.format(best_cd_gen_path))
            logger.info(f"Saved BEST Registration model to [{best_reg_gen_path}]")

    # Loading pre-trained CD network
    def load_network(self):
        cd_resume = self.opt['path_cd']['resume_state']
        reg_resume = self.opt['path_reg']['resume_state']
        if cd_resume is not None:
            logger.info(
                'Loading pretrained model for CD model [{:s}] ...'.format(cd_resume))
            gen_path = '{}_gen.pth'.format(cd_resume)
            opt_path = '{}_opt.pth'.format(cd_resume)
            
            # change detection model
            network = self.netCD
            if isinstance(self.netCD, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=True)
            
            if self.opt['phase'] == 'train':
                opt = torch.load(opt_path)
                self.optCD.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']


        if reg_resume is not None:
            logger.info(f'Loading pretrained Registration model from [{reg_resume}]...')
            reg_gen = f'{reg_resume}_gen.pth'
            reg_opt = f'{reg_resume}_opt.pth'

            reg_model = self.netReg.module if isinstance(self.netReg, nn.DataParallel) else self.netReg
            reg_model.load_state_dict(torch.load(reg_gen), strict=True)

            if self.opt['phase'] == 'train':
                reg_opt_state = torch.load(reg_opt)
                self.optReg.load_state_dict(reg_opt_state['optimizer'])
                self.begin_epoch = min(self.begin_epoch, reg_opt_state.get('epoch', self.begin_epoch))
    
    # Functions related to computing performance metrics for CD
    def _update_metric(self):
        """
        update metric
        """
        G_pred = self.pred_cm.detach()
        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=self.data['L'].detach().cpu().numpy())
        return current_score
    
    # Collecting status of the current running batch
    def _collect_running_batch_states(self):
        self.running_acc = self._update_metric()
        self.log_dict['running_acc'] = self.running_acc.item()

    def set_registration_model(self, reg_model):
        self.netReg = reg_model.to(self.device)
        self.optReg = torch.optim.Adam(self.netReg.decoder.parameters(), lr=self.opt['train']['lr'])  # only train decoder

    # Collect the status of the epoch
    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.log_dict['epoch_acc'] = self.epoch_acc.item()

        for k, v in scores.items():
            self.log_dict[k] = v
            #message += '%s: %.5f ' % (k, v)

    # Rest all the performance metrics
    def _clear_cache(self):
        self.running_metric.clear()

    # Finctions related to learning rate sheduler
    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_netCD.step()

        
