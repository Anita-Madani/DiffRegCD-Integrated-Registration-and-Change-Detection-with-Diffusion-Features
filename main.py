import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from model.cd_modules.cd_head import cd_head 
from misc.print_diffuse_feats import print_feats
from misc.flow_utils import flow_to_rgb
from flow_vis import flow_to_color


# --- FLOW VAL METRICS (paste near imports) ---
import torch.nn.functional as F
import math

@torch.no_grad()
def flow_val_metrics(logits, gt_flow, pred_norm, *, align_corners=True, temperature=None):
    """
    logits:   [B, C, Hc, Wc]  (C=r*r or r*r+1)
    gt_flow:  [B, 2, H_in, W_in] in *pixels* (input size)
    pred_norm:[B, 2, Hc, Wc] in [-1,1] (soft-argmax output at coarse size)
    """
    B, C, Hc, Wc = logits.shape
    device, dtype = logits.device, logits.dtype

    # infer r and has_nomatch
    r_sq = int(round(C ** 0.5))
    if r_sq * r_sq == C:
        r = r_sq; has_nomatch = False; C_eff = C
    else:
        r_sq1 = int(round((C - 1) ** 0.5))
        assert r_sq1 * r_sq1 + 1 == C, f"C={C} not r^2 or r^2+1"
        r = r_sq1; has_nomatch = True; C_eff = r * r

    # resize GT to coarse grid, then pixels -> normalized (w.r.t input H_in,W_in)
    gt_rs = F.interpolate(gt_flow.to(dtype), size=(Hc, Wc), mode='bilinear', align_corners=align_corners)
    H_in, W_in = gt_flow.shape[-2:]
    if align_corners:
        sx = 2.0 / (W_in - 1) if W_in > 1 else 0.0
        sy = 2.0 / (H_in - 1) if H_in > 1 else 0.0
    else:
        sx = 2.0 / W_in if W_in > 0 else 0.0
        sy = 2.0 / H_in if H_in > 0 else 0.0

    gt_norm = torch.empty_like(gt_rs)
    gt_norm[:, 0] = gt_rs[:, 0] * sx
    gt_norm[:, 1] = gt_rs[:, 1] * sy

    # valid mask (same as loss overflow)
    overflow = (gt_norm.abs().max(dim=1, keepdim=True).values > 1)   # [B,1,Hc,Wc]
    valid = (~overflow)[:, 0]                                        # [B,Hc,Wc]
    denom = valid.sum().clamp_min(1)

    # norm-space EPE (scale-invariant sanity check)
    epe_norm = ((pred_norm - gt_norm).abs().sum(1)[valid]).mean().item()

    # pixel-space AEPE and thresholds
    dx_px = (pred_norm[:, 0] - gt_norm[:, 0]) / (sx + 1e-12)
    dy_px = (pred_norm[:, 1] - gt_norm[:, 1]) / (sy + 1e-12)
    epe_px = torch.sqrt(dx_px**2 + dy_px**2)

    aepe_px = (epe_px[valid]).mean().item()
    acc1 = ((epe_px < 1.0) & valid).float().sum().div(denom).item()
    acc3 = ((epe_px < 3.0) & valid).float().sum().div(denom).item()
    acc5 = ((epe_px < 5.0) & valid).float().sum().div(denom).item()

    # Fl-all (KITTI)
    mag_px = torch.sqrt((gt_norm[:,0]/(sx+1e-12))**2 + (gt_norm[:,1]/(sy+1e-12))**2)
    thr = torch.maximum(torch.full_like(mag_px, 3.0), 0.05 * mag_px)
    fl_all = (((epe_px > thr) & valid).float().sum() / denom).item()

    # bin accuracy (needs the same r×r lattice)
    ys = torch.linspace(-1.0, 1.0, r, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, r, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    code = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).view(1, r*r, 2, 1, 1)

    gtv = gt_norm.unsqueeze(1)
    d2  = ((gtv - code)**2).sum(dim=2)                # [B,r*r,Hc,Wc]
    gt_idx   = d2.argmin(dim=1)                       # [B,Hc,Wc]
    pred_idx = logits[:, :r*r].argmax(dim=1)          # [B,Hc,Wc]
    bin_acc = ((pred_idx == gt_idx) & valid).float().sum().div(denom).item()

    # no-match stats if present (GT = overflow)
    no_match_prob = 0.0; no_match_prec=no_match_rec=no_match_f1=0.0
    if C == r*r + 1:
        probs = torch.softmax(logits, dim=1)
        no = probs[:, -1] > 0.5
        gt_no = overflow[:, 0]
        tp = ((no & gt_no) & valid).sum().float()
        fp = ((no & ~gt_no) & valid).sum().float()
        fn = (((~no) & gt_no) & valid).sum().float()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2*prec*rec / (prec + rec + 1e-9)
        no_match_prob = probs[:, -1][valid].mean().item()
        no_match_prec, no_match_rec, no_match_f1 = prec.item(), rec.item(), f1.item()

    return {
        "aepe_px": aepe_px,
        "epe_norm": epe_norm,
        "acc@1px": acc1,
        "acc@3px": acc3,
        "acc@5px": acc5,
        "fl_all": fl_all,
        "bin_acc": bin_acc,
        "no_match_prob": no_match_prob,
        "no_match_precision": no_match_prec,
        "no_match_recall": no_match_rec,
        "no_match_f1": no_match_f1,
        "overflow_ratio": overflow.float().mean().item(),
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ddpm_cd.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run either train(training + validation) or testing', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('test', opt['path']['log'], 'test', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        print("Initializing wandblog.")
        wandb_logger = WandbLogger(opt)
        # Training log
        wandb.define_metric('epoch')
        wandb.define_metric('training/train_step')
        wandb.define_metric("training/*", step_metric="train_step")
        # Validation log
        wandb.define_metric('validation/val_step')
        wandb.define_metric("validation/*", step_metric="val_step")
        # Initialization
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None

    # Loading change-detction datasets.
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'test':
            print("Creating [train] change-detection dataloader.")
            train_set   = Data.create_cd_dataset(dataset_opt, phase)
            train_loader= Data.create_dataloader(
                train_set, dataset_opt, phase)
            opt['len_train_dataloader'] = len(train_loader)

        elif phase == 'val' and args.phase != 'test':
            print("Creating [val] change-detection dataloader.")
            val_set   = Data.create_cd_dataset(dataset_opt, phase)
            val_loader= Data.create_cd_dataloader(
                val_set, dataset_opt, phase)
            opt['len_val_dataloader'] = len(val_loader)
        
        elif phase == 'test' and args.phase == 'test':
            print("Creating [test] change-detection dataloader.")
            print(phase)
            test_set   = Data.create_cd_dataset(dataset_opt, phase)
            test_loader= Data.create_cd_dataloader(
                test_set, dataset_opt, phase)
            opt['len_test_dataloader'] = len(test_loader)
    
    logger.info('Initial Dataset Finished')

    # Loading diffusion model
    diffusion = Model.create_model(opt)
    logger.info('Initial Diffusion Model Finished')

    # Set noise schedule for the diffusion model
    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    # Creating change-detection model
    change_detection = Model.create_CD_model_256(opt)
    
    #################
    # Training loop #
    #################
    n_epoch = opt['train']['n_epoch']
    best_mF1 = 0.0
    start_epoch = 0
    if opt['phase'] == 'train':
        for current_epoch in range(start_epoch, n_epoch):         
            change_detection._clear_cache()
            train_result_path = '{}/train/{}'.format(opt['path']
                                                 ['results'], current_epoch)
            os.makedirs(train_result_path, exist_ok=True)
            
            ################
            ### training ###
            ################
            message = 'lr: %0.7f\n \n' % change_detection.optCD.param_groups[0]['lr']
            logger.info(message)
            for current_step, train_data in enumerate(train_loader):
                # Feeding data to diffusion model and get features
                diffusion.feed_data(train_data)

                f_A=[]
                f_B=[]
                for t in opt['model_cd']['t']:
                    fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                    if opt['model_cd']['feat_type'] == "dec":
                        f_A.append(fd_A_t)
                        f_B.append(fd_B_t)
                        # Uncommet the following line to visualize features from the diffusion model
                        # for level in range(0, len(fd_A_t)):
                        #     print_feats(opt=opt, train_data=train_data, feats_A=fd_A_t, feats_B=fd_B_t, level=level, t=t)
                        # del fe_A_t, fe_B_t
                    else:
                        f_A.append(fe_A_t)
                        f_B.append(fe_B_t)
                        del fd_A_t, fd_B_t
                
                # for i in range(0, len(fd_A)):
                #     print(fd_A[i].shape)

                # Feeding features from the diffusion model to the CD model
                change_detection.feed_data(f_A, f_B, train_data)
                change_detection.optimize_parameters()
                train_step += 1
                change_detection._collect_running_batch_states()

                # log running batch status
                if current_step % opt['train']['train_print_freq'] == 0:
                    # message
                    logs = change_detection.get_current_log()
                    message = '[Training CD]. epoch: [%d/%d]. Itter: [%d/%d], CD_loss: %.5f, running_mf1: %.5f\n' %\
                      (current_epoch, n_epoch-1, current_step, len(train_loader), logs['l_cd'], logs['running_acc'])
                    logger.info(message)

                    #vissuals
                    visuals = change_detection.get_current_visuals()

                    img_mode = "single"
                    if img_mode == "single":
                        # Converting to uint8
                        img_A   = Metrics.tensor2img(train_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        img_B   = Metrics.tensor2img(train_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                        gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                        pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                        
                        pred_flow_rgb = visuals['pred_flow']
#                        print("%%%%%%%%%% shape of the predicted flow before vis %%%%%%% ", pred_flow_rgb.shape, visuals['pred_flow'].shape )
                        gt_flow_rgb = visuals['gt_flow']
                        # pred_flow_np = pred_flow_rgb[0].permute(2, 0, 1).detach().cpu().numpy()
                        # pred_color = flow_to_color(pred_flow_np.transpose(1, 2, 0))
                        # # print("shape of the gt flow",  gt_flow_rgb.shape)
                        # gt_flow_np = gt_flow_rgb[0].detach().cpu().numpy()
                        # # print("shape of the np flow", gt_flow_np.shape)
                        # gt_flow_color = flow_to_color(gt_flow_np.transpose(1, 2, 0))
                        # print("the pred shape ",  pred_color.shape )
                        def _to_hw2(t):
                            # t: (H,W,2) or (2,H,W)  ->  (H,W,2) on CPU numpy
                            t = t.detach().cpu()
                            if t.dim() == 3 and t.shape[-1] == 2:
                                return t.numpy()
                            elif t.dim() == 3 and t.shape[0] == 2:
                                return t.permute(1, 2, 0).numpy()
                            else:
                                raise ValueError(f"Unexpected flow shape {tuple(t.shape)}")

                        def _stack_two(imgs, pad=6):
                            # stack [img0, img1] vertically with padding; if one, just return it
                            if len(imgs) == 1:
                                return imgs[0]
                            H, W, _ = imgs[0].shape
                            out = np.zeros((H*2 + pad, W, 3), dtype=np.uint8)
                            out[0:H, :W] = imgs[0]
                            out[H+pad:H*2+pad, :W] = imgs[1]
                            return out

                        # Build colorized flows for up to two samples
                        pred_colors = []
                        gt_colors   = []
                        for i in range(min(2, pred_flow_rgb.shape[0])):
                            pred_colors.append(flow_to_color(_to_hw2(pred_flow_rgb[i])))
                            gt_colors.append(flow_to_color(_to_hw2(gt_flow_rgb[i])))

                        # Single collaged images (use these names so your save calls work)
                        pred_color    = _stack_two(pred_colors)   # one image with 2 samples
                        gt_flow_color = _stack_two(gt_colors) 

                        #save imgs
                        Metrics.save_img(
                                img_A, '{}/img_A_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                img_B, '{}/img_B_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                pred_cm, '{}/img_pred_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(
                                gt_cm, '{}/img_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                        Metrics.save_img(pred_color, f'{train_result_path}/flow_pred_rgb_e{current_epoch}_b{current_step}.png')
                        Metrics.save_img(gt_flow_color, f'{train_result_path}/flow_gt_rgb_e{current_epoch}_b{current_step}.png')
                    else:
                        # grid img
                        visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                        visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                        grid_img = torch.cat((  train_data['A'], 
                                    train_data['B'], 
                                    visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                    visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                    dim = 0)
                        grid_img = Metrics.tensor2img(grid_img)  # uint8
                        Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_e{}_b{}.png'.format(train_result_path, current_epoch, current_step))
                
            ### log epoch status ###
            change_detection._collect_epoch_states()
            logs = change_detection.get_current_log()
            message = '[Training CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                      (current_epoch, n_epoch-1, logs['epoch_acc'])
            for k, v in logs.items():
                message += '{:s}: {:.4e} '.format(k, v)
                tb_logger.add_scalar(k, v, current_step)
            message += '\n'
            logger.info(message)
            
            if wandb_logger:
                    wandb_logger.log_metrics({
                        'training/mF1': logs.get('epoch_acc', 0),
                        'training/mIoU': logs.get('miou',0),
                        'training/OA': logs.get('acc',0),
                        'training/change-F1': logs['F1_1'],
                        'training/no-change-F1': logs['F1_0'],
                        'training/change-IoU': logs['iou_1'],
                        'training/no-change-IoU': logs['iou_0'],
                        'training/l_cd': logs.get('l_cd', 0),
                        'training/l_flow': logs.get('l_flow', 0),
                        'training/l_total': logs.get('l_total', 0),
                        'training/train_step': current_epoch
                    })
                    wandb_logger._wandb.log({"train_step": train_step})

            change_detection._clear_cache()
            change_detection._update_lr_schedulers()
            
            ##################
            ### validation ###
            ##################
            if current_epoch % opt['train']['val_freq'] == 0:
                val_result_path = '{}/val/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                os.makedirs(val_result_path, exist_ok=True)

                for current_step, val_data in enumerate(val_loader):
                    # Feed data to diffusion model
                    diffusion.feed_data(val_data)
                    f_A=[]
                    f_B=[]
                    for t in opt['model_cd']['t']:
                        fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                        if opt['model_cd']['feat_type'] == "dec":
                            f_A.append(fd_A_t)
                            f_B.append(fd_B_t)
                            del fe_A_t, fe_B_t
                        else:
                            f_A.append(fe_A_t)
                            f_B.append(fe_B_t)
                            del fd_A_t, fd_B_t

                    # Feed data to CD model
                    change_detection.feed_data(f_A, f_B, val_data)
                    change_detection.test()
                    val_step += 1
                    change_detection._collect_running_batch_states()
                    
                    # log running batch status for val data
                    if current_step % opt['train']['val_print_freq'] == 0:
                        # message
                        logs        = change_detection.get_current_log()
                        message     = '[Validation CD]. epoch: [%d/%d]. Itter: [%d/%d], running_mf1: %.5f\n' %\
                                    (current_epoch, n_epoch-1, current_step, len(val_loader), logs['running_acc'])
                        logger.info(message)

                        #vissuals
                        visuals = change_detection.get_current_visuals()

                        img_mode = "single"
                        if img_mode == "single":
                            # Converting to uint8
                            img_A   = Metrics.tensor2img(val_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                            img_B   = Metrics.tensor2img(val_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                            gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                            pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                            pred_flow_rgb = visuals['pred_flow']
                            gt_flow_rgb = visuals['gt_flow']

                            def _to_hw2(t):
                                # t: (H,W,2) or (2,H,W)  ->  (H,W,2) on CPU numpy
                                t = t.detach().cpu()
                                if t.dim() == 3 and t.shape[-1] == 2:
                                    return t.numpy()
                                elif t.dim() == 3 and t.shape[0] == 2:
                                    return t.permute(1, 2, 0).numpy()
                                else:
                                    raise ValueError(f"Unexpected flow shape {tuple(t.shape)}")

                            def _stack_two(imgs, pad=6):
                                # stack [img0, img1] vertically with padding; if one, just return it
                                if len(imgs) == 1:
                                    return imgs[0]
                                H, W, _ = imgs[0].shape
                                out = np.zeros((H*2 + pad, W, 3), dtype=np.uint8)
                                out[0:H, :W] = imgs[0]
                                out[H+pad:H*2+pad, :W] = imgs[1]
                                return out

                            # Build colorized flows for up to two samples
                            pred_colors = []
                            gt_colors   = []
                            for i in range(min(2, pred_flow_rgb.shape[0])):
                                pred_colors.append(flow_to_color(_to_hw2(pred_flow_rgb[i])))
                                gt_colors.append(flow_to_color(_to_hw2(gt_flow_rgb[i])))

                            # Single collaged images (use these names so your save calls work)
                            pred_color    = _stack_two(pred_colors)   # one image with 2 samples
                            gt_flow_color = _stack_two(gt_colors) 

                            #save imgs
                            Metrics.save_img(
                                img_A, '{}/img_A_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                img_B, '{}/img_B_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                pred_cm, '{}/img_pred_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(
                                gt_cm, '{}/img_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))
                            Metrics.save_img(pred_color, f'{val_result_path}/flow_pred_rgb_e{current_epoch}_b{current_step}.png')
                            Metrics.save_img(gt_flow_color, f'{val_result_path}/flow_gt_rgb_e{current_epoch}_b{current_step}.png')
                        else:
                            # grid img
                            visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                            visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                            grid_img = torch.cat((  val_data['A'], 
                                        val_data['B'], 
                                        visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                        visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                        dim = 0)
                            grid_img = Metrics.tensor2img(grid_img)  # uint8
                            Metrics.save_img(
                            grid_img, '{}/img_A_B_pred_gt_e{}_b{}.png'.format(val_result_path, current_epoch, current_step))

                change_detection._collect_epoch_states()
                logs     = change_detection.get_current_log()
                message = '[Validation CD (epoch summary)]: epoch: [%d/%d]. epoch_mF1=%.5f \n' %\
                      (current_epoch, n_epoch-1, logs['epoch_acc'])
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    tb_logger.add_scalar(k, v, current_step)
                message += '\n'
                logger.info(message)

                if wandb_logger:
                    wandb_logger.log_metrics({
                        'validation/mF1': logs['epoch_acc'],
                        'validation/mIoU': logs['miou'],
                        'validation/OA': logs['acc'],
                        'validation/change-F1': logs['F1_1'],
                        'validation/no-change-F1': logs['F1_0'],
                        'validation/change-IoU': logs['iou_1'],
                        'validation/no-change-IoU': logs['iou_0'],
                        'validation/flow_loss': logs['l_flow'],
                        'validation/flow_epe': logs['flow_epe'],
                        'validation/flow_ce': logs['flow_ce'],
                        'validation/val_step': current_epoch               
                    })
                    wandb_logger._wandb.log({"val_step": val_step})
                
                if logs['epoch_acc'] > best_mF1:
                    is_best_model = True
                    best_mF1 = logs['epoch_acc']
                    logger.info('[Validation CD] Best model updated. Saving the models (current + best) and training states.')
                else:
                    is_best_model = False
                    logger.info('[Validation CD]Saving the current cd model and training states.')
                logger.info('--- Proceed To The Next Epoch ----\n \n')

                change_detection.save_network(current_epoch, is_best_model = is_best_model)
                change_detection._clear_cache()

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})
                
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation (testing).')
        test_result_path = '{}/test/'.format(opt['path']
                                                 ['results'])
        os.makedirs(test_result_path, exist_ok=True)
        logger_test = logging.getLogger('test')  # test logger
        change_detection._clear_cache()
        for current_step, test_data in enumerate(test_loader):
            # Feed data to diffusion model
            diffusion.feed_data(test_data)
            f_A=[]
            f_B=[]
            for t in opt['model_cd']['t']:
                fe_A_t, fd_A_t, fe_B_t, fd_B_t = diffusion.get_feats(t=t) #np.random.randint(low=2, high=8)
                if opt['model_cd']['feat_type'] == "dec":
                    f_A.append(fd_A_t)
                    f_B.append(fd_B_t)
                    del fe_A_t, fe_B_t
                else:
                    f_A.append(fe_A_t)
                    f_B.append(fe_B_t)
                    del fd_A_t, fd_B_t

            # Feed data to CD model
            change_detection.feed_data(f_A, f_B, test_data)
            change_detection.test()
            change_detection._collect_running_batch_states()

            # Logs
            logs        = change_detection.get_current_log()
            message     = '[Testing CD]. Itter: [%d/%d], running_mf1: %.5f\n' %\
                                    (current_step, len(test_loader), logs['running_acc'])
            logger_test.info(message)

            # Vissuals
            visuals = change_detection.get_current_visuals()
            img_mode = 'single'
            if img_mode == 'single':
                # Converting to uint8
                visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                img_A   = Metrics.tensor2img(test_data['A'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                img_B   = Metrics.tensor2img(test_data['B'], out_type=np.uint8, min_max=(-1, 1))  # uint8
                gt_cm   = Metrics.tensor2img(visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8
                pred_cm = Metrics.tensor2img(visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), out_type=np.uint8, min_max=(0, 1))  # uint8

                pred_flow_rgb = visuals['pred_flow']
                gt_flow_rgb = visuals['gt_flow']

                def _to_hw2(t):
                                # t: (H,W,2) or (2,H,W)  ->  (H,W,2) on CPU numpy
                    t = t.detach().cpu()
                    if t.dim() == 3 and t.shape[-1] == 2:
                        return t.numpy()
                    elif t.dim() == 3 and t.shape[0] == 2:
                        return t.permute(1, 2, 0).numpy()
                    else:
                        raise ValueError(f"Unexpected flow shape {tuple(t.shape)}")

                def _stack_two(imgs, pad=6):
                    # stack [img0, img1] vertically with padding; if one, just return it
                    if len(imgs) == 1:
                        return imgs[0]
                    H, W, _ = imgs[0].shape
                    out = np.zeros((H*2 + pad, W, 3), dtype=np.uint8)
                    out[0:H, :W] = imgs[0]
                    out[H+pad:H*2+pad, :W] = imgs[1]
                    return out

                # Build colorized flows for up to two samples
                pred_colors = []
                gt_colors   = []
                for i in range(min(2, pred_flow_rgb.shape[0])):
                    pred_colors.append(flow_to_color(_to_hw2(pred_flow_rgb[i])))
                    gt_colors.append(flow_to_color(_to_hw2(gt_flow_rgb[i])))

                # Single collaged images (use these names so your save calls work)
                pred_color    = _stack_two(pred_colors)   # one image with 2 samples
                gt_flow_color = _stack_two(gt_colors) 

                # Save imgs
                Metrics.save_img(
                    img_A, '{}/img_A_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    img_B, '{}/img_B_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    pred_cm, '{}/img_pred_cm{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    gt_cm, '{}/img_gt_cm{}.png'.format(test_result_path, current_step))
    
                Metrics.save_img(pred_color, f'{test_result_path}/flow_pred_rgb_b{current_step}.png')
                Metrics.save_img(gt_flow_color, f'{test_result_path}/flow_gt_rgb_b{current_step}.png')
            else:
                # grid img
                visuals['pred_cm'] = visuals['pred_cm']*2.0-1.0
                visuals['gt_cm'] = visuals['gt_cm']*2.0-1.0
                grid_img = torch.cat((  test_data['A'], 
                                        test_data['B'], 
                                        visuals['pred_cm'].unsqueeze(1).repeat(1, 3, 1, 1), 
                                        visuals['gt_cm'].unsqueeze(1).repeat(1, 3, 1, 1)),
                                        dim = 0)
                grid_img = Metrics.tensor2img(grid_img)  # uint8
                Metrics.save_img(
                    grid_img, '{}/img_A_B_pred_gt_{}.png'.format(test_result_path, current_step))

        change_detection._collect_epoch_states()
        logs = change_detection.get_current_log()
        message = '[Test CD summary]: Test mF1=%.5f \n' %\
                      (logs['epoch_acc'])
        for k, v in logs.items():
            message += '{:s}: {:.4e} '.format(k, v)
            message += '\n'
        logger_test.info(message)

        if wandb_logger:
            wandb_logger.log_metrics({
                'test/mF1': logs['epoch_acc'],
                'test/mIoU': logs['miou'],
                'test/OA': logs['acc'],
                'test/change-F1': logs['F1_1'],
                'test/no-change-F1': logs['F1_0'],
                'test/change-IoU': logs['iou_1'],
                'test/no-change-IoU': logs['iou_0'],
                'test/flow_loss': logs['l_flow'],
                'test/flow_epe': logs['flow_epe'],
                'test/flow_ce': logs['flow_ce'],
            })

        logger.info('End of testing...')
