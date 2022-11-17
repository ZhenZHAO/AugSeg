import argparse
import yaml
import os, sys
import os.path as osp
import pprint
import time
import pickle

import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from augseg.dataset.augs_ALIA import cut_mix_label_adaptive
from augseg.dataset.builder import get_loader
from augseg.models.model_helper import ModelBuilder
from augseg.utils.dist_helper import setup_distributed
from augseg.utils.loss_helper import get_criterion, compute_unsupervised_loss_by_threshold
from augseg.utils.lr_helper import get_optimizer, get_scheduler
from augseg.utils.utils import AverageMeter, intersectionAndUnion, load_state
from augseg.utils.utils import init_log, get_rank, get_world_size, set_random_seed, setup_default_logging

import warnings 
warnings.filterwarnings('ignore')


def main(in_args):
    args = in_args
    if args.seed is not None:
        # print("set random seed to", args.seed)
        set_random_seed(args.seed, deterministic=True)
        # set_random_seed(args.seed)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    rank, word_size = setup_distributed(port=args.port)

    ###########################
    # 1. output settings
    ###########################
    cfg["exp_path"] = osp.dirname(args.config)
    cfg["save_path"] = osp.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])
    cfg["log_path"] = osp.join(cfg["exp_path"], "log")
    flag_use_tb = cfg["saver"]["use_tb"]
    
    if not os.path.exists(cfg["log_path"]) and rank == 0:
        os.makedirs(cfg["log_path"])
    if not osp.exists(cfg["save_path"]) and rank == 0:
        os.makedirs(cfg["save_path"])
    # my favorate: logs
    if rank == 0:
        logger, curr_timestr = setup_default_logging("global", cfg["log_path"])
        csv_path = os.path.join(cfg["log_path"], "seg_{}_stat.csv".format(curr_timestr))
    else:
        logger, curr_timestr = None, ""
        csv_path = None
    # tensorboard
    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        if flag_use_tb:
            tb_logger = SummaryWriter(
                osp.join(cfg["log_path"], "events_seg",curr_timestr)
            )
        else:
            tb_logger = None
    else:
        tb_logger = None
    # make sure all folders and csv handler are correctly created on rank ==0.
    dist.barrier()

    ###########################
    # 2. prepare model 1
    ###########################
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    modules_head = [model.decoder]
    if cfg["net"].get("aux_loss", False):
        modules_head.append(model.auxor)
    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    ###########################
    # 3. data
    ###########################
    sup_loss_fn = get_criterion(cfg)
    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=args.seed)

    ##############################
    # 4. optimizer & scheduler
    ##############################
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )
    optimizer = get_optimizer(params_list, cfg_optim)

    ###########################
    # 5. prepare model more
    ###########################
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Teacher model -- freeze training
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher.cuda()
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    for p in model_teacher.parameters():
        p.requires_grad = False

    # initialize teacher model -- not neccesary if using warmup
    with torch.no_grad():
        for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
            t_params.data = s_params.data

    ######################################
    # 6. resume
    ######################################
    last_epoch = 0
    best_prec = 0
    best_epoch = -1
    best_prec_stu = 0
    best_epoch_stu = -1
    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    ######################################
    # 7. training loop
    ######################################
    if rank == 0:
        logger.info('-------------------------- start training --------------------------')
    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training
        res_loss_sup, res_loss_unsup = train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            tb_logger,
            logger,
            cfg
        )

        # Validation and store checkpoint
        if "cityscapes" in cfg["dataset"].get("type", "pascal"):
            if epoch % 10 == 0 or epoch > (cfg_trainer["epochs"]-50):
                if cfg_trainer.get("evaluate_student", True):
                    prec_stu = validate_citys(model, val_loader, epoch, logger, cfg)
                else:
                    prec_stu =-1000.0
                prec_tea = validate_citys(model_teacher, val_loader, epoch, logger, cfg)
                prec = prec_tea
            else:
                prec_stu = -1000.0
                prec_tea = -1000.0
                prec = prec_tea
        else:
            if cfg_trainer.get("evaluate_student", True):
                prec_stu = validate(model, val_loader, epoch, logger, cfg)
            else:
                prec_stu = -1000.0
            prec_tea = validate(model_teacher, val_loader, epoch, logger, cfg)
            prec = prec_tea

        if rank == 0:
            state = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "teacher_state": model_teacher.state_dict(),
                "best_miou": best_prec,
            }
            if prec_stu > best_prec_stu:
                best_prec_stu = prec_stu
                best_epoch_stu = epoch

            if prec > best_prec:
                best_prec = prec
                best_epoch = epoch
                state["best_miou"] = prec
                torch.save(state, osp.join(cfg["save_path"], "ckpt_best.pth"))

            torch.save(state, osp.join(cfg["save_path"], "ckpt.pth"))
            # save statistics
            tmp_results = {
                        'loss_lb': res_loss_sup,
                        'loss_ub': res_loss_unsup,
                        'miou_stu': prec_stu,
                        'miou_tea': prec_tea,
                        "best": best_prec,
                        "best-stu":best_prec_stu}
            data_frame = pd.DataFrame(data=tmp_results, index=range(epoch, epoch+1))
            if epoch > 0 and osp.exists(csv_path):
                data_frame.to_csv(csv_path, mode='a', header=None, index_label='epoch')
            else:
                data_frame.to_csv(csv_path, index_label='epoch')
            
            logger.info(" <<Test>> - Epoch: {}.  MIoU: {:.2f}/{:.2f}.  \033[34mBest-STU:{:.2f}/{}  \033[31mBest-EMA: {:.2f}/{}\033[0m".format(epoch, 
                prec_stu * 100, prec_tea * 100, best_prec_stu * 100, best_epoch_stu, best_prec * 100, best_epoch))
            if tb_logger is not None:
                tb_logger.add_scalar("mIoU val", prec, epoch)





def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    tb_logger,
    logger,
    cfg,
):

    ema_decay_origin = cfg["net"]["ema_decay"]
    rank, world_size = dist.get_rank(), dist.get_world_size()
    flag_extra_weak = cfg["trainer"]["unsupervised"].get("flag_extra_weak", False)
    model.train()
    
    # data loader
    loader_l.sampler.set_epoch(epoch)
    loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(loader_u), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, mixmatch!"

    # metric indicators
    sup_losses = AverageMeter(20)
    uns_losses = AverageMeter(20)
    batch_times = AverageMeter(20)
    learning_rates = AverageMeter(20)
    meter_high_pseudo_ratio = AverageMeter(20)
    
    # print freq 8 times for a epoch
    print_freq = len(loader_u) // 8 # 8 for semi 4 for sup
    print_freq_lst = [i * print_freq for i in range(1,8)]
    print_freq_lst.append(len(loader_u) -1)

    # start iterations
    model.train()
    model_teacher.eval()
    for step in range(len(loader_l)):
        batch_start = time.time()

        i_iter = epoch * len(loader_l) + step # total iters till now
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step() # lr is updated at the iteration level

        # obtain labeled and unlabeled data
        _, image_l, label_l = loader_l_iter.next()
        image_l, label_l = image_l.cuda(), label_l.cuda()
        _, image_u_weak, image_u_aug, _ = loader_u_iter.next()
        image_u_weak, image_u_aug = image_u_weak.cuda(), image_u_aug.cuda()
        
        # start the training
        if epoch < cfg["trainer"].get("sup_only_epoch", 0):
            # forward
            pred, aux = model(image_l)
            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                sup_loss = sup_loss_fn([pred, aux], label_l)
                del aux
            else:
                sup_loss = sup_loss_fn(pred, label_l)
                del pred

            # no unlabeled data during the warmup period
            unsup_loss = torch.tensor(0.0).cuda()
            pseduo_high_ratio = torch.tensor(0.0).cuda()

        else:
            # 1. generate pseudo labels
            p_threshold = cfg["trainer"]["unsupervised"].get("threshold", 0.95)
            with torch.no_grad():
                model_teacher.eval()
                pred_u, _ = model_teacher(image_u_weak.detach())
                pred_u = F.softmax(pred_u, dim=1)
                # obtain pseudos
                logits_u_aug, label_u_aug = torch.max(pred_u, dim=1)
                
                # obtain confidence
                entropy = -torch.sum(pred_u * torch.log(pred_u + 1e-10), dim=1)
                entropy /= np.log(cfg["net"]["num_classes"])
                confidence = 1.0 - entropy
                confidence = confidence * logits_u_aug
                confidence = confidence.mean(dim=[1,2])  # 1*C
                confidence = confidence.cpu().numpy().tolist()
                # confidence = logits_u_aug.ge(p_threshold).float().mean(dim=[1,2]).cpu().numpy().tolist()
                del pred_u
            model.train()
            
            # 2. apply cutmix
            trigger_prob = cfg["trainer"]["unsupervised"].get("use_cutmix_trigger_prob", 1.0)
            if np.random.uniform(0, 1) < trigger_prob and cfg["trainer"]["unsupervised"].get("use_cutmix", False):
                if cfg["trainer"]["unsupervised"].get("use_cutmix_adaptive", False):
                    image_u_aug, label_u_aug, logits_u_aug = cut_mix_label_adaptive(
                            image_u_aug,
                            label_u_aug,
                            logits_u_aug, 
                            image_l,
                            label_l, 
                            confidence
                        )

            # 3. forward concate labeled + unlabeld into student networks
            num_labeled = len(image_l)
            if flag_extra_weak:
                pred_all, aux_all = model(torch.cat((image_l, image_u_weak, image_u_aug), dim=0))
                del image_l, image_u_weak, image_u_aug
                pred_l= pred_all[:num_labeled]
                _, pred_u_strong = pred_all[num_labeled:].chunk(2)
                del pred_all
            else:
                pred_all, aux_all = model(torch.cat((image_l, image_u_aug), dim=0))
                del image_l, image_u_weak, image_u_aug
                pred_l= pred_all[:num_labeled]
                pred_u_strong = pred_all[num_labeled:]
                del pred_all

            # 4. supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = aux_all[:num_labeled]
                sup_loss = sup_loss_fn([pred_l, aux], label_l)
                del aux_all, aux
            else:
                sup_loss = sup_loss_fn(pred_l, label_l)

            # 5. unsupervised loss
            unsup_loss, pseduo_high_ratio = compute_unsupervised_loss_by_threshold(
                        pred_u_strong, label_u_aug.detach(),
                        logits_u_aug.detach(), thresh=p_threshold)
            unsup_loss *= cfg["trainer"]["unsupervised"].get("loss_weight", 1.0)
            del pred_l, pred_u_strong, label_u_aug, logits_u_aug

        loss = sup_loss + unsup_loss

        # update student model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        with torch.no_grad():
            if epoch > cfg["trainer"].get("sup_only_epoch", 0):
                ema_decay = min(
                    1
                    - 1
                    / (
                        i_iter
                        - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 0)
                        + 1
                    ),
                    ema_decay_origin,
                )
            else:
                ema_decay = 0.0
            # update weight
            for param_train, param_eval in zip(model.parameters(), model_teacher.parameters()):
                param_eval.data = param_eval.data * ema_decay + param_train.data * (1 - ema_decay)
            # update bn
            for buffer_train, buffer_eval in zip(model.buffers(), model_teacher.buffers()):
                buffer_eval.data = buffer_eval.data * ema_decay + buffer_train.data * (1 - ema_decay)
                # buffer_eval.data = buffer_train.data

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item() / world_size)

        reduced_uns_loss = unsup_loss.clone().detach()
        dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item() / world_size)

        reduced_pseudo_high_ratio = pseduo_high_ratio.clone().detach()
        dist.all_reduce(reduced_pseudo_high_ratio)
        meter_high_pseudo_ratio.update(reduced_pseudo_high_ratio.item() / world_size)

        # 12. print log information
        batch_end = time.time()
        batch_times.update(batch_end - batch_start)
        # if i_iter % 10 == 0 and rank == 0:
        if step in print_freq_lst and rank == 0:
            logger.info(
                "Epoch/Iter [{}:{:3}/{:3}].  "
                "Sup:{sup_loss.val:.3f}({sup_loss.avg:.3f})  "
                "Uns:{uns_loss.val:.3f}({uns_loss.avg:.3f})  "
                "Pseudo:{high_ratio.val:.3f}({high_ratio.avg:.3f})  "
                "Time:{batch_time.avg:.2f}  "
                "LR:{lr.val:.5f}".format(
                    cfg["trainer"]["epochs"], epoch, step,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    high_ratio=meter_high_pseudo_ratio,
                    batch_time=batch_times,
                    lr=learning_rates,
                )
            )
            if tb_logger is not None:
                tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
                tb_logger.add_scalar("Sup Loss", sup_losses.avg, i_iter)
                tb_logger.add_scalar("Uns Loss", uns_losses.avg, i_iter)
                tb_logger.add_scalar("High ratio", meter_high_pseudo_ratio.avg, i_iter)
    
    return sup_losses.avg, uns_losses.avg


def validate(
    model,
    data_loader,
    epoch,
    logger,
    cfg
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            output, _ = model(images)

        # get the output produced by model_teacher
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" [Test] -  class [{}] IoU {:.2f}".format(i, iou * 100))

    return mIoU


def validate_citys(
    model,
    data_loader,
    epoch,
    logger,
    cfg
):
    model.eval()
    data_loader.sampler.set_epoch(epoch)
    rank, world_size = dist.get_rank(), dist.get_world_size()

    num_classes = cfg["net"]["num_classes"]
    ignore_label = cfg["dataset"]["ignore_label"]
    if cfg["dataset"]["val"].get("crop", False):
        crop_size, _ = cfg["dataset"]["val"]["crop"].get("size", [800, 800])
    else:
        crop_size = 800

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        _, images, labels = batch
        images = images.cuda()
        labels = labels.long()
        batch_size, h, w = labels.shape

        with torch.no_grad():
            final = torch.zeros(batch_size, num_classes, h, w).cuda()
            row = 0
            while row < h:
                col = 0
                while col < w:
                    pred, _ = model(images[:, :, row: min(h, row + crop_size), col: min(w, col + crop_size)])
                    final[:, :, row: min(h, row + crop_size), col: min(w, col + crop_size)] += pred.softmax(dim=1)
                    col += int(crop_size * 2 / 3)
                row += int(crop_size * 2 / 3)
            # get the output
            output = final.argmax(dim=1).cpu().numpy()
            target_origin = labels.numpy()
            # print("="*50, output.shape, output.dtype, target_origin.shape, target_origin.dtype)

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )
        # # return ndarray, b*clas
        # print("="*20, type(intersection), type(union), type(target), intersection, union, target)

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" [Test] -  class [{}] IoU {:.2f}".format(i, iou * 100))
    return mIoU


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", default=None, type=int)
    args = parser.parse_args()
    main(args)
