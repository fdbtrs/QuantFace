import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_

from backbones.mobilefacenet import MobileFaceNet
from config.config_Quantization import config as cfg
from utils.dataset import MXFaceDataset, DataLoaderX
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100, iresnet50, freeze_model, unfreeze_model, iresnet18

torch.backends.cudnn.benchmark = True

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)

    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
        logging.info("load backbone!" + cfg.network)

    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet18":
        backbone = iresnet18(dropout=0.4, num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network =="mobilefacenet":
        backbone=MobileFaceNet().to(local_rank)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()

    if args.resume:
        try:
            backbone_pth = os.path.join(cfg.output32, str(cfg.global_step) + "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load backbone resume init, failed!")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)
    if cfg.network =="mobilefacenet":
        from backbones.mobilefacenet import quantize_model
        backbone_quant = quantize_model(backbone, cfg.wq, cfg.aq).to(local_rank)
    else:
        from backbones.iresnet import quantize_model
        backbone_quant=quantize_model(backbone,cfg.wq,cfg.aq).to(local_rank)
    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.eval()

    backbone_quant = DistributedDataParallel(
        module=backbone_quant, broadcast_buffers=True, device_ids=[local_rank])
    backbone_quant.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone_quant.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay,nesterov=True,)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)

    criterion =torch.nn.MSELoss()

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.rec)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)
    backbone_quant=unfreeze_model(backbone_quant)
    loss = AverageMeter()
    global_step = 0
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        backbone_quant=freeze_model(backbone_quant)
        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            if (global_step < 300):
              backbone_quant = unfreeze_model(backbone_quant)
            img = img.cuda(local_rank, non_blocking=True)

            features = F.normalize(backbone_quant(img))
            with torch.no_grad():
                features_1 = F.normalize(backbone(img))
            loss_v=criterion(features,features_1)
            loss_v.backward()

            clip_grad_norm_(backbone_quant.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_backbone.zero_grad()

            loss.update(loss_v.item(), 1)
            if (global_step %5000==0):
                logging.info(backbone_quant)
            callback_logging(global_step, loss, epoch)
            callback_verification(global_step, backbone_quant)
            backbone_quant = freeze_model(backbone_quant)



        scheduler_backbone.step()
        callback_checkpoint(global_step, backbone_quant, None,quantiza=True)
    callback_verification(5686, backbone_quant)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=1, help="resume training")
    args_ = parser.parse_args()
    main(args_)
