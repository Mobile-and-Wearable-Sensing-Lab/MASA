import argparse
import math
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

import torch.distributed as dist

import utils.warmup_scheduler as optim_warmup
from feeder import feeder_pretraining
import misc
import moco.builder_dist
from torch.utils.tensorboard import SummaryWriter
from dataset import get_pretraining


parser = argparse.ArgumentParser(description='MASA Pretraining')

# workers (accept both --workers and --worker)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--worker', dest='workers', type=int,
                    help='alias for --workers')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--schedule', default=[100, 160], nargs='*', type=int,
                    help='lr schedule milestones')
parser.add_argument('--optim', type=str, default='AdamW',
                    help='optimizer type: SGD or AdamW')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay', dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint')

# distributed
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local-rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')

parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training')
parser.add_argument('--checkpoint-path', default='./checkpoints', type=str)

parser.add_argument('--skeleton-representation', type=str, required=True,
                    help='graph-based / image-based / seq-based')
parser.add_argument('--pre-dataset', type=str, required=True,
                    help='which dataset protocol to use (keep SLR for this repo)')

# contrastive configs
parser.add_argument('--contrast-dim', default=128, type=int)
parser.add_argument('--contrast-k', default=32768, type=int)
parser.add_argument('--contrast-m', default=0.999, type=float)
parser.add_argument('--contrast-t', default=0.07, type=float)
parser.add_argument('--teacher-t', default=0.05, type=float)
parser.add_argument('--student-t', default=0.1, type=float)
parser.add_argument('--inter-weight', default=0.5, type=float)
parser.add_argument('--topk', default=1024, type=int)

parser.add_argument('--mlp', action='store_true')
parser.add_argument('--cos', action='store_true')
parser.add_argument('--inter-dist', action='store_true')
parser.add_argument('--warmup', action='store_true')
parser.add_argument('--contrast_weight', type=float, default=0.05)
parser.add_argument('--mask_ratio', type=float, default=0.9)


def main():
    args = parser.parse_args()
    misc.init_distributed_mode(args)

    # ---- make these ALWAYS defined ----
    global_rank = misc.get_rank()
    args.gpu = getattr(args, "gpu", None)  # in DDP, misc.init_distributed_mode usually sets args.gpu
    device = torch.device(args.device if args.device else "cuda")

    # if not distributed, put everything on cuda:0
    if not getattr(args, "distributed", False):
        if device.type == "cuda":
            torch.cuda.set_device(0)

    # seed
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # protocol / options
    from options import options_pretraining as options
    if args.pre_dataset == 'SLR':
        opts = options.opts_SLR()
    else:
        raise ValueError(f"Unsupported pre-dataset: {args.pre_dataset}")

    opts.train_feeder_args['input_representation'] = args.skeleton_representation
    opts.train_feeder_args['mask_ratio'] = args.mask_ratio

    print("Not using distributed mode" if not args.distributed else "Using distributed mode")
    print("=> creating model")
    model = moco.builder_dist.MASA(
        args.skeleton_representation, opts.num_class,
        args.contrast_dim, args.contrast_k, args.contrast_m, args.contrast_t,
        args.teacher_t, args.student_t, args.topk, args.mlp,
        inter_weight=args.inter_weight,
        inter_dist=args.inter_dist
    )

    print("options", opts.train_feeder_args)
    print(model)

    model.to(device)

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # args.gpu should exist in distributed mode; fallback to local_rank
        if args.gpu is None:
            args.gpu = args.local_rank if args.local_rank >= 0 else 0
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        print('Distributed data parallel model used')

    # ---- FIX 1: criterion must NOT depend on args.gpu ----
    criterion = nn.CrossEntropyLoss().to(device)

    # dataset
    train_dataset = get_pretraining(opts)

    if args.distributed:
        num_tasks = misc.get_world_size()
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        train_sampler = None

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=feeder_pretraining.collate_fn
    )

    # optimizer
    if args.optim.upper() == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        print(">>> using SGD optimizer!")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), args.lr,
            weight_decay=args.weight_decay
        )
        print(">>> using AdamW optimizer!")

    # ---- FIX 2: always define a scheduler (warmup or normal) ----
    if args.warmup:
        scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, gamma=0.5, milestones=[100, 200, 300, 350]
        )
        scheduler = optim_warmup.GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=20, after_scheduler=scheduler_steplr
        )
        print(">>> using warmup + MultiStepLR")
    else:
        if args.cos:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            print(">>> using CosineAnnealingLR")
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=args.schedule, gamma=0.1
            )
            print(">>> using MultiStepLR milestones:", args.schedule)

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location='cpu')
            args.start_epoch = checkpoint.get('epoch', 0)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint (epoch {args.start_epoch})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    # writer
    if global_rank == 0:
        os.makedirs(args.checkpoint_path, exist_ok=True)
        writer = SummaryWriter(log_dir=args.checkpoint_path)
    else:
        writer = None

    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast

    max_lambda_CL = args.contrast_weight
    max_lambda_epoch = 100

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # step LR
        if args.warmup:
            scheduler.step(epoch)
        else:
            scheduler.step()

        lambda_CL = min(max_lambda_CL * (epoch / max_lambda_epoch), max_lambda_CL)

        loss_joint, top1_joint, loss_hand_2d, loss_body_2d = train(
            train_loader, model, criterion, optimizer, epoch, autocast, scaler, lambda_CL, args, device
        )

        if writer is not None:
            writer.add_scalar('loss_joint', loss_joint.avg, global_step=epoch)
            writer.add_scalar('loss_hand_2d', loss_hand_2d.avg, global_step=epoch)
            writer.add_scalar('loss_body_2d', loss_body_2d.avg, global_step=epoch)
            writer.add_scalar('top1_joint', top1_joint.avg, global_step=epoch)
            writer.add_scalar('lambda_CL', lambda_CL, global_step=epoch)

            if epoch % 20 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best=False,
                   filename=os.path.join(args.checkpoint_path, f'checkpoint_{epoch:04d}.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, autocast, scaler, lambda_CL, args, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_joint = AverageMeter('Loss Joint', ':6.3f')
    losses_hand_2d = AverageMeter('Loss_hand_2d', ':6.3f')
    losses_body_2d = AverageMeter('Loss_body_2d', ':6.3f')
    top1_joint = AverageMeter('Acc Joint@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses_joint, losses_hand_2d, losses_body_2d, top1_joint],
        prefix=f"Epoch: [{epoch}] Lr_rate [{optimizer.param_groups[0]['lr']}] lambda_CL [{lambda_CL}]"
    )

    model.train()
    end = time.time()

    for i, (input_v1, input_v2) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # move to device
        for k, v in input_v1.items():
            for k_1, v_1 in v.items():
                input_v1[k][k_1] = v_1.float().to(device, non_blocking=True)
        for k, v in input_v2.items():
            for k_1, v_1 in v.items():
                input_v2[k][k_1] = v_1.float().to(device, non_blocking=True)

        with autocast():
            output, target, rh_loss, lh_loss, body_loss = model(input_v1, input_v2, self_dist=args.inter_dist)
            batch_size = output.size(0)

            target = target.to(device, non_blocking=True)
            loss_joint = criterion(output, target)
            loss = (lambda_CL * loss_joint) + (rh_loss + lh_loss + body_loss)

            if np.isinf(loss.item()) or np.isnan(loss.item()):
                print("Nan/Inf loss detected, skipping batch")
                continue

            losses_joint.update(loss_joint.item(), batch_size)
            losses_hand_2d.update((rh_loss.item() + lh_loss.item()) / 2.0, batch_size)
            losses_body_2d.update(body_loss.item(), batch_size)

        acc1_joint, _ = accuracy(output, target, topk=(1, 5))
        top1_joint.update(acc1_joint[0], batch_size)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses_joint, top1_joint, losses_hand_2d, losses_body_2d


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
