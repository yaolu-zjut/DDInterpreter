import time
import torch
import tqdm
import pdb

from utils_clom.logging import AverageMeter
from utils_clom.utils import accuracy, DiffAugment, augment

__all__ = ["train", "validate"]


def validate(val_loader, model, criterion, args):
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            # compute output
            output = model(images)
            if target.dim() == 2:
                target = target.squeeze().long()

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

    return top1.avg, top5.avg


def train(train_loader, model, criterion, optimizer, epoch, args, aug=False):
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        if aug:
            if args.dsa:
                images = DiffAugment(images, args.dsa_strategy, param=args.dsa_param)
            else:
                images = augment(images, args.dc_aug_param, device=args.device)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        if target.dim() == 2:
            target = target.squeeze().long()

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return top1.avg, top5.avg, losses.avg