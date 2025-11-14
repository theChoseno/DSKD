import os
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm
import pdb


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def validate(val_loader, distiller, epoch=None):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if epoch is None:
                output = distiller(image=image)
                loss = criterion(output, target)
            else:
                output, losses_dict = distiller.module.forward_train(image=image, target=target, epoch=epoch)
                loss = losses_dict['loss_kd']
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f} | loss_kd:{losses.avg:.2f}".format(
                top1=top1, top5=top5, losses=losses
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, top5.avg, losses.avg


def validate_cam(val_loader, distiller, epoch=None, file=None):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        if file:
            import cv2
#             pdb.set_trace()
            im = cv2.imread(file)
            mean=torch.tensor([0.485, 0.456, 0.406])
            std=torch.tensor([0.229, 0.224, 0.225])
            im = torch.tensor(im).float() / 255
            im = ((im - mean) / std).cuda()
            im = im.permute(2, 0, 1).unsqueeze(0)
            output = distiller.module.forward_cam(image=im, target=None, idx=0)
            exit()
        for idx, (image, target) in enumerate(val_loader):
#             pdb.set_trace()
            image = image.float()
            image = image.cuda(non_blocking=True)
#             for target in [label, torch.randint(1000, label.shape), torch.randint(1000, label.shape), torch.randint(1000, label.shape)]:
            target = target.cuda(non_blocking=True)
            if epoch is None:
                output = distiller.module.forward_cam(image=image, target=target, idx=idx)
                loss = criterion(output, target)
            else:
                output, losses_dict = distiller.module.forward_cam(image=image, target=target, epoch=epoch)
                loss = losses_dict['loss_kd']
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f} | loss_kd:{losses.avg:.2f}".format(
                top1=top1, top5=top5, losses=losses
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
#             if idx < 100:
#                 continue
    pbar.close()
    return top1.avg, top5.avg, losses.avg


def corr(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        res = torch.zeros(100, 100).cuda()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
#             output = output.softmax(dim=1)
            for id, tgt in enumerate(target):
#                 try:
                res[tgt] += output[id]
#                 except:
#                     pdb.set_trace()
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f} | loss_kd:{losses.avg:.2f}".format(
                top1=top1, top5=top5, losses=losses
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
            
    pbar.close()
#     res = res[:, :10]
#     res = res / res.max()
#     res = torch.nn.functional.normalize(res[:, :10], dim=1)
#     res = res @ res.T
#     pdb.set_trace()
    return top1.avg, top5.avg, losses.avg, res


def validate(val_loader, distiller, epoch=None, if_svd=False):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        stat = {}
        for idx, (image, target) in enumerate(val_loader):
            # if target[0] != 2:
            #     continue
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            if epoch is None:
                output = distiller(image=image)
                # import pdb
                # pdb.set_trace()
                if isinstance(output, list):
                    # if output[0].max() > 0:
                    #     output = output[0]
                    # elif output[1].max() > 0:
                    #     output = output[1]
                    # else:
                    #     output = output[2]
                    # # output = torch.stack(output, 2).max(2)[0]
                    output = output[2]
                loss = criterion(output, target)
            else:
                if if_svd:
                    output, losses_dict, _ = distiller.module.forward_train(image=image, target=target, epoch=epoch)
                else:
                    output, losses_dict = distiller.module.forward_train(image=image, target=target, epoch=epoch)
                loss = losses_dict['loss_ce']
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # kk = int(target[0])
            # if kk in stat.keys():
            #     stat[kk].append(int(acc1[0]))
            # else:
            #     stat[kk] = [int(acc1[0])]

            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f} | loss_ce:{losses.avg:.2f}".format(
                top1=top1, top5=top5, losses=losses
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    # pdb.set_trace()
    return top1.avg, top5.avg, losses.avg

def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
