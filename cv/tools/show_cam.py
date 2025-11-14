import argparse
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import sys
sys.path.append('.')

from mdistiller.distillers import CAM
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate_cam
from mdistiller.engine.cfg import CFG as cfg
import pdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-t", "--teacher_model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="imagenet",
        choices=["cifar100", "imagenet"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    parser.add_argument("-f", "--file", type=str)
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    if args.dataset == "imagenet":
        val_loader = get_imagenet_val_loader(args.batch_size)
#         val_loader.shuffle=True
#         if args.ckpt == "pretrain":
        teacher_model = imagenet_model_dict[args.teacher_model](pretrained=True)
        vanilla = imagenet_model_dict[args.model](pretrained=True)
#         else:
        model = imagenet_model_dict[args.model](pretrained=False)
#             model.load_state_dict(load_checkpoint(args.ckpt)["model"])
        try:
            model.load_state_dict(load_checkpoint(args.ckpt)["model"])
        except:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(load_checkpoint(args.ckpt)["model"])
    elif args.dataset == "cifar100":
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model, pretrain_model_path = cifar_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        try:
            model.load_state_dict(load_checkpoint(ckpt)["model"])
        except:
            model.load_state_dict(load_checkpoint(ckpt))
    model = CAM(model, teacher_model, cfg, vanilla)
    model.t_net = args.teacher_model
    model.s_net = args.model
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    test_acc, test_acc_top5, test_loss = validate_cam(val_loader, model, file=args.file)
