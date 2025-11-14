import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict


torch.manual_seed(3407)
def main(cfg, resume, opts, _time):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            try:
                model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            except:
                model_teacher.load_state_dict(load_checkpoint(pretrain_model_path))
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg
            )
    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )
    if args.ifsvd:
        print("########################")
        target_layers = []
        model_names = ["layer1", "layer2", "layer3"]
        convs = ["conv1","conv2"]
        teacher_svd_tensor = {}
        for model_name in model_names:
            model_layer = getattr(model_teacher, model_name)
            for i in range(len(model_layer)):
                for conv in convs:
                    model_conv = getattr(model_layer[i], conv)
                    conv_weight = model_conv.weight.detach()
                    conv_weight = conv_weight.reshape(conv_weight.shape[0] , conv_weight.shape[1] * conv_weight.shape[2] * conv_weight.shape[3])
                    U, S, V = torch.svd(conv_weight)
                    # print(model_conv)
                    conv_name = model_name + '.[' + str(i) + '].' + conv
                    teacher_svd_tensor[conv_name] = {
                        'U': U,
                        'S': S,
                        'V': V
                    }
        print("########################")

        rank_indeces = {}
        J_scores = {}

    global_step = 0
    total_step = len(train_loader) * cfg.SOLVER.EPOCHS

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg, args.ifsvd, teacher_svd_tensor, rank_indeces, J_scores,global_step, total_step, _time=_time
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="configs/cifar100/dkd/res32x4_res8x4.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--ifsvd", default=True)
    # parser.add_argument("--cd", default=0)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts, args.time)
