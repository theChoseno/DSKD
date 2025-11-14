import imp
import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import getpass
# from tensorboardX import SummaryWriter
from .utils import (
    AverageMeter,
    accuracy,
    validate,
    adjust_learning_rate,
    save_checkpoint,
    load_checkpoint,
    log_msg,
)
import pdb
from D_SVD import DynamicRankDistiller


scaler = torch.cuda.amp.GradScaler()


class BaseTrainer(object):
    def __init__(self, experiment_name, distiller, train_loader, val_loader, cfg, if_svd=False, teacher_svd_tensor=None, rank_indeces=None, J_scores=None, global_step=None, total_step=None, _time=False):
        self.cfg = cfg
        self.distiller = distiller
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = self.init_optimizer(cfg)
        self.best_acc = -1
        self._time = _time
        self.teacher_svd_tensor = teacher_svd_tensor
        self.rank_indeces = rank_indeces
        self.J_scores = J_scores
        self.global_step = global_step
        self.total_step = total_step
        self.if_svd = if_svd

        username = getpass.getuser()
        # init loggers
        self.log_path = os.path.join(cfg.LOG.PREFIX, experiment_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
#         self.tf_writer = SummaryWriter(os.path.join(self.log_path, "train.events"))

    def init_optimizer(self, cfg):
        if cfg.SOLVER.TYPE == "SGD":
            optimizer = optim.SGD(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                momentum=cfg.SOLVER.MOMENTUM,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif cfg.SOLVER.TYPE == "Adam":
            optimizer = optim.Adam(
                self.distiller.module.get_learnable_parameters(),
                lr=cfg.SOLVER.LR,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(cfg.SOLVER.TYPE)
        return optimizer

    def log(self, lr, epoch, log_dict):
        # tensorboard log
#         for k, v in log_dict.items():
#             self.tf_writer.add_scalar(k, v, epoch)
#         self.tf_writer.flush()
        # wandb log
        if self.cfg.LOG.WANDB:
            import wandb

            wandb.log({"current lr": lr})
            wandb.log(log_dict)
        if log_dict["test_acc"] > self.best_acc:
            self.best_acc = log_dict["test_acc"]
            if self.cfg.LOG.WANDB:
                wandb.run.summary["best_acc"] = self.best_acc
        # worklog.txt
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            lines = [
                "-" * 25 + os.linesep,
                "epoch: {}".format(epoch) + os.linesep,
                "lr: {:.2f}".format(float(lr)) + os.linesep,
            ]
            for k, v in log_dict.items():
                lines.append("{}: {:.2f}".format(k, v) + os.linesep)
            lines.append("-" * 25 + os.linesep)
            writer.writelines(lines)

    def train(self, resume=False):
        epoch = 1
        if resume:
            state = load_checkpoint(os.path.join(self.log_path, "latest"))
            try:
                epoch = state["epoch"] + 1
                self.distiller.load_state_dict(state["model"])
                self.optimizer.load_state_dict(state["optimizer"])
                self.best_acc = state["best_acc"]
            except:
#                 pdb.set_trace()
                epoch = 60
                self.distiller.module.student = torch.nn.DataParallel(self.distiller.module.student.cuda())
                self.distiller.module.student.load_state_dict(state["model"])
                self.distiller.module.student = self.distiller.module.student.module
                self.best_acc = 0
        while epoch < self.cfg.SOLVER.EPOCHS + 1:
            ret = self.train_epoch(epoch)
            if ret == False:
                return

            epoch += 1
        print(log_msg("Best accuracy:{}".format(self.best_acc), "EVAL"))
        with open(os.path.join(self.log_path, "worklog.txt"), "a") as writer:
            writer.write("best_acc\t" + "{:.2f}".format(float(self.best_acc)))

    def train_epoch(self, epoch):
        lr = adjust_learning_rate(epoch, self.cfg, self.optimizer)
        train_meters = {
            "training_time": AverageMeter(),
            "data_time": AverageMeter(),
            "losses": AverageMeter(),
            "top1": AverageMeter(),
            "top5": AverageMeter(),
        }
        num_iter = len(self.train_loader)
        pbar = tqdm(range(num_iter))

        # train loops
        self.distiller.train()
        if self._time:
            import time
            start = time.time()
        for idx, data in enumerate(self.train_loader):
            msg = self.train_iter(data, epoch, train_meters)
            pbar.set_description(log_msg(msg, "TRAIN"))
            pbar.update()
        pbar.close()
        if self._time:
            elapse = time.time() - start
            print(elapse)
            return False

        # validate
        test_acc, test_acc_top5, test_loss = validate(self.val_loader, self.distiller, epoch, self.if_svd)

        # log
        log_dict = OrderedDict(
            {
                "train_acc": train_meters["top1"].avg,
                "train_loss": train_meters["losses"].avg,
                "test_acc": test_acc,
                "test_acc_top5": test_acc_top5,
                "test_loss": test_loss,
            }
        )
        self.log(lr, epoch, log_dict)
        # saving checkpoint
        state = {
            "epoch": epoch,
            "model": self.distiller.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        student_state = {"model": self.distiller.module.student.state_dict()}
        save_checkpoint(state, os.path.join(self.log_path, "latest"))
        save_checkpoint(
            student_state, os.path.join(self.log_path, "student_latest")
        )
        if epoch % self.cfg.LOG.SAVE_CHECKPOINT_FREQ == 0:
            save_checkpoint(
                state, os.path.join(self.log_path, "epoch_{}".format(epoch))
            )
            save_checkpoint(
                student_state,
                os.path.join(self.log_path, "student_{}".format(epoch)),
            )
        # update the best
        if test_acc >= self.best_acc:
            save_checkpoint(state, os.path.join(self.log_path, "best"))
            save_checkpoint(
                student_state, os.path.join(self.log_path, "student_best")
            )
        return True

    def calculate_svd_loss(self, student_svd_tensor):
        count = 0
        model_names = ["layer1", "layer2", "layer3"]
        convs = ["conv1", "conv2"]
        total_svd_loss = 0
        student_svd_tensor_withograd = {}
        student_svd_allname = []
        if (self.cfg.EXPERIMENT.TAG == 'cam,res56,res20') or (self.cfg.EXPERIMENT.TAG == 'dkd,res56,res20'):
            for model_name in model_names:
                for i in range(3):
                    ind = i
                    ind_teacher = 4 * ind
                    if ind == 0:
                        ind_teacher = 0
                    for conv in convs:
                        count += 1
                        teacher_conv_name = model_name + '.[' + str(ind_teacher) + '].' + conv
                        student_conv_name = model_name + '.[' + str(ind) + '].' + conv

                        U_s = student_svd_tensor[student_conv_name]['U']
                        S_s = student_svd_tensor[student_conv_name]['S']
                        V_s = student_svd_tensor[student_conv_name]['V']

                        student_svd_tensor_withograd[student_conv_name] = {
                            'U': U_s.detach(),
                            'S': S_s.detach(),
                            'V': V_s.detach()
                        }
                        student_svd_allname.append(student_conv_name)

                        U_t = self.teacher_svd_tensor[teacher_conv_name]['U'].to(U_s.device)
                        S_t = self.teacher_svd_tensor[teacher_conv_name]['S'].to(U_s.device)
                        V_t = self.teacher_svd_tensor[teacher_conv_name]['V'].to(U_s.device)
                        k = S_s.shape[0]
                        if len(self.rank_indeces) == 0:
                            r_index = torch.arange(0, k).to(U_t.device)
                            score_weight = torch.ones(k).to(U_t.device)
                            sum_score_weight = score_weight.sum()
                            score_weight = score_weight / (sum_score_weight + 1e-6)
                        else:
                            r_index = self.rank_indeces[student_conv_name]
                            score_weight = self.J_scores[student_conv_name]
                            max_score = score_weight.max()
                            min_score = score_weight.min()
                            score_weight = (score_weight - min_score) / (max_score - min_score + 1e-6)
                            sum_score_weight = score_weight.sum()
                            score_weight = score_weight / (sum_score_weight + 1e-6)

                        # all_weight = torch.zeros(S_t.shape[0]).to(U_s.device)
                        # all_weight[r_index] = score_weight
                        all_weight = torch.abs(score_weight)
                        all_weight = torch.sqrt(all_weight)

                        student_matrix = all_weight * U_s @ torch.diag(S_s) @ V_s.T
                        teacher_matrix = all_weight * U_t @ torch.diag(S_t) @ V_t.T
                        svd_loss = torch.norm(student_matrix - teacher_matrix, p='fro') ** 2

                        total_svd_loss += svd_loss
            total_svd_loss = total_svd_loss / count

        elif (self.cfg.EXPERIMENT.TAG == 'cam,res110,res32') or (self.cfg.EXPERIMENT.TAG == 'dkd,res110,res32'):
            for model_name in model_names:
                for i in range(5):
                    ind = i
                    ind_teacher = 4*ind + 1
                    if ind == 0:
                        ind_teacher = 0
                    for conv in convs:
                        count += 1
                        teacher_conv_name = model_name + '.[' + str(ind_teacher) + '].' + conv
                        student_conv_name = model_name + '.[' + str(ind) + '].' + conv

                        U_s = student_svd_tensor[student_conv_name]['U']
                        S_s = student_svd_tensor[student_conv_name]['S']
                        V_s = student_svd_tensor[student_conv_name]['V']

                        student_svd_tensor_withograd[student_conv_name] = {
                            'U': U_s.detach(),
                            'S': S_s.detach(),
                            'V': V_s.detach()
                        }
                        student_svd_allname.append(student_conv_name)

                        U_t = self.teacher_svd_tensor[teacher_conv_name]['U'].to(U_s.device)
                        S_t = self.teacher_svd_tensor[teacher_conv_name]['S'].to(U_s.device)
                        V_t = self.teacher_svd_tensor[teacher_conv_name]['V'].to(U_s.device)
                        k = S_s.shape[0]
                        if len(self.rank_indeces) == 0:
                            r_index = torch.arange(0, k).to(U_t.device)
                            score_weight = torch.ones(k).to(U_t.device)
                            sum_score_weight = score_weight.sum()
                            score_weight = score_weight / (sum_score_weight + 1e-6)
                        else:
                            r_index = self.rank_indeces[student_conv_name]
                            score_weight = self.J_scores[student_conv_name]
                            max_score = score_weight.max()
                            min_score = score_weight.min()
                            score_weight = (score_weight - min_score) / (max_score - min_score + 1e-6)
                            sum_score_weight = score_weight.sum()
                            score_weight = score_weight / (sum_score_weight + 1e-6)

                        # all_weight = torch.zeros(S_t.shape[0]).to(U_s.device)
                        # all_weight[r_index] = score_weight
                        all_weight = torch.abs(score_weight)
                        all_weight = torch.sqrt(all_weight)

                        student_matrix = all_weight * U_s @ torch.diag(S_s) @ V_s.T
                        teacher_matrix = all_weight * U_t @ torch.diag(S_t) @ V_t.T
                        svd_loss = torch.norm(student_matrix - teacher_matrix, p='fro') ** 2

                        total_svd_loss += svd_loss
            total_svd_loss = total_svd_loss / count

        elif (self.cfg.EXPERIMENT.TAG == 'cam,res32x4,res8x4') or (self.cfg.EXPERIMENT.TAG == 'dkd,res32x4,res8x4'):
            for model_name in model_names:
                ind = 0
                ind_teacher = 0
                for conv in convs:
                    count += 1
                    teacher_conv_name = model_name + '.[' + str(ind_teacher) + '].' + conv
                    student_conv_name = model_name + '.[' + str(ind) + '].' + conv

                    U_s = student_svd_tensor[student_conv_name]['U']
                    S_s = student_svd_tensor[student_conv_name]['S']
                    V_s = student_svd_tensor[student_conv_name]['V']

                    student_svd_tensor_withograd[student_conv_name] = {
                        'U': U_s.detach(),
                        'S': S_s.detach(),
                        'V': V_s.detach()
                    }
                    student_svd_allname.append(student_conv_name)

                    U_t = self.teacher_svd_tensor[teacher_conv_name]['U'].to(U_s.device)
                    S_t = self.teacher_svd_tensor[teacher_conv_name]['S'].to(U_s.device)
                    V_t = self.teacher_svd_tensor[teacher_conv_name]['V'].to(U_s.device)
                    k = S_s.shape[0]
                    if len(self.rank_indeces) == 0:
                        r_index = torch.arange(0, k).to(U_t.device)
                        score_weight = torch.ones(k).to(U_t.device)
                        sum_score_weight = score_weight.sum()
                        score_weight = score_weight / (sum_score_weight + 1e-6)
                    else:
                        r_index = self.rank_indeces[student_conv_name]
                        score_weight = self.J_scores[student_conv_name]
                        max_score = score_weight.max()
                        min_score = score_weight.min()
                        score_weight = (score_weight - min_score) / (max_score - min_score + 1e-6)
                        sum_score_weight = score_weight.sum()
                        score_weight = score_weight / (sum_score_weight + 1e-6)

                    # all_weight = torch.zeros(S_t.shape[0]).to(U_s.device)
                    # all_weight[r_index] = score_weight
                    all_weight = torch.abs(score_weight)
                    all_weight = torch.sqrt(all_weight)

                    student_matrix = all_weight * U_s @ torch.diag(S_s) @ V_s.T
                    teacher_matrix = all_weight * U_t @ torch.diag(S_t) @ V_t.T
                    svd_loss = torch.norm(student_matrix - teacher_matrix, p='fro') ** 2

                    total_svd_loss += svd_loss
            total_svd_loss = total_svd_loss / count
        return total_svd_loss, student_svd_tensor_withograd, student_svd_allname

    def train_iter(self, data, epoch, train_meters):
        train_start_time = time.time()
        image, target, index = data
        if self.if_svd:
            self.global_step += 1
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

#         # forward
#         with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
#             preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)
#             # backward
#             loss = sum([l.mean() for l in losses_dict.values()])
            
#         scaler.scale(loss).backward()
#         scaler.step(self.optimizer)
#         scaler.update()
#         self.optimizer.zero_grad()

        if self.if_svd:
            preds, losses_dict, student_svd_tensor = self.distiller(image=image, target=target, epoch=epoch)
        # else:
        #     preds, losses_dict = self.distiller(image=image, target=target, epoch=epoch)
        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        if self.if_svd:
            svd_loss, student_svd_tensor, choose_name = self.calculate_svd_loss(student_svd_tensor)
            loss = loss + 0.2 * svd_loss
        loss.backward()
        self.optimizer.step()
        if self.if_svd and self.global_step % 100 == 0:
            alpha_t = self.global_step / self.total_step
            d_svd = DynamicRankDistiller(self.distiller.module.student, choose_name, alpha_t)
            self.J_scores = d_svd.compute_J_scores(student_svd_tensor)
        self.optimizer.zero_grad()

        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg


class CRDTrainer(BaseTrainer):
    def train_iter(self, data, epoch, train_meters):
        self.optimizer.zero_grad()
        train_start_time = time.time()
        image, target, index, contrastive_index = data
        train_meters["data_time"].update(time.time() - train_start_time)
        image = image.float()
        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        contrastive_index = contrastive_index.cuda(non_blocking=True)

        # forward
        preds, losses_dict = self.distiller(
            image=image, target=target, index=index, contrastive_index=contrastive_index
        )

        # backward
        loss = sum([l.mean() for l in losses_dict.values()])
        loss.backward()
        self.optimizer.step()
        train_meters["training_time"].update(time.time() - train_start_time)
        # collect info
        batch_size = image.size(0)
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        train_meters["losses"].update(loss.cpu().detach().numpy().mean(), batch_size)
        train_meters["top1"].update(acc1[0], batch_size)
        train_meters["top5"].update(acc5[0], batch_size)
        # print info
        msg = "Epoch:{}| Time(data):{:.3f}| Time(train):{:.3f}| Loss:{:.4f}| Top-1:{:.3f}| Top-5:{:.3f}".format(
            epoch,
            train_meters["data_time"].avg,
            train_meters["training_time"].avg,
            train_meters["losses"].avg,
            train_meters["top1"].avg,
            train_meters["top5"].avg,
        )
        return msg
