import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
import pdb

def removenan(x):
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return x

def msym(x):
    return (x + x.T)/2

def create_svd_with_custom_grad():
    class SVDWithCustomGrad(torch.autograd.Function):
        ## SVD操作，反向传播计算梯度 ##
        @staticmethod
        def forward(ctx, W):
            U, S, V = torch.svd(W) # m*r, r, n*r
            # U_k = U[:, rank_indeces]
            # S_k = S[rank_indeces]
            # V_k = V[:, rank_indeces]
            # ctx.save_for_backward(U_k, S_k, V_k)
            ctx.save_for_backward(U, S, V, W)
            return U, S, V # m*r, r, n*r

        @staticmethod
        def backward(ctx, dU, dS, dV):# m*k, k, n*k
            U, S, V, W = ctx.saved_tensors # m*k, k, n*k

            s_f = torch.square(S)
            f = removenan(1.0 / (s_f.reshape(-1, 1) - s_f.reshape(1, -1)))
            F = torch.where(torch.eye(f.size(-1), device=f.device) == 1, torch.zeros_like(f), f)
            S = torch.diag(S)
            s = torch.diag(1 / torch.diag(S))
            # s = 1 / S
            Im = torch.eye(U.shape[0], device=U.device)
            In = torch.eye(V.shape[0], device=V.device)
            Ik = torch.eye(S.shape[0], device=S.device)

            #### v2 ####
            E = dV @ s
            if dU.shape[0] <= dV.shape[0]:
                grad1 = U @ E.T
                grad2 = U @ (E.T @ V) @ V.T
                grad3 = 2 * U @ msym(F * (S @ V.T @ E)) @ S @ V.T
                grad = grad1 - grad2 - grad3
            else:
                grad = 2 * U @ S @ msym(F.T * (V.T @ dV)) @ V.T

            return removenan(grad)

    return SVDWithCustomGrad

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
#     pdb.set_trace()
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        ##########################################################
        student_svd_tensor = {}
        model_names = ["layer1", "layer2", "layer3"]
        convs = ["conv1", "conv2"]
        for model_name in model_names:
            model_layer = getattr(self.student, model_name)
            for i in range(len(model_layer)):
                for conv in convs:
                    model_conv = getattr(model_layer[i], conv)
                    with torch.enable_grad():
                        conv_weight = model_conv.weight
                        conv_weight = conv_weight.view(conv_weight.shape[0],
                                                       conv_weight.shape[1] * conv_weight.shape[2] *
                                                       conv_weight.shape[3])
                        # U_s, S_s, V_s = torch.svd(conv_weight)

                        svd_Function = create_svd_with_custom_grad()
                        U_s, S_s, V_s = svd_Function.apply(conv_weight)
                        U_s = removenan(U_s)
                        S_s = removenan(S_s)
                        V_s = removenan(V_s)
                    conv_name = model_name + '.[' + str(i) + '].' + conv
                    student_svd_tensor[conv_name] = {
                        'U': U_s,
                        'S': S_s,
                        'V': V_s
                    }
        #######################################################

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict, student_svd_tensor
