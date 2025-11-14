import torch
import numpy as np
from torch.autograd import Function
import re
# class DynamicSVDWithGrad(Function):
#     @staticmethod
#     def forward(ctx, W, select_s):

def hessian_vector_product(grad_0, params, v):
    grads = grad_0
    grad_flat = torch.cat([g.view(-1) for g in grads if g is not None])
    v_flat = torch.cat([v_.view(-1) for v_ in v])
    grad_dot_v = torch.dot(grad_flat, v_flat)
    hvp = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    return hvp

class DynamicRankDistiller:
    def __init__(self, student_layer, choose_name, alpha_t):
        self.student_layer = student_layer
        self.choose_name = choose_name
        self.alpha_t = alpha_t

    def compute_J_scores(self, student_svd_tensor):
        J_scores = {}

        model_names = ["layer1", "layer2", "layer3"]
        convs = ["conv1", "conv2"]
        for model_name in model_names:
            model_layer = getattr(self.student_layer, model_name)
            for i in range(len(model_layer)):
                for conv in convs:
                    param = getattr(model_layer[i], conv).weight
                    grad_W = param.grad
                    param = param.view(param.shape[0],
                                         param.shape[1] * param.shape[2] * param.shape[3])
                    grad_W = grad_W.view(grad_W.shape[0],
                                         grad_W.shape[1] * grad_W.shape[2] * grad_W.shape[3])
                    name = model_name + '.[' + str(i) + '].' + conv
                    U = student_svd_tensor[name]['U']
                    S = torch.diag(student_svd_tensor[name]['S'])
                    V = student_svd_tensor[name]['V']
                    delat_w = param - U @ S @ V.T
                    U_d, S_d, V_d = torch.svd(delat_w)
                    with torch.no_grad():
                        #### 计算梯度敏感性项 ####
                        grad_sensitive = torch.mm(torch.mm(U_d.T, grad_W), V_d) * torch.eye(U.shape[1], device=U.device)

                        # 计算曲率敏感性项（HVP）
                        # K = torch.einsum('ik,jk->ijk', V_d, U_d).flatten(start_dim=0, end_dim=1)

                        s_square = S_d ** 2
                        # s_square = 1
                        # grad_norm = torch.linalg.norm(grad_W, ord=2) ** 2
                        F1 = U_d.T @ grad_W @ grad_W.T @ U_d
                        F2 = V_d.T @ grad_W.T @ grad_W @ V_d

                        # I2 = torch.diag(abs(K.T @ grad_W.view(-1).unsqueeze(-1) @ grad_W.view(-1).unsqueeze(0) @ K) * torch.eye(U.shape[1], device=U.device))
                        # I2 = abs(s_square * grad_norm) * 0.5
                        I2 = torch.diag(abs(s_square * F1 @ F2) * 0.5)
                        I2_sum = I2.sum()
                        I2 = I2 / I2_sum

                        # 计算综合指标
                        # I1 = abs(torch.diag(grad_sensitive) * S_d)
                        I1 = abs(torch.diag(grad_sensitive) * S_d)
                        I1_sum = I1.sum()
                        I1 = I1 / I1_sum

                        score_final = (1 - self.alpha_t) * I1 + self.alpha_t * torch.diag(I2)
                        J_scores[name] = torch.sort(score_final, descending=True)[0]
        return J_scores
