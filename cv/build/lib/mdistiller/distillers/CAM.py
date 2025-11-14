import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller, get_fc
import cv2
import numpy as np
import imageio
import os
import pdb


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def save_cam(cam_teacher, cam_student, cam_vanilla, logits_teacher, logits_student, logits_vanilla, target, image, epoch, folder):
    c, b, h, w = cam_teacher.shape
    try:
        os.mkdir(os.path.join('cams', 'img'))
        os.mkdir(os.path.join('cams', 'viz'))
        os.mkdir(os.path.join('cams', 'cam'))
    except:
        pass
    output_t, idx = logits_teacher.softmax(dim=1).topk(3,dim=1)
    for i_b in range(b):
        if target is None or (output_t[i_b][-1] > 0.2 and idx[i_b][0] == target[i_b]) : #and (idx[i_b,0]-idx[i_b,1]).abs() > 10 and idx[i_b,1]).abs() > 10 and (idx[i_b,2]-idx[i_b,0]).abs() > 10:
            print('+1')
            original_image = image[i_b].permute(1, 2, 0).data.cpu().numpy()
            original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
            original_save = np.uint8(original_image * 255)
            if target is None:
                original_save = cv2.cvtColor(original_save, cv2.COLOR_RGB2BGR)
            imageio.imsave(os.path.join('cams', 'img', 'img_' + str(epoch) + '_' + str(i_b) + '_' + folder + '.jpg'), original_save)
            for i_c in idx[i_b]:
                cam_list = []
                vis_list = []
                for name, cams in ([['teacher', cam_teacher], ['student', cam_student], ['vanilla', cam_vanilla]]):
                    cams = torch.nn.functional.interpolate(cams, scale_factor=224/h, mode='bilinear', align_corners=False)
                    Res = cams[i_c, i_b]
                    Res = (Res - Res.min()) / (Res.max() - Res.min())
                    vis = show_cam_on_image(original_image, Res.detach().cpu().numpy())
                    vis =  np.uint8(255 * vis)
                    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
                    cam =  np.uint8(255 * Res.detach().cpu().numpy())
                    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
                    cam = cv2.cvtColor(np.array(cam), cv2.COLOR_RGB2BGR)
                    imageio.imsave(os.path.join('cams', 'viz', 'viz_' + str(epoch) + '_' + str(i_b) + '_' + str(i_c.item()) + name + folder + '.jpg'), vis)
                    imageio.imsave(os.path.join('cams', 'cam', 'cam_' + str(epoch) + '_' + str(i_b) + '_' + str(i_c.item()) + name + folder + '.jpg'), cam)

def save_img(cams, target, image, epoch, folder):
    c, b, h, w = cams.shape
    cams = torch.nn.functional.interpolate(cams, scale_factor=224/h, mode='bilinear', align_corners=False)
    i = 0
    original_image = image[i].permute(1, 2, 0).data.cpu().numpy()
    original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
    vis =  np.uint8(original_image * 255)
    imageio.imsave(os.path.join('cams', 'viz', 'viz_' + str(epoch) + '_' + str(i) + '_' + folder + '.jpg'), vis)

    
        
    
class CAM(Distiller):

    def __init__(self, student, teacher, cfg, vanilla=None):
        super(CAM, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.CAM.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.CAM.LOSS.FEAT_WEIGHT
        self.fc_s = get_fc(student)
        self.fc_t = get_fc(teacher)
        assert self.fc_s.bias is not None
#         self.pool_4 = nn.AdaptiveAvgPool2d([2, 2])
        self.pool_feature = None #nn.AdaptiveAvgPool3d([32, 2, 2])
        self.pool_fc = None #nn.AdaptiveAvgPool1d([32])
        self.dropout = nn.Dropout(0.1)
        self.warmup = cfg.CAM.WARMUP
        self.total_eps = 1 #cfg.SOLVER.EPOCHS
        self.T = cfg.CAM.T
        self.K = cfg.CAM.K
        if vanilla:
            self.vanilla = vanilla
            self.fc_v = get_fc(vanilla)
        self.kd_loss_weight = 0.9
        self.temperature = 4
        self.k = 30
        self.depth_t = self.fc_t.in_features
        self.depth_s = self.fc_s.in_features
            
    def get_loss_cam(self, cam_teacher, cam_student):
        with torch.no_grad():
            if cam_student.shape[-1] > cam_teacher.shape[-1]:
                cam_teacher = torch.nn.functional.interpolate(cam_teacher, scale_factor=cam_student.shape[-1]/cam_teacher.shape[-1], mode='bilinear', align_corners=False).cuda()
            elif cam_student.shape[-1] < cam_teacher.shape[-1]:
                cam_teacher = torch.nn.functional.interpolate(cam_teacher, scale_factor=cam_student.shape[-1]/cam_teacher.shape[-1], mode='bilinear', align_corners=False).cuda()
        # elif cam_student.shape[-1] < cam_teacher.shape[-1]:
        #     cam_student = torch.nn.functional.interpolate(cam_student, scale_factor=cam_teacher.shape[-1]/cam_student.shape[-1], mode='bilinear', align_corners=False).cuda()
        loss_cam = (((cam_student/self.T).softmax(dim=0).mean() - (cam_teacher/self.T).softmax(dim=0))**2).sum().mean()
        return loss_cam
        # return 1

    def forward_train(self, image, target, **kwargs):
        bs = image.shape[0]
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)
        # sz = feature_teacher['feats'][-1].shape[-1]
        topk_teacher = logits_teacher.topk(self.k)[1].view(bs, self.k, 1)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        num_classes = self.fc_s.weight.shape[0]
        # if self.pool_feature is None:
        #     sz = max(2, feature_student["feats"][-1].shape[-1] // 4)
        #     self.pool_feature = nn.AdaptiveAvgPool2d([sz, sz])
        #     self.pool_feature_4 = nn.AdaptiveAvgPool2d([4, 4])
        #     self.pool_feature_1 = nn.AdaptiveAvgPool2d([1, 1])
        cam_teacher = self.get_cam(feature_teacher, self.fc_t, num_classes, topk_teacher, bs, self.depth_t)
        cam_student = self.get_cam(feature_student, self.fc_s, num_classes, topk_teacher, bs, self.depth_s)
        # cam_teacher = cam_teacher.gather(dim=1, index=topk_teacher)
        # cam_student = cam_student.gather(dim=1, index=topk_teacher)
        loss_kd = self.kd_loss_weight * self.get_loss_cam(
             cam_teacher, cam_student
             )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict


            

    def forward_cam(self, image, target, idx, **kwargs):
        with torch.no_grad():
            logits_student, feature_student = self.student(image)
            logits_teacher, feature_teacher = self.teacher(image)
            logits_vanilla, feature_vanilla = self.vanilla(image)
        

        # losses
        num_classes = self.fc_s.weight.shape[0]
        feature_student = feature_student["feats"][-1]
        feature_teacher = feature_teacher["feats"][-1]
        feature_vanilla = feature_vanilla["feats"][-1]

        linear_student = self.fc_s.weight.view(num_classes, 1, -1, 1, 1)
        linear_teacher = self.fc_t.weight.view(num_classes, 1, -1, 1, 1)
        linear_vanilla = self.fc_v.weight.view(num_classes, 1, -1, 1, 1)
        cam_student = ((feature_student) * linear_student).mean(2).clamp(0)
        cam_teacher = ((feature_teacher) * linear_teacher).mean(2).clamp(0)
        cam_vanilla = ((feature_vanilla) * linear_vanilla).mean(2).clamp(0)

        save_cam(cam_teacher, cam_student, cam_vanilla, logits_teacher, logits_student, logits_vanilla, target, image, idx, self.t_net)
        return logits_student
    
    def get_cam(self, feature, fc, num_classes, index, bs, depth):
        feature = feature["feats"][-1]
        linear = fc.weight.view(1, num_classes, -1).repeat(bs, 1, 1).gather(1, index.repeat(1, 1, depth))
        cam = (feature.unsqueeze(1) * linear.unsqueeze(-1).unsqueeze(-1)).mean(2).clamp(0)#.permute(1, 0, 2, 3)
        return cam
