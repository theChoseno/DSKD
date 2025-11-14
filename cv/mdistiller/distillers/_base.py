import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

stat_output_1 = {49: 84.0, 33: 58.0, 72: 30.0, 51: 63.0, 71: 74.0, 92: 56.0, 15: 63.0, 14: 51.0, 23: 76.0, 0: 84.0, 75: 81.0, 81: 63.0, 69: 75.0, 40: 49.0, 43: 74.0, 97: 57.0, 70: 65.0, 53: 89.0, 29: 48.0, 21: 84.0, 16: 67.0, 39: 74.0, 8: 78.0, 20: 74.0, 61: 63.0, 41: 84.0, 93: 42.0, 56: 82.0, 73: 57.0, 58: 78.0, 11: 31.0, 25: 53.0, 37: 64.0, 63: 64.0, 24: 73.0, 22: 63.0, 17: 84.0, 4: 38.0, 6: 70.0, 9: 66.0, 57: 73.0, 2: 51.0, 32: 53.0, 52: 71.0, 42: 70.0, 77: 49.0, 27: 56.0, 65: 37.0, 7: 50.0, 35: 45.0, 82: 89.0, 66: 70.0, 90: 68.0, 67: 54.0, 91: 75.0, 10: 32.0, 78: 50.0, 54: 79.0, 89: 72.0, 18: 63.0, 13: 47.0, 50: 35.0, 26: 50.0, 83: 61.0, 47: 52.0, 95: 64.0, 76: 83.0, 59: 56.0, 85: 81.0, 19: 50.0, 46: 43.0, 1: 76.0, 74: 45.0, 60: 89.0, 64: 47.0, 45: 44.0, 36: 78.0, 87: 73.0, 30: 55.0, 99: 66.0, 80: 40.0, 28: 72.0, 98: 38.0, 12: 66.0, 94: 86.0, 68: 95.0, 44: 25.0, 31: 61.0, 79: 63.0, 34: 55.0, 55: 23.0, 62: 67.0, 96: 59.0, 84: 50.0, 38: 47.0, 86: 66.0, 5: 55.0, 48: 97.0, 3: 37.0, 88: 75.0}


def get_fc(model):
    layers = {k:v for k,v in model.named_modules()}.keys()
    if 'fc' in layers:
        return model.fc
    elif 'linear' in layers:
        return model.linear
    elif 'classifier' in layers:
        return model.classifier
    else:
        raise NotImplementedError
        
class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student
        self.fc_s = get_fc(student)

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]
    
    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        if isinstance(logits_student, list):
            logits = torch.stack(logits_student, 2).log_softmax(1)
            if self.training:
                granular_weight = (self.student.granular.weight.unsqueeze(0) * 10).softmax(-1)
            else:
                # pdb.set_trace()
                granular_weight = (self.student.granular.weight == self.student.granular.weight.max(1, keepdims=True)[0]).unsqueeze(0).long()
            logits = granular_weight * logits
            logits = logits.sum(-1)
            loss = -logits.gather(1, target.view(-1,1)).mean()
            return logits, {"ce": loss}
        elif isinstance(logits_student, None):
            loss = []
            # in each branch, less than one class > 0.5
            for br in logits_student:
                loss.append(br.sigmoid().topk(2)[0][:, 1].mean()) # len(loss)==3
            # in all branches, at least one class --> 1
            all_logits = torch.cat(logits_student, 1).sigmoid()
            loss.append(1 - all_logits.max(1)[0].mean()) # len(loss) == 4
            # for different classes, inner product --> 0
            class_diff = (target.unsqueeze(0) == target.unsqueeze(1)).logical_not()
            loss.append((all_logits @ all_logits.T)[class_diff].mean() * 2)
            # ce loss weighted 0.1
            # ce_loss = []
            # for br in logits_student:
                # ce_loss.append(1 * F.cross_entropy(br, target))
                # loss.append(1 * F.cross_entropy(br, target))
            # logits_pick = torch.zeros_like(logits_student[2])
            # for batch_id in range(len(logits_pick)):
            #     if stat_output_1[int(target[batch_id])] < 80:
            #         logits_pick[batch_id] = logits_student[1][batch_id]
            #     else:
            #         logits_pick[batch_id] = logits_student[2][batch_id]
            # loss.append(1 * F.cross_entropy(logits_pick, target))
            # loss.append(1 * F.cross_entropy(logits_student[2], target))
            # import pdb
            # pdb.set_trace()
            all_br = torch.stack(logits_student, 2).max(2)[0]
            loss.append(F.cross_entropy(all_br, target))
            # loss.append(min(ce_loss))
            # loss.append(ce_loss[-1])
            # assert(len(loss) == 8)
            
            loss = sum(loss)
            # logits_student_out = logits_student[2]
            logits_student_out = all_br
            # if logits_student[0].max() > 0:
            #     logits_student_out = logits_student[0]
            # elif logits_student[1].max() > 0:
            #     logits_student_out = logits_student[1]
            # else:
            #     logits_student_out = logits_student[2]
            # logits_student_out = logits_pick
            return logits_student_out, {"ce": loss}

            # loss = F.cross_entropy(logits_student[2], target)
            # return logits_student[2], {"ce": loss}
        else:
            loss = F.cross_entropy(logits_student, target)
            # return logits_student, {"ce": loss}
            num_classes = self.fc_s.weight.shape[0]
            cam_student = self.get_cam(feature_student, self.fc_s, num_classes, target)
            
            bs, c, h, w = cam_student.shape
            cams = torch.nn.functional.interpolate(cam_student, scale_factor=image.shape[-1]/h, mode='bilinear', align_corners=False)
            # print(cams.shape)
            
            cams_quantile_80 = cams.view(bs, -1).quantile(q=0.2, dim=-1).view(-1, 1, 1, 1)
            image_n_80 = image.clone()
            image_p_20 = image.clone()
            image_n_80[(cams <= cams_quantile_80).repeat(1, 3, 1, 1)] = 0
            image_p_20[(cams >= cams_quantile_80).repeat(1, 3, 1, 1)] = 0
            # pdb.set_trace()
            logits_n_80, _ = self.student(image_n_80)
            logits_p_20, _ = self.student(image_p_20)
            loss_n_80 = F.cross_entropy(logits_n_80, target)
            loss_p_20 = F.cross_entropy(logits_p_20, target)
            # pdb.set_trace()
            return logits_student, {"ce": loss + (0.3-loss.detach()).clamp(0)*(loss_n_80 + (4 - loss_p_20).clamp(0))}
        
    def get_cam(self, feature, fc, num_classes, target):
        feature = feature["feats"][-1]
        linear = fc.weight[target].detach()
        # pdb.set_trace()
        cam = (feature * linear.unsqueeze(-1).unsqueeze(-1)).mean(1, keepdims=True).clamp(0)#.permute(1, 0, 2, 3)
        return cam

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]

    
    def forward_cam(self, image, target, **kwargs):
        with torch.no_grad():
            logits_student, feature_student = self.student(image)
            logits_teacher, feature_teacher = self.teacher(image)
        pred_teacher = logits_teacher.softmax(1).max(1)

        # losses
        num_classes = self.fc_s.weight.shape[0]
        feature_student = feature_student["feats"][-1]
        feature_teacher = feature_teacher["feats"][-1]

        linear_student = self.fc_s.weight.view(num_classes, 1, -1, 1, 1)
        linear_teacher = self.fc_t.weight.view(num_classes, 1, -1, 1, 1)
        cam_student = ((feature_student) * linear_student).mean(2).clamp(0)
        cam_teacher = ((feature_teacher) * linear_teacher).mean(2).clamp(0)
        # pdb.set_trace()
        losses_dict = {}
        return logits_student, losses_dict