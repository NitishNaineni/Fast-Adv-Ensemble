import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

def adp_loss(inputs, targets, models, lamda, log_det_lamda, num_classes, label_smoothing):
    y_true = torch.zeros(inputs.size(0), num_classes).cuda()
    y_true.scatter_(1, targets.view(-1, 1), 1)

    loss_std = 0
    mask_non_y_pred = []
    ensemble_probs = 0

    for model in models:
        with autocast():
            outputs = model(inputs)
            loss_std += F.cross_entropy(outputs, targets, reduction='mean', label_smoothing=label_smoothing)
            y_pred = F.softmax(outputs, dim=-1)
        bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true, torch.ones_like(y_true))  
        mask_non_y_pred.append(torch.masked_select(y_pred, bool_R_y_true).reshape(-1, num_classes - 1))
        ensemble_probs += y_pred
    
    ensemble_probs = ensemble_probs / len(models)
    ensemble_entropy = torch.sum(-torch.mul(ensemble_probs, torch.log(ensemble_probs + 1e-20)),dim=-1).mean()

    mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
    
    mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1,keepdim=True)
    matrix = torch.matmul(mask_non_y_pred, mask_non_y_pred.permute(0, 2, 1))
    log_det = torch.logdet(matrix + 1e-6 * torch.eye(len(models), device=matrix.device).unsqueeze(0)).mean()

    log_det = log_det / len(models)**2
    loss_std = loss_std / len(models)
    loss = loss_std - lamda * ensemble_entropy - log_det_lamda * log_det

    return loss, torch.log(ensemble_probs)

def trades_loss(inputs, adv_inputs, targets, nets, beta, lamda, log_det_lamda, num_classes, label_smoothing):
    nat_loss, nat_outputs = adp_loss(
        inputs,
        targets, 
        nets, 
        lamda = lamda, 
        log_det_lamda = log_det_lamda, 
        num_classes= num_classes,
        label_smoothing = label_smoothing
    )
    adv_loss, adv_outputs = adp_loss(
        adv_inputs,
        targets, 
        nets, 
        lamda = lamda, 
        log_det_lamda = log_det_lamda, 
        num_classes= num_classes,
        label_smoothing = label_smoothing
    )
    robust_loss = F.kl_div(adv_outputs, nat_outputs, reduction='batchmean', log_target=True)
    loss = nat_loss + adv_loss + beta * robust_loss

    return loss, nat_outputs, adv_outputs