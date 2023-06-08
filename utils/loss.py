import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

def adp_loss(inputs, targets, ensemble, lamda, log_det_lamda, num_classes, label_smoothing):
    """
    Calculates the ADP (Adversarial Diversity Promoting) loss given the inputs and targets, a list of models, 
    regularization coefficients, number of classes, and label smoothing factor.

    Args:
    - inputs (tensor): input data
    - targets (tensor): ground truth labels
    - ensemble : ensemble of models
    - lamda (float): regularization coefficient for ensemble entropy
    - log_det_lamda (float): regularization coefficient for the log determinant
    - num_classes (int): number of classes
    - label_smoothing (float): label smoothing factor

    Returns:
    - loss (tensor): ADP loss
    - ensemble_probs (tensor): the softmax output of the ensemble model
    """

    # Create one-hot encoding of ground truth labels
    y_true = torch.zeros(inputs.size(0), num_classes).cuda()
    y_true.scatter_(1, targets.view(-1, 1), 1)

    loss_std = 0
    mask_non_y_pred = []
    ensemble_probs = torch.zeros(inputs.size(0), num_classes, device=inputs.device)

    with autocast():
        outputs = ensemble(inputs, ensemble=False)
        for output in outputs:
            loss_std += F.cross_entropy(output, targets, reduction='mean', label_smoothing=label_smoothing)
            # Softmax prediction probabilities
            y_pred = F.softmax(output, dim=-1)
            # Boolean mask to select the non-true classes in the one-hot encoding
            bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true, torch.ones_like(y_true))
            # Append non-true class prediction probabilities to a list
            mask_non_y_pred.append(torch.masked_select(y_pred, bool_R_y_true).reshape(-1, num_classes - 1))
            # Add the softmax output to ensemble probabilities
            ensemble_probs.add_(y_pred)

    # Average the ensemble probabilities
    ensemble_probs = ensemble_probs / ensemble.num_models
    # Calculate the ensemble entropy
    ensemble_entropy = torch.sum(-torch.mul(ensemble_probs, torch.log(ensemble_probs + 1e-20)),dim=-1).mean()

    # Stack the non-true class prediction probabilities
    mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
    # Normalize the stacked probabilities
    mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1, keepdim=True)
    # Calculate the matrix product of the stacked probabilities
    matrix = torch.matmul(mask_non_y_pred, mask_non_y_pred.permute(0, 2, 1))
    # Calculate the log determinant
    log_det = torch.logdet(matrix + 1e-6 * torch.eye(ensemble.num_models, device=matrix.device).unsqueeze(0)).mean()

    # Average the log determinant
    log_det = log_det / ensemble.num_models**2
    # Average the standard loss
    loss_std = loss_std / ensemble.num_models
    # Calculate the ADP loss
    loss = loss_std - lamda * ensemble_entropy - log_det_lamda * log_det

    return loss, torch.log(ensemble_probs)

def trades_loss(inputs, adv_inputs, targets, ensemble, beta, lamda, log_det_lamda, num_classes, label_smoothing):
    """
    Calculates the TRADES (Theoretically grounded Regularized Adversarial Defense with Ensemble of
    Shrinking and Expanding Prediction Sets) loss given the inputs, adversarial inputs, targets, a list of models,
    regularization coefficients, number of classes, and label smoothing factor.

    Args:
    - inputs (tensor): input data
    - adv_inputs (tensor): adversarial input data
    - targets (tensor): ground truth labels
    - ensemble : ensemble of models
    - beta (float): regularization coefficient for the KL divergence between the natural and adversarial
    prediction probabilities
    - lamda (float): regularization coefficient for ensemble entropy
    - log_det_lamda (float): regularization coefficient for the log determinant
    - num_classes (int): number of classes
    - label_smoothing (float): label smoothing factor

    Returns:
    - loss (tensor): TRADES loss
    - nat_outputs (tensor): log prob output of the ensemble model for the natural input
    - adv_outputs (tensor): log prob output of the ensemble model for the adversarial input
    """

    # Concatenate the natural and adversarial inputs and targets
    # stacked_inputs = torch.cat([inputs, adv_inputs], dim=0)
    # stacked_targets = torch.cat([targets, targets], dim=0)

    # Calculate the ADP loss for the concatenated inputs and targets
    std_loss, adv_outputs = adp_loss(
        adv_inputs,
        targets,
        ensemble,
        lamda=lamda,
        log_det_lamda=log_det_lamda,
        num_classes=num_classes,
        label_smoothing=label_smoothing
    )
    nat_outputs = ensemble(inputs)

    # Calculate the KL divergence between the adversarial and natural output probabilities
    robust_loss = F.kl_div(adv_outputs, nat_outputs, reduction='batchmean', log_target=True)
    # Calculate the TRADES loss
    loss = std_loss + beta * robust_loss

    return loss, nat_outputs, adv_outputs