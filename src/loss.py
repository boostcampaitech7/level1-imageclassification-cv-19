import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=1, gamma_neg=4, reduction='mean'):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        p_t = (probs * targets_one_hot).sum(dim=1)
        loss_pos = -((1 - p_t) ** self.gamma_pos) * torch.log(p_t)
        loss_neg = -((probs * (1 - targets_one_hot)).sum(dim=1) ** self.gamma_neg) * torch.log(1 - probs.sum(dim=1))
        loss = loss_pos + loss_neg
        return loss.mean()
# SupConLoss implementation
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels):
        features = F.normalize(features, dim=1)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        logits = torch.div(self.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0)), self.temperature)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        exp_logits = torch.exp(logits) * mask
        log_prob = torch.log(exp_logits / exp_logits.sum(dim=1, keepdim=True))
        return -(log_prob * mask).sum(dim=1).mean()

class Loss(nn.Module):
    def __init__(self, loss_type='focal'):
        super(Loss, self).__init__()
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(gamma=2)
        elif loss_type == 'asymmetric':
            self.loss_fn = AsymmetricLoss()
        else:
            raise ValueError("Unsupported loss type.")

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)
