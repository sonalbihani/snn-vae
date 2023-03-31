import torch.nn as nn
import torch
from torch.nn import functional as F


class VAELoss(nn.Module):
    def __init__(self, args=None):
        super(VAELoss, self).__init__()
        self.param_loss = nn.MSELoss()
    def forward(self, means, log_variances, output, target):
        KLD = torch.mean(-0.5 * torch.sum(1 + log_variances - means ** 2 - log_variances.exp(), dim = 1), dim = 0)
        ReconstructionLoss = self.param_loss(output, target)
        return ReconstructionLoss, KLD



class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss