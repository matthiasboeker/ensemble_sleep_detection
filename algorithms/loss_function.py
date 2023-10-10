import torch
import torch.nn as nn
import torch.nn.functional as F


def transform_to_ranked_class(tensor):
    # Compute the mean along the last dimension (assuming your sequences are along the last dimension)
    means = tensor.mean(dim=-1)

    # Map means to classes based on the above-defined bins
    # Using searchsorted here gives us the bins directly as it would return the
    # indices where elements should be inserted to maintain order.
    bins = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
    ranked_classes = torch.searchsorted(bins, means)

    return ranked_classes


def brier_score(predicted, target):
    return torch.mean((predicted - target) ** 2)


class BrierScoreLoss(nn.Module):
    def __init__(self):
        super(BrierScoreLoss, self).__init__()

    def forward(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): Tensor of predicted probabilities for the positive class.
            target (torch.Tensor): Tensor of true labels (0 or 1).
        Returns:
            torch.Tensor: The Brier Score loss.
        """
        return brier_score(predicted, target)


class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, predicted, target):
        s = torch.column_stack((1 - predicted, predicted))
        c = torch.column_stack((1 - target, target))
        logprobs = torch.log(c)
        return -(s * logprobs).sum() / predicted.shape[0]


class SupersetBrierScoreLoss(nn.Module):
    def __init__(self, optimism):
        super(SupersetBrierScoreLoss, self).__init__()
        self.optimism = optimism

    def forward(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): Tensor of predicted probabilities for the positive class.
            target (torch.Tensor): Superset of true labels (0 or 1).
        Returns:
            torch.Tensor: The Brier Score loss.
        """
        superset_brier = (target - predicted[:, None])**2
        return torch.mean(superset_brier.mean(dim=1))


class SupersetCrossEntropy(nn.Module):
    def __init__(self):
        super(SupersetCrossEntropy, self).__init__()

    def forward(self, predicted, target):
        """
        Args:
            predicted (torch.Tensor): Tensor of predicted probabilities for the positive class.
            target (torch.Tensor): Superset of true labels (0 or 1).
        Returns:
            torch.Tensor: The Brier Score loss.
        """
        losses = []
        for i in range(target.shape[1]):
            loss_i = F.binary_cross_entropy(predicted, target[:, i])
            losses.append(loss_i)

        # Combine individual losses and compute the mean
        return torch.stack(losses).mean()


def gaussian_kernel(x, a=1, b=3.5, c=1):
    return a * torch.exp(-(x - b)**2 / (2 * c**2))


class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        ranked_classes = transform_to_ranked_class(targets)
        ce_loss = F.cross_entropy(predictions, ranked_classes, reduction='none')
        weighted_loss = ce_loss * self.weights[ranked_classes]
        return weighted_loss.mean()
