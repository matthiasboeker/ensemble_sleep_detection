import torch
import matplotlib.pyplot as plt
import numpy as np


def brier_score_decomposition(predicted, target, bin_boundaries):
    N = predicted.size(0)

    # Binning the predicted probabilities
    bin_indices = torch.bucketize(predicted, bin_boundaries)
    bin_means = torch.bincount(bin_indices, weights=predicted) / torch.bincount(bin_indices)
    bin_observed = torch.bincount(bin_indices, weights=target) / torch.bincount(bin_indices)
    bin_counts = torch.bincount(bin_indices).float()

    overall_observed_freq = torch.mean(target)

    # Reliability
    reliability = torch.sum(bin_counts * (bin_observed - bin_means) ** 2) / N

    # Resolution
    resolution = torch.sum(bin_counts * (bin_observed - overall_observed_freq) ** 2) / N

    # Uncertainty
    uncertainty = overall_observed_freq * (1 - overall_observed_freq)

    return reliability, resolution, uncertainty


def calibration_data(predicted, target, bin_boundaries):
    bin_indices = torch.bucketize(predicted, bin_boundaries)

    bin_sums = torch.bincount(bin_indices, weights=predicted, minlength=len(bin_boundaries) + 1)
    bin_true = torch.bincount(bin_indices, weights=target, minlength=len(bin_boundaries) + 1)
    bin_counts = torch.bincount(bin_indices, minlength=len(bin_boundaries) + 1).float()

    bin_means = bin_sums / bin_counts
    bin_true_probs = bin_true / bin_counts

    return bin_means, bin_true_probs


def compute_ece(predicted, target, bin_boundaries):
    """
    Computes the Expected Calibration Error (ECE).

    Args:
        predicted (torch.Tensor): Tensor of predicted probabilities for the positive class.
        target (torch.Tensor): Tensor of true labels (0 or 1).
        bin_boundaries (list): List of boundaries that defines the bins, e.g., [0.1, 0.2, ..., 0.9].

    Returns:
        ece (torch.Tensor): The Expected Calibration Error.
    """
    #bin_indices = torch.bucketize(predicted, bin_boundaries)
    bin_indices = torch.bucketize(predicted, torch.tensor(bin_boundaries))
    bin_sums = torch.bincount(bin_indices, weights=predicted.detach().float())
    bin_true = torch.bincount(bin_indices, weights=torch.tensor(target).float())
    bin_counts = torch.bincount(bin_indices).float()

    bin_means = bin_sums / bin_counts
    bin_true_probs = bin_true / bin_counts

    ece = torch.sum(bin_counts / predicted.size(0) * torch.abs(bin_true_probs - bin_means))

    return ece


def plot_calibration_curve(predicted, target, bin_boundaries):
    bin_means, bin_true_probs = calibration_data(predicted, target, bin_boundaries)

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
    plt.plot(bin_means, bin_true_probs, marker='o', label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.legend(loc="best")
    plt.title("Calibration Plot")
    plt.show()


def ar_covariance_estimation(ts, max_lag=1):
    residuals = ts - torch.mean(ts, dim=0)

    # Compute the regular variance
    sigma_0 = torch.var(residuals)

    nw_variance = sigma_0

    for l in range(1, max_lag + 1):
        weight = 1 - l / (max_lag + 1)
        lagged_residuals = torch.roll(residuals, shifts=l, dims=0)
        lagged_residuals[:l] = 0  # set the first l residuals to zero

        sigma_l = torch.mean(residuals[l:] * lagged_residuals[l:])

        nw_variance += 2 * weight * sigma_l  # multiply by 2 because it's symmetric
    return nw_variance

