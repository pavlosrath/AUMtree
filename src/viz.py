from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def plot_differences(
        ax: plt.Axes, 
        scores: np.ndarray, 
        noise_mask: np.ndarray,
    ) -> plt.Axes:
    """
    Plot the AUM score over the range of estimators.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to use.
    scores : np.ndarray
        The label quality scores.
    noise_mask : np.ndarray
        A boolean mask indicating samples with label error.

    Returns
    -------
    plt.Axes
        The matplotlib axes.
    """
    # Split scores in clean and noisy
    clean_scores = scores[~noise_mask]
    noisy_scores = scores[noise_mask]

    # Compute statistics over samples along estimators
    mu_clean = np.mean(clean_scores, axis=0)
    sd_clean = np.std(clean_scores, axis=0)

    mu_noisy = np.mean(noisy_scores, axis=0)
    sd_noisy = np.std(noisy_scores, axis=0)

    # Plot the label quality scores with confidence intervals
    ax.fill_between(
        np.arange(scores.shape[1]), 
        mu_clean - sd_clean, 
        mu_clean + sd_clean, 
        alpha=0.2, 
        color='blue'
    )

    ax.fill_between(    
        np.arange(scores.shape[1]), 
        mu_noisy - sd_noisy, 
        mu_noisy + sd_noisy, 
        alpha=0.2, 
        color='red'
    )

    ax.plot(mu_clean, color='blue', label='Clean')
    ax.plot(mu_noisy, color='red', label='Noisy')

    ax.set_xlabel('Number of Estimators')
    ax.set_ylabel('Label Quality Score')
    ax.legend()

    return ax


def plot_histogram(
        ax: plt.Axes, 
        scores: np.ndarray, 
        noise_mask: np.ndarray,
    ) -> plt.Axes:
    """
    Plot the histogram of the label quality scores.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to use.
    scores : np.ndarray
        The label quality scores.
    noise_mask : np.ndarray
        A boolean mask indicating samples with label error.

    Returns
    -------
    plt.Axes
        The matplotlib axes.
    """
    # Split scores in clean and noisy
    clean_scores = scores[~noise_mask]
    noisy_scores = scores[noise_mask]

    # Plot the histogram of the label quality scores
    ax.hist(clean_scores.flatten(), bins=50, alpha=0.5, color='blue', label='Clean')
    ax.hist(noisy_scores.flatten(), bins=50, alpha=0.5, color='red', label='Noisy')

    ax.set_xlabel('Label Quality Score')
    ax.set_ylabel('Count')
    ax.legend()

    return ax   


def plot_roc_curve(
        ax: plt.Axes,
        scores: np.ndarray,
        noise_mask: np.ndarray,
    ) -> plt.Axes:
    """
    Plot the ROC curve of the label quality scores.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to use.
    scores : np.ndarray (n_samples,)
        The label quality scores.
    noise_mask : np.ndarray (n_samples,)
        A boolean mask indicating samples with label error.
    
    Returns
    -------
    plt.Axes
        The matplotlib axes.
    """
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(noise_mask, -scores)

    # Plot the ROC curve
    ax.plot(fpr, tpr, label='ROC', color='blue')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()

    # Compute the AUROC score
    auroc = roc_auc_score(noise_mask, -scores)
    ax.text(0.5, 0.3, f'AUROC: {auroc:.4f}', transform=ax.transAxes, ha='center')

    return ax


def plot_auroc(
        ax: plt.Axes, 
        scores: np.ndarray,
        noise_mask: np.ndarray,
        mark_max: bool=False,
        aggregate: bool=True,
    ) -> plt.Axes:
    """
    Plot the AUROC scores of the label quality scores over the range of estimators.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to use.
    scores : np.ndarray (n_samples, n_estimators)
        The label quality scores.
    noise_mask : np.ndarray
        A boolean mask indicating samples with label error.
    mark_max : bool (default=False)
        Whether to mark the maximum AUROC score in the legend.
    aggregate : bool (default=True)
        Whether to aggregate the scores over the estimators.
        Setting to False is useful e.g. with the cross-validation scores.

    Returns
    -------
    plt.Axes
        The matplotlib axes.
    """
    # Compute the AUROC score
    auroc = np.zeros(scores.shape[1])
    for i in range(scores.shape[1]):
        if aggregate:
            auroc[i] = roc_auc_score(noise_mask, -scores[:, :i+1].mean(axis=1))
        else:
            auroc[i] = roc_auc_score(noise_mask, -scores[:, i])

    # Plot the AUROC score
    ax.plot(range(1, scores.shape[1]+1), auroc, label='AUROC', color='blue')
    if mark_max:
        max_idx = np.argmax(auroc)
        label = f'Max AUROC {auroc[max_idx]:.4} @{max_idx+1}'
        ax.plot(max_idx+1, auroc[max_idx], 'ro', label=label)
        ax.legend()
    ax.set_xlabel('Number of Estimators')
    ax.set_ylabel('AUROC')

    return ax


def plot_spirals(
        ax: plt.Axes,  
        X: np.ndarray,
        y: np.ndarray,
        s: float=10.0, 
        alpha: float=0.7
    ) -> plt.Axes:
    """
    Plot the spirals dataset.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to use.
    X : np.ndarray (n_samples, 2)
        The input data.
    y : np.ndarray (n_samples,)
        The ground truth labels.
    s : float
        The marker size.
    alpha : float
        The marker transparency.

    Returns
    -------
    plt.Axes
        The matplotlib axes.
    """
    # Plot the spirals dataset
    for i in np.unique(y):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], s=s, alpha=alpha)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    return ax