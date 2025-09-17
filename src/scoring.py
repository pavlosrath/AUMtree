import numpy as np

from cleanlab.filter import find_label_issues
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.validation import compute_sublearner_probas


def compute_AUM_scores(
        model: XGBClassifier,  
        X: np.ndarray, 
        y_noisy: np.ndarray,
        additional_scores: bool = False
    ) -> np.ndarray:
    """ 
    Estimates label quality using the area under the margin (AUM) scores by 
    Pleiss et al. (2020).

    Computes the difference between the ground truth probability and the highest
    non ground truth probability for each sample in X and each range of estimators.
    These scores are called the area under the margin (AUM) scores in 
    Pleiss et al. (2020). To obtain per sample AUM scores, average over the 
    second axis (i.e., training steps / sub-estimators).

    Parameters
    ----------
    model : XGBClassifier
        The Gradient Boosted Trees model to use for prediction.
    X : np.ndarray (n_samples, n_features)
        The input data.
    y_noisy : np.ndarray (n_samples,)
        The noisy labels.
    additional_scores : bool
        Whether to return additional scores. Default is False.

    Returns
    -------
    np.ndarray (n_samples, n_estimators)
        The scores for each sample in X and each range of estimators.
    Optional:
    np.ndarray (n_samples, n_estimators), np.ndarray (n_samples, n_estimators)
        The ground truth and highest non ground truth probability for each sample 
        in X and each range of estimators.
    np.ndarray (n_samples, n_estimators)
        The naive scores.
    np.ndarray (n_samples, n_estimators)
        The entropy scores.
    """
    X_range = np.arange(X.shape[0])

    # Compute sublearner probabilities
    sublearner_probas = compute_sublearner_probas(model, X) 

    # Get the ground truth probability
    values_pos = sublearner_probas[X_range, :, y_noisy]
    # Get the highest non ground truth probability
    sub_learner_probas_copy = sublearner_probas.copy()
    sub_learner_probas_copy[X_range, :, y_noisy] = 0.
    values_neg = sub_learner_probas_copy.max(axis=2)

    # Calculate the (estimator-wise) outlier score as difference between the 
    # ground truth probability and the highest non ground truth probability
    scores = values_pos - values_neg

    if additional_scores:
        values_naive = (
            sublearner_probas.argmax(axis=2) == y_noisy[:, None]
        ).astype(float)
        values_entropy = -np.sum(
            sublearner_probas * np.log(sublearner_probas), axis=2
        )
        return scores, values_naive, values_pos, values_entropy
    
    return scores


def compute_all_cls_AUM(
        model: XGBClassifier,  
        X: np.ndarray,
        additional_scores: bool = False
    ) -> np.ndarray:
    """ 
    Estimates label quality using the area under the margin (AUM) scores by 
    Pleiss et al. (2020) for *each class*.

    Computes the difference between the ground truth probability and the highest
    non ground truth probability for each sample in X and each range of 
    estimators. These scores are called the area under the margin (AUM) scores 
    in Pleiss et al. (2020).

    To obtain per sample label quality scores use:
        `score_values.mean(axis=1)[np.arange(X.shape[0]), y_noisy]`

    Parameters
    ----------
    model : XGBClassifier
        The Gradient Boosted Trees model to use for prediction.
    X : np.ndarray (n_samples, n_features)
        The input data.
    additional_scores : bool
        Whether to return additional scores. Default is False.

    Returns
    -------
    scores: np.ndarray (n_samples, n_estimators, n_classes)
        The scores for each sample in X and each range of estimators.
    Optional:
    naive_scores: np.ndarray (n_samples, n_estimators, n_classes)
    values_entropy: np.ndarray (n_samples, n_estimators)
    values_pos: np.ndarray (n_samples, n_estimators, n_classes)
    
    """
    n_samples = X.shape[0]
    n_estimators = model.n_estimators
    n_classes = len(set(model.classes_))

    # Compute sublearner probabilities and predicted classes
    sublearner_probas = compute_sublearner_probas(model, X) 
    predicted_classes = sublearner_probas.argmax(axis=2, keepdims=True)
    sort_idxs = np.argsort(sublearner_probas, axis=2)

    # Compute the AUM scores
    values_pos = sublearner_probas.copy()
    
    # Set the highest activation as the negative
    values_neg = np.zeros((n_samples, n_estimators, n_classes))
    values_neg[:, :, :] = np.take_along_axis(
        sublearner_probas, sort_idxs[:, :, -1][:, :, None], axis=2
    )
    # ... unless it is the activation for the current class 
    mask = predicted_classes == model.classes_[None, None, :]
    values_neg[mask] = np.take_along_axis(
        sublearner_probas, sort_idxs[:, :, -2][:, :, None], axis=2
    )[:, :, 0].reshape(-1)

    # Calculate the (estimator-wise) outlier score as difference between the 
    # ground truth probability and the highest non ground truth probability
    scores = values_pos - values_neg

    if additional_scores:
        values_naive = (
            predicted_classes == model.classes_[None, None, :]
        ).astype(float)
        values_entropy = -np.sum(
            sublearner_probas * np.log(sublearner_probas), 
            axis=2, keepdims=True
        )
        return scores, values_naive, values_pos, values_entropy
    
    return scores


def cross_val_label_error_detection(
        y_noisy: np.ndarray,
        oos_probas: np.ndarray,
    ) -> np.ndarray:
    """
    Cross-validation for label error detection using the stratified k-fold 
    method.
    TODO: compare against Cleanlab

    Parameters
    ----------
    y_noisy : np.ndarray (n_samples,)
        The noisy labels.
    oos_probas : np.ndarray (n_samples, n_estimators, n_classes)

    Returns
    -------
    np.ndarray (n_samples, n_estimators)
        The out-of-sample (OOS) probabilities of each sample and each range of 
        estimators for the assigned class as a label quality score. I.e. self 
        confidence score, i.e. the self confidence scores.
    np.ndarray (n_samples, n_estimators)
        The normalized margins for each sample and each range of estimators.
        p(label = k) - max(p(label != k))
    """
    
    # Predict the scores
    label_error_scores = oos_probas[np.arange(oos_probas.shape[0]), :, y_noisy]
    estimator_probas = oos_probas.copy()
    estimator_probas[np.arange(oos_probas.shape[0]), :, y_noisy] = 0.
    normalized_margins = label_error_scores - estimator_probas.max(axis=2)
    
    return label_error_scores, normalized_margins


def cleanlab_label_error_detection(
        y_noisy: np.ndarray,
        oos_probas: np.ndarray,
        filter_by: str = "prune_by_noise_rate", 
        indices_ranked_by: str = "self_confidence",
    ) -> np.ndarray:
    """
    Label error detection using the Cleanlab library.

    Parameters
    ----------
    y_noisy : np.ndarray (n_samples,)
        The noisy labels.
    oos_probas : np.ndarray (n_samples, n_estimators, n_classes)
        The out-of-sample probabilities for each sample and each range of
        estimators.
    filter_by : str
        The method to use for filtering label issues. 
        Default is "prune_by_noise_rate".
    indices_ranked_by : str
        The method to use for ranking label issues.
        Default is "self_confidence".

    Returns
    -------
    np.ndarray (n_samples, n_estimators)
        The label quality scores for each sample and each range of estimators.
    """

    # Give non-issue samples a perfect label quality score
    # and rank all label issue samples by the selected method
    label_quality_scores = np.ones(oos_probas.shape[:2], dtype=float)

    for i, estimator_oos_probas in enumerate(np.swapaxes(oos_probas, 0, 1)):
        issue_idxs = find_label_issues(
            y_noisy, estimator_oos_probas,
            filter_by = filter_by,
            return_indices_ranked_by = indices_ranked_by,
            n_jobs=1 
        )
        label_quality_scores[issue_idxs, i] = np.linspace(
            0., 1., len(issue_idxs), endpoint=False
        )
    
    return label_quality_scores

def calc_val_scores(
        scores: np.ndarray, 
        noise_mask: np.ndarray,
        percentiles: list[float] = [0.95],
    ) -> tuple[float, list[float]]:
    """
    Calculate the AUROC and FPR at TPR xx% scores to evaluate the label error 
    detection.

    Parameters
    ----------
    scores : np.ndarray (n_samples,)
        The label quality scores. The higher the score, the cleaner the sample.
    noise_mask : np.ndarray (n_samples,)
        A boolean mask indicating samples with label error.
    percentiles: list[float]
        The percentiles to use for the FPR calculation. Default is [0.95]
    
    Returns
    -------
    float, list[float]
        The AUROC and FPR at TPR scores at the requested percentiles.
    """
    
    auroc = roc_auc_score(noise_mask, -scores)

    thresholds = [
        np.quantile(scores[noise_mask], percentile) 
        for percentile in percentiles
    ]
    fpr_at_tprs = [
        np.sum(scores[~noise_mask] <= threshold) / np.sum(scores <= threshold) 
        for threshold in thresholds
    ]

    return auroc, fpr_at_tprs