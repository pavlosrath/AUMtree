from typing import Optional

import json
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold

from src.label_noise import generate_noisy_labels
from scipy.special import softmax


def get_basescore(model: XGBClassifier) -> float:
    """Get base score from an XGBoost sklearn estimator."""
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    base_score = float(config["learner"]["learner_model_param"]["base_score"])
    return base_score


def compute_sublearner_probas(
        model: XGBClassifier, X: np.ndarray
    ) -> np.ndarray:
    """
    Computes the sublearner probabilities for each range of estimators.
    Computes the probabilities for multiclass classification by reconstructing
    the probabilities from the margins to reduce computation time.

    Parameters
    ----------
    model : XGBClassifier
        The model to use.
    X : np.ndarray (n_samples, n_features)
        The input data.

    Returns
    -------
    probas : np.ndarray (n_samples, n_estimators, n_classes)
        The probabilities for each sample and each range of estimators.
    """
    n_samples = X.shape[0]
    n_estimators = model.n_estimators
    n_classes = len(set(model.classes_))

    if n_classes == 2:
        # Due to some hidden reason, the cumulative approach does not work for 
        # binary classification
        probas = np.zeros((n_samples, n_estimators, n_classes))
        for i in range(n_estimators):
            probas[:, i] = model.predict_proba(X, iteration_range=(0, i+1))
    else:
        # Instead using iteration_range=(0, i+1), reduce the computation time
        # by reconstructing the probabilities from the margins
        base_score = get_basescore(model)
        margins = np.zeros((n_samples, n_estimators, n_classes))
        for i in range(n_estimators):
            margins[:, i] = model.predict(
                X, iteration_range=(i, i+1), output_margin=True,
            ) - base_score * (i > 0) # subtract the base score 

        cumulative_margins = np.cumsum(margins, axis=1)
        probas = softmax(cumulative_margins, axis=2)      

    return probas


def cross_validate_sublearners(
        model: XGBClassifier,
        X: np.ndarray,
        y: np.ndarray,
        transition_matrix: Optional[np.ndarray] = None,
        n_splits: int = 5,
        y_noisy: Optional[np.ndarray] = None,
        random_state: Optional[int] = 42,
    ) -> np.ndarray:
    """
    Performs stratified cross-validation for the additive model and returns
    the accuracy for each range of estimators. The model is trained on noisy
    labels and the accuracy is computed on the ground truth labels.

    Parameters
    ----------
    model : XGBClassifier
        The model to use.
    X : np.ndarray (n_samples, n_features)
        The input data.
    y : np.ndarray (n_samples,)
        The ground truth labels.
    transition_matrix : np.ndarray (n_classes, n_classes), optional
        The transition matrix.
    n_splits : int
        The number of splits to use
    y_noisy : np.ndarray (n_samples,), optional
        The noisy labels. If not provided, they are generated from the ground
        truth labels using the transition matrix.
    random_state : int, optional
        The random state to use for cross-validation.
    final_only : bool
        Whether to return only the accuracy for the final range of estimators.

    Returns
    -------
    accuracies : np.ndarray (n_estimators,)
        The accuracy for each range of estimators.
    oos_probas : np.ndarray (n_samples, n_estimators, n_classes)
        The out-of-sample probabilities for each sample and each range of 
        estimators.
    """
    assert (transition_matrix is None) ^ (y_noisy is None), \
        "Either the transition matrix or the noisy labels should be provided."
    # If the transition matrix is provided, generate the noisy labels
    if transition_matrix is not None:
        y_noisy, _ = generate_noisy_labels(y, transition_matrix)

    n_samples = X.shape[0]
    n_estimators = model.n_estimators
    n_classes = len(set(y)) # works bc. classes are 0, 1, ..., n_classes-1

    # One cannot use class-stratified cross-validation with the _GT_ labels here,
    # because they are not available in practise.
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    oos_probas = np.zeros((n_samples, n_estimators, n_classes))

    for train_index, test_index in skf.split(X, y_noisy):
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train_noisy = y_noisy[train_index]

        used_classes = np.unique(y_train_noisy)

        # Create custom class indexing
        cls_idx_proj_m = np.zeros((n_classes, len(used_classes)), dtype=int)
        cls_idx_mapping = {}
        for i, cls in enumerate(used_classes):
            cls_idx_proj_m[cls, i] = 1
            cls_idx_mapping[cls] = i
        forward_mapping = np.vectorize(cls_idx_mapping.get)

        # Fit the model
        model.fit(X_train, forward_mapping(y_train_noisy))

        # Predict the labels for each range of estimators
        oos_probas[test_index] = compute_sublearner_probas(
            model, X_test
        ) @ cls_idx_proj_m.T

    y_pred = np.argmax(oos_probas, axis=2)
    accuracies = np.mean(y_pred == y.reshape(-1, 1), axis=0)
    
    return accuracies, oos_probas