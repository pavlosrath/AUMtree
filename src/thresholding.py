
import numpy as np

from xgboost import XGBClassifier

from src.scoring import compute_AUM_scores


def pleiss_threshold(
        model: XGBClassifier,  
        X: np.ndarray, 
        y_noisy: np.ndarray,
        noise_rate: float = 0.1,
        percentile: float = 99.0,
        random_state: int = 42,
) -> float:
    """
    Computes the threshold for label errors based on Pleiss (2020).

    Parameters
    ----------
    model : XGBClassifier
        The trained XGBoost classifier.
    X : np.ndarray
        The input data.
    y_noisy : np.ndarray
        The noisy labels.
    noise_rate : float
        The noise rate.
    percentile : float
        The percentile to use for the threshold.
    random_state : int
        The random state to use.

    Returns
    -------
    float
        The threshold
    """
    rnd_gen = np.random.default_rng(random_state)

    cls_labels = set(y_noisy)
    n_classes = len(cls_labels)

    # Check that the classes are labeled incrementally
    assert list(cls_labels) == list(range(n_classes))

    y_thresholding = y_noisy.copy()
    # Samples to be assinged to the new class that mimics mislabeled data
    sample_idxs_for_thresholding = rnd_gen.choice(
        len(y_thresholding),
        size=int(noise_rate * len(y_thresholding)),
        replace=False,
    )
    y_thresholding[sample_idxs_for_thresholding] = n_classes

    model.fit(X, y_thresholding)

    scores_thresholding = compute_AUM_scores(
        model,
        X[sample_idxs_for_thresholding],
        y_thresholding[sample_idxs_for_thresholding],
    ).mean(axis=1)

    return np.percentile(scores_thresholding, percentile)