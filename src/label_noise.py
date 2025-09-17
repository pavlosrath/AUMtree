from typing import Tuple
import warnings
import numpy as np


def generate_noisy_labels(
        y: np.ndarray, 
        transition_matrix: np.ndarray,
        random_state: int = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate noisy labels given clean labels and a transition matrix.
    Returns the noisy labels and a boolean mask indicating samples with label 
    error (i.e., flipped labels).

    Parameters
    ----------
    y : np.ndarray
        The clean labels.
    transition_matrix : np.ndarray
        The transition matrix.
    random_state : int, optional
        The random seed to use.

    Returns
    -------
    noisy_labels : np.ndarray
        The noisy labels.
    noise_mask : np.ndarray
        A boolean mask indicating samples with label error.
    """
    rng_gen = np.random.default_rng(random_state)

    n_classes = len(transition_matrix)
    n_samples = len(y)

    actual_classes = np.unique(y)
    assert len(actual_classes) == n_classes, (
        f"The noise transition matrix describes transitions for {n_classes}"
        f"classes but there are {len(actual_classes)} classes in the dataset."
    )
    assert actual_classes[-1] == n_classes-1, (
        f"The indexing of the labels is not as expected."
        f"Labels should start at 0 and increment. {actual_classes}"
    )

    # Add label noise to the data by iterating over the classes
    noisy_labels = np.zeros((n_samples,), dtype=int)
    noise_mask = np.zeros((n_samples,), dtype=bool)
    for i in range(n_classes):
        target_mask = y == i
        noisy_labels[target_mask] = rng_gen.choice(
            n_classes, np.sum(target_mask), p=transition_matrix[i]
        )
        noise_mask[target_mask] = noisy_labels[target_mask] != i

    return noisy_labels, noise_mask


def create_uniform_transition_matrix(
        n_classes: int, 
        noise_level: float, 
    ) -> np.ndarray:
    """
    Create a uniform transition matrix for a given number of classes.
    I.e., the noise is uniformly distributed across the other classes.

    Parameters
    ----------
    n_classes : int
        The number of classes.
    noise_level : float
        The level of noise to add to the labels.

    Returns
    -------
    transition_matrix : np.ndarray (n_classes, n_classes)
        The transition matrix.
    """
    assert 0 <= noise_level <= 1, 'Noise level must be between 0 and 1.'
    assert noise_level < 0.5, 'Noise level too high.'

    # Distribute the noise uniformly across the classes
    noise_level = noise_level / (n_classes - 1)
    transition_matrix = np.zeros((n_classes, n_classes)) + noise_level
    transition_matrix += (1 - n_classes * noise_level) * np.eye(n_classes)

    return transition_matrix


def generate_derangement(n: int, random_state: int = 42) -> np.ndarray:
    """
    Generate a derangement of n elements. A derangement is a permutation of the
    elements of a sequence such that no element appears in its original position.

    I.e., for each i, perm[i] != i.
    Example: [1, 0, 2] is a derangement of [0, 1, 2], but [1, 2, 0] is not.
    
    Parameters
    ----------
    n : int
        The number of elements in the permutation.
    random_state : int, optional
        The random seed to use.

    Returns
    -------
    perm : np.ndarray (n,)
        The derangement.
    """
    rng_gen = np.random.default_rng(random_state)

    perm = np.arange(n)
    for i in range(n):
        # Swap the current element with a randomly chosen element
        # that is not its own index and whose index is not the current element
        # TODO check that there will always be indices available for swapping
        swap_index = rng_gen.choice(
            np.where((perm != i) & (np.arange(n) != perm[i]))[0]
        )
        perm[i], perm[swap_index] = perm[swap_index], perm[i]
    
    assert (perm != np.arange(n)).all(), 'Permutation is not valid.'
    return perm


def create_asymmetric_transition_matrix(
        n_classes: int, 
        noise_level: float, 
        method: str = 'random',
        random_state: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an asymmetric transition matrix for a given number of classes.
    An asymmetric transition matrix is one where the noise is not uniformly
    distributed across the classes but is concentrated on a single other class.

    Parameters
    ----------
    n_classes : int
        The number of classes.
    noise_level : float
        The level of noise to add to the labels.
    method : str
        The method to use to create the transition matrix.
        Case 'random': Randomly select a noisy class.
        Case 'shift': Shift the noisy class indices by one.
    random_state : int, optional
        The random seed to use for the derangement, i.e. the permutation.

    Returns
    -------
    transition_matrix : np.ndarray (n_classes, n_classes)
        The transition matrix.
    """
    assert 0 <= noise_level <= 1, 'Noise level must be between 0 and 1.'
    assert noise_level < 0.5, 'Noise level too high.'

    transition_matrix = np.zeros((n_classes, n_classes))
    if method == 'random':
        # Choose one noisy class at random per true class without
        # choosing the true class itself. Also choose each class only once.
        noisy_classes = generate_derangement(n_classes, random_state)
        for i, noisy_class in enumerate(noisy_classes):
            # Add noise to the transition matrix
            transition_matrix[i, i] = 1 - noise_level
            transition_matrix[i, noisy_class] = noise_level
    elif method == 'shift':
        for i in range(n_classes):
            # Shift the noisy class by one
            noisy_class = (i + 1) % n_classes
            # Add noise to the transition matrix
            transition_matrix[i, i] = 1 - noise_level
            transition_matrix[i, noisy_class] = noise_level

    # check that each label stills primarily contains elements of its own class
    for i in range(n_classes):
        max_other_prob = np.delete(transition_matrix[i], i).max()
        assert transition_matrix[i, i] > max_other_prob, (
            f"The transition matrix is not valid. {transition_matrix[i]}"
        )

    return transition_matrix


def estimate_transition_matrix(
        model: object,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
    """
    Estimate the transition_matrix from a model, by averaging the predicted
    probabilities for each class over unseen data.
    Ref.: Jakubik et al. (2024) https://arxiv.org/abs/2405.09602

    Parameters
    ----------
    model : object
        The model to use for prediction.
    X : np.ndarray (n_samples, n_features)
        The input data. Not seen during training.
    y : np.ndarray (n_samples,)
        The ground truth labels.

    Returns
    -------
    transition_matrix : np.ndarray (n_classes, n_classes)
        The estimated transition_matrix.
    """
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    transition_matrix = np.zeros((n_classes, n_classes))

    # Get the predicted probabilities
    probas = model.predict_proba(X)
    for label in unique_labels:
        # Get the average predicted probabilities for each class
        transition_matrix[label] = np.mean(probas[y == label], axis=0)

    # Normalize the transition matrix
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    return transition_matrix


def validate_label_transitions(
        name: str,
        y: np.ndarray,
        y_noisy: np.ndarray,
        verbose: bool = False,
        hard_mode: bool = False,
    ) -> None:
    """
    Validate the transition matrix by comparing the true labels with the noisy 
    labels.
    
    Throws an AssertionError if the transition matrix is not valid.

    We want each class to still be primarily represented by its label and 
    composed of its own class.

    This checks of the overall transition frequencies allows for more control 
    than checking the transition matrix directly.
    This is because here we can detect problems that depend on the relative 
    frequencies of the classes.

    Parameters
    ----------
    y : np.ndarray
        The true labels.
    y_noisy : np.ndarray
        The noisy labels.
    verbose : bool
        Whether to print the transition matrix.
    hard_mode : bool
        Whether to raise an error or a warning if the transition matrix is 
        invalid.
    """
    n_classes = len(np.unique(y))
    transition_matrix = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            transition_matrix[i, j] = np.mean((y == i) & (y_noisy == j))
    
    if verbose:
        per_class_transitions = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
        print(f"Transitions:\n{per_class_transitions.round(4)}")

    # Test that the diagonal entries are higher than the sum off-diagonal 
    # entries for each row and column
    diagonal = transition_matrix.diagonal()

    if hard_mode: # Raise Errors
        assert (2*diagonal > transition_matrix.sum(axis=1)).all(), (
            'There are classes that are not primarily'
            'represented by their own label.'
        )
        assert (2*diagonal > transition_matrix.sum(axis=0)).all(), (
            'There are classes (labels) that are not primarily' 
            'composed of their own class.'
        )
    else: # Warn
        if not (2*diagonal > transition_matrix.sum(axis=1)).all():
            warnings.warn(
                f"There are classes that are not primarily "
                f"represented by their own label in {name}."
            )
        if not (2*diagonal > transition_matrix.sum(axis=0)).all():
            warnings.warn(
                f"There are classes that are not primarily "
                f"represented by their own label in {name}."
            )


def propose_alternative_label(
        sample_scores: np.ndarray, 
        y_noisy: np.ndarray,
    ) -> np.ndarray:
    """
    Propose alternative labels for the given samples with these noisy labels.
    
    Parameters
    ----------
    sample_scores: np.ndarray (n_samples, n_classes)
        The scores for each class for each sample.
    y_noisy : np.ndarray (n_samples,)
        The noisy labels.

    Returns
    -------
    proposed_labels : np.ndarray (n_samples,)
        The proposed labels.
    """
    idxs = np.argsort(sample_scores)

    mask = idxs[:, -1] == y_noisy

    proposed_labels = idxs[:, -1]
    proposed_labels[mask] = idxs[mask, -2].copy()
    return proposed_labels
