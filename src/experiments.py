import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from src.data import load_dataset
from src.label_noise import (
    generate_noisy_labels, 
    create_uniform_transition_matrix,
    create_asymmetric_transition_matrix,
    validate_label_transitions,
)

from src.scoring import (
    compute_AUM_scores,
    cleanlab_label_error_detection,
    cross_val_label_error_detection,
    calc_val_scores,
)
from src.validation import (
    cross_validate_sublearners,
)
from src.models import (
    get_model_default_kwargs_for_ds,
)


def shared_experiment_logic(
        ds_name: str, 
        ds_kwargs: dict, 
        noise_type: str, 
        noise_level: float,
        model_config: dict={},
        n_folds: int=5,
        random_state: int=42,
        save_dir: str=None,
        verbose: bool=False, 
    ) -> tuple:
    """
    Shared logic for the experiments.

    Parameters
    ----------
    ds_name : str
        The name of the dataset to load.
    ds_kwargs : dict
        Keyword arguments to pass to the dataset loading function.
    noise_type : str
        The type of noise to apply to the dataset.
    noise_level : float
        The level of noise to apply to the dataset.
    model_config : dict
        The configuration for the model.
    n_folds : int
        The number of cross-validation folds to use.
    random_state : int
        The random state to use for the experiment.
    save_dir : str
        The directory to save/load the dataset to/from.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - X : The dataset features.
        - y : The dataset labels.
        - model : The trained model.
        - y_noisy : The noisy labels.
        - noise_mask : The noise mask.
        - accuracies : The cross-validation accuracies.
        - aum_values : The AUM values.
        - values_naive : The naive values.
        - values_pos : The positive values.
        - values_entropy : The entropy values.
        - oos_probas : The out-of-sample probabilities.
    """
    assert noise_type in ('uniform', 'asymmetric')

    # COMMENT: Unfortunately, we did not use a random state for the spirals
    # dataset generation during the experiments reported in the paper. Using the
    # same random state for the dataset generation and the noise generation
    # leads to a bug, where the results are systematically skewed. Using a 
    # different seed for the dataset generation and the noise generation fixes
    # this issue. Optimally, we would use a random generator and pass it to the
    # dataset generation and the noise generation functions. But since the 
    # results of the paper were generated without this, the reproducibility of
    # the results would not be possible. Therefore, we fix this issue by adding
    # 1 to the random state for the dataset generation.
    
    # Apply the random state to the dataset kwargs
    ds_kwargs['seed'] = random_state + 1
    
    # Load the dataset
    X, y = load_dataset(ds_name, save_dir, **ds_kwargs)
    n_classes = len(np.unique(y))

    # Initialize the transition matrix
    if noise_type == 'uniform':
        transition_matrix = create_uniform_transition_matrix(
            n_classes, noise_level
        )
    elif noise_type == 'asymmetric':
        transition_matrix = create_asymmetric_transition_matrix(
            n_classes, noise_level, random_state=random_state
        )

    # Generate noisy labels
    y_noisy, noise_mask = generate_noisy_labels(
        y, transition_matrix, random_state
    )

    # Validate the label noise transitions
    validate_label_transitions(ds_name, y, y_noisy)

    # Update the model configuration with the default values
    model_config.update(get_model_default_kwargs_for_ds(ds_name))    

    # Disable XGBoost warnings if not verbose
    if not verbose:
        model_config['verbosity'] = 0
        model_config['silent'] = True
    
    # Fit the model on the noisy labels
    model = XGBClassifier(**model_config)
    model.fit(X, y_noisy)

    # Compute the AUM label quality scores
    # Must be called with the model fitted on the entire noisy dataset
    aum_values, values_naive, values_pos, values_entropy  = compute_AUM_scores(
        model, X, y_noisy, additional_scores=True
    )

    # Compute the cross-validation accuracy of the model for each range of 
    # estimators and obtain the out-of-sample predictions on the data
    accuracies, oos_probas = cross_validate_sublearners(
        model, X, y, 
        transition_matrix=None, 
        n_splits=n_folds, 
        y_noisy=y_noisy, 
        random_state=random_state, 
    )

    return (
        X, y, model,
        y_noisy, noise_mask, 
        accuracies, aum_values, 
        values_naive, values_pos, 
        values_entropy, oos_probas, 
    )


def label_error_detection_trial(
        ds_name: str,
        ds_kwargs: dict= {},
        noise_type: str= 'uniform',
        noise_level: float= 0.1,
        random_state: int= 42,
        model_config: dict= {},
        n_folds: int= 5,
        save_dir: str= None,
        verbose: bool= False,
    ) -> pd.DataFrame:
    """
    This experiment evaluates the performance of different label error detection
    methods on a dataset. The methods evaluated are:
    - AUM: Average Uncertainty Margin
    - Naive: The naive label error detection method
    - Positive: The positive label error detection method
    - Entropy: The entropy label error detection method
    - Cross-Validation: The cross-validation label error detection method
    - Normalized Margins: The normalized margins label error detection method
    - Cleanlab: The cleanlab label error detection method

    """   
    (
        X, y, model,
        y_noisy, noise_mask, 
        accuracies, aum_values, 
        values_naive, values_pos, 
        values_entropy, oos_probas 
    ) = shared_experiment_logic(
        ds_name=ds_name, ds_kwargs=ds_kwargs,
        noise_type=noise_type, noise_level=noise_level,
        model_config=model_config, n_folds=n_folds,
        random_state=random_state, save_dir=save_dir, 
        verbose=verbose
    )

    # Compute the cross-validation based label quality scores
    cross_val_scores, normalized_margins = cross_val_label_error_detection(
        y_noisy, oos_probas
    )

    # Compute label quality scores using cleanlab
    cleanlab_scores = cleanlab_label_error_detection(
        y_noisy, oos_probas[:, -1:],
        # these two parameter values are generally the most performant
        filter_by='prune_by_noise_rate',
        indices_ranked_by='normalized_margin',
    )

    # Scores dictionary
    scores = {
        'aum': aum_values.mean(axis=1), 
        'naive': values_naive.mean(axis=1), 
        'positive': values_pos.mean(axis=1), 
        'entropy': -values_entropy.mean(axis=1), 
        'cross_val': cross_val_scores[:, -1], 
        'normalized_margins': normalized_margins[:, -1], 
        'cleanlab': cleanlab_scores[:, -1]
    }

    # Results DataFrame with multi-level index
    methods = list(scores.keys())
    index = pd.MultiIndex.from_product(
        [[ds_name], methods, [noise_type], [noise_level], [random_state]],
        names=['ds_name', 'method', 'noise_type', 'noise_level', 'random_state']
    )
    columns = [
        'accuracy_final', 'accuracy_max', 'auroc', 'fpr', 'fpr_50', 'fpr_10'
    ] 
    results = pd.DataFrame(index=index, columns=columns).copy(deep=True)

    # Select true positive threshold for false positive rates FPR@TPR
    tpr_thresholds = [0.95, 0.5, 0.1]

    # Iterate over the methods
    for method, score in scores.items():
        # Compute the validation scores
        auroc, (fpr, fpr_50, fpr_10) = calc_val_scores(
            score, noise_mask, tpr_thresholds
        )        

        # Write the results
        idx = (ds_name, method, noise_type, noise_level, random_state)
        results.loc[idx, 'accuracy_final'] = accuracies[-1]
        results.loc[idx, 'accuracy_max'] = np.max(accuracies)
        results.loc[idx, 'auroc'] = auroc
        results.loc[idx, 'fpr'] = fpr
        results.loc[idx, 'fpr_50'] = fpr_50
        results.loc[idx, 'fpr_10'] = fpr_10

    # Free memory
    del X, y, model, y_noisy, noise_mask, accuracies, aum_values, values_naive
    del values_pos, values_entropy, oos_probas, cross_val_scores
    del normalized_margins, cleanlab_scores

    return results


def performance_improvement_trial(
        ds_name: str, 
        percentiles: list[float],
        ds_kwargs: dict={},
        noise_type: str= 'uniform',
        noise_level: float= 0.05,
        random_state: int= 42,
        model_config: dict= {},
        n_folds: int= 5,
        save_dir: str= None,
        verbose: bool= False,
    ) -> pd.DataFrame:
    """
    This experiment evaluates the performance improvement of the model
    when the most uncertain samples are removed from the dataset.
    The cutoff is determined by the percentile of the label quality score.  
    The methods evaluated are:
    - AUM: Average Uncertainty Margin
    - Cross-Validation: The cross-validation label error detection method
    - Random: Randomly remove samples from the dataset  
    """
    (
        X, y, model,
        y_noisy, noise_mask, 
        accuracies, aum_values, 
        values_naive, values_pos, 
        values_entropy, oos_probas 
    ) = shared_experiment_logic(
        ds_name=ds_name, ds_kwargs=ds_kwargs,
        noise_type=noise_type, noise_level=noise_level,
        model_config=model_config, n_folds=n_folds,
        random_state=random_state, save_dir=save_dir, 
        verbose=verbose
    )

    # Compute the cross-validation based label quality scores
    cross_val_scores, _ = cross_val_label_error_detection(y_noisy, oos_probas)

    # Methods to evaluate
    methods = {
        'aum': lambda: aum_values.mean(axis=1),
        'cross_val': lambda: cross_val_scores[:, -1],
        'random': lambda: np.random.rand(len(y)),
    }

    # Results DataFrame with multi-level index
    index = pd.MultiIndex.from_product(
        [
            [ds_name], methods.keys(), [noise_type], [noise_level], 
            [random_state], percentiles
        ],
        names=[
            'ds_name', 'method', 
            'noise_type', 'noise_level', 
            'random_state', 'percentile'
        ]
    )
    results = pd.DataFrame(index=index, columns=["accuracy"])

    # Iterate over the methods
    for method, score_func in methods.items():
        scores = score_func()

        # Iterate over the percentiles
        for percentile in percentiles:
            accuracy = accuracies[-1]
            if not percentile == 0:
                # Mark the samples assumed to be labeled incorrectly
                exclusion_threshold = np.percentile(scores, percentile*100)
                excluded_sample_mask = scores < exclusion_threshold
                
                # Exclude the samples from the dataset
                X_included = X[~excluded_sample_mask]
                y_noisy_included = y_noisy[~excluded_sample_mask]

                y_pred = np.zeros(X.shape[0])
                y_pred_included = y_pred[~excluded_sample_mask]

                # Split the excluded samples into n_folds
                mislabel_test_indices = np.array_split(
                    np.random.permutation(np.where(excluded_sample_mask)[0]), 
                    n_folds
                )

                # Cross-validate the model
                skf = StratifiedKFold(
                    n_splits=n_folds, shuffle=True, random_state=42
                )
                for (train_index, test_index), mislabel_test_index in zip(
                        skf.split(X_included, y_noisy_included),
                        mislabel_test_indices
                    ):
                    # Split the data
                    X_train = X_included[train_index]
                    X_test = X_included[test_index]
                    y_train_noisy = y_noisy_included[train_index]

                    # Create custom class indexing to appease XGBoost
                    cls_idx_mapping = {
                        cls: i for i, cls in enumerate(np.unique(y_train_noisy))
                    }
                    inv_cls_idx_mapping = {
                        v: k for k, v in cls_idx_mapping.items()
                    }
                    forward_mapping = np.vectorize(cls_idx_mapping.get)
                    backward_mapping = np.vectorize(inv_cls_idx_mapping.get)

                    # Fit the model
                    model.fit(X_train, forward_mapping(y_train_noisy))

                    # Predict the labels
                    y_pred_included[test_index] = backward_mapping(
                        model.predict(X_test)
                    )
                    if mislabel_test_index.any():
                        y_pred[mislabel_test_index] = backward_mapping(
                            model.predict(X[mislabel_test_index])
                        )
                y_pred[~excluded_sample_mask] = y_pred_included

                # Compute accuracy
                accuracy = np.mean(y_pred == y)
        
            # Write the results
            index = (
                ds_name, method, noise_type, noise_level, 
                random_state, percentile
            )
            results.loc[index, 'accuracy'] = accuracy

    # Free memory
    del X, y, model, y_noisy, noise_mask, accuracies, aum_values
    del values_naive, values_pos, values_entropy, oos_probas
    del cross_val_scores

    return results