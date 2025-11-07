import os
import argparse
import pandas as pd

from tqdm import tqdm
from copy import deepcopy
from multiprocessing import Pool, cpu_count

from src.data import load_dataset, SLOW_DATASETS
from src.experiments import label_error_detection_trial
from src.utils import create_folder, StoreStringList, StoreFloatList


def _main_function_label_error_exp(
        datasets: list[str],
        noise_levels: list[float],
        noise_types: list[str],
        n_trials: int,
        n_jobs: int,
        device: str,
        save_dir: str,
        seed: int
    ) -> pd.DataFrame:
        
    # Download the datasets now to avoid concurrency issues
    for ds_name in datasets:
        path = save_dir + '/' + ds_name + '.csv'
        if not os.path.exists(path) and not ds_name in ["spirals", "digits"]:
            load_dataset(ds_name, save_dir=save_dir, verbose=True)

    # Initialize the pool, and set the number of threads
    n_experiments = len(datasets)*len(noise_levels)*len(noise_types)*n_trials
    n_jobs = min(8, cpu_count()) if n_jobs == -1 else min(n_jobs, n_experiments)
    trial_pool = Pool(n_jobs)
    print(f"Running {n_jobs} jobs in parallel")

    # Trial and model parameter
    base_trial_params = dict(n_folds=5)
    base_model_config = dict(device=device, nthread=cpu_count() // n_jobs)

    # Run the trials
    trial_info = []
    pbar = tqdm(total=n_experiments, desc="Trials")
    for noise_level in noise_levels:
        for noise_type in noise_types:
            for random_state in range(seed, seed+n_trials):
                for ds_name in datasets:
                    _model_config = deepcopy(base_model_config)
                    _model_config['random_state'] = random_state

                    # Force CPU for fast datasets
                    if ds_name not in SLOW_DATASETS:
                        _model_config['device'] = 'cpu'    

                    _trial_params = deepcopy(base_trial_params)  
                    _trial_params['noise_type'] = noise_type
                    _trial_params['noise_level'] = noise_level
                    _trial_params['ds_name'] = ds_name
                    _trial_params['random_state'] = random_state
                    _trial_params['model_config'] = _model_config
                    _trial_params['save_dir'] = save_dir
                    trial_result = trial_pool.apply_async(
                        label_error_detection_trial, 
                        kwds=_trial_params,
                        callback=lambda _: pbar.update(1)
                    )
                    trial_info.append(trial_result)
    
    # Collect the results
    trial_pool.close()
    trial_results = [trial_result.get() for trial_result in trial_info]
    trial_pool.terminate()
    pbar.close()

    return pd.concat(trial_results, axis=0)


if __name__ == '__main__':
    STUDY_DATASETS = [
        "digits",
        "spirals",
        "letters",
        "mushrooms",
        "satelite",
        "sensorless_drive",
        "cardiotocography",
        "credit_card_fraud",
        "human_activity_recognition",
    ]
    
    # Parse the arguments
    parser = argparse.ArgumentParser("Experiments on label error detectors")
    parser.add_argument(
        '-ds', '--datasets', default=STUDY_DATASETS, type=str, 
        action=StoreStringList, help="Datasets to use"
    )
    parser.add_argument(
        '-n', '--n_trials', default=10, type=int, 
        help="Number of trials to run per condition"
    )
    parser.add_argument(
        '-j', '--n_jobs', default=-1, type=int, 
        help="Number of jobs to run in parallel"
    )
    parser.add_argument(
        '-sd', '--save_dir', default="datasets", type=str,         
        help="Directory to save the datasets to"
    )
    parser.add_argument(
        '-sr', '--save_results', default="results", type=str,         
        help="Directory to save the results"
    )
    parser.add_argument(
        '-d', '--device', default="cuda", type=str, help="Device to use"
    )
    parser.add_argument(
        '-nl', '--noise_levels', default=[0.05, 0.1, 0.2], type=str,
        action=StoreFloatList, help="Noise levels to use"
    )
    parser.add_argument(
        '-nt', '--noise_types', default=['uniform', 'asymmetric'], type=str, 
        action=StoreStringList, help="Noise types to use"
    )
    parser.add_argument(
        '-s', '--seed', default=42, type=int, help="Random state"
    )
    args = parser.parse_args()

    # Run the main function for the label error detection experiments
    results = _main_function_label_error_exp(
        datasets=args.datasets,
        noise_levels=args.noise_levels,
        noise_types=args.noise_types,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        device=args.device,
        save_dir=args.save_dir,
        seed=args.seed
    )

    # Save the results
    create_folder(args.save_results)
    path = os.path.join(args.save_results, 'label_error_trials.csv')
    results.to_csv(path)

    print(f"Label error detection trials results saved at {path}")