import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from boosting_utils import BoostingDataLoader, Booster
from utils import parse_arguments


def run_boosting_pipeline(
    subject_id: str,
    eeg_dir: Path,
    feature_dir: Path,
    trf_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Run boosting pipeline for a single participant.

    Parameters
    ----------
    subject_id : str
        Subject ID.
    eeg_dir : pathlib.Path
        Path to EEG data.
    feature_dir : pathlib.Path
        Path to features.
    trf_params : dict
        dictionary containing the following parameters:
            model_list : list
                List of models to fit.
            sfreq : float
                Sampling frequency.
            tmin : float
                Start time of the epoch.
            tmax : float
                End time of the epoch.
            partition : int
                Number of partitions for cross-validation.

    Returns
    -------
    all_kernels : np.ndarray
        4D array of kernels (n_model, n_features, n_channels, n_times)
    all_scores : np.ndarray
        2D array of scores (n_model, n_times)
    times : np.ndarray
        1D array of time points (n_times)

    """

    model_list = trf_params['model_list']
    sfreq = trf_params['sfreq']
    tmin = trf_params['tmin']
    tmax = trf_params['tmax']
    partition = trf_params['partition']

    all_kernels = []
    all_scores = []

    for model in tqdm(model_list, desc=f'Fit {subject_id} models'):
        loader = BoostingDataLoader(subject_id, eeg_dir, feature_dir)
        feature, eeg_residuals = loader.get_data(model=model, trim_beginnings=True)

        booster = Booster(
            feature,
            eeg_residuals,
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            partition=partition,
        )

        kernels, scores, times = booster.get_results()

        all_kernels.append(np.stack(kernels, axis=0))
        all_scores.append(scores)

    all_kernels = np.array(all_kernels)
    all_scores = np.array(all_scores)

    return all_kernels, all_scores, times


if __name__ == '__main__':
    print(f'Running {__file__} ...')

    with open('config.json', 'r') as file:
        config = json.load(file)

    eeg_dir = Path(config['eeg_dir'])
    feature_dir = Path(config['feature_dir'])
    out_dir = Path(config['out_dir'])
    default_subjects = config['default_subjects']

    trf_dict = {
        "model_list": config['model_list'],
        "sfreq": config['sfreq'],
        "tmin": config['tmin'],
        "tmax": config['tmax'],
        "partition": config['partition']
    }

    out_dir.mkdir(exist_ok=True)
    (out_dir / 'kernels').mkdir(exist_ok=True)
    (out_dir / 'scores').mkdir(exist_ok=True)

    subjects = parse_arguments(default_subjects)

    for subject_id in subjects:
        all_kernels, all_scores, times = run_boosting_pipeline(subject_id, eeg_dir, feature_dir, trf_dict)

        np.save(out_dir / 'kernels' / f'{subject_id}.npy', all_kernels)
        np.save(out_dir / 'scores' / f'{subject_id}.npy', all_scores)

    np.save(out_dir / 'times.npy', times)
