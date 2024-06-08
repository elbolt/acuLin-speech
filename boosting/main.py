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
    """
    Run the boosting pipeline for a given subject and models.

    Parameters
    ----------
    subject_id : str
        The ID of the subject.
    eeg_dir : Path
        Path to the directory containing EEG data.
    feature_dir : Path
        Path to the directory containing feature data.
    trf_params : dict
        Dictionary containing the parameters for the TRF model:
        - model_attributes : dict
        - sfreq : int
        - tmin : float
        - tmax : float
        - partition : int

    Returns
    -------
    stacked_scores : np.ndarray
        Stacked boosting scores for all models.
    stacked_kernels : np.ndarray
        Stacked boosting kernels for all models.
    time_vector : np.ndarray
        Time vector of the estimated kernels.
    """
    # Extract TRF parameters
    model_attributes = trf_params['model_attributes']
    model_list = model_attributes.keys()
    sfreq = trf_params['sfreq']
    tmin = trf_params['tmin']
    tmax = trf_params['tmax']
    partition = trf_params['partition']

    all_scores = []
    all_kernels = []

    for model in tqdm(model_list, desc=f'Fit {subject_id} models'):
        loader = BoostingDataLoader(
            subject_id=subject_id,
            eeg_dir=eeg_dir,
            feature_dir=feature_dir,
            model_attributes=model_attributes,
            sfreq=sfreq
        )
        feature, eeg_residuals = loader.get_data(model=model, trim_beginnings=True)

        booster = Booster(
            feature=feature,
            eeg=eeg_residuals,
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            partition=partition
        )

        kernels, scores, time_vector = booster.get_results()

        all_scores.append(scores)
        all_kernels.append(kernels)

    stacked_scores = np.array(all_scores)
    stacked_kernels = np.vstack(all_kernels)

    return stacked_scores, stacked_kernels, time_vector


def main():
    """
    Main function to run the boosting pipeline for multiple subjects.
    """
    with open('config.json', 'r') as file:
        config = json.load(file)

    # Create TRF parameters dictionary
    trf_params = {
        "model_attributes": config['model_attributes'],
        "sfreq": config['sfreq'],
        "tmin": config['tmin'],
        "tmax": config['tmax'],
        "partition": config['partition'],
        "n_responses": config['n_responses']
    }

    # Path configuration
    eeg_dir = Path(config['eeg_dir'])
    feature_dir = Path(config['feature_dir'])
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'kernels').mkdir(exist_ok=True)
    (out_dir / 'scores').mkdir(exist_ok=True)

    # Participant parsing
    default_subjects = config['default_subjects']
    subjects = parse_arguments(default_subjects)

    for subject_id in subjects:
        scores, kernels, times = run_boosting_pipeline(
            subject_id=subject_id,
            eeg_dir=eeg_dir,
            feature_dir=feature_dir,
            trf_params=trf_params,
        )
        # Save results
        np.save(out_dir / 'scores' / f'{subject_id}.npy', scores)
        np.save(out_dir / 'kernels' / f'{subject_id}.npy', kernels)

    # Save time vector only once
    np.save(out_dir / 'time_vector.npy', times)


if __name__ == '__main__':
    main()
