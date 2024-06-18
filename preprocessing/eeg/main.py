import json
import numpy as np
from pathlib import Path
from eeg_utils import EEGLoader, EEGDownSegmenter
from utils import parse_arguments

import mne
mne.set_log_level('WARNING')


def run_eeg_preprocessing(
    raw_dir: Path,
    out_dir: Path,
    file_extension: str,
    default_subjects: list,
    IC_dict: dict,
    neuro_params: dict
) -> None:
    """ Runs EEG preprocessing pipeline on raw data and saves the result.

    Pipeline:
        - Load raw data (`EEGLoader` takes care of channel configuration)
        - Segment and downsample to 512 Hz (`EEGDownSegmenter` applies anti-alias filter)
        - Run ICA on high-pass filtered epochs copy
        - Remove bad components in original epochs
        - Interpolate bad channels
        - Anti-alias filter and downsample to 128 Hz
        - 1-25 Hz band-pass filter
        - Crop to 45 seconds
        - Global z-scoring
        - Save preprocessed data

    """
    tmin = neuro_params['tmin']
    tmax = neuro_params['tmax']
    final_length = neuro_params['final_length']
    sfreq_goal = neuro_params['sfreq_goal']

    subjects_list = parse_arguments(default_subjects)

    for _, subject_id in enumerate(subjects_list):
        print(f'Processing {subject_id}')

        eeg_loader = EEGLoader(subject_id, raw_dir, file_extension)
        raw = eeg_loader.get_raw()

        segmenter = EEGDownSegmenter(
            raw,
            subject_id,
            tmin=tmin,
            tmax=tmax,
            decimator=32
        )
        epochs = segmenter.get_epochs()

        epochs_ica_copy = epochs.copy()
        epochs_ica_copy.filter(
            l_freq=1.0,
            h_freq=None,
            method='fir',
            fir_window='hamming',
            phase='zero'
        )

        ica = mne.preprocessing.ICA(
            n_components=0.999,
            method='picard',
            max_iter=1000,
            fit_params=dict(fastica_it=5),
            random_state=1606
        )

        ica.fit(epochs_ica_copy)
        ica.exclude = IC_dict[subject_id]
        ica.apply(epochs)

        del epochs_ica_copy

        epochs.interpolate_bads(reset_bads=True)
    
        epochs.filter(
            l_freq=None,
            h_freq=sfreq_goal / 3.0,
            h_trans_bandwidth=sfreq_goal / 10.0,
            method='iir',
            iir_params=dict(order=3, ftype='butter', output='sos')
        )
        epochs.decimate(4)

        epochs.filter(
            l_freq=0.5,
            h_freq=25.0,
            method='iir',
            iir_params=dict(order=3, ftype='butter', output='sos')
        )

        epochs.crop(tmin=0, tmax=final_length)

        data = epochs.get_data(picks='eeg')
        data = data[..., 0:-1]  # Remove last sample to match speech features

        np.save(out_dir / f'{subject_id}.npy', data)


if __name__ == '__main__':
    print('Running: ', __file__)

    with open('config.json', 'r') as f:
        config = json.load(f)

    neuro_params = {
        'tmin': config['tmin'],
        'tmax': config['tmax'],
        'final_length': config['final_length'],
        'sfreq_goal': config['sfreq_goal']
    }

    default_subjects = config['default_subjects']

    raw_dir = Path(config['raw_dir'])
    file_extension = config['file_extension']

    IC_dict = config['IC_dict']

    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    run_eeg_preprocessing(
        raw_dir,
        out_dir,
        file_extension,
        default_subjects,
        IC_dict=IC_dict,
        neuro_params=neuro_params
    )
