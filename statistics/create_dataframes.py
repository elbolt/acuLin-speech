import json
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from helpers import get_relevant_data, get_largest_peaks


def create_scores_data(
        score_files: list[Path],
        models_dict: dict,
) -> pd.DataFrame:
    """ Create a DataFrame with scores data from the score files.

    Parameters
    ----------
    score_files : list[Path]
        List of score files.
    models_dict : dict
        Dictionary with the models and their corresponding indices.

    Returns
    -------
    pd.DataFrame
        DataFrame with scores data with structure:
        subject_id | model | score

    """
    all_subjects, all_models, all_scores = [], [], []

    for score_file in score_files:
        subject_id = score_file.stem
        score_data = np.load(score_file)

        for model, model_id in models_dict.items():
            mean_score = np.mean(score_data[model_id])  # average across channels
            all_subjects.append(subject_id)
            all_models.append(model)
            all_scores.append(mean_score)

    df = pd.DataFrame({
        'subject_id': all_subjects,
        'model': all_models,
        'score_r': all_scores
    })

    return df


def create_RMS_data(
        kernel_files: list[Path],
        responses_dict: dict,
        time_vector: np.ndarray,
        tmin: float,
        tmax: float,
        relevant_channels: list[str],
        channel_indices: list[int]
) -> pd.DataFrame:
    """ Create a DataFrame with RMS data from the kernel files.

    Parameters
    ----------
    kernel_files : list[Path]
        List of kernel files.
    score_files : list[Path]
        List of score files.
    responses_dict : dict
        Dictionary with the responses and their corresponding indices.
    time_vector : np.ndarray
        Time vector.
    tmin : float
        Minimum time.
    tmax : float
        Maximum time.
    relevant_channels : list[str]
        List of relevant channels.
    channel_indices : list[int]
        List of channel indices.

    Returns
    -------
    pd.DataFrame
        DataFrame with RMS data of the kernel files with structure:
        subject_id | response | electrode_id | RMS

    """
    combined_df = pd.DataFrame()

    for kernel_file in kernel_files:
        subject_id = kernel_file.stem
        kernel_data = np.load(kernel_file)

        relevant_kernel_data, _ = get_relevant_data(kernel_data, time_vector, tmin, tmax, channel_indices)
        RMS_data = np.sqrt(np.mean(relevant_kernel_data ** 2, axis=2))

        # REINTRODUCE THE CHANNELS

        subject_df = pd.DataFrame(RMS_data, index=list(responses_dict.keys()), columns=channel_indices)
        subject_df = subject_df.reset_index().melt(id_vars='index', var_name='electrode_id', value_name='RMS')
        subject_df.rename(columns={'index': 'response'}, inplace=True)
        subject_df['subject_id'] = subject_id
        subject_df = subject_df[['subject_id', 'response', 'electrode_id', 'RMS']]

        combined_df = pd.concat([combined_df, subject_df], ignore_index=True)

    return combined_df


def create_peak_data(
        kernel_files: list[Path],
        responses_dict: dict,
        time_vector: np.ndarray,
        tmin: float,
        tmax: float,
        peak_prominence: float,
        relevant_channels: list[str],
        channel_indices: list[int]
) -> pd.DataFrame:
    """ Create a DataFrame with peak data from the kernel files.

    Parameters
    ----------
    kernel_files : list[Path]
        List of kernel files.
    responses_dict : dict
        Dictionary with the responses and their corresponding indices.
    time_vector : np.ndarray
        Time vector.
    tmin : float
        Minimum time.
    tmax : float
        Maximum time.
    peak_prominence : float
        Peak prominence.
    relevant_channels : list[str]
        List of relevant channels.
    channel_indices : list[int]
        List of channel indices.


    Returns
    -------
    pd.DataFrame
        DataFrame with peak data with structure:
        subject_id | response | channel | latency_ms | amplitude

    """
    all_subjects, all_responses, all_latencies, all_channels, all_amplitudes = [], [], [], [], []

    for kernel_file in kernel_files:
        subject_id = kernel_file.stem
        kernel_data = np.load(kernel_file)

        relevant_kernel_data, relevant_time_lags = get_relevant_data(
            kernel_data,
            time_vector,
            tmin,
            tmax,
            channel_indices
        )

        n_responses, n_channels = relevant_kernel_data.shape[:2]

        for resp in range(n_responses):
            for ch in range(n_channels):
                response = relevant_kernel_data[resp, ch, ...]
                response_id = list(responses_dict.keys())[list(responses_dict.values()).index(resp)]

                latencies, amplitudes = get_largest_peaks(response, relevant_time_lags, prominence=peak_prominence)

                all_responses.extend([response_id] * len(latencies))
                all_subjects.extend([subject_id] * len(latencies))
                all_latencies.extend(latencies * 1e3)  # from seconds to milliseconds
                all_channels.extend([relevant_channels[ch]] * len(latencies))
                all_amplitudes.extend(amplitudes)

    df = pd.DataFrame({
        'subject_id': all_subjects,
        'response': all_responses,
        'channel': all_channels,
        'latency_ms': all_latencies,
        'amplitude': all_amplitudes
    })

    return df


if __name__ == "__main__":
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Configuration parameters
    kernels_dir = Path(config['kernels_dir'])
    scores_dir = Path(config['scores_dir'])
    dataframes_dir = Path(config['dataframes_dir'])
    dataframes_dir.mkdir(parents=True, exist_ok=True)
    subject_data_filename = config['subject_data_filename']
    scores_filename = config['scores_filename']
    rms_filename = config['rms_filename']
    peak_filename = config['peak_filename']
    models_dict = config['models_dict']
    responses_dict = config['responses_dict']
    time_vector = np.load(config['time_vector'])
    tmin, tmax = config['tmin'], config['tmax']
    peak_prominence = config['peak_prominence']

    # Get the relevant channels and their indices
    relevant_channels = config['relevant_channels']
    all_channels = mne.channels.make_standard_montage('biosemi32').ch_names
    channel_indices = [all_channels.index(channel) for channel in relevant_channels]

    # Get the list of kernel files, ignoring hidden files
    kernel_files = sorted(kernels_dir.glob('*.npy'))
    kernel_files = [file for file in kernel_files if not file.name.startswith('._')]

    # Get the list of score files, ignoring hidden files
    score_files = sorted(scores_dir.glob('*.npy'))
    score_files = [file for file in score_files if not file.name.startswith('._')]

    subject_df = pd.read_csv(dataframes_dir / subject_data_filename)

    scores_df = create_scores_data(score_files, models_dict)

    RMS_df = create_RMS_data(
        kernel_files,
        responses_dict,
        time_vector,
        tmin,
        tmax,
        relevant_channels,
        channel_indices
    )

    peak_df = create_peak_data(
        kernel_files,
        responses_dict,
        time_vector,
        tmin,
        tmax,
        peak_prominence,
        relevant_channels,
        channel_indices
    )

    scores_df_combined = pd.merge(scores_df, subject_df, on='subject_id')
    RMS_df_combined = pd.merge(RMS_df, subject_df, on='subject_id')
    peak_df_combined = pd.merge(peak_df, subject_df, on='subject_id')

    scores_df_combined.to_csv(dataframes_dir / scores_filename, index=False)
    RMS_df_combined.to_csv(dataframes_dir / rms_filename, index=False)
    peak_df_combined.to_csv(dataframes_dir / peak_filename, index=False)
