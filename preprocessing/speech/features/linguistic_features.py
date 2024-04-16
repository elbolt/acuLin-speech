import numpy as np
import pandas as pd
from tqdm import tqdm


def create(words_dir: str, phones_dir: str, out_dir: str, config: dict) -> None:
    """
    Create feature from linguistic time stamps and features.

    This function reads in the .txt files containing the linguistic time stamps and features and creates a feature.
    The feature is of the shape (samples, segments, features) and is saved as a .npy file.

    Acoustic feature shape: (samples, segments, [word onsets, phone onsets])
    Words feature shape: (samples, segments, [word surprisal, word frequency])
    Phones feature shape: (samples, segments, [phone surprisal, phone entropy])

    Parameters
    ----------
    words_dir : str
        Directory path to the words data.
    phones_dir : str
        Directory path to the phones data.
    out_dir : str
        Directory path to save the feature.
    config : dict
        Configuration dictionary containing the following keys:
        - elements: List of elements to extract.
        - vector_duration: Duration of the feature vector in seconds.
        - audio_snips: List of audio snips.
        - sfreq_target: Target sampling frequency.

    """
    elemenets = config['elements']
    vector_duration = config['vector_duration']
    audio_snips = config['audio_snips']
    sfreq_target = config['sfreq_target']

    n_snips = len(audio_snips)
    n_features = len(elemenets)

    puffer_length = 5
    full_length = (vector_duration + puffer_length) * sfreq_target
    final_length = (vector_duration) * sfreq_target

    segmentation = np.full((full_length, n_snips, n_features), np.nan)
    words = np.full((full_length, n_snips, n_features), np.nan)
    phones = np.full((full_length, n_snips, n_features), np.nan)

    silent_samples = np.zeros(n_snips)

    for e_idx, element in enumerate(elemenets):
        last_feature = 'frequency' if element == 'words' else 'entropy'
        dir_ = words_dir if element == 'words' else phones_dir
        snips = sorted(dir_.glob('*.txt'))

        onset_array = np.zeros((full_length, n_snips))
        surprisal_array = onset_array.copy()
        third_array = onset_array.copy()

        for idx, snip in enumerate(tqdm(snips, desc=element)):
            snip_df = pd.read_csv(snip, sep='\t')
            first_impulse = snip_df['onset'].iloc[0]
            cut = np.round(first_impulse * sfreq_target).astype(int)

            silent_samples[idx] = cut

            onsets = np.round(snip_df['onset'] * sfreq_target).astype(int)
            surprisal = snip_df['surprisal'].to_numpy()
            frequency = snip_df[last_feature].to_numpy()

            onsets_vector = np.zeros(full_length)
            onsets_vector[onsets] = 1

            surprisal_vector = np.zeros(full_length)
            surprisal_vector[onsets] = surprisal

            frequency_vector = np.zeros(full_length)
            frequency_vector[onsets] = frequency

            onsets_vector = onsets_vector[cut:]
            surprisal_vector = surprisal_vector[cut:]
            frequency_vector = frequency_vector[cut:]

            onset_array[:, idx] = np.pad(onsets_vector, (0, full_length - len(onsets_vector)), 'constant')
            surprisal_array[:, idx] = np.pad(surprisal_vector, (0, full_length - len(surprisal_vector)), 'constant')
            third_array[:, idx] = np.pad(frequency_vector, (0, full_length - len(frequency_vector)), 'constant')

        # Replace all NaNs with 0 in the three arrays
        onset_array = np.nan_to_num(onset_array)
        surprisal_array = np.nan_to_num(surprisal_array)
        third_array = np.nan_to_num(third_array)

        segmentation[..., e_idx] = onset_array

        if element == 'words':
            words[..., 0] = surprisal_array
            words[..., 1] = third_array
        elif element == 'phones':
            phones[..., 0] = surprisal_array
            phones[..., 1] = third_array

    # Reshape to match EEG-MNE shape (n_epochs, n_channels, n_times)
    segmentation = segmentation.transpose(1, 2, 0)
    words = words.transpose(1, 2, 0)
    phones = phones.transpose(1, 2, 0)

    # Cut to final length
    segmentation = segmentation[:, :, :final_length]
    words = words[:, :, :final_length]
    phones = phones[:, :, :final_length]

    print(f'Segmentation shape: {segmentation.shape}')
    print(f'Words shape: {words.shape}')
    print(f'Phones shape: {phones.shape}')

    np.save(out_dir / 'silent_samples.npy', silent_samples)
    np.save(out_dir / 'segmentation.npy', segmentation)
    np.save(out_dir / 'words.npy', words)
    np.save(out_dir / 'phones.npy', phones)
