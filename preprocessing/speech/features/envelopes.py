import numpy as np
import mne
from audio_utils import WaveProcessor
from tqdm import tqdm


def create(wav_dir: str, out_dir: str, config: dict) -> None:
    """ Create feature from audio snips.

    This function creates a feature from audio snips. The feature is an acoustic array of the shape
    (samples, segments, features) and is saved as a .npy file.
    The pipeline is as follows:
    1. Load audio snips.
    2. Downsample to 15 kHz.
    3. Extract Gammatone envelope.
    4. Downsample to 128 Hz.
    5. Filter between 0.5 and 25 Hz.
    6. Cut the silent samples to match the linguistic feature lengths.
    7. Pad the feature to the desired length.
    8. Normalize the feature globally and by absolute maximum (envelope) and maximum (onsets).

    Acoustic feature shape: (samples, segments, [envelope, envelope_onsets])

    Parameters
    ----------
    wav_dir : str
        Directory containing the audio snips.
    out_dir : str
        Directory to save the acoustic features.
    config : dict

    """

    vector_duration = config['vector_duration']
    sfreq_target = config['sfreq_target']
    vector_length = vector_duration * sfreq_target
    audio_snips = config['audio_snips']

    silent_samples = np.load(out_dir / 'silent_samples.npy')

    acoustic_elements = ['envelope', 'envelope_onsets']
    env_idx = acoustic_elements.index('envelope')
    ons_idx = acoustic_elements.index('envelope_onsets')

    acoustics = np.full((vector_length, len(audio_snips), len(acoustic_elements)), np.nan)

    for idx, snip_id in enumerate(tqdm(audio_snips, desc='envelopes')):
        processor = WaveProcessor(snip_id, wav_dir)
        processor.downsample(sfreq_goal=15000)
        processor.extract_Gammatone_envelope(num_filters=28, freq_range=(50, 5000), compression=0.6)
        processor.downsample(sfreq_goal=sfreq_target)

        sfreq_target, envelope = processor.get_wave()

        envelope = mne.filter.filter_data(
            envelope,
            sfreq=sfreq_target,
            l_freq=0.5,
            h_freq=25.0,
            method='iir',
            iir_params=dict(order=3, ftype='butter', output='sos')
        )

        onsets = np.maximum(0, np.diff(envelope, prepend=0))

        cut = int(silent_samples[idx])

        envelope = envelope[cut:]
        onsets = onsets[cut:]

        if len(envelope) < vector_length:
            envelope = WaveProcessor.padding(envelope, vector_duration, sfreq_target, pad_value=np.nan)
            onsets = WaveProcessor.padding(onsets, vector_duration, sfreq_target, pad_value=np.nan)
        else:
            envelope = envelope[:vector_length]
            onsets = onsets[:vector_length]

        acoustics[:, idx, env_idx] = envelope
        acoustics[:, idx, ons_idx] = onsets

    acoustics = np.nan_to_num(acoustics)

    # Reshape to match EEG-MNE shape (n_epochs, n_channels, n_times)
    # Now shape is (n_times, n_epochs, n_channels)
    acoustics = acoustics.transpose(1, 2, 0)

    np.save(out_dir / 'acoustics.npy', acoustics)
