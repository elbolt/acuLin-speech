import os
import numpy as np
import mne
from scipy.io import wavfile


class SpeechWaveExtractor():
    """ Class to load audio data.

    Methods
    -------
    extract_wave()
        Gets speech wave from audio segment.
    get_wave()
        Returns speech wave and sampling frequency.

    """
    def __init__(self, filename: str, directory: str) -> None:
        """ Initializes the SpeechWaveExtractor class.

        Parameters
        ----------
        filename : str
            Name of the file
        directory : str
            directory containing the file

        """
        self.filename = filename
        self.directory = directory

        self.extract_wave()

    def extract_wave(self) -> None:
        """ Gets speech wave from  audio segment. """
        filelocation = os.path.join(self.directory, f'{self.filename}.wav')
        self.sfreq, wave = wavfile.read(filelocation)
        self.wave = wave.astype(np.float64)

    def get_wave(self) -> tuple[int, np.ndarray]:
        """ Returns speech wave and sampling frequency. """
        return self.sfreq, self.wave


class WaveProcessor():
    """ Class to process speech wave arrays.

    This class is used to process speech wave arrays, such as downsampling, filtering, and extracting the Gammatone
    envelope. The class is initialized with a file name, directory, and TextGrid directory. The sampling frequency
    and speech wave signal are updated after each processing step.

    Methods
    -------
    downsample(sfreq_goal)
        Downsamples the speech wave to the desired sampling frequency.
    extract_Gammatone_envelope(num_filters, freq_range, compression)
        Extracts the Gammatone envelope from the speech wave.
    get_wave()
        Returns the sampling frequency and speech wave signal.

    """
    def __init__(self, filename: str, directory: str) -> None:
        """ Initializes the WaveProcessor class.

        Parameters
        ----------
        filename : str
            Name of the file
        directory : str
            directory containing the file

        """
        extractor = SpeechWaveExtractor(filename=filename, directory=directory)
        self.sfreq, self.wave = extractor.get_wave()

    def downsample(self, sfreq_goal: int) -> None:
        """ Downsamples the speech wave to the desired sampling frequency.

        Parameters
        ----------
        sfreq_goal : int
            Desired sampling frequency.

        """

        lowpass = sfreq_goal / 3.0

        wave_filtered = mne.filter.filter_data(
            self.wave,
            sfreq=self.sfreq,
            l_freq=None,
            h_freq=lowpass,
            method='iir',
            iir_params=dict(ftype='butter', order=3, output='sos')
        )

        wave_resampled = mne.filter.resample(wave_filtered, down=self.sfreq / sfreq_goal, npad='auto')

        self.sfreq = sfreq_goal
        self.wave = wave_resampled

    def extract_Gammatone_envelope(
        self,
        num_filters: int = 24,
        freq_range: tuple[int, int] = (100, 4000),
        compression: float = 0.3
    ) -> None:
        """ Extracts the Gammatone envelope from the speech wave. """
        from scipy import signal

        filterbank = WaveProcessor.gammatone_filterbank(
            sfreq=self.sfreq,
            num_filters=num_filters,
            freq_range=freq_range
        )

        gt_env = np.vstack([signal.filtfilt(filterbank[filt, :], 1.0, self.wave) for filt in range(num_filters)])

        gt_env = np.abs(gt_env)
        gt_env = np.power(gt_env, compression)
        gt_env = np.mean(gt_env, axis=0)

        self.wave = gt_env

    def get_wave(self) -> tuple[int, np.ndarray]:
        """ Returns the sampling frequency and speech wave signal. """
        return self.sfreq, self.wave

    @staticmethod
    def gammatone_filterbank(
        sfreq: int,
        num_filters: int,
        freq_range: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a Gammatone filterbank (Glasberg & Moore, 1990).

        This function generates a Gammatone filterbank, which is a set of bandpass filters that simulate the frequency
        response of the human auditory system. The filters are designed to be similar to the response of the cochlea,
        which is the organ in the inner ear responsible for processing sound.

        Parameters
        ----------
        sfreq : float
            Sampling rate of signal to be filtered.
        num_filters : int
            The number of filters in the filterbank.
        freq_range : tuple of (min_freq, max_freq)
            Frequency range of the filter.

        Returns
        -------
        tuple of (filter_bank, center_freqs)
            A tuple of (filter_bank, center_freqs), where filter_bank is a matrix of shape (num_filters, n), and
            center_freqs is a vector of shape (num_filters,).

        References
        ----------
        Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory filter shapes from notched-noise data.
        Hearing Research, 47(1-2), 103-138. doi:10.1016/0378-5955(90)90170-T

        """
        min_freq, max_freq = freq_range
        erb_min = 24.7 * (4.37 * min_freq / 1000 + 1)
        erb_max = 24.7 * (4.37 * max_freq / 1000 + 1)

        center_freqs_erb = np.linspace(erb_min, erb_max, num_filters)
        center_freqs_hz = (center_freqs_erb / 24.7 - 1) / 4.37 * 1000

        q = 1.0 / (center_freqs_erb * 0.00437 + 1.0)
        bandwidths = center_freqs_hz * q

        filter_bank = np.zeros((num_filters, 4))
        t = np.arange(4) / sfreq
        for i in range(num_filters):
            c = 2 * np.pi * center_freqs_hz[i]
            b = 1.019 * 2 * np.pi * bandwidths[i]

            envelope = (c**4) / (b * np.math.factorial(4)) * t ** 3 * np.exp(-b * t)
            sine_wave = np.sin(c * t)

            filter_bank[i, :] = sine_wave * envelope

        return filter_bank

    @staticmethod
    def padding(array: np.ndarray, length: float, sfreq: float, pad_value: float = np.nan) -> np.ndarray:
        """ Pads the input array with either NaN or 0 values to match the desired length in seconds,
        based on the given sampling frequency.

        Parameters
        ----------
        array : np.ndarray
            Array to be padded.
        length : float
            Desired length of the array in seconds.
        sfreq : float
            Sampling frequency of the array.

        Returns
        -------
        array_padded : np.ndarray
            Padded array.

        """
        padding_length = length * sfreq - array.shape[0]
        array_padded = np.pad(array, (0, padding_length), constant_values=pad_value)

        return array_padded
