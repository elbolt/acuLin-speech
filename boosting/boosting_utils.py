import numpy as np
import eelbrain
from pathlib import Path
from sklearn.linear_model import LinearRegression


class BoostingDataLoader():
    """ Class to load data for boosting analysis.

    In `eeg_dir`, my eeg data are stored as numpy arrays, shape: (n_epochs, n_channels, n_samples).
    Each eeg array is named after the subject ID.
    In `feature_dir, my speech data are stored as numpy arrays, shape: (n_epochs, n_features, n_samples).

    The main method of this class is `get_data`, which returns the features and the EEG residuals for the selected
    model. In line with Kries et al. (2023), I prepare the speech and feature data as follows:

    +-----------------+--------------------+--------------------+
    | Encoding model  | Regress out        | Boost on           |
    +-----------------+--------------------+--------------------+
    | "acoustic"      | `segmentation.npy` | `acoustics.npy`    |
    |                 | `words.npy`        |                    |
    |                 | `phones.npy`       |                    |
    +-----------------+--------------------+--------------------+
    | "segmentation"  | `acoustics.npy`    | `segmentation.npy` |
    |                 | `words.npy`        |                    |
    |                 | `phones.npy`       |                    |
    +-----------------+--------------------+--------------------+
    | "word-based"    | `acoustics.npy`    | `words.npy`        |
    |                 | `segmentation.npy` |                    |
    +-----------------+--------------------+--------------------+
    | "phoneme-based" | `acoustics.npy`    | `phones.npy`       |
    |                 | `segmentation.npy` |                    |
    +-----------------+--------------------+--------------------+

    The features contain the following n_features:
    - `acoustics.npy`: envelopes, envelope onsets
    - `segmentation.npy`: word onsets, phone onsets
    - `words.npy`: word surprisal, word frequency
    - `phones.npy`: phoneme surprisal, phonem entropy

    The EEG residuals are obtained by regressing out the features not of interest using a Linear Regression model.

    Methods
    -------
    load_data()
        Load data from files.
    check_data()
        Check if all arrays have the same number of epochs and samples.
    get_features(model: str)
        Get features and the features to regress out for the selected model.
    get_eeg_residuals(features_out)
        Get EEG for the selected model.

    Usage
    -----
    >>> from utils import BoostingDataLoader
    >>> data_loader = BoostingDataLoader(subject_id='01', eeg_dir='eeg', feature_dir='features')
    >>> features, eeg_residuals = data_loader.get_data(model='acoustic')

    """
    def __init__(
            self,
            subject_id: str,
            eeg_dir: str,
            feature_dir: str,
            sfreq: int = 128
    ) -> None:
        """ Initialize the class.

        Parameters
        ----------
        subject_id : str
            Subject ID.
        eeg_dir : str | Path
            Path to the EEG data.
        feature_dir : str | Path
            Path to the feature data.
        sfreq : int | 128
            Sampling frequency of the EEG data.

        """
        self.subject_id = subject_id
        self.eeg_dir = eeg_dir if eeg_dir.is_dir() else Path(eeg_dir)
        self.feature_dir = feature_dir if feature_dir.is_dir() else Path(feature_dir)
        self.sfreq = sfreq

        self.load_data()

    def load_data(self) -> None:
        """ Load data from files. """
        file_attributes = {
            'acoustics': 'acoustics.npy',
            'segmentation': 'segmentation.npy',
            'words': 'words.npy',
            'phones': 'phones.npy'
        }

        for attr, filename in file_attributes.items():
            setattr(self, attr, np.load(self.feature_dir / filename))

        self.eeg = np.load(self.eeg_dir / f'{self.subject_id}.npy')

        # In participant 45, the first epoch is missing because we forgot to start the recording.
        if self.subject_id == 'p45':
            self.acoustics = self.acoustics[1:, ...]
            self.segmentation = self.segmentation[1:, ...]
            self.words = self.words[1:, ...]
            self.phones = self.phones[1:, ...]

        self.check_data()

    def check_data(self) -> None:
        """ Check if all arrays have the same number of epochs and samples. """
        file_list = ['acoustics', 'segmentation', 'words', 'phones', 'eeg']
        shape_0 = getattr(self, file_list[0]).shape[0]
        shape_2 = getattr(self, file_list[0]).shape[2]

        for attr in file_list:
            if getattr(self, attr).ndim != 3:
                raise ValueError(f'Array {attr} does not have a 3D shape.')
            if getattr(self, attr).shape[0] != shape_0:
                raise ValueError('Not all arrays have the same number of epochs.')
            if getattr(self, attr).shape[2] != shape_2:
                raise ValueError('Not all arrays have the same number of samples.')

    def get_features(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        """ Get features and the features to regress out for the selected model.

        Parameters
        ----------
        model : str
            Encoding model to use. Must be one of 'acoustic', 'segmentation', 'word level', 'phoneme level'.

        Returns
        -------
        feature : ndarray (n_epochs, n_features, n_samples)
            feature data of interest for the selected model.
        features_out : ndarray (n_epochs, n_features, n_samples)
            feature data to regress out for the selected model.

        """

        if model == 'acoustic':
            feature = self.acoustics
            features_out = np.concatenate((self.segmentation, self.words, self.phones), axis=1)
        elif model == 'segmentation':
            feature = self.segmentation
            features_out = np.concatenate((self.acoustics, self.words, self.phones), axis=1)
        elif model in ['word level', 'phoneme level']:
            feature = self.words if model == 'word level' else self.phones
            features_out = np.concatenate((self.acoustics, self.segmentation), axis=1)
        else:
            raise ValueError(f'Invalid model: {model}')

        return feature, features_out

    def get_eeg_residuals(self, features_out: np.ndarray) -> np.ndarray:
        """ Get EEG for the selected model.

        In order to regress out the features not of interest, this method uses a Linear Regression model. The EEG
        residuals are obtained by subtracting the values predicted by the OLS model from the original EEG data.

        """
        eeg_residuals_list = []
        eeg = self.eeg.copy()

        for epoch in range(eeg.shape[0]):

            X = features_out[epoch, ...].reshape(features_out.shape[2], -1)
            Y = eeg[epoch, ...].reshape(eeg.shape[2], -1)

            model = LinearRegression()

            model.fit(X, Y)
            Y_pred = model.predict(X)
            eeg_residuals_list.append(Y - Y_pred)

        eeg_residuals_array = np.array(eeg_residuals_list).reshape(self.eeg.shape)

        return eeg_residuals_array

    def check_model(self, model: str) -> None:
        model_list = ['acoustic', 'segmentation', 'word level', 'phoneme level']
        if model not in model_list:
            raise ValueError(f'Invalid model: {model}. Model must be in {model_list}.')

    def get_data(self, model: str, trim_beginnings: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """ Get the data for the selected model.

        Parameters
        ----------
        model : str
            Encoding model to use. Must be one of 'acoustic', 'segmentation', 'word level', 'phoneme level'.

        Returns
        -------
        feature : ndarray (n_epochs, n_features, n_samples)
            feature data of interest for the selected model.
        eeg_residuals : ndarray (n_epochs, n_channels, n_samples)
            EEG residuals with respect to the features not of interest.
        trim_beginnings : bool | False
            Whether to cut of the first second of the data.

        """
        self.check_model(model=model)
        feature, features_out = self.get_features(model=model)
        eeg_residuals = self.get_eeg_residuals(features_out)

        if trim_beginnings:
            feature = feature[:, :, self.sfreq:]
            eeg_residuals = eeg_residuals[:, :, self.sfreq:]

        return feature, eeg_residuals


class Booster():
    """ Class to run boosting analysis.

    Methods
    -------
    get_NDVar_from_array(eeg, feature, sfreq)
        Take feature and EEG data and create Eelbrain NDVar objects.
    boosting(eeg, feature, sfreq, tmin, tmax, partition)
        Run boosting analysis using eelbrain.boosting.
    get_results()
        Get the boosting results.

    Usage
    -----
    >>> from boosting_utils import Booster
    >>> booster = Booster(feature=feature, eeg=eeg)
    >>> kernels, scores, time_vector = booster.get_results()

    """
    def __init__(
        self,
        feature: np.ndarray,
        eeg: np.ndarray,
        sfreq: int = 128,
        tmin: float = -100e-3,
        tmax: float = 600e-3,
        partition: int = 6
    ) -> None:
        """ Initialize the class.

        Parameters
        ----------
        feature : ndarray (n_epochs, n_features, n_samples)
            feature data of interest.
        eeg : ndarray (n_epochs, n_channels, n_samples)
            EEG data.
        sfreq : int | 128
            Sampling frequency of the EEG data.
        tmin : float | -100e-3
            Start time of the analysis window.
        tmax : float | 600e-3
            End time of the analysis window.
        partition : int | 6
            Number of partitions for the boosting analysis.

        """
        self.boosting(
            eeg=eeg,
            feature=feature,
            sfreq=sfreq,
            tmin=tmin,
            tmax=tmax,
            partition=partition
        )

    def get_NDVar_from_array(self, eeg: np.ndarray, feature: np.ndarray, sfreq: int) -> tuple[eelbrain.NDVar]:
        """ Take feature and EEG data and create Eelbrain NDVar objects.

        feature shape: (n_epochs, n_features, n_samples)
        EEG shape: (n_epochs, n_channels, n_samples); layout: 'Biosemi32'
        """

        timing = eelbrain.UTS(0, (1 / sfreq), eeg.shape[2])
        sensor = eelbrain.Sensor.from_montage('biosemi32')[:32]

        eel_eeg = eelbrain.NDVar(eeg, (eelbrain.Case, sensor, timing), name='EEG')
        first_feature = eelbrain.NDVar(feature[:, 0, :], (eelbrain.Case, timing), name='1st feature')
        second_feature = eelbrain.NDVar(feature[:, 1, :], (eelbrain.Case, timing), name='2nd feature')

        return eel_eeg, first_feature, second_feature

    def boosting(
        self,
        eeg: np.ndarray,
        feature: np.ndarray,
        sfreq: int,
        tmin: float,
        tmax: float,
        partition: int
    ) -> None:
        """ Run boosting analysis using eelbrain.boosting.

        Parameters
        ----------
        eeg : ndarray (n_epochs, n_channels, n_samples)
            EEG data.
        feature : ndarray (n_epochs, n_features, n_samples)
            feature data of interest.
        sfreq : int
            Sampling frequency of the EEG and feature data.
        tmin : float
            Start time of the kernel estimation window.
        tmax : float
            End time of the kernel estimation window.
        partition : int
            Number of partitions for the boosting analysis.

        Returns
        -------
        kernels : ndarray (n_features, n_channels, n_times)
            Estimated kernels.
        scores : ndarray (n_features, n_times)
            Boosting scores.
        time_vector : ndarray (n_times,)
            Time course of the estimated kernels.

        """

        eel_eeg, first_feature, second_feature = self.get_NDVar_from_array(
            eeg=eeg,
            feature=feature,
            sfreq=sfreq
        )

        boosting_result = eelbrain.boosting(
            y=eel_eeg,
            x=[first_feature, second_feature],
            tstart=tmin,
            tstop=tmax,
            scale_data=True,
            basis=100e-3,
            partitions=partition,
            partition_results=False,
            test=True,
        )

        time_vector = boosting_result.h_time.times
        first_kernel, second_kernel = boosting_result.h

        kernels = np.full((feature.shape[1], eeg.shape[1], time_vector.shape[0]), np.nan)
        kernels[0, ...] = first_kernel
        kernels[1, ...] = second_kernel

        scores = boosting_result.r.get_data()

        self.kernels = kernels
        self.scores = scores
        self.time_vector = time_vector

    def get_results(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Get the boosting results. """

        return self.kernels, self.scores, self.time_vector
