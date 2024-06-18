import numpy as np
import eelbrain
from pathlib import Path
from sklearn.linear_model import LinearRegression


class BoostingDataLoader():
    """ Class to load data for boosting analysis.

    This class is used to load EEG and speech data for boosting analysis. The EEG data is stored as numpy arrays in the
    `eeg_dir` directory, with each array named after the subject ID. The shape of the EEG arrays is (n_epochs,
    n_channels, n_samples). The speech data is stored as numpy arrays in the `feature_dir` directory, with a shape of
    (n_epochs, n_features, n_samples).

    The main method of this class is `get_data`, which returns the features and EEG residuals for the selected model.
    The features and EEG residuals are prepared according to the encoding model selected.
    The table below shows the encoding models and the features to regress out and boost on for each model:

    +-------------------------+---------------------+-------------------+
    | Encoding Model          | Regress Out         | Boost On          |
    +-------------------------+---------------------+-------------------+
    | "acoustic"              | word_segment.npy    | acoustic.npy      |
    |                         | phone_segment.npy   |                   |
    |                         | words.npy           |                   |
    |                         | phones.npy          |                   |
    +-------------------------+---------------------+-------------------+
    | "word-level             | acoustic.npy        | word_segment.npy  |
    | segmentation"           | words.npy           |                   |
    | `word_segment`          | phones.npy          |                   |
    +-------------------------+---------------------+-------------------+
    | "phoneme-level          | acoustic.npy        | phone_segment.npy |
    | segmentation"           | words.npy           |                   |
    | `phone_segment`         | phones.npy          |                   |
    +-------------------------+---------------------+-------------------+
    | "word-level linguistic" | acoustic.npy        | words.npy         |
    | `words`                 | word_segment.npy    |                   |
    |                         | phone_segment.npy   |                   |
    +-------------------------+---------------------+-------------------+
    | "phoneme-level          | acoustic.npy        | phones.npy        |
    | linguistic"             | word_segment.npy    |                   |
    | `phones`                | phone_segment.npy   |                   |
    +-------------------------+---------------------+-------------------+

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
            model_attributes: dict,
            sfreq: int = 128,
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        subject_id : str
            Subject ID.
        eeg_dir : str | Path
            Path to the EEG data.
        feature_dir : str | Path
            Path to the feature data.
        model_attributes : dict
            Dictionary of model attributes.
        sfreq : int, default=128
            Sampling frequency of the EEG data.

        """
        self.subject_id = subject_id
        self.eeg_dir = Path(eeg_dir)
        self.feature_dir = Path(feature_dir)
        self.sfreq = sfreq
        self.valid_models = list(model_attributes.keys())

        self.load_data(model_attributes)

    def load_data(self, model_attributes: dict) -> None:
        """ Load data from files. """
        try:
            for attr, filename in model_attributes.items():
                setattr(self, attr, np.load(self.feature_dir / filename))
            self.eeg = np.load(self.eeg_dir / f'{self.subject_id}.npy')
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading file: {e.filename}")
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading data: {e}")

        # In participant 45, the first epoch is missing because we forgot to start the recording.
        # if self.subject_id == 'p45':
        #     self.acoustic = self.acoustic[1:, ...]
        #     self.word_segment = self.word_segment[1:, ...]
        #     self.phone_segment = self.phone_segment[1:, ...]
        #     self.words = self.words[1:, ...]
        #     self.phones = self.phones[1:, ...]

        if self.subject_id == 'p45':
            for attr in model_attributes:
                setattr(self, attr, getattr(self, attr)[1:, ...])

        self.check_data(model_attributes)

    def check_data(self, model_attributes: dict) -> None:
        """ Check if all arrays have the same number of epochs and samples. """
        data_shapes = {attr: getattr(self, attr).shape for attr in model_attributes.keys()}

        shapes_0 = {shape[0] for shape in data_shapes.values()}
        shapes_2 = {shape[2] for shape in data_shapes.values()}

        if len(shapes_0) > 1 or len(shapes_2) > 1:
            raise ValueError('All arrays must have the same number of epochs and samples.')

        for attr, shape in data_shapes.items():
            if len(shape) != 3:
                raise ValueError(f'Array {attr} does not have a 3D shape.')

    def get_features(self, model: str) -> tuple[np.ndarray, np.ndarray]:
        """ Get features and the features to regress out for the selected model.

        Parameters
        ----------
        model : str
            Encoding model to use. Must be one of the model attributes defined in `model_attributes`.

        Returns
        -------
        feature : ndarray (n_epochs, n_features, n_samples)
            feature data of interest for the selected model.
        features_out : ndarray (n_epochs, n_features, n_samples)
            feature data to regress out for the selected model.

        """

        model_features = {
            'acoustic': (self.acoustic, [self.word_segment, self.phone_segment, self.words, self.phones]),
            'word_segment': (self.word_segment, [self.acoustic, self.words, self.phones]),
            'phone_segment': (self.phone_segment, [self.acoustic, self.words, self.phones]),
            'words': (self.words, [self.acoustic, self.word_segment, self.phone_segment]),
            'phones': (self.phones, [self.acoustic, self.word_segment, self.phone_segment])
        }

        if model not in model_features:
            raise ValueError(f'Invalid model: {model}')

        feature, features_out_list = model_features[model]
        features_out = np.concatenate(features_out_list, axis=1)

        return feature, features_out

    def get_eeg_residuals(self, features_out: np.ndarray) -> np.ndarray:
        """ Get EEG for the selected model.

        In order to regress out the features not of interest, this method uses a Linear Regression model. The EEG
        residuals are obtained by subtracting the values predicted by the OLS model from the original EEG data.

        Parameters
        ----------
        features_out : ndarray (n_epochs, n_features, n_samples)
            feature data to regress out for the selected model.

        Returns
        -------
        eeg_residuals_array : ndarray (n_epochs, n_channels, n_samples)
            EEG residuals with respect to the features not of interest.


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
        """ Check if the selected model is valid.

        Parameters
        ----------
        model : str
            Encoding model to use.
        model_list : list
            List of valid encoding models.

        """
        if model not in self.valid_models:
            raise ValueError(f'Invalid model: {model}. Valid models are: {", ".join(self.valid_models)}.')

    def get_data(self, model: str, trim_beginnings: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """ Get the data for the selected model.

        Parameters
        ----------
        model : str
            Encoding model to use. Must be one of 'acoustic', 'word-level segmentation',
            'phoneme-level segmentation', 'word-level linguistic', 'phoneme-level linguistic'

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
        tmin: int = -200e-3,
        tmax: int = 600e-3,
        partition: int = 6
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        feature : np.ndarray
            Feature data of interest. Shape: (n_epochs, n_features, n_samples).
        eeg : np.ndarray
            EEG data. Shape: (n_epochs, n_channels, n_samples).
        sfreq : int, default=128
            Sampling frequency of the EEG data.
        tmin : float, default=-200e-3
            Start time of the analysis window in seconds.
        tmax : float, default=600e-3
            End time of the analysis window in seconds.
        partition : int, default=6
            Number of partitions for the boosting analysis.

        """
        self.eeg = eeg
        self.feature = feature
        self.sfreq = sfreq
        self.tmin = tmin
        self.tmax = tmax
        self.partition = partition

        pass

    def get_NDVar_from_array(self, eeg: np.ndarray, feature: np.ndarray, sfreq: int) -> tuple[eelbrain.NDVar]:
        """ Take feature and EEG data and create Eelbrain NDVar objects.

        feature shape: (n_epochs, n_features, n_samples)
        EEG shape: (n_epochs, n_channels, n_samples); layout: 'Biosemi32'
        """

        timing = eelbrain.UTS(0, (1 / sfreq), eeg.shape[2])
        sensor = eelbrain.Sensor.from_montage('biosemi32')[:32]

        n_features = feature.shape[1]

        eel_eeg = eelbrain.NDVar(eeg, (eelbrain.Case, sensor, timing), name='EEG')
        features = []
        for i in range(n_features):
            feature_data = feature[:, i, :]
            feature_name = f'{i+1}st feature' if i == 0 else f'{i+1}nd feature'
            feature_var = eelbrain.NDVar(feature_data, (eelbrain.Case, timing), name=feature_name)
            features.append(feature_var)

        return (eel_eeg, [*features])

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
            Feature data.
        sfreq : int
            Sampling frequency of the EEG data.
        tmin : float
            Start time of the analysis window.
        tmax : float
            End time of the analysis window.
        partition : int
            Number of partitions for the boosting analysis.

        """
        try:
            eel_eeg, features = self.get_NDVar_from_array(
                eeg=eeg,
                feature=feature,
                sfreq=sfreq
            )

            boosting_result = eelbrain.boosting(
                y=eel_eeg,
                x=features,
                tstart=tmin,
                tstop=tmax,
                scale_data=True,
                basis=100e-3,
                partitions=partition,
                partition_results=False,
                test=True,
            )

            time_vector = boosting_result.h_time.times
            n_features = feature.shape[1]
            kernels = np.full((n_features, eeg.shape[1], time_vector.shape[0]), np.nan)

            for i in range(n_features):
                kernels[i, ...] = boosting_result.h[i]

            scores = boosting_result.r.get_data()

            return kernels, scores, time_vector

        except Exception as e:
            raise RuntimeError(f"An error occurred while running boosting analysis: {e}")

    def get_results(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Get the boosting results.

        Returns
        -------
        kernels : ndarray (n_features, n_channels, n_times)
            Estimated kernels.
        scores : ndarray (n_features, n_times)
            Boosting scores.
        time_vector : ndarray (n_times,)
            Time course of the estimated kernels.

        """

        kernels, scores, time_vector = self.boosting(
            eeg=self.eeg,
            feature=self.feature,
            sfreq=self.sfreq,
            tmin=self.tmin,
            tmax=self.tmax,
            partition=self.partition
        )

        return kernels, scores, time_vector
