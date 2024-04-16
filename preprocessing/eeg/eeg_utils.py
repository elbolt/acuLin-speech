import os
import json
import numpy as np

import mne


class EEGLoader():
    """ Loads EEG data from a fif file and sets reference channels.

    The EEG data is loaded from a fif file and bad as well as reference channels are set based on previous visual
    data inspection. The reference channels are set to the mastoids or to T7 and T8, depending on the subject.
    Bad channels are set to the ones that were previously identified as bad. The data object is returned with bad
    channels and reference channels set.

    Examples
    --------
    >>> eeg_loader = EEGLoader('p01', 'data', '.fif')
    >>> raw = eeg_loader.get_raw()

    """
    def __init__(
        self,
        subject_id: str,
        fif_dir: str,
        file_extension: str
    ) -> None:
        """ Loads EEG data from a fif file and sets reference channels.

        Parameters
        ----------
        subject_id : str
            Subject ID
        fif_dir : str
            fif_dir containing the fif file
        file_extension : str
            File extension of the fif file

        Errors
        ------
        ValueError
            If bad channels and reference channels overlap

        """
        self.subject_id = subject_id
        self.fif_dir = fif_dir
        self.filename = f'{self.subject_id}{file_extension}'
        self.filelocation = os.path.join(self.fif_dir, self.filename)
        self.raw_ = mne.io.read_raw_fif(self.filelocation, preload=False)

        self.bad_channels = None
        self.ref_channels = None

        self.configure_channels()
        self.set_reference()

    def configure_channels(self) -> None:
        """ Configures bad channels and reference channels based on the subject ID. """
        with open('config.json', 'r') as f:
            config = json.load(f)
        bad_channels_dict = config['bad_channels_dict']
        bad_refs = config['bad_references']

        self.bad_channels = bad_channels_dict[self.subject_id]
        self.raw_.info['bads'] = self.bad_channels

        if self.subject_id in bad_refs:
            self.ref_channels = ['T7', 'T8']
            print(f'Using reference channels: {self.ref_channels}.')
        else:
            self.ref_channels = ['EXG3', 'EXG4']

        if self.subject_id in bad_refs and set(self.ref_channels).issubset(set(self.bad_channels)):
            print(f'Bad channels and reference channels overlap for subject {self.subject_id}!')

    def set_reference(self) -> None:
        """ Sets reference channels and loads data. """
        self.raw_.load_data()
        self.raw_.set_eeg_reference(self.ref_channels)

    def get_raw(self) -> mne.io.Raw:
        return self.raw_


class EEGDownSegmenter():
    """ Segments raw EEG data into downsampled, anti-aliasing filtered epochs. An optional high pass filter can be
    applied to the data as well. The epochs are corrected for trigger delay.
    An instance of this class directly returns a mne.Epochs object with the data segmented into epochs, see examples.

    Examples
    --------
    >>> epochs = EEGDownSegmenter(raw, tmin=-0.2, tmax=0.5, decimator=4, highpass=0.5, is_subcortex=False)

    """
    def __init__(
        self,
        raw: mne.io.Raw,
        subject_id: str,
        tmin: float,
        tmax: float,
        decimator: int
    ) -> None:
        """ Segments raw EEG data into downsampled, anti-aliasing filtered epochs.

        Parameters
        ----------
        raw : mne.io.Raw
            Raw EEG data
        subject_id : str
            Subject ID
        tmin : float
            Start time of the epoch
        tmax : float
            End time of the epoch
        decimator : int
            Decimation factor

        """
        self.raw_ = raw
        self.subject_id = subject_id
        self.tmin = tmin
        self.tmax = tmax
        self.decimator = decimator

        self.epochs_ = None
        self.events_ = None

        self.create_epochs()

    def anti_aliasing_filter(self) -> None:
        """ Applies anti-aliasing filter to the raw data

        An anti-aliasing low pass filter at 1/3 of the target frequency is applied to the raw data.
        All frequencies are determined through the decimator
        parameter.

        An optional high pass filter can be applied to the data as well when specified in the constructor.

        """
        sfreq_goal = self.raw_.info['sfreq'] / self.decimator
        lowpass = sfreq_goal / 3.0

        self.raw_.filter(
            l_freq=None,
            h_freq=lowpass,
            method='iir',
            iir_params=dict(order=3, ftype='butter', output='sos')
        )

    def get_events(self) -> None:
        """ Finds audiobook events in the raw data and accounts for trigger delay and participant-related problems.

        The event code for audio onset is "256". The delay from the transductor to the eardrum `delta_t` is 1.07 ms.
        The trigger delay is accounted for by simply adding the delay in samples to the event onset.

        """
        events = mne.find_events(
            self.raw_,
            stim_channel='Status',
            min_duration=(1 / self.raw_.info['sfreq']),
            shortest_event=1,
            initial_event=True,
        )

        # Special case for participant 07, where three extra events occurred in the break
        if self.subject_id == 'p07':
            mask = np.ones(events.shape[0], dtype=bool)
            mask[39:42] = False
            events = events[mask]

        mask = events[:, 2] == 256
        audio_events = events[mask, :]
        audio_events[:, 0] = audio_events[:, 0]

        self.events_ = audio_events

    def create_epochs(self):
        """ Creates epochs from the raw data. """
        self.anti_aliasing_filter()
        self.get_events()

        epochs = mne.Epochs(
            self.raw_,
            self.events_,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=None,
            preload=True
        )

        epochs.decimate(self.decimator)

        self.epochs_ = epochs

    def get_epochs(self) -> mne.Epochs:
        """ Returns the segmented epochs. """
        return self.epochs_
