import numpy as np
from scipy.signal import find_peaks


def get_relevant_data(
        kernel_data: np.ndarray,
        time_array: np.ndarray,
        tmin: float,
        tmax: float,
        channel_indices: list
) -> tuple[np.ndarray, np.ndarray]:
    """ Get the relevant data from the kernel.

    Parameters
    ----------
    kernel_data : np.ndarray
        Kernel data of shape (n_responses, n_channels, n_times).
    time_array : np.ndarray
        Time array.
    tmin : float
        Minimum time to extract.
    tmax : float
        Maximum time to extract.
    channel_indices : list
        List of channel indices to extract.

    Returns
    -------
    np.ndarray
        Extracted kernel of shape (n_responses, len(channel_indices), n_times_within_range).
    np.ndarray
        Extracted time array.

    """

    time_mask = (time_array > tmin) & (time_array < tmax)
    extracted_time_array = time_array[time_mask]
    extracted_kernel = kernel_data[:, channel_indices][:, :, time_mask]

    return extracted_kernel, extracted_time_array


def get_largest_peaks(array: np.ndarray, time_arr: np.ndarray, prominence: float = 0.5) -> tuple:
    """ Find the two largest peak in an array between tmin and tmax.

    Parameters
    ----------
    array : np.ndarray
        Array of values.
    time_arr : np.ndarray
        Array of time lags in the array.

    Returns
    -------
    latencies : list
        List of peak latencies.
    amplitudes : list
        List of peak amplitudes.

    Note: The peaks are extracted from the normalized array but the amplitudes are taken from the original array.

    """

    normalized_array = (array - array.mean()) / array.std()

    peaks, properties = find_peaks(normalized_array, prominence=prominence)

    if len(peaks) == 0:
        return np.array([]), np.array([])
    else:
        sorted_peaks = sorted(zip(peaks, properties['prominences']), key=lambda x: x[1], reverse=True)
        top_peaks = sorted_peaks[:2]

        indices = [peak[0] for peak in top_peaks]
        indices = sorted(indices)
        latencies = time_arr[indices]
        amplitudes = array[indices]

        return latencies, amplitudes
