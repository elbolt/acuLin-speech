import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path


def get_window_boundary(df, make_plots: bool = False, figures_dir: Path = Path('figures')) -> np.ndarray:
    """ Get time boundaries for each response type.

    The dataframe `df` contains the two largest peaks (if found) previously identified for each subject_id, response
    and cluster type and stored in a dataframe. This function uses k-means clustering to find the boundary between
    the two peaks. The function returns a list of boundaries for each response type.

    `df` should be structured comparable to the following example:
    > subject_id    response      cluster    latency_ms   amplitude
    > p01           acoustic      frontal    90                2.5
    > p01           acoustic      temporal   92                3.1
    > p01           segmentation  frontal    90                2.5

    Parameters
    ----------
    df : pandas.DataFrame
            DataFrame containing the peak latency and amplitude for each response type.
    make_plots : bool
            Whether to plot and save a figure of the clustering.

    Returns
    -------
    boundaries : numpy.ndarray
            Array of shape (n_models, n_responses) containing the time boundaries for each response type.
    """

    response_name = list(df['response'].unique())
    boundaries = []

    if make_plots:
        Path(figures_dir).mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['font.family'] = 'Helvetica'

        speech_col = '#009988'  # teal
        word_col = '#000000'
        phone_col = '#767676'

        color_palette = [
                speech_col, speech_col,
                word_col, phone_col,
                word_col, word_col,
                phone_col, phone_col
        ]

        fig, axes = plt.subplots(4, 2, figsize=(4, 4.5), constrained_layout=True, sharex=True, sharey=True)
        for idx, response in enumerate(response_name):
            df_subset = df[df['response'] == response]

            kmeans = KMeans(n_clusters=2, random_state=24, n_init=100)

            kmeans.fit(df_subset['latency_ms'].values.reshape(-1, 1))
            centroids = kmeans.cluster_centers_
            boundary = np.mean(centroids)

            boundaries.append(boundary)

            sns.scatterplot(
                x='latency_ms',
                y='amplitude',
                data=df_subset,
                ax=axes[idx // 2, idx % 2],
                color=color_palette[idx],
                s=18,
                # edgecolor='none'
            )

            axes[idx // 2, idx % 2].axvline(boundary, color='black', linestyle='--')
            response_upper = response[0].upper() + response[1:]
            axes[idx // 2, idx % 2].set_title(f'{response_upper}, {boundary:.0f} ms', fontsize=10)
            axes[idx // 2, idx % 2].set_xlabel('Latency (ms)')
            axes[idx // 2, idx % 2].set_ylabel('Amplitude ($z$)')

        fig.savefig(figures_dir / 'time_clusters.eps', format='eps', dpi=300)

    else:
        for idx, response in enumerate(response_name):
            df_subset = df[df['response'] == response]

            kmeans = KMeans(n_clusters=2, random_state=24, n_init=100)

            kmeans.fit(df_subset['latency_ms'].values.reshape(-1, 1))
            centroids = kmeans.cluster_centers_
            boundary = np.mean(centroids)

            boundaries.append(boundary)

    # Back-transform to original time scale
    boundaries = np.array(boundaries)

    return boundaries


def determine_window(row: pd.Series) -> str:
    """ Determine if a peak is early or late based on the window boundaries.

    Parameters
    ----------
    row : pandas.Series
            Row of a DataFrame containing the peak latency and amplitude for each response type.

    Returns
    -------
    str
            'early' if the peak is early, 'late' if the peak is late.
    """
    boundary = response_boundaries[row['response']]

    return "early" if row['latency_ms'] < boundary else "late"


if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Configuration parameters
    dataframes_dir = Path(config['dataframes_dir'])
    figures_dir = Path(config['figures_dir'])
    peak_filename = config['peak_filename']
    peak_window_filename = config['peak_window_filename']
    df_peaks = pd.read_csv(dataframes_dir / peak_filename)

    # Get time boundaries for each response type
    boundaries = get_window_boundary(df_peaks, make_plots=True, figures_dir=figures_dir)

    # Determine if a peak is early or late based on the window boundaries
    responses = df_peaks['response'].unique()
    response_boundaries = dict(zip(responses, boundaries))
    df_peaks['window'] = df_peaks.apply(determine_window, axis=1)

    df_peaks.to_csv(dataframes_dir / peak_window_filename, index=False)
