""" Calculates word-based phoneme surprisal and entropy for each phoneme in the audiobook segment. """

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def create_word_phoneme_df(words: pd.DataFrame, phones: pd.DataFrame) -> pd.DataFrame:
    """ Builds a DataFrame that maps each word from the audiobook segments to its corresponding phonetic sequence.

    Parameters
    ----------
    words : pandas.DataFrame
        DataFrame containing words and their onsets in seconds, structured as follows:
        element onset
        word1   0.8
    phones : pandas.DataFrame
        DataFrame containing phones and their onsets in seconds, structured as follows:
        element onset
        phone1  0.8

    Returns
    -------
    pandas.DataFrame
        DataFrame containing words and their corresponding phonetic sequences, structured as follows:
        word    phones
        word1   phone1 phone2 phone3

    """
    word_phonemes = []

    for i, word_row in words.iterrows():
        current_onset = word_row['onset']
        next_onset = words['onset'].iloc[i+1] if i+1 < len(words) else float('inf')

        phonemes = phones[(phones['onset'] >= current_onset) & (phones['onset'] < next_onset)]

        phoneme_str = ' '.join(phonemes['element'].tolist())

        word_phonemes.append({'word': word_row['element'], 'phones': phoneme_str})

    return pd.DataFrame(word_phonemes)


def prepare_custom_dict(dict_dir: str, return_df: bool = False) -> tuple[dict, int]:
    """ Prepares custom dictionary for generating phonetic speech representations.

    Custom dict is transformed from a DataFrame with columns 'word', 'pronunciation', 'DeReKo_1gram' to a dictionary
    mapping words to their phonetic representations and to DeReKoGram frequencies. Entries with missing DeReKoGram
    frequencies are filled with 1.0 (Brodbeck et al., 2018). The total frequency of all words in the custom dictionary
    is computed.

    Parameters
    ----------
    dict_dir : str
        Path to the custom dictionary file.
    is_df : bool | False
        Whether the custom dictionary is a DataFrame or a dictionary.

    Returns
    -------
    dict
        Dictionary mapping words to their phonetic representations and to DeReKoGram frequencies.
    int
        Total frequency of all words in the custom dictionary.

    """
    custom_dict_df = pd.read_csv(dict_dir, sep='\t')

    custom_dict_df['DeReKo_1gram'] = custom_dict_df['DeReKo_1gram'].fillna(1.0)
    custom_dict = {row['pronunciation']: (row['word'], row['DeReKo_1gram']) for _, row in custom_dict_df.iterrows()}

    total_corpus_freq = sum(word_freq[1] for word_freq in custom_dict.values())

    custom_dict = custom_dict_df if return_df else custom_dict

    return custom_dict, total_corpus_freq


def get_phonetic_representations(
    phones_dir: str,
    words_dir: str,
    custom_dict: dict,
    total_corpus_freq: int,
    out_dir: str,
    segment: str
) -> None:
    """ Calculates word-based phoneme surprisal and entropy for each phoneme in the audiobook segment.

    - Phoneme surprisal is computed as the negative log probability of the phoneme given the current phonetic sequence
    (i.e., "activated cohort"), as defined by a custom dictionary.
    - Phoneme entropy is computed as the Shannon entropy of the cohort of words that share the current phonetic
    sequence.

    Parameters
    ----------
    phones_dir : str
        Path to the directory containing the phoneme files.
    words_dir : str
        Path to the directory containing the word files.
    custom_dict : dict
        Dictionary mapping words to their phonetic representations and to DeReKoGram frequencies.
    total_corpus_freq : int
        Total frequency of all words in the custom dictionary.
    out_dir : str
        Path to the output directory.
    segment : str
        Name of the audiobook segment.


    """
    # Get word-phoneme maps for current segment
    phones_df = pd.read_csv(phones_dir / segment, sep='\t')
    words_df = pd.read_csv(words_dir / segment, sep='\t')
    segment_map_df = create_word_phoneme_df(words_df, phones_df)

    all_surprisal = []
    all_entropies = []

    # Iterate over all words in segmentpet
    for word_idx in range(len(segment_map_df)):
        phones = segment_map_df.iloc[word_idx]['phones']
        phone_seq = phones.split(' ')
        phone_length = len(phone_seq)

        # Arrays for caching surprisal and entropy values until sequence is complete
        probabilities = np.zeros(phone_length)
        surprisal = np.zeros(phone_length)
        entropies = np.zeros(phone_length)

        start_freq = total_corpus_freq  # start with total frequency

        # Iterate over all phones in word
        for phone_idx in range(phone_length):
            phone_seq_ = ' '.join(phone_seq[:phone_idx + 1])

            cohort = [word_freq for pron, word_freq in custom_dict.items() if pron.startswith(phone_seq_)]

            phone_seq_freq = sum(freq for _, freq in cohort)

            cohort_probabilities = [freq / start_freq for _, freq in cohort if start_freq > 0]

            phone_entropy = -sum(p * np.log(p) for p in cohort_probabilities if p > 0)
            entropies[phone_idx] = phone_entropy

            if phone_seq_freq == 0:
                probabilities[phone_idx] = 0.99
                surprisal[phone_idx] = 0.01
            else:
                numerator = phone_seq_freq
                denominator = start_freq

                probabilities[phone_idx] = numerator / denominator
                surprisal_ = -np.log(probabilities[phone_idx]) if probabilities[phone_idx] > 0 else float('inf')
                surprisal[phone_idx] = surprisal_

            start_freq = phone_seq_freq  # update start_freq for next iteration

        all_surprisal.append(surprisal)
        all_entropies.append(entropies)

    phones_df['surprisal'] = np.concatenate(all_surprisal)
    phones_df['entropy'] = np.concatenate(all_entropies)

    phones_df.to_csv(out_dir / segment, sep='\t', index=False)


if __name__ == '__main__':
    print(f'Running {__file__} ...')

    with open('config.json') as f:
        config = json.load(f)

    phones_dir = Path(config['phones_dir'])
    words_dir = Path(config['words_dir'])

    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    dict_dir = Path(config['dict_dir'])  # Pronunciation dictionary and total corpus frequency
    custom_dict, total_corpus_freq = prepare_custom_dict(dict_dir)

    audiobook_segments = sorted(f for f in os.listdir(phones_dir) if f.endswith('.txt'))

    for segment in tqdm(audiobook_segments, desc='segment'):
        get_phonetic_representations(phones_dir, words_dir, custom_dict, total_corpus_freq, out_dir, segment)
