""" Calculate surprisal and frequency for each word in a sentence using the pre-trained German BERT model and DeReKo
1grams. """

import os
import json
import logging
import torch
import numpy as np
import pandas as pd
import tqdm

from pathlib import Path
from transformers import BertTokenizer, BertForMaskedLM
from get_phonetic_representations import prepare_custom_dict

# Set logging level to suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


def calculate_surprisal(tokenized_sentence: list) -> list:
    """ Calculate the surprisal of each word in a sentence using the pre-trained German BERT model.

    Parameters
    ----------
    tokenized_sentence : list
        A list of tokenized words representing a sentence.

    Returns
    -------
    surprisals : list
        A list of surprisal values for each word in the sentence.

    Example:
    >>> sentence = ['Ich', 'liebe', 'BERT']
    >>> calculate_surprisals(sentence)
    [2.345, 3.678, 1.234, 4.567]

    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-german-dbmdz-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-german-dbmdz-uncased')

    surprisals = []

    for i, word in enumerate(tokenized_sentence):
        if word.isalpha():
            partial_sentence = tokenized_sentence[:i + 1]  # create partial sentence up to current word
            partial_sentence[i] = '[MASK]'  # mask current word
            masked_text = ' '.join(partial_sentence)

            tokenized_text = tokenizer.tokenize(masked_text)
            masked_index = tokenized_text.index("[MASK]")

            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])

            # Predict all tokens
            with torch.no_grad():
                outputs = model(tokens_tensor)
                predictions = outputs[0]

            # Get the predicted probability of the original word
            predicted_index = tokenizer.convert_tokens_to_ids([word])[0]
            predicted_prob = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)[predicted_index].item()

            # Calculate surprisal
            surprisal = -np.log(predicted_prob) if predicted_prob > 0 else float('inf')
            surprisals.append(surprisal)

    return surprisals


if __name__ == '__main__':
    print(f'Running {__file__} ...')

    with open('config.json') as f:
        config = json.load(f)

    words_dir = Path(config['words_dir'])

    out_dir = Path(config['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    dict_dir = Path(config['dict_dir'])
    custom_dict_df, total_corpus_freq = prepare_custom_dict(dict_dir, return_df=True)

    snips = sorted(f for f in os.listdir(words_dir) if f.endswith('.txt'))

    # Calculate surprisals for each word in each snippet using DeReKo 1grams
    for snip in tqdm.tqdm(snips, desc='segment'):
        idx = snips.index(snip)

        snip_df = pd.read_csv(words_dir / snip, sep='\t')

        tokenized_sentence = snip_df.element.values.tolist()

        surprisals = calculate_surprisal(tokenized_sentence)

        snip_df['surprisal'] = surprisals
        snip_df['DeReKo_1gram'] = snip_df['element'].map(custom_dict_df.set_index('word')['DeReKo_1gram'])
        snip_df['probability'] = snip_df['DeReKo_1gram'] / total_corpus_freq
        snip_df['frequency'] = -np.log(snip_df['probability'])
        snip_df.drop(columns=['DeReKo_1gram', 'probability'], inplace=True)

        snip_df.to_csv(out_dir / snip, sep='\t', index=False)
