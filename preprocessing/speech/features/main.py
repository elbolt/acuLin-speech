"""
Run pipelines to extract envelope and speech features on the audio snippets.

"""
import json
import envelopes
import linguistic_features
from pathlib import Path

import mne
mne.set_log_level('WARNING')

if __name__ == '__main__':
    print('Running: ', __file__)

    with open('config.json') as f:
        config = json.load(f)

    wav_dir = Path(config['wav_dir'])
    words_dir = Path(config['words_dir'])
    phones_dir = Path(config['phones_dir'])
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(exist_ok=True)

    linguistic_features.create(words_dir, phones_dir, out_dir, config=config)
    envelopes.create(wav_dir, out_dir, config=config)
