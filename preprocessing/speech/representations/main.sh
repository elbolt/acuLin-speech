#!/bin/bash

#####################################################
# Run speech representation extraction scripts      #
#####################################################

ENV_NAME="linfeatures"

# Activate conda environment "environment"
source activate $ENV_NAME
echo "Conda environment $ENV_NAME activated."
# which python

# Run speech representations extraction pipelines
echo "Extract word representations from speech segments"
python get_word_representations.py
echo "Extract phonetic representations from speech segments"
python get_phonetic_representations.py

echo "Complete."
conda deactivate
