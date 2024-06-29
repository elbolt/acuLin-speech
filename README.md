# Acoustic and linguistic ecoding analysis pipelines for mTRF models based on natural, continuous speech

## Description

This repository hosts the pipelines for preprocessing electroencephalography (EEG) and speech signal data, alongside the modeling procedure for the multivariate Temporal Response Function (mTRF) models. These models are detailed in our manuscript entitled "Linguistic neural encoding of natural speech is not affected by cognitive decline alone, but decreases with combined hearing impairment", currently in preparation.

The code is shared for transparency purposes. The data—comprising EEG and audio files—cannot (yet) be shared. Note that the code is tailored to our specific environment and data infrastructure; it requires adjustments for use in different setups.

* The required modules for executing our code are listed in the `environment.yml` file.
* All data paths in the configuration file must be adapted to match your own data infrastructure.

## Structure

### Data preprocessing
The EEG and speech preprocessing pipelines are located under `preprocessing`, which contains two directories, `eeg` and `speech`:

* `preprocessing/eeg`: Contains the pipelines used for preprocessing the EEG data for mTRF analysis.
* `preprocessing/speech`: Comprises two subdirectories:
    * `representations`: Includes pipelines for computing linguistic markers (segmentation, word-based, and phoneme-based speech features for the mTRF models) from the output of the forced aligner. Please note that I used a different environment to run these scripts, detailed in `linfeatures.yml`.
    * `features`: Contains pipelines to generate acoustic and linguistic time series as speech features for the mTRF model.

### Boosting

The mTRF models were created using techniques from the Eelbrain Toolbox. For a detailed explanation and methodology, refer to [Brodbeck et al. (2023)](https://doi.org/10.7554/eLife.85012). The `boosting` directory holds all scripts used to set up the models as described in our paper.

### Statistics

This folder contains the R and Python scripts used to run the statistical models and perform the descriptive statistics reported in the main manuscript and supplementary material. These scripts ensure reproducibility and transparency of our data analysis processes.