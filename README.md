# Mic-E-Mouse: your computer mouse has big ears
![Mic-E-Mouse Visualization](./fig/micemouse.png)
image courtesy of GPT4/Dall-E-3, generated using the keywords "computer mouse with big ears and a microphone as a scroll whell"
## Description

Side-Channel Audio Attack through computer mice.

## Requirements
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/en/latest/index.html)
- A computer mouse with a high polling rate (4kHz < ), and a high DPI count (16,000 <).
## Installation
The conda/mamba environment can be installed with the following command: `conda env create -f environment.yml`
## Repository Structure
- waveformReconstruction_AUDIOMNIST.ipynb: Waveform Reconstruction Experiments for AudioMNIST, including collecting the samples used in the user study
- waveformReconstruction.ipynb: Waveform Reconstruction Experiments for VCTK
- models/: Directory containing all models trained and made public
- fig/: Directory containing all figures and visuals used in the paper and this repo
- core/nn/: Methods used for Neural Networks, dataloaders, network primitives, and utilities
- core/signal/: Methods used for signal processing
- src/: Mouse_Logger and OpenBlok modified sources
- utils/: HTML test to check if browsers can record audio
- vctk/: utils used to automate dataset creation
## DATA
Anonymous Data available [HERE](https://drive.google.com/drive/folders/1DcTldouupfp7BMteE1Br0lq7RCdQQ0Hc?usp=drive_link)

https://drive.google.com/drive/folders/1DcTldouupfp7BMteE1Br0lq7RCdQQ0Hc?usp=drive_link

## Authors
### KEPT ANONYMOUS WHILE UNDER SUBMISSION
## Citation
### KEPT ANONYMOUS WHILE UNDER SUBMISSION