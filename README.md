This repository contains the implementation of a soft-voting ensemble for detecting Patronising and Condescending Language (PCL).

## Repository Structure
- `dev.txt`: Model predictions on the official dev set (2094 lines).
- `test.txt`: Model predictions on the official test set (3832 lines).
- `BestModel/`: Contains the core implementation scripts.
  - `train.py`: Script for HPO via Optuna, LLRD implementation, and training the ensemble members.
  - `ensemble_inference.py`: Logic for loading the committee and performing soft-voting.

## Model Weights (External Link)
Due to file size constraints, the trained weights for the three ensemble members are hosted on OneDrive:
https://imperiallondon-my.sharepoint.com/:f:/g/personal/aam223_ic_ac_uk/IgB2OEFmfTB5TYOCc7J8vX-2ARbJTPKiV26f5fDl5p8aUos?e=i2gEe5
