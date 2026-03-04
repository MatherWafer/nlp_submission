This repository contains the implementation of a soft-voting ensemble for detecting Patronising and Condescending Language (PCL).

## Repository Structure
- `dev.txt`: Model predictions on the official dev set (2094 lines).
- `test.txt`: Model predictions on the official test set (3832 lines).
- `BestModel/`: Contains the core implementation scripts.
  - `train.py`: Script for HPO via Optuna, LLRD implementation, and training the ensemble members.
  - `ensemble_inference.py`: Logic for loading the committee and performing soft-voting.

## Model Weights (External Link)
Due to file size constraints, the trained weights for the three ensemble members are hosted on OneDrive:
https://imperiallondon-my.sharepoint.com/:f:/r/personal/aam223_ic_ac_uk/Documents/nlp_ensemble_models?csf=1&web=1&e=NLE8Sj
