Repo for NLP coursework.
Model-related code is in 'BestModel/'.
The trained best model is 'BestModel/best_model.pt'.
The threshold used for inference on official dev and test set that was found during training is stored in 'BestModel/threshold.json'.
This threshold is used during the evaluation.
This model is stored in git LFS due to its size but should be downloaded automatically when cloning this repo.

Run training with: 'uv run BestModel/train.py'
Run evaluation with: 'uv run evaluation.py'
Run data exploration with: 'uv run data-exploration.py'

Predictions are written to 'my-predictions/'.