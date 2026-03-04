Repo for NLP coursework.
Model-related code is in 'BestModel/'.
The trained best model is 'BestModel/best_model.pt'.

NOTE: The best model is not part of the git repo due to its size. To download it into 'BestModel/best_model.pt', run 'uv run download_model.py'.
It must be downloaded first before evaluation.py can run.

The threshold used for inference on official dev and test set that was found during training is stored in 'BestModel/threshold.json'.
This threshold is used during the evaluation.

Run training with: 'uv run BestModel/train.py'
Run evaluation with: 'uv run evaluation.py'
Run data exploration with: 'uv run data-exploration.py'

Predictions are written to 'my-predictions/'.
