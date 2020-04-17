
This page contains details of experimental setup for the following paper:

A Cross-Genre Ensemble Approach to Robust Reddit Part of Speech Tagging

## Flair

We trained Flair on different genres and used the prediction as features for our ensemble model. [Scripts/flair_train_predict.py](Scripts/flair_train_predict.py) can be used to train new models and then predict the POS tags. For more information on Flair, please visit [Flair](https://github.com/flairNLP/flair).

## XGBoost Ensemble

[Scripts/ensemble.py](Scripts/ensemble.py) contains the code for training and testing the XGBoost model. In our paper, we used Flair predictions and also named entities as features. The parameter used are based on parameter tuning with random search on the dev set.

## Data Sample

[data](data/) contains a very small sample data for both scripts, just to make the formatting clear.
