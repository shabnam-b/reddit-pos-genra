
This page contains details of the experimental setup for the following paper:

Behzad, Shabnam and Zeldes, Amir (2020) "[A Cross-Genre Ensemble Approach to Robust Reddit Part of Speech Tagging](http://arxiv.org/abs/2004.14312)".
In: Proceedings of the 12th Web as Corpus Workshop (WAC-XII).

## Flair

We trained Flair on different genres and used the prediction as features for our ensemble model. [Scripts/flair_train_predict.py](Scripts/flair_train_predict.py) can be used to train new models and then predict the POS tags. For more information on Flair, please visit [Flair](https://github.com/flairNLP/flair).

## XGBoost Ensemble

[Scripts/ensemble.py](Scripts/ensemble.py) contains the code for training and testing the XGBoost model. In our paper, we used Flair predictions and also named entities as features. The parameter used are based on parameter tuning with random search on the dev set.

## Data Sample

[data](data/) contains a very small sample data for both scripts, just to make the formatting clear.

## References

If you use these models in your research, please kindly cite the following paper: 

```
@InProceedings{BehzadZeldes2020,
  author    = {Shabnam Behzad and Amir Zeldes},
  title     = {A Cross-Genre Ensemble Approach to Robust {R}eddit Part of Speech Tagging},
  booktitle   = {Proceedings of the 12th Web as Corpus Workshop (WAC-XII)},
  year      = {2020},
}
```
If you use the Flair package, please cite as described [here](https://github.com/flairNLP/flair).

