from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
import pandas as pd

from IMLearn import BaseEstimator


def split_k_groups(x, y, k):
    if k > len(x):
        return None
    X_split = np.array_split(x, k)
    y_split = np.array_split(y, k)
    return X_split, y_split


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    X_df = pd.DataFrame(X)
    y_df = pd.DataFrame(y)

    # split X,y into cv disjoint sets (folds)
    X_df_split, y_df_split = split_k_groups(X_df, y_df, cv)

    train_scores = []
    validation_scores = []
    for i in range(cv):  # for all i!=j, j>=k, i>=1
        # use Si as development
        X_dev, y_dev = X_df_split[i], y_df_split[i]

        # train model on S\SiUSj
        X_train = np.squeeze(X_df.drop(X_dev.index).to_numpy())
        y_train = np.squeeze(y_df.drop(y_dev.index).to_numpy())
        estimator.fit(X_train, y_train)
        X_dev, y_dev = np.squeeze(X_dev.to_numpy()), np.squeeze(y_dev.to_numpy())
        # report mean and standard deviation of the k losses
        train_scores.append(scoring(y_train, estimator.predict(X_train)))
        validation_scores.append(scoring(y_dev, estimator.predict(X_dev)))
    return np.mean(train_scores), np.mean(validation_scores)
