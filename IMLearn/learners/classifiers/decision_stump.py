from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        split_feature_index = None
        split_thres = None
        split_sign = None

        min_err = 1
        for sign in {-1, 1}:
            thres_mat = np.apply_along_axis(self._find_threshold, 0, X, y, sign)
            min_err_index = np.argmin(thres_mat[1:], axis=1)
            threshold = thres_mat[0, min_err_index]
            curr_min_err = thres_mat[1, min_err_index][0]
            if curr_min_err < min_err:
                min_err = curr_min_err
                split_feature_index = min_err_index[0]
                split_thres = threshold[0]
                split_sign = sign

        self.threshold_ = split_thres
        self.j_ = split_feature_index
        self.sign_ = split_sign

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        n_samples = X.shape[0]  # number of samples
        responses = np.zeros((n_samples,))  # assign vector of zeros
        responses[X[:, self.j_] >= self.threshold_] = self.sign_  # assign 1 on values larger or equal to threshold
        responses[X[:, self.j_] < self.threshold_] = -self.sign_  # assign -1 on values lower than threshold
        return responses

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        thr = None
        thr_err = 1
        n_samples = labels.shape[0]  # number of samples

        # order by values
        sort_index = np.argsort(values)
        values_sorted, labels_sorted = values[sort_index], labels[sort_index]

        for i in range(0, n_samples - 1):
            temp_thres = (values_sorted[i] + values_sorted[i + 1]) / 2

            # assign predicted values
            y_pred = np.zeros((n_samples,))
            y_pred[values_sorted >= temp_thres] = sign  # assign sign on values larger or equal to threshold
            y_pred[values_sorted < temp_thres] = -sign  # assign -sign on values lower than threshold

            # calculate misclassification error
            curr_err = misclassification_error(labels_sorted, y_pred, True)

            if curr_err < thr_err:  # get threshold with minimum error
                thr = temp_thres
                thr_err = curr_err

        return thr, thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_predicted = self._predict(X)
        return misclassification_error(y, y_predicted)
