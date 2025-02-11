from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_, count = np.unique(y, return_counts=True)
        samples = np.size(y)  # number of samples
        classes = self.classes_.shape[0]  # number of classes
        features = X.shape[1]  # number of features

        self.mu_ = np.zeros((classes, features))
        self.vars_ = np.zeros((classes, features))

        for k, label in enumerate(self.classes_):
            x_in_label = X[y == label]
            self.mu_[k] = np.mean(x_in_label, axis=0)
            self.vars_[k] = np.var(x_in_label, axis=0)
        self.pi_ = count / samples

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # get maximum likelihood from X samples and return predictions vector
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        else:
            m = X.shape[0]
            likelihoods = np.zeros((m, self.classes_.size))
            for k in range(self.classes_.size):
                # calculate likelihood for X being in class k
                log_likelihood = -((X - self.mu_[k, :]) ** 2) / (2 * self.vars_[k, :]) + np.log(
                    1 / np.sqrt(self.vars_[k, :] * 2 * np.pi))
                likelihoods[:, k] = np.sum(log_likelihood, axis=1) + np.log(
                    self.pi_[k])  # sum all features for each sample and multiply by prior

            return likelihoods  ## this is not actually the likelihood!

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
        from ...metrics import misclassification_error
        y_predicted = self._predict(X)
        return misclassification_error(y, y_predicted)


