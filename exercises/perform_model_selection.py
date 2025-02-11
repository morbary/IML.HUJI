from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(-1, 2, n_samples)
    y_no_noise = f(X)
    y_with_noise = y_no_noise + np.random.normal(scale=np.sqrt(noise), size=n_samples)
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.DataFrame(y_with_noise), 2 / 3)

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(x=X, y=y_no_noise, name="True (noiseless) function", mode='lines', marker=dict(color='black')))
    fig1.add_trace(
        go.Scatter(x=X_train[0], y=y_train[0], name="Training function", mode='markers', marker=dict(color='red')))
    fig1.add_trace(
        go.Scatter(x=X_test[0], y=y_test[0], name="Test function", mode='markers', marker=dict(color='blue')))
    fig1.update_layout(title_text=f"True function and generated training and test data <br>"
                                  f"<sub>Number of samples = {n_samples}, Noise level = {noise}</sub>")
    fig1.update_xaxes(title_text="X value")
    fig1.update_yaxes(title_text="y value")
    fig1.show()
    # save_to1 = "../exercises/q1-" + str(noise) + "_noise--" + str(n_samples) + "_samples.png"
    # fig1.write_image(save_to1)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    X_train, y_train, X_test, y_test = np.squeeze(X_train.to_numpy()), np.squeeze(y_train.to_numpy()), \
                                       np.squeeze(X_test.to_numpy()), np.squeeze(y_test.to_numpy())
    k = list(range(0, 11))
    train_mse = []
    validation_mse = []
    for i in k:
        poly_fit = PolynomialFitting(i)
        train_score, validation_score = cross_validate(poly_fit, X_train, y_train,
                                                       mean_square_error, 5)
        train_mse.append(train_score)
        validation_mse.append(validation_score)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=k, y=train_mse, name="Training Error", mode='lines+markers', marker=dict(color='red')))
    fig2.add_trace(
        go.Scatter(x=k, y=validation_mse, name="Validation Error", mode='lines+markers', marker=dict(color='blue')))
    fig2.update_layout(title_text=f"MSE for different polynomial degrees on 5-fold cross-validation<br>"
                                  f"<sub>Number of samples = {n_samples}, Noise level = {noise}</sub>")
    fig2.update_xaxes(title_text="Polynomial Degree (k)")
    fig2.update_yaxes(title_text="Average MSE")
    fig2.show()
    # save_to2 = "../exercises/q2-" + str(noise) + "_noise--" + str(n_samples) + "_samples.png"
    # fig2.write_image(save_to2)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k_val = k[np.argmin(validation_mse)]
    poly_fit = PolynomialFitting(best_k_val)
    poly_fit.fit(X_train, y_train)
    y_pred = poly_fit.predict(X_test)
    test_mse = np.round(mean_square_error(y_test, y_pred), 2)
    print("Number of samples: ", n_samples, "Noise: ", noise)
    print(f"Test MSE for k={best_k_val}: {test_mse}")
    print(f"Validation MSE for k={best_k_val}: {np.round(validation_mse[best_k_val], 2)} \n")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    diabetes_ds = datasets.load_diabetes()
    X = diabetes_ds.data
    y = diabetes_ds.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_vals = np.linspace(0.00001, 2, n_evaluations)
    ridge_train_err, ridge_validation_err = [], []
    lasso_train_err, lasso_validation_err = [], []
    for lam in lam_vals:
        ridge_reg = RidgeRegression(lam)
        lasso_reg = Lasso(lam)
        ridge_train_score, ridge_validation_score = cross_validate(ridge_reg, X_train, y_train, mean_square_error, 5)
        lasso_train_score, lasso_validation_score = cross_validate(lasso_reg, X_train, y_train, mean_square_error, 5)
        ridge_train_err.append(ridge_train_score)
        ridge_validation_err.append(ridge_validation_score)
        lasso_train_err.append(lasso_train_score)
        lasso_validation_err.append(lasso_validation_score)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ridge Regularization", "Lasso Regularization"))

    fig.add_trace(go.Scatter(x=lam_vals, y=ridge_train_err, name="Ridge Training Error", mode='lines+markers', ), row=1,
                  col=1)
    fig.add_trace(go.Scatter(x=lam_vals, y=ridge_validation_err, name="Ridge Validation Error", mode='lines+markers', ),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=lam_vals, y=lasso_train_err, name="Lasso Training Error", mode='lines+markers', ), row=1,
                  col=2)
    fig.add_trace(go.Scatter(x=lam_vals, y=lasso_validation_err, name="Lasso Validation Error", mode='lines+markers', ),
                  row=1, col=2)
    fig.update_xaxes(title_text="Value of Regularization Parameter (λ)")
    fig.update_yaxes(title_text="Average MSE")
    fig.update_layout(height=500, width=1000, title_text=f"MSE on 5-fold cross-validation "
                                                         f"over different regularization parameter (λ) values <br>",
                      margin=dict(t=100))
    fig.show()
    # fig.write_image("../exercises/q7-MSE_ridge_lasso_over_lambda_val.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lam_ridge = lam_vals[np.argmin(ridge_validation_err)]
    print(f"Best Regularization parameter for Ridge regression: {best_lam_ridge}")
    best_lam_lasso = lam_vals[np.argmin(lasso_validation_err)]
    print(f"Best Regularization parameter for Lasso regression: {best_lam_lasso}\n")

    estimator_names = ["Ridge Regression", "Lasso Regression", "Linear Regression"]
    estimator_obj = [RidgeRegression(best_lam_ridge), Lasso(best_lam_lasso), LinearRegression()]
    for i, estimator in enumerate(estimator_obj):
        estimator.fit(X_train, y_train)
        print(f"{estimator_names[i]} Test Error: {mean_square_error(y_test, estimator.predict(X_test))}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(100, 5)  # q1-3
    select_polynomial_degree(100, 0)  # q4
    select_polynomial_degree(1500, 10)  # q5
    select_regularization_parameter()  # q6-8
