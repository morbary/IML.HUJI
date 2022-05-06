import pandas as pd

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import plotly.express as px

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    uni_guassian = UnivariateGaussian()
    # Question 1 - Draw samples and print fitted model
    X = np.random.normal(10, 1, size=1000)
    uni_guassian.fit(X)
    estimation = (uni_guassian.mu_, uni_guassian.var_)
    print(estimation)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)  # generate array of increasing sizes
    true_mu = 10
    estimation_error = []
    for size in ms:  # for sample size
        uni_guassian.fit(X[:size])
        estimation_error.append(np.abs(uni_guassian.mu_ - true_mu))  # calculate estimation error for Mean value

    fig = go.Figure(data=go.Scatter(x=ms, y=estimation_error))
    fig.update_layout(title='(Question 2) Error of Estimation of Mean Value as function of Sample size',
                      xaxis_title='Sample Size',
                      yaxis_title='Estimation Error')
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = uni_guassian.pdf(X)
    fig2 = go.Figure(data=go.Scatter(x=X, y=pdfs, mode='markers'))
    fig2.update_layout(title='(Question 3) PDFs as function of Sample value',
                       xaxis_title='Sample Value',
                       yaxis_title='PDF')
    fig2.show()


def test_multivariate_gaussian():
    multi_guassian = MultivariateGaussian()
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, sigma, 1000)
    multi_guassian.fit(X)
    print(multi_guassian.mu_)
    print(multi_guassian.cov_)

    # Question 5 - Likelihood evaluation
    max_likelihood = float('-inf')
    max_f1 = float('-inf')
    max_f3 = float('-inf')
    expectancies = np.linspace(-10, 10, 200)
    likelihoods = []
    for f1 in expectancies:
        for f3 in expectancies:
            current_mu = np.array([f1, 0, f3, 0])
            current_likelihood = MultivariateGaussian.log_likelihood(current_mu, sigma, X)
            if current_likelihood > max_likelihood:
                max_f1 = f1
                max_f3 = f3
                max_likelihood = current_likelihood
            likelihoods.append((current_likelihood, f1, f3))
    likelihood_df = pd.DataFrame(likelihoods, columns=['log likelihood', 'f1', 'f3'])
    layout = go.Layout(
        title='(Question 5) Multivariate Log Likelihood for Expectations ranging from -10 to 10 given a fixed '
              'covariance Matrix',
        xaxis=dict(title='f3'),
        yaxis=dict(title='f1')
    )
    fig = go.Figure(data=go.Heatmap(x=likelihood_df['f3'], y=likelihood_df['f1'], z=likelihood_df['log likelihood'],
                                    colorbar=dict(title="Log Likelihood")),
                    layout=layout)
    fig.show()

    # Question 6 - Maximum likelihood
    print("maximizing f1 value: ", round(max_f1, 3))
    print("maximizing f3 value: ", round(max_f3, 3))
    print("maximum likelihood calculated: ", round(max_likelihood, 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()


