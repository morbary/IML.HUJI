import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(lambda: DecisionStump(), n_learners)
    adaboost.fit(train_X, train_y)
    train_loss = []
    test_loss = []
    for j in range(1, n_learners + 1):
        train_loss.append(adaboost.partial_loss(train_X, train_y, j))
        test_loss.append(adaboost.partial_loss(test_X, test_y, j))

    num_learners = list(range(1, n_learners))
    fig = go.Figure()
    fig.add_traces(
        [go.Scatter(x=num_learners, y=train_loss, name="Training Error", mode="lines",
                    line=dict(color="pink")),
         go.Scatter(x=num_learners, y=test_loss, name="Testing Error", mode="lines",
                    line=dict(color="green"))])
    fig.update_layout(
        title=f'<b>AdaBoost: Performance Error on Data with Noise = {noise}</b>',
        xaxis_title="Number of Learners",
        yaxis_title="Error")
    fig.show()
    fig.write_image(f"../exercises/q1-error_as_function_of_learners_noise_{noise}.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[f"<sup><b>{t} learners<b></sup>" for t in T])
    for i, t in enumerate(T):
        fig2.add_traces([decision_surface(lambda x: adaboost.partial_predict(x, t), lims[0], lims[1], showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))],
                        rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig2.update_layout(title=f"<b>AdaBoost: Decision Boundaries on Test Set with Noise = {noise}</b>",
                       margin=dict(t=100),
                       yaxis1_range=[-1, 1], yaxis2_range=[-1, 1], yaxis3_range=[-1, 1], yaxis4_range=[-1, 1],
                       xaxis1_range=[-1, 1], xaxis2_range=[-1, 1], xaxis3_range=[-1, 1], xaxis4_range=[-1, 1]) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig2.show()
    fig2.write_image(f"../exercises/q2-decision_boundaries_noise_{noise}.png")

    # Question 3: Decision surface of best performing ensemble
    min_loss = 1
    best_num_of_learners = 1
    for z in list(range(1, 251)):
        curr_loss = adaboost.partial_loss(test_X, test_y, z)
        if curr_loss < min_loss:
            min_loss = curr_loss
            best_num_of_learners = z

    acc = accuracy(test_y, adaboost.predict(test_X))

    fig3 = go.Figure([decision_surface(lambda x: adaboost.partial_predict(x, t), lims[0], lims[1], showscale=False),
                      go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', showlegend=False,
                                 marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black", width=1)))])
    fig3.update_layout(height=600, width=900,
        title=f'<b>AdaBoost: Decision Surface of Ensemble with Lowest Error on Test set with Noise = {noise}</b><br>'
              f'<sup><b>Ensemble size = {best_num_of_learners}, Accuracy = {acc}</b></sup>')
    fig3.update_xaxes(range=[-1, 1], visible=False)
    fig3.update_yaxes(range=[-1, 1], visible=False)
    fig3.show()
    fig3.write_image(f"../exercises/q3-decision_surface_noise_{noise}.png")

    # Question 4: Decision surface with weighted samples
    normalized_D = adaboost.D_ / np.max(adaboost.D_) * 5
    fig4 = go.Figure([decision_surface(lambda x: adaboost.predict(x), lims[0], lims[1], showscale=False),
                      go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', showlegend=False,
                                 marker=dict(color=train_y, size=normalized_D, colorscale=[custom[0], custom[-1]],
                                             line=dict(color="black", width=1)))],
                     layout=go.Layout(height=600, width=900,
                         title=f"<b>Decision Surface of full Ensemble on Training set on Data with Noise = {noise}</b><br>"
                               f"<sup><b>Marker size proportional to sample weights</b></sup>"))
    fig4.update_xaxes(visible=False).update_yaxes(visible=False)
    fig4.update_xaxes(range=[-1, 1], visible=False)
    fig4.update_yaxes(range=[-1, 1], visible=False)
    fig4.show()
    fig4.write_image(f"../exercises/q4-decision_surface_w_distributions_noise_{noise}.png")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
