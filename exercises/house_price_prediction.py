from datetime import datetime

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # custom_date_parser = lambda x: datetime.strptime(x, "%Y%m%d%H")
    # ds = pd.read_csv(filename, parse_dates=["date"])
    ds = pd.read_csv(filename)
    ds["date"] = ds["date"].str.slice(0, 8)
    ds["date"] = pd.to_datetime(ds["date"], format="%Y%m%d", errors="coerce")
    ds = ds[ds['date'].notna()]
    ds['Year Sold'] = pd.DatetimeIndex(ds['date'], yearfirst=True).year
    ds['Month Sold'] = pd.DatetimeIndex(ds['date']).month
    ds = ds.drop(["id", "lat", "long","date","zipcode"], axis=1)
    for col in ["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
                "yr_built", "sqft_living15", "sqft_lot15"]:
        ds = ds[ds[col] > 0]
    for col in ["yr_renovated", "sqft_above", "sqft_basement"]:
        ds = ds[ds[col] >= 0]

    ds = ds[(ds["waterfront"] == 0) | (ds["waterfront"] == 1)]  ## change to yes or no
    ds = ds[(ds["view"] >= 0) & (ds["view"] <= 4)]
    ds = ds[(ds["condition"] >= 1) & (ds["condition"] <= 5)]

    response = ds["price"]
    response = response.rename('Price of Property')

    # ds = ds.drop(["date", "zipcode"], axis=1)  ##check how to numerize

    x = ds.drop(["price"], axis=1)

    x.rename(columns={'bedrooms': 'Number of Bedrooms',
                      'bathrooms': 'Number of Bathrooms',
                      'sqft_living': 'Size of living area in square feet',
                      'sqft_lot': 'Size of the lot in square feet',
                      'floors': 'Number of floors',
                      'waterfront': 'Waterfront property',
                      'view': 'Quality of the View from the Property',
                      'condition': 'Condition of the Property',
                      'grade': 'Construction Quality Classification',
                      'sqft_above': 'Square feet above ground',
                      'sqft_basement': 'Square feet below ground',
                      'yr_built': 'Year built',
                      'yr_renovated': 'Year renovated',
                      # 'zipcode': 'Zipcode',
                      'sqft_living15': 'Average size of interior housing living space for the closest '
                                       '15 houses, in square feet',
                      'sqft_lot15': 'Average size of land lots for the closest 15 houses, in square feet',
                      }, inplace=True)

    return x, response


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    # compute cov vector for each column in X with y
    # date = X["date"]
    # X = X.drop(["date"], axis=1)
    cov_x_y = np.dot(y.T - y.mean(), X - X.mean(axis=0)) / (y.shape[0])  # cov(x,y)= 1/n * sum[(xi-E[x])(yi-E[y])]
    # cov_date_y = np.dot(y.T-y.mean(), (date.T-date.T.mean()).astype(int))/ (y.shape[0])
    X_std = np.std(X, axis=0)  # compute standard deviation of each column in X
    y_std = np.std(y)  # compute standard deviation of y
    pearson_corr = cov_x_y / (X_std * y_std)  # compute pearson correlation

    response = y.name

    for (feature_name, feature_data) in X.iteritems():
        # print("CREATING PLOT: ", feature_name)
        index = X.columns.get_loc(feature_name)
        f = go.Figure(data=go.Scatter(x=feature_data.values, y=y, mode='markers'))
        f.update_layout(title=f'Correlation between {feature_name} and {response} <br> '
                              f'Pearson Correlation = {pearson_corr[index]}',
                        title_font_size=16,
                        xaxis_title=feature_name,
                        yaxis_title=response)
        save_to = output_path + "/" + feature_name + ".png"
        f.write_image(save_to)
        # f.show()
        # print("Pearson Correlation: \n", pearson_corr[index])


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "../exercises")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percentages = np.linspace(0.1, 1, 91)  # generate array of increasing percentages
    loss_data = pd.DataFrame(columns=["Percentage of Training Set",
                                      "Average Loss",
                                      "Loss Variance"])
    regression_estimator = LinearRegression(include_intercept=True)
    for p in percentages:
        loss_values = []
        for i in range(10):  # repeat 10 times for each p%
            #   1) Sample p% of the overall training data
            p_sample_X = train_X.sample(frac=p)
            p_sample_y = y.loc[p_sample_X.index]

            #   2) Fit linear model (including intercept) over sampled set
            regression_estimator.fit(p_sample_X.to_numpy(), p_sample_y)

            #   3) Test fitted model over test set
            loss = regression_estimator.loss(test_X.to_numpy(), test_y.to_numpy())
            loss_values.append(loss)

        #   4) Store average and variance of loss over test set
        loss_average_per_p = np.sum(loss_values) / 10
        loss_variance_per_p = np.var(loss_values)
        new_row = {"Percentage of Training Set": p * 100,
                   "Average Loss": loss_average_per_p,
                   "Loss Variance": loss_variance_per_p}
        loss_data = loss_data.append(new_row, ignore_index=True)

    loss_data["2*std(loss)"] = (2 * np.sqrt(loss_data["Loss Variance"]))
    fig = go.Figure([
        go.Scatter(
            name="Mean Loss",
            x=loss_data["Percentage of Training Set"],
            y=loss_data["Average Loss"],
            mode="markers+lines"),
        go.Scatter(
            name="upper bound",
            x=loss_data["Percentage of Training Set"],
            y=loss_data["Average Loss"] + loss_data["2*std(loss)"],
            mode="lines",
            # marker=dict(color="#444"),
            line=dict(color="lightgrey"),
            showlegend=False),
        go.Scatter(
            name="lower bound",
            x=loss_data["Percentage of Training Set"],
            y=loss_data["Average Loss"] - loss_data["2*std(loss)"],
            mode="lines",
            # marker=dict(color="#444"),
            line=dict(color="lightgrey"),
            fillcolor='rgba(68, 68, 68, 0.1)',
            fill='tonexty',
            showlegend=False)
    ])
    fig.update_layout(title='Mean Loss as a function of Training Set Percentage',
                      xaxis_title='Percentage of Training Set',
                      yaxis_title='Mean Loss')
    # fig.show()
    fig.write_image("../ex2/linear_reg_q4.png")
