import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "browser"
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    ds = pd.read_csv(filename, parse_dates=["Date"])
    # ds.dropna(axis=0)
    ds = ds[ds["Temp"] >= -20]  # clean data of outliers
    ds["DayOfYear"] = ds["Date"].dt.dayofyear
    return ds


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    city_temperature_data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    il_temperature_data = city_temperature_data[city_temperature_data["Country"] == "Israel"]
    il_temperature_data = il_temperature_data.astype({"Year": str})  # change year to string

    fig1 = px.scatter(
        il_temperature_data,
        x="DayOfYear",
        y="Temp",
        color="Year",
        title="Average daily temperature in Israel as a function of Day of the Year")
    # f.show()
    fig1.write_image("../exercises/poly_fit_q2_pt1.png")

    israel_temp_by_month_std = il_temperature_data.groupby("Month").agg({"Temp": np.std})
    israel_temp_by_month_std.reset_index(inplace=True)
    israel_temp_by_month_std = israel_temp_by_month_std.rename(columns={"Temp": "Temperature Standard Deviation"})
    fig2 = px.bar(israel_temp_by_month_std, x="Month", y="Temperature Standard Deviation",
                  title="Standard deviation of Monthly Temperature in Israel")
    fig2.update_xaxes(tickmode="linear")
    # f2.show()
    fig2.write_image("../exercises/poly_fit_q2_pt2.png")

    # Question 3 - Exploring differences between countries
    diff_between_countries = city_temperature_data.groupby(["Country", "Month"]).agg(Mean=("Temp", np.mean),
                                                                                     Std=("Temp", np.std))
    diff_between_countries.reset_index(inplace=True)
    fig3 = px.line(diff_between_countries, x="Month", y="Mean", error_y="Std", color="Country",
                   title="Average Monthly temperature")
    fig3.update_xaxes(tickmode="linear")
    # f3.show()
    fig3.write_image("../exercises/poly_fit_q3.png")

    # Question 4 - Fitting model for different values of `k`
    train_DayOfYear, train_Temp, test_DayOfYear, test_Temp = split_train_test(il_temperature_data["DayOfYear"],
                                                                              il_temperature_data["Temp"], 0.75)
    il_loss_data = pd.DataFrame(columns=["k", "Loss"])
    for k in range(1, 11):
        pf = PolynomialFitting(k)
        pf.fit(train_DayOfYear.to_numpy(), train_Temp.to_numpy())
        loss = pf.loss(test_DayOfYear.to_numpy(), test_Temp.to_numpy())

        new_row = {"k": k,
                   "Loss": round(loss, 2)}
        il_loss_data = il_loss_data.append(new_row, ignore_index=True)

    print(il_loss_data)
    fig4 = px.bar(il_loss_data, x="k", y="Loss",
                  title="Test Error as a function of k")
    fig4.update_xaxes(tickmode="linear")
    fig4.write_image("../exercises/poly_fit_q4.png")

    # Question 5 - Evaluating fitted model on different countries
    poly_fit_5 = PolynomialFitting(5)
    poly_fit_5.fit(il_temperature_data["DayOfYear"], il_temperature_data["Temp"])

    other_countries_loss_data = pd.DataFrame(columns=["Country", "Loss"])
    for country in ["Jordan", "South Africa", "The Netherlands"]:
        other_country_data = city_temperature_data[city_temperature_data["Country"] == country]
        other_country_loss = poly_fit_5.loss(other_country_data["DayOfYear"], other_country_data["Temp"])
        new_row = {"Country": country,
                   "Loss": round(other_country_loss, 2)}
        other_countries_loss_data = other_countries_loss_data.append(new_row, ignore_index=True)

    fig5 = px.bar(other_countries_loss_data, x="Country", y="Loss",
                  title="Loss of Temperature Prediction Per Country via Israel Training")
    fig5.write_image("../exercises/poly_fit_q5.png")
