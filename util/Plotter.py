import matplotlib.pyplot as plt

def plot_validation_forecast(model_fit, train, forecast, xlabel="date", ylabel="series"):
    """
    Plots forecasting results
    :param model_fit: ARIMA model fit
    :param train: train series
    :param forecast: forecast series
    :param xlabel: xlabel string
    :param ylabel: ylabel string
    :return: None
    """
    fig = plt.figure(facecolor="w", figsize=(13, 8))
    ax = fig.add_subplot(111)

    ax.plot(train.index.values, train, ls="-", c="#0072B2", label="Train Series")
    ax.plot(forecast["dates"].values, forecast["y"], ls="-", c="#696969", label="Validation")
    ax.plot(forecast["dates"].values, forecast["yhat"], ls="-", c="#FFA500", label="Forecast")
    ax.fill_between(forecast["dates"].values, forecast["yhat_lower"], forecast["yhat_upper"], color="#0072B2", alpha=0.3)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc="best")

    order = str(model_fit.specification["order"])
    seasonal_order = model_fit.specification["seasonal_order"]
    if seasonal_order != (0, 0, 0, 0):
        order+= " & " + str(seasonal_order)

    plt.title("Auto Forecast ARIMA "+order)
    plt.show()


def plot_forecast(train, forecast, xlabel="date", ylabel="series"):
    """
    Plots forecasting results
    :param model_fit: ARIMA model fit
    :param train: train series
    :param forecast: forecast series
    :param xlabel: xlabel string
    :param ylabel: ylabel string
    :return: None
    """
    fig = plt.figure(facecolor="w", figsize=(13, 8))
    ax = fig.add_subplot(111)

    ax.plot(train.index.values, train, ls="-", c="#0072B2", label="Train Series")
    ax.plot(forecast["dates"].values, forecast["yhat"], ls="-", c="#FFA500", label="Forecast")
    ax.fill_between(forecast["dates"].values, forecast["yhat_lower"], forecast["yhat_upper"], color="#0072B2", alpha=0.3)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc="best")
    plt.title("Auto Forecast ARIMA")
    plt.show()