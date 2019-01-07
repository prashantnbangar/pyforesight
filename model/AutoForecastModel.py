import warnings

from model.IterativeModel import RandomSearchARIMA
from model.StepwiseModel import StepwiseModel
from model.SeasonalEstimator import SeasonalEstimator
import pandas as pd

warnings.filterwarnings("ignore")


class AutoARIMA():

    def __init__(self):
        super().__init__()
        self.__model = None
        self.__seasonal_estimator = SeasonalEstimator()
        self.__seasonal_as_exogenous = False

    def fit(self, ts, seasonal_period=None, stepwise=True, test_ratio=0.20):
        """
        Fits the model on the passed time series
        :param ts: pandas series containing data
        :param test_ratio: test ratio to use from the passed data
        :param seasonal_period: The seasonal period of the time series
        :return: fitted model
        """

        seasonal_period, seasonal_series = self.__seasonal_estimator.decompose(ts, period=seasonal_period)
        if(stepwise):
            self.__model = StepwiseModel()
        else:
            self.__model = RandomSearchARIMA()

        # For longer seasonalities, treat seasonal component as an exogenous feature
        if seasonal_period > 200:
            seasonal_period = 0
            self.__seasonal_as_exogenous = True
            exogenous = seasonal_series
        else:
            exogenous = None

        self.__model.fit(ts, seasonal_period, test_ratio=test_ratio, exogenous=exogenous)

        return self.__model

    def predict(self, steps):
        """
        Returns the forecasted values using the trained model
        :param steps: number of steps to forecast
        :return: array of forecasted values
        """
        if self.__seasonal_as_exogenous:
            seasonal_series = self.__seasonal_estimator.predict_seasonal_series(steps=steps)
        else:seasonal_series = None
        return self.__model.predict(steps, exogenous=seasonal_series)

