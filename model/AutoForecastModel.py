import warnings

from model.RandomSearchModel import RandomSearchARIMA
from model.StepwiseModel import StepwiseModel

warnings.filterwarnings("ignore")


class AutoARIMA():

    def __init__(self):
        super().__init__()
        self.__model = None

    def fit(self, ts, seasonal_period=None, stepwise=True, test_ratio=0.20):
        """
        Fits the model on the passed time series
        :param ts: pandas series containing data
        :param test_ratio: test ratio to use from the passed data
        :param seasonal_period: The seasonal period of the time series
        :return: fitted model
        """
        if(stepwise):
            self.__model = StepwiseModel()
        else:
            self.__model = RandomSearchARIMA()

        self.__model.fit(ts, seasonal_period, test_ratio=test_ratio, exogenous=None)

        return self.__model


    def predict(self, steps, exogenous=None):
        """
        Returns the forecasted values using the trained model
        :param steps: number of steps to forecast
        :return: array of forecasted values
        """
        return self.__model.predict(steps, exogenous=exogenous)

