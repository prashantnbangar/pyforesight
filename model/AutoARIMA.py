import json
import os
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import mse
import random
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, 'parameters.json')) as f:
    parameters = json.load(f)


class AutoARIMA():

    def __init__(self):
        super().__init__()
        print(parameters)
        self.__seasonal = False
        self.__seasonal_period = None
        self.__best_model = None
        self.__best_model_fit = None

    def fit(self, ts, test_ratio=0.25, seasonal_period=None):
        """
        Fits the model on the passed time series
        :param ts: pandas series containing data
        :param test_ratio: test ratio to use from the passed data
        :param seasonal_period: The seasonal period of the time series
        :return: fitted model
        """
        if seasonal_period is None:
            self.__seasonal = False
        else:
            self.__seasonal = True
            self.__seasonal_period = seasonal_period
        self.__best_model, self.__best_model_fit = self.__evaluate_models(ts, test_ratio=test_ratio)
        return self.__best_model_fit

    def predict(self, steps):
        """
        Returns the forecasted values using the trained model
        :param steps: number of steps to forecast
        :return: array of forecasted values
        """
        if not self.__seasonal:
            return self.__best_model_fit.forecast(steps)[0]
        else:
            return self.__best_model_fit.forecast(steps)

    def __evaluate_models(self, ts, test_ratio, max_iterations=20):
        """
        Finds the best model for all combinations for parameters based on Mean Squared Error
        :param ts: time series
        :param test_ratio: test ratio to use
        :param max_iterations: max number of parameter combinations to test
        :return:
        """
        ts = ts.astype('float32')
        best_score, best_order, best_seasonal_order, best_model, best_model_fit = float("inf"), None, None, None, None
        order, seasonal_order = self.__generate_parameter_grid()
        for i in range(max_iterations):
            order_i = order[random.randint(0, len(order)-1)]
            seasonal_order_i = seasonal_order[random.randint(0, len(seasonal_order)-1)] if self.__seasonal else None
            try:
                score, model, model_fit = self.__evaluate_arima_model(ts, order_i, seasonal_order_i, test_ratio)
                if score < best_score:
                    best_score, best_order, best_seasonal_order, best_model, best_model_fit = \
                        score, order_i, seasonal_order_i, model, model_fit
                print('ARIMA%s %s MSE=%.3f' % (order_i, seasonal_order_i, score))
            except Exception as e:
                print(e)
        print('Best ARIMA%s %s MSE=%.3f' % (best_order, best_seasonal_order, best_score))
        print("Retraining model on entire dataset")
        best_model, best_model_fit = self.__train_model(ts, best_order, best_seasonal_order)
        return best_model, best_model_fit

    def __evaluate_arima_model(self, ts, order, seasonal_order, test_ratio):
        """
        Evaluates the ARIMA model with given order and seasonal order
        :param ts: time series
        :param order: (p,d,q)
        :param seasonal_order: (P,D,Q,m)
        :param test_ratio: test ratio to use
        :return: error, best_model, best_model_fit
        """
        # prepare training dataset
        train_size = int(len(ts) * (1-test_ratio))
        train, test = ts[0:train_size], ts[train_size:]

        model, model_fit = self.__train_model(train, order, seasonal_order)
        if not self.__seasonal:
            yhat = model_fit.forecast(len(test))[0]
        else:
            yhat = model_fit.forecast(len(test))

        # calculate out of sample error
        error = mse(test, yhat)
        return error, model, model_fit

    def __train_model(self, series, order, seasonal_order, max_iter=20):
        """
        Trains the ARIMA model and returns the model fit
        :param series: series to train model on
        :param order: (p,d,q)
        :param seasonal_order: (P,D,Q,m)
        :param max_iter: maximum iterations
        :return:
        """
        if not self.__seasonal:
            model = ARIMA(series, order=order)
            model_fit = model.fit(disp=0, maxiter=max_iter)
        else:
            model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                            enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=0, maxiter=max_iter)
        return model, model_fit

    def __generate_parameter_grid(self):
        """
        Generates the Parameter grid for the ARIMA models
        :return:
        """
        order = []
        seasonal_order = None
        grid = ParameterGrid({
            "p": range(parameters["min_p"], parameters["max_p"], 1),
            "q": range(parameters["min_q"], parameters["max_q"], 1),
            "d": range(parameters["min_d"], parameters["max_d"], 1)
        })
        for counter in range(len(grid)):
            param = grid[counter]
            order.append((param["p"], param["d"], param["q"]))

        if self.__seasonal:
            seasonal_order = []
            seasonal_grid = ParameterGrid({
                "P": range(parameters["min_P"], parameters["max_P"], 1),
                "Q": range(parameters["min_Q"], parameters["max_Q"], 1),
                "D": range(parameters["min_D"], parameters["max_D"], 1)
            })
            for counter in range(len(seasonal_grid)):
                param = seasonal_grid[counter]
                seasonal_order.append((param["P"], param["D"], param["Q"], self.__seasonal_period))

        return order, seasonal_order

