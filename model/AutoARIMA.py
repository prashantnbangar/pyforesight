import json
import os
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import mse
import random
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
        if seasonal_period == None:
            self.__seasonal = False
        else:
            self.__seasonal = True
        self.__best_model, best_model_fit = self.__evaluate_models(ts, test_ratio=test_ratio)
        self.__best_model_fit = self.__best_model.fit(ts)
        return best_model_fit

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
                    best_score, best_order, best_seasonal_order, model, model_fit = \
                        score, order_i, seasonal_order_i, model, model_fit
                print('ARIMA%s %s MSE=%.3f' % (order_i, seasonal_order_i, score))
            except Exception as e:
                print(e)
        print('Best ARIMA%s %s MSE=%.3f' % (best_order, best_seasonal_order, best_score))
        return best_model, best_model_fit

    def __evaluate_arima_model(self, ts, order, seasonal_order, test_ratio):
        """
        Evaluates the ARIMA model with given order and seasonal order
        :param ts: time series
        :param order: (p,d,q)
        :param seasonal_order: (P,D,Q)
        :param test_ratio: test ratio to use
        :return: error, best_model, best_model_fit
        """
        # prepare training dataset
        train_size = int(len(ts) * (1-test_ratio))
        train, test = ts[0:train_size], ts[train_size:]
        history = [x for x in train]
        # make predictions
        predictions = list()
        error=0
        model=None
        model_fit=None
        if not self.__seasonal:
            for t in range(len(test)):
                model = ARIMA(history, order=order)
                model_fit = model.fit(disp=0, maxiter=50)
                yhat = model_fit.forecast()[0]
                predictions.append(yhat)
                history.append(test[t])
                # calculate out of sample error
                error = mse(test, predictions)
        else:
            for t in range(len(test)):
                model = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                model_fit = model.fit()
                yhat = model_fit.forecast()
                predictions.append(yhat)
                history.append(test[t])
                # calculate out of sample error
                error = mse(test, predictions)
        return error, model, model_fit

    def __generate_parameter_grid(self, max_parameter_combinations=50):
        """
        Generates the Parameter grid for the ARIMA models
        :param max_parameter_combinations: max parameter combinations to return
        :return:
        """
        order = []
        seasonal_order = None
        seasonal_order=None
        for i in range(max_parameter_combinations):
            p = random.randint(parameters["min_p"], parameters["max_p"])
            q = random.randint(parameters["min_q"], parameters["max_q"])
            d = random.randint(parameters["min_d"], parameters["max_d"])
            order.append((p,d,q))

        if self.__seasonal:
            seasonal_order = []
            for i in range(max_parameter_combinations):
                P = random.randint(parameters["min_P"], parameters["max_P"])
                Q = random.randint(parameters["min_Q"], parameters["max_Q"])
                D = random.randint(parameters["min_D"], parameters["max_D"])
                seasonal_order.append((P, D, Q))

        return order, seasonal_order