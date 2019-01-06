import os
import json

from numpy.linalg import LinAlgError
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
import util.Plotter as Plotter

from util import Statistician
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

script_dir = os.path.dirname(__file__)
with open(os.path.join(script_dir, 'parameters.json')) as f:
    parameters = json.load(f)


class StepwiseModel():
    def __init__(self) -> None:
        super().__init__()
        self.__seasonal = False
        self.__seasonal_period = None
        self.__best_model = None
        self.__best_model_fit = None
        self.__drift = None

    def fit(self, ts, seasonal_period, exogenous=None, test_ratio=0.2):
        """
        Fits the step wise ARIMA model on the passed data
        :param ts: time series
        :param seasonal_period: seasonal span of the series if seasonal, else -1
        :param exogenous: exogenous variables if any
        :param test_ratio: test ratio to use
        :return: model, model_fit, score
        """
        self.__seasonal = True if seasonal_period != -1 else False
        self.__seasonal_period = seasonal_period

        # prepare training dataset
        train_size = int(len(ts) * (1 - test_ratio))
        train, test = ts[:train_size], ts[train_size:]

        if exogenous is not None:
            exogenous_train, exogenous_test = exogenous[:train_size], exogenous[train_size:]
        else:
            exogenous_train, exogenous_test = None, None

        self.__best_model_fit, self.__best_model = self.__train_stepwise(train, exogenous=exogenous_train)

        test_forecasts = self.predict(steps=len(test), exogenous=exogenous_test)
        rmse_score = rmse(test, test_forecasts["yhat"])
        model_order = self.__best_model_fit.specification["order"]
        seasonal_order = self.__best_model_fit.specification["seasonal_order"] if self.__seasonal else None
        drift = "" if self.__best_model_fit.specification["trend"] == "n" else "with drift"
        print('Best ARIMA%s %s %s MSE=%.3f' % (model_order, seasonal_order, drift, rmse_score))

        print("Retraining model on entire dataset")
        self.__best_model, self.__best_model_fit, score = self.__train_model(ts, model_order, seasonal_order)

        test_forecasts["y"] = test
        Plotter.plot_forecast(self.__best_model_fit, train, test_forecasts)
        return self.__best_model, self.__best_model_fit, rmse_score


    def predict(self, steps, exogenous=None):
        """
        Makes forecasts useing the fit model.
        :param steps: No of steps to make forecast
        :param exogenous: exogenous variables for foreasting
        :return: forecasted dataframe containing predicitons, confidence interval and index
        """
        start = self.__best_model_fit.nobs
        end = start + steps - 1

        predictions = self.__best_model_fit.get_prediction(start=start, end=end, exog=exogenous)
        confidence = predictions.conf_int()
        forecast = pd.DataFrame({
            "dates" : confidence.index,
            "yhat" : predictions.predicted_mean,
            "yhat_lower" : confidence.iloc[:, 0],
            "yhat_upper" : confidence.iloc[:, 1]
        })
        return forecast

    def __train_stepwise(self, ts, exogenous):
        """
        Trains the ARIMA model using the step wise algorithms to find the optimal parameters
        :param ts: time series
        :param exogenous: exogenous variables if any
        :return: fit, model
        """
        # Get the initial best model
        fit, model, score = self.__initialize_parameters(ts, exogenous=exogenous)

        best_score = score
        best_model = model
        best_fit = fit
        while True:
            p, d, q = best_fit.specification["order"]
            P, D, Q, m = best_fit.specification["seasonal_order"]
            parameter_space = self.__create_parameter_search_space(p, q, P, Q)
            count = 0
            for p_t, q_t, P_t, Q_t in parameter_space:
                if self.__seen[p_t, q_t, P_t, Q_t] == 1:
                    continue
                count+=1
                model_t, fit_t, score_t = self.__train_model(ts, order=(p_t, d, q_t),
                                                             seasonal_order=(P_t, D, Q_t, self.__seasonal_period),
                                                             exogenous=exogenous)
                if score_t is not None and score_t < best_score:
                    best_model = model_t
                    best_fit = fit_t
                    best_score = score_t
                    break
                self.__seen[p_t, q_t, P_t, Q_t] = 1
                if count >= 13:
                    break
            if score == best_score:
                break
            else:
                fit = best_fit
                model = best_model
                score = best_score

        return best_fit, best_model

    def __initialize_parameters(self, ts, exogenous=None):
        """
        Estimates the ARIMA parameters to start the stepwise model building process
        """

        # Identify seasonal order of differencing and define a basic parameter grid
        if not self.__seasonal:
            D = 0
            start_paramters = [(0, 1, 0, 0), (1, 0, 0, 0), (2, 2, 0, 0)]
        else:
            D = Statistician.ocsb(ts, self.__seasonal_period)
            start_paramters = [(0, 1, 0, 1), (1, 0, 1, 0), (2, 2, 1, 1)]

        # Identify the order of differencing
        if D == 0:
            test = ts
        else:
            test = ts.diff(self.__seasonal_period)[self.__seasonal_period:]

        d = Statistician.kpss(test)
        if d == 1 and D == 0:
            d = d + Statistician.kpss(test.diff()[1:])

        # Define a seen flag array to avoid re-evaluating same parameter model
        self.__seen = np.zeros((parameters["max_p"] + 1, parameters["max_q"] + 1, (parameters["max_P"] * self.__seasonal)+1,
                        (parameters["max_Q"] * self.__seasonal) + 1 + 1))

        # Evaluate a basic model with no p, d, P, D and drift terms
        model, fit, score = self.__train_model(ts, (0, d, 0), (0, D, 0, self.__seasonal_period), exogenous=exogenous)
        self.__seen[0, 0, 0, 0] = 1

        # Having more than 3 orders of differencing is not recommended
        # Initialize drift
        self.__drift = 0 if d+D >= 2 else 1

        # Identify the best model in the initial parameter grid
        for p, q, P, Q in start_paramters:
            model_x, fit_x, score_x = self.__train_model(ts, (p, d, q), (P, D, Q, self.__seasonal_period), exogenous=exogenous)
            self.__seen[p, q, P, Q] = 1
            if score_x is None or score_x < score:
                model = model_x
                fit = fit_x
                score = score_x

        return fit, model, score

    def __train_model(self, series, order, seasonal_order, exogenous=None, max_iterations=50):
        """
        Trains the ARIMA family of model and returns the best model and fit
        :param series: time series to train model on
        :param order: (p, d, q)
        :param seasonal_order: (P, D, Q, m)
        :param exogenous: exogenous variables array
        :return: model, fit, score
        """
        model, fit, score = None, None, None
        try:
            if not self.__seasonal:
                method="css-mle"
                model = ARIMA(series, exog=exogenous, order=order)
            else:
                method="lbfgs"
                trend = "n" if self.__drift == 0 else "c"
                model = SARIMAX(series, exog=exogenous, order=order, seasonal_order=seasonal_order, trend=trend,
                                enforce_stationarity=True)
            fit = model.fit(method=method, solver="lbfgs", maxiter=max_iterations, disp=0)
            score = fit.aic
            print("Order : " + str(order), ", Seasonal Order : " + str(seasonal_order) + ", AIC Score : " + str(score))
        except (ValueError, LinAlgError) as error:
            model, fit, score = None, None, None
            print(error)

        return model, fit, score

    def __create_parameter_search_space(self, p, q, P, Q, allow_drift=0):
        """Creates the ARIMA p, d, P, D parameter search space using the passed initial values identified earlier
        """
        p_up = min(parameters["max_p"], p+1)
        p_down = max(parameters["min_p"], p - 1)
        q_up = min(parameters["max_q"], q + 1)
        q_down = max(parameters["min_q"], q - 1)

        parameters_space = [(p_up, q, P, Q), (p_up, q_up, P, Q), (p, q_up, P, Q),
                                   (p_down, q, P, Q), (p_down, q_down, P, Q), (p, q_down, P, Q),
                                   (p_up, q_down, P, Q), (p_down, q_up, P, Q)]
        if(self.__seasonal):
            P_up = min(parameters["max_P"], P + 1)
            P_down = max(parameters["min_P"], P - 1)
            Q_up = min(parameters["max_Q"], Q + 1)
            Q_down = max(parameters["min_Q"], Q - 1)
            seasonal_parameter_space = [(p, q, P_up, Q), (p, q, P, Q_up), (p, q, P_up, Q_up),
                                        (p, q, P_down, Q), (p, q, P, Q_down), (p, q, P_down, Q_down),
                                        (p, q, P_up, Q_down), (p, q, P_down, Q_up)]

            parameters_space += seasonal_parameter_space

        # TODO : Revisit
        if allow_drift == 1:
            parameters_space = parameters_space + [(p, q, P, Q)]

        np.random.shuffle(parameters_space)
        return parameters_space