from seasonal import fit_seasons
import pandas as pd
from numpy import tile

class SeasonalEstimator():
    def decompose(self, data, period=None):
        """
        Decomposes the series and returns the seasonality and the seasonal component of the series
        :param ts: time series
        :param series_column: series column name
        :param period: seasonal period if known
        :return: seasonality and seasonal series
        """
        nobs = len(data)
        X = data.values.squeeze()
        self.__seasonal_pattern, trend = fit_seasons(X, trend="spline", period=period)
        trend = pd.Series(trend, index=data.index)

        self.__seasonality = 0
        if self.__seasonal_pattern is None:
            seasonal_series = pd.Series(0, index=data.index)
        else:
            self.__seasonality = len(self.__seasonal_pattern)
            seasonal_series = tile(self.__seasonal_pattern, nobs//self.__seasonality + 1)[:nobs]
            seasonal_series = pd.Series(seasonal_series, index=data.index)
            self.__diff = nobs % self.__seasonality

        return self.__seasonality, pd.DataFrame({"seasonal":seasonal_series})

    def predict_seasonal_series(self, steps):
        """
        Predicts the seasonal component of the series for given steps
        :param steps:
        :return: seasonal series
        """
        if self.__seasonal_pattern is None:
            return None
        extended_seasonal_series = list(self.__seasonal_pattern[:self.__diff])
        extended_seasonal_series += list(tile(self.__seasonal_pattern, steps // self.__seasonality + 1))
        extended_seasonal_series = extended_seasonal_series[: steps]
        return pd.DataFrame(extended_seasonal_series)