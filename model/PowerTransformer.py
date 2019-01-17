from scipy.stats import boxcox
import pandas as pd
from scipy import special


class PowerTransformer():
    def __init__(self):
        self.__lambda = None
        self.__offset = None

    def transform(self, series):
        """
        Transforms the series using the boxcox transformers
        :param series: series to be transformed
        :return: transformed_series, lambda, offset
        """
        offset = min(series)
        if offset <= 0:
            self.__offset = abs(offset) + 1
        else:
            self.__offset = 0

        transformed_series, self.__lambda = boxcox(series + self.__offset)
        transformed_series = pd.Series(transformed_series)
        transformed_series.index = series.index
        return transformed_series, self.__lambda, self.__offset

    def inverse_transform(self, series):
        """
        Inverse Transforms the passed series using the learned lambda and offset learnt earlier
        :param series: Series to be reverse transformed
        :return: inverse transformed series
        """
        transformed_series = pd.Series(special.inv_boxcox(series, self.__lambda) - self.__offset)
        transformed_series.index = series.index
        return transformed_series

