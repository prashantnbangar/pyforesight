import pandas as pd
import numpy as np


class PowerTransformer():
    def __init__(self):
        pass

    def transform(self, series):
        """
        Transforms the series using the inverse hyperbolic sin transformer
        :param series: series to be transformed
        :return: transformed_series
        """
        transformed_series = pd.Series(np.arcsinh(series.values))
        transformed_series.index = series.index
        return transformed_series

    def inverse_transform(self, series):
        """
        Inverse Transforms the passed series
        :param series: Series to be reverse transformed
        :return: inverse transformed series
        """
        transformed_series = pd.Series(np.sinh(series.values))
        transformed_series.index = series.index
        return transformed_series

