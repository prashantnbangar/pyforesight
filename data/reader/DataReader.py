import pandas as pd


def read_data(path_to_data, date_col):
    """
    Reads the datasets from the specified location and returns a pandas dataframe
    :param path_to_data: path of data file on disk
    :param file_type: type of data file
    :param date_col: date column name in data
    :return: pandas dataframe
    """
    """"""
    if path_to_data.endswith("csv"):
        return __read_csv(path_to_data, parse_dates=[date_col])
    else:
        return None


def __read_csv(path, seperator=",", index_col=None, parse_dates=False, infer_date_format=True):
    return pd.read_csv(path, sep=seperator, index_col=index_col, parse_dates=parse_dates,
                infer_datetime_format=infer_date_format)

