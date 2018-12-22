
def index_dates(dataframe, cols, frequency, drop_cols_used_as_index=True):
    """
    Sets the date col as dataframe index and resamples the dataframe to create empty rows for missing dates if any
    :param dataframe: dataframe with date column in it
    :param cols: list of date columns
    :param frequency: frequency of dates - [M, MS, D etc]
    :param drop_cols_used_as_index: boolean, drop columns set as index from dataframe
    :return: reindexed dataframe
    """
    dataframe.set_index(cols, drop=drop_cols_used_as_index, inplace=True)
    dataframe = dataframe.resample(frequency)
    return dataframe
