

def impute(dataframe):
    """
    Imputes the missing values and returns the dataframe
    :param dataframe: dataframe with date index
    :return: dataframe with imputed missing values
    """
    dataframe = dataframe.fillna(method="ffill")
    dataframe = dataframe.fillna(method="bfill")
    return dataframe
