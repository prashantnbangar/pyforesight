import data.reader.DataReader as DataReader
import data.preprocess.Indexer as Indexer
import data.preprocess.MissingValuesImputer as Imputer


def load_data(path, file_type, date_col, frequency):
    """
    Loads the dataset, indexes it and imputes the missing values
    :param path: path to the data set on disk
    :param file_type: file type of the data
    :param date_col: date column name in the dataset
    :param frequency: frequency of the dates
    :return: pandas dataframe
    """
    dataframe = DataReader.read_data(path, file_type, date_col=date_col)
    dataframe = Indexer.index_dates(dataframe, date_col, frequency=frequency)
    dataframe = Imputer.impute(dataframe)
    return dataframe