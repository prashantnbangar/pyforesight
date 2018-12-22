import data.DataLoader as DataLoader
import sys
from model.AutoARIMA import AutoARIMA


def main():
    if len(sys.argv) < 6:
        raise SyntaxError("Missing inputs to the script")

    DATASET_PATH = sys.argv[1]
    DATA_TYPE = sys.argv[2]
    DATE_COL = sys.argv[3]
    FREQUENCY=sys.argv[4]
    SEASONAL_PERIOD = sys.argv[5]

    dataframe = DataLoader.load_data(DATASET_PATH, DATA_TYPE, date_col=DATE_COL, frequency=FREQUENCY)

    model = AutoARIMA()
    model.fit(dataframe)


if __name__ == "__main__":
    main()
