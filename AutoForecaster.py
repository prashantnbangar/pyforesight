import data.DataLoader as DataLoader
import sys
from model.AutoARIMA import AutoARIMA


def main():
    if len(sys.argv) < 6:
        raise SyntaxError("Missing inputs to the script")

    DATASET_PATH = sys.argv[1]
    DATA_TYPE = sys.argv[2]
    DATE_COL = sys.argv[3]
    SERIES_NAME = sys.argv[4]
    FREQUENCY=sys.argv[5]
    SEASONAL_PERIOD = int(sys.argv[6])
    FORECAST_STEPS = int(sys.argv[7])

    dataframe = DataLoader.load_data(DATASET_PATH, DATA_TYPE, date_col=DATE_COL, frequency=FREQUENCY)

    model = AutoARIMA()
    model.fit(dataframe[SERIES_NAME], seasonal_period=SEASONAL_PERIOD)
    print(model.predict(FORECAST_STEPS))


if __name__ == "__main__":
    main()
