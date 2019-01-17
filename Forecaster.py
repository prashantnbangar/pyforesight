import data.DataLoader as DataLoader
import sys
from model.AutoForecastModel import AutoARIMA


def main():
    if len(sys.argv) < 4:
        raise SyntaxError("Missing inputs to the script")

    DATASET_PATH = sys.argv[1]
    DATE_COL = sys.argv[2]
    SERIES_NAME = sys.argv[3]
    FORECAST_STEPS = int(sys.argv[4])
    OPTIONAL_PARAMETERS = sys.argv[5]
    FREQUENCY=None
    SEASONAL_PERIOD=None

    # Extract optional parameters from the script inputs
    if OPTIONAL_PARAMETERS is not None and OPTIONAL_PARAMETERS != "":
        for parameter in OPTIONAL_PARAMETERS.split(","):
            vals = parameter.split("=")
            key = vals[0]
            value = vals[1]
            if key == "FREQUENCY":
                FREQUENCY = value
            elif key == "SEASONAL_PERIOD":
                SEASONAL_PERIOD = int(value)

    dataframe = DataLoader.load_data(DATASET_PATH, date_col=DATE_COL, frequency=FREQUENCY)
    model = AutoARIMA()
    model.fit(dataframe[SERIES_NAME], seasonal_period=SEASONAL_PERIOD)
    print(model.predict(FORECAST_STEPS))


if __name__ == "__main__":
    main()
