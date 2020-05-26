# pyforesight
Automatic time series forecasting using ARIMA family of models. 
Implements the basic Auto-ARIMA algorithm in python to handle any type of series for forecasting. Automatically chooses the best parameters for model based on the model score.


Usage:

forecast <data_file_path_on_disk> <date_column_name> <time_series_column_name> <forecast_steps> ["FREQUENCY=<date_frequency>,SEASONAL_PERIOD=<seasonal_period>"]

FREQUENCY and SEASONAL_PERIOD are optional parameters.

Output after model training on the Air Passengers dataset
![Airlines Dataset example](docs/AirlinesForecast.png?raw=true "Airlines Data Forecasting")
