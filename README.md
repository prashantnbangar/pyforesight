# pyautoarima
Enhanced Automatic time series forecasting using ARIMA family of models. 
Implements the basic Auto-ARIMA with additional functionalities in python creating a robust algorithm to handle any type of series for forecasting. 
Automatically chooses the best model for forecasting based on the model score.

Additional Features above the basic Auto-ARIMA implementation,
1. Handling Long Seasonalities
2. Handling multiple Seasonalities
3. Accepting Holiday feature and using it as an exogenous feature
4. Residual Analysis
5. Power transformation for handling multiplicative series


Usage:

forecast <data_file_disk_path> <data_file_type> <date_column_name> <time_series_column_name> <date_frequency> <seasonal_period> <forecast_steps>


Currently only csv data file type is supported.

Future scope:
1. Implement cross-validation for model selection
2. Consider change-points in the time series to improve the forecasting accuracy.
