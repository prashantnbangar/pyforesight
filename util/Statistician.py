from statsmodels.tsa import stattools
import numpy as np
from statsmodels.regression.linear_model import OLS

def ocsb(ts, period):
    """
    Returns the seasonal order of differencing required by the time series
    :param ts: time series
    :param period: seasonal order
    :return:
    """
    if len(ts) < (2*period)+5:
        return 0
    s_diff_series = ts.diff(period)[period:]
    diff_series = s_diff_series.diff()[1:]
    diff_series = diff_series[(1+period):]

    t1 = ts[1:].diff(period)[period:][period:-1]
    t2 = ts[period:].diff()[1:][1:-period]
    x_reg = np.stack((t1, t2), axis=1)

    test_value = ocsb_test_value(diff_series, x_reg, period)
    critical_value = ocsb_citical_value(period)
    return int(test_value >= critical_value)


def ocsb_test_value(diff_series, x_reg, period):
    try:
        fit = OLS(diff_series, x_reg).fit()
    except ValueError:
        # Regression Model cannot be fit
        return -np.inf

    t2 = np.sqrt(fit.cov_params()["x2"]["x2"])
    return fit.params["x2"]/t2


def ocsb_citical_value(period):
    log_p = np.log(period)
    return(-0.2937411 * np.exp(-0.2850853 *(log_p - 0.7656451)+
                               (-0.05983644)*((log_p - 0.7656451)**2)) - 1.652202)

def kpss(ts):
    """Returns the order of differencing required by a time series to make it stationary
    :param ts - time series
    """
    kpss_value, p_vaue, lags, critical_value = stattools.kpss(ts)
    if kpss_value < critical_value["5%"]:
        return 0
    else:
        return 1