# forecast.py
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


class Forecast:
    """
    Generates weekly demand forecasts per site and product from historical sales data.

    This class:
        - Aggregates daily sales into ISO week totals.
        - Fits baseline time-series models (ETS and ARIMA) for each site-product pair.
        - Backtests the models on a holdout period to select the best model.
        - Produces a user-specified horizon forecast with non-negative integer values.
        - Saves a timestamped CSV of forecasts for traceability.

    Attributes:
        input_sales (str): Path to CSV file containing historical sales data with columns 
            ['site_id', 'product_id', 'date', 'units_sold'].
        holdout_weeks (int): Number of weeks reserved for model backtesting.
        forecast_horizon (int): Number of future weeks to forecast.
    
    Methods:
        run(): Executes the forecast for all site-product pairs and returns a DataFrame with columns:
            ['site_id', 'product_id', 'week_start', 'forecast_units', 'method'].
    """

    def __init__(self, input_sales: str, holdout_weeks: int, forecast_horizon: int):
        self.input_sales = input_sales
        self.holdout_weeks = holdout_weeks
        self.forecast_horizon = forecast_horizon

    def __load_and_prep(self) -> pd.DataFrame:
        """
        Loads sales CSV, converts dates to ISO week starts (Mondays), 
        and aggregates units sold per week, site, and product.
        """
        df = pd.read_csv(self.input_sales, parse_dates=['date'])
        df['week_start'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
        weekly = df.groupby(['site_id', 'product_id', 'week_start'], as_index=False)['units_sold'].sum()
        return weekly

    def __fit_and_forecast(self, series: pd.Series, h: int) -> tuple[pd.Series, str]:
        """
        Fits ETS and ARIMA models on a time series and selects the best based on backtest RMSE.

        Args:
            series (pd.Series): Weekly units sold, indexed by week_start datetime.
            h (int): Forecast horizon (number of weeks).

        Returns:
            forecast (pd.Series): Forecasted weekly units (non-negative integers).
            method (str): Model used ('ETS' or 'ARIMA').
        """
        n = len(series)
        train = series.iloc[:max(1, n - self.holdout_weeks)]

        # ETS model
        try:
            ets = ExponentialSmoothing(train, seasonal='add', seasonal_periods=52).fit()
            ets_fore = ets.forecast(h)
            ets_err = (np.sqrt(((series.iloc[-self.holdout_weeks:] - ets.forecast(self.holdout_weeks))**2).mean())
                       if n > self.holdout_weeks else np.inf)
        except Exception as ex:
            print("ETS failed:", ex)
            ets_fore = pd.Series(np.repeat(train.mean(), h),
                                 index=pd.date_range(train.index[-1] + pd.Timedelta(7, 'd'),
                                                     periods=h, freq='7D'))
            ets_err = np.inf

        # ARIMA model
        try:
            ar = ARIMA(train, order=(1, 1, 1)).fit()
            ar_fore = ar.get_forecast(h).predicted_mean
            ar_err = (np.sqrt(((series.iloc[-self.holdout_weeks:] - ar.get_forecast(self.holdout_weeks).predicted_mean)**2).mean())
                      if n > self.holdout_weeks else np.inf)
        except Exception as ex:
            print("ARIMA failed:", ex)
            ar_fore = pd.Series(np.repeat(train.mean(), h),
                                index=pd.date_range(train.index[-1] + pd.Timedelta(7, 'd'),
                                                    periods=h, freq='7D'))
            ar_err = np.inf

        # Choose the model with lower error
        if ar_err < ets_err:
            chosen, method = ar_fore, 'ARIMA'
        else:
            chosen, method = ets_fore, 'ETS'

        # Ensure non-negative integers
        chosen = chosen.clip(lower=0).round().astype(int)
        return chosen, method

    def run(self) -> pd.DataFrame:
        """
        Runs the forecasting pipeline for all site-product pairs.

        Returns:
            pd.DataFrame: Forecast output with columns ['site_id', 'product_id', 
                                                       'week_start', 'forecast_units', 'method']
        """
        weekly = self.__load_and_prep()
        outputs = []

        for (site, prod), group in weekly.groupby(['site_id', 'product_id']):
            ts = group.set_index('week_start')['units_sold'].asfreq('7D').fillna(0)
            forecast, method = self.__fit_and_forecast(ts, self.forecast_horizon)
            for week_start, val in forecast.items():
                outputs.append({
                    'site_id': site,
                    'product_id': prod,
                    'week_start': pd.to_datetime(week_start),
                    'forecast_units': int(val),
                    'method': method
                })

        outdf = pd.DataFrame(outputs)
        stamp = datetime.now().strftime("%Y%m%d")
        outpath = f"outputs/forecasts_v{stamp}.csv"
        outdf.to_csv(outpath, index=False)
        print("Forecasts written to:", outpath)
        return outdf
