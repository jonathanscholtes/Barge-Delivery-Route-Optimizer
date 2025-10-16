# main.py
from forecast import Forecast
from optimize import Optimizer
import pandas as pd
from datetime import datetime

HOLDOUT_WEEKS = 12
FORECAST_HORIZON = 52
INPUT_SALES = r"../../data/sales_history.csv"
SITE_SPECS = r"../../data/site_specs.csv"
TRAVEL = r"../../data/travel_times.csv"
BARGE = r"../../data/barge_specs.csv"


def main():
    """
    Main pipeline:
        1. Generate weekly forecasts per site-product.
        2. Run CVRPTW optimizer for the specified week.
    """
    # 1 Forecasting
    forecaster = Forecast(INPUT_SALES, HOLDOUT_WEEKS, FORECAST_HORIZON)
    forecast_df = forecaster.run()
    print("Forecasting completed. Sample output:")
    print(forecast_df.head())

    # 2 Optimization
    optimizer = Optimizer(forecast_df, SITE_SPECS, TRAVEL, BARGE)
    # Example: run optimizer for a specific week
    week_start = '2026-04-13'
    print(f"\nRunning optimizer for week starting {week_start}...")
    optimizer.run(week_start=week_start)


if __name__ == "__main__":
    main()
