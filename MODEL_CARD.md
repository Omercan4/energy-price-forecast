# Model card: day-ahead price forecast, Germany

## Intended use
Show the expected shape of tomorrow's day-ahead electricity prices (DE-LU) as a median and an 80 % interval, for learning, dashboards
and discussion. **Not** for trading or any decision with financial risk.

## Model
LightGBM quantile regression (10 %, 50 %, 90 %), 400 trees each, 16 inputs. The interval is widened by an offset learned
on the last 60 days of the training data (conformalized quantile regression). Trained on hourly data from 1 Jan 2022.

## Inputs (all known before the auction at 12:00 on the day before)
- Calendar: hour, weekday, month, German public holiday.
- Weather forecasts for day D (Open-Meteo): temperature, solar radiation, wind at 100 m (North Sea and inland).
- Price history: the same local hour 1, 2, 3 and 7 days earlier; mean, minimum and maximum of the day before; mean of the last 7 days.

Not used in the released model: the grid operators' load, wind and solar forecasts (published too late), fuel and CO2 prices (no source).

## Evaluation
12-month rolling-origin backtest (each month predicted by a model trained on earlier data only). MAE 19.6 EUR/MWh vs. 37.4 for "same hour last week".
80 % interval coverage 79.0 %. Details, tables and figures in the README.

## Known limits
- Weather in training comes from the historical forecast archive (shorter lead time than the real morning-before forecast).
- Price spikes and negative-price hours are hard to predict; errors are largest in the evening ramp and in autumn.
- Markets change (new regulation, gas prices, new capacity). The monitoring box compares the rolling error with the backtest error
  and warns when it is 1.5 times higher, or when the inputs drift (PSI above 0.25).
- Prices are hourly means. Since October 2025 the market clears in 15-minute steps.
- Lags are on the local clock, with the repeated hour of the clock change matched by occurrence; a missing hour uses the hour before.

## Data
Bundesnetzagentur | SMARD.de (CC BY 4.0), Open-Meteo.com (CC BY 4.0, non-commercial use).
