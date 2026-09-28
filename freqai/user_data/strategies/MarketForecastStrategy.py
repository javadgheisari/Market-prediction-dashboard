"""FreqAI regression strategy for forward returns and trading signals."""

from functools import reduce

import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import DecimalParameter, IStrategy


class MarketForecastStrategy(IStrategy):
    """Predict the mean return over the configured forward candle window."""

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 80
    minimal_roi = {"0": 0.04, "240": 0.015, "720": 0}
    stoploss = -0.05
    use_exit_signal = True

    entry_return = DecimalParameter(
        0.001, 0.03, default=0.006, decimals=3, space="buy"
    )
    exit_return = DecimalParameter(
        -0.03, 0.0, default=-0.002, decimals=3, space="sell"
    )

    plot_config = {
        "main_plot": {},
        "subplots": {
            "FreqAI forecast": {
                "&-future_return": {"color": "#38bdf8"},
                "DI_values": {"color": "#f59e0b"},
            }
        },
    }

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)
        bands = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.0
        )
        dataframe["%-bb-width-period"] = (
            (bands["upper"] - bands["lower"]) / bands["mid"]
        )
        dataframe["%-relative-volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-range"] = (
            (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        )
        dataframe["%-raw-volume"] = dataframe["volume"]
        dataframe["%-raw-price"] = dataframe["close"]
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        dataframe["%-day-of-week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour-of-day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&-future_return"] = (
            dataframe["close"].shift(-horizon).rolling(horizon).mean()
            / dataframe["close"]
            - 1
        )
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-future_return"] > self.entry_return.value,
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda left, right: left & right, conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "freqai_return_long")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-future_return"] < self.exit_return.value,
        ]
        dataframe.loc[
            reduce(lambda left, right: left & right, conditions),
            ["exit_long", "exit_tag"],
        ] = (1, "freqai_return_exit")
        return dataframe
