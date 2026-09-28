"""FreqAI classifier strategy producing up, flat, and down signals."""

from functools import reduce

import numpy as np
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy


class MarketSignalStrategy(IStrategy):
    """Classify the configured forward window into actionable market regimes."""

    timeframe = "1h"
    can_short = False
    process_only_new_candles = True
    startup_candle_count = 80
    minimal_roi = {"0": 0.04, "240": 0.015, "720": 0}
    stoploss = -0.05
    use_exit_signal = True
    signal_threshold = 0.004

    plot_config = {
        "main_plot": {},
        "subplots": {
            "FreqAI prediction quality": {
                "do_predict": {"color": "#22c55e"},
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
        self.freqai.class_names = ["down", "flat", "up"]
        horizon = self.freqai_info["feature_parameters"]["label_period_candles"]
        future_return = dataframe["close"].shift(-horizon) / dataframe["close"] - 1
        dataframe["&-market_direction"] = np.select(
            [
                future_return < -self.signal_threshold,
                future_return > self.signal_threshold,
            ],
            ["down", "up"],
            default="flat",
        )
        dataframe.loc[future_return.isna(), "&-market_direction"] = np.nan
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return self.freqai.start(dataframe, metadata, self)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-market_direction"] == "up",
            dataframe["volume"] > 0,
        ]
        dataframe.loc[
            reduce(lambda left, right: left & right, conditions),
            ["enter_long", "enter_tag"],
        ] = (1, "freqai_up")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            dataframe["do_predict"] == 1,
            dataframe["&-market_direction"] == "down",
        ]
        dataframe.loc[
            reduce(lambda left, right: left & right, conditions),
            ["exit_long", "exit_tag"],
        ] = (1, "freqai_down")
        return dataframe
