import logging
from functools import reduce

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange.exchange_utils import *
from freqtrade.strategy import IStrategy, RealParameter

logger = logging.getLogger(__name__)


class ExampleLSTMStrategy(IStrategy):
    """
    Example strategy for spot trading (long only).
    Use at your own risk. For educational purposes only.
    """
    # Hyperspace parameters:
    buy_params = {
        "threshold_buy": 0.59453,
        "w0": 0.54347,
        "w1": 0.82226,
        "w2": 0.56675,
        "w3": 0.77918,
        "w4": 0.98488,
        "w5": 0.31368,
        "w6": 0.75916,
        "w7": 0.09226,
        "w8": 0.85667,
    }

    sell_params = {
        "threshold_sell": 0.80573,
    }

    # ROI table:
    minimal_roi = {
        "600": 0  # Let the model decide when to exit
    }

    # Stoploss:
    stoploss = -1  # Let the model decide when to sell

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.0139
    trailing_only_offset_is_reached = True

    timeframe = "1h"
    can_short = False  # Spot trading does not support shorting
    use_exit_signal = True
    process_only_new_candles = True

    startup_candle_count = 20

    threshold_buy = RealParameter(-1, 1, default=0, space='buy')
    threshold_sell = RealParameter(-1, 1, default=0, space='sell')

    # Weights for calculating the aggregate score - the sum of all weighted normalized indicators has to be 1!
    w0 = RealParameter(0, 1, default=0.10, space='buy')
    w1 = RealParameter(0, 1, default=0.15, space='buy')
    w2 = RealParameter(0, 1, default=0.10, space='buy')
    w3 = RealParameter(0, 1, default=0.15, space='buy')
    w4 = RealParameter(0, 1, default=0.10, space='buy')
    w5 = RealParameter(0, 1, default=0.10, space='buy')
    w6 = RealParameter(0, 1, default=0.10, space='buy')
    w7 = RealParameter(0, 1, default=0.05, space='buy')
    w8 = RealParameter(0, 1, default=0.15, space='buy')

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        enter_long_conditions = [
            df["do_predict"] == 1,
            df['&-target'] > self.threshold_buy.value,  # Target score above buy threshold
            df['volume'] > 0
        ]

        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "long")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conditions = [
            df["do_predict"] == 1,
            df['&-target'] < self.threshold_sell.value  # Target score below sell threshold
        ]

        df.loc[
            reduce(lambda x, y: x & y, exit_long_conditions), ["exit_long", "exit_tag"]
        ] = (1, "exit_long")

        return df
