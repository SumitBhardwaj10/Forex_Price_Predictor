import streamlit as st
from backtesting import Backtest,Strategy
from ta.momentum import RSIIndicator
import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.trend import CCIIndicator
from ta.volatility import BollingerBands
from ta.momentum import AwesomeOscillatorIndicator

#=============================================================================
# STRATEGY 1 RUNNER FUNCTION
#=============================================================================

class Strategy_01(Strategy):
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        super().next()
        current_ema_small = self.data.ema_small[-1]
        current_ema_big = self.data.ema_big[-1]
        previous_ema_small = self.data.ema_small[-2]
        previous_ema_big = self.data.ema_big[-2]

        if (previous_ema_small < previous_ema_big) and (current_ema_small > current_ema_big):
            current_price = self.data.Close[-1]
            SL = current_price - current_price * self.slperc
            TP = current_price + current_price * self.tpperc
            self.buy(size=self.my_size, sl=SL, tp=TP)
        elif (previous_ema_small > previous_ema_big) and (current_ema_small < current_ema_big):
            current_price = self.data.Close[-1]
            SL = current_price + current_price * self.slperc
            TP = current_price - current_price * self.tpperc
            self.sell(size=self.my_size, sl=SL, tp=TP)


#=============================================================================
# STRATEGY 2 RUNNER FUNCTION
#=============================================================================


class Strategy_02(Strategy):
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        super().next()
        buy_signal = self.data.buy_pattern[-1]
        sell_signal = self.data.sell_pattern[-1]

        if buy_signal > 0:
            current_price = self.data.Close[-1]
            SL = current_price - current_price * self.slperc
            TP = current_price + current_price * self.tpperc
            self.buy(size=self.my_size, sl=SL, tp=TP)
        elif sell_signal > 0:
            current_price = self.data.Close[-1]
            SL = current_price + current_price * self.slperc
            TP = current_price - current_price * self.tpperc
            self.sell(size=self.my_size, sl=SL, tp=TP)


#=============================================================================
# STRATEGY 3 RUNNER FUNCTION
#=============================================================================



class Strategy_03(Strategy):
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 10:
            return

        current_rsi = self.data.rsi[-1]
        previous_rsi = self.data.rsi[-10]
        current_price = self.data.Close[-1]
        previous_price = self.data.Close[-10]

        if current_price < previous_price and current_rsi > previous_rsi and current_rsi < 40:
            SL = current_price - current_price * self.slperc
            TP = current_price + current_price * self.tpperc
            self.buy(size=self.my_size, sl=SL, tp=TP)
        elif current_price > previous_price and current_rsi < previous_rsi and current_rsi > 60:
            SL = current_price + current_price * self.slperc
            TP = current_price - current_price * self.tpperc
            self.sell(size=self.my_size, sl=SL, tp=TP)


#=============================================================================
# STRATEGY 4 RUNNER FUNCTION
#=============================================================================


def macd_func(series, fast=12, slow=26, signal=9):
    s = pd.Series(series)
    macd = MACD(s, window_fast=fast, window_slow=slow, window_sign=signal)
    return macd.macd().values

class Strategy_04(Strategy):  # MACD Divergence
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04
    fast, slow, signal = 12, 26, 9

    def init(self):
        super().init()
        self.macd_line = self.I(macd_func, self.data.Close, self.fast, self.slow, self.signal)

    def next(self):
        lookback = 10
        price = self.data.Close
        if len(price) > lookback * 2:
            prev_low_idx = np.argmin(price[-lookback*2:-lookback]) + (len(price) - lookback*2)
            curr_low_idx = np.argmin(price[-lookback:]) + (len(price) - lookback)
            prev_high_idx = np.argmax(price[-lookback*2:-lookback]) + (len(price) - lookback*2)
            curr_high_idx = np.argmax(price[-lookback:]) + (len(price) - lookback)

            prev_low_price, curr_low_price = price[prev_low_idx], price[curr_low_idx]
            prev_high_price, curr_high_price = price[prev_high_idx], price[curr_high_idx]

            prev_low_macd, curr_low_macd = self.macd_line[prev_low_idx], self.macd_line[curr_low_idx]
            prev_high_macd, curr_high_macd = self.macd_line[prev_high_idx], self.macd_line[curr_high_idx]

            current_price = price[-1]

            if curr_low_price < prev_low_price and curr_low_macd > prev_low_macd:
                SL = current_price - current_price * self.slperc
                TP = current_price + current_price * self.tpperc
                self.buy(size=self.my_size, sl=SL, tp=TP)
            elif curr_high_price > prev_high_price and curr_high_macd < prev_high_macd:
                SL = current_price + current_price * self.slperc
                TP = current_price - current_price * self.tpperc
                self.sell(size=self.my_size, sl=SL, tp=TP)


#=============================================================================
# STRATEGY 5 RUNNER FUNCTION
#=============================================================================


def cci_func(high, low, close, window=20):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    return CCIIndicator(high=h, low=l, close=c, window=window).cci().values

class Strategy_05(Strategy):  # CCI OB/OS
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04
    cci_window = 20

    def init(self):
        super().init()
        self.cci = self.I(cci_func, self.data.High, self.data.Low, self.data.Close, self.cci_window)

    def next(self):
        current_cci = self.cci[-1]
        price = self.data.Close[-1]
        if self.cci[-2] < -100 and current_cci > -100:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif self.cci[-2] > 100 and current_cci < 100:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)



#=============================================================================
# STRATEGY 6 RUNNER FUNCTION
#=============================================================================


def bbands_func(series, window=20, window_dev=2):
    s = pd.Series(series)
    bb = BollingerBands(close=s, window=window, window_dev=window_dev)
    return bb.bollinger_mavg().values, bb.bollinger_hband().values, bb.bollinger_lband().values

class Strategy_06(Strategy):  # Bollinger Reversion
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04
    bb_window = 20
    bb_dev = 2

    def init(self):
        super().init()
        self.bb_mid, self.bb_upper, self.bb_lower = self.I(bbands_func, self.data.Close, self.bb_window, self.bb_dev)

    def next(self):
        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        if price < lower:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif price > upper:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)



#=============================================================================
# STRATEGY 7 RUNNER FUNCTION
#=============================================================================



class Strategy_07(Strategy):
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04
    # precomputed in data: 'supertrend', 'st_dir' (+1 long, -1 short), 'st_flip' boolean
    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 3:
            return
        price = self.data.Close[-1]
        flip = self.data.st_flip[-1]
        direction = self.data.st_dir[-1]  # +1 or -1
        if flip and direction == 1:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif flip and direction == -1:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)

#=============================================================================
# STRATEGY 8 RUNNER FUNCTION
#=============================================================================

class Strategy_08(Strategy):
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 3:
            return
        price = self.data.Close[-1]
        # Buy: %K crosses above %D while below oversold threshold
        if self.data.k_cross_up[-1] and self.data.stoch_k[-1] < self.data.os_level[-1]:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        # Sell: %K crosses below %D while above overbought threshold
        elif self.data.k_cross_dn[-1] and self.data.stoch_k[-1] > self.data.ob_level[-1]:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)

#=============================================================================
# STRATEGY 9 RUNNER FUNCTION
#=============================================================================

class Strategy_09(Strategy):
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04
    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 2:
            return
        price = self.data.Close[-1]
        if self.data.Close[-1] > self.data.donch_h[-2]:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif self.data.Close[-1] < self.data.donch_l[-2]:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)

#=============================================================================
# STRATEGY 10 RUNNER FUNCTION
#=============================================================================

class Strategy_10(Strategy):
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 2:
            return
        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]
        psar_now = self.data.psar[-1]
        psar_prev = self.data.psar[-2]
        # crossovers
        if prev_price < psar_prev and price > psar_now:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif prev_price > psar_prev and price < psar_now:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)

#=============================================================================
# STRATEGY 11 RUNNER FUNCTION
#=============================================================================

class Strategy_11(Strategy):
    my_size = 0.1
    slperc = 0.02
    tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 3:
            return
        price = self.data.Close[-1]
        tenkan, kijun = self.data.tenkan[-1], self.data.kijun[-1]
        span_a, span_b = self.data.span_a[-1], self.data.span_b[-1]
        cloud_top = max(span_a, span_b)
        cloud_bot = min(span_a, span_b)

        bull_cross = self.data.tenkan[-2] <= self.data.kijun[-2] and tenkan > kijun
        bear_cross = self.data.tenkan[-2] >= self.data.kijun[-2] and tenkan < kijun

        if bull_cross and price > cloud_top:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif bear_cross and price < cloud_bot:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)

#=============================================================================
# STRATEGY 12 RUNNER FUNCTION
#=============================================================================

class Strategy_12(Strategy):
    my_size = 0.1; slperc = 0.02; tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 3:
            return
        price = self.data.Close[-1]
        di_pos_now, di_neg_now = self.data.di_pos[-1], self.data.di_neg[-1]
        di_pos_prev, di_neg_prev = self.data.di_pos[-2], self.data.di_neg[-2]
        adx_now, thr = self.data.adx[-1], self.data.adx_thr[-1]

        bull_cross = di_pos_prev <= di_neg_prev and di_pos_now > di_neg_now
        bear_cross = di_pos_prev >= di_neg_prev and di_pos_now < di_neg_now

        if bull_cross and adx_now > thr:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif bear_cross and adx_now > thr:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)

#=============================================================================
# STRATEGY 13 RUNNER FUNCTION
#=============================================================================

class Strategy_13(Strategy):
    my_size = 0.1; slperc = 0.02; tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 3:
            return
        price = self.data.Close[-1]
        # enter only when we are in squeeze and price breaks upper/lower band
        if self.data.squeeze[-1] and self.data.Close[-1] > self.data.bb_upper[-1]:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif self.data.squeeze[-1] and self.data.Close[-1] < self.data.bb_lower[-1]:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)
#=============================================================================
# STRATEGY 14 RUNNER FUNCTION
#=============================================================================

class Strategy_14(Strategy):
    my_size = 0.1; slperc = 0.02; tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 3:
            return
        price = self.data.Close[-1]
        # Bounce up from 61.8% retracement in uptrend
        if self.data.fib_long_zone[-1]:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        # Bounce down from 61.8% in downtrend
        elif self.data.fib_short_zone[-1]:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)

#=============================================================================
# STRATEGY 15 RUNNER FUNCTION
#=============================================================================

class Strategy_15(Strategy):
    my_size = 0.1; slperc = 0.02; tpperc = 0.04

    def init(self):
        super().init()

    # precomputed: zscore, z_entry, z_exit
    def next(self):
        if len(self.data.Close) < 2:
            return
        price = self.data.Close[-1]
        z = self.data.zscore[-1]
        z_e = self.data.z_entry[-1]
        z_x = self.data.z_exit[-1]
        if z < -z_e:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif z > z_e:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)
        # Exit when zscore mean-reverts
        if self.position and abs(z) < z_x:
            self.position.close()

#=============================================================================
# STRATEGY 16 RUNNER FUNCTION
#=============================================================================

class Strategy_16(Strategy):
    my_size = 0.1; slperc = 0.02; tpperc = 0.04

    def init(self):
        super().init()

    # precomputed: vwap
    def next(self):
        if len(self.data.Close) < 2:
            return
        price = self.data.Close[-1]
        above_now = self.data.Close[-1] > self.data.vwap[-1]
        above_prev = self.data.Close[-2] > self.data.vwap[-2]
        if (not above_prev) and above_now:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif above_prev and (not above_now):
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)

#=============================================================================
# STRATEGY 17 RUNNER FUNCTION
#=============================================================================

class Strategy_17(Strategy):
    my_size = 0.1; slperc = 0.02; tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 2:
            return
        price = self.data.Close[-1]
        # green HA candle and above EMA -> buy; red HA and below EMA -> sell
        if self.data.ha_close[-1] > self.data.ha_open[-1] and self.data.Close[-1] > self.data.ema_trend[-1]:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif self.data.ha_close[-1] < self.data.ha_open[-1] and self.data.Close[-1] < self.data.ema_trend[-1]:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)
#=============================================================================
# STRATEGY 188 RUNNER FUNCTION
#=============================================================================

class Strategy_18(Strategy):
    my_size = 0.1; slperc = 0.02; tpperc = 0.04

    def init(self):
        super().init()

    def next(self):
        if len(self.data.Close) < 2:
            return
        price = self.data.Close[-1]
        if self.data.nr7[-1] and self.data.Close[-1] > self.data.range_high[-1]:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif self.data.nr7[-1] and self.data.Close[-1] < self.data.range_low[-1]:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)
#=============================================================================
# STRATEGY 19 RUNNER FUNCTION
#=============================================================================

from ta.volatility import KeltnerChannel
def keltner_func(high, low, close, window=20, mult=2.0):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    kc = KeltnerChannel(h, l, c, window=window, window_atr=window, original_version=True)
    return kc.keltner_channel_hband().values, kc.keltner_channel_lband().values

class Strategy_19(Strategy):
    my_size = 0.1; slperc = 0.02; tpperc = 0.04

    def init(self):
        super().init()

    # precomputed: kc_upper, kc_lower
    def next(self):
        price = self.data.Close[-1]
        if self.data.Close[-1] > self.data.kc_upper[-1]:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif self.data.Close[-1] < self.data.kc_lower[-1]:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)

#=============================================================================
# STRATEGY 20 RUNNER FUNCTION
#=============================================================================


def ao_func(high, low, s=5, l=34):
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    ao = AwesomeOscillatorIndicator(high=high_series, low=low_series, window1=s, window2=l)
    return ao.awesome_oscillator().values

class Strategy_20(Strategy):
    my_size = 0.1; slperc = 0.02; tpperc = 0.04
    ao_s = 5; ao_l = 34

    def init(self):
        super().init()
        self.ao_line = self.I(ao_func, self.data.High, self.data.Low, self.ao_s, self.ao_l)

    def next(self):
        if len(self.ao_line) < 2:
            return
        price = self.data.Close[-1]
        # zero-line cross logic
        if self.ao_line[-2] <= 0 and self.ao_line[-1] > 0:
            self.buy(size=self.my_size, sl=price - price * self.slperc, tp=price + price * self.tpperc)
        elif self.ao_line[-2] >= 0 and self.ao_line[-1] < 0:
            self.sell(size=self.my_size, sl=price + price * self.slperc, tp=price - price * self.tpperc)