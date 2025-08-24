import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from backtesting import Backtest,Strategy
from pygments.lexer import default
from tenacity import BaseAction
from ta.volatility import BollingerBands, KeltnerChannel
from ta.trend import MACD, EMAIndicator, CCIIndicator, PSARIndicator, IchimokuIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator
from backtest_container import backtest_strategies
import backtesting
import multiprocessing
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
backtesting.Pool = multiprocessing.Pool

st.set_page_config(layout="wide")
st.markdown(f"""
<style>
    /* --- General Styling --- */
    .stApp {{
        background: #F0F2F5; /* Soft, light grey background */
        color: #333333; /* Dark grey for default text */
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* --- Containers and Cards --- */
    .main-container, .info-container {{
        background: #FFFFFF; /* Clean white background */
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid #E0E0E0; /* Subtle light grey border */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); /* Soft shadow for depth */
        animation: fadeIn 1s ease-in-out;
    }}

    .info-card {{
        background: #FFFFFF;
        border-radius: 15px;
        padding: 1.5rem;
        transition: all 0.3s ease;
        border: 1px solid #EAEAEA; /* Slightly lighter border for cards */
        height: 100%;
    }}
    .info-card:hover {{
        transform: translateY(-5px);
        background: #FAFAFA; /* Subtle hover background color */
        border-color: #007BFF; /* Accent color border on hover */
    }}

    /* --- Typography --- */
    h1, h2, h3, p, label {{
        color: #212529 !important; /* Main text color - a dark, near-black */
    }}
    h2 {{
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        border-bottom: 3px solid #007BFF; /* Accent color underline */
        padding-bottom: 10px;
        margin-bottom: 2rem;
    }}
    h3 {{
        font-size: 1.5rem;
        font-weight: bold;
        color: #007BFF !important; /* Accent color for subheadings */
        margin-bottom: 1rem;
    }}

    /* --- Button Styling --- */
    div[data-testid="stButton"] > button {{
        border-radius: 10px;
        border: 2px solid #007BFF;
        background-color: transparent;
        color: #007BFF;
        transition: all 0.3s ease;
        padding: 10px 25px;
        font-weight: bold;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }}
    div[data-testid="stButton"] > button:hover {{
        background-color: #007BFF;
        color: #FFFFFF; /* White text on hover */
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.5); /* Softer blue glow */
    }}

    /* Target the st.dialog container by its data-testid */
    [data-testid="stDialog"] {{
    /* Set the text color for all elements inside the dialog */
    color: white;
    }}
    /* --- Other Widgets --- */
    div[data-testid="stFileUploader"] {{
        border: 2px dashed #00C49A;
        background-color: rgba(0, 196, 154, 0.1);
        padding: 1rem;
        border-radius: 10px;
    }}
     /* Targeting the container of the selectbox */
    div[data-testid="stSelectbox"] > div {{
        border: 2px solid #4CAF50; /* Green border */
        border-radius: 10px; /* Rounded corners */
        background-color: #f0f2f6; /* Light grey background */
    }}

    /* Targeting the label of the selectbox */
    div[data-testid="stSelectbox"] label {{
        font-weight: bold;
        color: #4CAF50; /* Green label text */
    }}

    /* You can be even more specific if needed */
    /* This targets the selected option text */
    div[data-testid="stSelectbox"] .st-emotion-cache-1f1dhpj {{
        color: #007bff; /* Blue text for selected option */
    }}
    </style>
""", unsafe_allow_html=True)

@st.dialog("Overview")
def show_features():
    st.header("Backtesting Techniques")
    strategies = {
        "Strategy 1": "EMA Crossover",
        "Strategy 2": "Candlestick Patterns (e.g., Engulfing, Doji)",
        "Strategy 3": "RSI Divergence",
        "Strategy 4": "MACD Crossover",
        "Strategy 5": "CCI (Commodity Channel Index) Overbought/Oversold",
        "Strategy 6": "Bollinger Bands Reversion",
        "Strategy 7": "Supertrend Indicator",
        "Strategy 8": "Stochastic Oscillator Crossover",
        "Strategy 9": "Price Breakout (Support & Resistance)",
        "Strategy 10": "Parabolic SAR",
        "Strategy 11": "Ichimoku Cloud Breakout",
        "Strategy 12": "ADX Trend Strength",
        "Strategy 13": "Bollinger Band Squeeze",
        "Strategy 14": "Fibonacci Retracement Levels",
        "Strategy 15": "Pairs Trading (Statistical Arbitrage)",
        "Strategy 16": "VWAP (Volume Weighted Average Price) Crossover",
        "Strategy 17": "Heikin-Ashi Trend Following",
        "Strategy 18": "Chart Patterns (e.g., Head and Shoulders, Triangles)",
        "Strategy 19": "Keltner Channel Strategy",
        "Strategy 20": "Awesome Oscillator Strategy"
    }

    for name,val in strategies.items():
        st.markdown(f"<p><b>{name}</b>: {val}</p>", unsafe_allow_html=True)

#=============================================================================
# This will show results
#=============================================================================


def show_results(stats,figure):
    st.subheader("📈 Optimized Strategy Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Return [%]",
            value=f"{stats['Return [%]']:.2f}%"
        )
    with col2:
        st.metric(
            label="Max Drawdown [%]",
            value=f"{stats['Max. Drawdown [%]']:.2f}%"
        )
    with col3:
        st.metric(
            label="Win Rate [%]",
            value=f"{stats['Win Rate [%]']:.2f}%"
        )
    with col4:
        st.metric(
            label="# of Trades",
            value=stats['# Trades']
        )
    st.divider()
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric(
            label="Sharpe Ratio",
            value=f"{stats['Sharpe Ratio']:.2f}"
        )
    with col6:
        st.metric(
            label="Sortino Ratio",
            value=f"{stats['Sortino Ratio']:.2f}"
        )
    with col7:
        st.metric(
            label="Profit Factor",
            value=f"{stats['Profit Factor']:.2f}"
        )
    with col8:
        st.metric(
            label="Avg. Trade [%]",
            value=f"{stats['Avg. Trade [%]']:.2f}%"
        )
    st.divider()
    with st.expander("🔍 Click to see all stats"):
        st.dataframe(stats)
    with st.expander("😶‍🌫️ Click to view the corssover place"):
        st.plotly_chart(figure)


#=============================================================================
# STRATEGY 1 RUNNER FUNCTION
#=============================================================================

if __name__ == '__main__':
    @st.cache_resource
    def run_strat_01(data, small, big):
        data["ema_small"] = EMAIndicator(data["Close"], window=small).ema_indicator()
        data["ema_big"] = EMAIndicator(data["Close"], window=big).ema_indicator()

        bt = Backtest(data, backtest_strategies.Strategy_01, cash=5000, margin=1 / 5, commission=0.0002,finalize_trades=True)
        stats = bt.optimize(slperc=[x/100 for x in range(1, 11)],
                            tpperc=[x/100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")

        for i in range(50,len(data)):
            if(data["ema_small"][i]>data["ema_big"][i] and data["ema_small"][i-1]<data["ema_big"][i-1]):
                mini_df=data.iloc[i-50:i+50]
                entry_index = data.index[i]  # or mini_df.index[50] since i-50 is the start
                entry_price = data["Close"][i]
                break
        figure=make_subplots(rows=1, cols=1, subplot_titles=("EMA Crossover"))
        figure.add_trace(go.Candlestick(x=mini_df.index,
                                        open=mini_df["Open"],
                                        high=mini_df["High"],
                                        low=mini_df["Low"],
                                        close=mini_df["Close"],
                                        name="Candlestick"),row=1,col=1)
        figure.add_trace(go.Scatter(x=mini_df.index,
                                 y=mini_df["ema_small"],
                                name=f"Ema{small}",mode="lines",
                                 marker=dict(color="green")),row=1,col=1)

        figure.add_trace(go.Scatter(x=mini_df.index,
                                y=mini_df["ema_big"],mode="lines",
                                 name=f"Ema{big}",
                                 marker=dict(color="red")),row=1,col=1)

        figure.add_trace(go.Scatter(x=[entry_index],
                                    y=[entry_price-5e-4],
                                    mode="markers",
                                    marker=dict(size=10,color="MediumPurple",symbol="x"),
                                    name="Entry Condition"),row=1,col=1)

        figure.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12,color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )

        return stats,figure


#=============================================================================
# STRATEGY 2 RUNNER FUNCTION
#=============================================================================

if __name__ == '__main__':
    @st.cache_resource
    def run_strat_02(data):
        properties = pd.DataFrame()
        properties['body_size'] = abs(data['Close'] - data['Open'])
        properties['upper_wick'] = data['High'] - np.maximum(data['Open'], data['Close'])
        properties['lower_wick'] = np.minimum(data['Open'], data['Close']) - data['Low']
        properties['total_range'] = data['High'] - data['Low']
        # Avoid division by zero for candles with no range
        properties['total_range'] = properties['total_range'].replace(0, 0.0001)
        properties['is_green'] = data['Close'] > data['Open']
        properties['is_red'] = data['Close'] < data['Open']

        data["bullish_marubozu"] = (
                (properties['body_size'] >= properties['total_range'] * 0.95) & (data["Close"] > data["Open"]))
        data["bearish_marubozu"] = (
                (properties['body_size'] >= properties['total_range'] * 0.95) & (data["Close"] < data["Open"]))

        data["hammer"] = ((properties['lower_wick'] >= properties['body_size'] * 2) &
                          (properties['upper_wick'] <= properties['body_size'] * 0.5) &
                          (properties['body_size'] > properties['total_range'] * 0.05))
        data["inverted_hammer"] = ((properties['upper_wick'] >= properties['body_size'] * 2) &
                                   (properties['lower_wick'] <= properties['body_size'] * 0.5) &
                                   (properties['body_size'] > properties['total_range'] * 0.05))

        prev_props = properties.shift(1)
        data["bullish_engulfing"] = ((properties['is_green']) &
                                     (prev_props['is_red']) &
                                     (data['Close'] > data['High'].shift(1)) &
                                     (data['Open'] < data['Low'].shift(1)))
        data["bearish_engulfing"] = ((properties['is_red']) &
                                     (prev_props['is_green']) &
                                     (data['Close'] < data['Low'].shift(1)) &
                                     (data['Open'] > data['High'].shift(1)))

        data["buy_pattern"] = data["bullish_engulfing"] + data["bullish_marubozu"] + data["hammer"]
        data["sell_pattern"] = data["bearish_engulfing"] + data["bearish_marubozu"] + data["inverted_hammer"]
        bt = Backtest(data, backtest_strategies.Strategy_02, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        for i in range(50,len(data)):
            if(data["buy_pattern"].iloc[i]>0):
                mini_df=data.iloc[i-50:i+50]
                entry_index = data.index[i]  # or mini_df.index[50] since i-50 is the start
                entry_price = data["Close"][i]
                break
        figure=make_subplots(rows=1, cols=1, subplot_titles=("EMA Crossover"))
        figure.add_trace(go.Candlestick(x=mini_df.index,
                                        open=mini_df["Open"],
                                        high=mini_df["High"],
                                        low=mini_df["Low"],
                                        close=mini_df["Close"],
                                        name="Candlestick"),row=1,col=1)

        figure.add_trace(go.Scatter(x=[entry_index],
                                    y=[entry_price-5e-4],
                                    mode="markers",
                                    marker=dict(size=10,color="MediumPurple",symbol="x"),
                                    name="Entry Condition"),row=1,col=1)

        figure.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12,color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )

        return stats,figure



#=============================================================================
# STRATEGY 3 RUNNER FUNCTION
#=============================================================================


if __name__ == '__main__':
    @st.cache_resource
    def run_strat_03(data,window):
        data["rsi"]=RSIIndicator(data["Close"],window=window).rsi()
        bt = Backtest(data, backtest_strategies.Strategy_03, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        for i in range(100,len(data)):
            current_rsi = data["rsi"].iloc[i]
            previous_rsi = data["rsi"].iloc[i - 10]
            current_price = data["Close"].iloc[i]
            previous_price = data["Close"].iloc[i - 10]
            if current_price < previous_price and current_rsi > previous_rsi and current_rsi < 40:
                mini_df=data.iloc[i-50:i+50]
                entry_index = data.index[i]
                entry_price = data["Close"][i]
                break

        figure=make_subplots(rows=2, cols=1, subplot_titles=("EMA Crossover"),shared_xaxes=True)
        figure.add_trace(go.Candlestick(x=mini_df.index,
                                        open=mini_df["Open"],
                                        high=mini_df["High"],
                                        low=mini_df["Low"],
                                        close=mini_df["Close"],
                                        name="Candlestick"),row=1,col=1)

        figure.add_trace(go.Scatter(x=[entry_index],
                                    y=[entry_price-5e-4],
                                    mode="markers",
                                    marker=dict(size=10,color="MediumPurple",symbol="x"),
                                    name="Entry Condition"),row=1,col=1)
        figure.add_trace(
            go.Scatter(
                x=mini_df.index,
                y=mini_df["rsi"],  # make sure RSI column exists in your data
                mode="lines",
                line=dict(color="orange"),
                name="RSI (14)"
            ),
            row=2, col=1
        )

        figure.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12,color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )

        return stats,figure



#=============================================================================
# STRATEGY 4 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_04(data):
        data["macd_line"]=MACD(data["Close"]).macd()
        data["signal_line"]=MACD(data["Close"]).macd_signal()
        bt = Backtest(data, backtest_strategies.Strategy_04, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 8)],
                            tpperc=[x / 100 for x in range(1, 8)],
                            fast=[13,18,24],
                            slow=[13,21,30],
                            constraint=lambda x: x.slperc < x.tpperc and x.slow>x.fast,
                            maximize="Return [%]")

        for i in range(50, len(data)):
            current_macd = data["macd_line"].iloc[i]
            previous_macd = data["macd_line"].iloc[i - 10]
            current_price = data["Close"].iloc[i]
            previous_price = data["Close"].iloc[i - 10]

            # Example: bullish divergence logic
            if current_price < previous_price and current_macd > previous_macd:
                mini_df = data.iloc[i - 50:i + 50]
                entry_index = data.index[i]
                entry_price = data["Close"].iloc[i]
                break


        figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Price Action", "MACD vs Signal"),
                               row_heights=[0.7, 0.3])
            # Candlestick
        figure.add_trace(go.Candlestick(
            x=mini_df.index,
            open=mini_df["Open"],
            high=mini_df["High"],
            low=mini_df["Low"],
            close=mini_df["Close"],
            name="Candlestick"),
            row=1, col=1
        )

        # Entry marker
        figure.add_trace(go.Scatter(
            x=[entry_index],
            y=[entry_price],
            mode="markers",
            marker=dict(size=10, color="MediumPurple", symbol="x"),
            name="Entry Condition"),
            row=1, col=1
        )

        # MACD line
        figure.add_trace(go.Scatter(
            x=mini_df.index,
            y=mini_df["macd_line"],  # must exist in data
            mode="lines",
            line=dict(color="cyan"),
            name="MACD Line"),
            row=2, col=1
        )

        # Signal line
        figure.add_trace(go.Scatter(
            x=mini_df.index,
            y=mini_df["signal_line"],  # must exist in data
            mode="lines",
            line=dict(color="orange"),
            name="Signal Line"),
            row=2, col=1
        )

        # Zero reference line
        figure.add_hline(y=0, line=dict(color="grey", dash="dot"), row=2, col=1)

        figure.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )

        return stats, figure


#=============================================================================
# STRATEGY 5 RUNNER FUNCTION
#=============================================================================

if __name__ == '__main__':
    @st.cache_resource
    def run_strat_05(data):
        data["cci"]=CCIIndicator(high=data["High"],low=data["Low"],close=data["Close"],window=20).cci()
        bt = Backtest(data, backtest_strategies.Strategy_05, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 8)],
                            tpperc=[x / 100 for x in range(1, 8)],
                            cci_window=[13,20,30],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        for i in range(50, len(data)):
            current_cci = data["cci"].iloc[i]
            prev_cci = data["cci"].iloc[i - 1]
            current_price = data["Close"].iloc[i]

            # Example bullish signal: oversold exit
            if prev_cci < -100 and current_cci > -100:
                mini_df = data.iloc[i - 50:i + 50]
                entry_index = data.index[i]
                entry_price = data["Close"].iloc[i]
                signal_type = "Bullish"
                break

            # Example bearish signal: overbought exit
            elif prev_cci > 100 and current_cci < 100:
                mini_df = data.iloc[i - 50:i + 50]
                entry_index = data.index[i]
                entry_price = data["Close"].iloc[i]
                signal_type = "Bearish"
                break

        # --- Build Plot ---
        figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Price Action", "CCI (20)"),
                               row_heights=[0.7, 0.3])

        # Candlestick chart
        figure.add_trace(go.Candlestick(
            x=mini_df.index,
            open=mini_df["Open"],
            high=mini_df["High"],
            low=mini_df["Low"],
            close=mini_df["Close"],
            name="Candlestick"),
            row=1, col=1
        )

        # Entry marker
        figure.add_trace(go.Scatter(
            x=[entry_index],
            y=[entry_price],
            mode="markers",
            marker=dict(size=12, color="MediumPurple", symbol="x"),
            name=f"{signal_type} Entry"),
            row=1, col=1
        )

        # CCI line
        figure.add_trace(go.Scatter(
            x=mini_df.index,
            y=mini_df["cci"],  # must exist in mini_df
            mode="lines",
            line=dict(color="orange"),
            name="CCI (20)"),
            row=2, col=1
        )

        # Overbought / Oversold reference lines
        figure.add_hline(y=100, line=dict(color="red", dash="dot"), row=2, col=1)
        figure.add_hline(y=-100, line=dict(color="green", dash="dot"), row=2, col=1)

        figure.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats,figure


#=============================================================================
# STRATEGY 6 RUNNER FUNCTION
#=============================================================================

if __name__ == '__main__':
    @st.cache_resource
    def run_strat_06(data):
        bb = BollingerBands(close=data["Close"], window=20, window_dev=2)
        data["bb_mid"] = bb.bollinger_mavg()
        data["bb_upper"] = bb.bollinger_hband()
        data["bb_lower"] = bb.bollinger_lband()
        bt = Backtest(data, backtest_strategies.Strategy_06, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 8)],
                            tpperc=[x / 100 for x in range(1, 8)],
                            bb_window=[13, 20, 30],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")

        # Example entry for visualization
        mini_df, entry_index, entry_price = None, None, None
        for i in range(50, len(data)):
            if data["Close"].iloc[i] < data["bb_lower"].iloc[i]:
                mini_df = data.iloc[i - 50:i + 50]
                entry_index, entry_price = data.index[i], data["Close"].iloc[i]
                break
            elif data["Close"].iloc[i] > data["bb_upper"].iloc[i]:
                mini_df = data.iloc[i - 50:i + 50]
                entry_index, entry_price = data.index[i], data["Close"].iloc[i]
                break

        # --- Plot ---
        figure = make_subplots(rows=1, cols=1, subplot_titles=("Bollinger Band Reversion"))

        figure.add_trace(go.Candlestick(
            x=mini_df.index if mini_df is not None else data.index,
            open=(mini_df["Open"] if mini_df is not None else data["Open"]),
            high=(mini_df["High"] if mini_df is not None else data["High"]),
            low=(mini_df["Low"] if mini_df is not None else data["Low"]),
            close=(mini_df["Close"] if mini_df is not None else data["Close"]),
            name="Candlestick"),
            row=1, col=1
        )

        if entry_index is not None:
            figure.add_trace(go.Scatter(
                x=[entry_index],
                y=[entry_price],
                mode="markers",
                marker=dict(size=12, color="MediumPurple", symbol="x"),
                name="Entry Condition"),
                row=1, col=1
            )

        # Bollinger bands
        figure.add_trace(go.Scatter(x=data.index, y=data["bb_upper"], mode="lines",
                                    line=dict(color="red", dash="dot"), name="Upper Band"), row=1, col=1)
        figure.add_trace(go.Scatter(x=data.index, y=data["bb_mid"], mode="lines",
                                    line=dict(color="blue"), name="Middle Band"), row=1, col=1)
        figure.add_trace(go.Scatter(x=data.index, y=data["bb_lower"], mode="lines",
                                    line=dict(color="green", dash="dot"), name="Lower Band"), row=1, col=1)

        figure.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )

        return stats, figure

#=============================================================================
# STRATEGY 7 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_07(data, atr_period, st_mult):
        data = data.copy()
        # Supertrend calculation
        tr1 = data["High"] - data["Low"]
        tr2 = (data["High"] - data["Close"].shift(1)).abs()
        tr3 = (data["Low"] - data["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()
        hl2 = (data["High"] + data["Low"]) / 2.0
        upperbasic = hl2 + st_mult * atr
        lowerbasic = hl2 - st_mult * atr

        upperband = upperbasic.copy()
        lowerband = lowerbasic.copy()
        for i in range(1, len(data)):
            upperband.iat[i] = min(upperbasic.iat[i], upperband.iat[i-1]) if data["Close"].iat[i-1] > upperband.iat[i-1] else upperbasic.iat[i]
            lowerband.iat[i] = max(lowerbasic.iat[i], lowerband.iat[i-1]) if data["Close"].iat[i-1] < lowerband.iat[i-1] else lowerbasic.iat[i]

        st_trend = pd.Series(np.nan, index=data.index)
        st_dir = pd.Series(1, index=data.index)
        for i in range(1, len(data)):
            prev = st_trend.iat[i-1] if not np.isnan(st_trend.iat[i-1]) else lowerband.iat[i]
            if data["Close"].iat[i] > upperband.iat[i]:
                st_trend.iat[i] = lowerband.iat[i]; st_dir.iat[i] = 1
            elif data["Close"].iat[i] < lowerband.iat[i]:
                st_trend.iat[i] = upperband.iat[i]; st_dir.iat[i] = -1
            else:
                st_trend.iat[i] = prev
                st_dir.iat[i] = st_dir.iat[i-1]
        st_flip = st_dir.diff().fillna(0).ne(0)

        data["supertrend"] = st_trend
        data["st_dir"] = st_dir
        data["st_flip"] = st_flip

        bt = Backtest(data, backtest_strategies.Strategy_07, cash=5000, margin=1/5, commission=0.0002, finalize_trades=True)
        stats = bt.optimize(slperc=[x/100 for x in range(1,11)],
                            tpperc=[x/100 for x in range(1,11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")

        # basic viz
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Supertrend",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"], name="Candles"), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["supertrend"], name="Supertrend", mode="lines"), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 8 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_08(data, k, d, smooth, ob_level, os_level):
        data = data.copy()
        so = StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"],
                                  window=k, smooth_window=smooth)
        data["stoch_k"] = so.stoch()
        data["stoch_d"] = data["stoch_k"].rolling(d).mean()
        data["k_cross_up"] = (data["stoch_k"].shift(1) <= data["stoch_d"].shift(1)) & (
                    data["stoch_k"] > data["stoch_d"])
        data["k_cross_dn"] = (data["stoch_k"].shift(1) >= data["stoch_d"].shift(1)) & (
                    data["stoch_k"] < data["stoch_d"])
        data["ob_level"] = ob_level
        data["os_level"] = os_level
        bt = Backtest(data, backtest_strategies.Strategy_08, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Price", "Stochastic"))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"], name="Candles"), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["stoch_k"], name="%K"), 2, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["stoch_d"], name="%D"), 2, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 9 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_09(data, donch_lookback):
        data = data.copy()
        data["donch_h"] = data["High"].rolling(donch_lookback).max()
        data["donch_l"] = data["Low"].rolling(donch_lookback).min()
        bt = Backtest(data, backtest_strategies.Strategy_09, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Donchian Breakout",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"], name="Candles"), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["donch_h"], name="Upper"), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["donch_l"], name="Lower"), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 10 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_10(data, psar_af, psar_max):
        data = data.copy()
        psar = PSARIndicator(high=data["High"], low=data["Low"], close=data["Close"], step=psar_af, max_step=psar_max)
        data["psar"] = psar.psar()
        bt = Backtest(data, backtest_strategies.Strategy_10, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Parabolic SAR",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"], name="Candles"), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["psar"], name="PSAR", mode="markers"), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 11 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_11(data, tenkan=9, kijun=26, senkou=52):
        data = data.copy()
        ichi = IchimokuIndicator(high=data["High"], low=data["Low"], window1=tenkan, window2=kijun, window3=senkou)
        data["tenkan"] = ichi.ichimoku_conversion_line()
        data["kijun"] = ichi.ichimoku_base_line()
        data["span_a"] = ichi.ichimoku_a()
        data["span_b"] = ichi.ichimoku_b()
        bt = Backtest(data, backtest_strategies.Strategy_11, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Ichimoku",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["tenkan"], name="Tenkan"), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["kijun"], name="Kijun"), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 12 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_12(data, adx_period=14, adx_thr=20):
        data = data.copy()
        adx = ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=adx_period)
        data["di_pos"] = adx.adx_pos()
        data["di_neg"] = adx.adx_neg()
        data["adx"] = adx.adx()
        data["adx_thr"] = adx_thr
        bt = Backtest(data, backtest_strategies.Strategy_12, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Price", "ADX"))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["adx"], name="ADX"), 2, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 13 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_13(data, bb_window=20, bb_dev=2.0, kc_window=20, kc_mult=1.5):
        data = data.copy()
        bb = BollingerBands(close=data["Close"], window=bb_window, window_dev=bb_dev)
        data["bb_upper"] = bb.bollinger_hband()
        data["bb_lower"] = bb.bollinger_lband()
        data["bb_mid"] = bb.bollinger_mavg()
        kc = KeltnerChannel(high=data["High"], low=data["Low"], close=data["Close"], window=kc_window,
                            window_atr=kc_window, original_version=True)
        data["kc_upper"] = kc.keltner_channel_hband()
        data["kc_lower"] = kc.keltner_channel_lband()
        # “Squeeze” when BB entirely within KC
        data["squeeze"] = (data["bb_upper"] < data["kc_upper"]) & (data["bb_lower"] > data["kc_lower"])
        bt = Backtest(data, backtest_strategies.Strategy_13, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("BB Squeeze",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["bb_upper"], name="BB Upper"), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["bb_lower"], name="BB Lower"), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 14 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_14(data, swing_lookback=20, rsi_window=14, rsi_long_max=40, rsi_short_min=60):
        data = data.copy()
        # swing highs/lows
        data["swing_high"] = data["High"].rolling(swing_lookback).max()
        data["swing_low"] = data["Low"].rolling(swing_lookback).min()
        # 61.8% retracement zones
        up_range = data["swing_high"] - data["swing_low"]
        data["fib_618_up"] = data["swing_high"] - 0.618 * up_range
        data["fib_618_dn"] = data["swing_low"] + 0.618 * up_range
        rsi = RSIIndicator(close=data["Close"], window=rsi_window).rsi()
        data["rsi"] = rsi
        # zones: long if price bounces above fib_618_up with RSI < rsi_long_max; short if below fib_618_dn with RSI > rsi_short_min
        data["fib_long_zone"] = (data["Close"].shift(1) <= data["fib_618_up"].shift(1)) & (
                    data["Close"] > data["fib_618_up"]) & (rsi < rsi_long_max)
        data["fib_short_zone"] = (data["Close"].shift(1) >= data["fib_618_dn"].shift(1)) & (
                    data["Close"] < data["fib_618_dn"]) & (rsi > rsi_short_min)
        bt = Backtest(data, backtest_strategies.Strategy_14, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Fibonacci Bounce",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["fib_618_up"], name="Fib 61.8% Up"), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["fib_618_dn"], name="Fib 61.8% Dn"), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 15 RUNNER FUNCTION
#=============================================================================

if __name__ == '__main__':
    @st.cache_resource
    def run_strat_15(data, z_window=50, z_entry=2.0, z_exit=0.3):
        data = data.copy()
        ma = data["Close"].rolling(z_window).mean()
        sd = data["Close"].rolling(z_window).std()
        data["zscore"] = (data["Close"] - ma) / sd
        data["z_entry"] = z_entry
        data["z_exit"] = z_exit
        bt = Backtest(data, backtest_strategies.Strategy_15, cash=5000, margin=1/5, commission=0.0002, finalize_trades=True)
        stats = bt.optimize(slperc=[x/100 for x in range(1,11)],
                            tpperc=[x/100 for x in range(1,11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Price", "Z-Score"))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["zscore"], name="z"), 2, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#============================================================================
# STRATEGY 16 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_16(data, vwap_window=20):
        data = data.copy()
        # rolling VWAP: sum(price*vol)/sum(vol) over window
        pv = (data["Close"] * data["Volume"]).rolling(vwap_window).sum()
        v = data["Volume"].rolling(vwap_window).sum()
        data["vwap"] = pv / v
        bt = Backtest(data, backtest_strategies.Strategy_16, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("VWAP Crossover",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["vwap"], name="VWAP"), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 17 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_17(data, ema_trend=50):
        data = data.copy()
        # Heikin-Ashi
        ha_close = (data["Open"] + data["High"] + data["Low"] + data["Close"]) / 4.0
        ha_open = ha_close.copy()
        ha_open.iloc[0] = data["Open"].iloc[0]
        for i in range(1, len(ha_open)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
        data["ha_open"] = ha_open
        data["ha_close"] = ha_close
        data["ema_trend"] = EMAIndicator(close=data["Close"], window=ema_trend).ema_indicator()
        bt = Backtest(data, backtest_strategies.Strategy_17, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Heikin-Ashi + EMA",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["ema_trend"], name="EMA Trend"), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 18 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_18(data):
        data = data.copy()
        rng = data["High"] - data["Low"]
        data["nr7"] = rng == rng.rolling(7).min()
        data["range_high"] = data["High"].rolling(7).max()
        data["range_low"] = data["Low"].rolling(7).min()
        bt = Backtest(data, backtest_strategies.Strategy_18, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("NR7 Breakout",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 19 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_19(data, kc_window=20, kc_mult=2.0):
        data = data.copy()
        kc = KeltnerChannel(high=data["High"], low=data["Low"], close=data["Close"], window=kc_window,
                            window_atr=kc_window, original_version=True)
        data["kc_upper"] = kc.keltner_channel_hband()
        data["kc_lower"] = kc.keltner_channel_lband()
        bt = Backtest(data, backtest_strategies.Strategy_19, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Keltner Channel",))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["kc_upper"], name="KC Upper"), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["kc_lower"], name="KC Lower"), 1, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig
#=============================================================================
# STRATEGY 20 RUNNER FUNCTION
#=============================================================================
if __name__ == '__main__':
    @st.cache_resource
    def run_strat_20(data, ao_s=5, ao_l=34):
        data = data.copy()
        ao = AwesomeOscillatorIndicator(high=data["High"], low=data["Low"], window1=ao_s, window2=ao_l)
        data["ao"] = ao.awesome_oscillator()
        # Strategy_20 computes AO internally via self.I for signal timing; we also store for plotting if needed.
        bt = Backtest(data, backtest_strategies.Strategy_20, cash=5000, margin=1 / 5, commission=0.0002,
                      finalize_trades=True)
        stats = bt.optimize(slperc=[x / 100 for x in range(1, 11)],
                            tpperc=[x / 100 for x in range(1, 11)],
                            constraint=lambda x: x.slperc < x.tpperc,
                            maximize="Return [%]")
        mini_df = data.tail(200)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Price", "AO"))
        fig.add_trace(go.Candlestick(x=mini_df.index, open=mini_df["Open"], high=mini_df["High"],
                                     low=mini_df["Low"], close=mini_df["Close"]), 1, 1)
        fig.add_trace(go.Scatter(x=mini_df.index, y=mini_df["ao"], name="AO"), 2, 1)
        fig.update_layout(
            width=800,
            height=600,
            plot_bgcolor="black",
            paper_bgcolor="black",
            font=dict(color="white", size=14),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(family="sans-serif", size=12, color="white"),
                bgcolor="black",
                bordercolor="grey",
                borderwidth=2,
            )
        )
        return stats, fig

#=============================================================================
# STRATEGY 1 MAIN FUNCTION
#=============================================================================

def start_01(df):
    data=df.copy()
    st.markdown("<div class='info-container'>",unsafe_allow_html=True)
    st.info("🥶 Enter the ema from below")
    ema1,ema2=None,None
    c1,c2=st.columns(2)
    with c1:
        ema1=st.number_input(min_value=2,max_value=200,step=1,value=20,label="Enter Here",key="ema1")
    with c2:
        ema2=st.number_input(min_value=2,max_value=200,step=1,value=50,label="Enter Here",key="ema2")
    if st.button("👁️Start Evaluation",key="start01"):
        if(ema1==ema2):
            st.error("❌ Ema cant be same choose different values")
        else:
            small=min(ema1,ema2)
            big=max(ema1,ema2)
            stats,figure=run_strat_01(data, small, big)
            show_results(stats,figure)


#=============================================================================
# STRATEGY 2 MAIN FUNCTION
#=============================================================================

def strat_02(df):
    data=df.copy()
    with st.expander("Click to view different candlesticks patterns"):
        st.header("🐂 Bullish Reversal Patterns", divider='rainbow')
        st.write("These patterns typically appear after a downtrend and signal a potential reversal to an uptrend.")
        # Hammer
        st.subheader("Hammer")
        col1, col2 = st.columns([1, 2],gap="small")
        with col1:
            st.image("Asset/hammer.png", width=400,caption="Hammer")
        with col2:
            st.write(
                """
                A **Hammer** has a short body at the top with a long lower wick and little to no upper wick.
                - **Appearance:** Looks like a hammer.
                - **Indication:** Signals that sellers pushed the price down, but buyers came in strongly to push it back up, suggesting a potential bottom and a **bullish reversal**.
                """
            )
        st.divider()
        # Bullish Engulfing
        st.subheader("Bullish Engulfing")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("Asset/bullish_engulf.png", use_container_width=True,caption="Bullish Engulfing")
        with col2:
            st.write(
                """
                This is a two-candle pattern. A smaller bearish candle is followed by a larger bullish candle that completely "engulfs" the body of the previous candle.
                - **Appearance:** A small red candle followed by a large green candle.
                - **Indication:** Shows a strong shift in momentum from sellers to buyers, signaling a powerful **bullish reversal**.
                """
            )
        st.divider()
        # Morning Star
        st.subheader("Morning Star")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("Asset/morning_star.png", caption="Morning Star Pattern", use_container_width=True)
        with col2:
            st.write(
                """
                A three-candle pattern: a long bearish candle, followed by a small-bodied candle (or Doji), and then a long bullish candle.
                - **Appearance:** A big red, a small middle, and a big green candle.
                - **Indication:** Represents a moment of indecision followed by a strong buyer takeover, signaling a likely **bullish reversal**.
                """
            )
        # --- Bearish Patterns ---
        st.header("🐻 Bearish Reversal Patterns", divider='rainbow')
        st.write("These patterns typically appear after an uptrend and signal a potential reversal to a downtrend.")

        # Shooting Star
        st.subheader("Shooting Star")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("Asset/shooting_star.png", caption="Bearish Shooting Star", use_container_width=True)
        with col2:
            st.write(
                """
                The inverse of a Hammer. It has a short body at the bottom with a long upper wick and little to no lower wick.
                - **Appearance:** An inverted hammer at the top of an uptrend.
                - **Indication:** Shows that buyers tried to push the price up, but sellers overpowered them, suggesting a potential peak and a **bearish reversal**.
                """
            )
        st.divider()
        # Bearish Engulfing
        st.subheader("Bearish Engulfing")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("Asset/bearish_engulf.png", caption="Bearish Engulfing Pattern", use_container_width=True)
        with col2:
            st.write(
                """
                A smaller bullish candle is followed by a larger bearish candle that completely "engulfs" the body of the previous one.
                - **Appearance:** A small green candle followed by a large red candle.
                - **Indication:** Represents a significant momentum shift from buyers to sellers, signaling a strong **bearish reversal**.
                """
            )
        st.divider()
        # --- Indecision Pattern ---
        st.header("🤔 Indecision Patterns", divider='rainbow')
        st.write(
            "These patterns can appear anytime and signal a potential pause, continuation, or reversal in the trend.")
        # Doji
        st.subheader("Doji")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image("Asset/doji.png", caption="Doji Candlestick", use_container_width=True)
        with col2:
            st.write(
                """
                A Doji forms when the open and close prices are virtually equal, resulting in a very small or non-existent body. It looks like a cross or plus sign.
                - **Appearance:** A cross shape with varying wick lengths.
                - **Indication:** Symbolizes **indecision** in the market. Neither buyers nor sellers were able to gain control. It can often precede a trend reversal.
                """
            )
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    if st.button("👁️Start Evaluation", key="start02"):
        stats,figure=run_strat_02(data)
        show_results(stats, figure)




#=============================================================================
# STRATEGY 3 MAIN FUNCTION
#=============================================================================

def strat_03(df):
    data=df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    st.write("Enter the RSI window")
    window = st.number_input(min_value=2, max_value=200, step=1, value=20, label="Enter Rsi Window Here", key="rsi_1")
    if st.button("👁️Start Evaluation", key="start03"):
        stats, figure = run_strat_03(data,window)
        show_results(stats, figure)

#=============================================================================
# STRATEGY 4 MAIN FUNCTION
#=============================================================================
def strat_04(df):
    data=df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    if st.button("👁️Start Evaluation", key="start04"):
        stats, figure = run_strat_04(data)
        show_results(stats, figure)

#=============================================================================
# STRATEGY 5 MAIN FUNCTION
#=============================================================================
def strat_05(df):
    data=df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    if st.button("👁️Start Evaluation", key="start05"):
        stats, figure = run_strat_05(data)
        show_results(stats, figure)


#=============================================================================
# STRATEGY 6 MAIN FUNCTION
#=============================================================================
def strat_06(df):
    data=df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    if st.button("👁️Start Evaluation", key="start06"):
        stats, figure = run_strat_06(data)
        show_results(stats, figure)
#=============================================================================
# STRATEGY 7 MAIN FUNCTION
#=============================================================================

def strat_07(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2 = st.columns(2)
    with c1: atr_period = st.number_input("ATR Period", 2, 100, 10, 1,key="atr_period")
    with c2: st_mult = st.number_input("Multiplier", 1.0, 10.0, 3.0, 0.1,key="st_mult")
    if st.button("👁️ Start Evaluation", key="start07"):
        stats, fig = run_strat_07(data, atr_period, st_mult)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 8 MAIN FUNCTION
#=============================================================================

def strat_08(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2, c3 = st.columns(3)
    with c1: k = st.number_input("%K Window", 2, 100, 14, 1,key="k")
    with c2: d = st.number_input("%D Smoothing", 2, 100, 3, 1,key="d")
    with c3: smooth = st.number_input("K Smoothing", 1, 10, 3, 1,key="smooth")
    c4, c5 = st.columns(2)
    with c4: ob = st.number_input("Overbought", 50, 100, 80, 1,key="ob")
    with c5: os = st.number_input("Oversold", 0, 50, 20, 1,key="os")
    if st.button("👁️ Start Evaluation", key="start08"):
        stats, fig = run_strat_08(data, k, d, smooth, ob, os)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 9 MAIN FUNCTION
#=============================================================================

def strat_09(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    lb = st.number_input("Donchian Lookback", 2, 200, 20, 1,key="lb_123")
    if st.button("👁️ Start Evaluation", key="start09"):
        stats, fig = run_strat_09(data, lb)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 10 MAIN FUNCTION
#=============================================================================

def strat_10(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2 = st.columns(2)
    with c1: af = st.number_input("PSAR AF (step)", 0.01, 0.5, 0.02, 0.01,key="af_11")
    with c2: mx = st.number_input("PSAR Max AF", 0.1, 1.0, 0.2, 0.01,key="mx_11")
    if st.button("👁️ Start Evaluation", key="start10"):
        stats, fig = run_strat_10(data, af, mx)
        show_results(stats, fig)

#=============================================================================
# STRATEGY 11 MAIN FUNCTION
#=============================================================================

def strat_11(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2, c3 = st.columns(3)
    with c1: tenkan = st.number_input("Tenkan", 2, 50, 9, 1,key="tenkan")
    with c2: kijun = st.number_input("Kijun", 5, 100, 26, 1,key="kijun")
    with c3: senkou = st.number_input("Senkou", 10, 150, 52, 1,key="senkou")
    if st.button("👁️ Start Evaluation", key="start11"):
        stats, fig = run_strat_11(data, tenkan, kijun, senkou)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 12 MAIN FUNCTION
#=============================================================================

def strat_12(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2 = st.columns(2)
    with c1: adx_p = st.number_input("ADX Period", 5, 100, 14, 1,key="adx_p_11")
    with c2: thr = st.number_input("ADX Threshold", 5, 50, 20, 1,key="thr_11")
    if st.button("👁️ Start Evaluation", key="start12"):
        stats, fig = run_strat_12(data, adx_p, thr)
        show_results(stats, fig)

#=============================================================================
# STRATEGY 13 MAIN FUNCTION
#=============================================================================

def strat_13(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2, c3, c4 = st.columns(4)
    with c1: bb_w = st.number_input("BB Window", 5, 100, 20, 1,key="bb_w_13")
    with c2: bb_d = st.number_input("BB Dev", 1.0, 4.0, 2.0, 0.1,key="bb_d_13")
    with c3: kc_w = st.number_input("KC Window", 5, 100, 20, 1,key="kc_w_13")
    with c4: kc_m = st.number_input("KC Mult", 1.0, 4.0, 1.5, 0.1,key="kc_m_13")
    if st.button("👁️ Start Evaluation", key="start13"):
        stats, fig = run_strat_13(data, bb_w, bb_d, kc_w, kc_m)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 14 MAIN FUNCTION
#=============================================================================
def strat_14(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2, c3, c4 = st.columns(4)
    with c1: sw = st.number_input("Swing Lookback", 5, 200, 20, 1,key="sw_14")
    with c2: rw = st.number_input("RSI Window", 2, 100, 14, 1,key="rw_14")
    with c3: rlong = st.number_input("RSI Long Max", 10, 60, 40, 1,key="rlong_14")
    with c4: rshort = st.number_input("RSI Short Min", 40, 90, 60, 1,key="rshort_14")
    if st.button("👁️ Start Evaluation", key="start14"):
        stats, fig = run_strat_14(data, sw, rw, rlong, rshort)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 15 MAIN FUNCTION
#=============================================================================
def strat_15(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2, c3 = st.columns(3)
    with c1: zw = st.number_input("Z-Score Window", 10, 200, 50, 1,key="zw_15")
    with c2: ze = st.number_input("Z-Entry Threshold", 0.5, 5.0, 2.0, 0.1,key="ze_15")
    with c3: zx = st.number_input("Z-Exit Threshold", 0.1, 2.0, 0.3, 0.1,key="zx_15")
    if st.button("👁️ Start Evaluation", key="start15"):
        stats, fig = run_strat_15(data, zw, ze, zx)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 16 MAIN FUNCTION
#=============================================================================
def strat_16(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    vw = st.number_input("VWAP Window", 5, 200, 20, 1,key="vw_16")
    if st.button("👁️ Start Evaluation", key="start16"):
        stats, fig = run_strat_16(data, vw)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 17 MAIN FUNCTION
#=============================================================================
def strat_17(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    et = st.number_input("EMA Trend", 5, 200, 50, 1,key="et_17")
    if st.button("👁️ Start Evaluation", key="start17"):
        stats, fig = run_strat_17(data, et)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 18 MAIN FUNCTION
#=============================================================================
def strat_18(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    st.info("NR7 Range Contraction — no extra parameters required")
    if st.button("👁️ Start Evaluation", key="start18"):
        stats, fig = run_strat_18(data)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 19 MAIN FUNCTION
#=============================================================================
def strat_19(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2 = st.columns(2)
    with c1: kc_w = st.number_input("KC Window", 5, 100, 20, 1,key="kc_w_19")
    with c2: kc_m = st.number_input("KC Multiplier", 1.0, 4.0, 2.0, 0.1,key="kc_m_19")
    if st.button("👁️ Start Evaluation", key="start19"):
        stats, fig = run_strat_19(data, kc_w, kc_m)
        show_results(stats, fig)
#=============================================================================
# STRATEGY 20 MAIN FUNCTION
#=============================================================================
def strat_20(df):
    data = df.copy()
    st.markdown("<div class='info-container'>", unsafe_allow_html=True)
    st.info("🥶 Click the button below to start the strategy")
    c1, c2 = st.columns(2)
    with c1: ao_s = st.number_input("AO Short Window", 2, 50, 5, 1,key="ao_s_20")
    with c2: ao_l = st.number_input("AO Long Window", 10, 100, 34, 1,key="ao_l_20")
    if st.button("👁️ Start Evaluation", key="start20"):
        stats, fig = run_strat_20(data, ao_s, ao_l)
        show_results(stats, fig)

#=============================================================================
# MAIN BODY OF CODE
#=============================================================================





def initialize():
    if "df_uploaded" not in st.session_state: st.session_state.df_uploaded = None
    if "pointer" not in st.session_state: st.session_state.pointer = None
    if "data_selector" not in st.session_state: st.session_state.data_selector = None
    if "load_strategy" not in st.session_state: st.session_state.load_strategy = None
initialize()


with st.container():
    st.markdown("<h1 style='text-align: center; font-size:4rem'>Backtesting Zone</h1>",unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; max-width: 800px; margin: auto;'>In this part"
        " you can try and test multiple backtesting strategies with your own data or we have some samples data as well.</p>",
        unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2,c3,c4,c5=st.columns([1,1,1,1,1])
    with c3:
        if st.button("🔍 Show Strategies",key="show_features"):
            show_features()
    st.markdown("<div class='main-container'>",unsafe_allow_html=True)

with (st.container()):
    if st.session_state.df_uploaded is None:
        st.markdown("<h2 style='text-align:left'>🏁 Step 1: Data Selection</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.pointer==None:
            st.info("🛷 Choose an option : You want to upload file or work on existing files")
            col1,col2=st.columns(2)
            with col1:
                if st.button("🐶 Upload a File"):
                    st.session_state.pointer=1
                    st.session_state.data_selector=1
                    st.rerun()
            with col2:
                if st.button("🐶 Choose a File"):
                    st.session_state.pointer = 1
                    st.session_state.data_selector = 2
                    st.rerun()

        if st.session_state.pointer is not None and st.session_state.data_selector==1:
            st.markdown("<p style:font-size:1.5rem>Upload your dataset in <b>CSV</b> or <b>Excel</b> "
                             "format to begin. For best result upload a clean and "
                             "well structured data</p>",unsafe_allow_html=True)
            uploaded_file=st.file_uploader("Drag and drop File here",type=["csv","xlsx","xls"],label_visibility="collapsed")
            st.warning("⚠️ Make sure your file should have columns ['Date','Open','High','Low','Close','Volume'] and date"
                       "in standard format.")
            if uploaded_file is not None:
                try:
                    with st.spinner('Reading your file...'):
                        if uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                        else:
                            df = pd.read_excel(uploaded_file)
                    # Store the uploaded dataframe in session state to persist it.
                    L=['date','Open','High','Low','Close','Volume']
                    if(len(df.columns)==len(L)):
                        df.columns=['date','Open','High','Low','Close','Volume']
                        df["date"]=pd.to_datetime(df["date"])
                        df.set_index("date",inplace=True)
                        st.session_state.df_uploaded = df
                        st.rerun()  # Rerun the script to move to the next step.
                    else:
                        st.error("❌ Extra columns present in data.")
                except Exception as e:
                    st.error(f"❌ Error: The file could not be read. Details: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.pointer is not None and st.session_state.data_selector == 2:
            st.info("These all are Forex data from and US Stock")
            c1,c2,c3,c4,c5,c6,c7,c8=st.columns(8)
            with c1:
                check1=st.checkbox("1 Minute Data")
            with c2:
                check2=st.checkbox("5 Minute Data")
            with c3:
                check3=st.checkbox("15 Minute Data")
            with c4:
                check4=st.checkbox("30 Minute Data")
            with c5:
                check5=st.checkbox("1 Hour Data")
            with c6:
                check6=st.checkbox("4 Hour Data")
            with c7:
                check7=st.checkbox("1 Day Data")
            with c8:
                check8=st.checkbox("US Stocks")
            Total_checks=[1 for x in [check1,check2,check3,check4,check5,check6,check7,check8] if x==True]
            Total_sum=sum(Total_checks)
            selected_file = None
            if(Total_sum>1):
                st.error("❌ You can select only one timeframe at a time")
            elif(Total_sum==1):
                if not check8:
                    data_list=["--Choose--","EURUSD","GBPUSD","USDCAD","USDCHF","USDJPY"]
                if check8:
                    data_list=["--Chosse--","Amazon","Apple","Cisco","Meta","Microsoft","Netflix","QCom","Starbucks","Tesla"]
                file_name=st.selectbox(options=data_list,label="Choose a File from below")
                if st.button("👍 Continue",key="Continue_!"):
                    if file_name=="--Choose--":
                        st.error("❌ Please specify file")
                    else:
                        if check1:
                            selected_file=pd.read_csv(f"./historical_data/{file_name}{1}.csv",sep="\t",header=None,names=["date","Open","High","Low","Close","Volume"],index_col="date")
                        elif check2:
                            selected_file=pd.read_csv(f"./historical_data/{file_name}{5}.csv",sep="\t",header=None,names=["date","Open","High","Low","Close","Volume"],index_col="date")
                        elif check3:
                            selected_file=pd.read_csv(f"./historical_data/{file_name}{15}.csv",sep="\t",header=None,names=["date","Open","High","Low","Close","Volume"],index_col="date")
                        elif check4:
                            selected_file=pd.read_csv(f"./historical_data/{file_name}{30}.csv",sep="\t",header=None,names=["date","Open","High","Low","Close","Volume"],index_col="date")
                        elif check5:
                            selected_file=pd.read_csv(f"./historical_data/{file_name}{60}.csv",sep="\t",header=None,names=["date","Open","High","Low","Close","Volume"],index_col="date")
                        elif check6:
                            selected_file=pd.read_csv(f"./historical_data/{file_name}{240}.csv",sep="\t",header=None,names=["date","Open","High","Low","Close","Volume"],index_col="date")
                        elif check7:
                            selected_file=pd.read_csv(f"./historical_data/{file_name}{1440}.csv",sep="\t",header=None,names=["date","Open","High","Low","Close","Volume"],index_col="date")
                        elif check8:
                            selected_file=pd.read_csv(f"./historical_data/{file_name}.csv",index_col="Date")
                            selected_file.index.name="date"
                            for colum in selected_file.columns:
                                if(colum!="Volume"):
                                    selected_file[colum]=selected_file[colum].apply(lambda x:x.replace("$",""))
                            selected_file=selected_file.astype(np.float64)
                            selected_file.columns=["Close","Volume","Open","High","Low"]
                        else:
                            st.error("❌ Please specify a timeframe")
                if selected_file is not None:
                    selected_file.index=pd.to_datetime(selected_file.index)
                    st.session_state.df_uploaded = selected_file
                    st.rerun()

    if st.session_state.df_uploaded is not None:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:left'>🔍 Step 2: Preview & Inspect</h2>", unsafe_allow_html=True)
        st.dataframe(st.session_state.df_uploaded.head())
        st.session_state.load_strategy=1
        if st.button("🔄 Upload a Different File"):
            # Clear all session state variables to reset the app.
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("📊 View Detailed Analysis", expanded=True):
            highest_high = st.session_state.df_uploaded['High'].max()
            lowest_low = st.session_state.df_uploaded['Low'].min()
            avg_daily_range = (st.session_state.df_uploaded['High'] - st.session_state.df_uploaded['Low']).mean()
            start_price = st.session_state.df_uploaded['Close'].iloc[0]
            end_price = st.session_state.df_uploaded['Close'].iloc[-1]
            net_change = end_price - start_price
            percentage_change = (net_change / start_price) * 100

            start=st.session_state.df_uploaded.index[0]
            end=st.session_state.df_uploaded.index[-1]
            col1,col2=st.columns(2)

            with col1:
                st.metric(
                    label="Starting Date",
                    value=f"{start}"
                )
            with col2:
                st.metric(
                    label="Ending Date",
                    value=f"{end}"
                )
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    label="Highest High 🔼",
                    value=f"{highest_high:.2f}"
                )
            with col2:
                st.metric(
                    label="Lowest Low 🔽",
                    value=f"{lowest_low:.2f}"
                )
            with col3:
                st.metric(
                    label="Avg. Candle  Range ↔️",
                    value=f"{avg_daily_range:.6f}",
                    help="The average difference between the high and low price each day."
                )
            with col4:
                st.metric(
                    label="Net Change 📊",
                    value=f"{net_change:.2f}",
                    delta=f"{percentage_change:.2f}%"  # Shows the change from the start
                )
            fig,ax=plt.subplots(2,2,figsize=(15,6),sharex=True)
            Weekly=st.session_state.df_uploaded.resample("W").max()
            Monthly=st.session_state.df_uploaded.resample("ME").max()
            Quaterly=st.session_state.df_uploaded.resample("4ME").max()
            Yearly=st.session_state.df_uploaded.resample("YE").max()
            with st.expander("📈 View Price Movements", expanded=True):
                # Create a list of available metrics
                metric_options = ["Close", "Volume"]

                # Create a selectbox for the user to choose
                selected_metric = st.selectbox("Select a metric to display:", metric_options)

                # Create the plot
                fig, ax = plt.subplots(2, 2, figsize=(15, 6), sharex=True)
                fig.suptitle(f"{selected_metric} Movements Across Different Timeframes", fontsize=16)

                # Plot the selected metric dynamically
                ax[0, 0].plot(Weekly.index, Weekly[selected_metric], label="Weekly", color="green")
                ax[0, 0].set_title("Weekly Movement")
                ax[0, 1].plot(Monthly.index, Monthly[selected_metric], label="Monthly", color="blue")
                ax[0, 1].set_title("Monthly Movement")
                ax[1, 0].plot(Quaterly.index, Quaterly[selected_metric], label="Quaterly", color="orange")
                ax[1, 0].set_title("Quaterly Movement")
                ax[1, 1].plot(Yearly.index, Yearly[selected_metric], label="Yearly", color="black")
                ax[1, 1].set_title("Yearly Movement")

                # Set a dynamic Y-label for all subplots
                for subplot in ax.flat:
                    subplot.set_ylabel(selected_metric)

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                st.pyplot(fig)




    if st.session_state.load_strategy is not None:
        st.markdown('<div class="step-container">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:left'>🦍 Step 3: Select Strategies</h2>", unsafe_allow_html=True)
        tab_list = ["Strategy 1🏁","Strategy 2🏁","Strategy 3🏁","Strategy 4🏁","Strategy 5🏁","Strategy 6🏁"
            ,"Strategy 7🏁","Strategy 8🏁","Strategy 9🏁","Strategy 10🏁","Strategy 11🏁"
            ,"Strategy 12🏁","Strategy 13🏁","Strategy 14🏁","Strategy 15🏁","Strategy 16🏁"
                    ,"Strategy 17🏁","Strategy 18🏁","Strategy 19🏁","Strategy 20🏁"]
        t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20= st.tabs(tab_list)
        with t1:
            st.markdown("<h3 style='text-align:left'>💯EMA Crossover</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The EMA Crossover is a popular trend-following strategy"
                        " used in technical analysis to identify potential shifts in market momentum. It involves"
                        " plotting two Exponential Moving Averages (EMAs) of different lengths on a price chart, "
                        "such as a fast-moving 12-period EMA and a slower 20-period EMA. A bullish signal, often called a"
                        " 'golden cross,' occurs when the shorter-term EMA crosses above the longer-term EMA, suggesting that "
                        "upward momentum is building. Conversely, a bearish signal, or 'death cross,' is "
                        "generated when the shorter-term EMA crosses below the longer-term EMA, indicating a potential"
                        " downtrend. While simple to use, the EMA Crossover is a lagging indicator that performs best in "
                        "learly trending markets and can produce false signals in choppy or sideways conditions.</p>", unsafe_allow_html=True)
            start_01(st.session_state.df_uploaded)


        with t2:
            st.markdown("<h3 style='text-align:left'>💯CandleStick Patterns</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>Candlestick patterns are a form of technical analysis "
                        "used by traders to visualize and interpret price movements in financial markets like "
                        "stocks, forex, and cryptocurrencies. 📈 Originating from 18th-century Japanese rice "
                        "traders, each 'candle' represents a specific time period (e.g., one day) and provides "
                        "four key pieces of information: the open, high, low, and close prices. The wide part of "
                        "the candle, called the real body, shows the range between the open and close price, with "
                        "its color indicating whether the price went up (typically green or white) or down "
                        "(typically red or black). The thin lines extending above and below the body, known "
                        "as wicks or shadows, represent the highest and lowest prices reached during that period. "
                        "Traders analyze patterns formed by single or multiple candlesticks, such as the Doji, "
                        "Hammer, or Engulfing patterns, to gauge market sentiment and predict potential trend "
                        "reversals or continuations. While highly popular, these patterns are most effective"
                        " when used in conjunction with other indicators and analysis techniques to confirm "
                        "trading signals..</p>",
                        unsafe_allow_html=True)
            strat_02(st.session_state.df_uploaded)

        with t3:
            st.markdown("<h3 style='text-align:left'>💯RSI Divergence</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The RSI divergence strategy is a powerful tool for "
                        "identifying potential market reversals by spotting a disconnect between price and momentum. "
                        "It occurs when the price of an asset moves in the opposite direction of the Relative Strength"
                        " Index (RSI) indicator. For instance, a bullish divergence forms when the price records a new"
                        " lower low, but the RSI simultaneously prints a higher low, signaling that downward momentum is"
                        " fading and a potential uptrend could be starting. Conversely, a bearish divergence appears when"
                        " the price achieves a new higher high while the RSI makes a lower high, suggesting that the"
                        " buying pressure is weakening and a downtrend may be imminent. This discrepancy indicates that "
                        "the current trend is losing steam. Traders often use this as an early warning signal but"
                        " typically wait for additional confirmation, like a key price level break or a specific "
                        "candlestick pattern, before executing a trade based on the divergence alone..</p>",
                        unsafe_allow_html=True)
            strat_03(st.session_state.df_uploaded)
        with t4:
            st.markdown("<h3 style='text-align:left'>💯MACD Crossover</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The MACD Crossover is a classic trading signal used "
                        "to identify shifts in market momentum, generated by the Moving Average Convergence "
                        "Divergence indicator's two main lines: the MACD line and the Signal line. A bullish "
                        "crossover 📈 occurs when the faster MACD line crosses above the slower Signal line,"
                        " an event often interpreted as a potential 'buy signal' suggesting upward momentum is"
                        " building. Conversely, a bearish crossover 📉 happens when the MACD line crosses below"
                        " the Signal line, which is typically viewed as a 'sell signal' indicating downward momentum"
                        "is taking over. Traders use these crossovers to time market entries and exits, but while"
                        " powerful, the signal is most reliable in trending markets and can give false alarms in "
                        "choppy conditions. Therefore, it's best used with other analysis for confirmation.    .</p>",
                        unsafe_allow_html=True)
            strat_04(st.session_state.df_uploaded)

        with t5:
            st.markdown("<h3 style='text-align:left'>💯CCI Overbought/Oversold</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The Commodity Channel Index (CCI) is a popular momentum oscillator "
                        "used to identify overbought and oversold conditions. It measures the current price level relative"
                        " to an average price level over a specific period. The CCI oscillates around a zero line, with key"
                        " reference levels typically set at +100 and -100. An overbought 📉 condition is signaled when the"
                        " CCI rises above the +100 level. This suggests the asset's price is unusually high and may be due"
                        " for a downward correction. Conversely, an oversold 📈 condition occurs when the CCI falls below "
                        "the -100 level. This indicates the price is unusually low and could be poised for an upward rally."
                        "Traders often wait for the CCI to cross back inside these boundaries to confirm a potential trade"
                        " signal. For example, a move from above +100 back down below it can act as a sell signal."
                        "While effective, the CCI can remain in overbought or oversold territory for extended periods"
                        " during strong trends. Therefore, it's often used with other indicators for confirmation before "
                        "making trading decisions.</p>",
                        unsafe_allow_html=True)
            strat_05(st.session_state.df_uploaded)

        with t6:
            st.markdown("<h3 style='text-align:left'>💯Boillinger Band Reversal</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The Bollinger Band Reversal is a mean-reversion trading strategy "
                        "based on the idea that price tends to revert to the middle band, which is a simple moving "
                        "average. A potential sell signal 📉 occurs when the price touches or moves above the upper "
                        "band, suggesting the market is 'overbought' and likely to correct downwards. Conversely, a "
                        "potential buy signal 📈 is flagged when the price touches or drops below the lower band,"
                        " indicating an 'oversold' condition that may lead to an upward rally. For stronger"
                        " confirmation, traders often wait for a price candle to close back inside the bands "
                        "before entering a trade. The primary risk is that in a strong trend, the price can "
                        "'walk the band,'' causing repeated false signals, making this strategy most effective "
                        "in sideways or range-bound markets.</p>",
                        unsafe_allow_html=True)
            strat_06(st.session_state.df_uploaded)

        with t7:
            st.markdown("<h3 style='text-align:left'>💯Supertrend Indicator</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The Supertrend is a straightforward trend-following indicator"
                        " plotted directly on a price chart to identify the current market direction and provide "
                        "clear signals. It appears as a single line that trails the price, turning green and moving "
                        "below the price to signal an uptrend 🟢, or turning red and moving above the price to "
                        "indicate a downtrend 🔴. A buy signal is generated when the indicator flips from red "
                        "to green, while a sell signal occurs when it flips from green to red. Because it uses "
                        "the Average True Range (ATR) for its calculation, it adjusts for market volatility, "
                        "making it an excellent tool for setting dynamic trailing stop-losses.</p>",
                        unsafe_allow_html=True)
            strat_07(st.session_state.df_uploaded)

        with t8:
            st.markdown("<h3 style='text-align:left'>💯Stochastic Oscillator</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The Stochastic Oscillator is a momentum indicator that "
                        "compares a security's closing price to its price range over a specific period, "
                        "oscillating between 0 and 100 to identify overbought and oversold conditions."
                        " A reading above 80 is considered overbought 📉, signaling the price is near "
                        "the top of its recent range and may be due for a pullback, while a reading below"
                        " 20 is considered oversold 📈, suggesting the price is near the bottom of its "
                        "range and could rally. The indicator features two lines, the main %K line and its "
                        "moving average, the %D line, with a common trading signal being the crossover of "
                        "these two lines, especially when it occurs within the overbought or oversold zones.</p>",
                        unsafe_allow_html=True)
            strat_08(st.session_state.df_uploaded)

        with t9:
            st.markdown("<h3 style='text-align:left'>💯Price Breakout</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>A Price Breakout is a significant trading "
                        "event where an asset's price moves decisively through a well-defined"
                        " level of support or resistance that previously contained it. A bullish"
                        " breakout 📈 occurs when the price moves above a resistance level, "
                        "signaling the potential start of a new uptrend, while a bearish breakout,"
                        " often called a breakdown 📉, happens when the price drops below a support "
                        "level, indicating a potential new downtrend. For a breakout to be considered"
                        " valid and strong, it is typically confirmed by a significant increase in trading "
                        "volume, showing strong conviction behind the move. Traders often watch for breakouts "
                        "following periods of market consolidation, like ranges or chart patterns, to enter "
                        "a trade at the beginning of a new trend.</p>",
                        unsafe_allow_html=True)
            strat_09(st.session_state.df_uploaded)

        with t10:
            st.markdown("<h3 style='text-align:left'>💯Parabolic SAR</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The Parabolic SAR (Stop and Reverse) is a trend-following"
                        " indicator used to determine trend direction and provide dynamic exit points. On a"
                        " chart, it appears as a series of dots either below or above the price candles. When"
                        " the dots are below the price 🟢, it signals an uptrend, and the dots themselves act"
                        " as a trailing stop-loss for a long position. Conversely, when the dots are above the"
                        " price 🔴, it indicates a downtrend and provides a trailing stop for a short position."
                        " A potential trend reversal is signaled when the price crosses the dots, causing them "
                        "to 'flip' to the opposite side, which is the core 'stop and reverse' mechanic of the"
                        " indicator. It works best in markets with clear trends but can generate frequent false"
                        " signals in choppy, sideways conditions.</p>",
                        unsafe_allow_html=True)
            strat_10(st.session_state.df_uploaded)

        with t11:
            st.markdown("<h3 style='text-align:left'>💯Ichimoku Cloud Breakout</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>An Ichimoku Cloud Breakout, also known as a"
                        " Kumo Breakout, is a powerful trend-following signal derived from the"
                        " Ichimoku Kinko Hyo indicator, where the 'Cloud' itself represents a "
                        "dynamic zone of support and resistance. A bullish breakout 📈 occurs when"
                        " the price moves decisively above the Cloud, suggesting a significant resistance "
                        "area has been cleared and a new uptrend is likely starting. Conversely, a bearish"
                        " breakout 📉 happens when the price breaks cleanly below the Cloud, indicating "
                        "a breach of a major support zone and the potential start of a strong downtrend."
                        " The significance of this signal is often greater when the breakout occurs through "
                        "a thicker Cloud, as this implies a more substantial shift in market sentiment and "
                        "confirms the beginning of a potentially durable trend.</p>",
                        unsafe_allow_html=True)
            strat_11(st.session_state.df_uploaded)

        with t12:
            st.markdown("<h3 style='text-align:left'>💯ADX Trend Strength</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The Average Directional Index (ADX) is a "
                        "technical indicator used to measure the strength of a market trend,"
                        " not its actual direction. As a non-directional indicator, a high ADX "
                        "value can signify either a strong uptrend or a strong downtrend. The "
                        "ADX line fluctuates between 0 and 100; a reading below 25 typically"
                        " indicates a weak or sideways market, while a value rising above 25 suggests"
                        " the presence of a strong trend 💪. While the ADX line itself only quantifies"
                        " the trend's momentum, its two companion lines, the Positive Directional Indicator "
                        "(+DI) and the Negative Directional Indicator (-DI), are used to determine whether "
                        "the trend is bullish or bearish. Traders often use the ADX as a filter, choosing"
                        " to deploy trend-following strategies only when the ADX is high and avoiding them "
                        "when it is low.</p>",
                        unsafe_allow_html=True)
            strat_12(st.session_state.df_uploaded)

        with t13:
            st.markdown("<h3 style='text-align:left'>💯Boillinger Band Squeeze</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>A Bollinger Band Squeeze is a technical pattern that occurs "
                        "when market volatility falls to a low level, causing the upper and lower bands to visibly"
                        " tighten or contract around the price. 1  This 'squeeze' signifies a period of consolidation"
                        " and is often seen as the calm before the storm, as it frequently precedes a significant "
                        "increase in volatility and a powerful price move. 2  The actual trading signal comes from "
                        "the subsequent breakout; when the price breaks decisively above the upper band after a "
                        "squeeze, it can signal the start of a strong uptrend 📈, while a break below the lower"
                        " band can indicate the beginning of a downtrend 📉. 3  Following the breakout, the bands "
                        "typically expand rapidly, confirming that volatility has returned to the market</p>",
                        unsafe_allow_html=True)
            strat_13(st.session_state.df_uploaded)

        with t14:
            st.markdown("<h3 style='text-align:left'>💯Fibonacci Retracement Levels</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>Fibonacci"
                        " Retracement is a popular technical analysis tool used to identify potential support and"
                        " resistance levels where a price reversal may occur. 1  After a significant price move "
                        "(a swing high to a swing low, or vice versa), traders apply Fibonacci percentages to the"
                        " chart to predict the extent of a potential pullback. 2  The core idea is that the market"
                        " will 'retrace' a portion of the initial move before continuing in the original direction, "
                        "with the most watched levels being 38.2%, 50%, and 61.8%. 1  These levels are anticipated "
                        "to act as support during an uptrend 📈 or resistance during a downtrend 📉, providing "
                        "strategic zones for traders to enter the market in the direction of the primary trend. </p>",
                        unsafe_allow_html=True)
            strat_14(st.session_state.df_uploaded)

        with t15:
            st.markdown("<h3 style='text-align:left'>💯Pairs Trading</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>Pairs Trading is a market-neutral strategy that aims to"
                        " profit from the price relationship between two highly correlated securities, "
                        "such as two companies in the same industry. The strategy is based on identifying "
                        "a pair of assets whose prices historically move together and then capitalizing on"
                        " temporary deviations from this relationship. When the correlation weakens and one "
                        "asset becomes temporarily overvalued relative to the other, a trader will simultaneously"
                        " sell (short) the outperforming asset and buy (long) the underperforming one ⚖️."
                        " The profit is realized when the prices revert to their historical mean and the"
                        " spread between them narrows, a process that is independent of the overall market's direction.</p>",
                        unsafe_allow_html=True)
            strat_15(st.session_state.df_uploaded)

        with t16:
            st.markdown("<h3 style='text-align:left'>💯VWAP (Volume weighted price crossover)</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The VWAP (Volume Weighted Average Price) Crossover is a popular "
                        "intraday trading signal that uses the VWAP line as a benchmark for short-term momentum. "
                        "The VWAP line represents the average price of a security throughout the day, weighted by"
                        " volume, and often acts as a dynamic level of support or resistance. A bullish crossover"
                        " 📈 occurs when the asset's price moves above the VWAP line, which many traders interpret"
                        " as a sign that buying pressure is increasing and the intraday trend is turning positive. "
                        "Conversely, a bearish crossover 📉 happens when the price falls below the VWAP line, suggesting"
                        " that sellers are gaining control and momentum is shifting downwards for the day.</p>",
                        unsafe_allow_html=True)
            strat_16(st.session_state.df_uploaded)

        with t17:
            st.markdown("<h3 style='text-align:left'>💯Heikin-Ashi Trend Following</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>Heikin-Ashi, which means 'average bar' in Japanese, is a "
                        "candlestick technique that smooths out price action to make identifying market trends easier."
                        " Unlike standard candles, Heikin-Ashi uses an averaging formula that filters out minor market "
                        "noise, resulting in a cleaner chart view. A trend-following strategy using this technique relies "
                        "on simple visual cues: a strong uptrend is characterized by a series of long-bodied green candles"
                        " with little or no lower wicks 🟢, while a strong downtrend is indicated by a series of long red"
                        " candles with little or no upper wicks 🔴. Traders typically enter a position when a clear trend"
                        " forms and hold it until the candles start to show signs of indecision or reversal, such as "
                        "shrinking bodies or the appearance of opposing wicks.</p>",
                        unsafe_allow_html=True)
            strat_17(st.session_state.df_uploaded)

        with t18:
            st.markdown("<h3 style='text-align:left'>💯Chart Patterns (e.g., Head and Shoulders, Triangles)</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>Chart Patterns are distinct formations on a price chart that are used by"
                        " technical analysts to forecast future price movements. These patterns, which visually represent "
                        "the tug-of-war between buyers and sellers, fall into two main categories: reversal and continuation"
                        ". Reversal patterns, like the well-known Head and Shoulders 📉, signal that the current trend is "
                        "likely changing direction. Continuation patterns, such as Triangles and Flags 📈, suggest a "
                        "temporary pause in the market before the original trend resumes. By recognizing these recurring"
                        " shapes, traders can anticipate potential breakouts or breakdowns to identify strategic entry "
                        "and exit points.</p>",
                        unsafe_allow_html=True)
            strat_18(st.session_state.df_uploaded)

        with t19:
            st.markdown("<h3 style='text-align:left'>💯Keltner Channel Strategy</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>A Keltner Channel strategy is typically a trend-following system "
                        "that uses a volatility-based indicator composed of a central Exponential Moving Average "
                        "(EMA) and two outer channels derived from the Average True Range (ATR). The most common "
                        "strategy involves trading breakouts, where a decisive price close outside of the channel"
                        " signals the potential start of a new trend. A close above the upper channel is interpreted"
                        " as a buy signal 📈, suggesting strong upward momentum, while a close below the lower "
                        "channel is viewed as a sell signal 📉, indicating the beginning of a downtrend. Once a"
                        " trade is initiated, the central EMA line often serves as a dynamic support or resistance"
                        " level and can be used as a reference for a trailing stop-loss.</p>",
                        unsafe_allow_html=True)
            strat_19(st.session_state.df_uploaded)

        with t20:
            st.markdown("<h3 style='text-align:left'>💯Awesome Oscillator Strategy</h3>", unsafe_allow_html=True)
            st.markdown("<p style:font-size:1.5rem>The Awesome Oscillator (AO) is a momentum indicator that measures"
                        " the market's driving force, displayed as a histogram that fluctuates above and below a central"
                        " zero line. The most fundamental strategy involves the Zero Line Crossover: a bullish signal 📈 "
                        "is generated when the histogram crosses from below to above the zero line, suggesting that"
                        " short-term momentum is overpowering long-term momentum. Conversely, a bearish signal 📉 occurs "
                        "when the histogram crosses from above to below the zero line, indicating that selling pressure "
                        "is gaining control. Beyond this simple crossover, traders also look for more nuanced signals such"
                        " as 'Twin Peaks' for divergence and 'Saucer' patterns for trend continuation entries.</p>",
                        unsafe_allow_html=True)
            strat_20(st.session_state.df_uploaded)


