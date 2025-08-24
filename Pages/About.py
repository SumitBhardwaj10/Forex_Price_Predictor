import streamlit as st
import time

st.set_page_config(layout="wide", page_title="Forex Price Predictor")

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
      color: white;
    }}

</style>
""", unsafe_allow_html=True)


@st.dialog("Overview")
def show_features_form():
    st.header("✨ Key Features")
    features = {
        "Price Prediction": "Predict forex prices for major, minor, and exotic pairs.",
        "Backtesting": "Test multiple FX strategies (trend-following, mean reversion, breakout).",
        "Simplicity": "Clean UI with one-click workflows.",
        "Pre-Trained Models": "Ready-to-use models trained on FX time series.",
        "Model Tuning": "Fine-tuned for volatility, sessions, and regime shifts.",
        "Backtest Evaluation": "Evaluate with FX-specific metrics (pips, hit-rate, Sharpe, max drawdown)."
    }
    st.markdown("<br>", unsafe_allow_html=True)
    for feature, desc in features.items():
        st.write(f"**{feature}:** {desc}")


with st.container():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; font-size: 4rem; font-weight: bold;'>Forex Price Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; max-width: 800px; margin: auto;'>Leverage advanced algorithms and robust backtesting to navigate the FX market—across London, New York, and Tokyo sessions—with data-driven confidence.</p>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    with col3:
        if st.button("🦑 Explore Features"):
            show_features_form()

# --- ABOUT THE MODEL SECTION ---
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("<h2>🔍 About The Model</h2>", unsafe_allow_html=True)
    st.markdown("<b>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown(
            """
            <div class='info-card'>
            <h3>Core Architecture</h3>
            <p>We use an ensemble of time-series models tailored for high-frequency FX dynamics:</p>
            <p><b>➡️ Long Short-Term Memory (LSTM) Networks:</b> Capture temporal dependencies, regime shifts, and session-based patterns in currency pairs.</p>
            <p><b>➡️ Gated Recurrent Units (GRUs):</b> Efficient recurrent layers for sequential FX data, helping stabilize training under volatility.</p>
            <p><b>➡️ Feature Engineering:</b> Session flags (London/NY/Tokyo), rolling volatility (ATR), RSI, moving averages, and price action features in pips.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.image("Asset/lstm.png", use_container_width=True)

    st.markdown("<b>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image("Asset/stocks.png", use_container_width=True)
    with col2:
        st.markdown(
            """
            <div class='info-card'>
            <h3>Training and Evaluation</h3>
            <p>Models are continuously updated on multi-pair FX data to remain aligned with liquidity cycles and macro events:</p>
            <p><b>➡️ Fine-Tuning:</b> Regular updates with recent tick/1m/5m/1h candles to adapt to volatility regimes.</p>
            <p><b>➡️ Backtesting Engine:</b> Simulated trades on historical FX data with realistic spread, slippage, and session filters.</p>
            <p><b>➡️ Performance Metrics:</b> Pips gained, hit rate, Sharpe ratio, profit factor, and max drawdown.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

# --- FOREX CONCEPTS EXPLAINED SECTION ---
with st.container():
    st.markdown('<div class="info-container">', unsafe_allow_html=True)
    st.markdown("<h2>📚 Forex Concepts Explained</h2>", unsafe_allow_html=True)
    st.markdown("<b>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown(
            """
            <div class='info-card'>
            <h3>Currency Pairs, Lots, and Pips</h3>
            <p><b>Currency Pairs:</b> FX is quoted as pairs (e.g., EUR/USD, GBP/JPY). The first is the <i>base</i> currency; the second is the <i>quote</i> currency.
            There are many currency pair available in the market but the majority of trade take place in 4 major currency pairs.</p>
            <p><b>Pips:</b> The standard unit of price movement in FX (typically the 4th decimal place, 2nd for JPY pairs). Strategy returns are often measured in pips.
            Each pip movement will earn around 10 points. Means if you chose 1 lot and price move by 1 pip your price fluctuation will be 
            1*1*10 = 10 dollar.</p>
            <p><b>Lot Sizes:</b> Standard (100k units), Mini (10k), and Micro (1k). Position sizing in lots determines risk per pip.
            You can select lot in decimal places even.</p>
            <p><b>Leverage & Margin:</b> Brokers allow leverage (e.g., 1:30, 1:100). Use carefully—pips scale both profit and loss.
            It play a crucial role in maximize profits if handle carefully otherwise the account will be blown up.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("Asset/stock_option.png")

    st.markdown('<br>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="small")
    with c1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.image("Asset/bid_ask.png")
    with c2:
        st.markdown(
            """
            <div class='info-card'>
            <h3>Bid, Ask, and Spread</h3>
            <p>Every FX quote has two prices:</p>
            <p><b>Bid Price:</b> The price at which you can sell the base currency.
            Like price of currency is 100 but you want to sell it on 102 that is is the bid price.</p>
            <p><b>Ask Price:</b> The price at which you can buy the base currency.
            Like price of currency is 100 but you want to buy in on 98 that is ask price.</p>
            <p>The difference is the <b>spread</b>, typically measured in pips. Lower spreads (e.g., EUR/USD during London/NY overlap) imply better liquidity and lower trading costs.</p>
            <p><b>Slippage:</b> The difference between expected and executed price—more common during news or low-liquidity hours.
            So the difference is 102-98 =4 this is our spread</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<br>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class='info-card'>
        <h3>Why These Concepts Matter</h3>
        <p>Mastering FX basics enables robust strategy design and realistic performance expectations:</p>
        <ul>
            <li><b>Currency Pairs & Sessions:</b> Liquidity varies by session (Tokyo, London, New York). Your strategy should account for time-of-day effects.</li>
            <li><b>Pips & Position Sizing:</b> Measuring returns in pips and sizing positions by risk per trade keeps drawdowns in check.</li>
            <li><b>Backtesting:</b> Validates rules on historical FX data including spread and slippage—critical before going live.</li>
            <li><b>Bid-Ask Spread:</b> A core transaction cost in FX; tighter spreads can significantly improve strategy efficiency.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown('<div class="info-card"></div>', unsafe_allow_html=True)
