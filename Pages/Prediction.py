import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
from ta.volatility import AverageTrueRange
from keras.losses import MeanSquaredError
import plotly.graph_objects as go
from keras.models import load_model
import joblib
from keras.metrics import MeanSquaredError
import xgboost
import numpy as np
import matplotlib.pyplot as plt

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

#================================================================================
# INITIALIZERS
#===============================================================================

def initializer():
    if "df" not in st.session_state: st.session_state.df=None
    if "zone_scale" not in st.session_state: st.session_state.zone_scale=None
    if "zone_model" not in st.session_state: st.session_state.zone_model=None
    if "lstm_model" not in st.session_state: st.session_state.lstm_model=None
    if "gru_model" not in st.session_state: st.session_state.gru_model=None
    if "lstm_f_scale" not in st.session_state: st.session_state.lstm_f_scale=None
    if "lstm_t_scale" not in st.session_state: st.session_state.lstm_t_scale=None
    if "gru_f_scale" not in st.session_state: st.session_state.gru_f_scale=None
    if "gru_t_scale" not in st.session_state: st.session_state.gru_t_scale=None
    if "zone_data" not in st.session_state: st.session_state.zone_data=None
    if "lstm_data" not in st.session_state: st.session_state.lstm_data=None
    if "gru_data" not in st.session_state: st.session_state.gru_data=None

initializer()

#================================================================================
# DAILOG BOX
#===============================================================================

@st.dialog("Overview")
def show_feature():
    st.header("😎 Key Models")
    feature={"Boosting":"Boosting techniques is used including XGBoost LightBoost",
             "RNN":"RNN techniques is used including LSTM(Long-Short term memory and GRU(Gated Recurrent Neural Network))",
             "Stacking":"Stacking Models",
             "ANN":"Artificial Neural Network"}
    for name,val in feature.items():
        st.markdown(f"<p><b>{name}</b> : {val}</p>",unsafe_allow_html=True)


#================================================================================
# DATA LOADING
#===============================================================================

def get_gbpusd_4h(limit=500, period="120d"):
    try:
        df_1h = yf.download("6B=F", interval="1h", period=period, progress=False)
        df_1h.columns = ["close", "high", "low", "open", "volume"]
        df_1h.index.name = "date"
        if df_1h.empty:
            st.error("⚠️ No data received from Yahoo Finance. Server might be busy.")
            return pd.DataFrame()

        df_4h = df_1h.resample("4H").agg({
            "open": "first",  # first value in 4h
            "high": "max",  # highest value in 4h
            "low": "min",  # lowest value in 4h
            "close": "last",  # last value in 4h
            "volume": "sum"  # total volume in 4h
        })
        df_4h.dropna(inplace=True)
        return df_4h

    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return pd.DataFrame()

#================================================================================
# FEATURE MAKING
#===============================================================================

def feature_engineering(data):
    data["ema_20"] = EMAIndicator(data["close"], window=20).ema_indicator()
    data["ema_50"] = EMAIndicator(data["close"], window=50).ema_indicator()
    data["ema_100"] = EMAIndicator(data["close"], window=100).ema_indicator()
    for ema, window in zip([20,20,50,50,50,100], [10,15,20,30,40,80]):
        mean = data[f"ema_{ema}"].rolling(window).mean()
        std = data[f"ema_{ema}"].rolling(window).std()
        data[f"zscore_{ema}_{window}"] = (data[f"ema_{ema}"] - mean) / std
    for i in (5,10, 30, 50):
        data[f"price_score_{i}"] = (data["close"] - data["close"].rolling(i).mean()) / data[
            "close"].rolling(i).std()
        data[f"pct_change_{i}"] = data["close"].pct_change(i)
    atr = AverageTrueRange(high=data["high"], low=data["low"], close=data["close"])
    data["atr"] = atr.average_true_range()
    data["candle_range"] = data["high"] - data["low"]
    data["ema"] = EMAIndicator(data["close"], window=21).ema_indicator()
    data["rsi"] = RSIIndicator(data["close"], window=13).rsi()
    bb = BollingerBands(data["close"])
    data["uperband"] = bb.bollinger_hband()
    data["lowerband"] = bb.bollinger_lband()
    data["bb_avg"] = bb.bollinger_mavg()
    data["mean_price"] = data["close"].rolling(13).mean()
    candle_mean = (data["high"] - data["low"]).mean()
    data["candle_strentgh"] = (data["high"] - data["low"]) / candle_mean

    return data

#================================================================================
# TREND PREDICTOR
#===============================================================================

def trend_predictor(data):
    features = st.session_state.zone_scale.transform(data)
    prediction = st.session_state.zone_model.predict(features)
    pred = (
        pd.Series(prediction)
        .map({0: -1, 1: 11, 2: 11, 3: 0, 4: 1, 5: 0})
        .dropna()
        .astype(int)
    )

    if pred.empty:
        st.warning("No valid predictions after mapping.")
        return
    zones = np.array_split(pred, 4)
    label_map = {-1: "Downtrend 📉", 0: "Sideways ➡️", 1: "Uptrend 📈"}
    zone_results = []
    non_empty_zones = 0

    cols = st.columns(4)
    for i, (z, c) in enumerate(zip(zones, cols), start=1):
        counts = z.value_counts()
        total = int(counts.sum())
        if total == 0:
            sig = 0
            conf = 0.0
            votes = {}
        else:
            sig = counts.idxmax()        # -1 / 0 / 1
            conf = counts.max() / total
            votes = counts.to_dict()
            non_empty_zones += 1

        lbl = label_map[sig]
        zone_results.append({"zone": i, "signal": sig, "confidence": conf, "label": lbl, "votes": votes})
        c.metric(f"Zone {i}", lbl, f"{conf*100:.1f}% agreement", help=f"Votes: {votes}")

    if non_empty_zones == 0:
        st.warning("All zones were empty.")
        return

    zone_signals = [r["signal"] for r in zone_results]
    zone_counts = pd.Series(zone_signals).value_counts()
    top_freq = zone_counts.max()
    top_signals = zone_counts[zone_counts == top_freq].index.tolist()

    if len(top_signals) == 1:
        final_signal = top_signals[0]
    else:
        best_sig = None
        best_conf = -1.0
        best_zone_idx = -1
        for idx, r in enumerate(zone_results):  # idx increases with recency
            if r["signal"] in top_signals:
                if (r["confidence"] > best_conf) or (r["confidence"] == best_conf and idx > best_zone_idx):
                    best_sig = r["signal"]
                    best_conf = r["confidence"]
                    best_zone_idx = idx
        final_signal = best_sig

    final_label = label_map[final_signal]
    final_support = zone_counts[final_signal] / max(non_empty_zones, 1) * 100.0

    st.metric("🔥🔥Final Trend🔥🔥", final_label, f"{final_support:.0f}% of zones")


#================================================================================
# PRICE PREDICTOR
#===============================================================================

@st._cache_data
def price_predictor(lstm_data,gru_data):
    L=[x for x in range(1,len(lstm_data.columns))]
    G=[x for x in range(1,len(gru_data.columns))]

    features1=lstm_data.values
    features2=gru_data.values

    features1[:,0]=st.session_state.lstm_t_scale.transform(features1[:,0].reshape(-1,1)).ravel()
    features1[:,L]=st.session_state.lstm_f_scale.transform(features1[:,L])
    features2[:,0]=st.session_state.gru_t_scale.transform(features2[:,0].reshape(-1,1)).ravel()
    features2[:,G]=st.session_state.gru_f_scale.transform(features2[:,G])


    features1 = features1.reshape(1,100,len(L)+1)
    features2 = features2.reshape(1,100,len(G)+1)

    prediction_1 = st.session_state.lstm_model.predict(features1, verbose=0)
    prediction_2 = st.session_state.gru_model.predict(features2, verbose=0)

    prediction_1 = st.session_state.lstm_t_scale.inverse_transform(np.asarray(prediction_1).reshape(-1, 1)).ravel()
    prediction_2 = st.session_state.gru_t_scale.inverse_transform(np.asarray(prediction_2).reshape(-1, 1)).ravel()

    return prediction_1, prediction_2

with st.container():
    st.markdown("<h1 style='text-align: center; font-size:4rem'>Predication Zone</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.2rem; max-width: 800px; margin: auto;'>In this part"
        " you can try and test multiple backtesting strategies with your own data or we have some samples data as well.</p>",
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # --- Top buttons ---
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c3:
        if st.button("🔍 Show Models"):
            show_feature()
    with c5:
        st.caption("Only for 4H GBP/USD")

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)


    st.markdown("<h2>📖 LSTM and GRU </h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Visualization Space")

    st.markdown("""
        **LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit)** are types of 
        Recurrent Neural Networks (RNNs) designed to learn **patterns across time**.  

        ---
        ### 🔑 Key Concept: Timesteps
        - A **timestep** is how many past data points the model looks back on.
        - Example: Timestep = 60 → Uses last 60 prices to predict the next.
        - Large timesteps = more context, slower training.
        - Small timesteps = faster, but less historical learning.

        ---
        ### 🌟 Benefits in Stock/Forex Market
        - Capture **sequential trends** (price depends on history).
        - Learn **long-term dependencies** (weeks/months).
        - Handle **noisy data** better than classical models.
        - Useful for:
          - Price prediction
          - Volatility forecasting
          - Trend detection
    """)
    st.info("👉 Below are predictions of LSTM on Forex Data")
    st.caption("Black line is real data | Orange is prediction on training data | Green is prediction on Testing data")
    st.image("Asset/output.png")


    st.markdown("<h2>📖 XGBoost with KMeans for Zone Prediction</h2>", unsafe_allow_html=True)

    st.markdown("""
        ### ⚡ XGBoost (Extreme Gradient Boosting)
        - A **boosting algorithm** that combines many weak learners (decision trees).
        - Great for **tabular and structured financial data**.

        ---
        ### 🔑 Why use KMeans + XGBoost?
        1. **KMeans (Clustering)**  
           - Groups prices into **zones** (support/resistance).  
           - These zones become **labels** for supervised learning.
        2. **XGBoost (Prediction)**  
           - Learns to predict which **zone** new price data belongs to.  
           - Helps identify **trends & market regimes**.

        ---
        ### 🌟 Benefits
        - Zone Prediction → find support & resistance
        - Pattern Learning → from clusters to unseen data
        - Speed & Accuracy → scalable on large datasets
        - Hybrid Power → clustering + prediction
    """)
    st.info("👉 Below are zones made by KMeans and predictions fitted by XGBoost. [0 = Downtrend] [4 = Uptrend]")
    st.caption("Overall Theory is if in range Number of point having Downtrend is more then its down else up or sideways")
    st.image("Asset/zone.png")


    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💻 Model Code Space")

    code = '''
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange
import tensorflow as tf
from tensorflow import keras

# ---------------- Data Loader ----------------
class DataLoader:
    def __init__(self, path):
        self.path = path

    @staticmethod
    def succesfull():
        print("✅ Data Loaded successfully")

    def load_data(self):
        data = pd.read_csv(self.path, sep="\\t", names=["date","open","high","low","close","volume"])
        data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d %H:%M")
        data.set_index("date", inplace=True)
        self.succesfull()
        print("👾 Shape of Data : ", data.shape)
        return data

# ---------------- Feature Engineering ----------------
class Features:
    @staticmethod
    def succesfull():
        print("✅ Features Created successfully")

    def make_features(self, data):
        data["ema_20"] = EMAIndicator(data["close"], window=20).ema_indicator()
        data["ema_50"] = EMAIndicator(data["close"], window=50).ema_indicator()

        for ema, window in zip([20, 50], [10, 20]):
            mean = data[f"ema_{ema}"].rolling(window).mean()
            std = data[f"ema_{ema}"].rolling(window).std()
            data[f"zscore_{ema}_{window}"] = (data[f"ema_{ema}"] - mean) / std

        for i in (5, 10):
            data[f"price_score_{i}"] = (data["close"] - data["close"].rolling(i).mean()) / data["close"].rolling(i).std()
            data[f"pct_change_{i}"] = data["close"].pct_change(i)

        atr = AverageTrueRange(high=data["high"], low=data["low"], close=data["close"])
        data["atr"] = atr.average_true_range()
        data["candle_range"] = data["high"] - data["low"]

        data.dropna(inplace=True)
        data.drop(["open","high","low"], axis=1, inplace=True)
        print("🏹 Shape Now is : ", data.shape)
        return data

# ---------------- Scaling + Timesteps ----------------
class Scaling_Timesteps:
    @staticmethod
    def succesfull1():
        print("✅ Data Scaled successfully")
    @staticmethod
    def succesfull2():
        print("✅ Timestep Added successfully")

    def start_scaling(self, data, timesteps, future):
        print("-"*50)
        print("🔎 All features : ", data.columns)
        print("-"*50)

        features = data.values
        column = data.columns
        rest_columns = []

        for idx in range(len(column)):
            if column[idx] == "close":
                forecast = idx
            else:
                rest_columns.append(idx)

        scaler1 = MinMaxScaler(feature_range=(0,1))
        scaler2 = MinMaxScaler(feature_range=(0,1))

        split = int(features.shape[0]*0.8)

        features[:split, forecast] = scaler1.fit_transform(features[:split, forecast].reshape(-1,1)).ravel()
        features[split:, forecast] = scaler1.transform(features[split:, forecast].reshape(-1,1)).ravel()

        features[:split, rest_columns] = scaler2.fit_transform(features[:split, rest_columns])
        features[split:, rest_columns] = scaler2.transform(features[split:, rest_columns])

        self.succesfull1()
        joblib.dump(scaler1, "lstm_target.pkl")
        joblib.dump(scaler2, "lstm_features.pkl")
        print("🤹🏻 Succesfully stored MinMaxScaler")

        X_new, y_new = [], []
        for i in range(len(features)-timesteps-future+1):
            X_new.append(features[i:i+timesteps])
            y_new.append(features[i+timesteps+future-1, forecast])

        self.succesfull2()
        X_new = np.array(X_new)
        y_new = np.array(y_new)

        print("❄️ Shape of Train Data : ", X_new.shape)
        print("❄️ Shape of Train Target : ", y_new.shape)
        return X_new, y_new

# ---------------- Model Training ----------------
class Model_Training:
    def start_training(self, X, y):
        split = int(X.shape[0]*0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        shape = X.shape[1:]
        model = keras.Sequential([
            keras.layers.LSTM(128, activation="tanh", return_sequences=True, input_shape=shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(128, activation="tanh", return_sequences=True),
            keras.layers.BatchNormalization(),
            keras.layers.LSTM(64, activation="tanh", return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(32, activation="tanh"),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1, activation="linear")
        ])

        model.compile(loss="mse", optimizer="adam", metrics=["mae"])
        callbacks = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=10, verbose=1, mode="min", restore_best_weights=True
        )

        print(model.summary())
        model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=callbacks)

        model.save("lstm_model.h5")
        print("✅ Model Successfully Trained and Saved as 'lstm_model.h5'")
'''

    code2='''class DataLoader:
    def __init__(self,path):
        self.path=path
        
    @staticmethod
    def succesfull():
        print("✅Data Loaded succesfully")
        
    def load_data(self):
        data=pd.read_csv(self.path,sep="\t",names=["date","open","high","low","close","volume"])
        data["date"]=pd.to_datetime(data["date"],format="%Y-%m-%d %H:%M")
        data.set_index("date",inplace=True)
        self.succesfull()
        print("👾Shape of Data : ",data.shape)
        return data

class Features:
    def __init__(self,data):
        self.data=data.copy()
        
    @staticmethod
    def succesfull():
        print("✅Features Created succesfully")
        
    def make_features(self):
        self.data["ema_20"]=EMAIndicator(self.data["close"],window=20).ema_indicator()
        self.data["ema_50"]=EMAIndicator(self.data["close"],window=50).ema_indicator()
        self.data["ema_100"]=EMAIndicator(self.data["close"],window=100).ema_indicator()
        for ema,window in zip([20,50,50,100],[15,30,40,80]):
            mean=self.data[f"ema_{ema}"].rolling(window).mean()
            std=self.data[f"ema_{ema}"].rolling(window).std()
            self.data[f"zscore_{ema}_{window}"]=(self.data[f"ema_{ema}"]-mean)/std
        for i in (10,30,50):
            self.data[f"price_score_{i}"]=(self.data["close"]-self.data["close"].rolling(i).mean())/self.data["close"].rolling(i).std()
            self.data[f"pct_change_{i}"]=self.data["close"].pct_change(i)        

        self.data.drop(["ema_20","ema_50","ema_100"],axis=1,inplace=True)
        self.data.dropna(inplace=True)
        print("🏹 Shape Now is : ",self.data.shape)
        return self.data

class Target:
    def __init__(self, data,name):
        self.zone = data.drop(name,axis=1).copy()
        self.data=data.copy()
        
    def make_targets(self):
        scaler=MinMaxScaler(feature_range=(0,1))
        self.zone=scaler.fit_transform(self.zone)

        pca=PCA(n_components=7)
        pca_data=pca.fit_transform(self.zone)
        wcss=[]                              #Finding optimal number of cluster for kmean using Kne method
        for i in range(2,11):
            kmean=KMeans(n_clusters=i,init="k-means++")
            kmean.fit(pca_data)
            wcss.append(kmean.inertia_)
        plt.plot(range(2,11),wcss,label="Knee_Method")
        plt.xticks(range(2,11))
        plt.show()
        
        kmean=KMeans(n_clusters=6,init="k-means++")     
        kmean.fit(pca_data)

        print("-"*50)
        print("Target_Has_Been_Created")
        self.data["zone"]=kmean.labels_
        print(self.data["zone"].value_counts())
        return self.data


class Visualization:
    def __init__(self):
        print("-"*50)
        print("💗 Graph below")
    def plot_graph(self,node):
        data=node.reset_index()
        plt.figure(figsize=(20,6))
        sns.scatterplot(data=data.iloc[20000:25000],x="date",y="close",hue="zone",palette="Set2")
        plt.show()

class Model:
    def __init__(self):
        print("-"*50)
        print("✅ Model Training has been initiated")

    
    def eval_model(self,model,features,X_train,X_test,y_train,y_test):
        # For cross validation
        pred=cross_val_predict(model,X_train,y_train,cv=5)
        # To get score using predict proba
        acc=accuracy_score(y_train,pred)  # return accuracy
        print("-"*50)
        print("✅ Accuracy of model is : ",acc)
        con_mtrx=confusion_matrix(y_train,pred)
        if hasattr(model,"feature_importances_"):  # Check if model have this function or not [Knn not have it]
            Imp=pd.DataFrame({"Name":features,"Importance":model.feature_importances_}).sort_values("Importance",ascending=False)
    
        fig,ax=plt.subplots(1,2,figsize=(14,8))
        sns.heatmap(con_mtrx,annot=True,linewidth=0.2,cmap="Blues",ax=ax[0],fmt="d")
        ax[0].set_title("Confusion_Matrix")
        if hasattr(model,"feature_importances_"):
    
            sns.barplot(data=Imp.head(10),y="Name",x="Importance",palette="viridis")
            ax[1].set_title("Top_10_features")
        
        plt.tight_layout()
        plt.show()
    
        cr=classification_report(y_train,pred)
        print(cr)
    
    def start_training(self,data,name):
        features=data.drop(name,axis=1).columns
        X=data.drop(name,axis=1).values
        print(X.shape)
        print("_"*50)
        print(features)
        print("_"*50)
        y=data["zone"].values
        split=int(X.shape[0]*0.8)
        X_train=X[:split]
        y_train=y[:split]
        X_test=X[split:]
        y_test=y[split:]

        scale_123=MinMaxScaler(feature_range=(0,1))
        X_train=scale_123.fit_transform(X_train)
        X_test=scale_123.transform(X_test)

        print("✅Scaler Has Been Dump")
        joblib.dump(scale_123,"zone_scaler_mm.pkl")
        
        model=XGBClassifier()
        model.fit(X_train,y_train)
        self.eval_model(model,features,X_train,X_test,y_train,y_test)
        
        joblib.dump(model,"zone_detector.pkl")
        print("🤹🏻MODEL SAVED🤹🏻")'''

    with st.expander("🏹 You can see the code from below to trai your own model"):
        st.subheader("For LSTM and GRU")
        st.code(code, language="python")
        st.subheader("For XGboost and KMean")
        st.code(code2, language="python")

    st.markdown("<div class='info-container'></div>", unsafe_allow_html=True)

    st.markdown("<h2>👾 Predict Future Price</h2>", unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.info("This Will only predict GBP/USD in 4h timeframe")
    if st.button("🔥Start Predict"):
        # Load the data
        st.session_state.df = get_gbpusd_4h(limit=500, period="120d")

        # Show data only if available
        if st.session_state.df is not None:
            with st.expander("⏰ View Data From Below"):
                st.dataframe(st.session_state.df.tail(10))


    if st.session_state.df is not None:
        st.session_state.df=feature_engineering(st.session_state.df)

        zone_name=['zscore_20_15', 'zscore_50_30', 'zscore_50_40', 'zscore_100_80',
       'price_score_10', 'pct_change_10', 'price_score_30', 'pct_change_30',
       'price_score_50', 'pct_change_50']
        lstm_name=['close', 'volume', 'ema_20', 'ema_50', 'zscore_20_10', 'zscore_50_20',
       'price_score_5', 'pct_change_5', 'price_score_10', 'pct_change_10',
       'atr', 'candle_range']
        gru_name=['close', 'volume', 'ema', 'rsi', 'uperband', 'lowerband', 'bb_avg',
       'mean_price', 'candle_strentgh']

        st.session_state.zone_data=st.session_state.df[zone_name]
        st.session_state.lstm_data=st.session_state.df[lstm_name]
        st.session_state.gru_data=st.session_state.df[gru_name]


        st.session_state.zone_scale=joblib.load("Models/zone_scaler_mm.pkl")
        st.session_state.zone_model=joblib.load("Models/zone_detector.pkl")
        st.session_state.lstm_model=load_model("Models/lstm_model.h5", compile=False)
        st.session_state.lstm_model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae"])
        st.session_state.gru_model=load_model("Models/gru_model.h5", compile=False)
        st.session_state.gru_model.compile(optimizer="adam", loss=MeanSquaredError(), metrics=["mae"])
        st.session_state.lstm_f_scale=joblib.load("Models/lstm_features.pkl")
        st.session_state.lstm_t_scale=joblib.load("Models/lstm_target.pkl")
        st.session_state.gru_f_scale=joblib.load("Models/gru_features.pkl")
        st.session_state.gru_t_scale=joblib.load("Models/gru_target.pkl")

        st.write("👾 Below is the recent chart of GBPUSD")
        st.caption("This is downloaded from yfinance some delay may present")
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(st.session_state.df["close"], color="green", linewidth=2, label="4Hour")
        ax.legend()
        ax.set_title("GBP/USD Chart")
        st.pyplot(fig)


        #trend predictor
        trend_predictor(st.session_state.zone_data)

        #GRU and LSTM predictor
        d1,d2=price_predictor(st.session_state.lstm_data.tail(100),st.session_state.gru_data.tail(100))

        lstm_df = st.session_state.lstm_data.tail(100)
        gru_df = st.session_state.gru_data.tail(100)

        # Pick a price series (prefer common names, else last numeric col) — inline, no functions

        price_series= lstm_df["close"].astype(float)
        last_price = float(price_series.iloc[-1])

        lstm_pred = float(d1)
        gru_pred = float(d2)+0.012 #Bias Added
        avg_pred = (lstm_pred + gru_pred) / 2.0

        # Deltas vs last
        lstm_delta_pct = (lstm_pred - last_price) / last_price * 100.0
        gru_delta_pct = (gru_pred - last_price) / last_price * 100.0
        avg_delta_pct = (avg_pred - last_price) / last_price * 100.0

        agree = np.sign(lstm_delta_pct) == np.sign(gru_delta_pct)

        # --- Header ---
        st.markdown("""
        <h2 style="text-align:center;margin:0;">📈 Next-Candle Outlook</h2>
        <p style="text-align:center;margin-top:4px;opacity:.75;">LSTM vs GRU with recent market context</p>
        """, unsafe_allow_html=True)

        # --- Highlight badges (agreement + move range) ---
        colA, colB = st.columns([1, 1])
        with colA:
            if agree:
                st.success("✅ Models agree on direction")
            else:
                st.warning("⚠️ Models disagree — use caution")
        with colB:
            band_low, band_high = min(lstm_pred, gru_pred), max(lstm_pred, gru_pred)
            st.info(f"Uncertainty band: {band_low:.5f} → {band_high:.5f}")

        # --- Metrics row ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Last Price", f"{last_price:.5f}")
        with c2:
            st.metric("LSTM → Next", f"{lstm_pred:.5f}", f"{lstm_delta_pct:+.2f}%")
        with c3:
            st.metric("GRU → Next", f"{gru_pred:.5f}", f"{gru_delta_pct:+.2f}%")
        with c4:
            st.metric("Average (LSTM+GRU)", f"{avg_pred:.5f}", f"{avg_delta_pct:+.2f}%")

        st.divider()

        hist = price_series
        if isinstance(hist.index, pd.DatetimeIndex):
            if len(hist.index) > 1:
                step = hist.index[-1] - hist.index[-2]
            else:
                step = pd.Timedelta(hours=1)
            next_idx = hist.index[-1] + step
        else:
            next_idx = hist.index[-1] + 1 if hasattr(hist.index, "__add__") else len(hist)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="History"))

        fig.add_trace(go.Scatter(x=[next_idx], y=[lstm_pred], mode="markers+text",
                                 name="LSTM", text=["LSTM"], textposition="top center"))
        fig.add_trace(go.Scatter(x=[next_idx], y=[gru_pred], mode="markers+text",
                                 name="GRU", text=["GRU"], textposition="bottom center"))
        fig.add_trace(go.Scatter(x=[next_idx], y=[avg_pred], mode="markers+text",
                                 name="Average", text=["AVG"], textposition="middle left"))

        fig.add_shape(type="rect",
                      x0=hist.index[-1], x1=next_idx,
                      y0=min(lstm_pred, gru_pred), y1=max(lstm_pred, gru_pred),
                      fillcolor="LightSkyBlue", opacity=0.22, line_width=0)

        fig.update_layout(
            title="Recent Price with Next-Candle Predictions",
            xaxis_title="Time" if isinstance(hist.index, pd.DatetimeIndex) else "Index",
            yaxis_title="Price",
            hovermode="x unified",
            height=460,
            margin=dict(l=30, r=20, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Compact summary table ---
        summary_df = pd.DataFrame({
            "Model": ["LSTM", "GRU", "Average"],
            "Prediction": [lstm_pred, gru_pred, avg_pred],
            "Δ vs Last": [f"{lstm_pred - last_price:+.6f}", f"{gru_pred - last_price:+.6f}",
                          f"{avg_pred - last_price:+.6f}"],
            "Δ %": [f"{lstm_delta_pct:+.2f}%", f"{gru_delta_pct:+.2f}%", f"{avg_delta_pct:+.2f}%"]
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

        # --- Optional: Candlesticks if OHLC exists (kept inline, no functions) ---
        ohlc_cols = {c.lower(): c for c in lstm_df.columns}
        if all(k in ohlc_cols for k in ("open", "high", "low", "close")):
            o, h, l, c = ohlc_cols["open"], ohlc_cols["high"], ohlc_cols["low"], ohlc_cols["close"]
            ohlc = lstm_df[[o, h, l, c]].copy()
            if not isinstance(ohlc.index, pd.DatetimeIndex):
                ohlc.index = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(ohlc), freq="H")
            fig2 = go.Figure(data=[go.Candlestick(
                x=ohlc.index, open=ohlc[o], high=ohlc[h], low=ohlc[l], close=ohlc[c], name="OHLC"
            )])
            fig2.add_hline(y=lstm_pred, line_dash="dot", annotation_text="LSTM next", opacity=0.5)
            fig2.add_hline(y=gru_pred, line_dash="dot", annotation_text="GRU next", opacity=0.5)
            fig2.add_hline(y=avg_pred, line_dash="dash", annotation_text="AVG next", opacity=0.5)
            fig2.update_layout(title="Candlestick (last 100) + Predicted Levels",
                               height=460, margin=dict(l=30, r=20, t=50, b=40))
            with st.expander("Candlestick view"):
                st.plotly_chart(fig2, use_container_width=True)

        # --- Export for logging/backtest ---
        export_df = pd.DataFrame({
            "last_price": [last_price],
            "lstm_pred": [lstm_pred],
            "gru_pred": [gru_pred],
            "avg_pred": [avg_pred],
            "lstm_delta_pct": [lstm_delta_pct],
            "gru_delta_pct": [gru_delta_pct],
            "avg_delta_pct": [avg_delta_pct],
            "agree": [agree]
        })
        st.download_button("Download snapshot (CSV)", export_df.to_csv(index=False), "next_candle_snapshot.csv")






