import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

# ------------------------------
# PAGE CONFIGURATION
# ------------------------------
st.set_page_config(page_title="📈 AAPL + TSLA Stock Trend Predictor", layout="centered")
st.title("📈 Stock Price Trend Predictor using LSTM (AAPL + TSLA)")

# ------------------------------
# USER INPUT
# ------------------------------
ticker = st.text_input("Enter Stock Symbol (AAPL or TSLA):", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

# ------------------------------
# FETCH STOCK DATA
# ------------------------------
st.info(f"Fetching {ticker} data from {start_date} to {end_date}...")
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("⚠️ No data found for this ticker or date range.")
    st.stop()

# Flatten MultiIndex if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data.reset_index(inplace=True)
data = data.dropna()

# ------------------------------
# FEATURE ENGINEERING
# ------------------------------
data['MA7'] = data['Close'].rolling(7).mean()
data['MA14'] = data['Close'].rolling(14).mean()
data['Pct_Change'] = data['Close'].pct_change()
data = data.dropna().reset_index(drop=True)

st.subheader("📊 Historical Closing Prices")
st.line_chart(data[['Date', 'Close']].set_index('Date'))

# ------------------------------
# LOAD MODEL & SCALERS
# ------------------------------
try:
    if os.path.exists("models/stock_lstm_aapl_tsla_final.keras"):
        model = load_model("models/stock_lstm_aapl_tsla_final.keras")
        feat_scaler = joblib.load("models/feat_scaler.save")
        target_scaler = joblib.load("models/target_scaler.save")
    else:
        model = load_model("stock_lstm_aapl_tsla_final.keras")
        feat_scaler = joblib.load("feat_scaler.save")
        target_scaler = joblib.load("target_scaler.save")
    st.success("✅ Model and scalers loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# ------------------------------
# DATA PREPARATION
# ------------------------------
FEATURES = ['Close', 'Open', 'High', 'Low', 'Volume', 'MA7', 'MA14', 'Pct_Change']
for c in FEATURES:
    if c not in data.columns:
        data[c] = 0

X_all = data[FEATURES].values.astype(float)
X_scaled = feat_scaler.transform(X_all)

# Sequence creation
TIME_STEP = 60
if len(X_scaled) < TIME_STEP + 1:
    st.warning("⚠️ Not enough data points. Please extend the date range.")
    st.stop()

X_seq = []
for i in range(TIME_STEP, len(X_scaled)):
    X_seq.append(X_scaled[i - TIME_STEP:i])
X_seq = np.array(X_seq)

# ------------------------------
# PREDICTION & EVALUATION
# ------------------------------
pred_s = model.predict(X_seq, verbose=0)
pred = target_scaler.inverse_transform(pred_s)
y_real = data['Close'].values[TIME_STEP:]

mse = mean_squared_error(y_real, pred)
rmse = np.sqrt(mse)

st.subheader("📉 Model Evaluation")
st.write(f"**MSE:** {mse:.4f}")
st.write(f"**RMSE:** {rmse:.4f}")

# ------------------------------
# NEXT-DAY FORECAST
# ------------------------------
X_input = X_scaled[-TIME_STEP:].reshape(1, TIME_STEP, len(FEATURES))
pred_next_s = model.predict(X_input)
pred_next = target_scaler.inverse_transform(pred_next_s)[0][0]
last_price = data['Close'].iloc[-1]

st.subheader("🔮 Next-Day Prediction")
st.write(f"**Last Real Closing Price:** ${last_price:.2f}")
st.write(f"**Predicted Next Closing Price:** ${pred_next:.2f}")

# Trend logic with slope check
recent = data['Close'].iloc[-7:]
slope = np.polyfit(range(len(recent)), recent, 1)[0] / last_price * 100
percent_change = (pred_next - last_price) / last_price * 100
trend_score = percent_change + slope
tol = 0.5

if trend_score > tol:
    st.success(f"📈 Predicted Trend: UPTREND (+{trend_score:.2f}%)")
elif trend_score < -tol:
    st.error(f"📉 Predicted Trend: DOWNTREND ({trend_score:.2f}%)")
else:
    st.info(f"➖ Predicted Trend: STABLE ({trend_score:.2f}%)")

# ------------------------------
# VISUALIZATION
# ------------------------------
st.subheader("📊 Actual vs Predicted (Test Section)")
N = min(300, len(y_real))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_real[-N:], label="Actual", color="blue")
ax.plot(pred[-N:], linestyle="--", label="Predicted", color="red")
ax.legend()
st.pyplot(fig)

st.subheader("📆 Last 60 Days + Next Prediction")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(data['Date'].iloc[-60:], data['Close'].iloc[-60:], label="Last 60 Days", color="blue")
ax2.scatter(data['Date'].iloc[-1] + pd.Timedelta(days=1), pred_next, color="red", s=80, label="Next Prediction")
ax2.legend()
plt.xticks(rotation=45)
st.pyplot(fig2)

st.caption("Developed by Team Market Forecasters — LSTM Stock Trend Predictor (AAPL + TSLA)")



