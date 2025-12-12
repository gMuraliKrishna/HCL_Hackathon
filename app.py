import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import time

# ------------------------------
# Configuration
# ------------------------------
RF_MODEL_PATH = "models/rf_model.pkl"
LSTM_MODEL_PATH = "models/lstm_model.h5"
DATA_FILE = "retail_store_inventory.csv"
LOOKBACK = 12  # for LSTM sequences

# ------------------------------
# Helper Functions
# ------------------------------

def load_rf_model(path):
    try:
        start = time.time()
        model = joblib.load(path)
        st.success(f"Random Forest model loaded in {time.time() - start:.2f}s")
        return model
    except Exception as e:
        st.error(f"Failed to load Random Forest model: {e}")
        return None

def load_lstm_model(path):
    try:
        start = time.time()
        model = load_model(path)
        st.success(f"LSTM model loaded in {time.time() - start:.2f}s")
        return model
    except Exception as e:
        st.error(f"Failed to load LSTM model: {e}")
        return None

def create_lstm_sequences(X, lookback=12):
    X_seq = []
    for i in range(len(X) - lookback):
        X_seq.append(X.iloc[i:i+lookback].values)
    return np.array(X_seq)

# ------------------------------
# App Layout
# ------------------------------
st.set_page_config(page_title="Retail Demand Forecasting", layout="wide")
st.title("Retail Store Demand Forecasting")
st.write("Forecast units sold using Random Forest and LSTM models.")

# ------------------------------
# Load Models
# ------------------------------
with st.spinner("Loading models..."):
    rf_model = load_rf_model(RF_MODEL_PATH)
    lstm_model = load_lstm_model(LSTM_MODEL_PATH)

# ------------------------------
# Load Dataset
# ------------------------------
st.sidebar.header("Upload Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        st.error(f"Default CSV '{DATA_FILE}' not found!")
        st.stop()

st.write("### Sample Data")
st.dataframe(df.head())

# ------------------------------
# Select store and product
# ------------------------------
stores = df["Store ID"].unique()
products = df["Product ID"].unique()

selected_store = st.sidebar.selectbox("Select Store", stores)
selected_product = st.sidebar.selectbox("Select Product", products)

df_filtered = df[(df["Store ID"] == selected_store) & (df["Product ID"] == selected_product)].sort_values("Date")

if df_filtered.empty:
    st.warning("No data for selected store/product!")
    st.stop()

# ------------------------------
# Preprocess features
# ------------------------------
feature_cols = df_filtered.drop(columns=["Units Sold", "Date"]).columns
X_input = df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

# ------------------------------
# Random Forest Prediction
# ------------------------------
if rf_model:
    with st.spinner("Predicting with Random Forest..."):
        rf_pred = rf_model.predict(X_input)
    st.write("### Random Forest Predictions")
    st.line_chart(pd.DataFrame({"Date": df_filtered["Date"], "Prediction": rf_pred}).set_index("Date"))

# ------------------------------
# LSTM Prediction
# ------------------------------
if lstm_model:
    if len(X_input) > LOOKBACK:
        with st.spinner("Predicting with LSTM..."):
            X_lstm_seq = create_lstm_sequences(X_input, LOOKBACK)
            lstm_pred = lstm_model.predict(X_lstm_seq, verbose=0).flatten()
            dates_lstm = df_filtered["Date"].iloc[LOOKBACK:].reset_index(drop=True)
        st.write("### LSTM Predictions")
        st.line_chart(pd.DataFrame({"Date": dates_lstm, "Prediction": lstm_pred}).set_index("Date"))
    else:
        st.warning(f"Not enough data for LSTM (requires more than {LOOKBACK} rows).")

# ------------------------------
# Actual Units Sold
# ------------------------------
st.write("### Actual Units Sold")
st.line_chart(pd.DataFrame({"Date": df_filtered["Date"], "Units Sold": df_filtered["Units Sold"]}).set_index("Date"))
