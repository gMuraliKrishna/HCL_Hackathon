import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ------------------------------
# Configuration
# ------------------------------
RF_MODEL_PATH = "models/rf_model.pkl"
LSTM_MODEL_PATH = "models/lstm_model.h5"
DATA_FILE = "retail_store_inventory.csv"

# ------------------------------
# Load Models
# ------------------------------
@st.cache_resource
def load_rf_model():
    return joblib.load(RF_MODEL_PATH)

@st.cache_resource
def load_lstm_model():
    return load_model(LSTM_MODEL_PATH)

rf_model = load_rf_model()
lstm_model = load_lstm_model()

# ------------------------------
# App Layout
# ------------------------------
st.title("Retail Store Demand Forecasting")
st.write("This app forecasts units sold using Random Forest and LSTM models.")

# ------------------------------
# Load dataset
# ------------------------------
st.sidebar.header("Upload Data (Optional)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(DATA_FILE)

st.write("### Sample Data")
st.dataframe(df.head())

# ------------------------------
# Select store and product
# ------------------------------
stores = df["Store ID"].unique()
products = df["Product ID"].unique()

selected_store = st.sidebar.selectbox("Select Store", stores)
selected_product = st.sidebar.selectbox("Select Product", products)

# Filter data
df_filtered = df[(df["Store ID"] == selected_store) & (df["Product ID"] == selected_product)].sort_values("Date")

# ------------------------------
# Preprocess features
# ------------------------------
feature_cols = df_filtered.drop(columns=["Units Sold", "Date"]).columns
X_input = df_filtered[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)

# ------------------------------
# Random Forest Prediction
# ------------------------------
rf_pred = rf_model.predict(X_input)
st.write("### Random Forest Predictions")
st.line_chart(rf_pred)

# ------------------------------
# LSTM Prediction
# ------------------------------
def create_lstm_sequences(X, lookback=12):
    X_seq = []
    for i in range(len(X) - lookback):
        X_seq.append(X.iloc[i:i+lookback].values)
    return np.array(X_seq)

lookback = 12
if len(X_input) > lookback:
    X_lstm_seq = create_lstm_sequences(X_input, lookback)
    lstm_pred = lstm_model.predict(X_lstm_seq).flatten()
    st.write("### LSTM Predictions")
    st.line_chart(lstm_pred)
else:
    st.warning(f"Not enough data for LSTM (requires > {lookback} rows)")

# ------------------------------
# Actual vs Predicted
# ------------------------------
st.write("### Actual Units Sold")
st.line_chart(df_filtered["Units Sold"].values)
