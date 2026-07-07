from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Daily Temperature Forecasting",
    page_icon="🌡️",
    layout="wide"
)

# Title
st.title("🌡️ Daily Minimum Temperature Forecasting")
st.write(
    "Predict the next day's minimum temperature using a trained LSTM model."
)

# Load Model and Scaler
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "temperature_lstm_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"

@st.cache_resource
def load_lstm_model():
    model = load_model(MODEL_PATH, compile=False)
    return model

@st.cache_resource
def load_scaler():
    scaler = joblib.load(SCALER_PATH)
    return scaler

try:
    model = load_lstm_model()
    scaler = load_scaler()

    st.success("Model and Scaler Loaded Successfully!")

except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Sidebar
st.sidebar.header("Input Temperature Values")

st.sidebar.write(
    "Enter the previous 30 days minimum temperatures."
)

# Default sample values
default_values = [10.0] * 30

temperatures = []

for i in range(30):
    temp = st.sidebar.number_input(
        f"Day {i+1}",
        value=float(default_values[i]),
        step=0.1
    )
    temperatures.append(temp)

# Prediction
if st.button("Predict Next Day Temperature"):

    input_data = np.array(temperatures).reshape(-1, 1)

    scaled_input = scaler.transform(input_data)

    X = scaled_input.reshape(1, 30, 1)

    prediction = model.predict(X)

    predicted_temp = scaler.inverse_transform(prediction)

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Next Day Minimum Temperature: "
        f"{predicted_temp[0][0]:.2f} °C"
    )

    # Plot
    chart_data = temperatures.copy()
    chart_data.append(predicted_temp[0][0])

    labels = [f"D{i+1}" for i in range(30)]
    labels.append("Prediction")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(labels[:-1], temperatures, marker='o')
    ax.plot(labels[-2:], chart_data[-2:], marker='o')
    ax.set_title("Temperature Forecast")
    ax.set_xlabel("Days")
    ax.set_ylabel("Temperature (°C)")
    plt.xticks(rotation=45)

    st.pyplot(fig)

# Dataset Upload Option
st.markdown("---")
st.subheader("Batch Prediction from CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV containing temperature values",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.write("Preview:")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):

            values = df.iloc[:, 0].values

            if len(values) < 30:
                st.error(
                    "CSV must contain at least 30 temperature values."
                )

            else:
                last_30 = values[-30:]

                input_data = np.array(last_30).reshape(-1, 1)

                scaled_input = scaler.transform(input_data)

                X = scaled_input.reshape(1, 30, 1)

                prediction = model.predict(X)

                predicted_temp = scaler.inverse_transform(prediction)

                st.success(
                    f"Predicted Temperature: "
                    f"{predicted_temp[0][0]:.2f} °C"
                )

    except Exception as e:
        st.error(f"Error processing file: {e}")

