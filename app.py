from pathlib import Path
import csv
import io

import streamlit as st
import numpy as np

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

# Load optional model assets if present
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "temperature_lstm_model.h5"
SCALER_PATH = BASE_DIR / "scaler.pkl"

@st.cache_resource
def load_optional_assets():
    model = None
    scaler = None

    try:
        from tensorflow.keras.models import load_model
        if MODEL_PATH.exists():
            model = load_model(MODEL_PATH, compile=False)
    except Exception:
        model = None

    try:
        import joblib
        if SCALER_PATH.exists():
            scaler = joblib.load(SCALER_PATH)
    except Exception:
        scaler = None

    return model, scaler

model, scaler = load_optional_assets()

if model is not None and scaler is not None:
    st.success("TensorFlow model and scaler loaded successfully.")
else:
    st.info("Using a lightweight fallback predictor because the TensorFlow model assets are not available in this environment.")


def predict_temperature(values):
    arr = np.asarray(values, dtype=float)

    if model is not None and scaler is not None and len(arr) >= 30:
        scaled_input = scaler.transform(arr.reshape(-1, 1))
        X = scaled_input.reshape(1, 30, 1)
        prediction = model.predict(X, verbose=0)
        return float(scaler.inverse_transform(prediction)[0][0])

    if len(arr) >= 2:
        x = np.arange(1, len(arr) + 1, dtype=float)
        slope, intercept = np.polyfit(x, arr, 1)
        return float(slope * (len(arr) + 1) + intercept)

    return float(arr[-1]) if len(arr) > 0 else 0.0

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
    predicted_temp = predict_temperature(temperatures)

    st.subheader("Prediction Result")
    st.success(
        f"Predicted Next Day Minimum Temperature: {predicted_temp:.2f} °C"
    )

    chart_values = temperatures + [predicted_temp]
    st.line_chart(chart_values)

# Dataset Upload Option
st.markdown("---")
st.subheader("Batch Prediction from CSV")

uploaded_file = st.file_uploader(
    "Upload a CSV containing temperature values",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        content = uploaded_file.getvalue().decode("utf-8-sig")
        reader = csv.reader(io.StringIO(content))
        rows = [row for row in reader if row]

        values = []
        for row in rows:
            try:
                values.append(float(row[0].strip()))
            except Exception:
                continue

        st.write("Preview:")
        st.write(values[:10])

        if st.button("Run Batch Prediction"):
            if len(values) < 30:
                st.error("CSV must contain at least 30 temperature values.")
            else:
                last_30 = values[-30:]
                predicted_temp = predict_temperature(last_30)
                st.success(f"Predicted Temperature: {predicted_temp:.2f} °C")

    except Exception as e:
        st.error(f"Error processing file: {e}")

