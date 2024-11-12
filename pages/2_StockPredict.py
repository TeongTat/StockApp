import streamlit as st

st.title("Stock Prediction")

st.write("Set up your prediction preferences below.")

# Sidebar options for prediction period and model
prediction_period = st.selectbox("Prediction Period", ["1 Month", "3 Months", "6 Months"])
prediction_model = st.selectbox("Prediction Model", ["Linear Regression", "LSTM", "ARIMA"])

st.write("Prediction settings selected:")
st.write(f"Prediction Period: {prediction_period}")
st.write(f"Prediction Model: {prediction_model}")

# Placeholder for prediction results
st.write("Predicted data will be displayed here.")
# Note: Add your prediction model implementation here.
