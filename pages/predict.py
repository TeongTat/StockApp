import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Streamlit App Title
st.title("Cryptocurrency Price Prediction using LSTM")

# Cryptocurrency selection
crypto_dict = {"Bitcoin (BTC)": "BTC-USD", "Ethereum (ETH)": "ETH-USD", "XRP (XRP)": "XRP-USD"}
crypto_name = st.selectbox("Select Cryptocurrency:", list(crypto_dict.keys()))
crypto_symbol = crypto_dict[crypto_name]

# Date selection
start_date = st.date_input("Select start date for historical data", pd.to_datetime("2023-01-01"))
end_date = st.date_input("Select end date", pd.to_datetime("today"))

# Button to trigger prediction
if st.button("Predict"):
    st.subheader(f"Fetching Data for {crypto_symbol}...")
    
    # Fetch historical data
    data = yf.download(crypto_symbol, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found! Try selecting a different date range.")
    else:
        st.write("Last 5 rows of historical data:")
        st.write(data.tail())
        
        # Prepare data for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[['Close']])
        
        # Create sequences for LSTM
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        seq_length = 50
        X_train, y_train = create_sequences(scaled_data, seq_length)
        
        # Reshape for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        # Build LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)
        
        # Predict next 5 days
        last_seq = scaled_data[-seq_length:]
        last_seq = np.reshape(last_seq, (1, seq_length, 1))
        predictions = []
        for _ in range(5):
            next_price = model.predict(last_seq)[0][0]
            predictions.append(next_price)
            last_seq = np.append(last_seq[:, 1:, :], [[[next_price]]], axis=1)
        
        # Convert predictions back to actual prices
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Display predictions
        future_dates = pd.date_range(data.index[-1], periods=6)[1:]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': predictions.flatten()})
        forecast_df.set_index("Date", inplace=True)
        st.subheader("Predicted Prices for Next 5 Days")
        st.write(forecast_df)
        
        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data.index[-50:], data['Close'].values[-50:], label="Historical Prices", color="blue")
        ax.plot(forecast_df.index, forecast_df['Predicted Price'], label="Predicted Prices", linestyle="dashed", color="red")
        ax.set_title(f"Cryptocurrency Price Prediction for {crypto_symbol}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        st.pyplot(fig)
