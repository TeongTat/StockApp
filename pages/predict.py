import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# List of S&P 500 stocks
sp500_stocks = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA"
}

# Streamlit UI
st.title("S&P 500 Stock Price Prediction using ARIMA")

# Dropdown for stock selection
stock_name = st.selectbox("Select a stock:", list(sp500_stocks.keys()))
stock_symbol = sp500_stocks[stock_name]

# Date input for historical data range
start_date = st.date_input("Select start date for historical data", pd.to_datetime("2023-01-01"))
end_date = st.date_input("Select end date", pd.to_datetime("today"))

# Button to fetch and predict
if st.button("Predict"):
    st.subheader(f"Fetching Data for {stock_symbol}...")
    
    # Fetch stock data
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    if stock_data.empty:
        st.error("No data found! Try selecting a different date range.")
    else:
        st.write("Last 5 rows of historical data:")
        st.write(stock_data.tail())

        # Train ARIMA model for Close, High, and Low prices
        def train_arima(series):
            model = ARIMA(series, order=(5, 1, 0))
            return model.fit()
        
        close_model = train_arima(stock_data['Close'].dropna())
        high_model = train_arima(stock_data['High'].dropna())
        low_model = train_arima(stock_data['Low'].dropna())

        # Forecast next 5 days
        forecast_days = 5
        close_forecast = close_model.forecast(steps=forecast_days)
        high_forecast = high_model.forecast(steps=forecast_days)
        low_forecast = low_model.forecast(steps=forecast_days)

        # Prepare DataFrame
        future_dates = pd.date_range(stock_data.index[-1], periods=forecast_days+1)[1:]
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close': close_forecast,
            'Predicted High': high_forecast,
            'Predicted Low': low_forecast
        })
        forecast_df.set_index("Date", inplace=True)
        
        # Display predictions
        st.subheader("Predicted Prices for Next 5 Days")
        st.write(forecast_df)

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        stock_data['Close'][-50:].plot(ax=ax, label="Historical Close Prices", color="blue")
        forecast_df['Predicted Close'].plot(ax=ax, label="Forecast Close", linestyle="dashed", color="red")
        forecast_df['Predicted High'].plot(ax=ax, label="Forecast High", linestyle="dotted", color="green")
        forecast_df['Predicted Low'].plot(ax=ax, label="Forecast Low", linestyle="dotted", color="orange")
        ax.set_title(f"Stock Price Prediction for {stock_symbol}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
