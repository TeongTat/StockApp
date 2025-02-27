import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Fetch full S&P 500 stock list
@st.cache
def get_sp500_stocks():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)[0]
    return dict(zip(sp500_table['Security'], sp500_table['Symbol']))

sp500_stocks = get_sp500_stocks()

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

        # Use 'Close' price for forecasting
        stock_prices = stock_data['Close'].dropna()

        # Train ARIMA model (p=5, d=1, q=0)
        model = ARIMA(stock_prices, order=(5, 1, 0))
        model_fit = model.fit()

        # Forecast next 5 days
        forecast = model_fit.forecast(steps=5)

        # Forecast high and low prices using a simple variation
        high_forecast = forecast * 1.02  # Assume high price is 2% higher
        low_forecast = forecast * 0.98   # Assume low price is 2% lower

        # Display predictions
        future_dates = pd.date_range(stock_prices.index[-1], periods=6)[1:]  # Exclude last known date
        forecast_df = pd.DataFrame({
            'Date': future_dates, 
            'Predicted Close Price': forecast,
            'Predicted High Price': high_forecast,
            'Predicted Low Price': low_forecast
        })
        forecast_df.set_index("Date", inplace=True)

        st.subheader("Predicted Prices for Next 5 Days")
        st.write(forecast_df)

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        stock_prices[-50:].plot(ax=ax, label="Historical Prices", color="blue")
        forecast_df["Predicted Close Price"].plot(ax=ax, label="Forecast Close", linestyle="dashed", color="red")
        forecast_df["Predicted High Price"].plot(ax=ax, label="Forecast High", linestyle="dotted", color="green")
        forecast_df["Predicted Low Price"].plot(ax=ax, label="Forecast Low", linestyle="dotted", color="purple")
        ax.set_title(f"Stock Price Prediction for {stock_symbol}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
