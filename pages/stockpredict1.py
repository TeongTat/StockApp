import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load full S&P 500 stock list
def load_sp500_stocks():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return dict(zip(table['Security'], table['Symbol']))

sp500_stocks = load_sp500_stocks()

# Streamlit UI
st.title("S&P 500 Stock Price Prediction using ARIMA")

# Dropdown for stock selection
stock_name = st.selectbox("Select a stock:", list(sp500_stocks.keys()))
stock_symbol = sp500_stocks[stock_name]

# Date input for historical data range
start_date = st.date_input("Select start date for historical data", pd.to_datetime("2023-01-01"))
end_date = st.date_input("Select end date", pd.to_datetime("today"))

@st.cache_data
def fetch_stock_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

# Button to fetch and predict
if st.button("Predict"):
    st.subheader(f"Fetching Data for {stock_symbol}...")
    
    # Fetch stock data
    stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
    
    if stock_data.empty:
        st.error("No data found! Try selecting a different date range.")
    else:
        st.write("Last 5 rows of historical data:")
        st.write(stock_data.tail())

        # Use 'Close', 'High', and 'Low' prices for forecasting
        stock_prices = stock_data[['Close', 'High', 'Low']].dropna()

        def train_arima(series):
            model = ARIMA(series, order=(5, 1, 0))
            return model.fit()

        model_close = train_arima(stock_prices['Close'])
        model_high = train_arima(stock_prices['High'])
        model_low = train_arima(stock_prices['Low'])

        # Forecast next 5 days
        forecast_close = model_close.forecast(steps=5)
        forecast_high = model_high.forecast(steps=5)
        forecast_low = model_low.forecast(steps=5)

        # Display predictions
        future_dates = pd.date_range(stock_prices.index[-1], periods=6)[1:]
        forecast_df = pd.DataFrame({
            'Date': future_dates, 
            'Predicted Close Price': forecast_close,
            'Predicted High Price': forecast_high,
            'Predicted Low Price': forecast_low
        })
        forecast_df.set_index("Date", inplace=True)

        st.subheader("Predicted Prices for Next 5 Days")
        st.write(forecast_df)

        # Plot results
        fig, ax = plt.subplots(figsize=(10, 5))
        stock_prices['Close'][-50:].plot(ax=ax, label="Historical Close Prices", color="blue")
        forecast_df["Predicted Close Price"].plot(ax=ax, label="Forecast Close", linestyle="dashed", color="red")
        forecast_df["Predicted High Price"].plot(ax=ax, label="Forecast High", linestyle="dashed", color="green")
        forecast_df["Predicted Low Price"].plot(ax=ax, label="Forecast Low", linestyle="dashed", color="orange")
        ax.set_title(f"Stock Price Prediction for {stock_symbol}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
