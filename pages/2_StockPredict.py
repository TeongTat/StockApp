import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from datetime import timedelta

# Streamlit Page Configuration
st.set_page_config(page_title="Stock Prediction with ARIMA", layout="wide")

# Title and Description
st.title("Stock Prediction with ARIMA")
st.write("Predict stock prices for the next day, including high, low, and close prices using ARIMA.")

# **1. Load S&P 500 Tickers and Company Names**
@st.cache_data
def load_sp500_tickers_names():
    """Fetch S&P 500 tickers and company names from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return dict(zip(table["Symbol"], table["Security"]))

tickers_names = load_sp500_tickers_names()

# Dropdown for Ticker Selection
selected_ticker = st.selectbox(
    "Select a stock ticker:",
    options=[f"{ticker} - {name}" for ticker, name in tickers_names.items()]
)

# Extract the ticker symbol from the dropdown selection
ticker_symbol = selected_ticker.split(" - ")[0]

# **2. Fetch Stock Data**
@st.cache_data
def get_stock_data(ticker, period="1y"):
    """Fetch historical stock data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    return stock.history(period=period)

st.subheader(f"Displaying Data for {ticker_symbol}")
stock_data = get_stock_data(ticker_symbol)

if stock_data.empty:
    st.warning("No data available for this ticker.")
else:
    st.write("Historical Stock Data (Last 1 Year):")
    st.dataframe(stock_data)

    # Prepare data for ARIMA (High, Low, and Close prices)
    stock_high = stock_data['High']
    stock_low = stock_data['Low']
    stock_close = stock_data['Close']

# **3. Train ARIMA and Forecast for High, Low, and Close**
def train_arima_and_forecast(data, steps=1):
    """Train ARIMA model and forecast future values."""
    model = ARIMA(data, order=(5, 1, 0))  # (p, d, q) values can be fine-tuned
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=steps)
    return forecast

if not stock_data.empty:
    st.write("Training ARIMA models for High, Low, and Close prices...")

    # Forecast High, Low, and Close for the next day
    next_day_high = train_arima_and_forecast(stock_high, steps=1)[0]
    next_day_low = train_arima_and_forecast(stock_low, steps=1)[0]
    next_day_close = train_arima_and_forecast(stock_close, steps=1)[0]

    # Display Results
    st.subheader("Next Day Predictions:")
    prediction_df = pd.DataFrame({
        "Metric": ["High Price", "Low Price", "Close Price"],
        "Predicted Value": [next_day_high, next_day_low, next_day_close]
    })
    st.write(prediction_df)

    # Visualize Close Predictions
    st.subheader("Historical Close Prices with Next Day Prediction")
    plt.figure(figsize=(10, 6))
    plt.plot(stock_close.index, stock_close, label="Historical Close Prices")
    plt.scatter(
        [stock_close.index[-1] + timedelta(days=1)], [next_day_close],
        color="red", label="Next Day Close Prediction"
    )
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Close Price Prediction")
    plt.legend()
    st.pyplot(plt)

    # **4. Download Prediction Data**
    csv_data = prediction_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Next Day Predictions as CSV",
        data=csv_data,
        file_name=f"{ticker_symbol}_next_day_predictions.csv",
        mime="text/csv"
    )
