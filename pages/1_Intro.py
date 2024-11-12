#core package
import streamlit as st
import requests
import streamlit.components.v1 as components
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import yfinance as yf
import streamlit as st
import pandas as pd

# Title for the app
st.title("S&P 500 Stock Information and Prediction")

# Sidebar for navigation - acts as a tab selector
st.sidebar.title("Navigation")
selected_option = st.sidebar.radio("Go to:", ["Stock Information", "Stock Prediction"])

# Load S&P 500 tickers and company names
@st.cache_data  # Cache the data to avoid reloading on every interaction
def load_sp500_tickers_names():
    # Scrape S&P 500 tickers and company names from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]  # Load the first table on the page
    # Create a dictionary with Ticker as key and Name as value
    tickers_names = dict(zip(table["Symbol"], table["Security"]))
    return tickers_names

# Function to get stock data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    return stock.history(period="1y")  # adjust the period as needed

# Load tickers and company names
tickers_names = load_sp500_tickers_names()

# Dropdown for ticker selection with company names
selected_ticker = st.selectbox(
    "Select a stock ticker:", 
    options=[f"{ticker} - {name}" for ticker, name in tickers_names.items()]
)

# Extract the actual ticker from the selected option
ticker_symbol = selected_ticker.split(" - ")[0]

# Display stock data
if ticker_symbol:
    st.subheader(f"Stock Data for {ticker_symbol}")
    stock_data = get_stock_data(ticker_symbol)
    st.write(stock_data)
    
    # Display charts
    st.line_chart(stock_data['Close'], width=0, height=0)
    st.line_chart(stock_data['Volume'], width=0, height=0)

# Function to display stock prediction options
def display_stock_prediction():
    st.subheader("Stock Prediction Settings")
    # Sidebar options for prediction period and model
    prediction_period = st.selectbox("Prediction Period", ["1 Month", "3 Months", "6 Months"])
    prediction_model = st.selectbox("Prediction Model", ["Linear Regression", "LSTM", "ARIMA"])
    
    st.write("Prediction settings selected:")
    st.write(f"Prediction Period: {prediction_period}")
    st.write(f"Prediction Model: {prediction_model}")
    
    # Placeholder for prediction results
    st.write("Predicted data will be displayed here.")
    # Note: Add your prediction model implementation here.

# Sidebar-based navigation
if selected_option == "Stock Information":
    display_stock_info()
elif selected_option == "Stock Prediction":
    display_stock_prediction()
