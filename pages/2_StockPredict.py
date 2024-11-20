import streamlit as st

st.title("Stock Prediction")

st.write("Set up your prediction preferences below.")

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

# Sidebar options for prediction period and model
prediction_period = st.selectbox("Prediction Period", ["1 Month", "3 Months", "6 Months"])
prediction_model = st.selectbox("Prediction Model", ["Linear Regression", "LSTM", "ARIMA"])

st.write("Prediction settings selected:")
st.write(f"Prediction Period: {prediction_period}")
st.write(f"Prediction Model: {prediction_model}")

# Placeholder for prediction results
st.write("Predicted data will be displayed here.")
# Note: Add your prediction model implementation here.
