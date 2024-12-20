import streamlit as st
import yfinance as yf
import pandas as pd


# Streamlit App Title
st.title("Stock Data Viewer")

# Sidebar Inputs
st.sidebar.header("Input Options")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

# Fetch Data
if ticker:
    try:
        st.write(f"Fetching data for **{ticker.upper()}** from **{start_date}** to **{end_date}**")
        data = yf.download(ticker, start=start_date, end=end_date)

        # Check if Data is Valid
        if not data.empty:
            # Display Raw Data
            st.subheader("Raw Data")
            st.write(data)

            # Plot Data
            st.subheader("Closing Price Over Time")
            st.line_chart(data["Close"])

            # Add Moving Averages (Optional)
            st.subheader("With Moving Averages")
            data["MA10"] = data["Close"].rolling(window=10).mean()
            data["MA50"] = data["Close"].rolling(window=50).mean()
            st.line_chart(data[["Close", "MA10", "MA50"]])
        else:
            st.warning("No data found for the given ticker and date range.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Enter a valid stock ticker to start.")
