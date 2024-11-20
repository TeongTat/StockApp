import streamlit as st
import requests

#set tittle page
st.tittle("Stock Predictor App")

st.write("""
This is a stock price and prediction modelling app that will assist investors on buying or selling the stocks. The stocks are based on all companies listed on S&P 500 and the price are up to date linking from Yahoo Finance server.
The app will display the following:
- Historical stock price trend up to the latest.
- Showcase stock price forecast (up to 10days).

The main purpose of this application is providing price guidance for investors on the price risks and market risks of the S&P 500 stocks.""")
