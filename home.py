import streamlit as st

st.set_page_config(layout="wide")

# Main title and description for the app
st.title("S&P 500 Stock Information & Prediction App")
st.write("Welcome! Use the sidebar to navigate between pages to view stock information or set up predictions.")

# Sidebar for navigation - acts as a tab selector
st.sidebar.title("Navigation")
selected_option = st.sidebar.radio("Go to:", ["Stock Information", "Stock Prediction"])
