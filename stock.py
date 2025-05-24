import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
 
import streamlit as st

start = '2010-01-10'
end = '2024-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2010-2024')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

model = load_model('pvs.h5')

st.subheader('Prediction vs Original')
ma100 = df.Close.rolling(10).mean()
ma200 = df.Close.rolling(50).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'b', label= 'Original Price')
plt.plot(ma200, 'r', label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


