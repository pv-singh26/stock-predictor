import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Streamlit UI
st.title('Stock Trend Prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Download data
start = '2010-01-10'
end = '2024-12-31'
df = yf.download(user_input, start=start, end=end)

# Show basic data info
st.subheader('Data from 2010-2024')
st.write(df.describe())

# Plot closing price
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price')
plt.legend()
st.pyplot(fig)

# Plot with 100 MA
st.subheader('Closing Price with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, label='100MA')
plt.plot(df['Close'], label='Closing Price')
plt.legend()
st.pyplot(fig)

# Plot with 100 & 200 MA
st.subheader('Closing Price with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'g', label='100MA')
plt.plot(ma200, 'r', label='200MA')
plt.plot(df['Close'], 'b', label='Closing Price')
plt.legend()
st.pyplot(fig)

# Helper function to create lagged features
def create_lagged_features(series, lag=50):
    df_lag = pd.DataFrame()
    for i in range(lag):
        df_lag[f'lag_{i+1}'] = series.shift(i+1)
    df_lag['target'] = series.values
    df_lag.dropna(inplace=True)
    return df_lag

# Create lagged dataset
lag = 50
df_lagged = create_lagged_features(df['Close'], lag=lag)

# Split into train and test sets
train_size = int(len(df_lagged) * 0.7)
train = df_lagged[:train_size]
test = df_lagged[train_size:]

x_train = train.drop('target', axis=1).values
y_train = train['target'].values
x_test = test.drop('target', axis=1).values
y_test = test['target'].values

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the model (increased depth and estimators)
model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
model.fit(x_train_scaled, y_train)

# Make predictions
y_predicted = model.predict(x_test_scaled)

# Show comparison (last 200 for readability)
st.subheader('Predictions vs Actual (last 200 points)')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test[-200:], 'b', label='Actual Price')
plt.plot(y_predicted[-200:], 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
