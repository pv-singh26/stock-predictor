import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

start = '2010-01-10'
end = '2024-12-31'

st.title('Stock Trend Prediction with Random Forest')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end)
st.subheader('Data from 2010-2024')
st.write(df.describe())

# Use closing prices only
close = df['Close']

# Create lagged features — e.g. use last 100 days to predict next day
lags = 100
data = pd.DataFrame()
for i in range(lags):
    data[f'lag_{i+1}'] = close.shift(i+1)
data['target'] = close.values

data.dropna(inplace=True)

# Split data into train and test
train_size = int(len(data) * 0.7)
train = data[:train_size]
test = data[train_size:]

X_train = train.drop('target', axis=1)
y_train = train['target']

X_test = test.drop('target', axis=1)
y_test = test['target']

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error: {mse:.2f}')

# Plot
st.subheader('Predictions vs Original Price')
fig = plt.figure(figsize=(12,6))
plt.plot(y_test.values, color='blue', label='Original Price')
plt.plot(y_pred, color='red', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
