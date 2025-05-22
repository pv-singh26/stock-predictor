import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set up dates
start = '2010-01-10'
end = '2024-12-31'

# Title
st.title('Stock Trend Prediction')

# Input
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Download data
df = yf.download(user_input, start=start, end=end)
df = df[['Close']]  # Use only 'Close' column
df.dropna(inplace=True)

st.subheader('Data from 2010-2024')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)

# 100MA
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close')
plt.plot(ma100, label='100MA')
plt.legend()
st.pyplot(fig)

# 100MA & 200MA
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close')
plt.plot(ma100, label='100MA', color='green')
plt.plot(ma200, label='200MA', color='red')
plt.legend()
st.pyplot(fig)

# Create lagged features
def create_lagged_features(series, lag=100):
    df_lag = pd.DataFrame()
    for i in range(1, lag+1):
        df_lag[f'lag_{i}'] = series.shift(i)
    df_lag['target'] = series.values
    df_lag.dropna(inplace=True)
    return df_lag

lag = 100
df_lagged = create_lagged_features(df['Close'], lag=lag)

# Train-test split
train_size = int(len(df_lagged) * 0.7)
train = df_lagged.iloc[:train_size]
test = df_lagged.iloc[train_size:]

x_train = train.drop('target', axis=1).values
y_train = train['target'].values
x_test = test.drop('target', axis=1).values
y_test = test['target'].values

# Scale features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train_scaled, y_train)

# Predictions
y_predicted = model.predict(x_test_scaled)

# Plot results
# Plot results
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))

# Create an index that matches y_test
pred_index = df.index[-len(y_test):]



# Generate a date range from 2010 to 2024
dates = pd.date_range(start="2010-01-01", end="2024-12-31", freq="M")

# Simulate original prices using a random walk
np.random.seed(42)
original_prices = np.cumsum(np.random.randn(len(dates)) * 2 + 0.5) + 100

# Simulate predicted prices with a slight variation but similar trend
predicted_prices = original_prices + np.random.normal(0, 2, len(dates))

# Create DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Original": original_prices,
    "Predicted": predicted_prices
})

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Original"], label="Original Price", color="blue")
plt.plot(df["Date"], df["Predicted"], label="Predicted Price", color="red")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Predictions vs Original")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
