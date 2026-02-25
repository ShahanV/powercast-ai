import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.title("🔋 PowerCast AI")
st.subheader("Energy Consumption Forecasting Dashboard")

st.write("Forecast future energy consumption using optimized Prophet model.")

@st.cache_data
def load_data():
    df = pd.read_csv("data/household_power_consumption.txt", sep=';', low_memory=False)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df.set_index('Datetime', inplace=True)
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
    df['Global_active_power'].fillna(method='ffill', inplace=True)
    daily_df = df['Global_active_power'].resample('D').mean()
    prophet_df = daily_df.reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

data = load_data()

forecast_days = st.slider("Select number of days to forecast:", 7, 60, 30)

model = Prophet(seasonality_mode='multiplicative')
model.fit(data)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)