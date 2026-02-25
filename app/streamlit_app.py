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

fig1 = model.plot(forecast)
st.pyplot(fig1)

train = data[:-30]
test = data[-30:]

model_eval = Prophet(seasonality_mode='multiplicative')
model_eval.fit(train)

future_eval = model_eval.make_future_dataframe(periods=30)
forecast_eval = model_eval.predict(future_eval)

forecast_test = forecast_eval[['ds','yhat']].tail(30)

actual = test['y'].values
predicted = forecast_test['yhat'].values

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

st.write("### Model Performance")
st.write(f"MAE: {mae:.4f}")
st.write(f"RMSE: {rmse:.4f}")

forecast_output = forecast[['ds', 'yhat']]
st.download_button(
    label="Download Forecast CSV",
    data=forecast_output.to_csv(index=False),
    file_name='forecast.csv',
    mime='text/csv'
)