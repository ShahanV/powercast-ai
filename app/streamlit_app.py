import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.set_page_config(page_title="PowerCast AI", layout="wide")

st.title("🔋 PowerCast AI")
st.subheader("Energy Consumption Forecasting Dashboard")
st.write("Forecast future energy consumption using optimized Prophet model.")

# -------------------------------------------------
# Load Data
# -------------------------------------------------
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

# -------------------------------------------------
# Load Saved Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/powercast_prophet_model.pkl")

model = load_model()

# -------------------------------------------------
# Forecast Section
# -------------------------------------------------
st.markdown("## 📈 Generate Forecast")

forecast_days = st.slider("Select number of days to forecast:", 7, 60, 30)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

fig1 = model.plot(forecast)
st.pyplot(fig1)

# -------------------------------------------------
# Forecast Table Display
# -------------------------------------------------
st.markdown("## 📊 Forecasted Energy Consumption")

forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)

forecast_output = forecast_output.round(3)

st.dataframe(forecast_output, use_container_width=True)

# -------------------------------------------------
# Evaluation Section
# -------------------------------------------------
st.markdown("## 📏 Model Performance")

train = data[:-30]
test = data[-30:]

future_eval = model.make_future_dataframe(periods=30)
forecast_eval = model.predict(future_eval)

forecast_test = forecast_eval[['ds', 'yhat']].tail(30)

actual = test['y'].values
predicted = forecast_test['yhat'].values

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", f"{mae:.4f}")

with col2:
    st.metric("RMSE", f"{rmse:.4f}")

# -------------------------------------------------
# Download Forecast
# -------------------------------------------------
st.markdown("## 💾 Export Forecast")

st.download_button(
    label="Download Forecast CSV",
    data=forecast_output.to_csv(index=False),
    file_name='forecast.csv',
    mime='text/csv'
)

# -------------------------------------------------
# Sidebar Info
# -------------------------------------------------
st.sidebar.markdown("## ℹ️ Model Info")
st.sidebar.write("Final Model: Prophet (Multiplicative Seasonality)")
st.sidebar.write("Selected after benchmarking and hyperparameter tuning.")
st.sidebar.write("Trained on daily aggregated household energy consumption data.")