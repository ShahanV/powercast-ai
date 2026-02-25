import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.title("🔋 PowerCast AI")
st.subheader("Energy Consumption Forecasting Dashboard")

st.write("Forecast future energy consumption using optimized Prophet model.")