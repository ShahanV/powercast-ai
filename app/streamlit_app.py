import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from datetime import datetime

# -------------------------------------------------
# Page Config & Custom CSS
# -------------------------------------------------
st.set_page_config(
    page_title="PowerCast AI — Energy Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS for a polished, modern look
st.markdown("""
<style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* ---------- Hero Section ---------- */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px;
        padding: 3rem 3rem 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }

    .hero-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        padding: 0.35rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.15;
        margin-bottom: 0.75rem;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: #94a3b8;
        font-weight: 300;
        line-height: 1.7;
        max-width: 600px;
        position: relative;
        z-index: 1;
    }

    /* ---------- Section Headers ---------- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.25rem;
    }

    .section-icon {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
    }

    .section-icon.purple { background: rgba(168, 85, 247, 0.12); }
    .section-icon.blue   { background: rgba(59, 130, 246, 0.12); }
    .section-icon.green  { background: rgba(34, 197, 94, 0.12);  }
    .section-icon.amber  { background: rgba(245, 158, 11, 0.12); }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }

    .section-desc {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        padding-left: 3.5rem;
    }

    /* ---------- Metric Cards ---------- */
    .metric-row {
        display: flex;
        gap: 1.25rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        flex: 1;
        background: linear-gradient(145deg, #1e1b4b 0%, #1e1e2e 100%);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    }

    .metric-label {
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    .metric-value.purple { color: #a78bfa; }
    .metric-value.blue   { color: #60a5fa; }
    .metric-value.green  { color: #4ade80; }
    .metric-value.amber  { color: #fbbf24; }

    .metric-sub {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* ---------- Forecast CTA ---------- */
    .cta-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
    }

    .cta-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.4rem;
    }

    .cta-desc {
        color: #94a3b8;
        font-size: 0.88rem;
        margin-bottom: 1.25rem;
    }

    /* ---------- Dataframe Styling ---------- */
    .forecast-table-wrapper {
        background: #1e1e2e;
        border: 1px solid rgba(99, 102, 241, 0.1);
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 2rem;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }

    .sidebar-logo {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .sidebar-version {
        font-size: 0.75rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }

    .sidebar-section-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #64748b;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
    }

    .sidebar-info-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .sidebar-info-label {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.25rem;
    }

    .sidebar-info-value {
        font-size: 0.92rem;
        color: #e2e8f0;
        font-weight: 500;
    }

    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(99, 102, 241, 0.1);
        margin: 1.25rem 0;
    }

    /* ---------- Download Button Override ---------- */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
    }

    /* ---------- Slider ---------- */
    .stSlider > div > div > div > div {
        background: #6366f1 !important;
    }

    /* ---------- Divider ---------- */
    .custom-divider {
        border: none;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
        margin: 2.5rem 0;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #475569;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
    }

    .footer a {
        color: #818cf8;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚡ PowerCast AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-version">v2.0 — Intelligent Forecasting</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">🧠 Model Details</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info-card">
        <div class="sidebar-info-label">Algorithm</div>
        <div class="sidebar-info-value">Prophet (Meta)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info-card">
        <div class="sidebar-info-label">Seasonality Mode</div>
        <div class="sidebar-info-value">Multiplicative</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info-card">
        <div class="sidebar-info-label">Granularity</div>
        <div class="sidebar-info-value">Daily Aggregation</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info-card">
        <div class="sidebar-info-label">Training Data</div>
        <div class="sidebar-info-value">UCI Household Power</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">📊 Pipeline</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.8;">
        1. Ingest raw consumption data<br>
        2. Clean & forward-fill missing values<br>
        3. Resample to daily means<br>
        4. Hyperparameter-tuned Prophet<br>
        5. Generate probabilistic forecast
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#475569; font-size:0.75rem; text-align:center;">'
        f'Session started {datetime.now().strftime("%b %d, %Y %H:%M")}'
        f'</div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------
# Hero Section
# -------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ AI-POWERED ENERGY INTELLIGENCE</div>
    <div class="hero-title">Predict Tomorrow's<br>Energy Demand, Today.</div>
    <div class="hero-subtitle">
        PowerCast AI uses advanced time-series forecasting to predict household energy
        consumption with confidence intervals — so you can plan smarter, save more,
        and reduce waste.
    </div>
</div>
""", unsafe_allow_html=True)


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
# Quick Stats Row
# -------------------------------------------------
total_days = len(data)
avg_power = data['y'].mean()
max_power = data['y'].max()
min_power = data['y'].min()

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-label">Historical Days</div>
        <div class="metric-value purple">{total_days:,}</div>
        <div class="metric-sub">data points available</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Avg. Daily Power</div>
        <div class="metric-value blue">{avg_power:.2f} kW</div>
        <div class="metric-sub">global active power</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Peak Recorded</div>
        <div class="metric-value amber">{max_power:.2f} kW</div>
        <div class="metric-sub">maximum daily mean</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Lowest Recorded</div>
        <div class="metric-value green">{min_power:.2f} kW</div>
        <div class="metric-sub">minimum daily mean</div>
    </div>
</div>
""", unsafe_allow_html=True)


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
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    <div class="section-icon purple">📈</div>
    <div class="section-title">Generate Forecast</div>
</div>
<div class="section-desc">Select your forecast horizon and watch the model predict future energy consumption in real time.</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cta-container">
    <div class="cta-title">🎯 How far into the future do you want to look?</div>
    <div class="cta-desc">Drag the slider to set your forecast window. The model will generate predictions with upper and lower confidence bounds.</div>
</div>
""", unsafe_allow_html=True)

forecast_days = st.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=90,
    value=30,
    step=1,
    help="Choose between 7 and 90 days for the forecast window.",
)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# --- Custom matplotlib chart ---
fig1, ax1 = plt.subplots(figsize=(14, 5))
fig1.patch.set_facecolor('#0e0e1a')
ax1.set_facecolor('#0e0e1a')

# Historical
hist = forecast[forecast['ds'] <= data['ds'].max()]
fut = forecast[forecast['ds'] > data['ds'].max()]

ax1.plot(hist['ds'], hist['yhat'], color='#6366f1', linewidth=1.2, alpha=0.8, label='Fitted')
ax1.fill_between(hist['ds'], hist['yhat_lower'], hist['yhat_upper'], color='#6366f1', alpha=0.08)

ax1.plot(fut['ds'], fut['yhat'], color='#a78bfa', linewidth=2.2, label='Forecast')
ax1.fill_between(fut['ds'], fut['yhat_lower'], fut['yhat_upper'], color='#a78bfa', alpha=0.18, label='Confidence Interval')

ax1.scatter(data['ds'], data['y'], color='#ffffff', s=2, alpha=0.25, label='Actual', zorder=2)

# Vertical line at forecast start
ax1.axvline(x=data['ds'].max(), color='#fbbf24', linestyle='--', linewidth=1, alpha=0.6)
ax1.text(data['ds'].max(), ax1.get_ylim()[1] * 0.95, '  Forecast Start ',
         color='#fbbf24', fontsize=8, fontweight='bold', ha='left', va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1b4b', edgecolor='#fbbf24', alpha=0.8))

ax1.set_xlabel('Date', color='#94a3b8', fontsize=10, labelpad=10)
ax1.set_ylabel('Global Active Power (kW)', color='#94a3b8', fontsize=10, labelpad=10)
ax1.tick_params(colors='#64748b', labelsize=8)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#1e293b')
ax1.spines['bottom'].set_color('#1e293b')
ax1.legend(facecolor='#1e1b4b', edgecolor='#312e81', fontsize=8, labelcolor='#e2e8f0', loc='upper left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
fig1.autofmt_xdate()

plt.tight_layout()
st.pyplot(fig1)


# -------------------------------------------------
# Forecast Table
# -------------------------------------------------
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    <div class="section-icon blue">📊</div>
    <div class="section-title">Forecast Results</div>
</div>
<div class="section-desc">Detailed daily predictions with confidence intervals. Scroll through or download the full table below.</div>
""", unsafe_allow_html=True)

forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days).copy()
forecast_output.columns = ['Date', 'Predicted (kW)', 'Lower Bound', 'Upper Bound']
forecast_output['Date'] = forecast_output['Date'].dt.strftime('%Y-%m-%d')
forecast_output = forecast_output.round(3)

st.dataframe(
    forecast_output,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Date": st.column_config.TextColumn("📅 Date", width="medium"),
        "Predicted (kW)": st.column_config.NumberColumn("⚡ Predicted (kW)", format="%.3f"),
        "Lower Bound": st.column_config.NumberColumn("🔽 Lower Bound", format="%.3f"),
        "Upper Bound": st.column_config.NumberColumn("🔼 Upper Bound", format="%.3f"),
    },
)


# -------------------------------------------------
# Model Performance
# -------------------------------------------------
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    <div class="section-icon green">🎯</div>
    <div class="section-title">Model Performance</div>
</div>
<div class="section-desc">Evaluated on the last 30 days of historical data. Lower values indicate better accuracy.</div>
""", unsafe_allow_html=True)

train = data[:-30]
test = data[-30:]

future_eval = model.make_future_dataframe(periods=30)
forecast_eval = model.predict(future_eval)
forecast_test = forecast_eval[['ds', 'yhat']].tail(30)

actual = test['y'].values
predicted = forecast_test['yhat'].values

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean Absolute Error</div>
        <div class="metric-value blue">{mae:.4f}</div>
        <div class="metric-sub">average absolute deviation</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Root Mean Squared Error</div>
        <div class="metric-value purple">{rmse:.4f}</div>
        <div class="metric-sub">penalizes large errors more</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mean Abs. % Error</div>
        <div class="metric-value amber">{mape:.2f}%</div>
        <div class="metric-sub">percentage-based accuracy</div>
    </div>
    """, unsafe_allow_html=True)

# Actual vs Predicted Chart
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

fig2, ax2 = plt.subplots(figsize=(14, 4))
fig2.patch.set_facecolor('#0e0e1a')
ax2.set_facecolor('#0e0e1a')

ax2.plot(test['ds'].values, actual, color='#4ade80', linewidth=2, label='Actual', marker='o', markersize=3)
ax2.plot(test['ds'].values, predicted, color='#f472b6', linewidth=2, label='Predicted', marker='s', markersize=3, linestyle='--')
ax2.fill_between(test['ds'].values, actual, predicted, color='#818cf8', alpha=0.08)

ax2.set_xlabel('Date', color='#94a3b8', fontsize=10, labelpad=10)
ax2.set_ylabel('Power (kW)', color='#94a3b8', fontsize=10, labelpad=10)
ax2.tick_params(colors='#64748b', labelsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#1e293b')
ax2.spines['bottom'].set_color('#1e293b')
ax2.legend(facecolor='#1e1b4b', edgecolor='#312e81', fontsize=9, labelcolor='#e2e8f0')
ax2.set_title('Actual vs. Predicted — Last 30 Days', color='#e2e8f0', fontsize=12, fontweight='bold', pad=15)

plt.tight_layout()
st.pyplot(fig2)


# -------------------------------------------------
# Download Section
# -------------------------------------------------
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="section-header">
    <div class="section-icon amber">💾</div>
    <div class="section-title">Export Forecast</div>
</div>
<div class="section-desc">Download the complete forecast table as a CSV file for offline analysis or integration into your systems.</div>
""", unsafe_allow_html=True)

export_col1, export_col2, export_col3 = st.columns([1, 1, 2])

with export_col1:
    st.download_button(
        label="⬇️  Download Forecast CSV",
        data=forecast_output.to_csv(index=False),
        file_name=f'powercast_forecast_{forecast_days}d_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )

with export_col2:
    st.download_button(
        label="📋  Download Evaluation CSV",
        data=pd.DataFrame({'Date': test['ds'].values, 'Actual': actual, 'Predicted': predicted}).to_csv(index=False),
        file_name=f'powercast_evaluation_{datetime.now().strftime("%Y%m%d")}.csv',
        mime='text/csv',
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Built with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a> &
    <a href="https://facebook.github.io/prophet/" target="_blank">Prophet</a><br>
    PowerCast AI © 2026 — Intelligent Energy Forecasting
</div>
""", unsafe_allow_html=True)