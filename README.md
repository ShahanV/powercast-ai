# PowerCast AI 🔋

PowerCast AI is a time-series forecasting project where I am building a machine learning model to predict future energy consumption using historical household electricity usage data. The goal of this project is to understand consumption patterns and develop an AI-based forecasting system that can help in smarter energy planning.

---

## 📌 Problem Statement

Energy consumption forecasting is important for efficient power management, load balancing, and cost optimization. 

The objective of this project is to:
- Analyze historical household energy usage data
- Identify patterns and trends over time
- Build forecasting models to predict future energy consumption
- Evaluate model performance using proper metrics

---

## 📊 Dataset

The project uses the **Household Electric Power Consumption Dataset** from the UCI Machine Learning Repository.

The dataset contains:
- Date and time information
- Global active power consumption
- Voltage and other electrical measurements

---

## ⚠️ Dataset Note

The dataset is not included in this repository due to GitHub file size limitations (files larger than 100MB cannot be pushed).  

Please download the **Household Electric Power Consumption Dataset** manually from the UCI Machine Learning Repository and place the file inside the `data/` folder before running the notebook.

Expected file path:

data/household_power_consumption.txt

---

## 🚀 Project Goals

- Perform data cleaning and preprocessing
- Conduct exploratory data analysis (EDA)
- Implement time-series forecasting models (Prophet, ARIMA)
- Evaluate model performance (MAE, RMSE)
- Build a simple dashboard for predictions (optional enhancement)

---

## 📅 Day 1 Progress

- Repository setup
- Dataset download and loading
- Dataset exploration
- Datetime conversion
- Initial visualization
- Missing value analysis

---

## 🛠 Tech Stack

- Python
- Pandas
- Matplotlib
- Prophet (planned)
- Streamlit (planned)