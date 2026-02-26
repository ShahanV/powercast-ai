# PowerCast AI

PowerCast AI is a time-series forecasting project where I am building a machine learning model to predict future energy consumption using historical household electricity usage data. The goal of this project is to understand consumption patterns and develop an AI-based forecasting system that can help in smarter energy planning.

---

## Problem Statement

Energy consumption forecasting is important for efficient power management, load balancing, and cost optimization. 

The objective of this project is to:
- Analyze historical household energy usage data
- Identify patterns and trends over time
- Build forecasting models to predict future energy consumption
- Evaluate model performance using proper metrics

---

## Dataset

The project uses the **Household Electric Power Consumption Dataset** from the UCI Machine Learning Repository.

The dataset contains:
- Date and time information
- Global active power consumption
- Voltage and other electrical measurements

---

## Dataset Note

The dataset is not included in this repository due to GitHub file size limitations (files larger than 100MB cannot be pushed).  

Please download the **Household Electric Power Consumption Dataset** manually from the UCI Machine Learning Repository and place the file inside the `data/` folder before running the notebook.

Expected file path:

data/household_power_consumption.txt

---

## Project Goals

- Perform data cleaning and preprocessing
- Conduct exploratory data analysis (EDA)
- Implement time-series forecasting models (Prophet, ARIMA)
- Evaluate model performance (MAE, RMSE)
- Build a simple dashboard for predictions (optional enhancement)

---

## Day 1 Progress

- Repository setup
- Dataset download and loading
- Dataset exploration
- Datetime conversion
- Initial visualization
- Missing value analysis

---

## Day 2 Progress

- Handled missing values using forward fill
- Resampled data to daily averages
- Performed monthly trend analysis
- Applied rolling mean for smoothing
- Created train-test split (last 30 days as test set)
- Prepared dataset in Prophet format (ds, y)

---

## Day 3 Progress

- Installed and implemented Prophet forecasting model
- Trained model on historical daily energy data
- Generated 30-day future forecast
- Visualized predictions with confidence intervals
- Evaluated model using MAE and RMSE
- Compared actual vs predicted energy consumption

---

## Day 4 Progress

- Implemented ARIMA model for time-series forecasting
- Generated 30-day ARIMA forecast
- Evaluated ARIMA using MAE and RMSE
- Compared Prophet vs ARIMA performance
- Selected Prophet as the better-performing model

---

## Day 5 Progress

- Stored baseline Prophet performance metrics
- Tested multiplicative seasonality configuration
- Tuned changepoint prior scale parameter
- Added explicit yearly and weekly seasonality
- Evaluated all configurations using MAE and RMSE
- Compared tuned models against baseline
- Selected multiplicative seasonality as the final optimized model

---

## Day 6 Progress

- Built interactive Streamlit dashboard
- Integrated optimized Prophet model
- Added forecast days slider
- Displayed live forecast visualization
- Showed MAE and RMSE in UI
- Added forecast CSV download functionality

---

## Day 7 Progress

- Serialized the final optimized Prophet model using joblib
- Integrated the pre-trained model into the Streamlit dashboard
- Removed retraining from the application for faster inference
- Displayed forecast results directly in the dashboard with confidence intervals
- Improved UI layout and organized sections for better clarity
- Finalized project structure for submission
- Prepared application for demo and deployment

---

## Tech Stack

- Python
- Pandas
- Matplotlib
- Prophet (planned)
- Streamlit (planned)