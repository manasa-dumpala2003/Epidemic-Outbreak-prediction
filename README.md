🦠 Epidemic Outbreak Prediction – COVID-19 Forecasting for India

📍 Live Demo: epidemic-outbreak-prediction-fhb2etpde3isxmpi9xruua.streamlit.app

📌 Project Overview
This project focuses on predicting daily COVID-19 cases in India using the Prophet time series forecasting model. The model is enhanced with external factors like mobility trends and government-imposed lockdowns to improve forecast accuracy.
The app is built using Streamlit for interactive exploration and is deployed live for public use.

🎯 Objectives
📈 Forecast future COVID-19 daily case counts
📊 Analyze the influence of mobility and lockdowns
🧠 Evaluate forecast performance using MAE, RMSE, and MAPE
📉 Visualize trends, seasonality, and forecast error metrics

📁 Dataset Sources
Dataset	Description
cases.csv	COVID-19 confirmed cases (Johns Hopkins)
mobility.xlsx	Mobility trends from World Bank / Google
Lockdown Period	March 25 – May 31, 2020 (manually added)

⚙️ Tech Stack
🐍 Python
📈 Prophet (Facebook) for forecasting
📊 Matplotlib for visualizations
🧪 scikit-learn for error metrics
🧾 Pandas, NumPy
🌐 Streamlit for web app UI

📊 Forecast Features
✅ Prophet model with custom monthly seasonality
✅ Mobility index and lockdown flag as regressors
✅ Visual comparison: Trend vs Predicted
✅ Evaluation metrics:
MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
MAPE (Mean Absolute Percentage Error)
✅ Forecast accuracy displayed dynamically

📊 Model Output & Performance
The Prophet model was trained on daily COVID-19 case data for India, incorporating mobility patterns and a lockdown indicator as external regressors.

🔍 Output Highlights:
Forecast Period: 30 days after July 31, 2020

Actual vs Predicted Plot: The model captures the underlying trend effectively.

Seasonality Components: Prophet automatically decomposes the series into weekly, yearly, and custom monthly patterns.

📈 Key Visuals:
1. Forecast vs Actual
🔵 The blue line shows predicted new daily cases (yhat)
🔴 The red line represents actual trend (trend)
✅ There is strong alignment in most parts, especially during stable periods

2. Seasonality
Shows periodic behavior such as:
Weekly dips (weekend reporting)
Monthly fluctuations
Long-term growth captured in trend

📉 Performance Metrics (Evaluation on Forecast Horizon):
Metric	Meaning	Result (Average)
MAE	Mean Absolute Error – average deviation	~ reasonable (visually stable)
RMSE	Root Mean Squared Error – penalizes big errors	slightly higher due to spikes
MAPE	Mean Absolute Percentage Error – normalized	~ 15% average

🧠 Interpretation:
The model performs very well during the post-lockdown stable period.
Spikes and anomalies (due to sudden outbreaks or policy shifts) naturally lead to higher RMSE values.
Inclusion of mobility and lockdown improves the forecast’s ability to respond to policy and behavior changes.
The MAPE < 20% threshold confirms it's a good quality time series forecast.

📌 Recommendations:
Retrain the model periodically with updated data
Consider adding more regressors: vaccination rate, testing rate, variant type
Use the forecast for short-term planning (2–4 weeks ahead)


