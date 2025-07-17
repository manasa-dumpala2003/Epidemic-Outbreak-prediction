🦠 Epidemic Outbreak Prediction – COVID-19 Forecasting for India

📍 **Live Demo**: [Click here to view the app](https://epidemic-outbreak-prediction-fhb2etpde3isxmpi9xruua.streamlit.app/)

## 📌 Project Overview

This project focuses on predicting daily COVID-19 cases in India using the **Prophet time series forecasting model**.  
The model is enhanced with external factors like **mobility trends** and **government-imposed lockdowns** to improve forecast accuracy.  
The app is built using **Streamlit** for interactive exploration and is deployed live for public use.

## 🎯 Objectives

- 📈 Forecast future COVID-19 daily case counts  
- 📊 Analyze the influence of mobility and lockdowns  
- 🧠 Evaluate forecast performance using MAE, RMSE, and MAPE  
- 📉 Visualize trends, seasonality, and forecast error metrics  

## 📁 Dataset Sources

| File            | Description                                    |
|------------------|------------------------------------------------|
| `cases.csv`      | COVID-19 confirmed cases (Johns Hopkins)       |
| `mobility.xlsx`  | Mobility trends (Google Mobility) |
| `lockdown`       | March 25 – May 31, 2020 (manually flagged)     |

## ⚙️ Tech Stack

- 🐍 Python  
- 📈 Prophet (Facebook) for forecasting  
- 📊 Matplotlib for visualizations  
- 🧪 Scikit-learn for error metrics  
- 🧾 Pandas, NumPy  
- 🌐 Streamlit for web app UI
  
## 📊 Forecast Features

- ✅ Prophet model with custom **monthly seasonality**  
- ✅ External regressors: **mobility index** and **lockdown indicator**  
- ✅ Forecast vs Actual case comparison  
- ✅ Error evaluation metrics:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)  
- ✅ Forecast **accuracy displayed live** in the dashboard  

## 📊 Model Output & Performance

The Prophet model was trained on daily COVID-19 case data for India, incorporating **mobility patterns** and a **lockdown indicator** as external regressors.

### 🔍 Output Highlights

- **Forecast Period**: 30 days after July 31, 2020  
- **Actual vs Predicted Plot**: Model effectively captures the underlying trend  
- **Seasonality Components**: Prophet automatically identifies:
  - Weekly seasonality
  - Monthly fluctuations
  - Long-term trends

### 📈 Key Visuals

#### 🟦 Forecast vs Actual:
- 🔵 **Blue**: Predicted daily cases (`yhat`)  
- 🔴 **Red**: Actual trend (`trend`)  
- ✅ Strong alignment during post-lockdown period

#### 📊 Seasonality:
- Weekly dips (weekend underreporting)  
- Monthly effects  
- Clear post-lockdown trend shifts  

### 📉 Performance Metrics (on 30-day forecast)

| Metric   | Description                              | Result         |
|----------|------------------------------------------|----------------|
| **MAE**  | Mean Absolute Error – avg deviation       | Low            |
| **RMSE** | Root Mean Squared Error – penalizes spikes| Acceptable     |
| **MAPE** | Mean Absolute Percentage Error – normalized| **9.44%**      |

> ✅ **Forecast Accuracy: 90.56%**  

### ✅ Forecast Accuracy Summary (Live Streamlit Output)

| Metric               | Value     |
|----------------------|-----------|
| 📈 **Forecast Accuracy** | **90.56%** |
| 📉 **Average MAPE**      | **9.44%**  |

### 🧠 Interpretation

- Performs well during **stable periods**, post-lockdown  
- Spike days (due to outbreaks or policy changes) lead to slightly higher RMSE  
- External regressors (mobility, lockdown) increase robustness  
- MAPE < 10% confirms it’s a **highly reliable forecast model**

## 📌 Recommendations

- 🔄 Retrain the model periodically with new data  
- ➕ Add more regressors like:
  - Vaccination rate
  - Testing rate
  - Variant impact
- 🗓️ Use for short-term planning (2–4 weeks) by health departments or researchers



