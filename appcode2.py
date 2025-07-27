"""import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# === Title ===
st.set_page_config(page_title="COVID-19 Forecast India", layout="wide")
st.title("🦠 COVID-19 Forecasting for India")
st.write("Using Prophet with Mobility and Lockdown Regressors")

# === Load Data ===
@st.cache_data
def load_data():
    covid_data = pd.read_csv("cases.csv")
    mobility_data = pd.read_excel("mobility.xlsx")
    
    covid_filtered = covid_data[covid_data['Country/Region'] == 'India']
    covid_country = covid_filtered.drop(columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum().T
    covid_country.index = pd.to_datetime(covid_country.index)
    df_cases = covid_country.copy()
    df_cases.columns = ['cases']

    mobility_filtered = mobility_data[mobility_data['Economy Name'] == 'India']
    mobility_avg = mobility_filtered.groupby('Economy Name').mean(numeric_only=True).T
    mobility_avg.index = pd.to_datetime(mobility_avg.index)
    df_mob = mobility_avg.copy()
    df_mob.columns = ['mobility']

    df = df_cases.join(df_mob, how='inner')
    df = df[(df.index >= '2020-01-01') & (df.index <= '2021-01-01')]
    df['y'] = df['cases'].diff()
    df = df.dropna().reset_index().rename(columns={'index': 'ds'})
    df['lockdown'] = 0
    df.loc[(df['ds'] >= '2020-03-25') & (df['ds'] <= '2020-05-31'), 'lockdown'] = 1
    return df

df = load_data()
st.success("✅ Data loaded successfully!")

# === Show Raw Data ===
if st.checkbox("🔍 Show Sample Data"):
    st.dataframe(df.head())

# === Model Training ===
train_df = df[df['ds'] >= '2020-02-15'].head(167)

with st.spinner("⚙️ Training Prophet model..."):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_regressor('mobility')
    model.add_regressor('lockdown')
    model.fit(train_df[['ds', 'y', 'mobility', 'lockdown']])

# === Forecasting ===
future = model.make_future_dataframe(periods=30)
future = future.merge(df[['ds', 'mobility', 'lockdown']], on='ds', how='left')
future['mobility'] = future['mobility'].ffill()
future['lockdown'] = future['lockdown'].fillna(0)

forecast = model.predict(future)
forecast['yhat'] = forecast['yhat'].clip(lower=0)

# === Visualization ===
st.subheader("📈 Forecasted vs Actual Cases")

fig1, ax = plt.subplots()
ax.plot(forecast['ds'], forecast['trend'], label='Trend', color='red')
ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='blue')
ax.set_title("COVID-19 Cases Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Daily New Cases")
ax.grid(True)
ax.legend()
st.pyplot(fig1)

# === Seasonality Plot ===
st.subheader("📊 Seasonality Components")
with st.spinner("🔄 Generating seasonality plots..."):
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

# === Forecast Error Evaluation ===
merged_eval = forecast.merge(df[['ds', 'y']], on='ds', how='left')
eval_start = train_df['ds'].max()
merged_eval['days_ahead'] = (merged_eval['ds'] - eval_start).dt.days
eval_df = merged_eval[merged_eval['days_ahead'] > 0].copy()

daily_metrics = []
for day in eval_df['ds'].unique():
    row = eval_df[eval_df['ds'] == day]
    y_true = row['y'].values
    y_pred = row['yhat'].values
    if len(y_true) > 0 and len(y_pred) > 0:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if y_true[0] != 0 else np.nan
        daily_metrics.append({'ds': day, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

metrics_df = pd.DataFrame(daily_metrics)

# === Metrics Plot ===
st.subheader("📉 Daily Forecast Error Metrics")

fig3, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
axs[0].bar(metrics_df['ds'], metrics_df['MAE'], color='orange')
axs[0].set_title("MAE")
axs[1].bar(metrics_df['ds'], metrics_df['RMSE'], color='green')
axs[1].set_title("RMSE")
axs[2].bar(metrics_df['ds'], metrics_df['MAPE'], color='purple')
axs[2].set_title("MAPE")
plt.xticks(rotation=90)
plt.tight_layout()
st.pyplot(fig3)

# === Summary Metrics ===
st.subheader("📌 Average Error Metrics")
st.write(metrics_df.describe()[['MAE', 'RMSE', 'MAPE']])

# === Overall Forecast Accuracy ===
mean_mape = metrics_df['MAPE'].mean()
accuracy = 100 - mean_mape

st.markdown("### ✅ Forecast Accuracy Summary")
col1, col2 = st.columns(2)
col1.metric(label="📈 Forecast Accuracy", value=f"{accuracy:.2f} %")
col2.metric(label="📉 Average MAPE", value=f"{mean_mape:.2f} %")

# === Detailed August Forecast Table ===
st.subheader("📅 August Actual vs Forecasted Cases with MAPE")

# Merge forecast with actual cases
result_df = forecast[['ds', 'yhat']].merge(df[['ds', 'y']], on='ds', how='left')
result_df.rename(columns={'y': 'actual_cases', 'yhat': 'predicted_cases'}, inplace=True)

# Calculate daily MAPE
result_df['MAPE'] = np.where(result_df['actual_cases'] > 0,
                              np.abs((result_df['actual_cases'] - result_df['predicted_cases']) / result_df['actual_cases']) * 100,
                              np.nan)

# Filter for August
august_result = result_df[(result_df['ds'] >= '2020-08-01') & (result_df['ds'] <= '2020-08-31')]

# Select and format columns
august_result = august_result[['ds', 'actual_cases', 'predicted_cases', 'MAPE']]
august_result.columns = ['Date', 'Actual Cases', 'Predicted Cases', 'MAPE (%)']

# Display table
st.dataframe(august_result.style.format({
    'Actual Cases': '{:.0f}',
    'Predicted Cases': '{:.0f}',
    'MAPE (%)': '{:.2f}'
}))

# Download button for CSV
csv = august_result.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download August Forecast as CSV",
    data=csv,
    file_name='august_forecast.csv',
    mime='text/csv'
)"""
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Streamlit Page Config ===
st.set_page_config(page_title="COVID-19 Forecast India", layout="wide")
st.title("🦠 COVID-19 Forecasting for India")
st.write("Using Prophet with Mobility, Lockdown & Holiday Effects")

# === File Upload Section ===
st.sidebar.header("Upload Your Data")
cases_file = st.sidebar.file_uploader("Upload COVID Cases (CSV)", type=["csv"])
mobility_file = st.sidebar.file_uploader("Upload Mobility Data (Excel)", type=["xlsx"])

# === Load Default Data if No Upload ===
@st.cache_data
def load_default_data():
    covid_data = pd.read_csv("cases.csv")
    mobility_data = pd.read_excel("mobility.xlsx")
    return covid_data, mobility_data

if cases_file and mobility_file:
    covid_data = pd.read_csv(cases_file)
    mobility_data = pd.read_excel(mobility_file)
else:
    st.info("Using default dataset")
    covid_data, mobility_data = load_default_data()

# === Preprocess Data ===
covid_filtered = covid_data[covid_data['Country/Region'] == 'India']
covid_country = covid_filtered.drop(columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum().T
covid_country.index = pd.to_datetime(covid_country.index)
df_cases = covid_country.copy()
df_cases.columns = ['cases']

mobility_filtered = mobility_data[mobility_data['Economy Name'] == 'India']
mobility_filtered['date'] = pd.to_datetime(mobility_filtered['date'])
df_mob = mobility_filtered[['date', 'mobility']].rename(columns={'date': 'ds'})
df_cases.reset_index(inplace=True)
df_cases.rename(columns={'index': 'ds'}, inplace=True)
df = df_cases.merge(df_mob, on='ds', how='inner')
df['y'] = df['cases'].diff()
df = df.dropna()
df['lockdown'] = 0
df.loc[(df['ds'] >= '2020-03-25') & (df['ds'] <= '2020-05-31'), 'lockdown'] = 1

# === Add Holidays ===
holidays = pd.DataFrame({
    'holiday': ['lockdown_start', 'lockdown_end', 'diwali'],
    'ds': pd.to_datetime(['2020-03-25', '2020-05-31', '2020-11-14']),
    'lower_window': 0,
    'upper_window': 1
})

# === Show Raw Data ===
if st.checkbox("🔍 Show Sample Data"):
    st.dataframe(df.head())

# === Forecast Horizon ===
forecast_days = st.slider("Select Forecast Horizon (Days)", 7, 90, 30)

# === Model Training ===
with st.spinner("⚙️ Training Prophet model..."):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_regressor('mobility')
    model.add_regressor('lockdown')
    model.fit(df[['ds', 'y', 'mobility', 'lockdown']])

# === Forecasting ===
future = model.make_future_dataframe(periods=forecast_days)
future = future.merge(df[['ds', 'mobility', 'lockdown']], on='ds', how='left')
future['mobility'] = future['mobility'].ffill()
future['lockdown'] = future['lockdown'].fillna(0)
forecast = model.predict(future)
forecast['yhat'] = forecast['yhat'].clip(lower=0)

# === Interactive Forecast Plot ===
st.subheader("📈 Forecasted vs Actual Cases")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual', line=dict(color='black')))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dot', color='lightblue')))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dot', color='lightblue')))
fig1.update_layout(title="COVID-19 Forecast with Confidence Intervals", xaxis_title="Date", yaxis_title="Daily Cases")
st.plotly_chart(fig1, use_container_width=True)

# === Seasonality Components ===
st.subheader("📊 Seasonality Components")
st.write("Hover to inspect values")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# === Evaluation Metrics ===
eval_df = forecast.merge(df[['ds', 'y']], on='ds', how='left').dropna()
y_true = eval_df['y']
y_pred = eval_df['yhat']
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
accuracy = max(0, 100 - mape)

st.subheader("📌 Evaluation Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("MAPE", f"{mape:.2f}%")
col4.metric("Accuracy", f"{accuracy:.2f}%")
