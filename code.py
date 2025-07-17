import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# === Load and Filter Data for India ===#2020-01-22 to 2023-03-09 ,2020-02-15 to 2022-10-15
covid_data = pd.read_csv(r'c:\users\Admin\cases.csv')
#print(type(covid_data))
#print(covid_data.head())

mobility_data = pd.read_excel(r'c:\users\Admin\mobility.xlsx')
#print(mobility_data.head())
#print(type(mobility_data))

country = 'India'
covid_filtered = covid_data[covid_data['Country/Region'] == country]
#print(covid_filtered)
covid_country = covid_filtered.drop(columns=['Province/State', 'Lat', 'Long']).groupby('Country/Region').sum().T
#print(covid_country.head())
covid_country.index = pd.to_datetime(covid_country.index)
#print(covid_country.head())

mobility_filtered = mobility_data[mobility_data['Economy Name'] == country]
#print(mobility_filtered.head())
mobility_avg = mobility_filtered.groupby('Economy Name').mean(numeric_only=True).T
#print(mobility_avg.head())
mobility_avg.index = pd.to_datetime(mobility_avg.index)
#print(mobility_avg.head())

# === Prepare Data ===

df_cases = covid_country[[country]].copy()
#print(df_cases.head())
df_cases.columns = ['cases']#renaming the column
#print(df_cases.head())
df_mob = mobility_avg[[country]].copy()
#print(df_mob.head())
df_mob.columns = ['mobility']#renaming the column
#print(df_mob.head())

df = df_cases.join(df_mob, how='inner')
#print(df.tail())
df = df[(df.index >= '2020-01-01') & (df.index <= '2021-01-01')]
#print(df.tail())
df['y'] = df['cases'].diff()
#print(df.head(30))
df = df.dropna()
#print(df.head())
df = df.reset_index().rename(columns={'index': 'ds'})
#print(df.head())

# === Add Lockdown Regressor ===
df['lockdown'] = 0
df.loc[(df['ds'] >= '2020-03-25') & (df['ds'] <= '2020-05-31'), 'lockdown'] = 1
#print(df.iloc[30:76])

# === Training Data ===
train_df = df[df['ds'] >= '2020-02-15'].head(167)#training 2020-02-15 to 2020-07-31
#print(train_df)

# === Model with Seasonality and Regressors ===
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add custom monthly seasonality
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

# Add external regressors
model.add_regressor('mobility')
model.add_regressor('lockdown')

# Fit the model
model.fit(train_df[['ds', 'y', 'mobility', 'lockdown']])

# === Forecasting ===

future = model.make_future_dataframe(periods=30)
#print(future.head())
future = future.merge(df[['ds', 'mobility', 'lockdown']], on='ds', how='left')
#print(future.head())
future['mobility'] = future['mobility'].ffill()
future['lockdown'] = future['lockdown'].fillna(0)
forecast = model.predict(future)
#print(forecast)

# === Postprocess Forecast ===
forecast_filtered = forecast[['ds', 'trend', 'yhat']].copy()
#print(forecast_filtered.head(20))
forecast_filtered.loc[:, 'yhat'] = forecast_filtered['yhat'].clip(lower=0)
#print(forecast_filtered)
forecast_filtered.loc[:, 'Country'] = country
forecast_filtered = forecast_filtered[forecast_filtered['ds'] <= '2020-08-30']
#print(forecast_filtered)
forecast_filtered.to_csv(f'forecast_{country.lower()}.csv', index=False)

# === Plot Forecast ===
plt.figure(figsize=(10, 6))
plt.plot(forecast_filtered['ds'], forecast_filtered['trend'], label='Actual Cases', color='red')
plt.plot(forecast_filtered['ds'], forecast_filtered['yhat'], label='Predicted Cases', color='blue')
plt.title(f'Forecast for {country}')
plt.xlabel('Date')
plt.ylabel('Covid-19 Cases')
plt.grid(True)
plt.legend()
plt.savefig(f'custom_forecast_plot_{country.lower()}.png')
plt.close()

# === Plot Seasonality Components ===
model.plot_components(forecast)
plt.savefig(f'seasonality_components_{country.lower()}.png')
plt.close()

# === Evaluation: Merge Forecast and Actuals ===#2020-08-13
merged_eval = forecast_filtered.merge(df[['ds', 'y']], on='ds', how='left')
#print(merged_eval.head())
eval_start = train_df['ds'].max()
merged_eval['days_ahead'] = (merged_eval['ds'] - eval_start).dt.days
eval_df = merged_eval[merged_eval['days_ahead'] > 0].copy()
#print(eval_df)

# ===calculating daily forecast error metrics (MAE, RMSE, MAPE)===
daily_metrics = []

for day in eval_df['ds'].unique():
    row = eval_df[eval_df['ds'] == day]
    y_true = row['y'].values
    y_pred = row['yhat'].values

    if len(y_true) > 0 and len(y_pred) > 0:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if y_true[0] != 0 else np.nan
        daily_metrics.append({'ds': day, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

# === Convert to DataFrame and Save ===
metrics_df = pd.DataFrame(daily_metrics)
metrics_df.to_csv(f'daily_metrics_{country.lower()}.csv', index=False)

# === Plot Daily Metrics ===
plt.subplot(1,3,1)
plt.bar(metrics_df['ds'], metrics_df['MAE'])
plt.title("MAE METRICS")
plt.xticks(rotation=90)
plt.subplot(1,3,2)
plt.bar(metrics_df['ds'], metrics_df['RMSE'])
plt.title("RMSE METRICS")
plt.xticks(rotation=90)
plt.subplot(1,3,3)
plt.bar(metrics_df['ds'], metrics_df['MAPE'])
plt.gca().set_yticks([10,30,50])
plt.title("MAPE METRICS")
plt.xticks(rotation=90)
#plt.figure(figsize=(10,10))
plt.suptitle('Daily Forecast Error Metrics')
plt.xlabel('Date')
plt.ylabel('Error Value')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'daily_metrics_plot_{country.lower()}.png')
plt.show()


