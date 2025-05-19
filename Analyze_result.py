# %% Spatial Model 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# %%
lstm_df = pd.read_csv('lstm_spatiotemporal.csv', header=None, names=['predicted'])
df = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv', index_col='date', parse_dates=True)

test_data = df[-len(lstm_df):]
test_data = test_data[(test_data['month'] >= 6) & (test_data['month'] <= 9)]
test_data_reset = test_data.reset_index()
compare = pd.merge(test_data_reset, lstm_df, left_index=True, right_index=True, how='inner')
compare['day'] = compare['date'].dt.date

group = compare.groupby(['day'])[['charged_energy', 'predicted']].sum().reset_index()
group['mae'] = np.abs(group['charged_energy'] - group['predicted'])

fig, axs = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

axs[0].plot(group['day'], group['charged_energy'], label='Actual', color='blue')
axs[0].plot(group['day'], group['predicted'], label='Predicted', color='orange')
axs[0].set_title('SpatioTemporal LSTM')
axs[0].set_ylabel('Energy Consumption (kWh)')
axs[0].legend()
axs[0].grid(True)
axs[0].xaxis.set_major_locator(mdates.MonthLocator())
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axs[0].tick_params(labelbottom=False)  # Hide x-axis labels

axs[1].plot(group['day'], group['mae'], label='MAE', color='purple')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Error Score')
axs[1].legend()
axs[1].grid(True)

# Show all months on second row
axs[1].xaxis.set_major_locator(mdates.MonthLocator())
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axs[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('spatiotemporal_peak_lstm.png', dpi=300)
plt.show()


### Spatial LSTM hour
cnn_df = pd.read_csv('lstm_spatiotemporal.csv', header=None, names=['predicted'])
df = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv', index_col='date', parse_dates=True)

# Extract test data and align with predictions
test_data = df[-len(cnn_df):]
test_data = test_data.copy()
test_data['predicted'] = cnn_df.values[:len(test_data)]

# Extract features
test_data['hour'] = test_data.index.hour
land_types_to_plot = ['commercial', 'construction']
filtered = test_data[test_data['land_types'].isin(land_types_to_plot)]

group_hourly = (filtered.groupby(['hour', 'land_types'])[['charged_energy', 'predicted']].sum().reset_index())
group_hourly['mae'] = np.abs(group_hourly['charged_energy'] - group_hourly['predicted'])

fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)  
for i, land in enumerate(land_types_to_plot):
    subset = group_hourly[group_hourly['land_types'] == land]  
    
    axs[0, i].plot(subset['hour'], subset['charged_energy'], label='Actual', marker='o', color='blue')
    axs[0, i].plot(subset['hour'], subset['predicted'], label='Predicted', marker='x', color='orange')
    axs[0, i].set_title(f"{land.capitalize()}")
    axs[0, i].set_xlim(0, 24)
    axs[0, i].set_xticks(np.arange(0, 25, 1))
    if i == 0:
        axs[0, i].set_ylabel('Energy Consumption (kWh)')

    axs[0, i].legend()
    axs[0, i].grid(True)

    axs[1, i].plot(subset['hour'], subset['mae'], label='MAE', marker='o', color='purple')
    axs[1, i].set_xlabel('Hour')
    axs[1, i].set_xlim(0, 24)
    axs[1, i].set_xticks(np.arange(0, 25, 1))

    if i == 0:
        axs[1, i].set_ylabel('Error Score')
    axs[1, i].legend()
    axs[1, i].grid(True)

plt.tight_layout()
plt.savefig('spatiotemporal_lstm_by_hour_day.png', dpi = 300)
plt.show()




# %% CNN-LSTM Spatial
cnn_df = pd.read_csv('hybrid_spatiotemporal.csv', header=None, names=['predicted'])
df = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv', index_col='date', parse_dates=True)

test_data = df[-len(cnn_df):]
test_data = test_data[(test_data['month'] >= 6) & (test_data['month'] <= 9)]
test_data_reset = test_data.reset_index()

compare1 = pd.merge(test_data_reset, cnn_df, left_index=True, right_index=True, how='inner')
compare1['day'] = compare1['date'].dt.date

group_cnn= compare1.groupby(['day'])[['charged_energy', 'predicted']].sum().reset_index()
group_cnn['mae'] = np.abs(group_cnn['charged_energy'] - group_cnn['predicted'])

fig, axs = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

axs[0].plot(group_cnn['day'], group_cnn['charged_energy'], color='blue', label='Actual')
axs[0].plot(group_cnn['day'], group_cnn['predicted'], color='orange', label='Predicted')
axs[0].set_title('SpatioTemporal CNN-LSTM')
axs[0].set_ylabel('Energy Consumption (kWh)')
axs[0].legend()
axs[0].grid(True)
axs[0].xaxis.set_major_locator(mdates.MonthLocator())
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axs[0].tick_params(labelbottom=False)  # Hide x-axis labels

axs[1].plot(group_cnn['day'], group_cnn['mae'], label='MAE', color='purple')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Error Score')
axs[1].legend()
axs[1].grid(True)
axs[1].xaxis.set_major_locator(mdates.MonthLocator())
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axs[1].tick_params(axis='x', rotation=0)
    
plt.tight_layout()
plt.savefig('spatiotemporal_comparison_hybrid.png', dpi = 300)  
plt.show()



### Spatial CNN-LSTM hourly prediction by regions
cnn_df = pd.read_csv('hybrid_spatiotemporal.csv', header=None, names=['predicted'])
df = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv', index_col='date', parse_dates=True)

# Extract test data and align with predictions
test_data = df[-len(cnn_df):]
test_data = test_data.copy()
test_data['predicted'] = cnn_df.values[:len(test_data)]

# Extract features
test_data['hour'] = test_data.index.hour
test_data['mae'] = np.abs(test_data['charged_energy'] - test_data['predicted'])

# Select land types to plot
land_types_to_plot = ['commercial', 'construction']
filtered = test_data[test_data['land_types'].isin(land_types_to_plot)]

group_hourly = (filtered.groupby(['hour', 'land_types'])[['charged_energy', 'predicted']].sum().reset_index())
group_hourly['mae'] = np.abs(group_hourly['charged_energy'] - group_hourly['predicted'])

fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)  
for i, land in enumerate(land_types_to_plot):
    subset = group_hourly[group_hourly['land_types'] == land]  
    
    axs[0, i].plot(subset['hour'], subset['charged_energy'], label='Actual', marker='o', color='blue')
    axs[0, i].plot(subset['hour'], subset['predicted'], label='Predicted', marker='x', color='orange')
    axs[0, i].set_title(f"{land.capitalize()}")
    axs[0, i].set_xlim(0, 24)
    axs[0, i].set_xticks(np.arange(0, 25, 1))
    if i == 0:
        axs[0, i].set_ylabel('Energy Consumption (kWh)')

    axs[0, i].legend()
    axs[0, i].grid(True)

    axs[1, i].plot(subset['hour'], subset['mae'], label='MAE', marker='o', color='purple')
    axs[1, i].set_xlabel('Hour')
    axs[1, i].set_xlim(0, 24)
    axs[1, i].set_xticks(np.arange(0, 25, 1))

    if i == 0:
        axs[1, i].set_ylabel('Error Score')
    axs[1, i].legend()
    axs[1, i].grid(True)

plt.tight_layout()
plt.savefig('spatiotemporal_cnn_by_hour_day.png', dpi = 300)
plt.show()





# %% Temporal Models
lstm_df1 = pd.read_csv('lstm_temporal_pred.csv', header=None, names=['predicted'])
df1 = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv', index_col='date', parse_dates=True)

test_data = df1[-len(lstm_df1):]
#test_data = test_data[(test_data['month'] >= 6) & (test_data['month'] <= 9)]
test_data_reset = test_data.reset_index()
compare = pd.merge(test_data_reset, lstm_df1, left_index=True, right_index=True, how='inner')
compare['day'] = compare['date'].dt.date

group = compare.groupby(['day'])[['charged_energy', 'predicted']].sum().reset_index()
group['mae'] = np.abs(group['charged_energy'] - group['predicted'])

fig, axs = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

axs[0].plot(group['day'], group['charged_energy'], label='Actual', color='blue')
axs[0].plot(group['day'], group['predicted'], label='Predicted', color='orange')
axs[0].set_title('Temporal LSTM')
axs[0].set_ylabel('Energy Consumption (kWh)')
axs[0].legend()
axs[0].grid(True)
axs[0].xaxis.set_major_locator(mdates.MonthLocator())
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axs[0].tick_params(labelbottom=False)  # Hide x-axis labels

axs[1].plot(group['day'], group['mae'], label='MAE', color='purple')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Error Score')
axs[1].legend()
axs[1].grid(True)

# Show all months on second row
axs[1].xaxis.set_major_locator(mdates.MonthLocator())
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axs[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('temporal_comparison_lstm.png', dpi =300)  
plt.show()


### LSTM plotting by land type and hour 
lstm_df = pd.read_csv('lstm_temporal_pred.csv', header=None, names=['predicted'])
df = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv', index_col='date', parse_dates=True)

# Extract test data and align with predictions
test_data = df[-len(lstm_df):]
test_data = test_data.copy()
test_data['predicted'] = lstm_df.values[:len(test_data)]

# Extract features
test_data['hour'] = test_data.index.hour
test_data['day_of_week'] = test_data.index.day_name()  # e.g., 'Monday'

# Select land types to plot
land_types_to_plot = ['commercial', 'construction']
filtered = test_data[test_data['land_types'].isin(land_types_to_plot)]

group_hourly = (filtered.groupby(['hour', 'land_types'])[['charged_energy', 'predicted']].sum().reset_index())
group_hourly['mae'] = np.abs(group_hourly['charged_energy'] - group_hourly['predicted'])

fig, axs = plt.subplots(2, 2, figsize=(15, 8), layout='constrained')

# Hour comparison
for i, land in enumerate(land_types_to_plot):
    subset = group_hourly[group_hourly['land_types'] == land]  
    
    axs[0, i].plot(subset['hour'], subset['charged_energy'], label='Actual', marker='o', color='blue')
    axs[0, i].plot(subset['hour'], subset['predicted'], label='Predicted', marker='x', color='orange')
    axs[0, i].set_title(f"{land.capitalize()}")
    axs[0, i].set_xlim(0, 24)
    axs[0, i].set_xticks(np.arange(0, 25, 1))
    if i == 0:
        axs[0, i].set_ylabel('Energy Consumption (kWh)')

    axs[0, i].legend()
    axs[0, i].grid(True)
    axs[0, i].tick_params(labelbottom=False) 

    axs[1, i].plot(subset['hour'], subset['mae'], label='MAE', marker='o', color='purple')
    axs[1, i].set_xlabel('Hour')
    axs[1, i].set_xlim(0, 24)
    axs[1, i].set_xticks(np.arange(0, 25, 1))

    if i == 0:
        axs[1, i].set_ylabel('Error Score')
    axs[1, i].legend()
    axs[1, i].grid(True)
    
plt.tight_layout()
plt.savefig('temporal_lstm_by_hour_day.png', dpi = 300)
plt.show()


# %% CNN-LSTM temporal
hybrid_df1 = pd.read_csv('hybrid_temporal_pred.csv', header=None, names=['predicted'])
df = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv', index_col='date', parse_dates=True)

test_data = df[-len(hybrid_df1):]
test_data = test_data[(test_data['month'] >= 6) & (test_data['month'] <= 9)]
test_data_reset = test_data.reset_index()

compare1 = pd.merge(test_data_reset, hybrid_df1, left_index=True, right_index=True, how='inner')
compare1['day'] = compare1['date'].dt.date

group_cnn= compare1.groupby(['day'])[['charged_energy', 'predicted']].sum().reset_index()
group_cnn['mae'] = np.abs(group_cnn['charged_energy'] - group_cnn['predicted'])

fig, axs = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')

axs[0].plot(group_cnn['day'], group_cnn['charged_energy'], color='blue', label='Actual')
axs[0].plot(group_cnn['day'], group_cnn['predicted'], color='orange', label='Predicted')
axs[0].set_title('Temporal CNN-LSTM')
axs[0].set_ylabel('Energy Consumption (kWh)')
axs[0].legend()
axs[0].grid(True)
axs[0].xaxis.set_major_locator(mdates.MonthLocator())
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axs[0].tick_params(labelbottom=False)  # Hide x-axis labels

axs[1].plot(group_cnn['day'], group_cnn['mae'], label='MAE', color='purple')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Error Score')
axs[1].legend()
axs[1].grid(True)
axs[1].xaxis.set_major_locator(mdates.MonthLocator())
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
axs[1].tick_params(axis='x', rotation=0)
    
    
plt.tight_layout()
plt.savefig('temporal_comparison_hybrid.png', dpi = 300)  
plt.show()



# ####
cnn_df = pd.read_csv('hybrid_temporal_pred.csv', header=None, names=['predicted'])
df = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv', index_col='date', parse_dates=True)

# Extract test data and align with predictions
test_data = df[-len(cnn_df):]
test_data = test_data.copy()
test_data['predicted'] = cnn_df.values[:len(test_data)]

# Extract features
test_data['hour'] = test_data.index.hour

land_types_to_plot = ['commercial', 'construction']
filtered = test_data[test_data['land_types'].isin(land_types_to_plot)]

group_hourly = (filtered.groupby(['hour', 'land_types'])[['charged_energy', 'predicted']].sum().reset_index())
group_hourly['mae'] = np.abs(group_hourly['charged_energy'] - group_hourly['predicted'])

fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True)  
for i, land in enumerate(land_types_to_plot):
    subset = group_hourly[group_hourly['land_types'] == land]  
    
    axs[0, i].plot(subset['hour'], subset['charged_energy'], label='Actual', marker='o', color='blue')
    axs[0, i].plot(subset['hour'], subset['predicted'], label='Predicted', marker='x', color='orange')
    axs[0, i].set_title(f"{land.capitalize()}")
    axs[0, i].set_xlim(0, 24)
    axs[0, i].set_xticks(np.arange(0, 25, 1))
    if i == 0:
        axs[0, i].set_ylabel('Energy Consumption (kWh)')

    axs[0, i].legend()
    axs[0, i].grid(True)
    axs[0, i].tick_params(labelbottom=False) 

    axs[1, i].plot(subset['hour'], subset['mae'], label='MAE', marker='o', color='purple')
    axs[1, i].set_xlabel('Hour')
    axs[1, i].set_xlim(0, 24)
    axs[1, i].set_xticks(np.arange(0, 25, 1))

    if i == 0:
        axs[1, i].set_ylabel('Error Score')
    axs[1, i].legend()
    axs[1, i].grid(True)

plt.tight_layout()
plt.savefig('temporal_cnn_by_hour_day.png', dpi = 300)
plt.show()












# %% By Date and Land types
lstm_df = pd.read_csv('hybrid_temporal_pred.csv', header=None, names=['predicted'])
df = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv', index_col='date', parse_dates=True)

test_data = df[-len(lstm_df):]
test_data_reset = test_data.reset_index()
compare = pd.merge(test_data_reset, lstm_df, left_index=True, right_index=True, how='inner')
compare['day'] = compare['date'].dt.date
group = compare.groupby(['day', 'land_types'])[['charged_energy', 'predicted']].sum().reset_index()
group['mae'] = np.abs(group['charged_energy'] - group['predicted'])

land_types_to_plot = ['commercial', 'construction']
filtered_group = group[group['land_types'].isin(land_types_to_plot)]

# Set up the subplots (2x2 layout)
fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')

# Plot Actual vs Predicted for "construction" land type (first row, first column)
construction_data = filtered_group[filtered_group['land_types'] == 'commercial']
axs[0, 0].plot(construction_data['day'], construction_data['charged_energy'], color='blue', label='Actual')
axs[0, 0].plot(construction_data['day'], construction_data['predicted'], color='orange', label='Predicted')
axs[0, 0].set_title('Commercial')
axs[0, 0].set_ylabel('Charged Energy (kWh)')
axs[0, 0].legend()
axs[0, 0].grid(True)

farmyard_data = filtered_group[filtered_group['land_types'] == 'construction']
axs[0, 1].plot(farmyard_data['day'], farmyard_data['charged_energy'], color='blue', label='Actual')
axs[0, 1].plot(farmyard_data['day'], farmyard_data['predicted'], color='orange', label='Predicted')
axs[0, 1].set_title('Construction')
axs[0, 1].legend()
axs[0, 1].grid(True)

construction_mae = construction_data['mae']
axs[1, 0].plot(construction_data['day'], construction_mae, color='purple', label='MAE')
axs[1, 0].set_xlabel('Date')
axs[1, 0].set_ylabel('MAE')
axs[1, 0].legend()
axs[1, 0].grid(True)

farmyard_mae = farmyard_data['mae']
axs[1, 1].plot(farmyard_data['day'], farmyard_mae, color='purple', label='MAE')
axs[1, 1].set_xlabel('Date')
axs[1, 1].legend()
axs[1, 1].grid(True)
for ax in axs.flat:
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('temp_land_cnn.jpg')  
plt.show()

# %%
