import pandas as pd, json
import numpy as np
import optuna
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Input, Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

tf.keras.backend.clear_session()
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# %%
df = pd.read_csv('feature_data.csv')
df["date"] = pd.to_datetime(df["date"])
df.set_index('date', inplace=True)
             
# %%
X = df[['duration', 'temperature', 'dewpoint', 'land_types', 'year', 'month',
       'day_of_week', 'weekend', 'holiday', 'lockdown', 'covid',
       'energy_movingavg', 'energy_movingstd', 'hour_sin', 'hour_cos']]
y = df['charged_energy']

train_size = int(len(X) * 0.8)  
val_size = int(len(X) * 0.1)   
test_size = len(X) - train_size - val_size 

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Standardize numeric features
numeric_features = ['duration', 'temperature', 'dewpoint',
                    'energy_movingavg', 'energy_movingstd', 'hour_sin', 'hour_cos']
categorical_features = list(set(X.columns) - set(numeric_features))

preprocessor = ColumnTransformer(transformers=[('num', MinMaxScaler(), numeric_features), 
                                               ('cat', 'passthrough', categorical_features)])

X_train = preprocessor.fit_transform(X_train, y_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)


# %% LSTM reshape
X_train_series = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_series = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_series = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

X_train_series = np.array(X_train_series, dtype=np.float32)
X_val_series = np.array(X_val_series, dtype=np.float32)
X_test_series = np.array(X_test_series, dtype=np.float32)

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

best_params = {'lstm_units': 128, 
               'dropout': 0.2, 
               'optimizer': 'adam', 
               'learning_rate': 0.00011543450935125595, 
               'two_layers': True}

lstm_units = best_params["lstm_units"]
dropout_rate = best_params["dropout"]
optimizer_name = best_params["optimizer"]
learning_rate = best_params["learning_rate"]
two_layers = best_params["two_layers"]

if optimizer_name == "adam":
    optimizer = Adam(learning_rate=learning_rate)
elif optimizer_name == "sgd":
    optimizer = SGD(learning_rate=learning_rate)
else:
    optimizer = RMSprop(learning_rate=learning_rate)
    
with tf.device(device):
    model = Sequential()
    model.add(Input(shape=(X_train_series.shape[1], X_train_series.shape[2])))
    model.add(LSTM(lstm_units//2, activation='relu', return_sequences=two_layers))
    model.add(Dropout(dropout_rate))
    
    if two_layers:
        model.add(LSTM(lstm_units, activation='relu'))
        model.add(Dropout(dropout_rate))
            
    model.add(Dense(1))
        
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(X_train_series, y_train, validation_data=(X_val_series, y_val),
                        epochs=30, batch_size=128, verbose=0,callbacks=[early_stopping])

model.save('lstm_temporal_model.keras')  
np.save("lstm_temporal_history.npy", history.history)

y_pred = model.predict(X_test_series, batch_size=32)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'Test set: \n MAE: {mae} \n RMSE: {rmse} \n R-squared: {r2}')
np.savetxt("lstm_temporal_pred.csv", y_pred, delimiter=",")

# %% CNN-LSTM reshape
X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_series = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_series = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

X_train_series = np.array(X_train_series, dtype=np.float32)
X_val_series = np.array(X_val_series, dtype=np.float32)
X_test_series = np.array(X_test_series, dtype=np.float32)

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

# %%
best_params = {
    "filters": 128,
    "kernel_size": 3,
    "lstm_units": 96,
    "dropout": 0.3,
    "learning_rate": 0.001001528738572488,
    "optimizer": "rmsprop"}

filters = best_params['filters']
kernel_size = best_params['kernel_size']
lstm_units = best_params['lstm_units']
dropout_rate = best_params['dropout']
learning_rate = best_params['learning_rate']
optimizers = best_params['optimizer']
    
if optimizers == "adam":
    optimizers = Adam(learning_rate=learning_rate)
elif optimizers == "sgd":
    optimizers = SGD(learning_rate=learning_rate)
else:
    optimizers = RMSprop(learning_rate=learning_rate)
    
with tf.device(device):
    model = Sequential()
    model.add(Input(shape=(X_train_series.shape[1], X_train_series.shape[2])))
    model.add(Conv1D(filters=filters// 2, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(Dropout(rate=dropout_rate))
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(Dropout(rate=dropout_rate))
    model.add(LSTM(units=lstm_units, activation='relu', return_sequences=False))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer=optimizers)
        
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    history = model.fit(X_train_series, y_train,
                        validation_data=(X_val_series, y_val),
                        epochs=10, batch_size=128, 
                        callbacks=[early_stopping], verbose=0)

model.save('hybrid_temporal_model.keras')  
np.save("hybrid_tenporal_history.npy", history.history)
    
y_pred = model.predict(X_test_series, batch_size=32)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'Test set: \n MAE: {mae} \n RMSE: {rmse} \n R-squared: {r2}')
np.savetxt("hybrid_temporal_pred.csv", y_pred, delimiter=",")


# %%
