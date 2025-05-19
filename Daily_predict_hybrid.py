# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as K
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, MaxPooling1D, LSTM, Dropout, Embedding, Flatten, Concatenate, Dense
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


K.clear_session()

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        # Restrict TensorFlow to only use GPU:0
        tf.config.set_visible_devices(gpus[1], 'GPU')
        print(f"Using only GPU: {gpus[1].name}")
    except RuntimeError as e:
        print(f"Error limiting GPUs: {e}")

# Allow memory growth on GPUs to prevent TensorFlow from allocating all GPU memory at once
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for all GPUs.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")

# Load dataset
df = pd.read_csv('feature_data.csv')
df = df.sort_values(['date'])
df.set_index('date', inplace=True)

encoder = LabelEncoder()
df['day_of_week_encoded'] = encoder.fit_transform(df['day_of_week'])
df['time_slot_encoded'] = encoder.fit_transform(df['time_slot'])
df['land_types_encoded'] = encoder.fit_transform(df['land_types'])

split_index = int(len(df) * 0.8)
train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

val_split = int(len(train_data) * 0.2)
val_data = train_data[-val_split:]
train_data = train_data[:-val_split]

# %% Categorical encoding
numeric_features = ['duration', 'temperature', 'year', 'dewpoint', 'road_density', 'commercial_density',
                   'residential_density', 'recreation_density', 'highway_proximity', 
                    'public_transport_proximity', 'evcs_proximity', 'center_proximity', 'parking_density',
                  'hour_sin', 'hour_cos']
binary_variables = ['weekend', 'holiday', 'covid']
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_features),
    ('cat', 'passthrough' , binary_variables)])

X_train = preprocessor.fit_transform(train_data[numeric_features + binary_variables])
X_val = preprocessor.transform(val_data[numeric_features + binary_variables])
X_test = preprocessor.transform(test_data[numeric_features + binary_variables])

y_train = train_data['charged_energy'].values
y_val = val_data['charged_energy'].values
y_test = test_data['charged_energy'].values

# Sequence data generator
SEQ_LENGTH = 24
PREDICT_HORIZON = 24

def create_sequences_cnn_lstm(X, y, seq_len, pred_len):
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_len - pred_len + 1):
        x_seq = X[i:i + seq_len]
        y_seq_step = y[i + seq_len:i + seq_len + pred_len]
        if y_seq_step.shape[0] == pred_len:
            X_seq.append(x_seq)
            y_seq.append(y_seq_step)
    
    # Convert lists to numpy arrays after the loop
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    return X_seq, y_seq

# Create sequences for CNN-LSTM model
X_train_seq, y_train_seq = create_sequences_cnn_lstm(X_train, y_train, SEQ_LENGTH, PREDICT_HORIZON)
X_val_seq, y_val_seq = create_sequences_cnn_lstm(X_val, y_val, SEQ_LENGTH, PREDICT_HORIZON)
X_test_seq, y_test_seq = create_sequences_cnn_lstm(X_test, y_test, SEQ_LENGTH, PREDICT_HORIZON)

# Now align the categorical features for each sequence
X_train_day = train_data['day_of_week_encoded'].values[SEQ_LENGTH:SEQ_LENGTH + len(X_train_seq)].reshape(-1, 1)
X_val_day = val_data['day_of_week_encoded'].values[SEQ_LENGTH:SEQ_LENGTH + len(X_val_seq)].reshape(-1, 1)
X_test_day = test_data['day_of_week_encoded'].values[SEQ_LENGTH:SEQ_LENGTH + len(X_test_seq)].reshape(-1, 1)

X_train_slot = train_data['time_slot_encoded'].values[SEQ_LENGTH:SEQ_LENGTH + len(X_train_seq)].reshape(-1, 1)
X_val_slot = val_data['time_slot_encoded'].values[SEQ_LENGTH:SEQ_LENGTH + len(X_val_seq)].reshape(-1, 1)
X_test_slot = test_data['time_slot_encoded'].values[SEQ_LENGTH:SEQ_LENGTH + len(X_test_seq)].reshape(-1, 1)

X_train_land = train_data['land_types_encoded'].values[SEQ_LENGTH:SEQ_LENGTH + len(X_train_seq)].reshape(-1, 1)
X_val_land = val_data['land_types_encoded'].values[SEQ_LENGTH:SEQ_LENGTH + len(X_val_seq)].reshape(-1, 1)
X_test_land = test_data['land_types_encoded'].values[SEQ_LENGTH:SEQ_LENGTH + len(X_test_seq)].reshape(-1, 1)

# Reshaping sequences for CNN-LSTM (add channels dimension)
X_train_seq = X_train_seq.reshape(X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2], 1)  # Adding channels dimension
X_val_seq = X_val_seq.reshape(X_val_seq.shape[0], X_val_seq.shape[1], X_val_seq.shape[2], 1)
X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2], 1)

best_params = {
    'filters': 224,
    'kernel_size': 5,
    'lstm_units': 96,
    'dropout': 0.2,
    'learning_rate': 0.0005144265260054973,
    'optimizer': 'rmsprop'}


# Optimizer selection
if best_params['optimizer'] == "adam":
    optimizer = Adam(learning_rate=best_params['learning_rate'])
elif best_params['optimizer'] == "sgd":
    optimizer = SGD(learning_rate=best_params['learning_rate'])
else:
    optimizer = RMSprop(learning_rate=best_params['learning_rate'])

  
# %%
with tf.device('/GPU:0'):
    sequence_input = Input(shape=(SEQ_LENGTH, X_train_seq.shape[2]), name='sequence_input')
    day_input = Input(shape=(1,), name='day_of_week_input')
    slot_input = Input(shape=(1,), name='time_slot_input')
    land_input = Input(shape=(1,), name='land_type_input')

    # Embedding layers for categorical features
    day_emb = Flatten()(Embedding(input_dim=7, output_dim=4)(day_input))
    slot_emb = Flatten()(Embedding(input_dim=6, output_dim=4)(slot_input))
    land_emb = Flatten()(Embedding(input_dim=11, output_dim=5)(land_input))
    combined_emb = Concatenate()([day_emb, slot_emb, land_emb])

    # CNN layers
    x = Conv1D(filters=best_params['filters'] // 2, kernel_size=best_params['kernel_size'], activation='relu', padding='same')(sequence_input)
    x = BatchNormalization()(x)
    x = Conv1D(filters=best_params['filters'], kernel_size=best_params['kernel_size'], activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
   #x = MaxPooling1D(pool_size=1)(x)
    x = LSTM(best_params['lstm_units'] // 2, kernel_regularizer=l2(0.00001))(x)
    x = Dropout(best_params['dropout'])(x)

    # Combine with embedding output
    combined = Concatenate()([x, combined_emb])
    output = Dense(PREDICT_HORIZON, kernel_regularizer=l2(0.00001))(combined)

    model = Model(inputs={'sequence_input': sequence_input,
                          'day_of_week_input': day_input,
                          'time_slot_input': slot_input,
                          'land_type_input': land_input}, 
                  outputs=output)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
    reduce_lp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(x={'sequence_input': X_train_seq,
                           'day_of_week_input': X_train_day,
                           'time_slot_input': X_train_slot,
                           'land_type_input': X_train_land},
                        y=y_train_seq,
                        validation_data=({'sequence_input': X_val_seq, 
                                          'day_of_week_input': X_val_day,
                                          'time_slot_input': X_val_slot,
                                          'land_type_input': X_val_land},
                                         y_val_seq),
                        epochs=30,
                        batch_size=128,
                        callbacks=[early_stop, reduce_lp], verbose=0)


#  %%
y_pred = model.predict({'sequence_input': X_test_seq,
                        'day_of_week_input': X_test_day,
                        'time_slot_input': X_test_slot,
                        'land_type_input': X_test_land})

y_test_flat = y_test_seq.flatten()
y_pred_flat = y_pred.flatten()

mae = mean_absolute_error(y_test_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
r2 = r2_score(y_test_flat, y_pred_flat)
print(f'CNN-LSTM Daily Prediction: \n MAE: {mae} \n RMSE: {rmse} \n R-squared: {r2}')
np.savetxt("hybrid_daily.csv", y_pred_flat, delimiter=",")

# %%
test_data.index = pd.to_datetime(test_data.index)

# Use the same indexing method
start_idx = test_data.index[SEQ_LENGTH + PREDICT_HORIZON - 1:]
index = np.tile(start_idx.values, PREDICT_HORIZON)
index = index[:len(y_pred_flat)]

# Create Series with datetime index
y_pred_series = pd.Series(y_pred_flat, index=index)
y_true_series = pd.Series(y_test_flat, index=index)

# Group predictions and actuals by day (summing over 24h)
daily_pred = y_pred_series.groupby(y_pred_series.index.date).mean()
daily_true = y_true_series.groupby(y_true_series.index.date).mean()

# Calculate daily MAE
daily_mae = (daily_pred - daily_true).abs()

num_days = min(35, len(daily_true))
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

axes[0].plot(daily_true.index[:num_days], daily_true[:num_days], label='Actual', marker='o', color='blue')
axes[0].plot(daily_pred.index[:num_days], daily_pred[:num_days], label='Predicted', marker='x', color='orange')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Energy Consumption (kWh)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(daily_mae.index[:num_days], daily_mae.values[:num_days], label='MAE', marker='o', color='purple')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Error Score')
axes[1].grid(True)
axes[1].legend()
plt.tight_layout()
plt.savefig('cnn_daily_prediction.jpg', dpi=300)
plt.show()

