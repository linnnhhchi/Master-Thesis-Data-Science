import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Embedding, Concatenate, Flatten, RepeatVector
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

K.clear_session()

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        # Restrict TensorFlow to only use GPU:0
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using only GPU: {gpus[0].name}")
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

# Load and preprocess the data
df = pd.read_csv('feature_data.csv')
df = df.sort_values(['date'])
df.set_index('date', inplace=True)

# Categorical encoding
encoder = LabelEncoder()
df['day_of_week_encoded'] = encoder.fit_transform(df['day_of_week'])
df['time_slot_encoded'] = encoder.fit_transform(df['time_slot'])
df['land_types_encoded'] = encoder.fit_transform(df['land_types'])

# Train-test split
split_index = int(len(df) * 0.8)
train_data = df.iloc[:split_index]
test_data = df.iloc[split_index:]

# Validation split
val_split = int(len(train_data) * 0.2)
val_data = train_data[-val_split:]
train_data = train_data[:-val_split]

numeric_features = ['duration', 'temperature', 'year', 'dewpoint', 'road_density', 'commercial_density',
                   'residential_density', 'recreation_density', 'highway_proximity', 
                    'public_transport_proximity', 'evcs_proximity', 'center_proximity', 'parking_density',
                  'hour_sin', 'hour_cos']

binary_variables = ['weekend', 'holiday', 'covid']

binary_variables = ['weekend']

preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), numeric_features),
    ('cat', 'passthrough' , binary_variables)])

X_train = preprocessor.fit_transform(train_data[numeric_features + binary_variables])
X_val = preprocessor.transform(val_data[numeric_features + binary_variables])
X_test = preprocessor.transform(test_data[numeric_features + binary_variables])

y_train = train_data['charged_energy'].values
y_val = val_data['charged_energy'].values
y_test = test_data['charged_energy'].values

# Sequence data generation
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

# Define model architecture
best_params = {
    'lstm_units': 112,
    'dropout': 0.2,
    'optimizer': 'rmsprop',
    'learning_rate': 0.00010791676136221328,
    'two_layers': True}

# Optimizer selection
if best_params['optimizer'] == "adam":
    optimizer = Adam(learning_rate=best_params['learning_rate'])
elif best_params['optimizer'] == "sgd":
    optimizer = SGD(learning_rate=best_params['learning_rate'])
else:
    optimizer = RMSprop(learning_rate=best_params['learning_rate'])

# LSTM Model
with tf.device('/GPU:0'):
    sequence_input = Input(shape=(SEQ_LENGTH, X_train_seq.shape[2]), name='sequence_input')
    day_input = Input(shape=(1,), name='day_of_week_input')
    slot_input = Input(shape=(1,), name='time_slot_input')
    land_input = Input(shape=(1,), name='land_type_input')

    # Embedding layers for categorical features
    day_emb = Flatten()(Embedding(input_dim=7, output_dim=4)(day_input))
    slot_emb = Flatten()(Embedding(input_dim=6, output_dim=3)(slot_input))
    land_emb = Flatten()(Embedding(input_dim=11, output_dim=5)(land_input))
    combined_emb = Concatenate()([day_emb, slot_emb, land_emb])

    # LSTM layer
    x = LSTM(best_params['lstm_units'] // 2, activation='relu', return_sequences = best_params['two_layers'])(sequence_input)
    x = Dropout(best_params['dropout'])(x)

    if best_params['two_layers']:
        x = LSTM(best_params['lstm_units'], activation='relu')(x)
        x = Dropout(best_params['dropout'])(x)

    combined = Concatenate()([x, combined_emb])
    output = Dense(PREDICT_HORIZON)(combined)

    model = Model(inputs={'sequence_input': sequence_input,
                          'day_of_week_input': day_input,
                          'time_slot_input': slot_input,
                          'land_type_input': land_input}, 
                  outputs=output)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Early stopping to avoid overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    # Training the model
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
                        batch_size=32,
                        callbacks=[early_stop], verbose=0)

# Predict on test data
y_pred = model.predict({'sequence_input': X_test_seq,
                        'day_of_week_input': X_test_day,
                        'time_slot_input': X_test_slot,
                        'land_type_input': X_test_land})

y_pred_flat = y_pred.flatten()
y_test_seq_flat = y_test_seq.flatten() 

if len(y_test_seq_flat) != len(y_pred_flat):
    y_pred_flat = y_pred_flat[:len(y_test_seq_flat)]

mae = mean_absolute_error(y_test_seq_flat, y_pred_flat)
rmse = np.sqrt(mean_squared_error(y_test_seq_flat, y_pred_flat))
r2 = r2_score(y_test_seq_flat, y_pred_flat)
print(f'LSTM Daily Test: \n MAE: {mae} \n RMSE: {rmse} \n R-squared: {r2}')
np.savetxt("lstm_daily.csv", y_pred_flat, delimiter=",")


# %% Plotting
test_data.index = pd.to_datetime(test_data.index)

# Use the same indexing method
start_idx = test_data.index[SEQ_LENGTH + PREDICT_HORIZON - 1:]
index = np.tile(start_idx.values, PREDICT_HORIZON)
index = index[:len(y_pred_flat)]

# Create Series with datetime index
y_pred_series = pd.Series(y_pred_flat, index=index)
y_true_series = pd.Series(y_test_seq_flat, index=index)

# Group predictions and actuals by day (summing over 24h)
daily_pred = y_pred_series.groupby(y_pred_series.index.date).mean()
daily_true = y_true_series.groupby(y_true_series.index.date).mean()

# Calculate daily MAE
daily_mae = (daily_pred - daily_true).abs()

num_days = min(35, len(daily_true))
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

axes[0].plot(daily_true.index[:num_days], daily_true[:num_days], label='Actual', marker='o', color='blue')
axes[0].plot(daily_pred.index[:num_days], daily_pred[:num_days], label='Predicted', marker='x', color='orange')
axes[0].set_ylabel('Energy Consumption (kWh)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(daily_mae.index[:num_days], daily_mae.values[:num_days], label='MAE',  marker='o', color='purple')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Error Score')
axes[1].grid(True)
axes[1].legend()
plt.tight_layout()
plt.savefig('lstm_daily_prediction_subplot.jpg', dpi=300)
plt.show()
