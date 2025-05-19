import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, Embedding, Flatten, RepeatVector, Concatenate
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# Load data
df = pd.read_csv('feature_data.csv')
df["date"] = pd.to_datetime(df["date"])
df.set_index('date', inplace=True)

X = df.drop(columns=['charged_energy', 'duration'])
y = df['charged_energy']

# Encode land_types for embedding
label_encoder = LabelEncoder()
X['land_types'] = label_encoder.fit_transform(X['land_types'])
X_land = np.array(X['land_types'], dtype=np.int32)
X = X.drop(columns=['land_types'])

numeric_features = ['temperature', 'year', 'dewpoint', 'road_density', 'commercial_density',
                    'residential_density', 'recreation_density', 'highway_proximity',
                    'public_transport_proximity', 'evcs_proximity', 'center_proximity',
                    'parking_density', 'hour_sin', 'hour_cos']
binary = list(set(X.columns) - set(numeric_features))

preprocessor = ColumnTransformer(transformers=[
    ('num', MinMaxScaler(), numeric_features),
    ('cat', OneHotEncoder(), binary)
])

train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

land_train = X_land[:train_size]
land_val = X_land[train_size:train_size + val_size]
land_test = X_land[train_size + val_size:]

X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)

# LSTM input reshaping
X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_seq = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

X_train_seq = np.array(X_train_seq, dtype=np.float32)
X_val_seq = np.array(X_val_seq, dtype=np.float32)
X_test_seq = np.array(X_test_seq, dtype=np.float32)

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

best_params = {
    'lstm_units': 128,
    'dropout': 0.3,
    'optimizer': 'rmsprop',
    'learning_rate': 0.00010791676136221328,
    'two_layers': True}

lstm_units = best_params["lstm_units"]
dropout_rate = best_params["dropout"]
optimizer_name = best_params["optimizer"]
learning_rate = best_params["learning_rate"]

if optimizer_name == "adam":
    optimizer = Adam(learning_rate=learning_rate)
elif optimizer_name == "sgd":
    optimizer = SGD(learning_rate=learning_rate)
else:
    optimizer = RMSprop(learning_rate=learning_rate)

with tf.device("/CPU:0"):  
    seq_input = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]), name='seq_input')
    land_input = Input(shape=(1,), dtype='int32', name='land_input')

    emb = Embedding(input_dim=np.max(X_land)+1, output_dim=5)(land_input)
    emb = Flatten()(emb)
    emb = RepeatVector(X_train_seq.shape[1])(emb)

    x = Concatenate()([seq_input, emb])
    x = LSTM(units=lstm_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1)(x)

    model = Model(inputs=[seq_input, land_input], outputs=output)
    model.compile(optimizer=optimizer, loss='mae')
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit([X_train_seq, land_train], y_train,
              validation_data=([X_val_seq, land_val], y_val),
              epochs=30, batch_size=32, callbacks=[early_stopping], verbose=1)

y_pred = model.predict([X_test_seq, land_test], verbose=1)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'LSTM Embedding Test:\n MAE: {mae:.4f} \n RMSE: {rmse:.4f} \n R-squared: {r2:.4f}')
np.savetxt("lstm_spatiotemporal.csv", y_pred, delimiter=",")

#CNN-LSTM
X_train_seq = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_seq = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_seq = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Land type sequences (embedding input)
land_train = X_land[:len(X_train_seq)]
land_val = X_land[len(X_train_seq):len(X_train_seq) + len(X_val_seq)]
land_test = X_land[len(X_train_seq) + len(X_val_seq):]

# Ensure correct data type
X_train_seq = np.array(X_train_seq, dtype=np.float32)
X_val_seq = np.array(X_val_seq, dtype=np.float32)
X_test_seq = np.array(X_test_seq, dtype=np.float32)

land_train = np.array(land_train, dtype=np.int32)
land_val = np.array(land_val, dtype=np.int32)
land_test = np.array(land_test, dtype=np.int32)

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

best_params = {
    'filters': 128,
    'kernel_size': 5,
    'lstm_units': 32,
    'dropout': 0.2,
    'learning_rate': 0.0005851446457315788,
    'optimizer': 'rmsprop'}

# Optimizer selection
if best_params['optimizer'] == "adam":
    optimizer = Adam(learning_rate=best_params['learning_rate'])
elif best_params['optimizer'] == "sgd":
    optimizer = SGD(learning_rate=best_params['learning_rate'])
else:
    optimizer = RMSprop(learning_rate=best_params['learning_rate'])

filters = best_params["filters"]
kernel_size = best_params["kernel_size"]
lstm_units = best_params["lstm_units"]
dropout_rate = best_params["dropout"]
optimizer_name = best_params["optimizer"]
learning_rate = best_params["learning_rate"]

with tf.device("/CPU:0"): 
    seq_input = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]), name='seq_input')
    land_input = Input(shape=(1,), dtype='int32', name='land_input')

    emb = Embedding(input_dim=np.max(X_land)+1, output_dim=5)(land_input)
    emb = Flatten()(emb)
    emb = RepeatVector(X_train_seq.shape[1])(emb)
    x = Concatenate()([seq_input, emb])
    
    # CNN Layers
    x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling1D(pool_size=1)(x)  
    
    x = LSTM(units=lstm_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(1)(x)

    model = Model(inputs=[seq_input, land_input], outputs=output)
    model.compile(optimizer=optimizer, loss='mae')
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit([X_train_seq, land_train], y_train,
              validation_data=([X_val_seq, land_val], y_val),
              epochs=30, batch_size=32, callbacks=[early_stopping], verbose=1)

y_pred = model.predict([X_test_seq, land_test], verbose=1)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'CNN-LSTM Embedding Test:\n MAE: {mae:.4f} \n RMSE: {rmse:.4f} \n R-squared: {r2:.4f}')
np.savetxt("hybrid_spatiotemporal.csv", y_pred, delimiter=",")

# Feature Importance analysis 
## LSTM
def get_all_feature_names(preprocessor):
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            try:
                names = transformer.get_feature_names_out(columns)
            except TypeError:
                names = transformer.get_feature_names_out()
            feature_names.extend(names)
        else:
            feature_names.extend(columns)
    return feature_names

def evaluate_model(model, X_test_series, y_test):
    y_pred = model.predict(X_test_series, batch_size=32)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2

baseline_mae, baseline_rmse, baseline_r2 = evaluate_model(model, X_test_seq, y_val)

feature_names = get_all_feature_names(preprocessor)
importances = []
for i in range(X_test_seq.shape[2]):  
    X_test_permuted = X_test_seq.copy()
    for t in range(X_test_permuted.shape[1]):
        X_test_permuted[:, t, i] = np.random.permutation(X_test_permuted[:, t, i])
    mae = mean_absolute_error(y_val, model.predict(X_test_permuted, batch_size=32))
    importances.append(mae - baseline_mae)

importances = np.array(importances) 
importances_percentage = importances / importances.sum() * 100  
sorted_idx = np.argsort(importances_percentage)
top_10_idx = sorted_idx[-10:]

plt.figure(figsize=(10, 6))
plt.barh(np.array(feature_names)[top_10_idx], importances_percentage[top_10_idx])
plt.xlabel("Permutation Importance (%)")
plt.tight_layout()
plt.savefig('feature_less_importance_cnn.jpg', dpi=600)
plt.show()


# %%  CNN-LSTM Feature Permutation
y_pred = model.predict(X_test_series, batch_size=32)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f'Test set: \n MAE: {mae} \n RMSE: {rmse} \n R-squared: {r2}')

# Baseline MAE with unshuffled data
y_pred_baseline = model.predict(X_test_seq)
baseline_mae = mean_absolute_error(y_test, y_pred_baseline)

# Set up for feature importance calculation
n_features = X_test_seq.shape[1]  # number of time steps
importance_scores = []

# Loop through each feature (time step) and shuffle
for i in range(n_features):
    X_permuted = X_test_seq.copy()
    X_permuted[:, i, 0] = np.random.permutation(X_permuted[:, i, 0])
    y_pred_permuted = model.predict(X_permuted)
    permuted_mae = mean_absolute_error(y_test, y_pred_permuted)
    
    importance = permuted_mae - baseline_mae
    importance_scores.append(importance)

importance_scores = np.array(importance_scores)

# Get top 10 important features
top_indices = np.argsort(importance_scores)[-10:][::-1]  
top_scores = importance_scores[top_indices]
all_feature_names = get_all_feature_names(preprocessor)
top_feature_names = [all_feature_names[i] for i in top_indices]

total_importance = np.sum(importance_scores)
importance_percentages = (importance_scores / total_importance) * 100

top_percentages = importance_percentages[top_indices]

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(range(len(top_percentages)), top_percentages)
plt.yticks(range(len(top_percentages)), top_feature_names)  
plt.xlabel("Permutation Importance (%)")
plt.gca().invert_yaxis()  
plt.tight_layout()
plt.show()
