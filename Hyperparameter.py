# %% Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dropout, Dense, Embedding, Flatten, RepeatVector, Concatenate
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
import optuna

# Device configuration
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# %% Load and preprocess data
df = pd.read_csv('feature_data.csv')
df["date"] = pd.to_datetime(df["date"])
df.set_index('date', inplace=True)

X = df.drop(columns=['charged_energy', 'duration'])
y = df['charged_energy']

label_encoder = LabelEncoder()
X['land_types'] = label_encoder.fit_transform(X['land_types'])
X_land = np.array(X['land_types'], dtype=np.int32)
X = X.drop(columns=['land_types'])

numeric_features = ['temperature', 'year', 'dewpoint', 'road_density', 'commercial_density',
                    'residential_density', 'recreation_density', 'highway_proximity',
                    'public_transport_proximity', 'evcs_proximity', 'center_proximity',
                    'parking_density', 'hour_sin', 'hour_cos']
multiple = ['land_types']
binary = list(set(X.columns) - set(numeric_features) - set(multiple))

# Preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', MinMaxScaler(), numeric_features),
    ('cat', OneHotEncoder(), binary)])

# Convert to DataFrames for indexing
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(X))


# Spatiotemporal LSTM
def objective(trial):
    lstm_units = trial.suggest_int("lstm_units", 16, 128, step=16)
    dropout_rate = trial.suggest_categorical("dropout", [0.2, 0.3, 0.5])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    two_layers = trial.suggest_categorical('two_layers', [True, False])

    if optimizer_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    val_losses = []

    for train_index, val_index in splits:
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        land_train_fold, land_val_fold = X_land[train_index], X_land[val_index]

        X_train_proc = preprocessor.fit_transform(X_train_fold, y_train_fold)
        X_val_proc = preprocessor.transform(X_val_fold)

        X_train_seq = X_train_proc.reshape((X_train_proc.shape[0], 1, X_train_proc.shape[1]))
        X_val_seq = X_val_proc.reshape((X_val_proc.shape[0], 1, X_val_proc.shape[1]))

        X_train_seq = np.array(X_train_seq, dtype=np.float32)
        X_val_seq = np.array(X_val_seq, dtype=np.float32)
        y_train_fold = np.array(y_train_fold, dtype=np.float32)
        y_val_fold = np.array(y_val_fold, dtype=np.float32)

        # Model with embedding
        with tf.device(device):
            # Inputs
            seq_input = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]), name='seq_input')
            land_input = Input(shape=(1,), dtype='int32', name='land_input')

            # Embedding
            embedding = Embedding(input_dim=np.max(X_land) + 1, output_dim=5)(land_input)
            embedding_flat = Flatten()(embedding)
            embedding_repeat = RepeatVector(X_train_seq.shape[1])(embedding_flat)

            # Merge embedding with input
            merged = Concatenate()([seq_input, embedding_repeat])

            # LSTM layers
            x = LSTM(lstm_units // 2, activation='relu', return_sequences=two_layers)(merged)
            x = Dropout(dropout_rate)(x)
            if two_layers:
                x = LSTM(lstm_units, activation='relu')(x)
                x = Dropout(dropout_rate)(x)

            output = Dense(1)(x)

            model = Model(inputs=[seq_input, land_input], outputs=output)
            model.compile(loss='mean_absolute_error', optimizer=optimizer)

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit([X_train_seq, land_train_fold], y_train_fold,
                      validation_data=([X_val_seq, land_val_fold], y_val_fold),
                      epochs=30, batch_size=32, verbose=0,
                      callbacks=[early_stopping])

            val_loss = model.evaluate([X_val_seq, land_val_fold], y_val_fold, verbose=0)
            val_losses.append(val_loss)

    return np.mean(val_losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best parameters:", study.best_params)


# Spatiotemporal CNN-LSTM
def objective(trial):
    filters = trial.suggest_categorical('filters', [16, 32, 64, 96, 128])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    lstm_units = trial.suggest_int('lstm_units', 16, 128, step=16)
    dropout_rate = trial.suggest_categorical('dropout', [0.2, 0.3, 0.5])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])

    if optimizer_choice == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == "sgd":
        optimizer = SGD(learning_rate=learning_rate)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    val_losses = []

    for train_index, val_index in splits:
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        land_train_fold, land_val_fold = X_land[train_index], X_land[val_index]

        X_train_proc = preprocessor.fit_transform(X_train_fold, y_train_fold)
        X_val_proc = preprocessor.transform(X_val_fold)

        X_train_seq = X_train_proc.reshape((X_train_proc.shape[0], X_train_proc.shape[1], 1))
        X_val_seq = X_val_proc.reshape((X_val_proc.shape[0], X_val_proc.shape[1], 1))

        X_train_seq = np.array(X_train_seq, dtype=np.float32)
        X_val_seq = np.array(X_val_seq, dtype=np.float32)
        y_train_fold = np.array(y_train_fold, dtype=np.float32)
        y_val_fold = np.array(y_val_fold, dtype=np.float32)

        # CNN-LSTM with embedding
        with tf.device(device):
            seq_input = Input(shape=(X_train_seq.shape[1], 1), name='seq_input')
            land_input = Input(shape=(1,), dtype='int32', name='land_input')

            emb = Embedding(input_dim=np.max(X_land)+1, output_dim=5)(land_input)
            emb = Flatten()(emb)
            emb = RepeatVector(X_train_seq.shape[1])(emb)

            x = Concatenate()([seq_input, emb])
            x = Conv1D(filters=filters//2, kernel_size=kernel_size, activation='relu', padding='same')(x)
            x = Dropout(dropout_rate)(x)
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
            x = Dropout(dropout_rate)(x)
            x = LSTM(units=lstm_units, activation='relu')(x)
            output = Dense(1)(x)

            model = Model(inputs=[seq_input, land_input], outputs=output)
            model.compile(loss='mae', optimizer=optimizer)

            early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

            model.fit([X_train_seq, land_train_fold], y_train_fold,
                      validation_data=([X_val_seq, land_val_fold], y_val_fold),
                      epochs=30, batch_size=32, callbacks=[early_stopping], verbose=0)

            val_loss = model.evaluate([X_val_seq, land_val_fold], y_val_fold, verbose=0)
            val_losses.append(val_loss)

    return np.mean(val_losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best parameters:", study.best_params)


# Temporal LSTM
X = df[['duration', 'temperature', 'dewpoint', 'land_types', 'year', 'month',
       'day_of_week', 'weekend', 'holiday', 'lockdown', 'covid',
       'energy_movingavg', 'energy_movingstd', 'hour_sin', 'hour_cos']]
y = df['charged_energy']

train_size = int(len(X) * 0.8)  
val_size = int(len(X) * 0.1)   
test_size = len(X) - train_size - val_size 

X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

numeric_features = ['duration', 'temperature', 'dewpoint',
                    'energy_movingavg', 'energy_movingstd', 'hour_sin', 'hour_cos']
categorical_features = list(set(X.columns) - set(numeric_features))

preprocessor = ColumnTransformer(transformers=[('num', MinMaxScaler(), numeric_features), 
                                               ('cat', 'passthrough', categorical_features)])

X_train = preprocessor.fit_transform(X_train, y_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)


X_train_series = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_series = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_series = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

X_train_series = np.array(X_train_series, dtype=np.float32)
X_val_series = np.array(X_val_series, dtype=np.float32)
X_test_series = np.array(X_test_series, dtype=np.float32)

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

def objective(trial):
    lstm_units = trial.suggest_int("lstm_units", 16, 128, step=16) 
    dropout_rate = trial.suggest_categorical("dropout", [0.2, 0.3, 0.5])  
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])  
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)  
    two_layers = trial.suggest_categorical('two_layers', [True, False])

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
        history = model.fit(
            X_train_series, y_train,
            validation_data=(X_val_series, y_val),
            epochs=20, batch_size=128,
            verbose=0,callbacks=[early_stopping])

    return history.history['val_loss'][-1]


study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 
print("Best parameters:", study.best_params)

# Temporal CNN-LSTM
X_train_series = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_series = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_series = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

X_train_series = np.array(X_train_series, dtype=np.float32)
X_val_series = np.array(X_val_series, dtype=np.float32)
X_test_series = np.array(X_test_series, dtype=np.float32)

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)

def objective(trial):
    filters = trial.suggest_categorical('filters', [16, 32, 64, 96, 128])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    lstm_units = trial.suggest_int('lstm_units', 16, 96, step=16)
    dropout_rate = trial.suggest_categorical('dropout', [0.2, 0.3, 0.5])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizers = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    
    if optimizers == "adam":
        optimizers = Adam(learning_rate=learning_rate)
    elif optimizers == "sgd":
        optimizers = SGD(learning_rate=learning_rate)
    else:
        optimizers = RMSprop(learning_rate=learning_rate)
    
    with tf.device(device):
        model = Sequential()
        model.add(Input(shape=(X_train_series.shape[1], X_train_series.shape[2])))  
        model.add(Conv1D(filters=filters//2, kernel_size=kernel_size, activation='relu', padding='same'))  
        model.add(Dropout(rate=dropout_rate))
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))  
        model.add(Dropout(rate=dropout_rate))
        model.add(LSTM(units=lstm_units, activation='relu', return_sequences=False)) 
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        
        model.compile(loss='mae', optimizer=optimizers)
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        history = model.fit(X_train_series, y_train,
            validation_data=(X_val_series, y_val),
            epochs=30, batch_size=128, 
            callbacks=[early_stopping],verbose=0)

    return history.history['val_loss'][-1]

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=50) 
print("Best parameters:", study.best_params)
