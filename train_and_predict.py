import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas_ta as ta
from tensorflow.keras.layers import Bidirectional


def create_indicators(data):
    # Convert datetime column to pandas datetime format
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Add technical indicators to the data
    data['SMA'] = ta.sma(data['Close'], length=14)  # Simple Moving Average
    data['EMA'] = ta.ema(data['Close'], length=14)  # Exponential Moving Average
    data['RSI'] = ta.rsi(data['Close'], length=14)  # Relative Strength Index
    data['MACD'] = ta.macd(data['Close']).iloc[:,1] # Moving Average Convergence Divergence

    bb = ta.bbands(data['Close'])
    data['BBL'] = bb.iloc[:,0]
    data['BBM'] = bb.iloc[:,1]
    data['BBU'] = bb.iloc[:,2]
    data['BB_BANDWIDTH'] = bb.iloc[:,3]

    # Add additional moving averages for different time horizons
    data['SMA_10'] = ta.sma(data['Close'], length=10)  # 10-minute SMA
    data['SMA_50'] = ta.sma(data['Close'], length=50)  # 50-minute SMA
    data['SMA_200'] = ta.sma(data['Close'], length=200)  # 200-minute SMA
    # data['SMA_1000'] = ta.sma(data['Close'], length=1000)  # 200-minute SMA

    data['EMA_10'] = ta.ema(data['Close'], length=10)  # 10-minute EMA
    data['EMA_50'] = ta.ema(data['Close'], length=50)  # 50-minute EMA
    data['EMA_200'] = ta.ema(data['Close'], length=200)  # 200-minute EMA
    # data['EMA_1000'] = ta.ema(data['Close'], length=1000)  # 200-minute EMA

    # Add differences between moving averages to the data
    data['SMA_10_50_diff'] = data['SMA_10'] - data['SMA_50']
    data['SMA_10_200_diff'] = data['SMA_10'] - data['SMA_200']
    data['SMA_50_200_diff'] = data['SMA_50'] - data['SMA_200']
    # data['SMA_10_1000_diff'] = data['SMA_10'] - data['SMA_1000']
    # data['SMA_50_1000_diff'] = data['SMA_50'] - data['SMA_1000']
    # data['SMA_200_1000_diff'] = data['SMA_200'] - data['SMA_1000']

    data['EMA_10_50_diff'] = data['EMA_10'] - data['EMA_50']
    data['EMA_10_200_diff'] = data['EMA_10'] - data['EMA_200']
    data['EMA_50_200_diff'] = data['EMA_50'] - data['EMA_200']
    # data['EMA_10_1000_diff'] = data['EMA_10'] - data['EMA_1000']
    # data['EMA_50_1000_diff'] = data['EMA_50'] - data['EMA_1000']
    # data['EMA_200_1000_diff'] = data['EMA_200'] - data['EMA_1000']

    # Calculate log returns instead of simple returns
    data['return_1'] = np.log(data['Close'] / data['Close'].shift(5))
    data['return_2'] = np.log(data['Close'] / data['Close'].shift(20))
    data['return_3'] = np.log(data['Close'] / data['Close'].shift(100))

    data['Close_raw'] = data['Close']  # Save unmodified prices

    # Drop rows with NaN values introduced by indicators
    data = data.dropna() 

    return data

def train_and_predict_lstm(data, 
                          feature_columns,
                          test_size=0.2,
                          sequence_length=1000,
                          threshold=0.03,
                          lower_threshold=0.0,
                          epochs=1,
                          batch_size=64,
                          learning_rate=0.001,
                          lstm_units=64,
                          dense_units=32,
                          dropout_rate=0.3):
    """
    Train an LSTM model on a subset of data and make predictions on the test set.
    After finding optimal threshold, retrains on full dataset for production use.
    
    Args:
        data: DataFrame with features and labels
        test_size: Fraction of data to use for testing (between 0 and 1)
        sequence_length: Length of input sequences
        threshold: Profit threshold for label creation
        lower_threshold: Lower threshold for label creation
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        lstm_units: Number of LSTM units
        dense_units: Number of dense units
        dropout_rate: Dropout rate
    
    Returns:
        dict: Contains predictions, probabilities, model, scaler, and performance metrics
    """
    
    # Define feature columns
    # feature_columns = [
    #     'Open', 'High', 'Low', 'Close', 'Volume',
    #     'MACD', 'BBL', 'BBM', 'BBU', 'BB_BANDWIDTH',
    #     'SMA', 'EMA', 'RSI',
    #     'SMA_10', 'SMA_50', 'SMA_200', 'EMA_10', 'EMA_50', 'EMA_200',
    #     'SMA_10_50_diff', 'SMA_10_200_diff', 'SMA_50_200_diff',
    #     'EMA_10_50_diff', 'EMA_10_200_diff', 'EMA_50_200_diff',
    #     'return_1', 'return_2', 'return_3'
    # ]
    
    # Create labels for entire dataset first (before splitting)
    def add_first_hit_labels(df, threshold=0.02, max_lookahead=60, lower_threshold=0.01):
        labels = []
        sell_price = df['Close_raw'].values * (1 - 0.002/2)
        buy_price = df['Close_raw'].values * (1 + 0.002/2)
        
        for i in range(len(df)):
            entry_price = buy_price[i]
            label = np.nan
            for j in range(1, max_lookahead + 1):
                if i + j >= len(sell_price):
                    break
                future_return = (sell_price[i + j] - entry_price) / entry_price
                if future_return >= threshold:
                    label = 1
                    break
                elif future_return <= -(threshold - lower_threshold):
                    label = 0
                    break
            labels.append(label)
        
        # for i in range(len(df)):
        #     entry_price = buy_price[i]
        #     label = np.nan
        #     for j in range(1, max_lookahead + 1):
        #         if i + j >= len(sell_price):
        #             future_return = (sell_price[i + j -1] - entry_price) / entry_price
        #             if future_return > 0:
        #                 label = 1
        #                 break
        #             else:
        #                 label = 0
        #                 break
        #         future_return = (sell_price[i + j] - entry_price) / entry_price
        #         if future_return >= threshold:
        #             label = 1
        #             break
        #         elif future_return <= -(threshold - lower_threshold):
        #             label = 0
        #             break
        #     labels.append(label)

        df[f'label_first_hit_{int(threshold*100)}'] = labels
        return df
    
    # Apply labels to entire dataset
    label_col = f'label_first_hit_{int(threshold*100)}'
    # data1 = data.iloc[:int(len(data)*(1-test_size)),:]
    # data1 = add_first_hit_labels(data1, threshold=threshold, max_lookahead=len(data1), lower_threshold=lower_threshold)
    # data2 = data.iloc[int(len(data)*(1-test_size)):,:]
    # data2 = add_first_hit_labels(data2, threshold=threshold, max_lookahead=len(data2), lower_threshold=lower_threshold)
    # data = pd.concat([data1, data2])
    data = add_first_hit_labels(data, threshold=threshold, max_lookahead=len(data), lower_threshold=lower_threshold)
    # Drop rows where label couldn't be assigned (before splitting)
    data = data.dropna(subset=[label_col])
    
    # Now split the cleaned data chronologically
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    # Create sequences for training
    X_train, y_train = [], []
    for i in range(len(train_data) - sequence_length + 1):
        X_train.append(train_data[feature_columns].iloc[i:i+sequence_length].values)
        y_train.append(train_data[label_col].iloc[i + sequence_length - 1])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train).astype(int)
    
    # Create sequences for testing
    X_test, y_test = [], []
    for i in range(len(test_data) - sequence_length + 1):
        X_test.append(test_data[feature_columns].iloc[i:i+sequence_length].values)
        y_test.append(test_data[label_col].iloc[i + sequence_length - 1])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test).astype(int)
    
    # Scale the data
    scaler = StandardScaler()
    
    # Flatten for scaling
    n_train, seq_len, n_feat = X_train.shape
    n_test = X_test.shape[0]
    
    # Reshape to 2D (samples * timesteps, features)
    X_train_2d = X_train.reshape(-1, n_feat)
    X_test_2d = X_test.reshape(-1, n_feat)
    
    # Fit on train, transform both
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_test_scaled = scaler.transform(X_test_2d)
    
    # Reshape back to 3D
    X_train = X_train_scaled.reshape(n_train, seq_len, n_feat)
    X_test = X_test_scaled.reshape(n_test, seq_len, n_feat)
    
    # Compute class weights
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
    
    # Build and train initial model (for threshold optimization)
    initial_model = Sequential([
        LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # # Model 2
    # initial_model = Sequential([
    #     LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    #     LSTM(32),
    #     Dense(32, activation='relu'),
    #     Dropout(0.2),
    #     Dense(1, activation='sigmoid')
    # ])

    # # Model 3
    # initial_model = Sequential([
    #     Bidirectional(LSTM(64, return_sequences=False), input_shape=(X_train.shape[1], X_train.shape[2])),
    #     Dropout(0.3),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    initial_model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the initial model
    initial_history = initial_model.fit(X_train, y_train, 
                                       validation_data=(X_test, y_test), 
                                       epochs=epochs, 
                                       batch_size=batch_size, 
                                       class_weight=class_weights,
                                       verbose=1)
    
    # Find optimal threshold using test set
    test_probs = initial_model.predict(X_test, verbose=0).flatten()
    
    from sklearn.metrics import precision_recall_curve, f1_score
    precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Find optimal threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    # Now retrain on full dataset (train + test) for production
    # Combine train and test data
    full_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    
    # Create sequences for full dataset
    X_full, y_full = [], []
    for i in range(len(full_data) - sequence_length + 1):
        X_full.append(full_data[feature_columns].iloc[i:i+sequence_length].values)
        y_full.append(full_data[label_col].iloc[i + sequence_length - 1])
    
    X_full = np.array(X_full)
    y_full = np.array(y_full).astype(int)
    
    # # Scale full dataset using the same scaler
    # n_full, seq_len, n_feat = X_full.shape
    # X_full_2d = X_full.reshape(-1, n_feat)
    # X_full_scaled = scaler.transform(X_full_2d)
    # X_full = X_full_scaled.reshape(n_full, seq_len, n_feat)
    
    # Compute class weights for full dataset
    full_class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_full),
        y=y_full
    )
    full_class_weights = {i: full_class_weights_array[i] for i in range(len(full_class_weights_array))}
    
    # # Build and train production model on full dataset
    # production_model = Sequential([
    #     LSTM(lstm_units, input_shape=(X_full.shape[1], X_full.shape[2]), return_sequences=False),
    #     Dropout(dropout_rate),
    #     Dense(dense_units, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    production_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_full.shape[1], X_full.shape[2])),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    # # Model 3
    # production_model = Sequential([
    #     Bidirectional(LSTM(64, return_sequences=False), input_shape=(X_full.shape[1], X_full.shape[2])),
    #     Dropout(0.3),
    #     Dense(32, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])
    
    production_model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train production model on full dataset
    production_history = production_model.fit(X_full, y_full, 
                                             epochs=epochs, 
                                             batch_size=batch_size, 
                                             class_weight=full_class_weights,
                                             verbose=1)
    
    # Make predictions with production model on test set for evaluation
    probs = production_model.predict(X_test, verbose=0).flatten()
    pred_labels = (probs > best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, pred_labels)
    
    # Get test data for return (without sequence_length offset)
    test_data_for_return = test_data.iloc[sequence_length:].reset_index(drop=True)
    
    
    return {
        'predictions': pred_labels,
        'probabilities': probs,
        'actual_labels': y_test,
        # 'X_test': X_test,
        'last_sequence': full_data.iloc[-sequence_length:,:],
        'last_sequence_test': test_data.iloc[-sequence_length:,:],
        'model': production_model,  # Return the production model trained on full dataset
        'scaler': scaler,
        'best_threshold': best_threshold,
        'accuracy': accuracy,
        'class_weights': full_class_weights,
        'training_history': production_history.history,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'full_samples': len(X_full),
        'feature_columns': feature_columns,
        'sequence_length': sequence_length,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'full_size': len(full_data)
    }

def train_and_predict_direction(data, 
                                         feature_columns,
                                         test_size=0.2,
                                         sequence_length=1000,
                                         lookahead_steps=5,  # Number of steps ahead to predict
                                         epochs=1,
                                         batch_size=64,
                                         learning_rate=0.001,
                                         lstm_units=64,
                                         dense_units=32,
                                         dropout_rate=0.3):
    """
    Train an LSTM model to predict whether the log return N steps ahead is positive or negative.
    
    Args:
        data: DataFrame with features
        feature_columns: List of feature column names
        test_size: Fraction of data to use for testing (between 0 and 1)
        sequence_length: Length of input sequences
        lookahead_steps: Number of steps ahead to predict log return direction
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        lstm_units: Number of LSTM units
        dense_units: Number of dense units
        dropout_rate: Dropout rate
    
    Returns:
        dict: Contains predictions, probabilities, model, scaler, and performance metrics
    """
    
    # Create labels for log return direction prediction
    def add_log_return_direction_labels(df, lookahead_steps=5):
        """
        Create binary labels based on whether log return N steps ahead is positive (1) or negative (0)
        """
        labels = []
        
        for i in range(len(df)):
            if i + lookahead_steps >= len(df):
                # For the last few points, we can't look ahead enough steps
                labels.append(np.nan)
            else:
                # Calculate log return N steps ahead
                current_price = df['Close'].iloc[i]
                future_price = df['Close'].iloc[i + lookahead_steps]
                log_return = np.log(future_price / current_price)
                
                # Label: 1 if positive log return, 0 if negative
                label = 1 if log_return > 0 else 0
                labels.append(label)
        
        df[f'log_return_direction_{lookahead_steps}'] = labels
        return df
    
    # Apply labels to entire dataset
    label_col = f'log_return_direction_{lookahead_steps}'
    data = add_log_return_direction_labels(data, lookahead_steps=lookahead_steps)
    
    # Drop rows where label couldn't be assigned (before splitting)
    data = data.dropna(subset=[label_col])
    
    # Scale the data
    scaler = StandardScaler()

    data = scaler.fit_transform(data)

    # Now split the cleaned data chronologically
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    # Create sequences for training
    X_train, y_train = [], []
    for i in range(len(train_data) - sequence_length + 1):
        X_train.append(train_data[feature_columns].iloc[i:i+sequence_length].values)
        y_train.append(train_data[label_col].iloc[i + sequence_length - 1])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train).astype(int)
    
    # Create sequences for testing
    X_test, y_test = [], []
    for i in range(len(test_data) - sequence_length + 1):
        X_test.append(test_data[feature_columns].iloc[i:i+sequence_length].values)
        y_test.append(test_data[label_col].iloc[i + sequence_length - 1])
    
    X_test = np.array(X_test)
    y_test = np.array(y_test).astype(int)
    
    # # Flatten for scaling
    # n_train, seq_len, n_feat = X_train.shape
    # n_test = X_test.shape[0]
    
    # X_train = X_train.reshape(-1, n_feat)
    # X_test = X_test.reshape(-1, n_feat)

    # # Reshape back to 3D
    # X_train = X_train.reshape(n_train, seq_len, n_feat)
    # X_test = X_test.reshape(n_test, seq_len, n_feat)

    # Compute class weights
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
    
    # Build and train initial model (for threshold optimization)
    initial_model = Sequential([
        LSTM(lstm_units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    initial_model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the initial model
    initial_history = initial_model.fit(X_train, y_train, 
                                       validation_data=(X_test, y_test), 
                                       epochs=epochs, 
                                       batch_size=batch_size, 
                                       class_weight=class_weights,
                                       verbose=1)
    
    # Find optimal threshold using test set
    test_probs = initial_model.predict(X_test, verbose=0).flatten()
    
    from sklearn.metrics import precision_recall_curve, f1_score
    precision, recall, thresholds = precision_recall_curve(y_test, test_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Find optimal threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    # Now retrain on full dataset (train + test) for production
    # Combine train and test data
    full_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    
    # Create sequences for full dataset
    X_full, y_full = [], []
    for i in range(len(full_data) - sequence_length + 1):
        X_full.append(full_data[feature_columns].iloc[i:i+sequence_length].values)
        y_full.append(full_data[label_col].iloc[i + sequence_length - 1])
    
    X_full = np.array(X_full)
    y_full = np.array(y_full).astype(int)
    
    # # Scale full dataset using the same scaler
    # n_full, seq_len, n_feat = X_full.shape
    # X_full_2d = X_full.reshape(-1, n_feat)
    # X_full_scaled = scaler.transform(X_full_2d)
    # X_full = X_full_scaled.reshape(n_full, seq_len, n_feat)
    
    # Compute class weights for full dataset
    full_class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_full),
        y=y_full
    )
    full_class_weights = {i: full_class_weights_array[i] for i in range(len(full_class_weights_array))}
    
    # Build and train production model on full dataset
    production_model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_full.shape[1], X_full.shape[2])),
        LSTM(32),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    production_model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train production model on full dataset
    production_history = production_model.fit(X_full, y_full, 
                                             epochs=epochs, 
                                             batch_size=batch_size, 
                                             class_weight=full_class_weights,
                                             verbose=1)
    
    # Make predictions with production model on test set for evaluation
    probs = production_model.predict(X_test, verbose=0).flatten()
    pred_labels = (probs > best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, pred_labels)
    
    # Get test data for return (without sequence_length offset)
    test_data_for_return = test_data.iloc[sequence_length:].reset_index(drop=True)
    
    return {
        'predictions': pred_labels,
        'probabilities': probs,
        'actual_labels': y_test,
        'last_sequence': full_data.iloc[-sequence_length:,:],
        'last_sequence_test': test_data.iloc[-sequence_length:,:],
        'model': production_model,  # Return the production model trained on full dataset
        'scaler': scaler,
        'best_threshold': best_threshold,
        'accuracy': accuracy,
        'class_weights': full_class_weights,
        'training_history': production_history.history,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'full_samples': len(X_full),
        'feature_columns': feature_columns,
        'sequence_length': sequence_length,
        'lookahead_steps': lookahead_steps,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'full_size': len(full_data)
    }

def prepare_input_for_pred(test_data, sequence_length, feature_columns, scaler=None):
    """
    Prepares input for LSTM prediction.
    If test_data is exactly sequence_length rows, returns a single sequence.
    If longer, returns all possible sequences.
    Optionally accepts a fitted scaler for consistent scaling.
    """
    X_test = []
    n = len(test_data)
    if n < sequence_length:
        raise ValueError(f"Not enough data: got {n} rows, need at least {sequence_length}")
    elif n == sequence_length:
        X_test.append(test_data[feature_columns].iloc[:sequence_length].values)
    else:
        for i in range(n - sequence_length + 1):
            X_test.append(test_data[feature_columns].iloc[i:i+sequence_length].values)
    X_test = np.array(X_test)
    # Scaling
    if scaler is not None:
        n_test, seq_len, n_feat = X_test.shape

        X_test_2d = X_test.reshape(-1, n_feat)
        X_test_scaled = scaler.transform(X_test_2d)
        X_test = X_test_scaled.reshape(n_test, seq_len, n_feat)

        # Scale each sequence individually by flattening it completely
        # X_test_scaled = []
        # for i in range(n_test):
        #     seq_flat = X_test[i].flatten()  # Flatten the entire sequence
        #     seq_scaled_flat = scaler.transform(seq_flat.reshape(1, -1)).flatten()
        #     seq_scaled = seq_scaled_flat.reshape(seq_len, n_feat)
        #     X_test_scaled.append(seq_scaled)
        # X_test = np.array(X_test_scaled)

    return X_test

def predict_next_lstm(data, feature_columns, scaler, model, threshold, sequence_length=1000):
    """
    Predict the label for the next data point using the most recent sequence_length data points.
    Args:
        data: DataFrame containing all data up to the current point (including the new point if available)
        feature_columns: List of feature columns used for the model
        scaler: Fitted StandardScaler from training
        model: Trained LSTM model
        threshold: Optimal threshold for decision
        sequence_length: Length of the input sequence for the LSTM
    Returns:
        pred: 0 or 1 (predicted label)
        prob: Probability output by the model
    """
    # Ensure there are enough data points
    if len(data) < sequence_length:
        raise ValueError(f"Not enough data to create a sequence of length {sequence_length}.")
    
    # Get the latest sequence_length rows
    seq = data.iloc[-sequence_length:][feature_columns].values
    # Scale using the fitted scaler (flatten first, then reshape)
    seq_flat = seq.flatten()
    seq_scaled_flat = scaler.transform(seq_flat.reshape(1, -1)).flatten()
    seq_scaled = seq_scaled_flat.reshape(sequence_length, len(feature_columns))
    # Reshape for LSTM input
    X_new = seq_scaled.reshape(1, sequence_length, len(feature_columns))
    # Predict probability
    prob = model.predict(X_new, verbose=0).flatten()[0]
    # Apply threshold
    pred = int(prob > threshold)
    return pred, prob

def sliding_window_evaluation(data, 
                            window_size=100000, 
                            test_size=0.2,
                            step_size=20000,
                            **model_params):
    """
    Perform sliding window evaluation of the LSTM model.
    
    Args:
        data: Full dataset
        window_size: Total size of each window
        test_size: Fraction of window to use for testing (between 0 and 1)
        step_size: How much to move the window forward each time
        **model_params: Parameters to pass to train_and_predict_lstm
    
    Returns:
        list: List of results from each window
    """
    
    results = []
    total_samples = len(data)
    
    # Calculate number of windows
    start_idx = 0
    window_count = 0
    
    while start_idx + window_size <= total_samples:
        print(f"Processing window {window_count + 1}: samples {start_idx} to {start_idx + window_size}")
        
        # Extract window data
        window_data = data.iloc[start_idx:start_idx + window_size].copy()
        
        # Train and predict
        try:
            result = train_and_predict_lstm(window_data, 
                                          test_size=test_size,
                                          **model_params)
            
            # Add window information
            result['window_start'] = start_idx
            result['window_end'] = start_idx + window_size
            result['window_count'] = window_count
            
            results.append(result)
            
            print(f"  Window {window_count + 1} completed - Accuracy: {result['accuracy']:.4f}")
            print(f"  Train samples: {result['train_samples']}, Test samples: {result['test_samples']}")
            
        except Exception as e:
            print(f"  Error in window {window_count + 1}: {e}")
        
        # Move window forward
        start_idx += step_size
        window_count += 1
    
    return results

def analyze_sliding_window_results(results):
    """
    Analyze and visualize the results from sliding window evaluation.
    
    Args:
        results: List of results from sliding_window_evaluation
    """
    
    if not results:
        print("No results to analyze")
        return
    
    # Extract metrics
    accuracies = [r['accuracy'] for r in results]
    thresholds = [r['best_threshold'] for r in results]
    window_counts = [r['window_count'] for r in results]
    train_samples = [r['train_samples'] for r in results]
    test_samples = [r['test_samples'] for r in results]
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Accuracy over time
    ax1.plot(window_counts, accuracies, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Window Number')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Across Windows')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=np.mean(accuracies), color='r', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.4f}')
    ax1.legend()
    
    # Plot 2: Optimal threshold over time
    ax2.plot(window_counts, thresholds, 'go-', linewidth=2, markersize=6)
    ax2.set_xlabel('Window Number')
    ax2.set_ylabel('Optimal Threshold')
    ax2.set_title('Optimal Threshold Across Windows')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=np.mean(thresholds), color='r', linestyle='--', 
                label=f'Mean: {np.mean(thresholds):.4f}')
    ax2.legend()
    
    # Plot 3: Sample sizes over time
    ax3.plot(window_counts, train_samples, 'b-', label='Train Samples', linewidth=2)
    ax3.plot(window_counts, test_samples, 'r-', label='Test Samples', linewidth=2)
    ax3.set_xlabel('Window Number')
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('Sample Sizes Across Windows')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Accuracy distribution
    ax4.hist(accuracies, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Accuracy')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Accuracies')
    ax4.axvline(x=np.mean(accuracies), color='r', linestyle='--', 
                label=f'Mean: {np.mean(accuracies):.4f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSliding Window Evaluation Summary:")
    print(f"==================================")
    print(f"Total windows evaluated: {len(results)}")
    print(f"Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Min accuracy: {np.min(accuracies):.4f}")
    print(f"Max accuracy: {np.max(accuracies):.4f}")
    print(f"Mean optimal threshold: {np.mean(thresholds):.4f} ± {np.std(thresholds):.4f}")
    print(f"Min threshold: {np.min(thresholds):.4f}")
    print(f"Max threshold: {np.max(thresholds):.4f}")
    print(f"Average train samples per window: {np.mean(train_samples):.0f}")
    print(f"Average test samples per window: {np.mean(test_samples):.0f}")
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_threshold': np.mean(thresholds),
        'std_threshold': np.std(thresholds),
        'accuracies': accuracies,
        'thresholds': thresholds,
        'train_samples': train_samples,
        'test_samples': test_samples
    }

def train_and_predict_lstm_memory_efficient(data, 
                                          feature_columns,
                                          test_size=0.2,
                                          sequence_length=1000,
                                          threshold=0.03,
                                          lower_threshold=0.0,
                                          epochs=1,
                                          batch_size=32,  # Reduced batch size
                                          learning_rate=0.001,
                                          lstm_units=32,  # Reduced LSTM units
                                          dense_units=16,  # Reduced dense units
                                          dropout_rate=0.3):
    """
    Memory-efficient version of train_and_predict_lstm that uses generators and smaller models.
    """
    
    # Create labels for entire dataset first (before splitting)
    def add_first_hit_labels(df, threshold=0.02, max_lookahead=60, lower_threshold=0.01):
        labels = []
        sell_price = df['Close_raw'].values * (1 - 0.002/2)
        buy_price = df['Close_raw'].values * (1 + 0.002/2)
        
        for i in range(len(df)):
            entry_price = buy_price[i]
            label = np.nan
            for j in range(1, max_lookahead + 1):
                if i + j >= len(sell_price):
                    break
                future_return = (sell_price[i + j] - entry_price) / entry_price
                if future_return >= threshold:
                    label = 1
                    break
                elif future_return <= -(threshold - lower_threshold):
                    label = 0
                    break
            labels.append(label)
        
        df[f'label_first_hit_{int(threshold*100)}'] = labels
        return df
    
    # Apply labels to entire dataset
    label_col = f'label_first_hit_{int(threshold*100)}'
    data1 = data.iloc[:int(len(data)*(1-test_size)),:]
    data1 = add_first_hit_labels(data1, threshold=threshold, max_lookahead=len(data1), lower_threshold=lower_threshold)
    data2 = data.iloc[int(len(data)*(1-test_size)):,:]
    data2 = add_first_hit_labels(data2, threshold=threshold, max_lookahead=len(data2), lower_threshold=lower_threshold)
    data = pd.concat([data1, data2])
    # Drop rows where label couldn't be assigned (before splitting)
    data = data.dropna(subset=[label_col])
    
    # Now split the cleaned data chronologically
    split_idx = int(len(data) * (1 - test_size))
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    # Create data generator for memory efficiency
    def create_sequences_generator(data_subset, feature_columns, label_col, sequence_length, batch_size=32):
        """Generator that yields batches of sequences to reduce memory usage"""
        n_samples = len(data_subset) - sequence_length
        
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = []
            y_batch = []
            
            for j in range(i, batch_end):
                X_batch.append(data_subset[feature_columns].iloc[j:j+sequence_length].values)
                y_batch.append(data_subset[label_col].iloc[j + sequence_length])
            
            yield np.array(X_batch), np.array(y_batch).astype(int)
    
    # Scale the data using a sample
    scaler = StandardScaler()
    
    # Use a sample of training data to fit the scaler
    sample_size = min(1000, len(train_data) - sequence_length)
    sample_indices = np.random.choice(len(train_data) - sequence_length, sample_size, replace=False)
    
    sample_data = []
    for idx in sample_indices:
        # Get the sequence and flatten it properly
        sequence = train_data[feature_columns].iloc[idx:idx+sequence_length].values
        sample_data.append(sequence.flatten())
    
    sample_data = np.array(sample_data)
    scaler.fit(sample_data)
    
    # Build and train initial model (for threshold optimization)
    initial_model = Sequential([
        LSTM(lstm_units, input_shape=(sequence_length, len(feature_columns)), return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    initial_model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train using generator
    train_generator = create_sequences_generator(train_data, feature_columns, label_col, sequence_length, batch_size)
    
    # Count total batches for steps_per_epoch
    total_train_sequences = len(train_data) - sequence_length
    steps_per_epoch = (total_train_sequences + batch_size - 1) // batch_size
    
    # Create validation data (smaller sample for memory efficiency)
    val_size = min(1000, len(test_data) - sequence_length)
    val_indices = np.random.choice(len(test_data) - sequence_length, val_size, replace=False)
    
    X_val, y_val = [], []
    for idx in val_indices:
        seq = test_data[feature_columns].iloc[idx:idx+sequence_length].values
        # Scale the flattened sequence
        seq_flat = seq.flatten()
        seq_scaled_flat = scaler.transform(seq_flat.reshape(1, -1)).flatten()
        seq_scaled = seq_scaled_flat.reshape(sequence_length, len(feature_columns))
        X_val.append(seq_scaled)
        y_val.append(test_data[label_col].iloc[idx + sequence_length])
    
    X_val = np.array(X_val)
    y_val = np.array(y_val).astype(int)
    
    # Compute class weights using a sample
    sample_labels = []
    for i in range(0, len(train_data) - sequence_length, 10):  # Sample every 10th
        sample_labels.append(train_data[label_col].iloc[i + sequence_length])
    
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(sample_labels),
        y=sample_labels
    )
    class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
    
    # Custom training loop for memory efficiency
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_generator = create_sequences_generator(train_data, feature_columns, label_col, sequence_length, batch_size)
        
        for step in range(steps_per_epoch):
            try:
                X_batch, y_batch = next(train_generator)
                
                # Scale the batch
                batch_shape = X_batch.shape
                # Flatten each sequence in the batch
                X_batch_flat = X_batch.reshape(batch_shape[0], -1)
                X_batch_scaled_flat = scaler.transform(X_batch_flat)
                X_batch = X_batch_scaled_flat.reshape(batch_shape)
                
                # Train on batch
                initial_model.train_on_batch(X_batch, y_batch, class_weight=class_weights)
                
                if step % 100 == 0:
                    print(f"  Step {step}/{steps_per_epoch}")
                    
            except StopIteration:
                break
    
    # Find optimal threshold using validation set
    test_probs = initial_model.predict(X_val, verbose=0).flatten()
    
    from sklearn.metrics import precision_recall_curve, f1_score
    precision, recall, thresholds = precision_recall_curve(y_val, test_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Find optimal threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    # Now retrain on full dataset for production (using smaller sample for memory)
    # Use a larger sample for production model
    prod_sample_size = min(5000, len(data) - sequence_length)
    prod_indices = np.random.choice(len(data) - sequence_length, prod_sample_size, replace=False)
    
    X_full, y_full = [], []
    for idx in prod_indices:
        seq = data[feature_columns].iloc[idx:idx+sequence_length].values
        # Scale the flattened sequence
        seq_flat = seq.flatten()
        seq_scaled_flat = scaler.transform(seq_flat.reshape(1, -1)).flatten()
        seq_scaled = seq_scaled_flat.reshape(sequence_length, len(feature_columns))
        X_full.append(seq_scaled)
        y_full.append(data[label_col].iloc[idx + sequence_length])
    
    X_full = np.array(X_full)
    y_full = np.array(y_full).astype(int)
    
    # Compute class weights for full dataset sample
    full_class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_full),
        y=y_full
    )
    full_class_weights = {i: full_class_weights_array[i] for i in range(len(full_class_weights_array))}
    
    # Build and train production model on full dataset sample
    production_model = Sequential([
        LSTM(lstm_units, input_shape=(sequence_length, len(feature_columns)), return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    production_model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train production model on full dataset sample
    production_history = production_model.fit(X_full, y_full, 
                                             epochs=epochs, 
                                             batch_size=batch_size, 
                                             class_weight=full_class_weights,
                                             verbose=1)
    
    # Make predictions with production model on validation set for evaluation
    probs = production_model.predict(X_val, verbose=0).flatten()
    pred_labels = (probs > best_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val, pred_labels)
    
    return {
        'predictions': pred_labels,
        'probabilities': probs,
        'actual_labels': y_val,
        'last_sequence': data.iloc[-sequence_length:,:],
        'model': production_model,
        'scaler': scaler,
        'best_threshold': best_threshold,
        'accuracy': accuracy,
        'class_weights': full_class_weights,
        'training_history': production_history.history,
        'train_samples': total_train_sequences,
        'test_samples': val_size,
        'full_samples': prod_sample_size,
        'feature_columns': feature_columns,
        'sequence_length': sequence_length,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'full_size': len(data)
    }

def calculate_directional_accuracy(pred_labels, test_data, lookahead_steps=1):
    """
    Calculate the accuracy of directional predictions by comparing predicted directions
    with actual log return directions.
    
    Args:
        pred_labels: Array of predicted labels (0 or 1)
        test_data: DataFrame containing the test data with 'Close' prices
        lookahead_steps: Number of steps ahead to calculate log returns (default=1)
    
    Returns:
        dict: Contains accuracy metrics and detailed results
    """
    actual_directions = []
    actual_log_returns = []
    
    # Calculate actual log return directions
    for i in range(len(test_data) - lookahead_steps):
        if i + lookahead_steps >= len(test_data):
            break
            
        current_price = test_data['Close'].iloc[i]
        future_price = test_data['Close'].iloc[i + lookahead_steps]
        log_return = np.log(future_price / current_price)
        
        # Direction: 1 if positive, 0 if negative
        actual_direction = 1 if log_return > 0 else 0
        
        actual_directions.append(actual_direction)
        actual_log_returns.append(log_return)
    
    # Align predictions with actual directions
    # Note: pred_labels might be longer than actual_directions due to sequence_length offset
    aligned_predictions = pred_labels[:len(actual_directions)]
    
    # Calculate accuracy metrics
    correct_predictions = sum(aligned_predictions == actual_directions)
    total_predictions = len(aligned_predictions)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate additional metrics
    true_positives = sum((aligned_predictions == 1) & (actual_directions == 1))
    true_negatives = sum((aligned_predictions == 0) & (actual_directions == 0))
    false_positives = sum((aligned_predictions == 1) & (actual_directions == 0))
    false_negatives = sum((aligned_predictions == 0) & (actual_directions == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate average log return for correct vs incorrect predictions
    correct_predictions_mask = aligned_predictions == actual_directions
    avg_log_return_correct = np.mean([actual_log_returns[i] for i in range(len(actual_log_returns)) if correct_predictions_mask[i]]) if sum(correct_predictions_mask) > 0 else 0
    avg_log_return_incorrect = np.mean([actual_log_returns[i] for i in range(len(actual_log_returns)) if not correct_predictions_mask[i]]) if sum(~correct_predictions_mask) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'avg_log_return_correct': avg_log_return_correct,
        'avg_log_return_incorrect': avg_log_return_incorrect,
        'actual_directions': actual_directions,
        'predicted_directions': aligned_predictions.tolist(),
        'actual_log_returns': actual_log_returns
    }

# Example usage:
"""
# Run sliding window evaluation
results = sliding_window_evaluation(
    data=data,
    window_size=100000,  # 100k samples per window
    test_size=0.2,       # 20% for testing (80% for training)
    step_size=20000,     # Move forward by 20k each time
    sequence_length=1000,
    threshold=0.03,
    lower_threshold=0.0,
    epochs=1,
    batch_size=64
)

# Analyze results
summary = analyze_sliding_window_results(results)
""" 