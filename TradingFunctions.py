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

def add_first_hit_labels(df, threshold=0.03, max_lookahead=60, lower_threshold=0.01):

    """
    Function that creates the target labels. For each price in df, it looks ahead to check whether returns
    first hit the upper threshold (threshold) or the lower threshold (lower_threshold). If not specified, this
    function looks 60 steps ahead at most. It accounts for the bid-ask spread
    """

    labels = []
    # Correct prices for bid-ask spread
    sell_price = df['Close_raw'].values * (1 - 0.002/2) 
    buy_price = df['Close_raw'].values * (1 + 0.002/2)
    
    # For each price it looks ahead and creates labels based on future returns. It ignores the last, unrealized return
    for i in range(len(df)):
        entry_price = buy_price[i]
        label = np.nan

        for j in range(1, max_lookahead + 1):
            if i + j >= len(sell_price): # Breaks if no more data
                break
            future_return = (sell_price[i + j] - entry_price) / entry_price

            if future_return >= threshold:
                label = 1 # Label 1 if return crosses upper threshold
                break
            elif future_return <= -(threshold - lower_threshold):
                label = 0 # Label 0 if return crosses lower threshold
                break

        labels.append(label)
    
    # Similar to the previous loop but it registers the last return even if it didn't cross any threshold
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

    # Add a column with labels
    df[f'label_first_hit_{int(threshold*100)}'] = labels
    return df

def train_and_predict_lstm(data, 
                          feature_columns,
                          label_col,
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
    Trains an LSTM model on a subset of data and makes predictions on the test set. Predictions on the test set
    are used to determine the optimal threshold. After finding optimal threshold, retrains on full dataset
    for production use.
    
    Args:
        data: DataFrame with features and labels
        feature_columns: list of features to include in the model
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
    
    """
    In the part below, we estimate the optimal threshold used to determine whether the predicted probabilities
    in the output nodes should be translated to 1 or 0
    """

    # Split the original train data into a smaller train dataset and a test test for threshold optimization
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
    # Model 1
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
    
    """
    In the part below, we return to our original training dataset and train another LSTM model which will be used
    to generate our final predictions
    """
    
    # Combine train and test data
    full_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    
    # Create sequences for full dataset
    X_full, y_full = [], []
    for i in range(len(full_data) - sequence_length + 1):
        X_full.append(full_data[feature_columns].iloc[i:i+sequence_length].values)
        y_full.append(full_data[label_col].iloc[i + sequence_length - 1])
    
    X_full = np.array(X_full)
    y_full = np.array(y_full).astype(int)
    
    # Scale full dataset using the same scaler
    n_full, seq_len, n_feat = X_full.shape
    X_full_2d = X_full.reshape(-1, n_feat)
    X_full_scaled = scaler.fit_transform(X_full_2d)
    X_full = X_full_scaled.reshape(n_full, seq_len, n_feat)
    
    # Compute class weights for full dataset
    full_class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_full),
        y=y_full
    )
    full_class_weights = {i: full_class_weights_array[i] for i in range(len(full_class_weights_array))}
    
    # # Build and train production model on full dataset
    # # Model 1
    # production_model = Sequential([
    #     LSTM(lstm_units, input_shape=(X_full.shape[1], X_full.shape[2]), return_sequences=False),
    #     Dropout(dropout_rate),
    #     Dense(dense_units, activation='relu'),
    #     Dense(1, activation='sigmoid')
    # ])

    # Model 2
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

def train_and_predict_direction(data, 
                                         feature_columns,
                                         label_col,
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