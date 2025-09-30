import pandas as pd
import pandas_ta as ta
from datetime import datetime
import numpy as np

from simulate_trades_modified import simulate_trades_max_positions, simulate_trades, simulate_trades2, simulate_trades_max_positions_with_shorts
from train_and_predict import train_and_predict_lstm, create_indicators, prepare_input_for_pred, train_and_predict_lstm_memory_efficient

results = []
results2 = []
results3 = []
results4 = []

start_point = 25000
train_size = 6000
test_size = 500

for i in range(20):
    data = pd.read_csv('btcusd_15-min_data.csv').iloc[-start_point + i*test_size:-(start_point - train_size) + i*test_size,:]

    og_data = data.copy()

    data = create_indicators(data)

    ts = 0.03
    lts = 0
    sequence_length=1000

    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MACD', 'BBL', 'BBM', 'BBU', 'BB_BANDWIDTH',
        'SMA', 'EMA', 'RSI',
        'SMA_10', 'SMA_50', 'SMA_200', 'EMA_10', 'EMA_50', 'EMA_200',
        'SMA_10_50_diff', 'SMA_10_200_diff', 'SMA_50_200_diff',
        'EMA_10_50_diff', 'EMA_10_200_diff', 'EMA_50_200_diff',
        'return_1', 'return_2', 'return_3'
    ]

    result = train_and_predict_lstm(data, feature_columns, epochs=1, sequence_length=sequence_length,
    threshold=ts, lower_threshold=lts)

    production_model = result['model']
    best_threshold = result['best_threshold'] 
    scaler = result['scaler']
    first_pred = result['probabilities'][-1]
    new_data = pd.read_csv('btcusd_15-min_data.csv').iloc[-(start_point - train_size) + i*test_size:-(start_point - train_size - test_size) + i*test_size,:]

    train_data = result['last_sequence_test'].iloc[:,:-1]

    # Make prediction for latest data point available
    last_data = pd.concat([og_data, new_data])
    last_data = create_indicators(last_data)

    last_data = pd.concat([train_data.iloc[-sequence_length+1:,:], last_data.iloc[-test_size:,:]])

    X_test = prepare_input_for_pred(last_data, sequence_length, feature_columns, scaler=scaler)

    probs = production_model.predict(X_test, verbose=0).flatten()
    pred_labels = (probs > best_threshold).astype(int)

    # Strat 1
    sim_results, max_open_trades, signals_ignored, open_trades_history = simulate_trades_max_positions_with_shorts(
        last_data.iloc[-test_size:,:], pred_labels.flatten(), max_positions=10, max_lookahead=len(last_data), threshold=ts, lower_threshold=lts
    )

    # Strat 2
    sim_results2, max_open_trades2, signals_ignored2, open_trades_history2 = simulate_trades_max_positions(
        last_data.iloc[-test_size:,:], pred_labels.flatten(), max_positions=10, max_lookahead=len(last_data), threshold=ts, lower_threshold=lts
    )

    # Strat 3
    pred_labels2 = (probs > 0.5).astype(int)

    sim_results3, max_open_trades3, signals_ignored3, open_trades_history3 = simulate_trades_max_positions(
        last_data.iloc[-test_size:,:], pred_labels2.flatten(), max_positions=10, max_lookahead=len(last_data), threshold=ts, lower_threshold=lts
    )

    # Strat 3
    pred_labels3 = (probs > 0.5).astype(int)

    sim_results4, max_open_trades4, signals_ignored4, open_trades_history4 = simulate_trades_max_positions_with_shorts(
        last_data.iloc[-test_size:,:], pred_labels2.flatten(), max_positions=10, max_lookahead=len(last_data), threshold=ts, lower_threshold=lts
    )

    total_profit = sim_results['profit_usd'].sum()
    win_rate = (sim_results['result'] == 'win').mean()
    avg_return = sim_results['pnl_pct'].mean()
    cum_profit = sim_results['profit_usd'].cumsum()

    results.append({
        "Total Profit": total_profit,
        "Win Rate": win_rate,
        "Average Return per Trade": avg_return,
        "Number of Trades": len(sim_results),
        "Maximum Open Trades": max_open_trades,
        "Max drawdown": cum_profit.min()
    })

    if sum(pred_labels)==0:
        results2.append({
            "Total Profit": None,
            "Win Rate": None,
            "Average Return per Trade": None,
            "Number of Trades": None,
            "Maximum Open Trades": None,
            "Max drawdown": None
        })
    else:
        # Print performance summary for max positions version
        total_profit = sim_results2['profit_usd'].sum()
        win_rate = (sim_results2['result'] == 'win').mean()
        avg_return = sim_results2['pnl_pct'].mean()
        cum_profit = sim_results2['profit_usd'].cumsum()

        results2.append({
            "Total Profit": total_profit,
            "Win Rate": win_rate,
            "Average Return per Trade": avg_return,
            "Number of Trades": len(sim_results2),
            "Maximum Open Trades": max_open_trades,
            "Max drawdown": cum_profit.min()
        })

    if sum(pred_labels2)==0:
        results3.append({
            "Total Profit": None,
            "Win Rate": None,
            "Average Return per Trade": None,
            "Number of Trades": None,
            "Maximum Open Trades": None,
            "Max drawdown": None
        })
    else:
    # Print performance summary for max positions version
        total_profit = sim_results3['profit_usd'].sum()
        win_rate = (sim_results3['result'] == 'win').mean()
        avg_return = sim_results3['pnl_pct'].mean()
        cum_profit = sim_results3['profit_usd'].cumsum()

        results3.append({
            "Total Profit": total_profit,
            "Win Rate": win_rate,
            "Average Return per Trade": avg_return,
            "Number of Trades": len(sim_results3),
            "Maximum Open Trades": max_open_trades,
            "Max drawdown": cum_profit.min()
        })

    if sum(pred_labels3)==0:
        results4.append({
            "Total Profit": None,
            "Win Rate": None,
            "Average Return per Trade": None,
            "Number of Trades": None,
            "Maximum Open Trades": None,
            "Max drawdown": None
        })
    else:
    # Print performance summary for max positions version
        total_profit = sim_results4['profit_usd'].sum()
        win_rate = (sim_results4['result'] == 'win').mean()
        avg_return = sim_results4['pnl_pct'].mean()
        cum_profit = sim_results4['profit_usd'].cumsum()

        results4.append({
            "Total Profit": total_profit,
            "Win Rate": win_rate,
            "Average Return per Trade": avg_return,
            "Number of Trades": len(sim_results4),
            "Maximum Open Trades": max_open_trades,
            "Max drawdown": cum_profit.min()
        })