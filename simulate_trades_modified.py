import pandas as pd
import numpy as np

def simulate_trades_with_open_tracking(data, preds, max_lookahead=60, threshold=0.02, spread=0.002, slippage=0.0005, trade_size=1000,
                    transaction_cost=5, lower_threshold=0.01):
    """
    Simulate long trades when the model predicts a 1 (entry signal).
    Considers spread, slippage, and realistic exit logic.
    Tracks maximum number of open trades simultaneously.
    
    Returns:
    - DataFrame with trade results
    - Maximum number of open trades simultaneously
    - List of open trade counts at each time step
    """
    close_prices = data['Close_raw'].values
    trades = []
    
    # Track open trades: list of (entry_index, entry_price, entry_time)
    open_trades = []
    max_open_trades = 0
    open_trades_history = []  # Track number of open trades at each time step
    
    for i in range(len(preds)):
        # Check for trade exits first
        still_open_trades = []
        for trade in open_trades:
            entry_index, entry_price, entry_time = trade
            
            # Check if this trade should exit at current time
            if i >= entry_index:  # Trade has been open for at least one period
                current_price = close_prices[i]
                pnl = (current_price - entry_price) / entry_price
                
                if pnl >= threshold:
                    # Trade wins
                    exit_price = current_price * (1 - spread / 2 - slippage)
                    final_return = (exit_price - entry_price) / entry_price
                    profit = trade_size * final_return - transaction_cost
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': i,
                        'entry_time': entry_time,
                        'exit_time': i,
                        'result': 'win',
                        'pnl_pct': final_return,
                        'profit_usd': profit,
                        'duration': i - entry_index
                    })
                elif pnl <= -(threshold - lower_threshold):
                    # Trade loses
                    exit_price = current_price * (1 - spread / 2 - slippage)
                    final_return = (exit_price - entry_price) / entry_price
                    profit = trade_size * final_return - transaction_cost
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': i,
                        'entry_time': entry_time,
                        'exit_time': i,
                        'result': 'loss',
                        'pnl_pct': final_return,
                        'profit_usd': profit,
                        'duration': i - entry_index
                    })
                elif i - entry_index >= max_lookahead:
                    # Trade times out
                    exit_price = current_price * (1 - spread / 2 - slippage)
                    final_return = (exit_price - entry_price) / entry_price
                    profit = trade_size * final_return - transaction_cost
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': i,
                        'entry_time': entry_time,
                        'exit_time': i,
                        'result': 'timeout',
                        'pnl_pct': final_return,
                        'profit_usd': profit,
                        'duration': i - entry_index
                    })
                else:
                    # Trade is still open
                    still_open_trades.append(trade)
        
        open_trades = still_open_trades
        
        # Check for new trade entry
        if preds[i] == 1:
            entry_price = close_prices[i] * (1 + spread / 2 + slippage)
            open_trades.append((i, entry_price, i))
        
        # Update max open trades
        current_open_trades = len(open_trades)
        max_open_trades = max(max_open_trades, current_open_trades)
        open_trades_history.append(current_open_trades)
    
    # Close any remaining open trades at the end
    for trade in open_trades:
        entry_index, entry_price, entry_time = trade
        final_price = close_prices[-1] * (1 - spread / 2 - slippage)
        final_return = (final_price - entry_price) / entry_price
        profit = trade_size * final_return - transaction_cost
        trades.append({
            'entry_index': entry_index,
            'exit_index': len(close_prices) - 1,
            'entry_time': entry_time,
            'exit_time': len(close_prices) - 1,
            'result': 'end_of_data',
            'pnl_pct': final_return,
            'profit_usd': profit,
            'duration': len(close_prices) - 1 - entry_index
        })

    return pd.DataFrame(trades), max_open_trades, open_trades_history

def simulate_trades_single_only(data, preds, max_lookahead=60, threshold=0.02, spread=0.002, slippage=0.0005, trade_size=1000,
                    transaction_cost=5, lower_threshold=0.01):
    """
    Simulate long trades when the model predicts a 1 (entry signal).
    Only allows ONE trade at a time - no new trades can be opened while a trade is active.
    Considers spread, slippage, and realistic exit logic.
    
    Returns:
    - DataFrame with trade results
    - Number of signals ignored due to existing open trade
    """
    close_prices = data['Close_raw'].values
    trades = []
    
    # Track single open trade: (entry_index, entry_price, entry_time) or None
    current_trade = None
    signals_ignored = 0
    
    for i in range(len(preds)):
        # Check if current trade should exit
        if current_trade is not None:
            entry_index, entry_price, entry_time = current_trade
            
            # Check if this trade should exit at current time
            if i >= entry_index:  # Trade has been open for at least one period
                current_price = close_prices[i]
                pnl = (current_price - entry_price) / entry_price
                
                if pnl >= threshold:
                    # Trade wins
                    exit_price = current_price * (1 - spread / 2 - slippage)
                    final_return = (exit_price - entry_price) / entry_price
                    profit = trade_size * final_return - transaction_cost
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': i,
                        'entry_time': entry_time,
                        'exit_time': i,
                        'result': 'win',
                        'pnl_pct': final_return,
                        'profit_usd': profit,
                        'duration': i - entry_index
                    })
                    current_trade = None  # Trade closed
                    
                elif pnl <= -(threshold - lower_threshold):
                    # Trade loses
                    exit_price = current_price * (1 - spread / 2 - slippage)
                    final_return = (exit_price - entry_price) / entry_price
                    profit = trade_size * final_return - transaction_cost
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': i,
                        'entry_time': entry_time,
                        'exit_time': i,
                        'result': 'loss',
                        'pnl_pct': final_return,
                        'profit_usd': profit,
                        'duration': i - entry_index
                    })
                    current_trade = None  # Trade closed
                    
                elif i - entry_index >= max_lookahead:
                    # Trade times out
                    exit_price = current_price * (1 - spread / 2 - slippage)
                    final_return = (exit_price - entry_price) / entry_price
                    profit = trade_size * final_return - transaction_cost
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': i,
                        'entry_time': entry_time,
                        'exit_time': i,
                        'result': 'timeout',
                        'pnl_pct': final_return,
                        'profit_usd': profit,
                        'duration': i - entry_index
                    })
                    current_trade = None  # Trade closed
        
        # Check for new trade entry (only if no trade is currently open)
        if preds[i] == 1 and current_trade is None:
            entry_price = close_prices[i] * (1 + spread / 2 + slippage)
            current_trade = (i, entry_price, i)
        elif preds[i] == 1 and current_trade is not None:
            # Signal ignored because trade is already open
            signals_ignored += 1
    
    # Close any remaining open trade at the end
    if current_trade is not None:
        entry_index, entry_price, entry_time = current_trade
        final_price = close_prices[-1] * (1 - spread / 2 - slippage)
        final_return = (final_price - entry_price) / entry_price
        profit = trade_size * final_return - transaction_cost
        trades.append({
            'entry_index': entry_index,
            'exit_index': len(close_prices) - 1,
            'entry_time': entry_time,
            'exit_time': len(close_prices) - 1,
            'result': 'end_of_data',
            'pnl_pct': final_return,
            'profit_usd': profit,
            'duration': len(close_prices) - 1 - entry_index
        })

    return pd.DataFrame(trades), signals_ignored

def simulate_trades_max_positions(data, preds, max_positions=3, max_lookahead=60, threshold=0.02, spread=0.002, slippage=0.0005, trade_size=1000,
                    transaction_cost=5, lower_threshold=0.01):
    """
    Simulate long trades when the model predicts a 1 (entry signal).
    Allows a maximum number of open trades simultaneously.
    Considers spread, slippage, and realistic exit logic.
    
    Args:
        max_positions: Maximum number of trades that can be open simultaneously
    
    Returns:
    - DataFrame with trade results
    - Maximum number of open trades simultaneously (should not exceed max_positions)
    - Number of signals ignored due to position limit
    - List of open trade counts at each time step
    """
    close_prices = data['Close_raw'].values
    trades = []
    
    # Track open trades: list of (entry_index, entry_price, entry_time)
    open_trades = []
    max_open_trades = 0
    signals_ignored = 0
    open_trades_history = []  # Track number of open trades at each time step
    
    for i in range(len(preds)):
        # Check for trade exits first
        still_open_trades = []
        for trade in open_trades:
            entry_index, entry_price, entry_time = trade
            
            # Check if this trade should exit at current time
            if i >= entry_index:  # Trade has been open for at least one period
                current_price = close_prices[i]
                pnl = (current_price - entry_price) / entry_price
                
                if pnl >= threshold:
                    # Trade wins
                    exit_price = current_price * (1 - spread / 2 - slippage)
                    final_return = (exit_price - entry_price) / entry_price
                    profit = trade_size * final_return - transaction_cost
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': i,
                        'entry_time': entry_time,
                        'exit_time': i,
                        'result': 'win',
                        'pnl_pct': final_return,
                        'profit_usd': profit,
                        'duration': i - entry_index
                    })
                elif pnl <= -(threshold - lower_threshold):
                    # Trade loses
                    exit_price = current_price * (1 - spread / 2 - slippage)
                    final_return = (exit_price - entry_price) / entry_price
                    profit = trade_size * final_return - transaction_cost
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': i,
                        'entry_time': entry_time,
                        'exit_time': i,
                        'result': 'loss',
                        'pnl_pct': final_return,
                        'profit_usd': profit,
                        'duration': i - entry_index
                    })
                elif i - entry_index >= max_lookahead:
                    # Trade times out
                    exit_price = current_price * (1 - spread / 2 - slippage)
                    final_return = (exit_price - entry_price) / entry_price
                    profit = trade_size * final_return - transaction_cost
                    trades.append({
                        'entry_index': entry_index,
                        'exit_index': i,
                        'entry_time': entry_time,
                        'exit_time': i,
                        'result': 'timeout',
                        'pnl_pct': final_return,
                        'profit_usd': profit,
                        'duration': i - entry_index
                    })
                else:
                    # Trade is still open
                    still_open_trades.append(trade)
        
        open_trades = still_open_trades
        
        # Check for new trade entry (only if under position limit)
        if preds[i] == 1 and len(open_trades) < max_positions:
            entry_price = close_prices[i] * (1 + spread / 2 + slippage)
            open_trades.append((i, entry_price, i))
        elif preds[i] == 1 and len(open_trades) >= max_positions:
            # Signal ignored because position limit reached
            signals_ignored += 1
        
        # Update max open trades
        current_open_trades = len(open_trades)
        max_open_trades = max(max_open_trades, current_open_trades)
        open_trades_history.append(current_open_trades)
    
    # Close any remaining open trades at the end
    for trade in open_trades:
        entry_index, entry_price, entry_time = trade
        final_price = close_prices[-1] * (1 - spread / 2 - slippage)
        final_return = (final_price - entry_price) / entry_price
        profit = trade_size * final_return - transaction_cost
        trades.append({
            'entry_index': entry_index,
            'exit_index': len(close_prices) - 1,
            'entry_time': entry_time,
            'exit_time': len(close_prices) - 1,
            'result': 'end_of_data',
            'pnl_pct': final_return,
            'profit_usd': profit,
            'duration': len(close_prices) - 1 - entry_index
        })

    return pd.DataFrame(trades), max_open_trades, signals_ignored, open_trades_history

def simulate_trades_max_positions_with_shorts(data, preds, max_positions=3, max_lookahead=60, threshold=0.02, spread=0.002, slippage=0.0005, trade_size=1000,
                    transaction_cost=5, lower_threshold=0.01):
    """
    Simulate both long and short trades with position limiting.
    - 1 = go long
    - 0 = go short
    Allows a maximum number of open trades simultaneously.
    Considers spread, slippage, and realistic exit logic.
    
    Args:
        max_positions: Maximum number of trades that can be open simultaneously
        preds: Predictions where 1 = long, 0 = short
    
    Returns:
    - DataFrame with trade results
    - Maximum number of open trades simultaneously (should not exceed max_positions)
    - Number of signals ignored due to position limit
    - List of open trade counts at each time step
    """
    close_prices = data['Close_raw'].values
    trades = []
    
    # Track open trades: list of (entry_index, entry_price, entry_time, direction)
    # direction: 'long' or 'short'
    open_trades = []
    max_open_trades = 0
    signals_ignored = 0
    open_trades_history = []  # Track number of open trades at each time step
    
    for i in range(len(preds)):
        # Check for trade exits first
        still_open_trades = []
        for trade in open_trades:
            entry_index, entry_price, entry_time, direction = trade
            
            # Check if this trade should exit at current time
            if i >= entry_index:  # Trade has been open for at least one period
                current_price = close_prices[i]
                
                if direction == 'long':
                    pnl = (current_price - entry_price) / entry_price
                    
                    if pnl >= threshold:
                        # Long trade wins
                        exit_price = current_price * (1 - spread / 2 - slippage)
                        final_return = (exit_price - entry_price) / entry_price
                        profit = trade_size * final_return - transaction_cost
                        trades.append({
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'entry_time': entry_time,
                            'exit_time': i,
                            'direction': 'long',
                            'result': 'win',
                            'pnl_pct': final_return,
                            'profit_usd': profit,
                            'duration': i - entry_index
                        })
                    elif pnl <= -(threshold - lower_threshold):
                        # Long trade loses
                        exit_price = current_price * (1 - spread / 2 - slippage)
                        final_return = (exit_price - entry_price) / entry_price
                        profit = trade_size * final_return - transaction_cost
                        trades.append({
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'entry_time': entry_time,
                            'exit_time': i,
                            'direction': 'long',
                            'result': 'loss',
                            'pnl_pct': final_return,
                            'profit_usd': profit,
                            'duration': i - entry_index
                        })
                    elif i - entry_index >= max_lookahead:
                        # Long trade times out
                        exit_price = current_price * (1 - spread / 2 - slippage)
                        final_return = (exit_price - entry_price) / entry_price
                        profit = trade_size * final_return - transaction_cost
                        trades.append({
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'entry_time': entry_time,
                            'exit_time': i,
                            'direction': 'long',
                            'result': 'timeout',
                            'pnl_pct': final_return,
                            'profit_usd': profit,
                            'duration': i - entry_index
                        })
                    else:
                        # Long trade is still open
                        still_open_trades.append(trade)
                
                elif direction == 'short':
                    pnl = (entry_price - current_price) / entry_price
                    
                    if pnl >= (threshold - lower_threshold):
                        # Short trade wins
                        exit_price = current_price * (1 + spread / 2 + slippage)
                        final_return = (entry_price - exit_price) / entry_price
                        profit = trade_size * final_return - transaction_cost
                        trades.append({
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'entry_time': entry_time,
                            'exit_time': i,
                            'direction': 'short',
                            'result': 'win',
                            'pnl_pct': final_return,
                            'profit_usd': profit,
                            'duration': i - entry_index
                        })
                    elif pnl <= -threshold:
                        # Short trade loses
                        exit_price = current_price * (1 + spread / 2 + slippage)
                        final_return = (entry_price - exit_price) / entry_price
                        profit = trade_size * final_return - transaction_cost
                        trades.append({
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'entry_time': entry_time,
                            'exit_time': i,
                            'direction': 'short',
                            'result': 'loss',
                            'pnl_pct': final_return,
                            'profit_usd': profit,
                            'duration': i - entry_index
                        })
                    elif i - entry_index >= max_lookahead:
                        # Short trade times out
                        exit_price = current_price * (1 + spread / 2 + slippage)
                        final_return = (entry_price - exit_price) / entry_price
                        profit = trade_size * final_return - transaction_cost
                        trades.append({
                            'entry_index': entry_index,
                            'exit_index': i,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'entry_time': entry_time,
                            'exit_time': i,
                            'direction': 'short',
                            'result': 'timeout',
                            'pnl_pct': final_return,
                            'profit_usd': profit,
                            'duration': i - entry_index
                        })
                    else:
                        # Short trade is still open
                        still_open_trades.append(trade)
        
        open_trades = still_open_trades
        
        # Check for new trade entry (only if under position limit)
        if preds[i] in [0, 1] and len(open_trades) < max_positions:
            if preds[i] == 1:
                # LONG TRADE
                entry_price = close_prices[i] * (1 + spread / 2 + slippage)
                open_trades.append((i, entry_price, i, 'long'))
            else:
                # SHORT TRADE
                entry_price = close_prices[i] * (1 - spread / 2 - slippage)
                open_trades.append((i, entry_price, i, 'short'))
        elif preds[i] in [0, 1] and len(open_trades) >= max_positions:
            # Signal ignored because position limit reached
            signals_ignored += 1
        
        # Update max open trades
        current_open_trades = len(open_trades)
        max_open_trades = max(max_open_trades, current_open_trades)
        open_trades_history.append(current_open_trades)
    
    # Close any remaining open trades at the end
    for trade in open_trades:
        entry_index, entry_price, entry_time, direction = trade
        final_price = close_prices[-1]
        
        if direction == 'long':
            exit_price = final_price * (1 - spread / 2 - slippage)
            final_return = (exit_price - entry_price) / entry_price
        else:  # short
            exit_price = final_price * (1 + spread / 2 + slippage)
            final_return = (entry_price - exit_price) / entry_price
        
        profit = trade_size * final_return - transaction_cost
        trades.append({
            'entry_index': entry_index,
            'exit_index': len(close_prices) - 1,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_time': entry_time,
            'exit_time': len(close_prices) - 1,
            'direction': direction,
            'result': 'end_of_data',
            'pnl_pct': final_return,
            'profit_usd': profit,
            'duration': len(close_prices) - 1 - entry_index
        })

    return pd.DataFrame(trades), max_open_trades, signals_ignored, open_trades_history

# Example usage:
"""
# For multiple trades tracking:
sim_results, max_open_trades, open_trades_history = simulate_trades_with_open_tracking(
    test_data, pred_labels.flatten(), max_lookahead=len(test_data), threshold=ts, lower_threshold=lts
)

# For single trade only:
sim_results, signals_ignored = simulate_trades_single_only(
    test_data, pred_labels.flatten(), max_lookahead=len(test_data), threshold=ts, lower_threshold=lts
)

# For maximum positions (e.g., max 3 trades):
sim_results, max_open_trades, signals_ignored, open_trades_history = simulate_trades_max_positions(
    test_data, pred_labels.flatten(), max_positions=3, max_lookahead=len(test_data), threshold=ts, lower_threshold=lts
)

# Print performance summary for max positions version
total_profit = sim_results['profit_usd'].sum()
win_rate = (sim_results['result'] == 'win').mean()
avg_return = sim_results['pnl_pct'].mean()

print(f"\nMax Positions Simulation Results (max {3} positions):")
print(f"------------------------------------------------")
print(f"Total Profit: ${total_profit:.2f}")
print(f"Win Rate: {win_rate:.2%}")
print(f"Average Return per Trade: {avg_return:.2%}")
print(f"Number of Trades: {len(sim_results)}")
print(f"Maximum Open Trades: {max_open_trades}")
print(f"Signals Ignored (due to position limit): {signals_ignored}")
print(f"Total Signals: {len(sim_results) + signals_ignored}")
print(f"Signal Utilization: {len(sim_results)/(len(sim_results) + signals_ignored):.2%}")

# Optional: Plot open trades over time
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(open_trades_history)
plt.axhline(y=3, color='r', linestyle='--', label=f'Max Positions ({3})')
plt.title('Number of Open Trades Over Time')
plt.xlabel('Time Step')
plt.ylabel('Number of Open Trades')
plt.legend()
plt.grid(True)
plt.show()
""" 

def simulate_trades2(data, preds, max_lookahead=60, threshold=0.02, spread=0.002, slippage=0.0005,
                    trade_size=1000, transaction_cost=5, lower_threshold=0.01):
    """
    Simulate both long and short trades based on model predictions.
    - 1 = go long
    - 0 = go short
    """
    close_prices = data['Close_raw'].values
    trades = []

    for i in range(len(preds)):
        direction = preds[i]  # 1 = long, 0 = short
        if direction not in [0, 1]:
            continue  # skip invalid predictions

        if direction == 1:
            # LONG TRADE
            entry_price = close_prices[i] * (1 + spread / 2 + slippage)
            result = None
            exit_price = None
            
            for j in range(1, max_lookahead + 1):
                if i + j >= len(close_prices):
                    break
                future_price = close_prices[i + j]
                pnl = (future_price - entry_price) / entry_price

                if pnl >= threshold:
                    exit_price = future_price * (1 - spread / 2 - slippage)
                    result = 'long_win'
                    break
                elif pnl <= -(threshold - lower_threshold):
                    exit_price = future_price * (1 - spread / 2 - slippage)
                    result = 'long_loss'
                    break
            
            # If no win/loss condition was hit, mark as 'end' and use final price
            if result is None:
                result = 'long_end'
                exit_price = close_prices[min(i + max_lookahead, len(close_prices) - 1)] * (1 - spread / 2 - slippage)

            final_return = (exit_price - entry_price) / entry_price
            profit = trade_size * final_return - transaction_cost

        else:
            # SHORT TRADE
            entry_price = close_prices[i] * (1 - spread / 2 - slippage)
            result = None
            exit_price = None
            
            for j in range(1, max_lookahead + 1):
                if i + j >= len(close_prices):
                    break
                future_price = close_prices[i + j]
                pnl = (entry_price - future_price) / entry_price

                if pnl >= (threshold - lower_threshold):
                    exit_price = future_price * (1 + spread / 2 + slippage)
                    result = 'short_win'
                    break
                elif pnl <= -threshold:
                    exit_price = future_price * (1 + spread / 2 + slippage)
                    result = 'short_loss'
                    break
            
            # If no win/loss condition was hit, mark as 'end' and use final price
            if result is None:
                result = 'short_end'
                exit_price = close_prices[min(i + max_lookahead, len(close_prices) - 1)] * (1 + spread / 2 + slippage)

            final_return = (entry_price - exit_price) / entry_price
            profit = trade_size * final_return - transaction_cost

        trades.append({
            'entry_index': i,
            'result': result,
            'pnl_pct': final_return,
            'profit_usd': profit
        })

    return pd.DataFrame(trades)



def simulate_trades(data, preds, max_lookahead=60, threshold=0.02, spread=0.002, slippage=0.0005, trade_size=1000,
                    transaction_cost=5, lower_threshold=0.01):
    """
    Simulate long trades when the model predicts a 1 (entry signal).
    Considers spread, slippage, and realistic exit logic.
    """
    close_prices = data['Close_raw'].values
    trades = []
    
    for i in range(len(preds)):
        if preds[i] != 1:
            continue  # Only simulate trades for predicted longs
        
        entry_price = close_prices[i] * (1 + spread / 2 + slippage)
        exit_price = None
        result = None

        for j in range(1, max_lookahead + 1):
            if i + j >= len(close_prices):
                break

            future_price = close_prices[i + j]
            pnl = (future_price - entry_price) / entry_price

            if pnl >= threshold:
                exit_price = future_price * (1 - spread / 2 - slippage)
                result = 'win'
                break
            elif pnl <= -(threshold - lower_threshold):
                exit_price = future_price * (1 - spread / 2 - slippage)
                result = 'loss'
                break
        
        # if result is not None:
        #     final_return = (exit_price - entry_price) / entry_price
        #     profit = trade_size * final_return - transaction_cost
        #     trades.append({
        #         'entry_index': i,
        #         'result': result,
        #         'pnl_pct': final_return,
        #         'profit_usd': profit
        #     })

        # If no win/loss condition was hit, mark as 'end' and use final price
        if result is None:
            result = 'end'
            exit_price = close_prices[min(i + max_lookahead, len(close_prices) - 1)] * (1 - spread / 2 - slippage)

        final_return = (exit_price - entry_price) / entry_price
        profit = trade_size * final_return - transaction_cost
        trades.append({
            'entry_index': i,
            'result': result,
            'pnl_pct': final_return,
            'profit_usd': profit
        })

    return pd.DataFrame(trades)