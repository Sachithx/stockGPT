import pandas as pd
from datetime import datetime, timedelta
from utils import load_data, calculate_performance_metrics, visualize_results



def backtest_strategy(data, initial_capital=100, position_size=0.25, commission=0.001,
                      profit_target=0.25, stop_loss=0.05, max_holding_days=30):
    """
    Simulate trading based on:
    - Buy signals from sentiment
    - Sell signals based on profit targets or stop losses

    Parameters:
    - data: DataFrame with price data and trading signals
    - initial_capital: Starting capital for the simulation
    - position_size: Percentage of capital to allocate per trade
    - commission: Commission rate per trade
    - profit_target: Target profit percentage for selling (e.g., 0.1 = 10%)
    - stop_loss: Stop loss percentage (e.g., 0.05 = 5%)
    - max_holding_days: Maximum days to hold a position before forced selling

    Returns:
    - performance: DataFrame with portfolio value over time
    - metrics: Dictionary with performance metrics
    """
    # Make a copy of the data
    df = data.copy()

    # Ensure we have price data (adjust column names if needed)
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    available_columns = [col for col in price_columns if col in df.columns]

    if not available_columns:
        raise ValueError("No price data columns found. Need at least one of: Open, High, Low, Close, Adj Close")

    # Use the first available price column
    price_col = available_columns[0]
    print(f"Using '{price_col}' column for price data")

    # Initialize portfolio tracking
    df['Position'] = 0  # Number of shares held
    df['Capital'] = initial_capital
    df['Portfolio_Value'] = initial_capital
    df['Trade_Type'] = ''  # Track reason for trades

    # Track trades
    trades = []

    # Simulate trading day by day
    position = 0  # Current position (number of shares)
    capital = initial_capital  # Current cash
    entry_price = 0  # Price at which we bought
    days_held = 0  # Number of days we've held the position
    waiting_period = 0  # Days to wait after selling before buying again

    for i in range(1, len(df)):
        yesterday = df.iloc[i-1]
        today = df.iloc[i]

        # Get today's price and sentiment signal
        price = today[price_col]
        buy_signal = today['Signal'] == 1

        # Calculate portfolio value (cash + shares)
        portfolio_value = capital + position * price

        # SELLING LOGIC: Check if we should sell based on profit target or stop loss
        if position > 0:
            days_held += 1
            current_return = (price / entry_price) - 1

            # Sell if profit target reached
            if current_return >= profit_target:
                # Calculate trade value
                trade_value = position * price
                commission_cost = trade_value * commission

                # Execute sell
                capital += (trade_value - commission_cost)

                # Record trade
                trades.append({
                    'Date': today['Date'],
                    'Action': 'SELL',
                    'Reason': 'PROFIT_TARGET',
                    'Price': price,
                    'Entry_Price': entry_price,
                    'Return_Pct': current_return * 100,
                    'Shares': position,
                    'Value': trade_value,
                    'Commission': commission_cost,
                    'Days_Held': days_held,
                    'Portfolio_Value': portfolio_value
                })

                # Reset position
                position = 0
                entry_price = 0
                days_held = 0
                waiting_period = 5  # Wait 5 days before considering buying again
                df.loc[df.index[i], 'Trade_Type'] = 'PROFIT_TAKE'

            # Sell if stop loss triggered
            elif current_return <= -stop_loss:
                # Calculate trade value
                trade_value = position * price
                commission_cost = trade_value * commission

                # Execute sell
                capital += (trade_value - commission_cost)

                # Record trade
                trades.append({
                    'Date': today['Date'],
                    'Action': 'SELL',
                    'Reason': 'STOP_LOSS',
                    'Price': price,
                    'Entry_Price': entry_price,
                    'Return_Pct': current_return * 100,
                    'Shares': position,
                    'Value': trade_value,
                    'Commission': commission_cost,
                    'Days_Held': days_held,
                    'Portfolio_Value': portfolio_value
                })

                # Reset position
                position = 0
                entry_price = 0
                days_held = 0
                waiting_period = 10  # Wait longer after a stop loss (10 days)
                df.loc[df.index[i], 'Trade_Type'] = 'STOP_LOSS'

            # Sell if maximum holding period reached
            elif days_held >= max_holding_days:
                # Calculate trade value
                trade_value = position * price
                commission_cost = trade_value * commission

                # Execute sell
                capital += (trade_value - commission_cost)

                # Record trade
                trades.append({
                    'Date': today['Date'],
                    'Action': 'SELL',
                    'Reason': 'MAX_HOLDING',
                    'Price': price,
                    'Entry_Price': entry_price,
                    'Return_Pct': current_return * 100,
                    'Shares': position,
                    'Value': trade_value,
                    'Commission': commission_cost,
                    'Days_Held': days_held,
                    'Portfolio_Value': portfolio_value
                })

                # Reset position
                position = 0
                entry_price = 0
                days_held = 0
                waiting_period = 3  # Short waiting period
                df.loc[df.index[i], 'Trade_Type'] = 'MAX_HOLDING'

        # BUYING LOGIC: Check if we should buy based on sentiment signal
        elif position == 0 and waiting_period <= 0:
            if buy_signal:
                # Calculate position size
                trade_value = portfolio_value * position_size
                shares_to_buy = trade_value / price

                # Account for commission
                commission_cost = trade_value * commission

                if trade_value + commission_cost <= capital:
                    # Execute buy
                    position = shares_to_buy
                    entry_price = price
                    capital -= (trade_value + commission_cost)
                    days_held = 0

                    # Record trade
                    trades.append({
                        'Date': today['Date'],
                        'Action': 'BUY',
                        'Reason': 'SENTIMENT',
                        'Price': price,
                        'Entry_Price': entry_price,
                        'Return_Pct': 0,
                        'Shares': shares_to_buy,
                        'Value': trade_value,
                        'Commission': commission_cost,
                        'Days_Held': 0,
                        'Portfolio_Value': portfolio_value
                    })

                    df.loc[df.index[i], 'Trade_Type'] = 'BUY'
        else:
            # Decrease waiting period counter
            waiting_period = max(0, waiting_period - 1)

        # Update tracking
        df.loc[df.index[i], 'Position'] = position
        df.loc[df.index[i], 'Capital'] = capital
        df.loc[df.index[i], 'Portfolio_Value'] = capital + position * price

    # Create a trades dataframe
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # Calculate performance metrics
    metrics = calculate_performance_metrics(df, initial_capital, trades_df)

    return df, trades_df, metrics


# Main function to run the entire analysis
def run_sentiment_trading_simulation(sentiment_file, stock_price_file, strategy='threshold',
                                     threshold=0.5, initial_capital=100,
                                     profit_target=0.25, stop_loss=0.05, max_holding_days=30):
    """
    Run a complete sentiment-based trading simulation with profit targets.

    Parameters:
    - sentiment_file: Path to the sentiment analysis results
    - stock_price_file: Path to the stock price historical data
    - strategy: Trading strategy to use ('threshold', 'relative', or 'trend')
    - threshold: Sentiment threshold for buy signals
    - initial_capital: Starting capital for the simulation
    - profit_target: Target profit percentage for selling (e.g., 0.1 = 10%)
    - stop_loss: Stop loss percentage (e.g., 0.05 = 5%)
    - max_holding_days: Maximum days to hold a position
    """
    print("=" * 80)
    print(f"RUNNING SENTIMENT-BASED TRADING SIMULATION WITH PROFIT TARGETS")
    print(f"Strategy: {strategy}, Sentiment Threshold: {threshold}, Initial Capital: ${initial_capital}")
    print(f"Profit Target: {profit_target*100:.1f}%, Stop Loss: {stop_loss*100:.1f}%, Max Holding: {max_holding_days} days")
    print("=" * 80)

    # Step 1: Load and merge data
    merged_data = load_data(sentiment_file, stock_price_file)

    # Step 2: Generate trading signals (only buy signals based on sentiment)
    signals_data = generate_trading_signals(merged_data, strategy, threshold, profit_target, stop_loss)

    # Step 3: Run backtest simulation with profit targets
    performance, trades, metrics = backtest_strategy(
        signals_data,
        initial_capital=initial_capital,
        profit_target=profit_target,
        stop_loss=stop_loss,
        max_holding_days=max_holding_days
    )

    # Step 4: Print performance metrics
    print("\nPERFORMANCE METRICS:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Step 5: Visualize results
    strategy_name = f"{strategy.capitalize()} Buy / {profit_target*100:.0f}% Profit Target"
    visualize_results(signals_data, performance, trades, metrics, strategy_name)

    # Step 6: Analyze trade types
    if not trades.empty and 'Reason' in trades.columns:
        reason_counts = trades[trades['Action'] == 'SELL']['Reason'].value_counts()
        print("\nSELL TRADE ANALYSIS:")
        print("-" * 40)
        total_sells = len(trades[trades['Action'] == 'SELL'])
        for reason, count in reason_counts.items():
            print(f"{reason}: {count} trades ({count/total_sells*100:.1f}%)")

    # Step 7: Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save signals data
    signals_data.to_csv(f"trading_signals_{strategy}_profit{int(profit_target*100)}_{timestamp}.csv", index=False)

    # Save performance data
    performance.to_csv(f"trading_performance_{strategy}_profit{int(profit_target*100)}_{timestamp}.csv", index=False)

    # Save trades data
    if not trades.empty:
        trades.to_csv(f"trading_trades_{strategy}_profit{int(profit_target*100)}_{timestamp}.csv", index=False)

    print(f"\nDetailed results saved with timestamp {timestamp}")

    return performance, trades, metrics