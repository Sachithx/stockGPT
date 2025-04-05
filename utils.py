import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from constants import SENTIMENT_FILE, STOCK_PRICE_FILE

# Create trading signals based on sentiment (for buy) and profit targets (for sell)
def generate_trading_signals(data, strategy='threshold', threshold=0.5, profit_target=0.25, stop_loss=0.05):
    """
    Generate buy signals based on sentiment data.
    Sell signals are determined by profit targets, not sentiment.

    Parameters:
    - data: DataFrame with sentiment and price data
    - strategy: Strategy to use ('threshold', 'relative', or 'trend')
    - threshold: Sentiment threshold for buy decisions
    - profit_target: Target profit percentage for selling (e.g., 0.1 = 10%)
    - stop_loss: Optional stop loss percentage (e.g., 0.05 = 5%)

    Returns:
    - data: DataFrame with added signal column
    """
    # Make a copy to avoid modifying the original
    df = data.copy()

    # Initialize signal column (0 = hold, 1 = buy, -1 = sell)
    df['Signal'] = 0

    # Generate BUY signals based on sentiment
    if strategy == 'threshold':
        # Buy when positive sentiment exceeds threshold
        df.loc[df['Positive_Ratio'] >= threshold, 'Signal'] = 1

    elif strategy == 'relative':
        # Compare positive vs negative sentiment
        # Buy when positive significantly exceeds negative
        df['Sentiment_Difference'] = df['Positive_Ratio'] - df['Negative_Ratio']
        df.loc[df['Sentiment_Difference'] >= threshold, 'Signal'] = 1

    elif strategy == 'trend':
        # Look at sentiment trends over time
        window = 3  # Consider last 3 days

        # Calculate moving averages of sentiment
        df['Positive_MA'] = df['Positive_Ratio'].rolling(window=window).mean()

        # Generate signals based on trend
        df['Positive_Trend'] = df['Positive_Ratio'] > df['Positive_MA']

        # Buy when positive sentiment is trending up and above threshold
        df.loc[(df['Positive_Trend']) & (df['Positive_Ratio'] > threshold), 'Signal'] = 1

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Count the number of signals generated
    buy_signals = (df['Signal'] == 1).sum()
    hold_signals = (df['Signal'] == 0).sum()

    print(f"Strategy '{strategy}' generated:")
    print(f"- Buy signals: {buy_signals} ({buy_signals/len(df)*100:.1f}%)")
    print(f"- Hold signals: {hold_signals} ({hold_signals/len(df)*100:.1f}%)")
    print(f"- Sell signals will be generated based on profit targets during simulation")

    # Note: We don't generate sell signals here - they'll be determined dynamically
    # during the backtest based on profit targets

    return df

def load_data(sentiment_file=SENTIMENT_FILE, stock_price_file=STOCK_PRICE_FILE):
    """
    Load and prepare both sentiment and stock price data.

    Parameters:
    - sentiment_file: Path to the sentiment analysis results (daily aggregation)
    - stock_price_file: Path to the stock price historical data

    Returns:
    - merged_data: A dataframe with both sentiment and price data
    """
    print(f"Loading sentiment data from {sentiment_file}")

    try:
        sentiment_df = pd.read_csv(sentiment_file)

        # Ensure date is in datetime format
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

        print(f"Successfully loaded sentiment data with {len(sentiment_df)} rows")
        print(f"Sentiment data columns: {sentiment_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading sentiment data: {e}")
        # Check if the file exists
        import os
        if not os.path.exists(sentiment_file):
            print(f"The file {sentiment_file} does not exist.")
            print("Available files in current directory:")
            print(os.listdir('.'))
        raise

    print(f"Loading stock price data from {stock_price_file}")
    try:
        prices_df = pd.read_csv(stock_price_file)

        # Ensure date is in datetime format
        date_col = None
        possible_date_cols = ['Date', 'date', 'DATE', 'Timestamp', 'timestamp', 'Time', 'time']

        for col in possible_date_cols:
            if col in prices_df.columns:
                date_col = col
                break

        if date_col is None:
            print(f"Warning: Could not find date column. Available columns: {prices_df.columns.tolist()}")
            # Try to use the first column as date
            date_col = prices_df.columns[0]
            print(f"Using {date_col} as date column")

        prices_df['Date'] = pd.to_datetime(prices_df[date_col])

        # If we used a different column for the date, make sure we keep that original column
        if date_col != 'Date':
            prices_df[date_col] = prices_df['Date']

        print(f"Successfully loaded price data with {len(prices_df)} rows")
        print(f"Price data columns: {prices_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading stock price data: {e}")
        # Check if the file exists
        import os
        if not os.path.exists(stock_price_file):
            print(f"The file {stock_price_file} does not exist.")
            print("Available files in current directory:")
            print(os.listdir('.'))
        raise

    # Check for required price columns
    required_price_cols = ['Open', 'High', 'Low', 'Close']
    available_price_cols = [col for col in required_price_cols if col in prices_df.columns]

    if not available_price_cols:
        print(f"Warning: No standard price columns found (Open, High, Low, Close)")
        print(f"Available columns: {prices_df.columns.tolist()}")

        # Try to guess price columns based on common patterns
        possible_renames = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adj close': 'Adj Close',
            'adj_close': 'Adj Close',
            'adjusted close': 'Adj Close',
            'adjusted_close': 'Adj Close'
        }

        for old, new in possible_renames.items():
            for col in prices_df.columns:
                if old.lower() in col.lower():
                    prices_df[new] = prices_df[col]
                    print(f"Using {col} as {new}")

    # Merge sentiment data with stock price data
    print(f"Merging data on Date column")
    merged_data = pd.merge(sentiment_df, prices_df, on='Date', how='inner')

    if len(merged_data) == 0:
        print("WARNING: Merged data is empty! Check date formats and ranges.")
        print(f"Sentiment data date range: {sentiment_df['Date'].min()} to {sentiment_df['Date'].max()}")
        print(f"Price data date range: {prices_df['Date'].min()} to {prices_df['Date'].max()}")

        # Try creating a sample of merged data for debugging
        print("\nDate format comparison:")
        print(f"Sentiment date example: {sentiment_df['Date'].iloc[0]} (type: {type(sentiment_df['Date'].iloc[0])})")
        print(f"Price date example: {prices_df['Date'].iloc[0]} (type: {type(prices_df['Date'].iloc[0])})")

        raise ValueError("Merged data is empty. Unable to continue.")

    print(f"Data merged successfully. Found {len(merged_data)} days with both sentiment and price data.")
    return merged_data



def print_signal_report(report):
    """
    Print a nicely formatted version of the signal report.
    
    Parameters:
    - report: The report dictionary returned by generate_signal_report
    """
    if report["status"] == "error":
        print(f"\n❌ ERROR: {report['message']}")
        return
    
    print("\n" + "="*70)
    print(f"📊 SENTIMENT-BASED TRADING SIGNAL REPORT")
    print(f"⏱️  Generated at: {report['report_generated_at']}")
    print("="*70)
    
    # Print parameters
    params = report["parameters"]
    print(f"\n🔧 STRATEGY PARAMETERS:")
    print(f"  • Sentiment threshold: {params['sentiment_threshold']*100:.1f}%")
    print(f"  • Profit target: {params['profit_target']*100:.1f}%")
    print(f"  • Stop loss: {params['stop_loss']*100:.1f}%")
    print(f"  • Maximum holding period: {params['max_holding_days']} days")
    
    # Print summary if available
    if "summary" in report:
        summary = report["summary"]
        print(f"\n📅 PERIOD: {summary['period']} ({summary['total_days']} days)")
        print(f"  • Buy signals: {summary['buy_signals']} days ({summary['buy_signals']/summary['total_days']*100:.1f}%)")
        print(f"  • Hold signals: {summary['hold_signals']} days ({summary['hold_signals']/summary['total_days']*100:.1f}%)")
        print(f"  • Average positive sentiment: {summary['avg_positive_sentiment']*100:.1f}%")
        print(f"  • Average negative sentiment: {summary['avg_negative_sentiment']*100:.1f}%")
    
    # Print each day's data
    print("\n📈 DAILY SIGNALS:")
    for day in report["data"]:
        date = day["date"]
        signal = day["signal"]
        signal_emoji = "🟢" if signal == "BUY" else "⚪"
        
        print(f"\n  {signal_emoji} {date} - SIGNAL: {signal}")
        print(f"    • Sentiment: +{day['sentiment']['positive_ratio']*100:.1f}% | -{day['sentiment']['negative_ratio']*100:.1f}% | "
              f"Neutral: {day['sentiment']['neutral_ratio']*100:.1f}%")
        print(f"    • Reason: {day['signal_explanation']}")
        
        # Print trading guidelines if it's a BUY signal
        if signal == "BUY" and "trading_guidelines" in day:
            guide = day["trading_guidelines"]
            print(f"    • Entry price: ${guide['entry_price']:.2f}")
            print(f"    • Profit target: ${guide['profit_target_price']:.2f} (+{guide['profit_target_percentage']})")
            print(f"    • Stop loss: ${guide['stop_loss_price']:.2f} (-{guide['stop_loss_percentage']})")
            print(f"    • Maximum hold time: {guide['max_holding_period']}")
    
    print("\n" + "="*70)
    
# Calculate performance metrics for the strategy
def calculate_performance_metrics(performance_data, initial_capital, trades_df):
    """
    Calculate key performance metrics for the trading strategy.

    Parameters:
    - performance_data: DataFrame with portfolio value over time
    - initial_capital: Starting capital amount
    - trades_df: DataFrame with trade details

    Returns:
    - metrics: Dictionary with performance metrics
    """
    metrics = {}

    # Extract performance data
    df = performance_data.copy()

    # Calculate basic metrics
    final_value = df['Portfolio_Value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # Calculate buy & hold return for comparison
    first_price = df.iloc[0]['Close'] if 'Close' in df.columns else df.iloc[0]['Adj Close'] if 'Adj Close' in df.columns else None
    last_price = df.iloc[-1]['Close'] if 'Close' in df.columns else df.iloc[-1]['Adj Close'] if 'Adj Close' in df.columns else None

    if first_price is not None and last_price is not None:
        buy_hold_return = (last_price - first_price) / first_price
    else:
        buy_hold_return = None

    # Calculate annualized return
    days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    years = days / 365.25
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else None

    # Calculate volatility (standard deviation of daily returns)
    df['Daily_Return'] = df['Portfolio_Value'].pct_change()
    volatility = df['Daily_Return'].std() * (252 ** 0.5)  # Annualized

    # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else None

    # Maximum drawdown
    df['Cumulative_Max'] = df['Portfolio_Value'].cummax()
    df['Drawdown'] = (df['Portfolio_Value'] - df['Cumulative_Max']) / df['Cumulative_Max']
    max_drawdown = df['Drawdown'].min()

    # Trade statistics
    num_trades = len(trades_df)

    if num_trades > 0:
        winning_trades = trades_df[trades_df['Action'] == 'SELL'].copy()
        if not winning_trades.empty:
            # For each sell, find the corresponding buy
            buy_prices = []
            for idx, sell_trade in winning_trades.iterrows():
                # Find the most recent buy before this sell
                buy_trades = trades_df[(trades_df['Action'] == 'BUY') &
                                      (trades_df['Date'] < sell_trade['Date'])]
                if not buy_trades.empty:
                    most_recent_buy = buy_trades.iloc[-1]
                    buy_prices.append(most_recent_buy['Price'])
                else:
                    buy_prices.append(None)

            winning_trades['Buy_Price'] = buy_prices
            winning_trades = winning_trades.dropna(subset=['Buy_Price'])

            if not winning_trades.empty:
                winning_trades['Profit'] = (winning_trades['Price'] - winning_trades['Buy_Price']) * winning_trades['Shares']
                winning_trades['Profit'] -= winning_trades['Commission']  # Account for commission

                win_count = (winning_trades['Profit'] > 0).sum()
                loss_count = (winning_trades['Profit'] <= 0).sum()

                win_rate = win_count / len(winning_trades) if len(winning_trades) > 0 else 0

                avg_profit = winning_trades[winning_trades['Profit'] > 0]['Profit'].mean() if win_count > 0 else 0
                avg_loss = winning_trades[winning_trades['Profit'] <= 0]['Profit'].mean() if loss_count > 0 else 0

                profit_factor = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
            else:
                win_rate = 0
                avg_profit = 0
                avg_loss = 0
                profit_factor = 0
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
    else:
        win_rate = 0
        avg_profit = 0
        avg_loss = 0
        profit_factor = 0

    # Combine all metrics
    metrics = {
        'Initial Capital': initial_capital,
        'Final Portfolio Value': final_value,
        'Total Return (%)': total_return * 100,
        'Buy & Hold Return (%)': buy_hold_return * 100 if buy_hold_return is not None else None,
        'Strategy Outperformance (%)': (total_return - buy_hold_return) * 100 if buy_hold_return is not None else None,
        'Annualized Return (%)': annualized_return * 100 if annualized_return is not None else None,
        'Volatility (%)': volatility * 100,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Number of Trades': num_trades,
        'Win Rate (%)': win_rate * 100,
        'Avg Profit ($)': avg_profit,
        'Avg Loss ($)': avg_loss,
        'Profit Factor': profit_factor
    }

    return metrics

# Visualize results
def visualize_results(data, performance, trades_df, metrics, strategy_name):
    """
    Create visualizations of trading performance.

    Parameters:
    - data: Original data with sentiment and signals
    - performance: Portfolio performance data
    - trades_df: DataFrame with trade details
    - metrics: Dictionary with performance metrics
    - strategy_name: Name of the strategy for plot titles
    """
    try:
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Plot 1: Stock price with buy/sell signals
        ax1 = axes[0]

        # Plot the stock price
        price_col = 'Close' if 'Close' in data.columns else 'Adj Close' if 'Adj Close' in data.columns else None

        if price_col:
            ax1.plot(data['Date'], data[price_col], label='Stock Price', color='blue')

            # Find buy and sell points from the trades dataframe instead of signals
            if trades_df is not None and not trades_df.empty:
                # Add buy signals
                buy_trades = trades_df[trades_df['Action'] == 'BUY']
                if not buy_trades.empty:
                    ax1.scatter(buy_trades['Date'], buy_trades['Price'],
                               color='green', label='Buy', marker='^', s=100)

                # Add sell signals
                sell_trades = trades_df[trades_df['Action'] == 'SELL']
                if not sell_trades.empty:
                    ax1.scatter(sell_trades['Date'], sell_trades['Price'],
                               color='red', label='Sell', marker='v', s=100)

            ax1.set_title(f'Stock Price with {strategy_name} Trading Signals', fontsize=14)
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Plot 2: Portfolio value over time
        ax2 = axes[1]
        ax2.plot(performance['Date'], performance['Portfolio_Value'], label='Portfolio Value', color='purple')

        # Add profit target and stop loss references if available
        if 'Entry_Price' in trades_df.columns and price_col:
            for _, trade in trades_df.iterrows():
                if trade['Action'] == 'BUY':
                    # Plot profit target level
                    if 'profit_target' in metrics:
                        profit_level = trade['Price'] * (1 + metrics['profit_target'])
                        ax1.axhline(y=profit_level, color='green', linestyle=':', alpha=0.3)

                    # Plot stop loss level
                    if 'stop_loss' in metrics:
                        stop_level = trade['Price'] * (1 - metrics['stop_loss'])
                        ax1.axhline(y=stop_level, color='red', linestyle=':', alpha=0.3)

        ax2.set_title('Portfolio Value Over Time', fontsize=14)
        ax2.set_ylabel('Value ($)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Sentiment over time
        ax3 = axes[2]
        ax3.plot(data['Date'], data['Positive_Ratio'], label='Positive Sentiment', color='green')
        ax3.plot(data['Date'], data['Negative_Ratio'], label='Negative Sentiment', color='red')

        # Add a threshold line for buy signals
        if 'threshold' in strategy_name.lower():
            # Extract the threshold value from the strategy name or use default
            threshold = 0.4  # Default

            # Try to extract from strategy name if available
            threshold_str = strategy_name.split('threshold=')
            if len(threshold_str) > 1:
                try:
                    threshold = float(threshold_str[1].split(')')[0])
                except:
                    pass

            ax3.axhline(y=threshold, color='black', linestyle='--', alpha=0.5, label=f'Buy Threshold ({threshold})')

        ax3.set_title('Sentiment Ratios Over Time', fontsize=14)
        ax3.set_xlabel('Date', fontsize=8)
        ax3.set_ylabel('Sentiment Ratio', fontsize=8)
        ax3.legend()
        ax3.grid(True, alpha=0.7)

        # Add metrics as text
        plt.figtext(0.01, 0.01, format_metrics(metrics), fontsize=6,
                   bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom')

        plt.tight_layout()

        # Try to save the figure in a safe way
        try:
            # Safe filename conversion
            safe_name = ''.join([c if c.isalnum() else '_' for c in strategy_name])
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"trading_results_{safe_name}_{timestamp}.png"

            # Try different paths in case of permission issues
            try_paths = [
                filename,  # Current directory
                f"/tmp/{filename}",  # Temp directory
                f"/content/{filename}",  # Google Colab directory
            ]

            saved = False
            for path in try_paths:
                try:
                    plt.savefig(path, dpi=300, bbox_inches='tight')
                    print(f"Visualization saved as {path}")
                    saved = True
                    break
                except Exception as e:
                    print(f"Could not save to {path}: {e}")

            if not saved:
                print("Warning: Could not save visualization to file. Displaying only.")

        except Exception as e:
            print(f"Warning: Could not save visualization: {e}")

        # Display the figure
        plt.show()

    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
        print("Continuing without visualization...")


# Format metrics for display
def format_metrics(metrics):
    """Format the metrics dictionary as a readable string."""
    text = "Performance Metrics:\n\n"
    for key, value in metrics.items():
        if isinstance(value, float):
            text += f"{key}: {value:.2f}\n"
        else:
            text += f"{key}: {value}\n"
    return text
