from utils import generate_trading_signals
from utils import load_data
import json
from openai import OpenAI

def generate_signal_report(sentiment_file, stock_price_file, date=None, days_back=7, 
                          threshold=0.4, profit_target=0.25, stop_loss=0.03, max_holding_days=10):
    """
    Generate a structured report about trading signals for a specific date or the last N days.
    
    Parameters:
    - sentiment_file: Path to the sentiment analysis results
    - stock_price_file: Path to the stock price historical data
    - date: Specific date to report on (format: 'YYYY-MM-DD' or datetime object)
            If None, will use the most recent date in the data
    - days_back: Number of days to include in the report if showing historical data
    - threshold: Sentiment threshold used for buy signals
    - profit_target: Target profit percentage for selling
    - stop_loss: Stop loss percentage
    - max_holding_days: Maximum days to hold a position
    
    Returns:
    - report: Dictionary with structured report information
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Load and merge the data
    try:
        merged_data = load_data(sentiment_file, stock_price_file)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to load data: {str(e)}"
        }
    
    # Generate trading signals (we only need this to check which days would trigger buys)
    signals_data = generate_trading_signals(
        merged_data, 
        strategy='threshold', 
        threshold=threshold,
        profit_target=profit_target,
        stop_loss=stop_loss
    )
    
    # Convert date parameter to datetime if provided as string
    if date is not None and isinstance(date, str):
        try:
            date = pd.to_datetime(date)
        except:
            return {
                "status": "error",
                "message": f"Invalid date format. Please use 'YYYY-MM-DD' format."
            }
    
    # If no date specified, use the most recent date in the data
    if date is None:
        date = signals_data['Date'].max()
        
    # Filter data for the requested date or date range
    if days_back > 1:
        # Calculate the start date based on days_back
        start_date = date - timedelta(days=days_back-1)
        
        # Filter data for the date range
        report_data = signals_data[
            (signals_data['Date'] >= start_date) & 
            (signals_data['Date'] <= date)
        ].copy()
        
        # Sort by date
        report_data = report_data.sort_values('Date')
    else:
        # Filter for just the single date
        report_data = signals_data[signals_data['Date'] == date].copy()
    
    # Check if we have data for the requested date(s)
    if len(report_data) == 0:
        return {
            "status": "error",
            "message": f"No data available for the specified date range."
        }
    
    # Create the report structure
    report = {
        "status": "success",
        "report_generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "sentiment_threshold": threshold,
            "profit_target": profit_target,
            "stop_loss": stop_loss,
            "max_holding_days": max_holding_days
        },
        "data": []
    }
    
    # Extract key price columns if available
    price_col = 'Close' if 'Close' in report_data.columns else 'Adj Close' if 'Adj Close' in report_data.columns else None
    
    # Process each date in the report data
    for _, row in report_data.iterrows():
        date_str = row['Date'].strftime("%Y-%m-%d")
        
        # Determine signal
        signal = "BUY" if row['Signal'] == 1 else "HOLD"
        
        # Create the day's report
        day_report = {
            "date": date_str,
            "sentiment": {
                "positive_ratio": row['Positive_Ratio'],
                "negative_ratio": row['Negative_Ratio'],
                "neutral_ratio": 1 - row['Positive_Ratio'] - row['Negative_Ratio'] if 'Neutral_Ratio' not in row else row['Neutral_Ratio']
            },
            "signal": signal,
            "signal_explanation": f"{'Positive sentiment' if signal == 'BUY' else 'Sentiment'} is {row['Positive_Ratio']*100:.1f}% which is {'above' if signal == 'BUY' else 'below'} the {threshold*100:.1f}% threshold"
        }
        
        # Add price information if available
        if price_col:
            day_report["price"] = {
                "current": row[price_col],
                "profit_target": row[price_col] * (1 + profit_target) if signal == "BUY" else None,
                "stop_loss": row[price_col] * (1 - stop_loss) if signal == "BUY" else None
            }
        
        # Add trading guidelines if BUY signal
        if signal == "BUY":
            day_report["trading_guidelines"] = {
                "entry_price": row[price_col] if price_col else "N/A",
                "profit_target_price": row[price_col] * (1 + profit_target) if price_col else "N/A",
                "profit_target_percentage": f"{profit_target*100:.1f}%",
                "stop_loss_price": row[price_col] * (1 - stop_loss) if price_col else "N/A",
                "stop_loss_percentage": f"{stop_loss*100:.1f}%",
                "max_holding_period": f"{max_holding_days} days"
            }
        
        # Add to the report
        report["data"].append(day_report)
    
    # Add summary information if we have multiple days
    if len(report["data"]) > 1:
        num_buy_signals = sum(1 for day in report["data"] if day["signal"] == "BUY")
        report["summary"] = {
            "period": f"{report['data'][0]['date']} to {report['data'][-1]['date']}",
            "total_days": len(report["data"]),
            "buy_signals": num_buy_signals,
            "hold_signals": len(report["data"]) - num_buy_signals,
            "avg_positive_sentiment": sum(day["sentiment"]["positive_ratio"] for day in report["data"]) / len(report["data"]),
            "avg_negative_sentiment": sum(day["sentiment"]["negative_ratio"] for day in report["data"]) / len(report["data"])
        }
    
    return report


def get_llm_recommendation(user_question, signal_report, api_key):
    """
    Send the trading signal report and user question to OpenAI LLM and get trading advice.
    
    Parameters:
    - user_question: The question or message from the user
    - signal_report: The dictionary returned by generate_signal_report
    - api_key: Your OpenAI API key
    
    Returns:
    - LLM's response with trading recommendation
    """
    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=api_key)
    
    # Convert the report to a JSON string for inclusion in the prompt
    report_json = json.dumps(signal_report, indent=2)
    
    # Create the system message that defines the assistant's role and provides context
    system_message = """
You are StockGPT, a specialized financial assistant that provides trading recommendations based on sentiment analysis of social media data.

You have access to a sentiment-based trading signal report for Tesla stock that includes:
1. Sentiment metrics (positive, negative, neutral percentages)
2. Trading signals (BUY or HOLD) based on optimized parameters
3. Trading guidelines for execution (entry prices, profit targets, stop losses)

Your task is to help the user understand what trading decision they should make today based on this data.

Key rules for your responses:
- Be clear and direct about the recommended action (BUY, HOLD, or SELL if already in a position)
- Explain the rationale behind the recommendation using the sentiment data
- If a BUY signal is present, provide specific entry, profit target, and stop loss levels
- Keep responses conversational but focused on actionable advice
- Never recommend anything outside what the data supports
- Consider both sentiment trends and absolute levels when advising
- If the data is insufficient or outdated, be transparent about limitations
"""

    # Format user prompt to include both their question and the report data
    user_prompt = f"""
USER QUESTION: {user_question}

SENTIMENT TRADING REPORT:
{report_json}

Please analyze this report and provide a clear trading recommendation that answers the user's question. Focus on what action they should take today based on the most recent data available in the report.
"""

    # Send the request to the OpenAI API using the new client interface
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.4,  # Lower temperature for more focused/predictable responses
            max_tokens=500,   # Adjust based on how detailed you want the responses
        )
        
        # Extract the assistant's message (new response format)
        return {
            "status": "success",
            "recommendation": response.choices[0].message.content
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting recommendation: {str(e)}"
        }
    

# Example of how to integrate this with your chatbot
def process_user_message(user_message, sentiment_file, stock_price_file, openai_api_key):
    """
    Process a user message in the chatbot and provide trading recommendations.
    
    Parameters:
    - user_message: The message from the user
    - sentiment_file: Path to the sentiment data file
    - stock_price_file: Path to the stock price data file
    - openai_api_key: Your OpenAI API key
    
    Returns:
    - Response to send back to the user
    """
    # Generate the signal report (using the most recent data by default)
    signal_report = generate_signal_report(
        sentiment_file=sentiment_file,
        stock_price_file=stock_price_file
    )
    
    # Handle any errors in generating the report
    if signal_report["status"] == "error":
        return f"I'm sorry, I couldn't analyze the trading signals: {signal_report['message']}"
    
    # Get recommendation from LLM
    llm_response = get_llm_recommendation(
        user_question=user_message,
        signal_report=signal_report,
        api_key=openai_api_key
    )
    
    # Handle any errors in getting the recommendation
    if llm_response["status"] == "error":
        return f"I'm sorry, I had trouble processing your request: {llm_response['message']}"
    
    # Return the recommendation
    return llm_response["recommendation"]

# Example usage in a chatbot
"""
user_message = "Should I buy Tesla stock today based on Twitter sentiment?"
response = process_user_message(
    user_message=user_message,
    sentiment_file="tesla_daily_sentiment.csv",
    stock_price_file="TSLA_historical_prices.csv",
    openai_api_key="your-api-key-here"
)
print(response)
"""