import openai
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import json
from flask import Flask, request, jsonify, render_template
constants import OPENAI_API_KEY
# OpenAI API configuration
openai.api_key = OPENAI_API_KEY  # Replace with your actual API key

# Load your optimized parameters
OPTIMAL_PARAMS = {
    'threshold': 0.4,  # Sentiment threshold for buying
    'max_holding_days': 10,  # Maximum days to hold a position
    'profit_target': 0.25,  # Target profit percentage
    'stop_loss': 0.03  # Stop loss percentage
}

# File paths for data
SENTIMENT_FILE = "tesla_daily_sentiment_latest.csv"
STOCK_PRICE_FILE = "TSLA_historical_prices.csv"

# Flask application
app = Flask(__name__)


# ====== SENTIMENT ANALYSIS FUNCTIONS ======

def analyze_new_tweets(tweets):
    """
    Analyze new tweets for sentiment using GPT-3.5.

    Parameters:
    - tweets: List of tweet texts to analyze

    Returns:
    - List of sentiment predictions (Positive, Negative, Neutral)
    """

    # Function to sanitize tweets for better processing
    def sanitize_tweet(tweet):
        if not isinstance(tweet, str):
            return ""

        # Replace problematic characters
        sanitized = (str(tweet)
                     .replace('\\', '\\\\')
                     .replace('"', '\\"')
                     .replace('\n', ' ')
                     .replace('\r', ' ')
                     .replace('\t', ' ')
                     .replace('\b', ' '))

        # Truncate overly long tweets
        if len(sanitized) > 280:
            sanitized = sanitized[:277] + "..."

        return sanitized

    # Sanitize tweets in this batch
    sanitized_tweets = [sanitize_tweet(tweet) for tweet in tweets]

    # Create a numbered batch for the model
    numbered_tweets = [f"Tweet {j + 1}: {tweet}" for j, tweet in enumerate(sanitized_tweets)]
    tweets_text = "\n".join(numbered_tweets)

    # Create the prompt
    system_prompt = "You are a financial sentiment analyst who classifies tweets about Tesla and its stock as Positive, Negative, or Neutral."

    user_prompt = f"""Analyze the sentiment of each of these Tesla-related tweets.
For each tweet, determine if the sentiment toward Tesla or its stock is Positive, Negative, or Neutral.

{tweets_text}

For EACH tweet, respond with ONLY a number and sentiment classification in this EXACT format:
1: Positive
2: Negative
3: Neutral

Provide one line per tweet, numbered exactly as above.
"""

    # Get response from OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=1024
    )

    content = response['choices'][0]['message']['content']

    # Parse the response
    sentiment_pattern = r'^(\d+): (Positive|Negative|Neutral)$'

    results = []
    import re
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue

        match = re.match(sentiment_pattern, line)
        if match:
            tweet_idx = int(match.group(1)) - 1
            if 0 <= tweet_idx < len(tweets):
                results.append({
                    "Tweet": tweets[tweet_idx],
                    "Sentiment": match.group(2)
                })

    return results


def get_daily_sentiment_summary(date=None):
    """
    Get the daily sentiment summary for a specific date.
    If date is None, return the most recent date.

    Parameters:
    - date: Date to get sentiment for (optional)

    Returns:
    - Dictionary with sentiment summary
    """
    try:
        # Load the sentiment data
        sentiment_df = pd.read_csv(SENTIMENT_FILE)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

        # If no date specified, use the most recent date
        if date is None:
            date = sentiment_df['Date'].max()
        else:
            date = pd.to_datetime(date)

        # Filter for the specified date
        daily_data = sentiment_df[sentiment_df['Date'] == date]

        if len(daily_data) == 0:
            return {
                "date": date.strftime("%Y-%m-%d"),
                "status": "error",
                "message": "No sentiment data available for this date."
            }

        # Get the sentiment metrics
        positive_ratio = daily_data['Positive_Ratio'].values[0]
        negative_ratio = daily_data['Negative_Ratio'].values[0]
        tweet_count = daily_data['Tweet_Count'].values[0]

        # Determine the signal based on our optimal threshold
        signal = "BUY" if positive_ratio >= OPTIMAL_PARAMS['threshold'] else "HOLD"

        # Return the summary
        return {
            "date": date.strftime("%Y-%m-%d"),
            "status": "success",
            "sentiment_summary": {
                "positive_ratio": positive_ratio,
                "negative_ratio": negative_ratio,
                "neutral_ratio": 1 - positive_ratio - negative_ratio,
                "tweet_count": tweet_count,
                "signal": signal,
                "threshold": OPTIMAL_PARAMS['threshold']
            }
        }

    except Exception as e:
        return {
            "date": datetime.now().strftime("%Y-%m-%d") if date is None else date.strftime("%Y-%m-%d"),
            "status": "error",
            "message": f"Error retrieving sentiment data: {str(e)}"
        }


def get_sentiment_trend(days=7):
    """
    Get the sentiment trend for the past N days.

    Parameters:
    - days: Number of days to include in trend

    Returns:
    - Dictionary with sentiment trend data
    """
    try:
        # Load the sentiment data
        sentiment_df = pd.read_csv(SENTIMENT_FILE)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

        # Sort by date and get the last N days
        sentiment_df = sentiment_df.sort_values('Date', ascending=False)
        sentiment_df = sentiment_df.head(days)

        # Sort back to chronological order for display
        sentiment_df = sentiment_df.sort_values('Date')

        # Format the trend data
        trend_data = []
        for _, row in sentiment_df.iterrows():
            date_str = row['Date'].strftime("%Y-%m-%d")
            signal = "BUY" if row['Positive_Ratio'] >= OPTIMAL_PARAMS['threshold'] else "HOLD"

            trend_data.append({
                "date": date_str,
                "positive_ratio": row['Positive_Ratio'],
                "negative_ratio": row['Negative_Ratio'],
                "tweet_count": row['Tweet_Count'],
                "signal": signal
            })

        return {
            "status": "success",
            "trend_data": trend_data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving sentiment trend: {str(e)}"
        }


def get_current_portfolio_status():
    """
    Get the current status of the portfolio based on latest sentiment and price data.

    Returns:
    - Dictionary with portfolio status
    """
    try:
        # Load price data
        prices_df = pd.read_csv(STOCK_PRICE_FILE)
        prices_df['Date'] = pd.to_datetime(prices_df['Date'])

        # Get the most recent price
        latest_price = prices_df.sort_values('Date', ascending=False).iloc[0]
        price = latest_price['Close'] if 'Close' in latest_price else latest_price['Adj Close']

        # Load sentiment data
        sentiment_df = pd.read_csv(SENTIMENT_FILE)
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])

        # Get the most recent sentiment
        latest_sentiment = sentiment_df.sort_values('Date', ascending=False).iloc[0]

        # Determine current signal
        current_signal = "BUY" if latest_sentiment['Positive_Ratio'] >= OPTIMAL_PARAMS['threshold'] else "HOLD"

        # Calculate profit target and stop loss levels
        profit_target_price = price * (1 + OPTIMAL_PARAMS['profit_target'])
        stop_loss_price = price * (1 - OPTIMAL_PARAMS['stop_loss'])

        return {
            "status": "success",
            "portfolio_status": {
                "current_price": price,
                "current_signal": current_signal,
                "last_update": latest_price['Date'].strftime("%Y-%m-%d"),
                "profit_target": profit_target_price,
                "stop_loss": stop_loss_price,
                "max_holding_days": OPTIMAL_PARAMS['max_holding_days']
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Error retrieving portfolio status: {str(e)}"
        }


# ====== CHATBOT FUNCTIONS ======

def process_chat_message(user_message):
    """
    Process a chat message and generate a response about sentiment and trading signal.

    Parameters:
    - user_message: User's input message

    Returns:
    - Chatbot response with relevant sentiment information
    """
    # First, use GPT to understand the intent of the message
    system_prompt = """You are a financial assistant that helps users interpret Tesla stock sentiment data and make trading decisions.
You should identify what the user is asking for, which could be:
1. Current sentiment summary
2. Trading signal (buy/sell/hold)
3. Historical sentiment trend
4. Portfolio advice
5. General question about the system
Output ONLY one of these categories as a single word: SENTIMENT, SIGNAL, TREND, PORTFOLIO, or GENERAL."""

    user_prompt = f"Classify this message: '{user_message}'"

    # Get intent classification from GPT
    intent_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
        max_tokens=20
    )

    intent = intent_response['choices'][0]['message']['content'].strip().upper()

    # Get relevant data based on intent
    data = {}
    if intent == "SENTIMENT" or intent == "SIGNAL":
        data = get_daily_sentiment_summary()
    elif intent == "TREND":
        data = get_sentiment_trend()
    elif intent == "PORTFOLIO":
        data = get_current_portfolio_status()

    # Format data for GPT response
    data_str = json.dumps(data, indent=2)

    # Generate a natural language response with GPT
    response_prompt = f"""The user asked: "{user_message}"

Based on their question, I've identified their intent as: {intent}

Here is the relevant data:
{data_str}

Please generate a helpful, conversational response that addresses their question using this data.
Include specific numbers and insights from the data.
If they asked about a trading signal, be clear about what action they should take (BUY, SELL, or HOLD) based on the optimal strategy.
Explain the reasoning behind the recommendation.
Keep your response under 150 words.
"""

    chat_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful financial assistant that explains sentiment-based trading signals for Tesla stock."},
            {"role": "user", "content": response_prompt}
        ],
        temperature=0.7,
        max_tokens=250
    )

    return chat_response['choices'][0]['message']['content']


# ====== FLASK ROUTES ======

@app.route('/')
def home():
    """Render the home page with chat interface."""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages and return responses."""
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"error": "No message provided"})

    response = process_chat_message(user_message)
    return jsonify({"response": response})


@app.route('/api/sentiment/today', methods=['GET'])
def today_sentiment():
    """Get today's sentiment summary."""
    summary = get_daily_sentiment_summary()
    return jsonify(summary)


@app.route('/api/sentiment/trend', methods=['GET'])
def sentiment_trend():
    """Get sentiment trend."""
    days = request.args.get('days', 7, type=int)
    trend = get_sentiment_trend(days)
    return jsonify(trend)


@app.route('/api/portfolio/status', methods=['GET'])
def portfolio_status():
    """Get current portfolio status."""
    status = get_current_portfolio_status()
    return jsonify(status)


@app.route('/api/analyze/tweets', methods=['POST'])
def analyze_tweets():
    """Analyze new tweets for sentiment."""
    tweets = request.json.get('tweets', [])
    if not tweets:
        return jsonify({"error": "No tweets provided"})

    results = analyze_new_tweets(tweets)
    return jsonify({"results": results})


# ====== HTML TEMPLATE ======
@app.route('/templates/index.html')
def get_template():
    """Serve the index.html template."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tesla Sentiment Trading Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
        }
        .chat-container {
            height: 70vh;
            overflow-y: auto;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e9ecef;
            padding: 10px 15px;
            border-radius: 18px;
            margin-bottom: 10px;
            max-width: 80%;
            align-self: flex-end;
            margin-left: auto;
        }
        .bot-message {
            background-color: #d1e7dd;
            padding: 10px 15px;
            border-radius: 18px;
            margin-bottom: 10px;
            max-width: 80%;
        }
        .dashboard-card {
            margin-top: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .signal-buy {
            color: green;
            font-weight: bold;
        }
        .signal-hold {
            color: orange;
            font-weight: bold;
        }
        .signal-sell {
            color: red;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="text-center mb-4">Tesla Sentiment Trading Assistant</h1>

        <div class="row">
            <div class="col-md-8">
                <div class="chat-container d-flex flex-column" id="chatContainer">
                    <div class="bot-message">
                        Hello! I'm your Tesla sentiment trading assistant. I can help you understand the current sentiment around Tesla stock and provide trading signals based on optimal parameters. How can I help you today?
                    </div>
                </div>

                <div class="input-group mt-3">
                    <input type="text" id="userInput" class="form-control" placeholder="Ask about Tesla sentiment or trading signals...">
                    <button class="btn btn-primary" id="sendButton">Send</button>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card dashboard-card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Current Trading Signal</h5>
                    </div>
                    <div class="card-body" id="signalDisplay">
                        Loading...
                    </div>
                </div>

                <div class="card dashboard-card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">Sentiment Summary</h5>
                    </div>
                    <div class="card-body" id="sentimentSummary">
                        Loading...
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const chatContainer = document.getElementById('chatContainer');
            const userInput = document.getElementById('userInput');
            const sendButton = document.getElementById('sendButton');
            const signalDisplay = document.getElementById('signalDisplay');
            const sentimentSummary = document.getElementById('sentimentSummary');

            // Load initial data
            fetchTodaySentiment();

            // Send message on button click
            sendButton.addEventListener('click', sendMessage);

            // Send message on Enter key
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });

            function sendMessage() {
                const message = userInput.value.trim();
                if (message === '') return;

                // Add user message to chat
                addMessageToChat(message, 'user');
                userInput.value = '';

                // Send to backend
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    // Add bot response to chat
                    addMessageToChat(data.response, 'bot');
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                })
                .catch(error => {
                    console.error('Error:', error);
                    addMessageToChat('Sorry, there was an error processing your request.', 'bot');
                });
            }

            function addMessageToChat(message, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = sender === 'user' ? 'user-message' : 'bot-message';
                messageDiv.textContent = message;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function fetchTodaySentiment() {
                fetch('/api/sentiment/today')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        updateSentimentDisplay(data.sentiment_summary);
                    } else {
                        sentimentSummary.innerHTML = `<div class="alert alert-danger">${data.message}</div>`;
                        signalDisplay.innerHTML = `<div class="alert alert-danger">Unable to determine signal</div>`;
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    sentimentSummary.innerHTML = '<div class="alert alert-danger">Error loading sentiment data</div>';
                });
            }

            function updateSentimentDisplay(sentiment) {
                // Update signal display
                let signalClass = sentiment.signal === 'BUY' ? 'signal-buy' : 'signal-hold';
                signalDisplay.innerHTML = `
                    <h2 class="text-center ${signalClass}">${sentiment.signal}</h2>
                    <p class="text-center">Based on optimal strategy parameters</p>
                    <hr>
                    <p><strong>Current Price:</strong> $<span id="currentPrice">Loading...</span></p>
                    <p><strong>Sentiment Threshold:</strong> ${(sentiment.threshold * 100).toFixed(1)}%</p>
                `;

                // Update sentiment summary
                sentimentSummary.innerHTML = `
                    <div class="d-flex justify-content-between mb-2">
                        <span>Positive:</span>
                        <span class="text-success">${(sentiment.positive_ratio * 100).toFixed(1)}%</span>
                    </div>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-success" role="progressbar" style="width: ${sentiment.positive_ratio * 100}%"></div>
                    </div>

                    <div class="d-flex justify-content-between mb-2">
                        <span>Negative:</span>
                        <span class="text-danger">${(sentiment.negative_ratio * 100).toFixed(1)}%</span>
                    </div>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-danger" role="progressbar" style="width: ${sentiment.negative_ratio * 100}%"></div>
                    </div>

                    <div class="d-flex justify-content-between mb-2">
                        <span>Neutral:</span>
                        <span class="text-secondary">${((1 - sentiment.positive_ratio - sentiment.negative_ratio) * 100).toFixed(1)}%</span>
                    </div>
                    <div class="progress mb-3">
                        <div class="progress-bar bg-secondary" role="progressbar" style="width: ${(1 - sentiment.positive_ratio - sentiment.negative_ratio) * 100}%"></div>
                    </div>

                    <p class="text-center mt-3">Based on ${sentiment.tweet_count} tweets</p>
                `;

                // Get current price
                fetch('/api/portfolio/status')
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        document.getElementById('currentPrice').textContent = data.portfolio_status.current_price.toFixed(2);
                    }
                });
            }
        });
    </script>
</body>
</html>
    """
    return html


# Start the Flask app if running directly
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)