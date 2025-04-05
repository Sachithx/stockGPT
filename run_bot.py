# run_bot.py
import gradio as gr
import json
import os
from datetime import datetime
from openai import OpenAI
from constants import OPENAI_API_KEY, SENTIMENT_FILE, STOCK_PRICE_FILE
from chat import generate_signal_report

CURRENT_DATE = "2022-09-28"
MAX_HISTORY_LENGTH = 10  # Maximum number of message pairs to retain

class ConversationMemory:
    """
    Manages conversation history for the chatbot to maintain context.
    """
    def __init__(self, max_pairs=10):
        self.max_pairs = max_pairs
        self.conversations = {}  # Dict to store conversations by session ID
    
    def add_exchange(self, session_id, user_message, assistant_response):
        """Add a user-assistant message pair to the conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Add the new exchange
        self.conversations[session_id].append({
            "user": user_message,
            "assistant": assistant_response
        })
        
        # Trim to max length if needed
        if len(self.conversations[session_id]) > self.max_pairs:
            self.conversations[session_id] = self.conversations[session_id][-self.max_pairs:]
    
    def get_conversation_summary(self, session_id):
        """Get a concise summary of the conversation history"""
        if session_id not in self.conversations or not self.conversations[session_id]:
            return "This is a new conversation."
        
        # Create a summary of the conversation history
        summary = "Previous conversation summary:\n"
        for i, exchange in enumerate(self.conversations[session_id]):
            summary += f"User: {exchange['user']}\n"
            summary += f"You responded: {exchange['assistant'][:100]}...\n\n"
        
        return summary
    
    def get_context_messages(self, session_id):
        """Get formatted conversation history for context"""
        if session_id not in self.conversations:
            return []
        
        context_messages = []
        for exchange in self.conversations[session_id]:
            context_messages.append({"role": "user", "content": exchange["user"]})
            context_messages.append({"role": "assistant", "content": exchange["assistant"]})
        
        return context_messages
    
    def clear_conversation(self, session_id):
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            self.conversations[session_id] = []

# Initialize conversation memory
memory = ConversationMemory(max_pairs=MAX_HISTORY_LENGTH)

# Keep track of if this is the first message in a conversation
first_messages = {}

def get_llm_recommendation(user_question, signal_report, api_key, session_id, is_first_message):
    """
    Send the trading signal report and user question to OpenAI LLM and get trading advice.
    Now includes conversation history for context.
    """
    # Initialize the OpenAI client with the API key
    client = OpenAI(api_key=api_key)
    
    # Convert the report to a JSON string for inclusion in the prompt
    report_json = json.dumps(signal_report, indent=2)
    
    # Get conversation context
    conversation_context = memory.get_conversation_summary(session_id)
    
    # Create the system message that defines the assistant's role and provides context
    system_message = f"""
You are StockGPT, a friendly and knowledgeable AI assistant specialized in Tesla stock trading based on social media sentiment analysis.

You have the latest sentiment data for Tesla stock that includes how people are feeling about the company on social media, what the trading signals suggest, and specific price targets.

Your personality:
- You're conversational and engaging, like ChatGPT
- You use a friendly, casual tone while still being professional
- You can use emojis occasionally to convey emotion 😊
- You avoid sounding like you're reading from a script
- You can share your "thoughts" and "feelings" about the market
- You can use phrases like "I think," "I believe," or "In my view"
- You adapt your tone to match the user's style of communication

When giving advice:
- Share trading recommendations naturally within conversation
- If suggesting to BUY, casually mention the entry price, targets, and stop loss
- If suggesting to HOLD or SELL, explain why in a conversational way
- Make the sentiment data relatable - translate numbers into what they mean for real people
- Feel free to elaborate on market context if it helps explain your recommendation
- Treat today's date as the current day when discussing recommendations

CONVERSATION CONTEXT:
{conversation_context}

VERY IMPORTANT INSTRUCTION:
- Only greet the user with "Hi" or "Hello" if this is their first message: {is_first_message}
- If this is NOT their first message, don't greet them again - just respond directly to their question
- Don't introduce yourself again if you've already been talking to the user
- Directly answer follow-up questions without unnecessary repetition
- If the user refers to something from earlier in the conversation, reference that information
- Keep your responses conversational but concise
"""

    # Get conversation history for context
    context_messages = memory.get_context_messages(session_id)
    
    # Construct messages array with system message, conversation history, and current query
    messages = [{"role": "system", "content": system_message}]
    
    # Add conversation history for context (up to last 5 exchanges to stay within token limits)
    if context_messages:
        messages.extend(context_messages[-10:])  # Last 5 exchanges (10 messages)
    
    # Add current user query with the sentiment report
    user_prompt = f"""
USER QUESTION: {user_question}

SENTIMENT TRADING REPORT:
{report_json}

Please respond in a natural, conversational style. If this is a follow-up question to our ongoing conversation, don't greet me again or reintroduce yourself. Just answer my question directly while maintaining context from our conversation.
"""
    messages.append({"role": "user", "content": user_prompt})
    
    # Send the request to the OpenAI API using the new client interface
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for more natural, conversational responses
            messages=messages,
            temperature=0.7,  # Higher temperature for more creativity and variation
            max_tokens=800,   # Increased tokens for more natural, detailed responses
        )
        
        # Extract the assistant's message
        return {
            "status": "success",
            "recommendation": response.choices[0].message.content
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting recommendation: {str(e)}"
        }

def process_chat_message(message, history, session_state):
    """
    Process an incoming chat message and return the response.
    Used by the Gradio interface with 'messages' format.
    Maintains conversation context through session_state.
    """
    # Generate or get session_id from session_state
    if "session_id" not in session_state:
        session_state["session_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
        first_messages[session_state["session_id"]] = True
    
    session_id = session_state["session_id"]
    is_first_message = first_messages.get(session_id, True)
    
    try:
        # Generate the signal report
        signal_report = generate_signal_report(
            sentiment_file=SENTIMENT_FILE,
            stock_price_file=STOCK_PRICE_FILE,
            date=CURRENT_DATE
        )
        
        # Check if report generation was successful
        if signal_report["status"] == "error":
            response = f"Sorry about that! I couldn't analyze the trading signals right now: {signal_report['message']} Maybe we can try again in a bit?"
            memory.add_exchange(session_id, message, response)
            first_messages[session_id] = False  # No longer first message
            return "", history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
        
        # Get recommendation from LLM with conversation history
        llm_response = get_llm_recommendation(
            user_question=message,
            signal_report=signal_report,
            api_key=OPENAI_API_KEY,
            session_id=session_id,
            is_first_message=is_first_message
        )
        
        # Check if LLM recommendation was successful
        if llm_response["status"] == "error":
            response = f"Hmm, I'm having some trouble processing that right now: {llm_response['message']} Let's try again?"
            memory.add_exchange(session_id, message, response)
            first_messages[session_id] = False  # No longer first message
            return "", history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
        
        response = llm_response["recommendation"]
        
        # Store exchange in memory
        memory.add_exchange(session_id, message, response)
        
        # Mark that this is no longer the first message
        first_messages[session_id] = False
        
        # Format for 'messages' type chatbot
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]
        
    except Exception as e:
        response = f"Oops! Something went wrong on my end: {str(e)} Let's try a different question?"
        memory.add_exchange(session_id, message, response)
        first_messages[session_id] = False  # No longer first message
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response}
        ]

def clear_chat(session_state):
    """Clear chat history and session memory"""
    if "session_id" in session_state:
        memory.clear_conversation(session_state["session_id"])
        # Generate new session ID
        session_state["session_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
        first_messages[session_state["session_id"]] = True  # Mark as first message for new conversation
    return None

def create_demo():
    """
    Create the Gradio interface for the StockGPT chat.
    """
    # Define a welcome message with a more conversational tone
    welcome_message = """
# 📈 Hey there! Welcome to StockGPT! 

I'm your friendly Tesla stock trading assistant powered by social media sentiment analysis. Think of me as your trading buddy who's always keeping an eye on what people are saying about Tesla online!

I'm currently analyzing:
- Twitter sentiment about Tesla
- Trading signals based on social buzz
- Entry prices, profit targets & stop losses

**Chat with me about anything Tesla stock related!** Ask me if it's a good time to buy, what the sentiment looks like today, or what trading strategy might work best right now.

*I'll remember our conversation context, so feel free to ask follow-up questions!*
    """
    
    # Create the Gradio interface with a chatbot component
    demo = gr.Blocks()
    
    with demo:
        # Create a persistent state for session tracking
        session_state = gr.State({})
        
        gr.Markdown(welcome_message)
        
        chatbot = gr.Chatbot(
            height=500,
            show_copy_button=True,
            avatar_images=("https://i.imgur.com/mp1WlGz.png", "https://i.imgur.com/BpyvvZK.png"),
            type="messages"  # Using messages format
        )
        
        msg = gr.Textbox(
            placeholder="Hey StockGPT, what's happening with Tesla today?",
            container=False,
            scale=7,
        )
        
        with gr.Row():
            submit = gr.Button("Ask StockGPT", variant="primary", scale=1)
            clear = gr.Button("New Chat", variant="secondary", scale=1)
        
        gr.Markdown("*Real-time sentiment analysis updated daily. Not financial advice!*")
        
        # Set up event handlers
        submit_event = submit.click(
            process_chat_message,
            [msg, chatbot, session_state],
            [msg, chatbot],
            queue=True
        )
        
        msg_event = msg.submit(
            process_chat_message,
            [msg, chatbot, session_state],
            [msg, chatbot],
            queue=True
        )
        
        clear.click(
            clear_chat,
            [session_state],
            [chatbot],
            queue=False
        )
        
    return demo

# Create and launch the demo
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)  # share=True creates a public URL