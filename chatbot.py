# -*- coding: utf-8 -*-

# %pip install langchain_groq langchain_core langchain_community pypdf chromadb sentence_transformers gradio tf_keras ipywidgets vaderSentiment gradio

import os
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from langchain_groq import ChatGroq
import datetime
import random
import gradio as gr

# -------------------------------
# Helper Functions & Global Setup
# -------------------------------

USER_DATA_FILE = "user_data.json"

UNIVERSITY_RESOURCES = {
    "Centennial College": "Visit the Student Wellness Centre: https://www.centennialcollege.ca/student-health",
    "University of Toronto": "Check U of T’s mental health services: https://mentalhealth.utoronto.ca/",
}

MOTIVATIONAL_QUOTES = [
    "Stay focused! Every small step brings you closer to success. 💪",
    "You’re capable of amazing things—keep pushing forward!",
    "Don’t let stress take over! Take breaks, breathe, and keep going. 🚀"
]

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}  # Return empty dict if file doesn't exist

def save_user_data(user_data):
    with open(USER_DATA_FILE, "w") as file:
        json.dump(user_data, file, indent=4)  # indent=4 makes JSON human-readable

def update_user_data(user_id, key, value):
    user_data = load_user_data()
    if user_id not in user_data:
        user_data[user_id] = {}  # Initialize empty dict for new user
    user_data[user_id][key] = value
    save_user_data(user_data)

def update_student_profile(user_id, major, year_of_study, common_stressors, university):
    user_data = load_user_data()
    if user_id not in user_data:
        user_data[user_id] = {}
    # Store all profile details in user's data
    user_data[user_id]["major"] = major
    user_data[user_id]["year_of_study"] = year_of_study
    user_data[user_id]["common_stressors"] = common_stressors
    user_data[user_id]["university"] = university
    save_user_data(user_data)

def get_mental_health_resources(user_id):
    user_data = load_user_data()
    # Default to generic message if university not specified
    university = user_data.get(user_id, {}).get("university", "your university")
    if university in UNIVERSITY_RESOURCES:
        return f"If you need support, check out {UNIVERSITY_RESOURCES[university]}"
    else:
        return "I recommend checking your university's website for student wellness resources."

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores["compound"]
    # Classify sentiment based on compound score thresholds
    if compound_score <= -0.5:
        return "sad"
    elif -0.5 < compound_score <= -0.1:
        return "frustrated"
    elif -0.1 < compound_score < 0.1:
        return "neutral"
    elif 0.1 <= compound_score < 0.5:
        return "happy"
    else:
        return "excited"

def load_llm():
    return ChatGroq(
        temperature=0.6,  # Controls randomness; 0.6 balances creativity and coherence
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"  # Specific LLM model used
    )

def generate_summary(conversation_history, llm):
    prompt = f"""
    You are a helpful assistant. Summarize this conversation in 1-2 sentences.

    Conversation:
    {" ".join(conversation_history)}

    Summary:
    """
    response = llm.invoke(prompt)
    return response.content.strip()

def check_deadlines(user_id):
    user_data = load_user_data()
    deadlines = user_data.get(user_id, {}).get("deadlines", {})
    today = datetime.date.today()
    # Check for deadlines within next 3 days
    upcoming = [
        task for task, date in deadlines.items()
        if datetime.date.fromisoformat(date) <= today + datetime.timedelta(days=3)
    ]
    if upcoming:
        return f"Reminder! You have upcoming deadlines: {', '.join(upcoming)}. Don’t forget to plan ahead!"
    return "No major deadlines soon. Keep up the good work!"

def send_daily_motivation():
    return random.choice(MOTIVATIONAL_QUOTES)  # Randomly select a quote

# -----------------------------------------
# Chatbot Function for Gradio Interface
# -----------------------------------------

def chatbot_response(user_message, history, user_id):
    """
    This function takes the latest user message, the conversation history,
    and the user_id. It constructs a conversation prompt (as a string) for the LLM,
    updates the internal history, and returns the UI history in the expected format.
    """
    llm = load_llm()
    user_data = load_user_data()
    major = user_data.get(user_id, {}).get("major", "student")  # Default to "student" if no major

    messages = []

    # Construct system message with user context
    system_content = (
        f"You are a mental health assistant for students. The user is studying {major}.\n"
        f"- Name: {user_id}\n"
        f"- Last emotion: {user_data.get(user_id, {}).get('last_emotion', 'None')}\n"
        f"- Last conversation: {user_data.get(user_id, {}).get('last_conversation', 'None')}\n\n"
        "Respond with empathy and helpful advice. Keep your responses brief and focused.\n"
        "Ask at most one follow-up question per response. Prioritize clarity and conciseness."
    )

    if "schedule" in user_message.lower():
        deadline_info = check_deadlines(user_id)
        system_content += f"\n- Upcoming deadlines: {deadline_info}\nInclude these deadlines in your response."

    sentiment = analyze_sentiment(user_message)
    if sentiment in ["sad", "frustrated"] or "help" in user_message.lower():
        resource_info = get_mental_health_resources(user_id)
        system_content += f"\n- Mental health resources: {resource_info}\nIf resources are provided, include them in your response to support the user."

    messages.append({"role": "system", "content": system_content})

    # Add history to messages
    for pair in history:
        user_text, bot_text = pair
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": bot_text})

    messages.append({"role": "user", "content": user_message})

    # Convert messages list to a single string for LLM
    prompt_str = ""
    for msg in messages:
        role = msg["role"].capitalize()
        prompt_str += f"{role}: {msg['content']}\n"

    response = llm.invoke(prompt_str)
    bot_reply = response.content.strip()

    history.append((user_message, bot_reply))

    # Format history for Gradio UI
    ui_history = []
    for pair in history:
        ui_history.append({"role": "user", "content": pair[0]})
        ui_history.append({"role": "assistant", "content": pair[1]})

    return ui_history, history

# ----------------------------
# Gradio Blocks UI Definition
# -----------------------------

def setup_profile(name, major_val, year, stressors, univ):
    """
    Save the profile to our JSON file and update the user_id state.
    """
    if not name.strip():
        return "Please enter a valid name.", ""  # Validate non-empty name
    update_user_data(name, "name", name)
    update_student_profile(name, major_val, year, stressors, univ)
    status = f"Profile set up for {name}. Welcome!"
    return status, name  # Return status and user_id

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Mental Health Chatbot")

        with gr.Tab("User Setup"):
            gr.Markdown("### Set up your profile")
            user_name = gr.Textbox(label="Enter your name")
            major = gr.Textbox(label="Your major")
            year_of_study = gr.Textbox(label="Year of Study")
            common_stressors = gr.Textbox(label="Common stressors")
            university = gr.Textbox(label="University")
            setup_button = gr.Button("Set Up Profile")
            setup_output = gr.Textbox(label="Setup Status", interactive=False)

        with gr.Tab("Chat"):
            gr.Markdown("### Chat with the Bot")
            chatbot = gr.Chatbot(label="Conversation", type="messages")
            msg = gr.Textbox(label="Your Message")
            clear = gr.Button("Clear Chat")

        state = gr.State([])  # Stores internal conversation history
        user_id_state = gr.State("")  # Tracks current user ID

        setup_button.click(
            setup_profile,
            inputs=[user_name, major, year_of_study, common_stressors, university],
            outputs=[setup_output, user_id_state]
        )

        def respond(message, chat_history, user_id):
            return chatbot_response(message, chat_history, user_id)

        msg.submit(respond, [msg, state, user_id_state], [chatbot, state])
        clear.click(lambda: ([], []), None, [chatbot, state])  # Reset both UI and internal history

    demo.launch(share=True)  # Share=True generates a public URL

if __name__ == "__main__":
    main()
