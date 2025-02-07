import os
import json

DATA_PATH = "intents.json"

def load_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, 'r') as file:
            return json.load(file)
    else:
        raise FileNotFoundError(f"Data file {DATA_PATH} not found.")

def sanitize_data(data):
    # Example sanitization step: Remove empty intents
    return [intent for intent in data['intents'] if intent['patterns']]
