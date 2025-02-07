import json
import random
from nlp_engine import preprocess_input, predict_intent

# Load intents from the JSON file
with open('intents.json', 'r') as file:
    intents_data = json.load(file)

def get_response(user_input):
    # Preprocess input
    processed_input = preprocess_input(user_input)

    # Predict intent
    intent_tag = predict_intent(processed_input)

    # Find response
    for intent in intents_data['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])

    return "I'm not sure how to respond to that. Could you rephrase?"
