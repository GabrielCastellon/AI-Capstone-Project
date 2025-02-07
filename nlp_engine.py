from transformers import pipeline

# Initialize NLP model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def preprocess_input(user_input):
    return user_input.strip().lower()

def predict_intent(user_input):
    # Dummy function to match intents; replace with ML model logic if needed
    result = classifier(user_input)
    sentiment = result[0]['label'].lower()

    if "positive" in sentiment:
        return "happy"
    elif "negative" in sentiment:
        return "sad"
    else:
        return "neutral-response"
