import argparse
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def load_model_and_tokenizer(model_dir):
    # Laden des Modells und Tokenizers
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

def predict_sentiment(text, tokenizer, model):
    # Tokenisieren des Textes
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Vorhersage durchführen
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
    
    # Rückgabe des Sentiments
    return "positive" if prediction == 1 else "negative"

def main():
    # Argumente parsen
    parser = argparse.ArgumentParser(description="Sentiment Analysis mit fine-tuntem BERT-Modell")
    parser.add_argument("--text", type=str, required=True, help="Text zur Sentiment-Analyse")
    args = parser.parse_args()

    # Modell und Tokenizer laden
    model_dir = "../models/bert-finetuned"
    tokenizer, model = load_model_and_tokenizer(model_dir)

    # Sentiment vorhersagen
    sentiment = predict_sentiment(args.text, tokenizer, model)
    print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()