# inference.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Load model
model_path = "./models/feedback_roberta"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example feedbacks
feedbacks = [
    "The product quality is amazing!",
    "I want a refund for my order.",
    "Can you explain how the new feature works?"
]

for f in feedbacks:
    result = classifier(f)
    print(f"Feedback: {f}\nPrediction: {result}\n")
