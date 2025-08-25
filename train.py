# train.py
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch

# Load dataset
df = pd.read_csv("data/customer_feedback.csv")
dataset = Dataset.from_pandas(df)

# Split into train/test
dataset = dataset.train_test_split(test_size=0.2)

# Tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["feedback"], truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Labels: Sentiment + Intent
label2id = {
    "positive": 0, "negative": 1, "neutral": 2,
    "praise": 0, "complaint": 1, "query": 2
}

# For simplicity, we fine-tune sentiment classifier
num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training args
training_args = TrainingArguments(
    output_dir="./models",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save model
trainer.save_model("./models/feedback_roberta")
tokenizer.save_pretrained("./models/feedback_roberta")

print("Training complete! Model saved in ./models/feedback_roberta")
