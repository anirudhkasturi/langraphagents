import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd
from datasets import Dataset

print("HALAMA")
print(os.getcwd())
#Define Labels
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive" : 1}

# Load the dataset
df = pd.read_csv('hackathon2024/model_training/sensitiveDataLlmTraining.csv')
column_to_clean = 'text'
# Clean the new line characters from the dataset 'text' rows 
df[column_to_clean] = df[column_to_clean].str.replace(r'[\n\r]', '', regex=True)
print(df.head())
dataset = Dataset.from_pandas(df)
print("Dataset")
print(dataset)
# Load pre-trained model and tokenizer
model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, 
id2label=id2label, label2id=label2id)

# Tokenize the input texts
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and test sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
print(train_test_split['train'])
print(train_test_split['test'])

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test_split['train'],
    eval_dataset=train_test_split['test'],
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./sentiment_model")